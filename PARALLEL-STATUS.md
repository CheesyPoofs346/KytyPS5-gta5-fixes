# Parallel recording — what is built, what to test, what is not done

Written overnight 2026-09-03. Everything below compiles and links; tests are 53 ok.
**Nothing here has been run in game.** No launches happened while you were asleep.

---

## Test it like this

```bash
# 1. baseline, unchanged code path (all flags off)
./run-gta5.sh --present-mode Mailbox

# 2. batching - this is the RAM and overhead fix, still one thread
./run-gta5.sh --present-mode Mailbox --secondary-record true

# 3. deferred translation through the collection queue, drained in guest order
./run-gta5.sh --present-mode Mailbox --secondary-record true --draw-queue true

# 4. only if you want to help finish the upload work - EXPECTED TO BE WRONG somewhere,
#    this is the flag whose remaining consumer has not been found yet
./run-gta5.sh --present-mode Mailbox --secondary-record true --defer-uploads true
```

Run 2 is the important one. Compare against last night's `--secondary-record true`, which ate
all your RAM.

**Expect:** memory bounded and far lower, framerate somewhere near baseline. Batching removes the
per-draw `vkCmdExecuteCommands`, so the overhead that made the single-worker path slow is gone,
but no work has moved off the main thread yet, so do not expect an fps win.

**Watch for:** any rendering difference from run 1. Batching changed *when* the render pass opens
(once per batch instead of once per draw), so if anything regressed visually it will be a barrier
or ordering problem, and the fix is a missing flush hook.

---

## What landed

| commit | what |
|---|---|
| `e9ceeb2` | worker pool, per-worker `VkCommandPool`, secondary buffers with dynamic-rendering inheritance |
| `5158761` | per-worker `DescriptorHeap`, selected by a thread-local slot |
| `3889e46` | batch-scoped buffer ownership - no `Buffer` destroyed while a batch is open |
| `8d350dd` | per-worker stream rings, 16 MiB each, capped at 8 slots (112 MiB extra) |
| `ca796cf` | single-worker end-to-end secondary path - **verified rendering clean, 0 validation errors** |
| `eb57598` | reclaim secondaries before growing the ring (the RAM bug) |
| `bdc7de9` | **batching** - one secondary carries many draws, cut on render-state change |
| `23eb846` | reader-writer lock on the page table, deferred per-worker LRU |
| `f2c5cde` | `--draw-workers`, ownership scope tied to the batch |

## Night 2: deferred uploads built, gated off, and why

`--defer-uploads` (default false) makes resolve *stage* buffer uploads instead of recording them,
with `FlushPendingUploads` recording the list on the main thread before the batch opens its render
pass. Both paths share one `RecordUpload`.

It is off by default deliberately. Deferral is only correct if every consumer of an upload drains
the list first, and enumerating consumers by hand failed twice in twenty minutes: the suite went
53 -> 49 on `RenderExecutorColorVolumeDiscovery` ("upload/readback lost the final Z slice"), and
hooking `Flush`, `FlushAndWait`, `DownloadBufferMemory` and `ReadMemoryOnGpu` still was not
enough. Something else reads back through a path not yet found. Guessing at more hook points is
the same mistake in a new costume, so the semantics now change only under a flag.

**To finish it properly:** find the remaining consumer by running with `--defer-uploads true` and
watching that one test, or better, invert the design - rather than enumerating consumers, make
`CommandScheduler::Current()` drain pending uploads whenever the caller is about to record
something that is not a batched draw. That is a chokepoint rather than a list, so it cannot miss a
caller. It needs a re-entry guard, since `RecordUpload` calls `Current()` itself.

## Step 2 is still blocked, and buffers were only half of it

Even with uploads deferred, `ParallelFor` cannot be wired: **`textureCache` records into the
primary during resolve as well** - three `m_scheduler.Current()` sites covering image uploads,
clears and barriers, all reachable from `RebindImages`/`ResolveTexture` on the per-draw path.
Texture uploads need the same staging treatment as buffer uploads before any resolve work can
leave the main thread.

So the remaining order is: finish the buffer-upload chokepoint, do the same for texture uploads,
then wire `ParallelFor`. Wiring it before those two would produce eight threads recording into one
primary out of order, which renders as corruption rather than as an error.

## Step 2 blocker, original evidence

`ParallelFor` is not wired over draws, because resolve cannot run off the main thread as the code
stands. `BufferCache::SynchronizeBuffer` - reached from `ObtainBuffer`, which every draw hits for
every buffer - does three things a worker may not do:

1. calls `command.EndRendering()`, which now flushes the secondary batch;
2. records a barrier, a `copyBuffer` and another barrier into the **primary** command buffer;
3. allocates out of the shared staging ring via `UploadCopies`.

So "resolve" is not a pure phase that produces descriptors - it emits commands. Running it on
eight threads would have eight threads recording into one primary, out of order, whatever the
caches do about locking. No amount of page-table locking fixes that, which is why the locks landed
but the dispatch did not.

**The unblocker, and the next thing to build:** resolve must produce a *list* of required uploads
instead of recording them - address, size, source offset - and a serial phase must apply that list
in guest order before the batch is replayed. The memory tracker's upload bookkeeping has to move
with it, since `ForEachUploadRange` also clears CPU-dirty state as a side effect. That is a
contained change to `bufferCache`, but it changes when uploads become visible to the GPU, so it
wants a run to validate rather than a night of unverified edits.

## Assumption the queue relies on

A queued draw holds raw guest pointers (`index_addr`, and everything resolve reads). Deferring
translation means reading that memory later than the packet arrived. This is the same assumption
the console makes - its command processor consumes packets asynchronously, so a title cannot
overwrite data a submitted draw still needs - but it is an assumption, and if `--draw-queue`
produces corruption that plain `--secondary-record` does not, this is the first thing to suspect.

## What is NOT done - read this before assuming 16 cores are running

**Draws are still translated serially.** `--draw-workers 8` creates the pool, engages the locks
and the deferred LRU, and sizes the per-worker resources - but `DrawIndex` still resolves and
records every draw on the calling thread. Nothing calls `ParallelFor` over draws yet.

What remains is the restructure that makes parallelism possible at all: the command processor
handles one draw packet at a time and translates it synchronously, so there is no group of draws
to hand out. Draws have to be collected - packet plus its 188 ns register snapshot - and the group
resolved and recorded across workers, before the merge replays them in guest order.

That is a real architectural change to the draw path and it cannot be validated without running
the game, which is why it is not half-built here. The pieces it needs are all in place:
snapshots (0.188 us, 1.04/draw), ownership rules, per-worker resources, locks.

## Amdahl, so the target is honest

Serial PM4 ingest was measured at 7.65-8.42 us/draw against a ~12 us `DrawIndex`, but that number
fails its own sanity check - `DrawIndex` plus ingest exceeds the measured frame cost, so the phase
is still double-counting somewhere (one nesting path was fixed, another remains). Until it is
exact, the ceiling is unknown. Even at half, ~4 us of serial ingest against ~12 us of parallelisable
draw work caps the whole plan near 4x, not 16x.

Worth fixing that measurement before investing in the collection restructure, because it decides
whether the restructure buys 4x or 1.5x.

---

# Update 2026-09-03 evening — the number that redirects the plan

Tests **53 ok**, everything links. No game launches happened while you were away.

## Recording is not the bottleneck; resolution is

The drain census reported `phase3_record = 83.7%`, which reads as "recording is the bottleneck".
It is not. Phase 3 is *everything after the SRT walk*, and almost all of it is resource
resolution against shared caches. Splitting it by the measured per-phase profile:

| Phase 3 work | us/draw | parallelisable? |
|---|---|---|
| PrepareGraphicsBindings | 3.373 | no - buffer/texture caches |
| vertex+index buffers | 1.035 | no - buffer cache |
| AcquireRenderTargets | 0.575 | no - texture cache |
| CreateGraphicsPipeline | 0.323 | no - locked pipeline cache |
| PrepareDrawRenderState + preamble | 0.710 | no |
| Commit vb/desc/ib | 0.618 | yes |
| BeginRendering+bind | 0.426 | yes |
| SetGraphicsDynamicParams | 0.087 | yes |
| EmitDrawPrimitives | 0.044 | yes |

**Actual Vulkan recording is ~1.18 us of ~11.9 - about 10%**, and part of that 0.618 is image
transitions that must stay serial anyway. Multi-threading recording has a ceiling of roughly
**5-9%**, not 83.7%.

The work worth doing is parallelising **resolution** (~5.9 us/draw). It needs exactly the same
prerequisite - stopping resolve from touching the primary command buffer - so the plan was right,
the payoff is just somewhere else than it looked. That is what tonight went into.

## Landed tonight

| commit | what |
|---|---|
| `3568035` | SRT resource walk runs across the worker pool |
| `dd3d1cb` | worker pool sized from --draw-workers, not by whoever asks first |
| `9642fcc` | image layout transitions staged for a serial pre-pass (--defer-transitions) |

### The pool bug, and why the 8-wide walk has never run

`GetDrawWorkerPool(count)` created the pool on first use and the first caller won -
`FlushSecondaryBatch` asking for 1, which always runs before the first queue drain. The pool was
built with **one worker**, so ParallelFor had two runners and the walk got 2x instead of 8x. That
is exactly the two-cores-busy reading. It now takes no argument and reads the config itself.

**So the 15.75 us/draw result was measured 2-wide. The 8-worker walk has never actually run.**
First thing to check on return.

### Transition staging

`Image::Transit` writes a barrier straight into the primary, from `CommitBindings`, for every
sampled image of every draw - the single thing stopping resolution running on a worker. The four
layout cases now compute one target layout and access mask, then either transition inline
(default) or stage a `PendingTransition` applied before the batch opens its render pass. The
descriptor is written with the layout the image will hold.

Ordering is unchanged by construction: a batch's draws already replay after every transition
their resolve recorded, so the staged pre-pass is what the code already did - only who records
them moves.

## Not done, deliberately

Parallel recording is not wired. At a ~9% ceiling it is not worth the risk ahead of the
resolution work, and it needs texture clears deferred too - the three `m_scheduler.Current()`
sites in `textureCache` are image *clears*, which transition and clear, so they need the same
treatment as `Transit`.

Remaining blockers for parallel resolution, in order:

1. `--defer-uploads` still has one unfound consumer. Built, gated off.
2. Texture clears need staging, same pattern as transitions.
3. `ObtainBuffer` -> `SynchronizeBuffer` records uploads and calls `EndRendering`; stageable once
   1 and 2 are done.
4. Then resolution moves to ParallelFor, and recording comes with it.

## Test plan on return

```bash
# 1. the 8-wide walk, which has never actually run 8-wide
./run-gta5.sh --present-mode Mailbox --secondary-record true --draw-queue true --draw-workers 8

# 2. correctness of staged transitions - expect no visual change
./run-gta5.sh --present-mode Mailbox --secondary-record true --draw-queue true               --draw-workers 8 --defer-transitions true --vulkan-validation true
```

Baselines: **14.08 us/draw** (batching only, sec13), **15.75** (walk at 2 runners, par8),
**16.37** (par9, census overhead included).

Watch `DrainCensus`: `phase2_walk` should fall from ~10% toward 2-3% if the pool fix worked, and
TOTAL should land near 13 us/draw. That is the honest expected size.

Run 2 is purely a correctness check. If anything renders differently with `--defer-transitions`,
the staged layout does not match what `Transit` would have produced, and `binding.layout` is the
first thing to look at. Validation now survives the title's three pre-existing faults and stays
fatal only for secondary command buffer errors, which is the class this work can break.

---

# Update 2026-09-03 late — both remaining blockers cleared

Tests **53 ok**, builds clean, no game launched.

## The staging rule changed, and that is the important part

Deferral used to be controlled by a flag, which meant every consumer of a staged upload had to be
found and drained. That failed twice. The failing test shows why it was never going to work: it
calls `ObtainBuffer` and then records a `copyBuffer` from that buffer on the raw handle. A
"consumer" is anyone who obtains a resource and then records against it - an unbounded set.

Staging now keys on **who is running**, not on a flag:

```
MustStageForWorker()  ->  t_draw_worker_index != 0
```

- A main-thread caller records inline, exactly as it always has. No consumer can be surprised,
  and no enumeration is needed.
- A worker always stages, which is the property correctness actually depends on.
- The `--defer-uploads` / `--defer-transitions` flags remain as force-on overrides, for testing
  the staged paths deliberately.

This is what closes the `--defer-uploads` blocker: there was never a single missing consumer to
find.

## Texture clears staged

Both sites (`ClearImageFromBuffer`, and the DCC metadata clear) end the render pass, transition
the image to TransferDst and clear it - three recordings into the primary from the resolve path.
They now stage a `PendingClear`, and `TextureCache::FlushPendingClears` records them in the batch
pre-pass.

Pre-pass order is now: **clears, then transitions, then buffer uploads.** Clears write images that
the transitions afterwards put into their sampled layout, so the order is load-bearing.

## Where that leaves parallel resolution

Everything that recorded into the primary from the per-draw resolve path is now stageable:

| what | status |
|---|---|
| image layout transitions (`CommitBindings`) | staged, `9642fcc` |
| texture clears (2 sites) | staged, `c0a0c97` |
| buffer uploads (`SynchronizeBuffer`) | staged, `c3a9111` + `c0a0c97` |

Remaining before resolution can move to `ParallelFor`:

1. **`SynchronizeBuffer` still calls `EndRendering` on the immediate path.** Harmless on the main
   thread; on a worker it must not run at all. Verify the staged path never reaches it.
2. **Texture cache spinlock** (`m_lock`) is taken during resolution. It is a lock, so it is safe,
   but eight workers contending on one spinlock will not scale - it wants the same treatment the
   page table got (reader-writer, deferred writes).
3. **`AcquireRenderTargets`** has not been audited for primary-buffer recording at all. It is
   0.575 us/draw of the resolve path and it is the last unexamined piece.

Only after those three does `ParallelFor` over resolution make sense. Wiring it before then gives
eight threads racing on a spinlock and recording into one primary, which renders as corruption
rather than erroring.

## Test plan unchanged from the previous section

The two runs listed above still stand. Nothing in this update changes behaviour on the main
thread, so runs 1 and 2 should behave exactly as they would have before it - which is itself the
thing to confirm first.
