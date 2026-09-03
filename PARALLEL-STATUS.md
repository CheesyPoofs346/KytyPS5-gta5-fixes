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

## Step 2 is blocked, and here is the exact reason

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
