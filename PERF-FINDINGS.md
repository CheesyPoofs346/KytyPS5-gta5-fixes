# GTA V performance findings

## CONTRIBUTOR (not root cause): the game was in "Performance RT" mode

Settings -> Graphics -> **Performance** (not Performance RT).

Ray tracing was enabled in-game for the entire investigation. Measured effect of turning it off:

```
Performance RT : 3,737 draws/frame, 14.4 fps
Performance    : 3,109 draws/frame, 16.6 fps
```

**-17% draws, +15% fps. Real, but it is not the whole problem** - the city is still ~16.6 fps,
not 30. RT removed 628 draws/frame; it did not change the ~19us cost per draw, which remains
the actual bottleneck. Check it before profiling anyway, since it inflates every measurement:

| observation | explanation |
|---|---|
| 4,000-6,000 draws/frame (real GTA V does ~2,000-3,000) | RT passes |
| GPU only 17-22% utilised while frames crawled | RT structure building is CPU-side |
| "looking up/down is smooth, city is not" | fewer objects -> fewer draws (holds with RT off too) |

Still true after the RT fix: `frame time = draws x ~19us`, single-threaded. 30 fps at 3,100
draws needs ~10.7us/draw. That gap still requires parallelism or bindless descriptors.

**Check the game's own graphics settings before profiling the emulator.** A full day went into
optimising the draw path when the workload itself was inflated by a setting in the pause menu.


Machine: Ryzen 9700X (16 threads), CPU ~29% / GPU ~22% during gameplay.
Run with: `./run-gta5.sh --present-mode Mailbox`

## The core relationship

`frame time = draw count x ~17 us`. This held across every measurement, all day.

- Looking at ground / interior / cutscene: ~400-1000 draws -> 30+ fps, smooth
- Los Santos street level: 3,000-5,000 draws -> 13-17 fps

There is no hidden stall. The cost is per-draw CPU work on a single thread.

## Where the time goes (Tracy, 41M zones, 20s of driving)

| zone | us/draw | notes |
|---|---|---|
| MaterializeResources | 3.80 | **pure function** - no shared state. Best parallel candidate. |
| ExecutePreparedDraw (self) | 2.69 | |
| RebindBuffers | 1.87 | per-object uniform data; genuinely changes each draw |
| PrepareBindings | 0.95 | |
| FindBuffers | 0.77 | |
| DrawIndex (self) | 0.53 | |
| CommitBindings | 0.56 | descriptor writes are cheap |
| RebindImages | 0.37 | |
| **total measured** | **~12.5** | |

To reach 30 fps at 4,500 draws you need ~7 us/draw. Single-threaded optimisation cannot
get there; the remaining work is legitimate per-draw work.

## CORRECTION — net performance gain was ZERO

Final measurement, memo disabled for correctness:

```
session start           : 18.69 us/draw, 14.4 fps
after a full day        : 18.74 us/draw, 15.4 fps   <- per-draw cost UNCHANGED
```

Both large "wins" reported during the session were corrupted rendering, not speed:

- **Evaluator memo (-10%)**: produced wrong descriptor values. Light coronas rendered as huge
  red blobs. Now `shader_memo = false` by default.
- **Skipping "redundant" global barriers (-8%)**: skipped barriers that copies/uploads/clears
  genuinely needed. Same red-blob corruption.

The tell in both cases was identical: a cheap, large gain in code just written, not
investigated for a catch. Any future optimisation here must be verified visually in a
light-heavy scene (night, under an overpass) before its number is believed.

The remaining safe changes (validation gates, allocation removal, 8KB memset, XXH3 hash,
dead probe removal) are correct but individually and collectively within measurement noise.

**The only real win of the session was `--present-mode Mailbox`** - the default FIFO mode
quantised framerate to 60/N, displaying ~19 fps of work as a hard 15.

## Landed and safe

- `--present-mode Mailbox` — the default FIFO quantised fps to 60/N, so ~19 fps of real
  work displayed as a hard 15. Biggest single win, costs nothing.
- SRT evaluator memo: was a vector scanned linearly per node (O(nodes^2)); now an
  open-addressed table with generation stamps.
- `ValidateResourceSpecialization` gated off in both hot call sites (ran twice per draw).
- Removed an 8 KB `memset` per call in `HashGuestEdges` (per image, per draw).
- Pipeline key hashed as a block (XXH3) instead of byte-by-byte.
- Removed 2 per-call heap allocations in `MaterializeResources` (~2M calls / 20s).

Net: **18.69 -> 16.68 us/draw (-10.7%)**. The 30 fps threshold moved from ~1,783 to
~1,997 draws. Below that the game is smooth; above it, it is not.

Additional landed after that measurement:
- Evaluator memo table resized 1024 -> 256 slots (24KB -> 6KB) so it stays in L1. Verified
  net positive: memo ON 16.96 us/draw vs memo OFF 18.78 us/draw. Bypass with `--shader-memo false`.
- Removed dead per-draw diagnostic probes (DegenerateViewportZ, NoDepthDraw, CollapsedZDraw,
  GtaDegenerateDepth, HDR probe hooks) - 3 atomics per draw from investigations long closed.

## Measured and RULED OUT — do not re-investigate

| suspect | measurement |
|---|---|
| texture streaming CPU | 0.4 ms/frame (detiling is GPU-side) |
| garbage collection | 4.6 us per run |
| page protection / VirtualProtect | 0.3 ms/frame |
| buffer-upload render-pass breaks | 12.6 per frame |
| pipeline cache lock contention | 0.03 us |
| GPU | 22% utilised — not the limit |
| forced readback in PrepareStorageSampledOverlap | 50 drains, 0.08 ms each = 4.2 ms total |
| whole-snapshot materialization cache | 4-12% hit rate — key includes per-object constants |

## Failed attempts and WHY (each cost a broken build)

1. **Prefetch worker replaying submissions** — PM4 packet length is not derivable from the
   header; the real parser uses each handler's return value. Skipping packets by header
   length misaligns the stream and dereferences garbage.
2. **Per-frame shader compile budget** — capping at N/frame meant ~600 shaders took ~300
   frames to land; geometry (traffic lights) stayed missing for ~10s.
3. **Async shader compilation** — the recompiler is not thread-safe, and non-null
   `ShaderProgram` invariants are assumed throughout the draw path.
4. **Per-descriptor caching (x3)** — buffers/images share `source` ids (different descriptor
   widths); the evaluator memo hides dependencies from a hit; subtree dependency scoping
   still leaked. Code left inert in SrtWalker.cpp with `constexpr bool cacheable = false`.
5. **Skipping "redundant" global barriers** — WRONG. Copies, image uploads, clears and blits
   also write memory. Skipping barriers after those corrupted light rendering. The apparent
   +13% was just skipping necessary work.

## Profile breakdown inside materialization (Tracy)

| | us/call | calls | us/draw |
|---|---|---|---|
| EvaluateRuntimeSources | 2.01 | 1.75/draw | **3.5** |
| MaterializeSnapshot | 0.27 | 1.75/draw | 0.5 |

Evaluation is 93% of materialization and ~21% of the whole frame. It is recursive tree
walking with a memo probe per node. The proper fix is a precomputed flat evaluation order
per program (the expression structure is fixed; only leaf values change), turning recursion
into a linear loop with no hashing. Single-threaded, no threading risk.

`ExecutePreparedDraw` self time (2.89 us) is actual Vulkan command recording - irreducible.

## The change that would actually reach 30 fps

Parallelise `MaterializeResources` across draws. It is pure: const program data + user data
+ read-only guest memory in, its own snapshot out. Touches no texture cache, no command
buffer, no shared mutable state.

Blocker: the PM4 parser hands each draw straight to recording, so there is no lookahead
window for workers to fill. Requires decoupling parse-ahead from command recording.

Note: the VS->PS ordering constraint is weaker than it appears. `ShaderGetStaticInputInfoPS`
needs only `vs_info.stage.program` (for push-constant offsets), which comes from the program
cache lookup, NOT from materialization. So vertex and pixel materialization can overlap once
both program pointers are known.

Arithmetic for the full version: per draw is ~13 us of *prepare* work (materialize 3.5,
bindings 3.8, shader lookup, target resolve) and ~3 us of *record* work. Prepare across 4
workers = 3.25 + 3 = ~6.3 us effective, which is 30+ fps even at 4,500 draws. Record must
stay ordered on one thread.

Blocker beyond the lookahead window: `BindImage` mutates shared texture-cache state
(`is_bound`, `shader_write`, `force_general`) mid-prepare. Workers need per-thread staging
or that state has to move out of the prepare phase.

### Prerequisite already in the tree

`ProgramCache::TryLookupUnique()` and `ProgramCache::MaterializeInto()` in pipelineCache.cpp
split the shader lookup from materialisation. This is the piece that makes parallel
materialisation possible: the pixel stage's PrepareProgram reads only
`vs_info.stage.program`, which `TryLookupUnique` publishes - so BOTH stages can be looked up
first, then materialised concurrently.

### What is NOT done, and the trap in it

An attempt at `PipelineCache::MaterializeGraphicsStages()` was removed because it recomputed
`PrepareProgram` for the pixel stage with a DEFAULT `target_export_mapping` instead of the
real one, which would specialise a different shader than the lookup matched. Whatever drives
the parallel path must carry the real mapping (and the real ShaderParams) through to
materialisation rather than rebuilding them.

Second option: bindless descriptors (persistent descriptor heaps, as shadPS4 uses) would
remove most of `RebindBuffers`/`PrepareBindings` outright. Larger rewrite.

## Still open (not performance)

- Fog missing (regression, cause unknown — check DCC clear path and validation gates)
- Palm trees vanish close up: 46.5k draws/session dropped, all stage mask 0x0200210d,
  primitive type 9 (patch list) = tessellation, unimplemented
- `UnifiedTextureCacheFlow` test fails — PRE-EXISTING, unrelated to this work

## Draw-count investigation: SETTLED (no pathological shader)

Built an A/B harness keyed on `PsStageRegisters::chksum` (stable across launches, unlike the
shader address which is a guest pointer that changes every run).

```
ShaderCensus (city):  2,400-2,700 draws/frame
  top shader   19.3%   |  no single effect dominates
  2nd          12.7%   |  long tail after the top three
  3rd          12.4%   |
```

Skipping the top shader (verified: it vanished from the census, 240k draws dropped) removed
19.3% of draws and **19.3% of the work. us/draw stayed at ~20.** No shader is disproportionately
expensive; the draws are legitimate and uniformly priced.

**This closes the draw-count theory.** Frame time cannot be reduced by finding a redundant pass,
because there isn't one. The only remaining lever is per-draw cost, and halving it single-
threaded was attempted seven ways and failed every time.

### Usage

```bash
./run-gta5.sh --present-mode Mailbox                          # census prints every 300 frames
./run-gta5.sh --present-mode Mailbox --skip-ps-chksum 0x<hex> # A/B one shader
```

Three bugs were built into this harness before it measured anything, all mine: keyed on the
per-run address instead of the checksum; counted draws *before* the skip check so a skipped
draw was indistinguishable from an executed one; and both errors produced plausible-looking
numbers that were pure scene variation. **Verify a measurement tool measures the thing before
trusting a single number from it.**

## Parallel shader resolution: BUILT, MEASURED, NEGATIVE RESULT

Implemented and measured (flag `--parallel-resolve`, defaults off):

```
serial                   : n=29  median 20.30 us/draw  (p25 18.68, p75 21.73)
parallel (always)        : n=39  median 20.17 us/draw  (p25 19.06, p75 22.31)
parallel (>=6 resources) : n=15  median 19.76 us/draw  (p25 18.57, p75 21.66)
```

**No usable gain.** Quartile ranges overlap almost exactly. 53 tests pass, no corruption.

**Why two-way overlap cannot work here:** only two things per draw are independent (vertex and
pixel resource resolution, ~3us each). The cross-core handoff - atomic store, spin, cache-line
transfer, plus both threads pulling the same guest memory - consumes most of what overlapping
saves. Gating dispatch on descriptor count (so only expensive shaders pay the handoff) moved it
from 0% to ~2.7%, still inside noise.

**What the architecture now has** (kept, correct, flag-gated off):
- `PrepareVertexParams` / `MaterializeVertexProgram` - the serial/parallel seam
- `LookupVertexProgram` / `ResolveVertexResources` - permutation lookup split from resource
  resolution. Lookup publishes `stage.program`, which is all pixel prep needs, so pixel params
  can be built before either stage resolves resources.
- `ResolveWorker` - spin-then-yield worker (a condvar wake costs more than the work saved)

**HAZARD documented in pipelineCache.h:** `ShaderParams::user_data` is a span into LIVE register
state. Anything deferring materialisation past the next register write must copy it first.

### The only remaining path to 30 fps

Queue-based parallelism: buffer many draws so workers process a queue instead of trading one job
back and forth. That needs parse-ahead, which needs per-draw register snapshots - the piece that
killed the earlier prefetch attempt (it tried to re-parse the PM4 stream instead of buffering
already-parsed draws). ShaderParams itself is small and copyable, so the job payload is cheap;
the register state is the hard part.

## Batching (multi-draw-indirect): RULED OUT WITH DATA

```
BatchPotential: 100000 draws | 100000 distinct states | 0 batchable (0.0%) | avg run 1.00
```

**Every draw has unique state.** Not one consecutive pair shares pipeline + bindings. GTA gives
each object its own buffer offsets, so MDI batching would require moving per-object data into
arrays indexed by `gl_DrawID` - i.e. modifying the shader recompiler. Not viable as a bolt-on.

This was the one idea that could have delivered 3-4x (cost divides by run length). It is dead,
and it cost one measurement instead of days of implementation.

## What actually works: attacking SRT evaluation overhead

Verified at city scale (>2500 draws), which matters - lighter scenes flatter the numbers:

```
baseline                    19.88 us/draw  (n=27)
flat L1 memo                15.70 us/draw  (n=23)   -21%
```

The evaluator was using an `unordered_map` memo - two hash operations per expression node, and
expressions have tens of nodes. Replacing it with a 4KB open-addressed table that stays in L1
was worth 3.6 us/draw. **Scoped to a single evaluation** (never reused across draws); cross-draw
reuse is what corrupted light coronas earlier - see the warning at the declaration.

Also landed, unmeasured (built after the last data run):
- cycle check moved into the memo (was a linear `m_visiting` vector scan on every node)
- single-entry memo in front of the graphics pipeline hash map (key is ~200 bytes, hashing walks
  every byte; consecutive draws often produce an identical key)

## CACHE IDENTITY BUG - crashed the emulator, worth remembering

The descriptor cache used the `Program`'s **raw pointer** as identity. Programs are shared_ptr
owned and freed on eviction; a later allocation reused the address, a stale entry matched, and a
descriptor from a *different shader* was returned:

```
ShaderRecompiler PS failed hash=0x0000000010960993: image descriptor 0 has 4 dwords
```

Fixed by keying on the shader hash (not recycled) plus the expected dword count.

**Pattern to watch:** three caching schemes in one session were unsafe (evaluator memo, barrier
skip, descriptor cache). Each time the fast path's *logic* was right and its *identity
assumptions* were wrong. Validate what makes an entry addressable, not just what it computes.

## Staged and awaiting one measurement run

All build clean; 53 tests pass (only the pre-existing UnifiedTextureCacheFlow failure remains).

| change | expected | measured? |
|---|---|---|
| flat L1 memo (replaced unordered_map in the SRT evaluator) | -21% | **YES** (19.88 -> 15.70 us/draw, city scale) |
| cycle check folded into the memo (was a per-node vector scan) | ~5-8% | no |
| pipeline lookup memo (~200-byte key, exact compare, owner-scoped) | ~1.5 us | no |
| binding storage recycling (~18 allocs/draw -> ~0) | ~1 us | no |
| dynamic-state redundancy filter (11 vkCmdSet* per draw) | ~0.5-1 us | no |
| pipeline + index-buffer bind filtering | ~0.2-0.4 us | no |

Projection if the unmeasured ones land near estimate: **~12-13 us/draw ~= 27-30 fps** in the city.
Treat that as a hypothesis, not a result.

### Correctness notes for whoever tests this

Every one of these is a cache or a redundancy filter, and three such schemes were unsafe earlier
in the same session. If anything renders wrong, bisect in this order (cheapest to revert first):

1. `--cache-descriptors` is DEFAULT OFF and unproven - leave it off unless testing it specifically.
2. Dynamic-state / bind filtering: Vulkan dynamic state is per COMMAND BUFFER. The cache
   invalidates on handle change (`DynamicStateCache::Retarget`). If state leaks between buffers,
   this is the cause.
3. Pipeline memo: scoped to its owning PipelineCache and only valid because pipelines are never
   erased outside the destructor (verified).
4. Flat memo: scoped to a single evaluation, never reused across draws. Cross-draw reuse is what
   corrupted light coronas earlier - do not "optimise" that scoping away.

## New analysis: call-frequency, not total time

Every prior investigation ranked Tracy zones by total time and optimised the top one. Ranking by
**calls per draw** instead exposes redundancy that total-time ranking hides:

| zone | calls/draw | us/draw |
|---|---|---|
| TileGetTextureSize | **3.42** | 0.24 |
| ResolveRenderColorTarget | **2.76** | 0.47 |
| CpOpSetShaderReg | 1.80 | 0.07 |
| **RebindBuffers** | 1.73 | **1.90** |
| PrepareBindings | 1.73 | 1.07 |
| FindBuffers | 1.73 | 0.47 |
| RebindImages | 1.71 | 0.27 |

`RebindBuffers` at 1.90 us/draw is the largest single item outside the SRT evaluator, and had
never been examined - the whole session went into shader resource resolution instead.

### What ObtainBuffer does per buffer, per draw

1. `IsRegionGpuModified` + `IsRegionCpuModified` - each takes a lock and iterates regions
2. for small read-only buffers: `m_stream_buffer.Map()` + **`TryReadBacking` (a memcpy of the
   guest data)**
3. slot lookup / `FindBuffer`
4. `TouchBuffer`, `SynchronizeBuffer`

### Latent inefficiency worth someone's attention

The stream path never clears the CPU-dirty flag. The normal upload path does, via
`ForEachModifiedRange<DirtySource::Cpu, true>`. So once a small uniform buffer is marked
CPU-modified it stays marked, and **its contents are re-copied on every draw that binds it**,
whether or not the guest wrote to it again.

Clearing that flag safely requires re-arming page protection so the next guest write is still
detected - get it wrong and buffers go stale. Not attempted; flagged as the highest-value
remaining idea in this area.

### Landed instead (safe subset)

Per-draw dedup of buffer resolution. Vertex and pixel stages bind the same buffer frequently and
`RebindBuffers` runs once per stage, so the identical `ObtainBuffer` request was served twice per
draw. Scoped to exactly one draw (`BeginDrawBufferScope`), because between draws the guest may
write the memory and the answer can legitimately change. Written buffers are never cached.
Both entry points reset the scope - the compute path reaches PrepareBindings without going
through PrepareGraphicsBindings and would otherwise inherit the previous draw's entries.

## 2026-09-02: the structural finding - GPU-driven rendering is expanded on the CPU

`CommandProcessor::DrawIndirectMulti` (graphicsRun.cpp) handles `DRAW_INDEX_INDIRECT_MULTI` by
reading the count out of guest memory on the CPU and then looping:

```cpp
draw_count = *count_addr;          // CPU reads memory a GPU compute pass wrote
for (uint32_t i = 0; i < draw_count; i++) { /* one individual draw each */ }
```

On a PS5 this packet is consumed by the hardware command processor: one packet, N draws, zero
CPU cost, count never leaves the GPU. Here, one packet becomes N draws each paying the full
~16us translation. **This is the answer to "why does a weaker PS5 hold 60fps" - the console
never performs the work we perform.** GPU utilisation is 22%; the GPU was never the limit.

**Vulkan has the native path: `vkCmdDrawIndexedIndirectCount`, core since 1.2. This project
targets 1.3 (`VULKAN_TARGET_API_VERSION`). It is available and unused.**

This is NOT the batching that was ruled out. That was merging independent consecutive draws
with distinct state (run length 1.00, correctly dead). An indirect-multi batch shares state by
construction - the game already declared it as one packet.

**NEXT STEP, one drive, before building anything:** `IndirectMultiCensus` now prints to stdout
(it existed but used LOGF, which is silenced in this build, so it had never been seen). It
reports calls / zero_count / draws_issued. That fraction decides whether native passthrough is
the biggest available win or a footnote.

### Why threads do not help yet
Only `MaterializeResources` (3.80us) is pure, and it still will not split: all requests in a
draw share two `Evaluator` objects and splitting destroys the cross-request memo reuse that is
the only verified win (-21%). Everything else goes through `bufferCache` (no lock at all) or
`textureCache` (spin lock). Real parallelism = parse/prepare/record pipelining, which requires
making those caches concurrent first. Frame cost then becomes max(stage) not sum(stage);
recording becomes the wall at ~3.2us/draw ~= 14ms at 4450 draws.

### Also closed this session
- Occlusion query index recycling FIXED - see [[gta5-occlusion-queries-broken]]. Cost fps
  (17.8 -> 14.0) because the old number came from wrongly culled geometry.
- PM4 predication: not a lever, never reached 20k eligible packets in a session.
- Render distance: raising it adds draws; +19% draws measured -1.1 fps.
