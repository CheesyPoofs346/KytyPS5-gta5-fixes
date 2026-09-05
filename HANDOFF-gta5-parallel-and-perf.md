# Handoff — KytyPS5 / GTA V, parallelism and performance

Branch `perf-occlusion-fix`. Supersedes `HANDOFF-gta5-cpu-and-perf.md` for everything about
parallelism; that document's §0 rules and dead-end table still apply.

**Bottom line: per-draw work is memory-bandwidth bound, not lock bound and not CPU bound. The
parallel architecture now works correctly and does not scale. Four independent measurements agree.**

---

## §0 Rules (additions to the previous handoff — all learned the hard way)

1. **Never launch the exe directly. Always `./run-gta5.sh`.** It backs up the save and refuses to
   launch if slots are missing; launching from elsewhere starts a new game in North Yankton.
2. **The pipeline cache is invalidated by every rebuild** unless the fix below is present. A cold
   cache costs **4.4 µs/draw (~29% of frame time)** in `CreateGraphicsPipeline`. **Any A/B where one
   side was freshly built is contaminated.** This invalidated at least three comparisons here.
3. **`µs/draw` does not normalise scene differences.** A lighter scene can have more expensive
   draws. Only within-run phase percentages are safe to compare across runs.
4. **`git stash -u` stashes 209 untracked files including `run-gta5.sh`.** Use the worktree at
   `C:/Users/konze/kyty-baseline` for baseline builds.
5. **The test suite has a pre-existing flake** (`GpuCommandLane`, ~1 in 5). Run it 3–5 times.
6. Heredoc `\n` escapes get eaten by this shell and land as literal newlines inside string
   literals. Use the edit tool for format strings.

---

## §1 What landed and works

### Pipeline cache persistence (the largest single win)

`DriverCacheSignature` included `BuildFingerprint()` — the executable's **file size and last-write
timestamp** — so every rebuild discarded the whole cache. A Vulkan pipeline cache blob is only
invalidated by driver/device identity, which was already in the signature; if our SPIR-V changes the
driver simply misses and recompiles, it cannot produce wrong output.

```
CreateGraphicsPipeline  4.874 -> 0.457 µs/draw
TOTAL DrawIndex        14.202 -> 9.765 µs/draw
```

### RCU on VirtualRanges (§2) and atomic dirty bits (§3)

Readers take an immutable `shared_ptr` snapshot and no lock; writers mutate `m_ranges` under the
existing lock and publish a copy. All 47 mutation sites untouched.

```
ClampCensus miss=6848 cyc (lock=6595 work=619)  ->  miss=669 cyc (lock=42 work=627)
ClampRangeSize   0.723 -> 0.156 µs/draw
tracker queries  0.670 -> 0.538 µs/draw
```

**Boot completes.** `std::shared_mutex` here froze the game — Windows SRWLock is not
starvation-free and 8 readers starved the guest allocator (44 draws/frame vs 4850). RCU has no
reader lock at all, so there is nothing to starve with.

**Payoff was ~2%.** `TOTAL DrawIndex` moved 9.952 → 9.765 despite the two phases giving back 0.70 µs.
The rest was absorbed, as in every previous attempt.

### Current() audit (§4)

All 39 `Current()` sites in the renderer classified. **`CreateBuffer` was the dangerous one**: it
looks assertion-only but does `buffer.CopyFrom(command, ...)`, recording into the primary, and was
reachable from a worker via `ObtainBuffer -> FindBuffer`. That is silent corruption, not an abort.
Only found because the compiler flagged an unused variable after the assertion was removed.

Recorders (`TryDownloadImage`, `Image::Upload/Download/CopyImage/Resolve/CopyMip`,
`TileManager::Record`) now assert on workers. Assertion-only sites (`FindImage`, `NativeUpload`,
`CreateBuffer`'s precondition) gate on `!MustStageForWorker()`.

### Two-phase acquisition/binding split (§2b) — commit `a958dfa`

Three attempts to run the whole resolve on workers aborted, each on a different creation route. The
cause was structural: `PrepareGraphicsBindings` conflated **acquisition** (create if missing,
allocate VRAM, upload, record a CopyFrom) with **binding** (turn what exists into descriptor
writes). Only the second is parallel-safe.

The split falls on existing function boundaries, which is what made it small:

```
phase 1   serial     shader permutation lookup
phase 2a  PARALLEL   SRT walk
phase 2b  serial     PrepareBindings + FindBuffers   <- creates what is missing
phase 2c  PARALLEL   RebindBuffers + RebindImages    <- every lookup hits
phase 3   serial     record
```

Every creation tripwire stays armed. **Result: no aborts, 5088 draws/frame, serial recording fell
from 77.6% to 42.3%, `TOTAL DrawIndex` 9.765 -> 6.700 µs.** The refactor did exactly what it was
designed to do.

---

## §2 Why it still does not reach 30 fps

Phase 2c got **slower** with more workers:

```
                  2 workers    8 workers
phase2a_walk       16.1%   ->   9.9%
phase2b_acquire    13.9%   ->  14.5%
phase2c_bind       14.9%   ->  22.7%    <- negative scaling
phase3_record      44.8%   ->  42.3%
bailouts            7,747  -> 24,448
```

`RunnerCensus` shows the work distributed evenly — caller 1.71M items, workers 1.41–1.43M each,
under 2% spread. Not starvation, and the locks are gone.

**Mechanism:** `RebindBuffers` is dominated by `StreamBuffer::Copy` — memcpy into host-visible,
**write-combined** GPU memory. Each worker has its own ring so there is no software contention, but
they share one memory controller and WC write-buffer path. Past saturation, more threads means more
contention rather than more throughput.

**Four independent measurements now agree the per-draw work is memory-bound:**

| experiment | result |
|---|---|
| SRT walk, 8 workers, even split | ~2.4× at best, **0 fps** |
| texture-cache shared lock | **0 fps** |
| VirtualRanges RCU (lock 6595 -> 42 cycles) | **~2%** |
| phase 2c, 2 -> 8 workers | **negative** |

A 9700X has the cores. It does not have the memory bandwidth to push 4,200 draws' worth of
descriptor and stream-buffer traffic eight ways at once.

**Do not spend more time on parallelism.** The remaining serial phase (phase3_record, 42.3%) would
hit the same wall — recording is `vkCmd*` calls writing into command buffer memory.

---

## §3 Draw-count culling — tried and REMOVED

Frame time is linear in draw count, so cutting draws is arithmetically the only remaining lever.
It was implemented (`--cull-small-draws N`, skipping draws with an index count between a
fullscreen-quad guard of 32 and N) and **removed after testing.**

```
threshold 96 -> 13.5% of draws culled -> UNPLAYABLE
```

Roads, pavements and building floors disappeared. **Index count is a proxy for triangle count, not
for screen area, and the two are inversely related for exactly the geometry that matters.** A road
surface or building floor is an enormous flat mesh made of very few triangles, so a
"cull small draws" band targets the largest, most structural surfaces in the scene first. The
premise was backwards.

Any future attempt at draw reduction needs a real size signal - projected bounding-box area, or a
distance derived from the draw's transform - not the index count. That data is not currently
available at the point where a draw could be cheaply skipped.

## §4 Useful flags

```
--draw-profile true            per-phase breakdown, FrameThreads, ClampCensus, DrainCensus
--draw-workers N               worker pool size
--test-parallel-bindings true  the 2b/2c split (works, does not scale)
--log-ui-draws true            UI quad geometry (samples into gameplay)
--skip-ps-chksum 0xHASH        drop draws by pixel-shader CHECKSUM
```

**Trap:** `ShaderCensus` prints its values next to `--skip-ps`, but that flag matches `data_addr`
while the printed values are `chksum`. Pasting the census line verbatim silently matches nothing.

---

## §5 Still open

**Minimap renders ~4× oversized, overflowing the rectangle its own HUD bars mark out.** Confirmed
pre-existing — reproduces on `435cba4` in a clean worktree, so nothing in this work caused it. **Four
theories checked and all four wrong**: save-file corruption, `PROFILEB`, a 2560 vs 3840 resolution
mismatch, and a fresh profile. Viewport and scissor come straight from guest registers with no
scaling anywhere in that path.

Next step is a RenderDoc capture (`--rd`, tooling in `_RenderDoc/`) with the minimap on screen, to
identify the draw outright instead of theorising. Do not guess a fifth time.
