# Handoff — KytyPS5 / GTA V performance

Branch `perf-occlusion-fix`. Goal: a stable 30 fps. **Currently ~16.4–17.5 fps and no change has
moved it.** Read §0 before proposing anything.

---

## §0 Rules that cost us real time to learn

1. **Never launch the exe directly. Always `./run-gta5.sh`.** The emulator resolves `_SaveData`
   relative to cwd; launching elsewhere finds no save slots and starts a new game in North Yankton.
   The script backs the save up first and refuses to launch if slots are missing. A baseline
   variant exists: `run-gta5-baseline.sh` (same guard rails, different exe).
2. **Do not change `--screen-width/--height`.** Dropping to 640×360 lost the Vulkan device
   (`vkQueueSubmit ErrorDeviceLost`) during load. Use `--skip-ps-chksum` to remove GPU work instead.
3. **`LOGF` is silenced in this build.** Only `std::printf` reaches stdout. Several existing
   diagnostics (`SceneViewport`, `BandScissor`) have therefore never printed once.
4. **The test suite has a pre-existing flake.** `GpuCommandLane` fails ~1 run in 5, including on
   `435cba4`. Run the suite 3–5 times; do not read one green run as proof.
5. **A counter's name is not evidence of what it measures.** Two full design cycles were wasted on
   this — see §4.
6. **Heredoc `\n` escapes get eaten** by the shell used here; they land as literal newlines inside
   string literals and break the build. Use the edit tool for format strings.
7. `git stash -u` in this repo stashes **209 untracked files** including `run-gta5.sh`. Use a
   worktree for baseline builds. One already exists at commit `435cba4`.

---

## §1 Where the frame actually goes (measured 2026-09-04)

Median gameplay frame: **4223 draws, 61.1 ms, 16.4 fps.**

```
FrameThreads: 65.0 ms/frame | guest blocked 62.4 ms (96%) | gpu busy 63.3 ms (97%) | headroom 1.6 ms
  blocked-queue polls: 1.3/frame costing 0.90 ms/frame (1% of wall)

TOTAL DrawIndex        9.952 us/draw   ->  ~41 ms/frame   (~62%)
PM4 non-draw packets   6.210 us/draw   ->  ~26 ms/frame   (~39%)
                                            sum ~67 ms, matches the observed frame
```

This accounting **closes for the first time**. It did not before, because PM4 non-draw was
double-counting every draw (§4).

### Why the CPU sits at 20–30%

Not lock stalls, not idle waiting. **One thread is pegged at 96–98% and the other fifteen have no
work.** `GuestGpu` owns exactly one `std::jthread`, and it does PM4 parsing, draw translation,
Vulkan recording and submission — one submission at a time, in order. The guest thread is blocked
96% of the frame because it genuinely has almost nothing left to do.

Consequence: **the only way to use more cores is to split the work on that one thread.** Every
attempt so far has failed (§3).

---

## §2 The two live targets

### (a) PM4 non-draw — ~26 ms/frame, 39%, never examined

~20,000 non-draw packets per frame at ~1.3 µs each, on the single GPU thread. This is now the
largest unexamined cost in the emulator, larger than anything inside a draw. Nobody has looked
because the number was nonsense until `6517df5`.

**Start here.** Nothing is known about the distribution — which packet types dominate, whether any
are redundant, whether register writes could be batched. The first step is a per-opcode histogram in
the packet dispatch loop (`graphicsRun.cpp` ~line 908, where the `Pm4NonDraw` timer sits).

### (b) `PrepareGraphicsBindings` — 3.4–3.8 µs/draw, ~38% of a draw

```
PrepareGraphicsBindings 3.752
  PrepareBindings       1.063
  RebindBuffers         1.489
    buffer loop         1.245
      tracker queries   0.670   <- IsRegionGpuModified + IsRegionCpuModified
      SynchronizeBuffer 0.218
      FindBuffer        0.117
    NativeUpload x2     0.123
  FindBuffers           0.851  (ClampRangeSize 0.723)
  RebindImages          0.423
```

`tracker queries` are ~11 per draw at ~230 cycles each. Measured **uncontended** (single-threaded
run), so most of that is `Iterate` + region bitmap scan, not lock contention — a lock-free rewrite
would recover little. Two cheap unexplored items:

- The two queries run back-to-back on the **same range**, each doing its own traversal + lock +
  scan. Fusing them into one traversal is mechanical, worth maybe 0.3 µs/draw.
- Nobody has checked how often the fast path they gate actually **fires**. If it rarely does, the
  fix is not to make the queries cheaper but to stop making them.

---

## §3 Dead ends — measured, do not repeat

| attempt | result |
|---|---|
| Parallel SRT walk, 8 workers | even split, ~2.4× at best, **0 fps**. Memory-bound. |
| Texture-cache shared lock | **0 fps**. |
| `PrepareGraphicsBindings` into phase 2 | **17.10 fps at 2 workers and 17.10 at 8.** Identical. |
| Descriptor caching | measured a net **loss**. |
| `VirtualRanges` `std::shared_mutex` | **froze the game.** Windows SRWLock is not starvation-free; 8 readers starved the guest allocator's `unique_lock` and the game never left boot (44 draws/frame vs 4850). Reverted. |
| Skip no-op `Protect` writes | **hung at the gameplay transition.** Reverted, cause never established. |
| Parallel secondary command recording | abandoned pre-implementation: all Vulkan recording is **9.5%** of a draw. |
| `--frame-pipelining` (built, default off) | headroom measured at **1.6–2.3 ms of 65 ms ≈ 3%**. Not worth its risk. |
| 41% of draws deleted via `--skip-ps-chksum` | **no fps change.** Proves **not GPU-bound.** |

**`--test-parallel-bindings` is unsafe.** It aborted a run on
`TextureCache: image lookup requires a valid command buffer`. There is an unaudited class of
"requires a valid command buffer" preconditions reachable from workers; `bufferCache.cpp:617` is
fixed, `textureCache.cpp:1347/1691` are not, and `textureCache.cpp:1784/2105/2367` use
`m_scheduler.Current()` with **no guard at all** — those would record into whatever buffer happens
to be current, i.e. silent corruption rather than a clean abort. Keep the flag off.

---

## §4 The measurement trap that cost two design cycles

`DrainCensus` prints `phase3_record`. That field times the **whole serial `DrawIndex` call**,
derivation and recording together — not Vulkan recording. Reading the label literally produced a
complete design document for parallel secondary command buffer recording, which targets 9.5% of a
draw. Killed by `--draw-profile` before implementation.

Similarly `PM4 non-draw` read 44.6 ms/frame beside `DrawIndex`'s 39.9 ms inside a 57.2 ms frame —
impossible, because `IT_INDIRECT_BUFFER` recurses into nested draws and the timer wrapped them.

**Check what a timer actually encloses before trusting its name.**

---

## §5 Open, unrelated to performance

**Minimap renders ~4× oversized** (screenshot: map quad huge, icons at correct small positions).
Confirmed **pre-existing** — reproduces on `435cba4` built in a clean worktree, so none of this
session's commits caused it. Two theories were checked and **both were wrong**: the 360p run changed
only a counter and a checksum in `Profile` (5 bytes), and `PROFILEB` holds `2560`, correct.

`--log-ui-draws` dumps viewport/scissor/framebuffer for every ≤64-index quad. **Its 400-line cap
fills during boot**, so raise the cap or gate it on a keypress to catch the HUD in gameplay. It
discriminates: viewport and scissor come straight from guest registers with **no scaling anywhere**
in that path, so if the map's viewport is already ~4× too large the guest was handed the wrong
display size; if it is correct the fault is downstream in the composite.

---

## §6 Useful flags

```
--draw-profile true          per-phase draw breakdown + FrameThreads + ClampCensus
--log-ui-draws true          UI quad geometry (cap fills at boot, see §5)
--frame-pipelining true      guest runs ahead; ~3% ceiling, default off
--pipeline-depth N           bound on how far ahead (default 4)
--skip-ps-chksum 0xHASH      drop draws by pixel-shader CHECKSUM
--draw-workers N             worker pool size
--test-parallel-bindings     UNSAFE, see §3
```

**Trap:** `ShaderCensus` prints its values next to `--skip-ps`, but that flag matches `data_addr`
while the printed values are `chksum`. Pasting the census line verbatim silently matches nothing and
looks like "no effect". Use `--skip-ps-chksum`.

---

## §7 Recommended next step

Profile the PM4 non-draw path (§2a). It is ~39% of the frame, entirely unexamined, on the thread
that is already the bottleneck, and it carries none of the concurrency risk that has produced two
hangs and two crashes. A per-opcode histogram is a few lines and one drive.

Do **not** start with more parallelism. Three attempts measured exactly zero, and §1 explains why:
the work is on one thread and the shared state it touches serializes anything you hand out.
