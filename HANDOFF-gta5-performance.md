# GTA V performance — session handoff

Written 2026-09-02 at the end of a ~3-day session. Read this **before** touching anything.
It records what is established with measurements, what is ruled out, what is unverified,
and the landmines that have already cost days.

**Branch: `perf-occlusion-fix`** (6 commits ahead of `font-fix`, which is ahead of `main`).
Working tree is clean except `_SaveData/.../Profile` and `sce_param.bin`, which are runtime
state and should stay uncommitted.

---

## 0. STANDING USER CONSTRAINTS — these are not negotiable

1. **ALWAYS launch via `./run-gta5.sh`. NEVER launch the exe directly.**
   The emulator resolves `_SaveData` **relative to the working directory**
   (`src/libs/libSaveData.cpp`: `SAVE_DATA_DIR = "_SaveData"`). Launching from anywhere else
   finds no save slots, creates a blank profile, and starts a **new game in North Yankton**.
   The user has said, in caps, more than once, that this must never happen. `run-gta5.sh`
   does `cd "$ROOT"` and then *refuses to launch* if no `SAVEDATASGTA5*` slots are visible.
   It also backs the save up to `C:/Users/konze/KytyPS5-SaveBackups/` on every launch.

2. **Do not launch the game unless the user explicitly says to.** They test manually and have
   done 20+ runs already. Reading logs and code is expected; launching is not, unless asked.

3. **The user's save is old but deliberate.** Do not suggest starting fresh.

4. **Ignore whatever "codex" is working on** — the user has another agent in this repo.

5. The user is direct and gets frustrated with over-promising. **Give calibrated numbers, say
   what is measured vs projected, and report failures plainly.** See section 8.

---

## 1. Where performance actually stands

Measured in Los Santos, steady state, `--present-mode Mailbox`, occlusion queries ON:

```
draws/frame   median 4448   (p10 3851, p90 4645)
fps           median 14.00  (min 9.8, max 16.9)
us/draw       median 15.96
```

Machine: Ryzen 9700X, 16 threads. CPU ~29%, **GPU ~22%** during gameplay.

**The single relationship that has held across every measurement, for three days:**

```
frame time = draw count x ~16 us,  single-threaded
```

There is no hidden stall. The cost is per-draw CPU work on one thread.

| target | budget at 4450 draws |
|---|---|
| 60 fps | 3.7 us/draw |
| 30 fps | 7.5 us/draw |
| today  | 16.0 us/draw |

**30 fps requires cutting per-draw cost by ~53%.** Everything verified so far totals ~21%.

### Framerate went DOWN this session, on purpose

`17.8 fps -> 14.0 fps`. This is not a regression to fix. The old number came from broken
occlusion queries returning zero (= "occluded"), so the guest was culling geometry it should
have drawn — including the sun. Fixing the queries made rendering correct and the workload
honest. Draws went `3470 -> 4448` for the same reason. **Do not "restore" the old behaviour.**

---

## 2. THE core insight (this is the answer to "why is a PS5 faster")

A PS5 game writes **PM4 command packets** into a ring buffer and a **hardware command
processor** consumes them in silicon. CPU cost per draw ~= 0.

This emulator must translate every packet into Vulkan: parse PM4, decode GCN/RDNA registers,
walk SRT expression trees to resolve descriptors, look up/compile pipelines, resolve buffers
and textures through caches, build descriptor sets, record a Vulkan draw. **~16 us of CPU per
draw that the console spends nothing on.**

The GPU is at 22% utilisation. **It has never been the bottleneck.** The comparison is not
their GPU vs our GPU — it is *dedicated command-processor silicon* vs *a software translation
layer*.

Full writeup (same content, nicer to read):
https://claude.ai/code/artifact/43dc27be-5255-440a-b74b-1a56edae5b09

---

## 3. THE NEXT ACTION — do this before building anything

### Read `IndirectMultiCensus`

`CommandProcessor::DrawIndirectMulti` (`src/graphics/guest_gpu/graphicsRun.cpp`, ~line 1013)
handles `DRAW_INDEX_INDIRECT_MULTI` — GPU-driven rendering — by **expanding it on the CPU**:

```cpp
draw_count = *count_addr;          // CPU reads memory a GPU compute pass wrote
for (uint32_t i = 0; i < draw_count; i++) {
    // read args from memory, issue ONE individual draw each
}
```

On hardware that entire packet is free and the count never leaves the GPU. Here, one packet
becomes N draws each paying the full ~16 us.

**`vkCmdDrawIndexedIndirectCount` is core in Vulkan 1.2. This project targets 1.3**
(`VULKAN_TARGET_API_VERSION` in `src/graphics/host_gpu/graphicContext.h`). The native path is
available and completely unused.

A census already existed in that function but used `LOGF`, which is **silenced in this build**,
so it had **never printed once**. It now uses `std::printf`. It reports:

```
IndirectMultiCensus: calls=N zero_count=N draws_issued=N from_count_addr=0|1
```

**That fraction decides whether native passthrough is the biggest win in the project or a
footnote.** One city drive answers it. Do not build the passthrough before reading it.

**Important:** this is NOT the draw batching that was ruled out. That was merging *independent
consecutive* draws with distinct state (100,000 draws -> 100,000 distinct states, run length
1.00 — correctly dead). An indirect-multi batch shares state **by construction**: the game
already declared it as one packet with one pipeline. Different mechanism entirely.

---

## 4. Ranked opportunities

| # | Change | Expected | Risk | Notes |
|---|---|---|---|---|
| 1 | Read the indirect census | — | none | free, sizes #2 |
| 2 | Native `vkCmdDrawIndexedIndirectCount` passthrough | removes whole draws | medium | magnitude set by #1; does **not** need the cache refactor |
| 3 | Bindless descriptors (descriptor indexing) | ~2–2.5 us/draw | medium | also removes most per-draw cache touching → enables #5 |
| 4 | Stream-buffer dirty-flag fix | ~1.9 us/draw | **high** | see section 6 |
| 5 | parse/prepare/record pipelining | ~16 → ~9 us/draw | **very high** | needs #6 first |
| 6 | Make bufferCache + textureCache concurrent | prerequisite for #5 | **very high** | riskiest code in the renderer |
| — | Trimming the "cheap" translation steps | ~1–1.5 us | low | not worth the effort |

**Stacking 2 + 3 + 5 is the only combination that plausibly reaches 30 fps in the dense city.
No single one gets there.**

### Why threads don't help today (checked, don't redo this)

- Only `MaterializeResources` (3.80 us/draw) is pure. **It still won't split**: every request
  in a draw shares two `Evaluator` objects (`SrtWalker.cpp` ~line 1081), each construction
  zeroes a ~6 KB memo, and splitting destroys memo reuse *across* requests — which is the only
  verified optimisation in the project (−21%).
- Everything else goes through `bufferCache` (**no lock at all**) or `textureCache`
  (`TrackingSpinLock m_lock`). Throwing threads at those gives data races or contention.
- The existing `--parallel-resolve` worker returned **2.7%** because it splits *one draw*
  across two threads, pays a fixed ~1 us handoff, then immediately blocks on `Wait()`.
  It behaved exactly as designed; the design was the ceiling.

### Per-draw cost breakdown (Tracy, 41M zones, 20s driving)

| stage | us/draw | parallel-safe |
|---|---|---|
| MaterializeResources | 3.80 | pure, but unsplittable (above) |
| ExecutePreparedDraw + DrawIndex | 3.22 | no — recording is serial |
| RebindBuffers | 1.87 | no — bufferCache |
| PrepareBindings | 0.95 | no |
| FindBuffers | 0.77 | no — bufferCache |
| CommitBindings | 0.56 | no |
| RebindImages | 0.37 | no — textureCache spin lock |

---

## 5. What landed this session

All on `perf-occlusion-fix`, each independently revertable:

| commit | what | verified? |
|---|---|---|
| `62b9665` | **Occlusion query index ownership** | YES — 100% success over 32,000 queries |
| `10b4624` | Buffer dedup cache key (was missing 3 of 5 inputs) | correctness only |
| `32dd6a4` | A/B flags + F6 harness + health counters | works |
| `3c13142` | Carry-over of pre-existing uncommitted work | as-is |
| `331aeb0` | Research notes + `IndirectMultiCensus` to stdout | — |
| `baaf110` | **Tile layout memo** + scratch vector reuse | correctness YES, **speed UNMEASURED** |

### `62b9665` — the real fix

`OpenSegment` allocated query pool indices with `m_next = (m_next + 1) % QueryCount` and
tracked nothing in flight. Results are read later on the **deferred-operation thread** with
`eWait`. Once the counter wrapped, `resetQueryPool` ran on slots whose read hadn't happened —
so "successful" queries returned zero, which reads as "occluded".

Now: mutex-guarded free list, indices released **only inside the deferred read callback**,
after the value is consumed. Acquire failure sets `m_overflowed` and falls back to "visible".

Verified: `OcclusionHealth: total=32000 ok=32000 (100.0%)`.

### `baaf110` — needs measuring

`TileGetTextureSize` is pure arithmetic over 5 scalars but runs **3.42x per draw** at 0.24 us
(~0.82 us/draw, ~5% of frame). Memoised into a 16-entry thread_local table (~8.7 KB — kept
small deliberately; an earlier memo here measured *worse* at 24 KB than 6 KB once it left L1).

Two behaviours preserved on purpose — read these before touching it:
- the unknown-format `EXIT` still fires **only when the caller passed a non-null `total_size`**,
  via an explicit `report_unknown` flag;
- **failed computations (total size 0) are never cached**, so failures still reach the `EXIT`
  instead of being handed a silent zero-size layout.

Test `TileSizeMemo` drives 32 distinct keys through 16 slots interleaved (exercises eviction
AND slot collisions) and checks partial vs full requests. It was watched failing before being
trusted to pass.

---

## 6. Known-real but NOT shipped

**Stream buffer never clears the CPU-dirty flag.** `BufferCache::ObtainBuffer`
(`src/graphics/host_gpu/renderer/cache/bufferCache.cpp` ~line 462):

```cpp
if (!is_written && size <= CACHING_PAGESIZE &&
    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
    auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
    if (mapped != nullptr && TryReadBacking(vaddr, mapped, size)) { ... }
}
```

The fast path fires only when the region **is** CPU-dirty and never clears the flag. So once
the guest writes a uniform buffer, **every subsequent draw re-maps and re-memcpys it forever**.
`RebindBuffers` is 1.87 us/draw and this is likely most of it.

Proper fix: after copying, clear the CPU-dirty flag (requires re-arming page protection) and
remember `(vaddr,size) -> stream offset`, reusing it while the guest hasn't written. Watch the
stream ring wrapping per frame.

**Deliberately not shipped** — it is the exact class of change that corrupted rendering three
times this session, and it could not be measured with the user away. Do it with a run available.

---

## 7. DEAD ENDS — closed with data, do not re-investigate

| theory | what killed it |
|---|---|
| Draw batching / merging | 100k draws -> 100k distinct states, run length 1.00 |
| Occlusion-driven culling as an fps lever | 99.6% of queries report occluded, guest **still** issues 4450 draws — it isn't culling on them |
| PM4 predication | `PredicationHealth`/`PredicationSet` never reached print thresholds (20k eligible packets / 2k calls) in a full session |
| Thread pool over MaterializeResources | shared evaluators + memo reuse make it unsplittable |
| Raising render distance | adds draws; +19% draws measured −1.1 fps |
| Texture streaming CPU | 0.4 ms/frame (detiling is GPU-side) |
| Garbage collection | 4.6 us per run |
| Page protection / VirtualProtect | 0.3 ms/frame |
| Buffer-upload render-pass breaks | 12.6 per frame |
| Pipeline cache lock contention | 0.03 us |
| Forced readback in PrepareStorageSampledOverlap | 50 drains x 0.08 ms |
| Whole-snapshot materialization cache | 4–12% hit rate |
| Any single pixel shader dominating | none above 19% of draws |
| GPU being the limit | 22% utilised |

---

## 8. LANDMINES — every one of these has already cost time

1. **`LOGF` is silenced in this build.** Only `std::printf` reaches stdout. An entire
   instrumentation round produced zero output because of this. Check `grep -c` on the log
   before believing "the counter says nothing happened".

2. **Three caches corrupted rendering this session**, all the same way: *the key omitted an
   input that changes the answer.*
   - evaluator memo reused across draws -> light coronas rendered as huge red blobs
   - "redundant" global barrier skipping -> same red blobs (copies/uploads/clears needed them)
   - descriptor cache keyed on a recycled raw `Program*` -> crash "image descriptor 0 has 4 dwords"
   - buffer dedup keyed on 2 of `ObtainBuffer`'s 5 inputs (fixed in `10b4624`)
   **Rule: before adding any cache, list every input the cached function reads and prove the
   key contains all of them.** Verify visually in a light-heavy scene (night, under an overpass).

3. **Command buffer handles are pool-RECYCLED.** An unchanged `VkCommandBuffer` handle does not
   mean the same recording. Use `CurrentCommandGeneration()` (added in `context.cpp`), not the
   handle. This was caught by review, not by tests.

4. **Three "wins" measured as noise or were fake.** Parallel resolve estimated 10%, delivered
   2.7%. GC fix estimated major, delivered 0. Two large "wins" were corrupted rendering.
   See memory note `perf-measurement-discipline`.

5. **A/B harness must key on something stable.** Two harnesses measured nothing because they
   keyed on a per-run guest address instead of the stable `chksum`, and because counters sat
   *before* the skip check.

6. **`KYTY_PM4_LEN(header)` already includes the header.** Adding `+1` overruns the buffer
   (crashed at `graphicsRun.cpp:870`).

7. **Editing files from Python: several source files are CRLF.** Multi-line `str.replace`
   patterns with `\n` silently fail to match. Use line-index surgery, or split/join on `\n`
   preserving `\r`. Also, bash heredocs eat `\n` inside Python string literals — use
   `BS = chr(92)` and concatenate, or the emitted C++ gets a literal newline inside a string
   and won't compile.

8. **A git reset once wiped a day of uncommitted work.** Commit early. That is why
   `3c13142` exists.

9. **The exe is locked while the game runs** — `ninja` link fails. Close it first
   (`taskkill //PID <pid> //F`), and verify a save backup exists before killing.

10. **Ambiguous build freshness.** Exe and source timestamps can share a minute. To confirm a
    rebuild actually happened: `touch` the source, rebuild, and check ninja lists the
    `.obj` and the link. Verify a string landed with `grep -a "<marker>" <exe>`.

---

## 9. How to build, test, run

```bash
# build (from repo root)
cd _Build/profile-buffer-readback && ninja kyty_emulator

# tests: expect 53 "ok". UnifiedTextureCacheFlow fails and is PRE-EXISTING, unrelated.
./_Build/profile-buffer-readback/shader_recompiler_compute_tests.exe 2>&1 | grep -c " ok$"

# run — ALWAYS this script, never the exe directly
./run-gta5.sh --present-mode Mailbox > run.log 2>&1
```

`--present-mode Mailbox` matters: the default FIFO quantises framerate to 60/N, displaying
~19 fps of real work as a hard 15. Biggest free win found, costs nothing.

### Runtime flags

| flag | default | purpose |
|---|---|---|
| `--real-occlusion-queries` | script forces `true` | config default is `false` |
| `--dyn-state-cache` | true | filters redundant `vkCmdSet*` |
| `--pipeline-memo` | true | single-entry pipeline key memo |
| `--buffer-dedup` | true | per-draw buffer resolution dedup |
| `--cache-descriptors` | false | descriptor cache — was crashing, now guarded |
| `--parallel-resolve` | false | the 2.7% two-thread worker |

### F6 A/B harness

In-game, **F6** cycles 5 configurations, prints the average fps of the segment it is *leaving*,
then resets the average. **Hold each step 25–30 seconds** — a previous run lost 11 of 13
segments to fast presses.

```
step 0 ALL ON (baseline) | 1 dyn-state OFF | 2 pipeline-memo OFF
step 3 buffer-dedup OFF  | 4 ALL OFF
```

Other keys: F2 pause, F4 fps overlay, F5 reset fps average, F1 RenderDoc capture, F11 fullscreen.

### Log analysis snippet (reuse it, don't rewrite it)

```python
import io, re, statistics
frame_re = re.compile(r'Frame:\s+(\d+) draws/frame,\s+([\d.]+) ms \(([\d.]+) fps\),\s+([\d.]+) us/draw')
rows=[]
for ln in io.open('run.log','r',encoding='utf-8',errors='replace'):
    m=frame_re.search(ln)
    if m: rows.append((int(m.group(1)),float(m.group(3)),float(m.group(4))))
g=[r for r in rows if r[0]>500]          # drop loading frames
s=g[len(g)//3:]                          # steady state only
print("draws=%5.0f fps=%5.2f us/draw=%5.2f"%(
    statistics.median([r[0] for r in s]),
    statistics.median([r[1] for r in s]),
    statistics.median([r[2] for r in s])))
```

Segment by `[F6 step N]` markers to attribute per-configuration numbers, and **discard the
first 2 frames after each toggle** (cold caches, and the print itself costs a frame).

---

## 10. Open bugs not related to performance

1. **99.6% of successful occlusion queries return ZERO samples.** Not a plumbing problem any
   more — the reads succeed. The game is being told almost nothing is visible, which is
   implausible for a dense city. This project has a history of depth bugs with exactly this
   flavour (depth clear wiping the Z-prepass; HTile RMW misread as a fast clear). **Suspect
   the depth buffer state when the occlusion test draws run.** Will not buy fps (the guest
   isn't culling on them) but it is a real correctness bug.

2. **Palm trees and thin power-line cables disappear up close.** 46.5k draws dropped, all with
   `VGT_SHADER_STAGES_EN` stage mask `0x0200210d`, primitive type 9 (patch list) — i.e.
   **tessellation**. Needs hull/domain shader support. See `HANDOFF-gta5-invisible-models.md`.

3. **Missing fog.** Deferred, never investigated.

4. **UI text drops glyphs.** See `HANDOFF-gta5-text-glyphs.md`. Atlas and 16-bit shader
   semantics both ruled out; next step is a RenderDoc glyph comparison.

---

## 11. Memory notes worth reading

In `C:\Users\konze\.claude\projects\C--Users-konze-KytyPS5-brandostrong\memory\`:

- `gta5-gpu-driven-draws` — the structural finding (section 3 here)
- `gta5-occlusion-queries-broken` — the fix and the remaining 99.6%-zero mystery
- `gta5-perf-root-cause` — draws x 16-20us, 11 eliminated theories
- `perf-measurement-discipline` — three fake wins in one day; verify mechanism AND correctness
  before reporting a number
- `gta5-missing-models-stage-mask` — the tessellation lead

Also `PERF-FINDINGS.md` in the repo root — longer form, same material, and the bisect order if
something renders wrong.

---

## 12. If you do one thing

Ask the user for one city drive, read `IndirectMultiCensus` and the tile-memo effect on
`us/draw` from the same log, and report both numbers plainly. That single run resolves the
biggest open unknown in the project *and* validates the last commit. Everything in section 4
should be planned around what that census says — not around a guess.
