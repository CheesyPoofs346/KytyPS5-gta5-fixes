# Per-draw architecture — where the time goes and how to restructure it

Written 2026-09-02 after instrumenting the whole draw path. Everything in section 1 is
**measured**, stable to three decimals across 8–9 million draws per run. Everything in sections
3–5 is **projected** and labelled as such.

Companion to `HANDOFF-gta5-performance.md`, which holds the session history and landmines.

---

## 1. Where the time actually goes (measured, `--draw-profile true`)

```
RefreshShaders                  3.77 us   26%
PrepareGraphicsBindings         4.67 us   32%
  of which PrepareBindings      1.10
  of which FindBuffers          1.62
  of which RebindBuffers        1.52
  of which RebindImages         0.45
vertex+index buffers            1.29 us    9%
AcquireRenderTargets            0.69 us
Commit vb/desc/ib               0.65 us
CreateGraphicsPipeline          0.56 us
PrepareDrawRenderState          0.50 us
BeginRendering+bind             0.45 us
preamble+checks                 0.32 us
SetGraphicsDynamicParams        0.09 us
EmitDrawPrimitives              0.05 us    0.3%
TOTAL DrawIndex                14.84 us
UNACCOUNTED                     1.82 us   12%
```

Plus **~3 us/draw outside `DrawIndex` entirely** (frame time reports ~17.6 us/draw against a
14.8 us DrawIndex). That is PM4 parsing and register-write packets. Never profiled.

**`EmitDrawPrimitives` — the actual Vulkan draw call — is 0.05 us of 14.84.** 99.7% of the cost
of issuing a draw is translation. This is the handoff's core insight as a number.

### Run-to-run variance is ±1.2 us

TOTAL across seven runs: 14.29, 14.27, 14.71, 14.33, 14.26, 13.66, 14.84. **Any claimed win
smaller than ~1.2 us cannot be established by comparing TOTAL between runs.** Compare a phase
against its parent within one run instead. This invalidated one win reported during the session.

---

## 2. Why it costs this much: everything is re-derived every draw

The single structural fact behind every number above:

> The emulator recomputes the full draw state from guest registers and guest memory on every
> draw. A PS5 title sets state once and issues many draws against it; the command processor
> consumes only the *deltas*. Here, draw 2000 does exactly as much work as draw 1.

Concretely, per draw, for each of two shader stages:

- `PrepareProgram` zeroes a **~2 KB `ShaderVertexInputInfo`**, re-reads the shader's AGC
  metadata, re-walks the vertex attribute tables out of guest memory, and rebuilds the input
  layout (`shader.cpp:681`). For a shader that has not changed.
- `MaterializeResources` walks the SRT expression graph — **141 descriptor resolutions per
  draw**, ~27 ns each — to produce descriptors that are mostly identical to last draw's.
- `FindBuffers` resolves **6.2 buffer descriptors** against the guest range table.
- `RebindBuffers` re-derives every buffer view; `RebindImages` re-resolves every texture.
- The pipeline key is rebuilt and hashed; render targets are re-acquired.

Nothing is remembered between draws except a handful of small single-entry memos. **There is no
dirty tracking anywhere.** `HW::Context` (`hardwareContext.h:705`) is a plain struct of registers
with plain setters — no version, no dirty mask, no way for a consumer to ask "did this change?".

That absence is the root cause, and it is why individually optimising leaves has hit a wall: each
leaf is only 0.5–2 us, so even perfect execution on all of them yields 20–25% while 30 fps needs
53%.

### What was already tried and is dead (do not redo)

| approach | outcome |
|---|---|
| Per-descriptor caching with dependency revalidation | **measured net loss**, +1.3–2.8 us/draw. 141 resolutions at ~27 ns each; no revalidation beats 27 ns |
| Native `vkCmdDrawIndexedIndirectCount` passthrough | `DRAW_INDEX_INDIRECT_MULTI` is **never called** in this title. Removes nothing |
| Gating the ~125 per-draw register validation branches | **56 ns/draw**, 0.35%. Not worth defaulting off |
| Draw batching / merging consecutive draws | 100k draws → 100k distinct states, run length 1.00 |
| Thread pool over `MaterializeResources` | shared evaluators + memo reuse make it unsplittable |

The one thing that did work — halving `ClampRangeSize` — worked because it removed *waiting*, not
computing: `MergeUnlocked` was sorting and reallocating the whole guest range table on each of
~147k mutations while holding a lock the render thread blocks on.

---

## 3. Proposal: 7 stages → 4, by deriving on change instead of per draw

The current path has seven stages because each one re-derives its inputs from raw registers. If
state changes are tracked, five of them collapse into one conditional stage.

### Stage 1 — Ingest (PM4, no derivation)

Register-write packet handlers do what they already do, plus set a **dirty bit** for the group
they touched. One counter per group: shader regs, context regs (blend/depth/raster), render
targets, viewport/scissor, user data.

Cost: one `|=` per register packet. The hook is a small number of places
(`CpOpSetContextReg` and siblings in `pm4Dispatch.cpp:197-214`), not the ~100 individual setters.

### Stage 2 — Resolve-if-dirty (replaces RefreshShaders + pipeline + render targets)

```
if (dirty & SHADER_REGS)  -> resolve programs + input info   (else reuse last draw's)
if (dirty & RT_REGS)      -> acquire render targets          (else reuse)
if (dirty & PIPELINE_REGS)-> look up/compile pipeline        (else reuse the handle)
```

Everything here is keyed on state that changes far less often than once per draw. The static
half of `PrepareProgram` — the 2 KB zero, the AGC parse, the attribute-table walk — is a pure
function of `(shader address, chksum)` and can be cached outright.

**This is gated on one unknown**: how often consecutive draws share state. That is now
instrumented (`8395fc4`) and prints as `state reuse: NN.N% of draws reuse the previous draw's
shader pair`. **Read that number before building any of this.** If it is 70%+, stage 2 nearly
vanishes for most draws. If it is under 20%, this whole section is dead and the effort belongs in
section 4 instead.

### Stage 3 — Bind (per-draw, but only what changed)

Descriptors genuinely depend on per-object user data, so this stage stays per-draw. It shrinks
by construction rather than by caching:

- **Bindless / descriptor indexing.** Textures and buffers live in one large descriptor array;
  the draw passes indices in push constants instead of building and writing a descriptor set.
  This removes most of `PrepareBindings` (1.10), `RebindImages` (0.45) and `CommitBindings`, and
  removes the per-draw `vkUpdateDescriptorSets`/`pushDescriptorSet` entirely. It is what
  vkd3d-proton and DXVK moved to for exactly this reason, and `VK_EXT_descriptor_buffer` exists
  to make it cheaper still.
- The SRT walk still has to produce the descriptor *contents*, but 141 resolutions per draw at
  27 ns is 3.8 us that a compiled fetch list could cut: each program's SRT expression graph is
  fixed, so it can be compiled once into a flat array of "read dword at offset, shift, mask,
  store" operations and executed as a tight loop instead of interpreted as a graph per draw.

### Stage 4 — Emit

Push constants, bind what changed, draw. Already 0.05 us.

### What that leaves

| stage | today | after |
|---|---|---|
| 1 Ingest | (inside preamble) | ~0.3 |
| 2 Resolve-if-dirty | 3.77 + 0.56 + 0.69 = 5.02 | ~1.0–1.5 (projected, reuse-rate dependent) |
| 3 Bind | 4.67 + 0.65 = 5.32 | ~2.0–2.5 (projected, bindless + compiled SRT) |
| 4 Emit | 0.05 | 0.05 |
| other (vertex/index, dyn state, begin rendering) | 1.83 | ~1.5 |
| **total** | **14.8** | **~5–6 projected** |

That is roughly 14 fps → 30–35 fps. It does **not** reach PS5 parity, and no single-threaded
design will, because the console spends ~0 CPU per draw and we cannot spend less than the cost
of building a Vulkan draw.

---

## 4. The part that actually reaches 60: use the other 15 cores

Measured during gameplay: **CPU ~29%, GPU ~22%**, on a 16-thread 9700X. One thread does all the
translation while fifteen idle and the GPU waits. That is the headroom, and it is large.

The work is naturally three-stage — parse PM4, resolve state/resources, record Vulkan — and those
stages are independent across draws. Recording into **secondary command buffers on N threads**,
or pipelining parse/resolve/record, converts a 5–6 us serial draw into ~1–2 us of effective
latency. That is 60 fps territory at 4450 draws/frame.

The blocker is stated in the handoff and confirmed by this session's findings: `bufferCache` has
**no lock at all**, `textureCache` has a spin lock, and the guest range table has a contended
mutex that a single sort-and-reallocate was enough to make visible. Those three have to become
genuinely concurrent (or per-thread with a merge step) before any of it is safe.

**Order matters:** section 3 first. It reduces the work that has to be made concurrent, and it is
recoverable if wrong. Threading first would be building a parallel version of work that should not
exist.

---

## 5. Ranked plan

| # | change | expected | risk | prerequisite |
|---|---|---|---|---|
| 1 | Read the `state reuse` number | — | none | one run |
| 2 | Cache the static half of `PrepareProgram` on (addr, chksum) | ~1–1.5 us | low | #1 says reuse is high |
| 3 | Dirty bits + resolve-if-dirty for shaders/pipeline/RTs | ~1.5–2 us | medium | #1, #2 |
| 4 | Compile SRT graphs to flat fetch lists | ~1.5–2 us | medium | none |
| 5 | Bindless descriptors | ~2–2.5 us | medium | none |
| 6 | Profile the ~3 us outside `DrawIndex` | unknown | none | one run |
| 7 | Concurrent buffer/texture caches | prerequisite | **very high** | #2–#5 done |
| 8 | Multi-threaded recording | the rest of the way to 60 | **very high** | #7 |

### Rules carried from this session

1. **Compare phases within a run, never TOTAL between runs** — variance is ±1.2 us.
2. **Instrument before fixing.** Three confident mechanism stories in a row were wrong: the
   descriptor cache (net loss), `hw_check` (20x overestimate), and the clamp fast path (which hit
   96% and still did not help, because the cost was lock contention elsewhere).
3. **A cache must beat the thing it replaces.** At 27 ns per descriptor resolution, nothing does.
   Prefer deleting work and restructuring over caching.
4. **Before any cache, enumerate every input the cached function reads and prove the key contains
   all of them.** Four separate bugs of this exact shape are documented in the handoff.
