# Parallel secondary command buffer recording — design scope

Target: `phase3_record`, stable at **79.5%** of drain time across every run. Phases 1 and 2 are
capped at a combined 20% ceiling, and two separate attempts to move them have now measured zero
(descriptor caching, texture-cache shared lock). Recording is the only remaining place the time is.

Nothing here is written yet. This is the design and the risk list.

---

## 0. The blocker to fix first: `g_command_generation` is global

This is the finding that matters most, and it is a **pre-existing latent bug**, not something
partitioning introduces.

`DynamicStateCache` (`renderDraw.cpp:392`) filters redundant state. It is already `thread_local`,
and it already invalidates on a generation change, which is exactly the right shape:

```cpp
bool Retarget() {
    const auto current = CurrentCommandGeneration();
    if (current == generation && Config::DynStateCacheEnabled()) return false;
    *this = {};                       // everything must be re-issued
    generation = current;
    return true;
}
```

But the generation it keys on is a **single global atomic bumped only by
`CommandBuffer::Begin()`** — the primary (`context.cpp:37`). `DrawWorkerPool::BeginSecondary`
never touches it.

Consequence: a thread that records secondary A, then secondary B, with no primary `Begin()` in
between, sees an unchanged generation. `Retarget()` returns false. **The pipeline bind, index
buffer bind and all 11 dynamic-state calls are skipped in secondary B** — which inherits none of
them, because Vulkan dynamic state is per command buffer. The draw executes against undefined
state.

This is the visual-corruption / driver-crash mode, and it is reachable today: phase 3 runs on the
main thread, so the main thread's cache persists across every secondary it opens within one
submission.

**Fix** — make the generation follow the recording target, not the primary:

```cpp
inline thread_local uint64_t t_recording_generation = 0;   // 0 = not recording a secondary

uint64_t CurrentCommandGeneration() {
    return t_recording_generation != 0 ? t_recording_generation : g_command_generation;
}
```

`BeginSecondary` assigns from a global `fetch_add`; `EndSecondary` restores 0. Per-thread, so one
worker's begin cannot spuriously invalidate another's cache — which rules out the naive fix of
bumping the shared global in `BeginSecondary`, since with 8 workers that would miss on every draw
and throw away the 0.5–1 µs/draw the cache exists to save.

**Verify before building on it.** I have not proven this misfires today, only that the mechanism is
unsound. Cheap check: count `BeginSecondary` calls against `Retarget()` fires. If retargets <
secondaries, state is leaking across secondary boundaries right now.

---

## 1. State invalidation across secondaries

Cheaper than expected, because the two categories of state are already handled differently and both
already work.

**Descriptors and push constants need no carry mechanism at all.** `CommitBindings`
(`descriptors.cpp:1419`) emits `pushConstants` and `pushDescriptorSetKHR` / `bindDescriptorSets`
**unconditionally on every draw** — there is no cache, no dirty bit, no skip path. Every draw is
already self-sufficient for descriptor state. Chunk boundaries are invisible to it.

**Pipeline, index buffer and dynamic state are filtered**, and those are exactly what §0's
generation fix invalidates. At draw 0 of each chunk `Retarget()` fires and re-emits:

| State | Call | Re-emit cost |
|---|---|---|
| Pipeline | `bindPipeline` | 1 |
| Viewport / scissor | `setViewport`, `setScissor` | 2 |
| Line width, depth bias | `setLineWidth`, `setDepthBiasEnable`, `setDepthBias` | 3 |
| Stencil | `setStencilCompareMask/WriteMask/Reference` × front+back | 6 |
| Index buffer | `bindIndexBuffer` | 1 |
| Vertex buffers | `bindVertexBuffers` | already per-draw |

**~13 redundant driver calls per chunk**, once, at chunk head. At the ~50–90 ns/call implied by the
existing "0.5–1 µs per 11 calls" measurement, that is **≈0.7–1.2 µs per chunk** — under one draw's
worth of phase 3 time (~11.4 µs/draw).

So there is **no snapshot to build**. The existing cache plus a correct generation source is the
whole state-carry mechanism. That is the good news in this design.

---

## 2. Batch partitioning

From the last clean run (`drains=120000`, `draws/batch=67.7`):

```
sizes  1:11742   2-4:18055   5-16:20283   17-64:23662   65-255:36091   256:10167
```

Weighting by bucket midpoint (≈9.23M draws total):

- batches **≥65 draws carry ~87% of all draws**
- batches **≥17 draws carry ~97%**
- batches of 1–4 draws are 25% of *batches* but **under 1% of draws**

So the long tail of tiny batches is irrelevant to throughput and should not be chunked at all —
same conclusion `ParallelFor`'s `kInlineThreshold = 16` already reached empirically.

**Split contiguously, never round-robin.** Contiguous chunks make guest order fall out of the
`executeCommands` array order for free (§3). Round-robin would require interleaving at replay,
which Vulkan cannot express.

**Proposed rule:**

```
chunks = clamp(batch_size / 16, 1, worker_count)
```

- < 32 draws → 1 chunk, existing single-secondary path unchanged
- 64 → 4 chunks × 16
- 256 → 8 chunks × 32

**Minimum 16 draws per chunk**, derived from §1: ~1 µs of chunk overhead against ~11.4 µs/draw
means 16 draws holds the overhead near 0.5%. Going to 8 would still be ~1%, so 16 is conservative
rather than tight — worth revisiting once measured, and the census should report the achieved
chunk-size distribution so it can be.

---

## 3. Replay on the primary

The cheap part. Vulkan executes secondaries **in array order**, so ordering costs nothing:

```cpp
scheduler.BeginRendering(ResumeStateFor(batch.rendering), /*secondary contents=*/true);
primary.executeCommands(chunk_count, batch.buffers.data());   // one call, chunk order
scheduler.EndRendering();
```

`OpenSecondaryBatch` becomes an ordered array instead of `{buffer, worker}`:

```cpp
struct OpenSecondaryBatch {
    std::array<vk::CommandBuffer, kMaxDrawWorkers> buffers {};
    std::array<uint32_t,          kMaxDrawWorkers> workers {};
    uint32_t chunk_count = 0;
    ...
};
```

Everything else in `FlushSecondaryBatch` (`renderDraw.cpp:2500`) already holds:

- staged work — clears, deferred touches, deferred tracks, transitions, uploads — is drained on the
  main thread **before** `BeginRendering`, so every barrier is emitted outside a render pass, which
  is where Vulkan requires them. Unchanged by chunking.
- `SecondaryRenderingFormats` is already a batch-level invariant ("every draw in one batch must
  agree on them"). Chunk boundaries cannot change it, so all chunks share one inheritance struct.
- occlusion queries already work: `BeginSecondary` sets `occlusionQueryEnable = VK_TRUE` with empty
  `queryFlags`, matching how the primary begins the query. Needs the `inheritedQueries` device
  feature, which is already the reason the earlier
  `VUID-vkCmdExecuteCommands-commandBuffer-00101` was fixed.

Replay overhead is therefore **one `executeCommands` regardless of chunk count**.

---

## 4. What is not yet proven — audit list before writing code

The infrastructure is further along than the fps suggests: per-worker command pools, per-worker
descriptor pools, per-worker stream rings, staged transitions, deferred touches/tracks, and a
worker-safe `FindTexture` (0 bailouts, 0 lock diagnostics, 0 VUIDs over 120k drains) all exist and
work. What is unproven is whether **`DrawIndex`'s full path** is worker-safe, since until now only
the *resolve* ran on workers, never the *record*.

Must be audited before writing:

1. **Occlusion query begin/end placement.** If a draw can begin or end a query mid-batch, that
   cannot be expressed inside a secondary. Highest-risk unknown; check first.
2. **`CommitBindings`' non-push path** — `updateDescriptorSets` plus a descriptor-set allocation.
   Per-worker descriptor pools exist; confirm the allocation actually routes through
   `CurrentDrawWorker()` and not a shared pool.
3. **`SetGraphicsDynamicParams`** — verify it only records and derives, and mutates no shared state.
4. **`m_push_constants`** — a member buffer on `RenderExecutor`, written per draw. If it is one
   shared array, 8 workers writing it concurrently is a straight race. Likely needs to be per-worker.
5. **`DynState()`** is `thread_local` already ✓ — but confirm no other `static` (non-thread_local)
   caches sit in the record path. `g_last_flushed_state` and the frame-accounting statics in
   `DrawIndex` are main-thread-only today and would become concurrent.
6. **`BeginRendering` / `EndRendering` calls inside `DrawIndex`** must not fire on the worker path;
   the `secondary` flag already guards this, but re-check every early return.

Items 4 and 5 are the ones I would expect to bite. They are cheap to check and expensive to debug
after the fact.

---

## 5. Expected return, stated honestly

Phase 3 is 79.5% of drain and fully serial. Perfect 8-way parallelism would take it to ~10%,
putting drain at roughly 30% of today's — but nothing here will be perfect, and this session has
already produced two parallelization efforts that measured **zero** because the bottleneck was
memory, not concurrency.

The credible claim is: **recording is driver-call-bound, not memory-bound**, which is the reason to
expect it to scale where the SRT walk did not. That is a hypothesis, not a result. The cheapest way
to test it before committing to the full build is to chunk a batch **two** ways and see whether
phase 3 wall time drops at all. If a 2-way split does not move it, an 8-way split will not either,
and this whole design should be abandoned rather than finished.

**Recommended order:**

1. Fix §0 (generation), add the retarget-vs-secondary counter, verify. Small, standalone, fixes a
   real latent bug regardless of what follows.
2. Audit §4 items 1, 4, 5.
3. 2-way chunk split behind a flag. Measure. **Decision point.**
4. Only then generalize to `clamp(size/16, 1, workers)`.
