# Guest backing memory: lifetime audit and contract

Source-backed audit of `src/kernel/memoryAddressSpace.inc` (`GuestBackingStore` and the
address-space wrapper). Purpose: establish what may safely happen outside the store's mutex,
so the stream-upload path can be improved without guessing.

## 1. What the mutex actually protects

`std::mutex m_mutex` (one per store, one store per process) is taken at 23 sites. Grouped by
what they mutate:

| group | functions | mutates |
|---|---|---|
| mapping table | `MapExistingPlaceholderFixed`, `Unmap`, `UnmapOne`, `MapBacking`, `UnmapBacking` | `m_maps` entries |
| reservation | `ReserveAligned`, `ReserveFixed`, `FindFreeAligned`, `ReleaseFree`, `TestContainsFree` | free-list / placeholders |
| commitment | `Commit`, `ReleaseCommitted` | committed guest ranges |
| protection | `Protect`, `ProtectTransient` | guest page protection |
| readers | `TryTransferBacking`, `TryReadSparseBacking`, `Contains`, `Owns`, `OverlapsOwned` | nothing |

So the mutex protects **the mapping table and the guest address space**. It does not protect
the backing storage, because the backing storage is not mutated by any of these.

## 2. The backing alias is permanent

This is the load-bearing fact.

`m_backing_base` is assigned exactly once, in the constructor, as a whole-store writable alias:

```cpp
m_backing_base = static_cast<uint8_t*>(
    MapViewOfFile(m_handle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, m_size));   // Windows
m_backing_base = static_cast<uint8_t*>(mmap(nullptr, m_size, PROT_READ | PROT_WRITE,
                                            MAP_SHARED, m_fd, 0));            // Linux
```

It is released **only in the destructor** (`UnmapViewOfFile` / `munmap`). Grep confirms no
other assignment or release: lines 243, 285 (create), 295, 308 (destroy).

Consequences, each verified:

- `UnmapOne` calls `UnmapViewPreservePlaceholder(old.vaddr, old.size)` and erases from
  `m_maps`. It unmaps the **guest-address view**. It never touches `m_backing_base`.
- `Protect` / `ProtectTransient` call `ProtectMappedUnlocked(vaddr, ...)`, operating on
  **guest addresses**. Guest protection changes cannot make the alias inaccessible.
- `m_size` never changes after construction.

**Therefore `m_backing_base + offset`, for any `offset < m_size`, is a valid mapped readable
host address for the entire lifetime of the store, independent of any guest mapping.**

## 3. Correcting an earlier claim

I previously said moving the copy outside the lock risks a **use-after-free**, because a
copied `MapEntry` "retains nothing". The first half was wrong.

A `MapEntry` does indeed retain nothing — but it does not need to. The bytes it points at live
in a permanent alias, not in storage that unmapping frees. **An unlocked read of
`m_backing_base + offset` cannot fault.**

The real hazard is narrower and semantic:

> Between the lookup and the copy, the guest may unmap the range and the same
> `backing_offset` may be re-committed to a different allocation. The read then returns the
> **current bytes of a different allocation** — not freed memory.

Failure mode: one draw reads wrong vertex or constant data. A visual glitch for one frame,
not a crash. That is a materially different risk class from what I claimed, and it changes
which designs are viable.

## 4. Design 1 — unlocked reads

**Requirement:** detect that the mapping relevant to this read did not change while the lock
was released. Storage validity is already guaranteed by §2, so only *identity* must be checked.

**Existing mechanism to prefer:** every mutation of `m_maps` already happens under `m_mutex`
at the 5 sites in group "mapping table", plus `Commit`/`ReleaseCommitted`. There is no
existing generation counter, but adding one to those sites is strictly smaller than any RCU
scheme and needs no new ownership model.

**Exact changes:**

1. Add `uint64_t m_map_generation = 0;` to `GuestBackingStore`.
2. Increment it in each function that mutates `m_maps` or committed ranges — the
   mapping-table and commitment groups in §1. All already hold the lock, so a plain
   non-atomic increment suffices for writers; readers load it via `std::atomic_ref` (or make
   the member `std::atomic<uint64_t>` and use relaxed loads).
3. In `TryTransferBacking`, for the single-contained-range case only (already 100% of calls,
   measured):
   - take the lock, look up the entry, read the generation, **release the lock**
   - perform the copy
   - re-read the generation; if it changed, discard and retry the whole operation under the
     lock using the existing two-pass path
4. Leave the spanning path entirely under the lock, unchanged.

Retry is safe because the destination is a stream allocation that has not been committed yet
(`stream.Commit()` runs after `TryReadBacking` returns), so a discarded copy has no observable
effect.

**Why it may help even though the fast-path A/B did not:** this removes the ~1417-byte memcpy
from inside the critical section. Lock **hold** time falls to a map lookup. Shorter holds
reduce contention for every other thread, which is a different lever from removing a lookup.
Benefit remains unmeasured until an A/B.

**Correctness tests required:**

- concurrent `UnmapOne` racing a read of the same range: result is either correct bytes or a
  retry, never a fault
- `backing_offset` recycled to a different allocation between lookup and copy: generation
  changed, retry taken
- generation increments verified at every mutation site (a test that mutates via each public
  entry point and asserts the counter moved)
- spanning and unmapped ranges still take the locked path, with no-partial-copy preserved
- the existing 5 focused fast-path tests must continue to pass

## 5. Design 2 — cached GPU snapshots

**Requirement:** these own their copied bytes, so per §2 and §3 they need **no** relationship
to the guest backing after the protected copy completes. What they need instead is
(a) correct content/version selection and (b) GPU lifetime.

**Existing ownership mechanisms to prefer — both already in the codebase:**

- `m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); })`
  in `bufferCache.cpp` already retains an allocation until the GPU is done with it. This is
  the mechanism for (b); no new lifetime machinery is needed.
- `StreamBuffer`'s watch/fence list (`WaitPendingOperations`, `m_current_watches`) already
  ties ring regions to scheduler ticks.

**The unsolved half is (a), version selection.** The CPU dirty bit cannot serve: it is page
granular (16 KB) while buffers average 1417 bytes, and `ForEachUploadRange` — the only path
that clears it — is not called on the stream path. A cache keyed on address needs a version
that increments on guest write, which does not currently exist at sub-page granularity.
Content hashing is the alternative and must beat a memcpy it duplicates the reads of.

**Verdict:** blocked on a versioning signal that does not exist. Not proposed now.

## 6. Recommended bounded experiment

**Design 1**, scoped to the single-contained-range fast path, behind `--backing-unlocked-read`
(default off), A/B'd with profiling off against the existing baselines.

Chosen over Design 2 because its correctness argument rests entirely on facts established
here (permanent alias, generation check, safe retry) and it reuses the existing lock rather
than introducing an ownership model. Design 2 needs a new versioning signal first.

## 7. Qualification carried forward

The batching census gives **~1,750 host calls/frame as a conditional floor**: conditional on
the current pipeline identities and on ordinary MDI, where draws with different pipelines
cannot share a call. It is not a floor for every possible backend design — vertex pulling
could reduce the pipeline set itself, since vertex layout specializes `vs_shader_id` via
`BuildStageStaticKey`.

**Vertex pulling and descriptor indexing remain deferred, not disproven.** The census measured
batching opportunity only. Descriptor indexing could reduce binding and preparation cost
without reducing draw count, and that benefit was never measured here.
