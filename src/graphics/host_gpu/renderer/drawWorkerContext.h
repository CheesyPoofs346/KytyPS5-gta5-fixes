#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWWORKERCONTEXT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWWORKERCONTEXT_H_

#include <cstdint>

namespace Libs::Graphics {

// Which worker slot the calling thread is acting as. Index 0 is the ingest/main thread, which is
// also a runner in DrawWorkerPool::ParallelFor.
//
// Resources that Vulkan requires to be externally synchronised - descriptor pools, command pools,
// stream-buffer rings - are held one per slot and selected through this. Threading the index
// through every call site instead would touch the whole draw path for no behavioural gain, and
// the draw path is already ~40 functions deep.
// Capped rather than "one per hardware thread": every slot costs a descriptor pool, a command
// pool and a stream ring, and the stream rings are host-visible memory. Eight bounds that at
// 112 MiB of extra rings while still giving 8-way parallelism across a ~12 us draw.
inline constexpr uint32_t kMaxDrawWorkers = 8;

inline thread_local uint32_t t_draw_worker_index = 0;

[[nodiscard]] inline uint32_t CurrentDrawWorker() noexcept {
	return t_draw_worker_index;
}

// Work that records into the primary command buffer must be staged when it originates on a
// worker, and need not be when it originates on the main thread.
//
// This replaces trying to enumerate the consumers of a staged upload. That failed twice: a
// consumer is anyone who obtains a resource and then records commands touching it, which is
// unbounded - the test that caught it does exactly that, obtaining a buffer and immediately
// recording a copyBuffer from it. A main-thread caller sees the recording happen inline, as it
// always has, so no consumer can be surprised. Only workers stage, and the batch flush drains
// their staged work on the main thread before any of it is replayed.
[[nodiscard]] inline bool MustStageForWorker() noexcept {
	return t_draw_worker_index != 0;
}

// Raised when a worker reaches work only the main thread may do, so the draw it was resolving can
// be abandoned and re-resolved serially in phase 3.
//
// This is a bail-out rather than a deferral, and the difference is deliberate. Deferring suits
// work that is a pure state change replayable at the flush - an LRU touch, a page-watcher re-arm.
// It does not suit work that uploads guest memory and records transfer commands mid-resolve:
// replaying that away from its call site means reconstructing the descriptor it was feeding, which
// is far more machinery than the case deserves. The census settles which it deserves - 59 hashes
// in 5.2M refreshes across a city drive - so the draws that hit this are rare enough that losing
// their parallelism costs nothing measurable, and phase 3's inline resolve is a path that already
// exists and is already exercised by every uncompiled permutation.
inline thread_local bool t_draw_worker_bailed = false;

inline void RequestWorkerBailout() noexcept {
	t_draw_worker_bailed = true;
}

// Reads and clears, so the caller must treat the result as the answer for exactly one item. A flag
// left set would silently invalidate the next draw the slot resolves.
[[nodiscard]] inline bool TakeWorkerBailout() noexcept {
	const bool bailed    = t_draw_worker_bailed;
	t_draw_worker_bailed = false;
	return bailed;
}

// Scoped so a worker slot cannot leak past the batch that set it - a stale index would silently
// point a later main-thread draw at a worker's pool.
class ScopedDrawWorker {
public:
	explicit ScopedDrawWorker(uint32_t index) noexcept
	    : m_previous(t_draw_worker_index) {
		t_draw_worker_index = index;
	}

	~ScopedDrawWorker() noexcept { t_draw_worker_index = m_previous; }

	ScopedDrawWorker(const ScopedDrawWorker&)            = delete;
	ScopedDrawWorker& operator=(const ScopedDrawWorker&) = delete;

private:
	uint32_t m_previous;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWWORKERCONTEXT_H_ */
