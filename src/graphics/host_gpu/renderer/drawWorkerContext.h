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
