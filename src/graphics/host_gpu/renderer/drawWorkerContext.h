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
inline thread_local uint32_t t_draw_worker_index = 0;

[[nodiscard]] inline uint32_t CurrentDrawWorker() noexcept {
	return t_draw_worker_index;
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
