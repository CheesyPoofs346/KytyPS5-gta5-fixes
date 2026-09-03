#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWWORKERPOOL_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWWORKERPOOL_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/renderTarget.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class MasterSemaphore;

// Secondary command buffers inherit a render pass. This renderer has no VkRenderPass objects - it
// uses vkCmdBeginRendering - so the inheritance chain must carry the attachment formats instead,
// via VkCommandBufferInheritanceRenderingInfo. These are the values a batch records against, and
// every draw in one batch must agree on them.
struct SecondaryRenderingFormats {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	uint32_t                                             color_count    = 0;
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	vk::SampleCountFlagBits                              samples = vk::SampleCountFlagBits::e1;
	uint32_t                                             view_mask = 0;

	bool operator==(const SecondaryRenderingFormats&) const = default;
};

// A pool of worker threads, each owning its own VkCommandPool and secondary command buffers.
//
// Vulkan command pools are externally synchronised: two threads may not record into buffers from
// the same pool concurrently. One pool per worker is what makes parallel recording legal at all,
// and it is why this cannot be bolted onto the existing single CommandScheduler pool.
//
// Recording order is preserved by the caller: batches are handed to vkCmdExecuteCommands in guest
// order, so blending and depth behave exactly as they do serially. Workers may finish out of
// order; the merge does not.
class DrawWorkerPool {
public:
	DrawWorkerPool(GraphicContext& graphics, MasterSemaphore& master, uint32_t worker_count);
	~DrawWorkerPool();
	KYTY_CLASS_NO_COPY(DrawWorkerPool);

	[[nodiscard]] uint32_t WorkerCount() const noexcept {
		return static_cast<uint32_t>(m_workers.size());
	}

	// Runs body(item, worker_index) for item in [0, count), across the pool, and returns once every
	// item has completed. The calling thread participates, so a pool of N gives N+1 runners and a
	// count of 1 never touches a worker at all.
	void ParallelFor(uint32_t count, const std::function<void(uint32_t, uint32_t)>& body);

	// Acquires a secondary command buffer for this worker, already begun with the inheritance the
	// formats describe, and flagged as continuing the caller's render pass.
	vk::CommandBuffer BeginSecondary(uint32_t worker, const SecondaryRenderingFormats& formats);
	void              EndSecondary(uint32_t worker);

	// Secondary buffers stay alive until the GPU has consumed them. Called once the tick they were
	// submitted under has retired.
	void RecycleRetired();

private:
	struct Worker;

	void WorkerLoop(uint32_t index, std::stop_token stop);

	GraphicContext&                      m_graphics;
	MasterSemaphore&                     m_master;
	std::vector<std::unique_ptr<Worker>> m_workers;

	// ParallelFor state. One outstanding batch at a time: the draw stream is ordered, so there is
	// no case where two independent batches are in flight.
	std::mutex                                     m_mutex;
	std::condition_variable                        m_work_available;
	std::condition_variable                        m_work_done;
	const std::function<void(uint32_t, uint32_t)>* m_body  = nullptr;
	uint32_t                                       m_count = 0;
	std::atomic<uint32_t>                          m_next {0};
	std::atomic<uint32_t>                          m_outstanding {0};
	uint64_t                                       m_generation = 0;
	std::vector<std::jthread>                      m_threads;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWWORKERPOOL_H_ */
