#include "graphics/host_gpu/renderer/drawWorkerPool.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

namespace Libs::Graphics {

namespace {

constexpr size_t kSecondaryGrowStep = 8;

} // namespace

// One command pool per worker, plus the secondary buffers recorded from it. A buffer is only
// reset once the tick it was submitted under has retired, so the GPU is never reading a buffer
// that is being re-recorded.
struct DrawWorkerPool::Worker {
	vk::CommandPool                m_pool = nullptr;
	std::vector<vk::CommandBuffer> m_buffers;
	std::vector<uint64_t>          m_ticks;
	size_t                         m_next_free = 0;
	vk::CommandBuffer              m_recording = nullptr;
};

DrawWorkerPool::DrawWorkerPool(GraphicContext& graphics, MasterSemaphore& master,
                               uint32_t worker_count)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(worker_count == 0);
	m_workers.reserve(worker_count);
	for (uint32_t i = 0; i < worker_count; i++) {
		auto worker = std::make_unique<Worker>();

		vk::CommandPoolCreateInfo create {};
		create.sType            = vk::StructureType::eCommandPoolCreateInfo;
		create.queueFamilyIndex = graphics.queue_family;
		// Transient because every buffer is re-recorded each frame; ResetCommandBuffer because
		// buffers are recycled individually as their ticks retire rather than by resetting the
		// whole pool, which would invalidate buffers the GPU has not finished with.
		create.flags = vk::CommandPoolCreateFlagBits::eTransient |
		               vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		const auto result = graphics.device.createCommandPool(&create, nullptr, &worker->m_pool);
		EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

		m_workers.push_back(std::move(worker));
	}

	// Worker 0 is the calling thread's slot: ParallelFor runs items on the caller too, so only
	// worker_count - 1 threads are spawned.
	m_threads.reserve(worker_count - 1);
	for (uint32_t i = 1; i < worker_count; i++) {
		m_threads.emplace_back([this, i](std::stop_token stop) { WorkerLoop(i, std::move(stop)); });
	}
}

DrawWorkerPool::~DrawWorkerPool() {
	for (auto& thread: m_threads) {
		thread.request_stop();
	}
	{
		std::lock_guard lock(m_mutex);
		m_generation++;
	}
	m_work_available.notify_all();
	m_threads.clear();   // jthread joins on destruction

	for (auto& worker: m_workers) {
		if (worker->m_pool != nullptr) {
			m_graphics.device.destroyCommandPool(worker->m_pool, nullptr);
		}
	}
}

void DrawWorkerPool::WorkerLoop(uint32_t index, std::stop_token stop) {
	KYTY_PROFILER_THREAD("DrawWorker");
	uint64_t seen = 0;
	for (;;) {
		const std::function<void(uint32_t, uint32_t)>* body = nullptr;
		{
			std::unique_lock lock(m_mutex);
			m_work_available.wait(lock, [this, &seen, &stop] {
				return stop.stop_requested() || m_generation != seen;
			});
			if (stop.stop_requested()) {
				return;
			}
			seen = m_generation;
			body = m_body;
		}
		if (body == nullptr) {
			continue;
		}
		for (;;) {
			const auto item = m_next.fetch_add(1, std::memory_order_relaxed);
			if (item >= m_count) {
				break;
			}
			(*body)(item, index);
		}
		if (m_outstanding.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			std::lock_guard lock(m_mutex);
			m_work_done.notify_all();
		}
	}
}

void DrawWorkerPool::ParallelFor(uint32_t                                       count,
                                 const std::function<void(uint32_t, uint32_t)>& body) {
	if (count == 0) {
		return;
	}
	// One item is not worth waking anything: the handoff costs more than the work.
	if (count == 1 || m_threads.empty()) {
		for (uint32_t i = 0; i < count; i++) {
			body(i, 0);
		}
		return;
	}

	{
		std::lock_guard lock(m_mutex);
		m_body  = &body;
		m_count = count;
		m_next.store(0, std::memory_order_relaxed);
		// Threads plus this one: the caller is a runner, and is counted so the last finisher is
		// unambiguous whoever it turns out to be.
		m_outstanding.store(static_cast<uint32_t>(m_threads.size()) + 1, std::memory_order_release);
		m_generation++;
	}
	m_work_available.notify_all();

	for (;;) {
		const auto item = m_next.fetch_add(1, std::memory_order_relaxed);
		if (item >= count) {
			break;
		}
		body(item, 0);
	}
	if (m_outstanding.fetch_sub(1, std::memory_order_acq_rel) != 1) {
		std::unique_lock lock(m_mutex);
		m_work_done.wait(lock,
		                 [this] { return m_outstanding.load(std::memory_order_acquire) == 0; });
	}

	std::lock_guard lock(m_mutex);
	m_body  = nullptr;
	m_count = 0;
}

vk::CommandBuffer DrawWorkerPool::BeginSecondary(uint32_t                         worker_index,
                                                 const SecondaryRenderingFormats& formats) {
	EXIT_IF(worker_index >= m_workers.size());
	auto& worker = *m_workers[worker_index];
	EXIT_IF(worker.m_recording != nullptr);

	if (worker.m_next_free >= worker.m_buffers.size()) {
		// Reclaim before growing. Without this the ring only ever grows: one secondary per draw at
		// ~4450 draws a frame exhausts host memory in seconds, which is exactly what it did.
		RecycleRetired();
	}
	if (worker.m_next_free >= worker.m_buffers.size()) {
		const auto first = worker.m_buffers.size();
		worker.m_buffers.resize(first + kSecondaryGrowStep);
		worker.m_ticks.resize(first + kSecondaryGrowStep, 0);

		vk::CommandBufferAllocateInfo allocate {};
		allocate.sType              = vk::StructureType::eCommandBufferAllocateInfo;
		allocate.commandPool        = worker.m_pool;
		allocate.level              = vk::CommandBufferLevel::eSecondary;
		allocate.commandBufferCount = static_cast<uint32_t>(kSecondaryGrowStep);
		const auto result =
		    m_graphics.device.allocateCommandBuffers(&allocate, worker.m_buffers.data() + first);
		EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	}

	const auto buffer = worker.m_buffers[worker.m_next_free];
	worker.m_ticks[worker.m_next_free] = m_master.CurrentTick();
	worker.m_next_free++;

	vk::CommandBufferInheritanceRenderingInfo rendering {};
	rendering.sType                = vk::StructureType::eCommandBufferInheritanceRenderingInfo;
	rendering.viewMask             = formats.view_mask;
	rendering.colorAttachmentCount = formats.color_count;
	rendering.pColorAttachmentFormats = formats.color_formats.data();
	rendering.depthAttachmentFormat   = formats.depth_format;
	rendering.stencilAttachmentFormat = formats.stencil_format;
	rendering.rasterizationSamples    = formats.samples;

	vk::CommandBufferInheritanceInfo inheritance {};
	inheritance.sType = vk::StructureType::eCommandBufferInheritanceInfo;
	inheritance.pNext = &rendering;
	// The primary may have an occlusion query running when this secondary is executed, so the
	// secondary has to declare that it inherits one. Paired with the inheritedQueries feature.
	inheritance.occlusionQueryEnable = VK_TRUE;
	inheritance.queryFlags           = vk::QueryControlFlagBits::ePrecise;

	vk::CommandBufferBeginInfo begin {};
	begin.sType = vk::StructureType::eCommandBufferBeginInfo;
	// RenderPassContinue is required even under dynamic rendering: it is what tells the driver the
	// inheritance chain above applies. OneTimeSubmit because these are re-recorded every frame.
	begin.flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue |
	              vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	begin.pInheritanceInfo = &inheritance;
	EXIT_NOT_IMPLEMENTED(buffer.begin(&begin) != vk::Result::eSuccess);

	worker.m_recording = buffer;
	return buffer;
}

void DrawWorkerPool::EndSecondary(uint32_t worker_index) {
	EXIT_IF(worker_index >= m_workers.size());
	auto& worker = *m_workers[worker_index];
	EXIT_IF(worker.m_recording == nullptr);
	worker.m_recording.end();
	worker.m_recording = nullptr;
}

void DrawWorkerPool::RecycleRetired() {
	for (auto& worker: m_workers) {
		if (worker->m_recording != nullptr) {
			continue;   // mid-record: its buffers are by definition still in use
		}
		// Conservative on purpose: the ring is only rewound when every buffer handed out has
		// retired. Recycling a prefix would need the ring to be a real ring, and reusing a buffer
		// the GPU is still reading is undefined behaviour rather than a slow frame.
		bool all_retired = true;
		for (size_t i = 0; i < worker->m_next_free; i++) {
			if (!m_master.IsFree(worker->m_ticks[i])) {
				all_retired = false;
				break;
			}
		}
		if (all_retired) {
			worker->m_next_free = 0;
		}
	}
}

} // namespace Libs::Graphics
