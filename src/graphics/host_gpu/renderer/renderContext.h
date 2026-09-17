#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_

#include "graphics/host_gpu/renderer/drawWorkerContext.h"
#include "graphics/host_gpu/renderer/drawWorkerPool.h"
#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/gpuResourceManager.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/gpuTimestamps.h"
#include "graphics/host_gpu/renderer/hdrProbe.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "kernel/eventQueue.h"

#include <memory>
#include <vector>

namespace Libs::VideoOut {
class VideoOutDriver;
}

namespace Libs::Graphics {

class GuestGpu;

class RenderContext {
public:
	explicit RenderContext(GraphicContext& graphics);
	~RenderContext();
	KYTY_CLASS_NO_COPY(RenderContext);

	[[nodiscard]] GraphicContext&           GetGraphics() const noexcept { return m_graphics; }
	void                                    InitializeGpu(VideoOut::VideoOutDriver* video_out);
	void                                    ShutdownGpu();
	[[nodiscard]] GuestGpu&                 GetGpu() const;
	// True once InitializeGpu has run. Unit tests drive the renderer without a GuestGpu, so any
	// frame-scoped logic must check this before asking for a frame number.
	[[nodiscard]] bool                      HasGpu() const noexcept { return m_gpu != nullptr; }
	[[nodiscard]] VideoOut::VideoOutDriver& GetVideoOut() const;

	Common::Mutex&      GetMutex() { return m_mutex; }
	CommandScheduler&   GetCommandScheduler() { return m_command_scheduler; }
	PipelineCache&      GetPipelineCache() { return m_pipeline_cache; }
	// Descriptor pools are externally synchronised, so each worker slot gets its own. Slot 0 is
	// the main thread and keeps the original heap, so single-threaded behaviour is unchanged.
	DescriptorHeap&     GetDescriptorHeap() {
		const auto worker = CurrentDrawWorker();
		if (worker == 0 || worker > m_worker_descriptor_heaps.size()) {
			return m_descriptor_heap;
		}
		return *m_worker_descriptor_heaps[worker - 1];
	}

	// Called once the worker count is known; safe to call before any worker exists.
	void CreateWorkerDescriptorHeaps(uint32_t worker_count);

	// Created on first use, sized from --draw-workers.
	//
	// Takes no count on purpose. It used to, and the first caller won: FlushSecondaryBatch asked
	// for 1 and always ran before the first queue drain, so the pool was built with a single
	// worker and every later request for eight was ignored. ParallelFor then had two runners -
	// the caller and one worker - which is exactly the 2x the walk measured against an expected
	// 8x.
	DrawWorkerPool& GetDrawWorkerPool();
	SamplerCache&       GetSamplerCache() { return m_sampler_cache; }
	GpuResourceManager& GetGpuResources() { return m_gpu_resources; }
	BufferCache&        GetBufferCache() { return m_gpu_resources.GetBufferCache(); }
	TextureCache&       GetTextureCache() { return m_gpu_resources.GetTextureCache(); }
	RenderExecutor&     GetRenderExecutor() { return m_render_executor; }
	HdrProbe&           GetHdrProbe() { return m_hdr_probe; }
	GpuTimestamps&      GetGpuTimestamps() { return m_gpu_timestamps; }

	void AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void TriggerInterrupt(int event_id, uint32_t context_id);

private:
	struct InterruptEqRegistration {
		LibKernel::EventQueue::KernelEqueue eq       = LibKernel::EventQueue::KERNEL_EQUEUE_INVALID;
		int                                 event_id = 0;
	};

	GraphicContext&           m_graphics;
	Common::Mutex             m_mutex;
	RenderExecutor            m_render_executor;
	CommandScheduler          m_command_scheduler;
	DescriptorHeap            m_descriptor_heap;
	std::vector<std::unique_ptr<DescriptorHeap>> m_worker_descriptor_heaps;
	std::unique_ptr<DrawWorkerPool>             m_draw_worker_pool;
	PipelineCache             m_pipeline_cache;
	SamplerCache              m_sampler_cache;
	GpuResourceManager        m_gpu_resources;
	HdrProbe                  m_hdr_probe;
	GpuTimestamps             m_gpu_timestamps;
	std::unique_ptr<GuestGpu> m_gpu;
	VideoOut::VideoOutDriver* m_video_out = nullptr;

	Common::Mutex                        m_interrupt_mutex;
	std::vector<InterruptEqRegistration> m_interrupt_eqs;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
