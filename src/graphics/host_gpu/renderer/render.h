#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "graphics/host_gpu/renderer/drawBatchQueue.h"
#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct RenderDepthInfo;
struct RenderColorInfo;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DrawRenderState;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	EopWrite,
	EopInterrupt,
	EopWriteBack,
	EopFlip,
	EopWriteBackFlip,
	EopOnlyFlip,
	Unknown,
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

// Identifies the recording a per-command-buffer state cache (dynamic state, bind filtering) is
// currently feeding. Pool-recycled handles make handle comparison unsafe - a fresh recording can
// reuse the previous handle - so compare this instead.
//
// It answers "which command buffer am I recording into", not "how many have begun". A secondary
// is a separate recording that inherits no dynamic state, no pipeline and no index binding, so it
// needs its own value or a cache built for the primary will skip re-emitting all of it and the
// draws execute against state the buffer was never given.
uint64_t CurrentCommandGeneration();

// Claims a fresh generation for a secondary recording on the calling thread, and returns the
// previous value for ScopedRecordingGeneration to restore.
//
// Per thread on purpose. Bumping a shared counter would work for one worker and fail for eight:
// every worker's begin would invalidate every other worker's cache, so every draw would miss and
// the 0.5-1 us/draw the cache exists to save would be lost precisely when parallel recording is
// meant to be paying off.
uint64_t BeginRecordingGeneration();
void     RestoreRecordingGeneration(uint64_t previous);

// Counts secondaries begun against dynamic-state cache invalidations, so "state leaked across a
// secondary boundary" is observed rather than assumed. Fires < begins means leakage.
void NoteStateRetarget();
void ReportRecordingCensus();

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const;

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	// secondary_contents: draws will arrive via vkCmdExecuteCommands rather than inline. Vulkan
	// forbids mixing the two in one render pass instance, so this is decided per BeginRendering.
	void BeginRendering(const RenderState& state, bool secondary_contents = false) const;
	void EndRendering() const;

	[[nodiscard]] vk::CommandBuffer Handle() const;
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&      GetRegisters() const noexcept {
		const auto& view = ThreadView();
		return view.context != nullptr ? *view.context : *m_registers;
	}
	[[nodiscard]] HW::UserConfig& GetUserConfig() const noexcept {
		const auto& view = ThreadView();
		return view.user_config != nullptr ? *view.user_config : *m_user_config;
	}
	[[nodiscard]] HW::Shader& GetShaders() const noexcept {
		const auto& view = ThreadView();
		return view.shaders != nullptr ? *view.shaders : *m_shaders;
	}

	// What the draw path reads its register state through. Swappable so a deferred draw can be
	// translated against the snapshot taken when it was queued rather than against registers the
	// packet loop has since overwritten.
	struct RegisterView {
		HW::Context*    context     = nullptr;
		HW::UserConfig* user_config = nullptr;
		HW::Shader*     shaders     = nullptr;
	};

	// Returns the view that was installed, for the caller to restore.
	//
	// The installed view is per thread. It describes the translation in flight on the calling
	// thread, not a property of the command buffer, and parallel chunk recording has two threads
	// translating different draws through the same CommandBuffer - a shared view would let one
	// chunk's swap retarget the other chunk's draw mid-translation, silently translating it
	// against the wrong registers.
	//
	// An unset view falls back to the bound one, so any path that never swaps - every non-batched
	// draw, dispatch and blit - reads exactly what it read before.
	RegisterView SwapRegisterView(const RegisterView& view) const noexcept {
		const RegisterView previous = ThreadView();
		ThreadView()                = view;
		return previous;
	}

private:
	// One per thread, shared across CommandBuffer instances. That is correct rather than a
	// compromise: at most one translation is in flight per thread, and it is the translation the
	// view belongs to. Nested translation of two different buffers on one thread does not happen.
	[[nodiscard]] static RegisterView& ThreadView() noexcept {
		static thread_local RegisterView view;
		return view;
	}

	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	vk::CommandBuffer   m_buffer          = nullptr;
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
};

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DrawIndex(uint64_t submit_id, CommandBuffer& buffer, uint32_t index_type_and_size,
	               uint32_t index_count, const void* index_addr, uint32_t flags, uint32_t type,
	               uint32_t instance_count = 1, uint32_t render_target_slice_offset = 0,
	               int32_t vertex_offset_add = 0, uint32_t first_instance = 0,
	               const PreparedShaders* prepared = nullptr);
	void DrawAuto(uint64_t submit_id, CommandBuffer& buffer, uint32_t index_count, uint32_t flags,
	              uint32_t render_target_slice_offset = 0, uint32_t instance_count = 1,
	              uint32_t first_vertex = 0, uint32_t first_instance = 0);
	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode);

	[[nodiscard]] PreparedBindings PrepareBindings(const ShaderStageRuntime& runtime);
	void                           FindBuffers(PreparedBindings& bindings);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	// record_target null means record binds into buffer itself; a non-null target sends them to a
	// secondary while barriers still go to the primary.
	// Queues a draw for later translation against the snapshot supplied, and reports whether the
	// queue reached capacity and should be drained.
	// Locates both shader permutations for a queued draw and fills the static half of its input
	// info. Serial: the permutation lookup takes the program-cache mutex.
	bool PrepareQueuedShaders(CommandBuffer& buffer, PreparedShaders& prepared);

	// Runs the resource walk for an already-located pair. Safe on a worker: it reads guest memory
	// and writes only into prepared, touching no command buffer, image or cache.
	void ResolveQueuedShaders(PreparedShaders& prepared);

	// A layout transition a draw needs, recorded for later rather than emitted inline.
	//
	// Transit writes a barrier into the primary command buffer, which is the single thing that
	// stops resource resolution running on a worker. Staging them changes no ordering: a batch's
	// draws already replay after every transition their resolve recorded, so applying the list as
	// an explicit pre-pass is what the code does today, made explicit.
	struct PendingTransition {
		ImageId                             image_id;
		vk::ImageLayout                     layout = vk::ImageLayout::eUndefined;
		vk::AccessFlags2                    access {};
		std::optional<ImageSubresourceRange> range;
	};

	// Applies every staged transition, in the order requested. Main thread only.
	void FlushPendingTransitions(CommandBuffer& buffer);
	[[nodiscard]] bool HasPendingTransitions() const noexcept {
		return !m_pending_transitions.empty();
	}

	bool EnqueueDrawIndex(QueuedDraw&& draw);
	void DrainDrawQueue(CommandBuffer& buffer);
	[[nodiscard]] bool DrawQueueEmpty() const noexcept { return m_draw_queue.Empty(); }

	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings,
	                    vk::CommandBuffer                  record_target = nullptr);

private:
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);
	[[nodiscard]] GraphicsBindings PrepareGraphicsBindings(const ShaderStageRuntime& vertex,
	                                                       const ShaderStageRuntime& pixel,
	                                                       bool                      pixel_active);
	void ResolveRenderColorTarget(uint64_t submit_id, CommandBuffer& buffer,
	                              RenderColorInfo& target, uint32_t render_target_slice_offset = 0,
	                              uint32_t render_target_slot = UINT32_MAX,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(uint64_t submit_id, CommandBuffer& buffer,
	                              RenderDepthInfo& target);
	[[nodiscard]] bool PrepareDrawRenderState(uint64_t submit_id, CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          bool log_setup_phases, DrawRenderState& state);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable, bool log_pipeline_phase,
	                         bool set_bind_debug, bool set_auto_debug);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth);
	[[nodiscard]] bool        ResolveColorTargets(uint64_t submit_id, CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      TrackImageBinding(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);

	// DB_RENDER_CONTROL.DEPTH_CLEAR_ENABLE is a persistent context register, but the clear it
	// requests must happen ONCE. Without this, every render-state change re-materialized the
	// loadOp clear and wiped the depth written by earlier geometry in the same frame.
	// Address of the depth buffer whose pending clear has already been folded into a render
	// state. Tracked per buffer: consuming the clear for one depth target must not suppress it
	// for a different one bound later in the same frame.
	bool                                  m_depth_clear_consumed      = false;
	uint64_t                              m_depth_clear_consumed_addr = 0;
	// Last frame in which each depth buffer was cleared, so a Z-prepass filled earlier in the
	// frame is not wiped by a second clear before the passes that consume it.
	uint64_t                              m_depth_clear_frame_addr    = 0;
	// Depth buffer that has had geometry written into it since its last clear, and the frame that
	// happened in. A clear arriving after the Z-prepass has filled the buffer would wipe exactly the
	// depth the later passes read back, so it must be suppressed no matter how many clears the frame
	// has already issued.
	uint64_t                              m_depth_dirty_addr          = 0;
	uint32_t                              m_depth_dirty_frame         = UINT32_MAX;
	uint32_t                              m_depth_clear_frame         = UINT32_MAX;
	RenderContext&                        m_context;
	std::vector<ImageId>                  m_bound_images;
	std::vector<vk::DescriptorBufferInfo> m_descriptor_buffers;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	// The per-draw descriptor-write and push-constant scratch used to live here, as shared
	// members. They are thread_local in descriptors.cpp now: they are rebuilt from scratch every
	// draw and never read across draws, so they were never shared state in intent - only in
	// storage. Eight workers recording concurrently would have interleaved writes into one array
	// and pushed a torn mix of two draws' constants.
	std::vector<uint32_t>                 m_image_occurrences;
	std::vector<PendingTransition> m_pending_transitions;
	DrawBatchQueue      m_draw_queue;
	bool                m_draining = false;


	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeImageClear(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

// Proves a straight-line uint4 constant-buffer fill without reading guest memory.
[[nodiscard]] bool ResolveComputeConstantBufferFill(const ShaderComputeInputInfo& input,
                                                    uint32_t group_x, uint32_t group_y,
                                                    uint32_t group_z, uint32_t mode,
                                                    ShaderBufferResource& destination,
                                                    ShaderBufferResource& constants);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
