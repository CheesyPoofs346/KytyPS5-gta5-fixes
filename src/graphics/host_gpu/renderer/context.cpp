#include "graphics/host_gpu/renderer/secondaryBatch.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vma.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <bit>
#include <cstring>
namespace Libs::Graphics {

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {}

bool CommandBuffer::IsInvalid() const {
	return m_buffer == nullptr;
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	return m_buffer;
}

namespace {
// Monotonic id for "a command buffer began recording". Handles come from a pool and are
// RECYCLED, so comparing handles cannot tell a fresh recording from a continuing one - anything
// caching per-command-buffer Vulkan state (dynamic state, bind filtering) must key on this.
std::atomic<uint64_t> g_command_generation {0};

// Which secondary the calling thread is recording into, or 0 for the primary.
//
// Thread-local rather than another shared counter: with eight workers a shared bump would make
// every worker's BeginSecondary invalidate every other worker's dynamic-state cache, so every
// draw would re-emit all 13 filtered calls. That trades the exact saving the cache exists for.
thread_local uint64_t t_recording_generation = 0;

// Secondaries are numbered from a distinct high range so a secondary generation can never collide
// with a primary one, whatever order the two counters advance in.
std::atomic<uint64_t> g_secondary_generation {1ull << 40};

std::atomic<uint64_t> g_secondaries_begun {0};
std::atomic<uint64_t> g_state_retargets {0};
} // namespace

uint64_t CurrentCommandGeneration() {
	// A secondary inherits no dynamic state, no pipeline and no index binding, so while one is
	// being recorded its own id is the answer. Falling through to the primary's counter here is
	// the bug this exists to fix: the primary's value does not change when a secondary begins, so
	// a cache keyed on it would report "same recording" and skip re-emitting everything.
	if (t_recording_generation != 0) {
		return t_recording_generation;
	}
	return g_command_generation.load(std::memory_order_relaxed);
}

uint64_t BeginRecordingGeneration() {
	const auto previous = t_recording_generation;
	t_recording_generation =
	    g_secondary_generation.fetch_add(1, std::memory_order_relaxed) + 1;
	g_secondaries_begun.fetch_add(1, std::memory_order_relaxed);
	return previous;
}

void RestoreRecordingGeneration(uint64_t previous) {
	t_recording_generation = previous;
}

void NoteStateRetarget() {
	g_state_retargets.fetch_add(1, std::memory_order_relaxed);
}

void ReportRecordingCensus() {
	const auto begun     = g_secondaries_begun.load(std::memory_order_relaxed);
	const auto retargets = g_state_retargets.load(std::memory_order_relaxed);
	// retargets < begun means at least one secondary was recorded without its state being
	// re-emitted, which is the silent-corruption case. Above 1.0 is expected and fine: the primary
	// retargets too, and a bypassed cache retargets on every draw.
	std::printf("RecordCensus: secondaries_begun=%llu state_retargets=%llu ratio=%.2f%s\n",
	            static_cast<unsigned long long>(begun),
	            static_cast<unsigned long long>(retargets),
	            begun > 0 ? static_cast<double>(retargets) / static_cast<double>(begun) : 0.0,
	            (begun > 0 && retargets < begun) ? "  <-- STATE LEAKED ACROSS A SECONDARY" : "");
	std::fflush(stdout);
}

void CommandBuffer::Begin() {
	g_command_generation.fetch_add(1, std::memory_order_relaxed);
	EXIT_IF(m_rendering || IsInvalid());
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.sType            = vk::StructureType::eCommandBufferBeginInfo;
	begin_info.pNext            = nullptr;
	begin_info.flags            = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	begin_info.pInheritanceInfo = nullptr;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::End() const {
	EndRendering();
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;
}

void CommandBuffer::BeginRendering(const RenderState& state, bool secondary_contents) const {
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	if (m_rendering && m_render_state == state) {
		return;
	}
	EndRendering();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].sType        = vk::StructureType::eRenderingAttachmentInfo;
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp =
		    attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp                 = vk::AttachmentStoreOp::eStore;
		colors[i].clearValue.color.uint32 = attachment.clear_value;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.sType       = vk::StructureType::eRenderingAttachmentInfo;
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.sType       = vk::StructureType::eRenderingAttachmentInfo;
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.sType                = vk::StructureType::eRenderingInfo;
	rendering.flags                = secondary_contents
	                                     ? vk::RenderingFlags {vk::RenderingFlagBits::eContentsSecondaryCommandBuffers}
	                                     : vk::RenderingFlags {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	Handle().beginRendering(rendering);
	m_render_state = state;
	m_rendering    = true;
}

void CommandBuffer::EndRendering() const {
	// The chokepoint for batched draws. Nine call sites end the pass here to record a transfer -
	// buffer uploads, texture uploads, clears, stream copies, compute - and each one then reads a
	// target the batch has not written yet, because its draws are still sitting in an unreplayed
	// secondary. That is the black screen: the work was recorded and submitted, but after whatever
	// consumed the target.
	//
	// Hooking the scheduler's EndRendering was not enough, and enumerating the nine would be the
	// same mistake as enumerating upload consumers. Flushing here cannot miss a caller.
	// Re-entrant: the flush ends the pass it opens, and its own guard stops the recursion.
	FlushSecondaryBatch(m_context);

	if (!m_rendering) {
		return;
	}
	Handle().endRendering();
	m_rendering    = false;
	m_render_state = {};
}

} // namespace Libs::Graphics
