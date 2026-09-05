#include "graphics/host_gpu/renderer/renderDraw.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/drawProfile.h"
#include "graphics/host_gpu/renderer/secondaryBatch.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <thread>
#include <chrono>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace {

// One secondary per draw was correct but pathological: an executeCommands per draw is pure
// overhead, and the ring could only rewind once every buffer retired, so it grew until host
// memory ran out. A batch holds draws until the render state or attachment formats change, which
// is the only thing a secondary's inheritance is tied to.
struct OpenSecondaryBatch {
	vk::CommandBuffer         buffer = nullptr;
	RenderState               rendering {};
	SecondaryRenderingFormats formats {};
	uint32_t                  worker   = 0;
	uint32_t                  draws    = 0;
	bool                      open      = false;
	bool                      flushing  = false;
	// True while a draw is actively recording into the secondary. A flush during that window ends
	// the buffer mid-draw and every later command in the draw is dropped - which is exactly what
	// VUID-vkCmdPushConstants-commandBuffer-recording reported.
	bool                      recording = false;
};

thread_local OpenSecondaryBatch g_secondary_batch;

// Clears the recording flag on every exit path. ExecutePreparedDraw returns early in several
// places after the batch has been joined - shader-skip filters among them - and a flag set by
// hand there stays set forever: every later flush is suppressed, and the next batch tries to
// begin a secondary on a worker that never stopped recording.
class ScopedBatchRecording {
public:
	ScopedBatchRecording() { g_secondary_batch.recording = true; }
	~ScopedBatchRecording() { g_secondary_batch.recording = false; }
	ScopedBatchRecording(const ScopedBatchRecording&)            = delete;
	ScopedBatchRecording& operator=(const ScopedBatchRecording&) = delete;
};

// The state the last flush opened a pass with, and whether there was one.
//
// A pass clears when the guest asked it to, and CommandBuffer::BeginRendering only honours that
// once because it early-outs while the same state is already rendering. Batching ends the pass at
// every flush, so a batch split purely by capacity would re-open with loadOp Clear and wipe what
// the previous batch drew - at 4450 draws against a 256 cap that is ~17 clears a frame, and only
// the last batch survives.
thread_local RenderState g_last_flushed_state {};
thread_local bool        g_have_last_flushed = false;

// Strips the clear requests from a state that is resuming a pass the previous flush already
// cleared. A genuine state change is not affected: it compares unequal and clears as the guest
// asked.
RenderState ResumeStateFor(const RenderState& state) {
	if (!g_have_last_flushed || !(g_last_flushed_state == state)) {
		return state;
	}
	RenderState resumed = state;
	for (auto& attachment: resumed.color_attachments) {
		attachment.is_clear = false;
	}
	resumed.depth_stencil_attachment.is_clear      = false;
	resumed.depth_stencil_attachment.depth_clear   = false;
	resumed.depth_stencil_attachment.stencil_clear = false;
	return resumed;
}

} // namespace


int32_t ResolveVertexOffset(uint32_t index_offset, const ShaderVertexInputInfo& vs_input_info) {
	if (index_offset != 0 || !vs_input_info.fetch_embedded) {
		return static_cast<int32_t>(index_offset);
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = *vs_input_info.stage.program;
	const auto& resources = *vs_input_info.stage.resources;
	if (program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			return static_cast<int32_t>(resources.user_data[index]);
		}
	}

	return 0;
}

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;
static std::atomic<uint32_t> g_shader_stage_log_count = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(RenderColorType type) {
	switch (type) {
		case RenderColorType::NoColorOutput: return "NoColorOutput";
		case RenderColorType::RenderTexture: return "RenderTexture";
		default: return "Unknown";
	}
}

static bool IsDualSourceBlendFactor(uint32_t factor) {
	return factor >= 0x0fu && factor <= 0x12u;
}

static void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF(
	    "DrawFramebufferSkip[%u]: %s color=%s color_addr=0x%010" PRIx64 " color_size=0x%016" PRIx64
	    " color_image=%s depth_format=%s depth_image=%s depth_vaddr_num=%d target_mask=0x%08" PRIx32
	    " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	    log_id, draw_name, RenderColorTypeName(color.type), color.base_addr, color.buffer_size,
	    color.image_id ? "yes" : "no", VulkanToString(depth.format).c_str(),
	    depth.image_id ? "yes" : "no", depth.vaddr_num, ctx.GetRenderTargetMask(),
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	bool interesting = rt_mask != 0x0f || (cb_shader_mask & ~0x0fu) != 0 ||
	                   IsDualSourceBlendFactor(bc0.color_srcblend) ||
	                   IsDualSourceBlendFactor(bc0.color_destblend) ||
	                   (bc0.separate_alpha_blend && (IsDualSourceBlendFactor(bc0.alpha_srcblend) ||
	                                                 IsDualSourceBlendFactor(bc0.alpha_destblend)));

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (color.type == RenderColorType::NoColorOutput) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.program->info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.kind == ShaderRecompiler::IR::ResourceKind::Image ||
		           image.kind == ShaderRecompiler::IR::ResourceKind::ImageUint;
	    });

	vk::Extent2D extent = color.image_id ? color.extent : vk::Extent2D {};
	auto         sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " clear=%s clear_rgba=(%.3f,%.3f,%.3f,%.3f) cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name,
	    RenderColorTypeName(color.type), color.base_addr, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    color.color_clear_enable ? "true" : "false", color.color_clear_value.float32[0],
	    color.color_clear_value.float32[1], color.color_clear_value.float32[2],
	    color.color_clear_value.float32[3], cc.mode, cc.op, bc.enable ? "true" : "false",
	    bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
	    static_cast<int>(ps_resources.images.size()), static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color.type),
	     color.base_addr, index_type_and_size, index_count, reinterpret_cast<uint64_t>(index_addr),
	     vs_input_info.resources_num, vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u attr_num=%d\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index, b.attr_num);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < b.attr_num; ai++) {
					const auto  res_index = b.attr_indices[ai];
					const auto& r         = vs_input_info.resources[res_index];
					const auto& rd        = vs_input_info.resources_dst[res_index];
					const auto  offset    = b.attr_offsets[ai];
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < b.attr_num; ai++) {
			const auto  res_index = b.attr_indices[ai];
			const auto& r         = vs_input_info.resources[res_index];
			const auto& rd        = vs_input_info.resources_dst[res_index];
			LOGF("DrawInputState[%u]: attr[%d] res=%d offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, res_index, b.attr_offsets[ai], rd.register_start, rd.registers_num,
			     rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

namespace {

// Redundant dynamic-state filtering.
//
// Every draw issues 11 vkCmdSet* calls (viewport, scissor, line width, depth bias, six stencil),
// but viewport/scissor/stencil rarely differ between consecutive draws. Each call is driver
// overhead, so skipping the unchanged ones is worth roughly 0.5-1us/draw.
//
// Vulkan dynamic state is per COMMAND BUFFER, so the cache is invalidated whenever the handle
// changes - otherwise a new buffer would inherit state it was never given.
struct DynamicStateCache {
	// Generation, not handle: command buffers come from a pool and handles are RECYCLED, so an
	// unchanged handle does not mean the recording is the same one.
	uint64_t            generation = UINT64_MAX;
	vk::Viewport        viewport {};
	vk::Rect2D          scissor {};
	float               line_width        = -1.0f;
	int                 depth_bias_enable = -1;
	float               bias_constant     = 0.0f;
	float               bias_clamp        = 0.0f;
	float               bias_slope        = 0.0f;
	bool                bias_valid        = false;
	uint32_t            stencil[6] {};
	bool                stencil_valid = false;
	// Bind calls are per-command-buffer state too, so they share this cache's lifetime rules.
	VkPipeline          pipeline      = VK_NULL_HANDLE;
	VkBuffer            index_buffer  = VK_NULL_HANDLE;
	uint64_t            index_offset  = UINT64_MAX;
	uint32_t            index_type    = UINT32_MAX;

	// Returns true when a new command buffer recording started, meaning everything must be
	// re-issued because Vulkan dynamic state does not survive across command buffers.
	bool Retarget() {
		const auto current = CurrentCommandGeneration();
		if (current == generation && Config::DynStateCacheEnabled()) {
			return false;
		}
		*this = {};
		NoteStateRetarget();
		if (!Config::DynStateCacheEnabled()) {
			// Bypassed for A/B: every later comparison must miss. A zeroed viewport would
			// compare equal to a genuinely zero one, so use values that cannot match -
			// NaN never compares equal, and no real scissor is UINT32_MAX wide.
			viewport.width       = std::numeric_limits<float>::quiet_NaN();
			scissor.extent.width = UINT32_MAX;
		}
		generation = current;
		return true;
	}
};

DynamicStateCache& DynState() {
	static thread_local DynamicStateCache cache;
	return cache;
}

bool SameViewport(const vk::Viewport& a, const vk::Viewport& b) {
	return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height &&
	       a.minDepth == b.minDepth && a.maxDepth == b.maxDepth;
}

bool SameScissor(const vk::Rect2D& a, const vk::Rect2D& b) {
	return a.offset.x == b.offset.x && a.offset.y == b.offset.y &&
	       a.extent.width == b.extent.width && a.extent.height == b.extent.height;
}

} // namespace

static void SetGraphicsDynamicParams(const CommandBuffer& buffer, vk::CommandBuffer vk_buffer,
                                     const RenderColorInfo* colors, uint32_t color_count,
                                     const RenderDepthInfo& depth,
                                     uint32_t ps_mrt_output_mask, uint32_t index_count) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(colors == nullptr);
	const auto& ctx = buffer.GetRegisters();

	const auto&  vp = ctx.GetScreenViewport();
	vk::Extent2D framebuffer_extent {};
	if (color_count > 0 && colors[0].image_id) {
		framebuffer_extent = colors[0].extent;
	} else if (depth.image_id) {
		framebuffer_extent = {depth.width, depth.height};
	} else {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		framebuffer_extent = {limits.maxFramebufferWidth, limits.maxFramebufferHeight};
	}

	const auto final_scissor = calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent);

	vk::Viewport viewport {};
	viewport.x        = vp.viewports[0].xoffset - vp.viewports[0].xscale;
	viewport.y        = vp.viewports[0].yoffset - vp.viewports[0].yscale;
	viewport.width    = vp.viewports[0].xscale * 2.0f;
	viewport.height   = vp.viewports[0].yscale * 2.0f;
	viewport.minDepth = vp.viewports[0].zoffset;
	viewport.maxDepth = vp.viewports[0].zscale + vp.viewports[0].zoffset;
	// A collapsed Z range pins every fragment to one depth value. This must be substituted for
	// EVERY affected draw, not just depth-writing ones: correcting the Z-prepass alone while
	// leaving the shading pass on the collapsed range guarantees a compare mismatch. The
	// zoffset test keeps the legitimate [1,1] sky/probe range out of this.
	// A backdrop/skybox legitimately pins itself to the far plane with a collapsed Z range, and it
	// draws as a fullscreen quad (a handful of indices). Remapping those to [0,1] lifts the sky off
	// the far plane and it paints over the scene, so only rescue real meshes.
	//
	// This is load-bearing, not cosmetic. Measured: when the guest collapses the range to [0,0],
	// z = z_ndc * zscale + zoffset makes EVERY depth write land on 0. The depth buffer then reads
	// all-zero even though ~247 depth-writing passes ran, deferred lighting reconstructs every
	// pixel at the far plane, and the scene renders black - which looks like geometry vanishing at
	// certain camera angles. Substituting a usable range is what keeps depth meaningful.
	constexpr uint32_t FullscreenQuadIndexLimit = 32;
	if (Config::FixDegenerateViewportZ() && vp.viewports[0].zscale < 0.001f &&
	    vp.viewports[0].zoffset < 0.001f && index_count > FullscreenQuadIndexLimit) {
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
	}
	// The "mountain range" and the missing models appear/disappear together, so they are the same
	// geometry drawn into a squashed viewport rather than two separate problems. Record the actual
	// viewport rectangle for these draws against the framebuffer it targets.
	if (viewport.height < 0.0f ? (-viewport.height) * 2.0f < static_cast<float>(framebuffer_extent.height)
	                           : viewport.height * 2.0f < static_cast<float>(framebuffer_extent.height)) {
		static std::atomic<uint32_t> squash_log {0};
		if (squash_log.fetch_add(1, std::memory_order_relaxed) < 30) {
			LOGF("SquashedViewport: vp=(%.1f,%.1f) %.1fx%.1f fb=%ux%u idx=%" PRIu32
			     " ps=0x%010" PRIx64 " zs=%.9f\n",
			     static_cast<double>(viewport.x), static_cast<double>(viewport.y),
			     static_cast<double>(viewport.width), static_cast<double>(viewport.height),
			     framebuffer_extent.width, framebuffer_extent.height, index_count,
			     buffer.GetShaders().GetPs().ps_regs.data_addr,
			     static_cast<double>(vp.viewports[0].zscale));
		}
	}
	// Is scene geometry being rendered into a narrow horizontal strip instead of the full target?
	// If so, everything visible lives inside that strip and pitching it off-screen would blank the
	// view - which matches the reported symptom exactly.
	{
		static std::atomic<uint32_t> vp_log {0};
		const float vh = viewport.height < 0.0f ? -viewport.height : viewport.height;
		// Log ONLY partial-height viewports on a full-size target: that is the guest asking us to
		// render into a strip of the scene, which is the state that coincides with the artifact.
		const bool partial = framebuffer_extent.height >= 720 &&
		                     vh + 2.0f < static_cast<float>(framebuffer_extent.height);
		if (index_count > 32 && partial &&
		    vp_log.fetch_add(1, std::memory_order_relaxed) < 60) {
			LOGF("SceneViewport: vp=(%.0f,%.0f) %.0fx%.0f fb=%ux%u idx=%" PRIu32 "\n",
			     static_cast<double>(viewport.x), static_cast<double>(viewport.y),
			     static_cast<double>(viewport.width), static_cast<double>(vh),
			     framebuffer_extent.width, framebuffer_extent.height, index_count);
		}
	}
	auto&      dyn     = DynState();
	const bool retarget = dyn.Retarget();
	if (retarget || !SameViewport(dyn.viewport, viewport)) {
		vk_buffer.setViewport(0, 1, &viewport);
		dyn.viewport = viewport;
	}

	vk::Rect2D scissor {};
	scissor.offset = {final_scissor.left, final_scissor.top};
	scissor.extent = {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
	                  static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)};
	// Diagnostic: a scissor far shorter than the framebuffer confines a full-screen pass to a
	// horizontal band. That is the signature of the sky/terrain strip artifact.
	if (framebuffer_extent.height >= 64 &&
	    scissor.extent.height * 2u < framebuffer_extent.height) {
		static std::atomic<uint32_t> band_log {0};
		if (band_log.fetch_add(1, std::memory_order_relaxed) < 24) {
			LOGF("BandScissor: scissor=(%d,%d) %ux%u framebuffer=%ux%u vp_y=%.1f vp_h=%.1f"
			     " window_offset=%d,%d window_br=%d,%d\n",
			     scissor.offset.x, scissor.offset.y, scissor.extent.width, scissor.extent.height,
			     framebuffer_extent.width, framebuffer_extent.height,
			     static_cast<double>(viewport.y), static_cast<double>(viewport.height),
			     vp.window_offset_x, vp.window_offset_y, vp.window_scissor_right, vp.window_scissor_bottom);
		}
		if (Config::SuppressBandPass()) {
			// Collapse the scissor to nothing so the band never reaches the framebuffer. Scissoring
			// rather than skipping the draw keeps all other pipeline state and side effects intact.
			scissor.extent = {0, 0};
		}
	}
	if (retarget || !SameScissor(dyn.scissor, scissor)) {
		vk_buffer.setScissor(0, 1, &scissor);
		dyn.scissor = scissor;
	}

	// UI-draw dump, for the oversized minimap. The icons render at the right size and the map does
	// not, so whatever differs between them is visible here: both are low-index quads drawn late in
	// the frame. Viewport and scissor come straight from guest registers with no scaling anywhere
	// in this path, so if the map's viewport is already ~4x too large the guest was told the wrong
	// display size, and if it is correct the fault is downstream in the composite.
	if (Config::LogUiDrawsEnabled() && index_count <= 64) {
		// Log a burst, then go quiet, then burst again. A plain first-400 cap spent itself entirely
		// on boot quads and never saw the HUD - twice. This keeps sampling into gameplay.
		static std::atomic<uint32_t> ui_seen {0};
		const auto                   seen = ui_seen.fetch_add(1, std::memory_order_relaxed);
		constexpr uint32_t           kPeriod = 20000;
		constexpr uint32_t           kBurst  = 60;
		const auto                   n       = seen % kPeriod;
		if (n < kBurst) {
			std::printf("UiDraw[%u]: idx=%" PRIu32 " vp=(%.1f,%.1f) %.1fx%.1f scissor=(%d,%d) %ux%u "
			            "fb=%ux%u ps=0x%010" PRIx64 "\n",
			            n, index_count, static_cast<double>(viewport.x),
			            static_cast<double>(viewport.y), static_cast<double>(viewport.width),
			            static_cast<double>(viewport.height), scissor.offset.x, scissor.offset.y,
			            scissor.extent.width, scissor.extent.height, framebuffer_extent.width,
			            framebuffer_extent.height, buffer.GetShaders().GetPs().ps_regs.data_addr);
			std::fflush(stdout);
		}
	}

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	if (retarget || dyn.line_width != line_width) {
		vk_buffer.setLineWidth(line_width);
		dyn.line_width = line_width;
	}

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	if (retarget || dyn.depth_bias_enable != (depth_bias_enable ? 1 : 0)) {
		vk_buffer.setDepthBiasEnable(depth_bias_enable ? VK_TRUE : VK_FALSE);
		dyn.depth_bias_enable = depth_bias_enable ? 1 : 0;
	}
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor =
		    ConvertPolygonOffsetConstantFactor(guest_constant_factor, poly_offset, depth.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		if (retarget || !dyn.bias_valid || dyn.bias_constant != constant_factor ||
		    dyn.bias_clamp != poly_offset.clamp || dyn.bias_slope != slope_factor) {
			vk_buffer.setDepthBias(constant_factor, poly_offset.clamp, slope_factor);
			dyn.bias_constant = constant_factor;
			dyn.bias_clamp    = poly_offset.clamp;
			dyn.bias_slope    = slope_factor;
			dyn.bias_valid    = true;
		}
	}

	if (depth.stencil_test_enable) {
		const uint32_t stencil[6] {
		    depth.stencil_dynamic_front.compareMask, depth.stencil_dynamic_back.compareMask,
		    depth.stencil_dynamic_front.writeMask,   depth.stencil_dynamic_back.writeMask,
		    depth.stencil_dynamic_front.reference,   depth.stencil_dynamic_back.reference};
		if (retarget || !dyn.stencil_valid ||
		    std::memcmp(dyn.stencil, stencil, sizeof(stencil)) != 0) {
			vk_buffer.setStencilCompareMask(vk::StencilFaceFlagBits::eFront, stencil[0]);
			vk_buffer.setStencilCompareMask(vk::StencilFaceFlagBits::eBack, stencil[1]);
			vk_buffer.setStencilWriteMask(vk::StencilFaceFlagBits::eFront, stencil[2]);
			vk_buffer.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, stencil[3]);
			vk_buffer.setStencilReference(vk::StencilFaceFlagBits::eFront, stencil[4]);
			vk_buffer.setStencilReference(vk::StencilFaceFlagBits::eBack, stencil[5]);
			std::memcpy(dyn.stencil, stencil, sizeof(stencil));
			dyn.stencil_valid = true;
		}
	}

#if defined(__APPLE__)
	// MoltenVK has no VK_EXT_color_write_enable; the pipeline is created without the
	// eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead.
#else
	vk::Bool32 enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	// Color-control operation selects special color-buffer paths, not the normal component write
	// mask. Attachment availability therefore follows the target write mask.
	for (uint32_t i = 0; i < color_count; i++) {
		// The CB leaves an MRT untouched when the pixel shader does not export it, but Vulkan
		// leaves an unwritten attachment UNDEFINED. Binding a G-buffer target the shader never
		// writes would therefore fill it with garbage instead of preserving it, so mask those
		// attachments off. A zero mask means "unknown", and keeps the previous behaviour.
		// Only meaningful for real MRT passes; a single attachment cannot hit this, so leave the
		// common path exactly as it was.
		// Off by default: unverified, and if mrt_output_mask ever under-reports a shader export
		// this would disable a colour write that should happen - i.e. it could CAUSE missing
		// geometry, the very symptom being debugged. Enable with --mask-unwritten-mrt to A/B it.
		const bool shader_writes_slot =
		    !Config::MaskUnwrittenMrt() || color_count < 2 || ps_mrt_output_mask == 0 ||
		    (ps_mrt_output_mask & (1u << colors[i].target_slot)) != 0;
		enable[i] =
		    (render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot) != 0 &&
		     shader_writes_slot)
		        ? VK_TRUE
		        : VK_FALSE;
	}
	if (color_count != 0) {
		vk_buffer.setColorWriteEnableEXT(color_count, enable);
	}
#endif
}

// Draw-acceptance census: every early return in the draw path is silent, so missing geometry
// is invisible without counting each reason.
static std::atomic<uint64_t> g_draw_accepted {0};
static std::atomic<uint64_t> g_draw_skip_empty {0};
static std::atomic<uint64_t> g_draw_skip_metadata {0};
static std::atomic<uint64_t> g_draw_skip_no_vs {0};
static std::atomic<uint64_t> g_draw_skip_ge {0};

static std::atomic<uint64_t> g_draw_census_total {0};

// Runs for every draw, so keep the common path to one relaxed increment and a modulo; only
// gather the individual counters on the rare reporting tick.
// Per-shader draw census.
//
// Frame time = draws x ~19us, so the question worth asking is WHICH shaders produce the draws.
// Keyed on the shader checksum, not its address: addresses are guest pointers that change every
// launch, so an address captured in one run silently matches nothing in the next.
namespace {

constexpr size_t kCensusSize = 512;

struct CensusSlot {
	std::atomic<uint64_t> addr {0};
	std::atomic<uint64_t> count {0};
};
std::array<CensusSlot, kCensusSize> g_census;
std::atomic<uint64_t>               g_census_frames {0};

void NoteShaderDraw(uint64_t ps_addr) {
	size_t i = static_cast<size_t>((ps_addr * 0x9e3779b97f4a7c15ull) >> 55u) % kCensusSize;
	for (size_t probe = 0; probe < 8; probe++, i = (i + 1) % kCensusSize) {
		auto&      slot = g_census[i];
		const auto cur  = slot.addr.load(std::memory_order_relaxed);
		if (cur == ps_addr) {
			slot.count.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		if (cur == 0) {
			uint64_t expected = 0;
			if (slot.addr.compare_exchange_strong(expected, ps_addr, std::memory_order_relaxed) ||
			    slot.addr.load(std::memory_order_relaxed) == ps_addr) {
				slot.count.fetch_add(1, std::memory_order_relaxed);
				return;
			}
		}
	}
}

void ReportCensus() {
	const auto frames = g_census_frames.exchange(0, std::memory_order_relaxed);
	if (frames == 0) {
		return;
	}
	std::array<std::pair<uint64_t, uint64_t>, kCensusSize> rows {};
	uint64_t                                              total = 0;
	for (size_t i = 0; i < kCensusSize; i++) {
		rows[i] = {g_census[i].count.exchange(0, std::memory_order_relaxed),
		           g_census[i].addr.load(std::memory_order_relaxed)};
		total += rows[i].first;
	}
	if (total == 0) {
		return;
	}
	std::sort(rows.begin(), rows.end(),
	          [](const auto& a, const auto& b) { return a.first > b.first; });
	std::printf("ShaderCensus: %.0f draws/frame over %llu frames - top pixel shaders:\n",
	            static_cast<double>(total) / static_cast<double>(frames),
	            static_cast<unsigned long long>(frames));
	for (size_t i = 0; i < 12 && rows[i].first != 0; i++) {
		std::printf("  --skip-ps 0x%010llx   %6.0f draws/frame  %5.1f%%\n",
		            static_cast<unsigned long long>(rows[i].second),
		            static_cast<double>(rows[i].first) / static_cast<double>(frames),
		            static_cast<double>(rows[i].first) / static_cast<double>(total) * 100.0);
	}
	std::fflush(stdout);
}

} // namespace

static void DrawCensusTick() {
	const auto total = g_draw_census_total.fetch_add(1, std::memory_order_relaxed) + 1;
	if (total % 20000 != 0) {
		return;
	}
	// LOGF is silenced in this build, so this census has never printed once and the skip rate is
	// still unknown - and skipped draws dilute every per-draw average the profiler reports.
	std::printf("DrawCensus: accepted=%" PRIu64 " skip_empty=%" PRIu64 " skip_metadata=%" PRIu64
	     " skip_no_vs=%" PRIu64 " skip_ge=%" PRIu64 "\n",
	     g_draw_accepted.load(std::memory_order_relaxed),
	     g_draw_skip_empty.load(std::memory_order_relaxed),
	     g_draw_skip_metadata.load(std::memory_order_relaxed),
	     g_draw_skip_no_vs.load(std::memory_order_relaxed),
	     g_draw_skip_ge.load(std::memory_order_relaxed));
	std::fflush(stdout);
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.gs_regs.chksum != 0 && ShaderAddressValid(vs.es_regs.data_addr);
}

static bool PixelShaderHasDepthOrCoverageSideEffects(const HW::ShaderRegisters& sh_regs) {
	const auto& db = sh_regs.db_shader_control;
	return sh_regs.shader_z_format != 0 || db.shader_kill_enable || db.shader_z_export_enable ||
	       db.shader_mask_export_enable || db.shader_dual_export_enable ||
	       db.shader_execute_on_noop;
}

static bool ShouldSkipGeShader(const CommandBuffer& buffer) {
	const auto& ctx         = buffer.GetRegisters();
	const auto& ucfg        = buffer.GetUserConfig();
	const auto& sh_ctx      = buffer.GetShaders();
	const auto& sh_regs     = ctx.GetShaderRegisters();
	const auto& ge_cntl     = ucfg.GetGeControl();
	const auto& vertex_info = sh_ctx.GetVs();
	const auto  stages      = ctx.GetShaderStages();

	const auto is_known_gs_out_prim_type = [](uint32_t value) {
		switch (static_cast<Prospero::GsOutputPrimitiveType>(value)) {
			case Prospero::GsOutputPrimitiveType::kPoints:
			case Prospero::GsOutputPrimitiveType::kLines:
			case Prospero::GsOutputPrimitiveType::kTriangles:
			case Prospero::GsOutputPrimitiveType::k2dRectangle:
			case Prospero::GsOutputPrimitiveType::kRectList: return true;
		}

		return false;
	};

	const bool ps5_ngg_vertex_path = stages == 0x02002000 && vertex_info.es_regs.data_addr != 0 &&
	                                 vertex_info.gs_regs.chksum != 0 &&
	                                 sh_regs.m_vgtGsMaxVertOut == 0x00000000 &&
	                                 is_known_gs_out_prim_type(sh_regs.m_vgtGsOutPrimType);

	const bool unsupported_stage_mask = (stages != 0 && stages != 0x02002000);
	const bool unsupported_gs_stage = (vertex_info.es_regs.data_addr != 0 &&
	                                   vertex_info.gs_regs.data_addr != 0 && !ps5_ngg_vertex_path);
	// GE_CNTL group sizes control guest scheduling and do not constrain the host vertex path.
	const bool ge_shader_regs =
	    (sh_regs.m_geNggSubgrpCntl != 0x00000000 && sh_regs.m_geNggSubgrpCntl != 0x00000001) ||
	    sh_regs.m_vgtGsMaxVertOut != 0x00000000 ||
	    !is_known_gs_out_prim_type(sh_regs.m_vgtGsOutPrimType) ||
	    sh_regs.m_geMaxOutputPerSubgroup > 0x00000040;

	if (unsupported_stage_mask || unsupported_gs_stage || ge_shader_regs) {
		// Which pipelines are actually being dropped, and how many draws does that cost? Thin
		// geometry (palm fronds, power cables) is missing in-game and tessellated draws are the
		// prime suspect.
		{
			static std::atomic<uint64_t> s_drops {0};
			static std::atomic<uint64_t> s_mask_tess {0};
			static std::atomic<uint64_t> s_mask_gs {0};
			static std::atomic<uint64_t> s_mask_ge {0};
			s_mask_tess.fetch_add(unsupported_stage_mask ? 1 : 0, std::memory_order_relaxed);
			s_mask_gs.fetch_add(unsupported_gs_stage ? 1 : 0, std::memory_order_relaxed);
			s_mask_ge.fetch_add(ge_shader_regs ? 1 : 0, std::memory_order_relaxed);
			const auto n = s_drops.fetch_add(1, std::memory_order_relaxed) + 1;
			if (n % 20000 == 0) {
				std::printf("DroppedDraws: 20000 | stage_mask=%llu gs=%llu ge_regs=%llu | "
				            "last stages=0x%08x prim=%u\n",
				            static_cast<unsigned long long>(
				                s_mask_tess.exchange(0, std::memory_order_relaxed)),
				            static_cast<unsigned long long>(
				                s_mask_gs.exchange(0, std::memory_order_relaxed)),
				            static_cast<unsigned long long>(
				                s_mask_ge.exchange(0, std::memory_order_relaxed)),
				            stages,
				            static_cast<uint32_t>(buffer.GetUserConfig().GetPrimType()));
				std::fflush(stdout);
			}
		}
		static std::once_flag warning_once;
		std::call_once(warning_once, [] {
			std::printf("Warning: game uses unsupported graphics pipelines; some draw calls were "
			            "skipped.\n");
		});

		const auto log_id = g_shader_stage_log_count.fetch_add(1);
		if (log_id < 32) {
			LOGF("Skipping unsupported GE shader draw: stages=0x%08" PRIx32
			     " prim_group=0x%04" PRIx16 " vert_group=0x%04" PRIx16 " ngg=0x%08" PRIx32
			     " max_out=0x%08" PRIx32 " gs_max_vert=0x%08" PRIx32 " gs_out_prim=0x%08" PRIx32
			     " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     stages, ge_cntl.primitive_group_size, ge_cntl.vertex_group_size,
			     sh_regs.m_geNggSubgrpCntl, sh_regs.m_geMaxOutputPerSubgroup,
			     sh_regs.m_vgtGsMaxVertOut, sh_regs.m_vgtGsOutPrimType,
			     vertex_info.es_regs.data_addr, vertex_info.gs_regs.data_addr);
		}
		return true;
	}

	return false;
}

struct DrawRenderState {
	RenderDepthInfo       depth_info;
	RenderColorInfo       color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t              color_count                              = 0;
	bool                  ps_active                                = true;
	RenderState           rendering;
	ShaderVertexInputInfo vs_input_info;
	ShaderPixelInputInfo  ps_input_info;
	ShaderProgram         vertex_program;
	ShaderProgram         pixel_program;
};

struct DrawCallInfo {
	const char*          name           = nullptr;
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             flags          = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;
};

static bool ResolveDccAttachmentClear(TextureCache& cache, const RenderColorInfo& target,
                                      const ImageViewInfo& view, vk::ClearColorValue& clear_value) {
	if (target.desc.info.metadata.kind != ImageMetadataKind::Dcc) {
		return false;
	}
	return cache.ResolveDccMetaClear(target.desc.info.metadata.range.address, view.base_layer,
	                                 view.layer_count, clear_value);
}

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth) {
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = color_count;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto old_image = cache.m_slot_images.try_get(target.image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			target.image_id = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
		target.image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto& image       = cache.GetImage(target.image_id);
		SetVulkanObjectNameF(m_context.GetGraphics().device, image.backing.image,
		                     "Kyty.MRT{}.Image[guest=0x{:016x} size=0x{:x} format={}]",
		                     target.target_slot, image.info.data.address, image.info.data.size,
		                     static_cast<uint32_t>(image.info.pixel_format));
		SetVulkanObjectNameF(m_context.GetGraphics().device, target.image_view,
		                     "Kyty.MRT{}.View[guest=0x{:016x} mip={} layer={}+{}]",
		                     target.target_slot, image.info.data.address,
		                     target.desc.view_info.base_level, target.desc.view_info.base_layer,
		                     target.desc.view_info.layer_count);
		EXIT_IF(image.backing.samples != target.samples || target.image_view == nullptr);
		if (attachment_samples == 0) {
			attachment_samples = target.samples;
		} else if (attachment_samples != target.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, target.samples);
		}
		const auto& view = target.desc.view_info;
		// Sample the target's incoming contents before this pass writes to it. The
		// following Transit restores the attachment layout.
		// GetGpu() aborts when no GPU is attached (the host test harness drives RenderContext
		// directly), so only reach for the frame number when the probe is actually enabled.
		if (m_context.GetHdrProbe().Enabled()) {
			m_context.GetHdrProbe().Capture(
			    image, static_cast<uint32_t>(m_context.GetGpu().GetFrameNum()), view.base_level,
			    view.base_layer, image.binding.is_bound,
			    static_cast<uint32_t>(target.export_mapping.packed));
		}
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		// Audited as part of the parallel-resolution work: this is a primary-buffer recording on
		// the per-draw resolve path, same as CommitBindings' transitions, so it stages the same
		// way. The attachment layout written below is the one being requested, which is what
		// BeginRendering must agree with either way.
		const ImageSubresourceRange color_range {view.base_level, view.level_count, view.base_layer,
		                                         view.layer_count};
		constexpr auto color_access = vk::AccessFlagBits2::eColorAttachmentRead |
		                              vk::AccessFlagBits2::eColorAttachmentWrite;
		if (Config::DeferTransitionsEnabled() || MustStageForWorker()) {
			m_pending_transitions.push_back(
			    PendingTransition {target.image_id, layout, color_access, color_range});
		} else {
			image.Transit(layout, color_access, color_range, buffer.Handle());
		}
		state.width             = std::min(state.width, target.extent.width);
		state.height            = std::min(state.height, target.extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		auto& attachment        = state.color_attachments[i];
		attachment.image_view   = target.image_view;
		attachment.image_layout = layout;
		attachment.clear_value  = target.color_clear_value.uint32;
		vk::ClearColorValue metadata_clear_value {};
		const bool          metadata_clear =
		    ResolveDccAttachmentClear(cache, target, view, metadata_clear_value);
		if (metadata_clear) {
			attachment.clear_value = metadata_clear_value.uint32;
		}
		attachment.is_clear = target.color_clear_enable || metadata_clear;
	}
	if (depth.image_id) {
		const auto owner = cache.m_slot_images.try_get(depth.image_id);
		if (owner == nullptr || !owner->registered || owner->binding.needs_rebind) {
			EXIT("depth target changed after render-state discovery\n");
		}
		depth.image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		if (depth.htile && depth.depth_clear_enable && !cache.ClearMeta(depth.htile_buffer_vaddr)) {
			EXIT("failed to acquire HTile metadata for a depth clear\n");
		}
		depth.depth_meta_clear_enable =
		    depth.htile &&
		    cache.IsMetaCleared(depth.htile_buffer_vaddr, depth.desc.view_info.base_layer);
		depth.depth_load_clear_enable = depth.depth_clear_enable || depth.depth_meta_clear_enable;
		if (depth.depth_meta_clear_enable &&
		    !cache.TouchMeta(depth.htile_buffer_vaddr, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		SetVulkanObjectNameF(m_context.GetGraphics().device, image.backing.image,
		                     "Kyty.DepthTarget.Image[guest=0x{:016x} size=0x{:x} format={}]",
		                     image.info.data.address, image.info.data.size,
		                     static_cast<uint32_t>(image.info.pixel_format));
		SetVulkanObjectNameF(m_context.GetGraphics().device, depth.image_view,
		                     "Kyty.DepthTarget.View[guest=0x{:016x} layer={}+{}]",
		                     image.info.data.address, depth.desc.view_info.base_layer,
		                     depth.desc.view_info.layer_count);
		EXIT_IF(depth.image_view == nullptr || image.backing.samples != depth.samples);
		if (attachment_samples == 0) {
			attachment_samples = depth.samples;
		} else if (attachment_samples != depth.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.samples);
		}
		const auto layout = depth_attachment_layout(depth);
		const auto writes = depth.AttachmentWriteAspects();
		auto       access = vk::AccessFlags2 {vk::AccessFlagBits2::eDepthStencilAttachmentRead};
		if (writes) {
			access |= vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		}
		const auto& view = depth.desc.view_info;
		const ImageSubresourceRange depth_range {view.base_level, view.level_count, view.base_layer,
		                                         view.layer_count};
		if (Config::DeferTransitionsEnabled() || MustStageForWorker()) {
			m_pending_transitions.push_back(
			    PendingTransition {depth.image_id, layout, access, depth_range});
		} else {
			image.Transit(layout, access, depth_range, buffer.Handle());
		}
		state.width               = std::min(state.width, depth.width);
		state.height              = std::min(state.height, depth.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = depth.image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		bool depth_clear_allowed = depth.depth_load_clear_enable;
		if (Config::DepthClearPerFrame() && depth_clear_allowed && m_context.HasGpu()) {
			const auto frame_now = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
			const bool already_has_geometry = m_depth_dirty_frame == frame_now &&
			                                  m_depth_dirty_addr == depth.depth_buffer_vaddr;
			if (already_has_geometry) {
				// The prepass has already filled this buffer this frame; clearing now would discard
				// it and every later depth-testing pass would fail.
				depth_clear_allowed = false;
			} else if (m_depth_clear_frame == frame_now &&
			           m_depth_clear_frame_addr == depth.depth_buffer_vaddr) {
				// Already cleared this buffer this frame: a second clear would discard the depth a
				// Z-prepass just produced, which later GEQUAL passes read back.
				depth_clear_allowed = false;
			} else {
				m_depth_clear_frame      = frame_now;
				m_depth_clear_frame_addr = depth.depth_buffer_vaddr;
			}
		}
		attachment.depth_clear    = depth_clear_allowed;
		if (!depth_clear_allowed && depth.depth_write_enable && m_context.HasGpu()) {
			// Geometry is about to write depth into this buffer; remember that so a later clear in the
			// same frame cannot wipe it.
			m_depth_dirty_frame = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
			m_depth_dirty_addr  = depth.depth_buffer_vaddr;
		}
		if (depth.depth_load_clear_enable) {
			// The clear has now been folded into a render state and will be applied as a loadOp.
			// DB_RENDER_CONTROL.DEPTH_CLEAR_ENABLE is a persistent context register, so without
			// consuming it here every later render-state change in the same episode would re-arm the
			// clear and wipe depth mid-frame. ResolveRenderDepthTarget re-arms it once the guest
			// lowers the register again.
			m_depth_clear_consumed      = true;
			m_depth_clear_consumed_addr = depth.depth_buffer_vaddr;
		}
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear  = depth.stencil_clear_enable;
		m_context.GetHdrProbe().NoteDepthClear(depth.depth_load_clear_enable,
		                                       depth.depth_meta_clear_enable,
		                                       depth.depth_clear_value, depth.depth_test_enable,
		                                       depth.depth_write_enable,
		                                       static_cast<uint32_t>(depth.depth_compare_op));
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	} else if (attachment_samples == 0 ||
	           vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {}) {
		EXIT("render state has no valid attachments\n");
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

static bool DrawHasActivePixelShader(const CommandBuffer& buffer) {
	const auto& ctx              = buffer.GetRegisters();
	const auto& sh_regs          = ctx.GetShaderRegisters();
	const bool  has_color_output = (ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask) != 0;
	return ShaderAddressValid(buffer.GetShaders().GetPs().ps_regs.data_addr) &&
	       (has_color_output || PixelShaderHasDepthOrCoverageSideEffects(sh_regs));
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

static bool ConsumeMetadataColorOperation(const CommandBuffer& buffer) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	// These special modes run color-buffer metadata or decompression operations. The shader is a
	// vehicle for that operation, and its exported color must not be applied as a normal draw.
	// Kyty stores expanded Vulkan images rather than compressed guest surfaces, so no equivalent
	// hardware pass is emitted. Tracked DCC clear state is materialized on attachment bind;
	// future CMask/FMask support can consume its state through the same TextureCache path.
	return mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear) ||
	       mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

struct DrawEmitInfo {
	bool     indexed       = false;
	int32_t  vertex_offset = 0;
	uint32_t first_vertex  = 0;
};

struct DrawIndexBufferSource {
	bool          enabled   = false;
	uint64_t      address   = 0;
	const void*   host_data = nullptr;
	uint64_t      size      = 0;
	vk::IndexType type      = vk::IndexType::eUint16;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	uint64_t       size   = 0;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

static uint64_t VertexBufferDescriptorSize(const ShaderVertexInputBuffer& buffer) {
	return (buffer.stride != 0 ? static_cast<uint64_t>(buffer.stride) * buffer.num_records
	                           : buffer.num_records);
}

struct VertexBufferRange {
	uint64_t                     base_address  = 0;
	uint64_t                     requested_end = 0;
	uint64_t                     acquired_end  = 0;
	std::pair<Buffer*, uint64_t> binding;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }
};

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers {};
	std::array<vk::DeviceSize, MaxBuffers> offsets {};
	uint32_t                               count = 0;
};

static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&               buffer,
                                                  const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	// Collect the non-empty guest vertex ranges.
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges {};
	uint32_t                                                      range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(vertex);
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vertex.addr, size);
		}
		ranges[range_count++] = {vertex.addr, vertex.addr + size};
	}

	std::sort(ranges.begin(), ranges.begin() + range_count,
	          [](const VertexBufferRange& left, const VertexBufferRange& right) {
		          return left.base_address < right.base_address;
	          });

	// Merge overlapping or touching ranges before acquiring host buffers.
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged_ranges {};
	uint32_t                                                      merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = ranges[i];
		if (merged_count != 0 &&
		    merged_ranges[merged_count - 1].requested_end >= range.base_address) {
			merged_ranges[merged_count - 1].requested_end =
			    std::max(merged_ranges[merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		merged_ranges[merged_count++] = {range.base_address, range.requested_end};
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		// PPSA20298
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		range.acquired_end = range.base_address + size;
		range.binding      = cache.ObtainBuffer(range.base_address, size, false);
		SetVulkanObjectNameF(
		    buffer.GetContext().GetGraphics().device, range.binding.first->Handle(),
		    "Kyty.VertexBufferRange[guest=0x{:016x} size=0x{:x}]", range.base_address, size);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(vertex);
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto range = std::find_if(merged_ranges.begin(), merged_ranges.begin() + merged_count,
		                                [&](const VertexBufferRange& value) {
			                                return vertex.addr >= value.base_address &&
			                                       vertex.addr < value.acquired_end;
		                                });
		if (range == merged_ranges.begin() + merged_count) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->binding.first->Handle();
		prepared.offsets[i] = range->binding.second + vertex.addr - range->base_address;
		SetVulkanObjectNameF(
		    buffer.GetContext().GetGraphics().device, prepared.buffers[i],
		    "Kyty.VertexBuffer[slot={} guest=0x{:016x} size=0x{:x} stride={} records={}]", i,
		    vertex.addr, size, vertex.stride, vertex.num_records);
	}

	return prepared;
}

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              uint32_t phase) {
	EXIT_IF(draw.name == nullptr);

	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count,
	                    draw.flags, draw.instance_count, draw.first_instance);
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, bool auto_draw,
                            vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kRectList:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kRectListLegacy:
			if (!auto_draw) {
				EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}

	return true;
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer, vk::PrimitiveTopology topology,
                                    uint32_t index_type_and_size) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}
	if (topology != vk::PrimitiveTopology::eLineStrip &&
	    topology != vk::PrimitiveTopology::eTriangleStrip &&
	    topology != vk::PrimitiveTopology::eTriangleFan) {
		return false;
	}

	uint32_t index_mask = 0;
	switch (static_cast<Prospero::IndexType>(index_type_and_size)) {
		case Prospero::IndexType::kIndex8: index_mask = 0xffu; break;
		case Prospero::IndexType::kIndex16: index_mask = 0xffffu; break;
		case Prospero::IndexType::kIndex32: index_mask = 0xffffffffu; break;
		default: EXIT("unknown index_type_and_size: %u\n", index_type_and_size);
	}

	const auto reset_index = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	EXIT_NOT_IMPLEMENTED((reset_index & index_mask) != index_mask);
	return true;
}

bool RenderExecutor::PrepareDrawRenderState(uint64_t submit_id, CommandBuffer& buffer,
                                            const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
                                            bool log_setup_phases, DrawRenderState& state) {
	EXIT_IF(draw.name == nullptr);
	auto& ctx = buffer.GetRegisters();

	if (ResolveColorTargets(submit_id, buffer, render_target_slice_offset)) {
		return false;
	}
	if (log_setup_phases) {
		LogDrawPhase(draw.name, "ResolveRenderColorTarget");
	}
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if (slot == 0 || (render_target_mask_slot(ctx.GetRenderTargetMask(), slot) != 0 &&
		                  ctx.GetRenderTarget(slot).base.addr != 0)) {
			ResolveRenderColorTarget(submit_id, buffer, state.color_info[state.color_count],
			                         render_target_slice_offset, slot);
			if (state.color_info[state.color_count].image_id) {
				state.color_count++;
			}
		}
	}
	if (log_setup_phases) {
		LogDrawPhase(draw.name, "ResolveRenderDepthTarget");
	}
	ResolveRenderDepthTarget(submit_id, buffer, state.depth_info);

	state.ps_active       = DrawHasActivePixelShader(buffer);
	const bool with_depth = (state.depth_info.format != vk::Format::eUndefined &&
	                         static_cast<bool>(state.depth_info.image_id));
	if (state.color_count == 0 && !with_depth && !state.ps_active) {
		LogFramebufferSkip(draw.name, state.color_info[0], state.depth_info, buffer,
		                   draw.index_count, draw.flags);
		return false;
	}

	return true;
}

namespace {

// One persistent worker for shader resource resolution.
//
// Resolution is ~21% of the frame and is pure - it reads guest memory and writes only its own
// input_info. Vertex and pixel resolution are independent once both permutations are located,
// so one runs here while the other runs on the render thread.
//
// Spin-then-yield rather than a condition variable: the handoff happens thousands of times per
// frame and a futex wake costs more than the work saved. The cores are idle anyway.
class ResolveWorker {
public:
	static ResolveWorker& Instance() {
		static ResolveWorker worker;
		return worker;
	}

	// Returns false if the worker is busy; caller then does the work inline.
	bool Dispatch(std::function<void()> job) {
		if (m_state.load(std::memory_order_acquire) != State::Idle) {
			return false;
		}
		m_job = std::move(job);
		m_state.store(State::Pending, std::memory_order_release);
		return true;
	}

	void Wait() {
		for (uint32_t spins = 0; m_state.load(std::memory_order_acquire) != State::Done; spins++) {
			if (spins > 2000) {
				std::this_thread::yield();
			}
		}
		m_state.store(State::Idle, std::memory_order_release);
	}

private:
	enum class State : uint32_t { Idle, Pending, Done };

	ResolveWorker() {
		m_thread = std::thread([this] {
			for (;;) {
				uint32_t spins = 0;
				while (m_state.load(std::memory_order_acquire) != State::Pending) {
					if (m_stop.load(std::memory_order_acquire)) {
						return;
					}
					if (++spins > 2000) {
						std::this_thread::yield();
					}
				}
				m_job();
				m_state.store(State::Done, std::memory_order_release);
			}
		});
	}

	~ResolveWorker() {
		m_stop.store(true, std::memory_order_release);
		if (m_thread.joinable()) {
			m_thread.join();
		}
	}

	std::atomic<State>    m_state {State::Idle};
	std::atomic<bool>     m_stop {false};
	std::function<void()> m_job;
	std::thread           m_thread;
};

} // namespace

static void RefreshShaders(CommandBuffer& buffer, const DrawCallInfo& draw, bool log_phases,
                           DrawRenderState& state, PreparedShaders* prepared = nullptr) {
	// Already resolved on a worker: the walk is the expensive half and it is done, so take the
	// result rather than repeat it. Only a permutation that needed compiling arrives unprepared.
	if (prepared != nullptr && prepared->valid) {
		state.vs_input_info  = prepared->vs_input_info;
		state.ps_input_info  = prepared->ps_input_info;
		state.vertex_program = prepared->vertex_program;
		state.pixel_program  = prepared->pixel_program;
		return;
	}
	EXIT_IF(draw.name == nullptr);
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	const auto& vertex_shader_info = sh_ctx.GetVs();
	const auto& pixel_shader_info  = sh_ctx.GetPs();
	const auto& shader_regs        = ctx.GetShaderRegisters();

	state.vertex_program = {};
	state.pixel_program  = {};
	state.ps_input_info  = {};
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
	    target_export_mapping {};
	for (uint32_t i = 0; i < state.color_count; i++) {
		target_export_mapping[state.color_info[i].target_slot] = state.color_info[i].export_mapping;
	}
	if (log_phases) {
		LogDrawPhase(draw.name, "GetVertexProgram");
	}
	auto& pipeline_cache = buffer.GetContext().GetPipelineCache();

	// Lookup is locked, resolution is not. The default path used to call GetVertexProgram, which
	// holds PipelineCache::m_mutex across the whole SRT walk - 3.4 us/draw, 26% of the draw, spent
	// inside a lock that shader compilation on other threads also wants. Splitting it is a
	// prerequisite for any parallel recording, and costs nothing single-threaded.
	if (state.ps_active) {
		// Locate both permutations first (cheap, locked). Vertex lookup publishes
		// vs_input_info.stage.program, which is all pixel preparation needs - so pixel params can
		// be built before either stage's resources are resolved. The two resolutions are then
		// independent and overlap.
		const auto vs_params = pipeline_cache.PrepareVertexParams(vertex_shader_info, shader_regs,
		                                                          state.vs_input_info);
		PipelineCache::ProgramRef vs_program;
		ShaderProgram             vs_handle {};
		if (pipeline_cache.LookupVertexProgram(vs_params, state.vs_input_info, vs_handle,
		                                       vs_program)) {
			const auto ps_params =
			    pipeline_cache.PreparePixelParams(pixel_shader_info, shader_regs,
			                                      state.vs_input_info, target_export_mapping,
			                                      state.ps_input_info);
			PipelineCache::ProgramRef ps_program;
			ShaderProgram             ps_handle {};
			if (pipeline_cache.LookupPixelProgram(ps_params, state.ps_input_info, ps_handle,
			                                      ps_program)) {
				// The cross-core handoff costs a fixed ~1us (atomic store, spin, cache-line
				// transfer). Resolution cost scales with descriptor count, so handing off a
				// two-descriptor shader loses money. Only dispatch when there is enough work.
				const auto vs_resources = vs_program != nullptr
				                              ? vs_program->info.buffers.size() +
				                                    vs_program->info.images.size() +
				                                    vs_program->info.samplers.size()
				                              : 0u;
				constexpr size_t kWorthDispatching = 6;

				bool       vs_ok      = false;
				const bool dispatched = Config::ParallelResolveEnabled() &&
				                        vs_resources >= kWorthDispatching &&
				                        ResolveWorker::Instance().Dispatch([&] {
					vs_ok = pipeline_cache.ResolveVertexResources(vs_program, vs_params,
					                                              state.vs_input_info);
				});
				const bool ps_ok =
				    pipeline_cache.ResolvePixelResources(ps_program, ps_params, state.ps_input_info);
				if (dispatched) {
					ResolveWorker::Instance().Wait();
				} else {
					vs_ok = pipeline_cache.ResolveVertexResources(vs_program, vs_params,
					                                              state.vs_input_info);
				}
				if (vs_ok && ps_ok) {
					state.vertex_program = vs_handle;
					state.pixel_program  = ps_handle;
					return;
				}
			}
		}
		// Anything unresolved (needs compiling, or a permutation mismatch) falls through to the
		// serial path, which handles compilation and re-materialisation correctly.
		state.vs_input_info = {};
		state.ps_input_info = {};
	}

	state.vertex_program =
	    pipeline_cache.GetVertexProgram(vertex_shader_info, shader_regs, state.vs_input_info);

	if (!state.ps_active) {
		return;
	}
	if (log_phases) {
		LogDrawPhase(draw.name, "GetPixelProgram");
	}
	state.pixel_program =
	    pipeline_cache.GetPixelProgram(pixel_shader_info, shader_regs, state.vs_input_info,
	                                   target_export_mapping, state.ps_input_info);
}

static PreparedVertexBuffers PrepareVertexBuffers(uint64_t submit_id, CommandBuffer& buffer,
                                                  const DrawCallInfo&          draw,
                                                  const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(draw.name == nullptr);
	(void)submit_id;

	LogDrawPhase(draw.name, "PrepareVertexBuffers");
	return AcquireVertexBuffers(buffer, vs_input_info);
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	PreparedIndexBuffer prepared;
	if (!source.enabled) {
		return prepared;
	}
	EXIT_IF(source.size == 0);
	prepared.size = source.size;
	prepared.type = source.type;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
	}
	if (source.host_data != nullptr) {
		SetVulkanObjectNameF(buffer.GetContext().GetGraphics().device, prepared.buffer,
		                     "Kyty.IndexBuffer[guest=transient size=0x{:x} type={}]", source.size,
		                     static_cast<uint32_t>(source.type));
	} else {
		SetVulkanObjectNameF(buffer.GetContext().GetGraphics().device, prepared.buffer,
		                     "Kyty.IndexBuffer[guest=0x{:016x} size=0x{:x} type={}]",
		                     source.address, source.size, static_cast<uint32_t>(source.type));
	}
	return prepared;
}

static void CommitVertexBuffers(vk::CommandBuffer            vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF(prepared.buffers[i] == nullptr);
	}
	if (prepared.count != 0) {
		vk_buffer.bindVertexBuffers(0, prepared.count, prepared.buffers.data(),
		                            prepared.offsets.data());
	}
}

static void CommitIndexBuffer(vk::CommandBuffer vk_buffer, const PreparedIndexBuffer& prepared) {
	if (prepared.size == 0) {
		return;
	}
	EXIT_IF(prepared.buffer == nullptr);
	{
		auto&      dyn = DynState();
		const bool retarget_idx = dyn.Retarget();
		auto*      raw = static_cast<VkBuffer>(prepared.buffer);
		if (retarget_idx || dyn.index_buffer != raw || dyn.index_offset != prepared.offset ||
		    dyn.index_type != static_cast<uint32_t>(prepared.type)) {
			vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
			dyn.index_buffer = raw;
			dyn.index_offset = prepared.offset;
			dyn.index_type   = static_cast<uint32_t>(prepared.type);
		}
	}
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
                                 const DrawRenderState& state, bool always_log,
                                 bool force_legacy_rect_log, uint32_t index_type_and_size,
                                 const void* index_addr) {
	EXIT_IF(draw.name == nullptr);

	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!always_log && !force_legacy_rect_log) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.name, state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, draw.flags);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vs_input_info, index_type_and_size,
	                  draw.index_count, index_addr);
	// LogDrawTextureState(draw.name, state.color_info[0], state.ps_input_info);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, vk::CommandBuffer vk_buffer,
                               const ShaderVertexInputInfo& vs_input_info, const DrawCallInfo& draw,
                               const DrawEmitInfo& emit) {
	EXIT_IF(draw.name == nullptr);

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
			if (emit.indexed) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      draw.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count, emit.first_vertex,
				               draw.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kRectListLegacy:
			if (emit.indexed) {
				EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			// Sarah
			EXIT_NOT_IMPLEMENTED(!(draw.index_count == 3 && vs_input_info.buffers_num == 0));
			vk_buffer.draw(4, draw.instance_count, emit.first_vertex, draw.first_instance);
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (emit.indexed) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      draw.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count, i + emit.first_vertex,
					               draw.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         vk::PrimitiveTopology topology, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source,
                                         bool primitive_restart_enable, bool log_pipeline_phase,
                                         bool set_bind_debug, bool set_auto_debug,
                                         PreparedShaders* prepared) {
	DrawPhaseTimer exec_entry_timer(DrawPhase::ExecEntry);
	m_context.GetHdrProbe().NoteDrawShader(buffer.GetShaders().GetPs().ps_regs.data_addr, 0);
	{
		const auto& dc_probe  = buffer.GetRegisters().GetDepthControl();
		const auto& vp_probe  = buffer.GetRegisters().GetScreenViewport().viewports[0];
		const auto& z_probe   = buffer.GetRegisters().GetDepthRenderTarget();
		const auto& clip_probe = buffer.GetRegisters().GetClipControl();
		// A depth-tested draw whose viewport Z range has collapsed pins every fragment to a single
		// depth, so under GEQUAL (reversed-Z) it is rejected anywhere earlier geometry already wrote
		// depth. Log what these draws actually are: a full-screen quad is a legitimate depth reset,
		// a high index count is real geometry being wrongly discarded.
		if (dc_probe.z_enable && vp_probe.zscale < 0.001f && vp_probe.zoffset < 0.001f) {
			static std::atomic<uint32_t> degenerate_log {0};
			if (degenerate_log.fetch_add(1, std::memory_order_relaxed) < 40) {
				LOGF("DegenerateViewportZ: zscale=%.9f zoffset=%.9f xscale=%.1f index_count=%" PRIu32
				     " instances=%" PRIu32 " ps=0x%010" PRIx64 " zfunc=%u zwrite=%d zmin=%.4f zmax=%.4f dx_clip=%d zexport=%d colors=%" PRIu32 " ps_active=%d\n",
				     static_cast<double>(vp_probe.zscale), static_cast<double>(vp_probe.zoffset),
				     static_cast<double>(vp_probe.xscale), draw.index_count, draw.instance_count,
				     buffer.GetShaders().GetPs().ps_regs.data_addr,
				     static_cast<uint32_t>(dc_probe.zfunc),
				     static_cast<int>(dc_probe.z_write_enable),
				     static_cast<double>(vp_probe.zmin), static_cast<double>(vp_probe.zmax),
				     static_cast<int>(clip_probe.dx_clip_space),
				     static_cast<int>(
				         buffer.GetRegisters().GetShaderRegisters().db_shader_control
				             .shader_z_export_enable),
				     state.color_count, static_cast<int>(state.ps_active));
			}
		}
		// The horizon band is distant geometry landing on the composited image. Identify every draw
		// that runs with depth testing OFF, since those cannot be occluded by the interior and are
		// the only things that can paint over a finished scene.
		if (!dc_probe.z_enable) {
			static std::atomic<uint32_t> nodepth_log {0};
			if (nodepth_log.fetch_add(1, std::memory_order_relaxed) < 60) {
				LOGF("NoDepthDraw: ps=0x%010" PRIx64 " index_count=%" PRIu32 " prim=%" PRIu32
				     " zfunc=%u vp=%.0fx%.0f vp_y=%.0f\n",
				     buffer.GetShaders().GetPs().ps_regs.data_addr, draw.index_count,
				     static_cast<uint32_t>(buffer.GetUserConfig().GetPrimType()),
				     static_cast<uint32_t>(dc_probe.zfunc),
				     static_cast<double>(vp_probe.xscale * 2.0f),
				     static_cast<double>(vp_probe.yscale * -2.0f),
				     static_cast<double>(vp_probe.yoffset));
			}
		}
		// The collapsed-Z pass writes depth ~0, so it only survives where the buffer is still 0.
		// Record which depth buffer it binds and what clear state that buffer is in, to see whether
		// the guest expects a freshly cleared target here.
		if (vp_probe.zscale < 0.001f && vp_probe.zoffset < 0.001f && draw.index_count > 32) {
			static std::atomic<uint32_t> cz_log {0};
			if (cz_log.fetch_add(1, std::memory_order_relaxed) < 30) {
				LOGF("CollapsedZDraw: ps=0x%010" PRIx64 " idx=%" PRIu32 " zread=0x%010" PRIx64
				     " zwrite_base=0x%010" PRIx64 " z=%d/%d func=%u\n",
				     buffer.GetShaders().GetPs().ps_regs.data_addr, draw.index_count,
				     z_probe.z_read_base_addr, z_probe.z_write_base_addr,
				     static_cast<int>(dc_probe.z_enable), static_cast<int>(dc_probe.z_write_enable),
				     static_cast<uint32_t>(dc_probe.zfunc));
			}
		}
		m_context.GetHdrProbe().NoteDrawDepth(
		    dc_probe.z_enable, dc_probe.z_write_enable, dc_probe.zfunc, vp_probe.zoffset,
		    vp_probe.zscale + vp_probe.zoffset, z_probe.z_write_base_addr);
	}
	EXIT_IF(draw.name == nullptr);
	auto& ucfg = buffer.GetUserConfig();

	LogDrawPhase(draw.name, "PrepareBindings");
	exec_entry_timer.Stop();
	DrawPhaseTimer bindings_timer(DrawPhase::Bindings);
	// Phase 2 builds these on a worker when --test-parallel-bindings is on. Moved rather than
	// copied: they own six heap vectors per stage, and the prepared entry is dead once its draw is
	// recorded. A draw that bailed out arrives with bindings_valid false and rebuilds here, which
	// is byte-for-byte the path every draw took before.
	auto bindings = (prepared != nullptr && prepared->bindings_valid)
	                    ? std::move(prepared->bindings)
	                    : PrepareGraphicsBindings(state.vs_input_info.stage,
	                                              state.ps_input_info.stage, state.ps_active);
	if (prepared != nullptr) {
		prepared->bindings_valid = false;
	}
	bindings_timer.Stop();
	DrawPhaseTimer vertex_index_timer(DrawPhase::VertexIndex);
	auto vertex_bindings = PrepareVertexBuffers(submit_id, buffer, draw, state.vs_input_info);
	auto index_binding   = PrepareIndexBuffer(buffer, index_source);
	vertex_index_timer.Stop();
	DrawPhaseTimer probes_timer(DrawPhase::Probes);
	// GTA's missing world models all arrive here with the same pixel shader and reversed-Z
	// GEQUAL. Capture the target immediately before that test; AcquireRenderTargets restores the
	// attachment layout after this diagnostic copy.
	constexpr uint64_t GtaModelDepthProbePs = 0x025b20b200ull;
	if (m_context.GetHdrProbe().Enabled() && state.depth_info.image_id &&
	    buffer.GetShaders().GetPs().ps_regs.data_addr == GtaModelDepthProbePs) {
		auto& depth_image = m_context.GetTextureCache().GetImage(state.depth_info.image_id);
		m_context.GetHdrProbe().CaptureDepth(
		    depth_image, static_cast<uint32_t>(m_context.GetGpu().GetFrameNum()),
		    state.depth_info.desc.view_info.base_layer, state.depth_info.depth_buffer_vaddr);
	}
	probes_timer.Stop();
	DrawPhaseTimer render_target_timer(DrawPhase::RenderTargets);
	state.rendering =
	    AcquireRenderTargets(buffer, state.color_info, state.color_count, state.depth_info);
	render_target_timer.Stop();
	{
		const auto& dc_probe = buffer.GetRegisters().GetDepthControl();
		const auto& vp_probe = buffer.GetRegisters().GetScreenViewport().viewports[0];
		// The GTA model draws use reversed-Z GEQUAL with a near-zero viewport depth range. Record
		// the fully resolved attachment state after AcquireRenderTargets() so the next run can prove
		// whether a lingering clear, metadata clear, or wrong target is rejecting them.
		if (dc_probe.z_enable && vp_probe.zscale < 0.001f && vp_probe.zoffset < 0.001f) {
			static std::atomic<uint32_t> depth_attachment_log {0};
			if (depth_attachment_log.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("GtaDegenerateDepth: ps=0x%010" PRIx64
				     " index_count=%" PRIu32 " addr=0x%010" PRIx64
				     " size=%ux%u fmt=%u clear=%d load_clear=%d meta_clear=%d value=%.9f"
				     " test=%d write=%d compare=%u consumed=%d consumed_addr=0x%010" PRIx64 "\n",
				     buffer.GetShaders().GetPs().ps_regs.data_addr, draw.index_count,
				     state.depth_info.depth_buffer_vaddr, state.depth_info.width,
				     state.depth_info.height, static_cast<uint32_t>(state.depth_info.format),
				     static_cast<int>(state.depth_info.depth_clear_enable),
				     static_cast<int>(state.depth_info.depth_load_clear_enable),
				     static_cast<int>(state.depth_info.depth_meta_clear_enable),
				     static_cast<double>(state.depth_info.depth_clear_value),
				     static_cast<int>(state.depth_info.depth_test_enable),
				     static_cast<int>(state.depth_info.depth_write_enable),
				     static_cast<uint32_t>(state.depth_info.depth_compare_op),
				     static_cast<int>(m_depth_clear_consumed), m_depth_clear_consumed_addr);
			}
		}
	}
	if (log_pipeline_phase) {
		LogDrawPhase(draw.name, "CreatePipeline");
	}
	DrawPhaseTimer pipeline_timer(DrawPhase::Pipeline);
	auto& pipeline = m_context.GetPipelineCache().CreateGraphicsPipeline(
	    std::span {state.color_info, state.color_count}, state.depth_info, state.vs_input_info, buffer,
	    state.ps_active ? &state.ps_input_info : nullptr, topology, primitive_restart_enable,
	    state.vertex_program, state.pixel_program, draw.index_count);
	pipeline_timer.Stop();
	DrawPhaseTimer batch_open_timer(DrawPhase::BatchOpen);

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory.
	auto vk_buffer = buffer.Handle();
	if (set_bind_debug) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x100u);
	}
	if (set_auto_debug) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x200u);
	}
	// Secondary recording. A secondary inherits none of the primary's bound state - not the
	// pipeline, descriptors, vertex/index buffers or dynamic state - so everything the draw needs
	// must be recorded into it. Only the barriers stay on the primary, which is why they are
	// emitted by CommitBindings before BeginRendering rather than moved here.
	const bool        secondary = Config::SecondaryRecordEnabled();
	vk::CommandBuffer record    = vk_buffer;
	std::optional<ScopedBatchRecording> batch_recording;
	if (secondary) {
		auto& pool = m_context.GetDrawWorkerPool();
		SecondaryRenderingFormats formats {};
		formats.color_count = state.color_count;
		for (uint32_t i = 0; i < state.color_count; i++) {
			formats.color_formats[i] = state.color_info[i].format;
		}
		formats.depth_format = state.depth_info.format;
		// A stencil buffer is present iff the target allocates one; the depth target carries a
		// single combined format for both aspects.
		formats.stencil_format = state.depth_info.stencil_buffer_size != 0
		                             ? state.depth_info.format
		                             : vk::Format::eUndefined;
		formats.samples = vulkan_sample_count(state.color_count > 0 ? state.color_info[0].samples
		                                                            : state.depth_info.samples);
		// A secondary's inheritance fixes its attachments, so a change in either ends the batch.
		auto& batch = g_secondary_batch;
		if (batch.open && (!(batch.rendering == state.rendering) || !(batch.formats == formats))) {
			FlushSecondaryBatch(m_context);
		}
		if (!batch.open) {
			// Ownership scope is exactly the batch: while it is open no Buffer is destroyed and
			// LRU touches are recorded per worker instead of hitting the shared cache.
			auto& buffer_cache = m_context.GetBufferCache();
			buffer_cache.SetConcurrent(pool.WorkerCount() > 1);
			buffer_cache.BeginBatch();
			batch.worker    = CurrentDrawWorker();
			batch.buffer    = pool.BeginSecondary(batch.worker, formats);
			batch.rendering = state.rendering;
			batch.formats   = formats;
			batch.open      = true;
			// The cache used to be reset by hand here, because it keyed on the primary's recording
			// generation and that does not change when a secondary begins. BeginSecondary now
			// claims a generation of its own, so Retarget() fires on the secondary's first draw
			// and re-emits pipeline, index and every vkCmdSet* without help. Resetting here as well
			// would only hide whether that is working from RecordCensus.
		}
		batch.draws++;
		record = batch.buffer;
		batch_recording.emplace();
	}

	batch_open_timer.Stop();
	DrawPhaseTimer commit_timer(DrawPhase::Commit);
	CommitVertexBuffers(record, vertex_bindings);
	if (bindings.pixel.has_value()) {
		if (set_auto_debug) {
			SetDrawDebugPhase(buffer, submit_id, draw, 0x300u);
		}
	}
	std::array<PreparedBindings*, 2> descriptor_stages {&bindings.vertex, nullptr};
	const size_t                     descriptor_stage_count = bindings.pixel.has_value() ? 2u : 1u;
	if (bindings.pixel) {
		descriptor_stages[1] = &*bindings.pixel;
	}
	CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline,
	               std::span {descriptor_stages.data(), descriptor_stage_count}, record);
	// Bindings are fully consumed by CommitBindings; hand their heap buffers back so the next
	// draw reuses them instead of allocating ~18 fresh vectors.
	ReturnPooledBindingStorage(bindings.vertex);
	if (bindings.pixel) {
		ReturnPooledBindingStorage(*bindings.pixel);
	}
	CommitIndexBuffer(record, index_binding);
	commit_timer.Stop();

	DrawPhaseTimer dyn_state_timer(DrawPhase::DynState);
	SetGraphicsDynamicParams(buffer, record, state.color_info, state.color_count,
	                         state.depth_info,
	                         state.ps_active ? state.ps_input_info.mrt_output_mask : 0u, draw.index_count);
	dyn_state_timer.Stop();

	LogDrawPhase(draw.name, "BeginRendering");
	if (set_auto_debug) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x400u);
	}
	DrawPhaseTimer begin_rendering_timer(DrawPhase::BeginRendering);
	// Batched draws do not open the pass here; FlushSecondaryBatch opens it once for the batch.
	if (!secondary) {
		m_context.GetCommandScheduler().BeginRendering(state.rendering);
	}
	{
		// Consecutive draws frequently reuse a pipeline even when their bindings differ, so this
		// skips a driver call without changing what gets bound.
		auto&      dyn = DynState();
		const bool retarget_pipe = dyn.Retarget();
		auto*      raw = static_cast<VkPipeline>(pipeline.pipeline);
		if (retarget_pipe || dyn.pipeline != raw) {
			record.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
			dyn.pipeline = raw;
		}
	}
	if (set_auto_debug) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x500u);
	}
	if (Config::ShouldSkipPixelShaderChksum(buffer.GetShaders().GetPs().ps_regs.chksum)) {
		static std::atomic<uint64_t> s_skipped {0};
		if ((s_skipped.fetch_add(1, std::memory_order_relaxed) + 1) % 20000 == 0) {
			std::printf("SkipPsChksum: 20000 draws skipped\n");
			std::fflush(stdout);
		}
		return;
	}
	// Counted AFTER the skip: counting before made skipped draws indistinguishable from executed
	// ones, so every A/B measured scene variation instead of the skip.
	NoteShaderDraw(buffer.GetShaders().GetPs().ps_regs.chksum);
	if (Config::ShouldSkipPixelShader(buffer.GetShaders().GetPs().ps_regs.data_addr)) {
		// Diagnostic: drop every draw that uses this guest pixel shader.
		LogDrawPhase(draw.name, "DrawSkippedByShaderFilter");
	} else {
		g_draw_accepted.fetch_add(1, std::memory_order_relaxed);
		// Per-frame draw count: if geometry vanishes at some camera angles while this stays flat,
		// the draws are being submitted and failing to render. If it drops, the guest is culling.
		{
			static std::atomic<uint32_t> last_frame {UINT32_MAX};
			static std::atomic<uint32_t> frame_draws {0};
			const auto frame_now = m_context.HasGpu()
			                           ? static_cast<uint32_t>(m_context.GetGpu().GetFrameNum())
			                           : 0u;
			const auto prev = last_frame.load(std::memory_order_relaxed);
			if (prev != frame_now) {
				if (prev != UINT32_MAX) {
					LOGF("FrameDraws: frame=%" PRIu32 " draws=%" PRIu32 "\n", prev,
					     frame_draws.load(std::memory_order_relaxed));
				}
				last_frame.store(frame_now, std::memory_order_relaxed);
				frame_draws.store(0, std::memory_order_relaxed);
			}
			frame_draws.fetch_add(1, std::memory_order_relaxed);
		}
		// Minimal frame accounting: draws per frame and wall time, so a settings change (e.g.
	// Performance RT vs Performance) can be measured rather than guessed at.
	{
		static std::atomic<uint32_t> s_last_frame {UINT32_MAX};
		static std::atomic<uint64_t> s_draws {0};
		s_draws.fetch_add(1, std::memory_order_relaxed);
		const auto frame_now = m_context.HasGpu()
		                           ? static_cast<uint32_t>(m_context.GetGpu().GetFrameNum())
		                           : 0u;
		const auto prev = s_last_frame.exchange(frame_now, std::memory_order_relaxed);
		if (prev != frame_now && prev != UINT32_MAX) {
			static std::chrono::steady_clock::time_point s_t0 {};
			static uint32_t                              s_frames = 0;
			g_census_frames.fetch_add(1, std::memory_order_relaxed);
			if (s_frames % 300 == 299) {
				ReportCensus();
			}
			if (++s_frames % 60 == 0) {
				const auto now = std::chrono::steady_clock::now();
				if (s_t0.time_since_epoch().count() != 0) {
					const auto ms =
					    std::chrono::duration_cast<std::chrono::nanoseconds>(now - s_t0).count() /
					    60.0 / 1e6;
					const auto d = s_draws.exchange(0, std::memory_order_relaxed) / 60;
					std::printf("Frame: %llu draws/frame, %.1f ms (%.1f fps), %.2f us/draw\n",
					            static_cast<unsigned long long>(d), ms, ms > 0 ? 1000.0 / ms : 0.0,
					            d != 0 ? ms * 1000.0 / static_cast<double>(d) : 0.0);
					std::fflush(stdout);
				} else {
					s_draws.exchange(0, std::memory_order_relaxed);
				}
				s_t0 = now;
			}
		}
	}
	DrawCensusTick();
		begin_rendering_timer.Stop();
		DrawPhaseTimer emit_timer(DrawPhase::Emit);
		EmitDrawPrimitives(ucfg, record, state.vs_input_info, draw, emit);
	}

	// Runs to the end of the function: barrier derivation and the trailing debug phases.
	DrawPhaseTimer teardown_timer(DrawPhase::Teardown);
	if (set_auto_debug) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x600u);
	}
	vk::PipelineStageFlags shader_write_stages = {};
	if (HasShaderBufferWrites(state.vs_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eVertexShader;
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
	LogDrawPhase(draw.name, "DrawComplete");
	if (set_auto_debug) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x700u);
	}
}

void RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               uint32_t index_type_and_size, uint32_t index_count,
                               const void* index_addr, uint32_t flags, uint32_t type,
                               uint32_t instance_count, uint32_t render_target_slice_offset,
                               int32_t vertex_offset_add, uint32_t first_instance,
                               PreparedShaders* prepared) {
	KYTY_PROFILER_FUNCTION();

	DrawProfileBeginDraw();
	struct DrawProfileEndGuard {
		~DrawProfileEndGuard() { DrawProfileEndDraw(); }
	} draw_profile_end_guard;
	DrawPhaseTimer draw_total_timer(DrawPhase::Total);
	DrawPhaseTimer draw_preamble_timer(DrawPhase::Preamble);

	EXIT_IF(buffer.IsInvalid());
	{
		// Calls MasterSemaphore::Refresh -> vkGetSemaphoreCounterValue, a driver round trip,
		// plus a mutex acquire, on every draw - to poll a queue that is almost always empty
		// and a counter that only advances on submission. ~62k driver calls a second.
		DrawPhaseTimer pending_timer(DrawPhase::PendingOps);
		m_context.GetCommandScheduler().PopPendingOperations();
	}
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    index_count, flags, type, instance_count,
	                    reinterpret_cast<uint64_t>(index_addr));

	Common::LockGuard lock(m_context.GetMutex());
	if (index_count == 0 || instance_count == 0) {
		g_draw_skip_empty.fetch_add(1, std::memory_order_relaxed);
		DrawCensusTick();
		return;
	}

	if (ConsumeMetadataColorOperation(buffer)) {
		g_draw_skip_metadata.fetch_add(1, std::memory_order_relaxed);
		DrawCensusTick();
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		g_draw_skip_no_vs.fetch_add(1, std::memory_order_relaxed);
		DrawCensusTick();
		return;
	}

	if (ShouldSkipGeShader(buffer)) {
		g_draw_skip_ge.fetch_add(1, std::memory_order_relaxed);
		DrawCensusTick();
		return;
	}

	// Diagnostic: the backdrop/sky pass is identified by its collapsed viewport Z range. Dropping
	// it tests whether its presence is what triggers the guest to cull scene geometry.
	if (Config::SkipBackdropPass()) {
		const auto& vp_skip = buffer.GetRegisters().GetScreenViewport().viewports[0];
		if (vp_skip.zscale < 0.001f && vp_skip.zoffset < 0.001f) {
			return;
		}
	}

	// Identification aid for the "props render on top, through walls" bug. GTA5 draws a small group
	// of blended, depth-read-only geometry that tests the reversed-Z scene depth with a LESS-family
	// compare. Mirroring that compare was tried and made them cover the scene instead, which says
	// their Z is wrong rather than their compare. Skipping them answers the remaining question:
	// whether these draws ARE the visible panels/props.
	if (Config::SkipSceneSoftTransparent()) {
		const auto& bc_skip = buffer.GetRegisters().GetBlendControl(0);
		const auto& dc_skip = buffer.GetRegisters().GetDepthControl();
		const bool  less_family =
		    dc_skip.zfunc == static_cast<uint8_t>(vk::CompareOp::eLess) ||
		    dc_skip.zfunc == static_cast<uint8_t>(vk::CompareOp::eLessOrEqual);
		if (bc_skip.enable && dc_skip.z_enable && !dc_skip.z_write_enable && less_family) {
			static std::atomic<uint32_t> skip_log {0};
			if (skip_log.fetch_add(1, std::memory_order_relaxed) < 20) {
				LOGF("SkipSoftTransparent: idx=%" PRIu32 " zfunc=%u ps=0x%010" PRIx64 "\n",
				     index_count, static_cast<uint32_t>(dc_skip.zfunc),
				     buffer.GetShaders().GetPs().ps_regs.data_addr);
			}
			return;
		}
	}

	// The distant-scenery layer renders into a variable-height strip of a 1536x768 HDR target; its
	// composite is what appears as the horizon band. Dropping it removes the band at the cost of
	// distant detail.
	if (Config::SkipDistantLayer()) {
		const auto& rt_skip = buffer.GetRegisters().GetRenderTarget(render_target_first_bound_slot(buffer));
		if (rt_skip.attrib2.width + 1u == 1536u && rt_skip.attrib2.height + 1u == 768u) {
			return;
		}
	}

	if (graphics_debug_dump_enabled()) {
		sh_print("GraphicsRenderDrawIndex():Shader:", sh_ctx);
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t flags               = 0x%08" PRIx32 "\n"
		     "\t type                = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t rt_slice_offset     = 0x%08" PRIx32 "\n"
		     "\t vertex_offset_add   = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     index_type_and_size, index_count, reinterpret_cast<uint64_t>(index_addr), flags, type,
		     instance_count, render_target_slice_offset, static_cast<uint32_t>(vertex_offset_add),
		     first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, false, topology)) {
		return;
	}

	vk::IndexType index_type           = vk::IndexType::eUint16;
	uint64_t      index_size           = 0;
	bool          expand_index8_to_u16 = false;
	const bool primitive_restart = ResolvePrimitiveRestart(buffer, topology, index_type_and_size);

	switch (static_cast<Prospero::IndexType>(index_type_and_size)) {
		case Prospero::IndexType::kIndex16:
			index_type = vk::IndexType::eUint16;
			index_size = 2 * static_cast<uint64_t>(index_count);
			break;
		case Prospero::IndexType::kIndex32:
			index_type = vk::IndexType::eUint32;
			index_size = 4 * static_cast<uint64_t>(index_count);
			break;
		// Some games use it - need vulkan extension
		case Prospero::IndexType::kIndex8:
			index_type           = vk::IndexType::eUint16;
			index_size           = static_cast<uint64_t>(index_count);
			expand_index8_to_u16 = true;
			break;
		default: EXIT("unknown index_type_and_size: %u\n", index_type_and_size);
	}

	EXIT_NOT_IMPLEMENTED(flags != 0);
	EXIT_NOT_IMPLEMENTED(type != 1);
	const DrawCallInfo    draw {"DrawIndex",    CommandBufferDebugOp::DrawIndex,
	                            index_count,    flags,
	                            instance_count, first_instance};
	std::vector<uint16_t> expanded_indices;
	if (expand_index8_to_u16) {
		EXIT_NOT_IMPLEMENTED(index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(index_addr);
		expanded_indices.resize(index_count);
		for (uint32_t i = 0; i < index_count; i++) {
			expanded_indices[i] = primitive_restart && src[i] == 0xffu ? 0xffffu : src[i];
		}
	}

	DrawIndexBufferSource index_source {};
	index_source.enabled = true;
	index_source.address = reinterpret_cast<uint64_t>(index_addr);
	index_source.host_data =
	    expanded_indices.empty() ? nullptr : static_cast<const void*>(expanded_indices.data());
	index_source.size =
	    expanded_indices.empty() ? index_size : expanded_indices.size() * sizeof(uint16_t);
	index_source.type = index_type;

	DrawRenderState state {};
	draw_preamble_timer.Stop();
	bool render_state_ok = false;
	{
		DrawPhaseTimer timer(DrawPhase::RenderState);
		render_state_ok =
		    PrepareDrawRenderState(submit_id, buffer, draw, render_target_slice_offset, true, state);
	}
	if (!render_state_ok) {
		ResetBindings();
		return;
	}

	{
		const auto& vs_ident = buffer.GetShaders().GetVs();
		const auto& ps_ident = buffer.GetShaders().GetPs();
		DrawProfileNoteShaders(vs_ident.es_regs.data_addr, vs_ident.gs_regs.chksum,
		                       ps_ident.ps_regs.data_addr, ps_ident.ps_regs.chksum);
		DrawPhaseTimer timer(DrawPhase::RefreshShaders);
		RefreshShaders(buffer, draw, true, state, prepared);
	}

	LogDrawStateIfNeeded(buffer, draw, state, true, false, index_type_and_size, index_addr);

	const auto vertex_offset =
	    ResolveVertexOffset(ucfg.GetIndexOffset(), state.vs_input_info) + vertex_offset_add;

	DrawEmitInfo emit {};
	emit.indexed       = true;
	emit.vertex_offset = vertex_offset;

	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart, true, true, false, prepared);
	ResetBindings();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, uint32_t index_count,
                              uint32_t flags, uint32_t render_target_slice_offset,
                              uint32_t instance_count, uint32_t first_vertex,
                              uint32_t first_instance) {
	KYTY_PROFILER_FUNCTION();

	DrawProfileBeginDraw();
	struct DrawProfileEndGuard {
		~DrawProfileEndGuard() { DrawProfileEndDraw(); }
	} draw_profile_end_guard;
	DrawPhaseTimer draw_total_timer(DrawPhase::Total);

	EXIT_IF(buffer.IsInvalid());
	{
		// Calls MasterSemaphore::Refresh -> vkGetSemaphoreCounterValue, a driver round trip,
		// plus a mutex acquire, on every draw - to poll a queue that is almost always empty
		// and a counter that only advances on submission. ~62k driver calls a second.
		DrawPhaseTimer pending_timer(DrawPhase::PendingOps);
		m_context.GetCommandScheduler().PopPendingOperations();
	}
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    index_count, flags, first_vertex, instance_count, first_instance);

	Common::LockGuard lock(m_context.GetMutex());
	if (index_count == 0 || instance_count == 0) {
		return;
	}

	if (ConsumeMetadataColorOperation(buffer)) {
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}

	if (ShouldSkipGeShader(buffer)) {
		return;
	}

	if (graphics_debug_dump_enabled()) {
		sh_print("GraphicsRenderDrawIndexAuto():Shader:", sh_ctx);
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t flags               = 0x%08" PRIx32 "\n"
		     "\t rt_slice_offset     = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     index_count, flags, render_target_slice_offset, instance_count, first_vertex,
		     first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	EXIT_NOT_IMPLEMENTED(flags != 0);
	const DrawCallInfo draw {"DrawIndexAuto", CommandBufferDebugOp::DrawIndexAuto,
	                         index_count,     flags,
	                         instance_count,  first_instance};

	DrawRenderState state {};
	if (!PrepareDrawRenderState(submit_id, buffer, draw, render_target_slice_offset, false,
	                            state)) {
		ResetBindings();
		return;
	}

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, true, topology)) {
		ResetBindings();
		return;
	}
	RefreshShaders(buffer, draw, false, state);

	const bool rect_list = topology == vk::PrimitiveTopology::ePatchList;
	if (rect_list && state.vs_input_info.buffers_num == 0 &&
	    state.vs_input_info.param_export_mask == 0 && state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.chksum,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, false,
	                     ucfg.GetPrimType() == Prospero::PrimitiveType::kRectListLegacy, 0,
	                     nullptr);

	const auto   vertex_offset = ResolveVertexOffset(ucfg.GetIndexOffset(), state.vs_input_info) +
	                             static_cast<int32_t>(first_vertex);
	DrawEmitInfo emit {};
	emit.first_vertex = static_cast<uint32_t>(vertex_offset);

	DrawIndexBufferSource index_source {};
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source, false, false,
	                    false, true);
	ResetBindings();
}

bool RenderExecutor::ResolveColorTargets(uint64_t submit_id, CommandBuffer& buffer,
                                         uint32_t render_target_slice_offset) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(submit_id, buffer, src, render_target_slice_offset, 0, true, true);
	ResolveRenderColorTarget(submit_id, buffer, dst, render_target_slice_offset, 1, true, true);
	if (!src.image_id || !dst.image_id || src.type == RenderColorType::NoColorOutput ||
	    dst.type == RenderColorType::NoColorOutput) {
		return false;
	}
	if (src.base_addr == dst.base_addr && src.base_mip_level == dst.base_mip_level &&
	    src.base_array_layer == dst.base_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source, {src.base_mip_level, 1, src.base_array_layer, 1},
	                    {dst.base_mip_level, 1, dst.base_array_layer, 1});
	return true;
}

void RenderExecutor::FlushPendingTransitions(CommandBuffer& buffer) {
	if (m_pending_transitions.empty()) {
		return;
	}
	// Applied in request order, on the primary, before the batch opens its render pass - which is
	// where these barriers were already landing. Staging only changes who records them, not when
	// the GPU sees them relative to the draws that need them.
	auto& texture_cache = m_context.GetTextureCache();
	auto  vk_buffer     = buffer.Handle();
	for (const auto& transition: m_pending_transitions) {
		auto* image = texture_cache.m_slot_images.try_get(transition.image_id);
		if (image == nullptr) {
			continue;   // retired between staging and flush; nothing to transition
		}
		image->Transit(transition.layout, transition.access, transition.range, vk_buffer);
	}
	m_pending_transitions.clear();
}

bool RenderExecutor::PrepareQueuedShaders(CommandBuffer& buffer, PreparedShaders& prepared) {
	auto&       pipeline_cache = buffer.GetContext().GetPipelineCache();
	const auto& vertex_info    = buffer.GetShaders().GetVs();
	const auto& pixel_info     = buffer.GetShaders().GetPs();
	const auto& shader_regs    = buffer.GetRegisters().GetShaderRegisters();

	prepared.valid     = false;
	prepared.ps_active = DrawHasActivePixelShader(buffer);
	if (!DrawHasValidVertexShader(buffer.GetShaders())) {
		return false;
	}

	// Colour export mapping feeds the pixel permutation key, so it has to be rebuilt here rather
	// than read from a later stage.
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> target_export_mapping {};

	prepared.vs_params = pipeline_cache.PrepareVertexParams(vertex_info, shader_regs,
	                                                        prepared.vs_input_info);
	PipelineCache::ProgramRef vs_program;
	if (!pipeline_cache.LookupVertexProgram(prepared.vs_params, prepared.vs_input_info,
	                                        prepared.vertex_program, vs_program)) {
		return false;   // needs compiling; the serial path handles that
	}
	if (!prepared.ps_active) {
		prepared.vs_program_ref = vs_program;
		prepared.valid          = true;
		return true;
	}
	prepared.ps_params =
	    pipeline_cache.PreparePixelParams(pixel_info, shader_regs, prepared.vs_input_info,
	                                      target_export_mapping, prepared.ps_input_info);
	PipelineCache::ProgramRef ps_program;
	if (!pipeline_cache.LookupPixelProgram(prepared.ps_params, prepared.ps_input_info,
	                                       prepared.pixel_program, ps_program)) {
		return false;
	}
	prepared.vs_program_ref = vs_program;
	prepared.ps_program_ref = ps_program;
	prepared.valid          = true;
	return true;
}

void RenderExecutor::ResolveQueuedShaders(PreparedShaders& prepared) {
	if (!prepared.valid) {
		return;
	}
	// The parallel half. Deliberately unlocked - see PipelineCache::ResolveVertexResources - and
	// it writes only into prepared, so eight of these can run at once.
	auto& pipeline_cache = m_context.GetPipelineCache();
	if (!pipeline_cache.ResolveVertexResources(prepared.vs_program_ref, prepared.vs_params,
	                                           prepared.vs_input_info)) {
		prepared.valid = false;
		return;
	}
	if (prepared.ps_active &&
	    !pipeline_cache.ResolvePixelResources(prepared.ps_program_ref, prepared.ps_params,
	                                          prepared.ps_input_info)) {
		prepared.valid = false;
	}
}

void RenderExecutor::AcquireQueuedBindings(PreparedShaders& prepared) {
	// Phase 2b. Serial, main thread. Creates every buffer and image the draw needs so that the
	// parallel pass that follows can only ever hit.
	prepared.bindings_valid = false;
	if (!prepared.valid) {
		return;
	}
	prepared.bindings =
	    AcquireGraphicsBindings(prepared.vs_input_info.stage, prepared.ps_input_info.stage,
	                            prepared.ps_active);
	prepared.bindings_valid = true;
}

void RenderExecutor::BindQueuedResources(PreparedShaders& prepared) {
	// Phase 2c. Parallel. Acquisition guaranteed every lookup below resolves to something that
	// already exists, so nothing here can allocate.
	if (!prepared.valid || !prepared.bindings_valid) {
		return;
	}
	BindGraphicsResources(prepared.bindings);
}

bool RenderExecutor::EnqueueDrawIndex(QueuedDraw&& draw) {
	m_draw_queue.Push(std::move(draw));
	return m_draw_queue.Full();
}

void RenderExecutor::DrainDrawQueue(CommandBuffer& buffer) {
	// A queued draw can itself hit a batch boundary and flush, and the flush drains the queue.
	// Without this guard that re-enters while the drain is already walking the same draws.
	if (m_draining) {
		return;
	}
	m_draining = true;
	m_draw_queue.Drain(*this, buffer);
	m_draining = false;
}

bool SecondaryBatchOpen() noexcept {
	return g_secondary_batch.open;
}

void FlushSecondaryBatch(RenderContext& context) {
	// The draw queue is deliberately NOT drained here.
	//
	// Draining translates draws, and translation re-enters buffer and texture resolution. This
	// function runs from CommandBuffer::EndRendering, which resolution itself calls while holding
	// the memory tracker's region lock - so draining here deadlocks on that lock, which is the
	// "recursive region tracking lock" abort. The queue is drained from the command processor
	// instead, where no cache lock is held.

	auto& batch = g_secondary_batch;
	// flushing guards re-entry: the EndRendering below is itself a flush hook.
	if (!batch.open || batch.flushing) {
		return;
	}
	// Resolve calls EndRendering to record uploads and barriers, and that now routes here. It does
	// not need the batch replayed: no render pass is open while a batch is being built - one is
	// only opened at flush - so the transfer is legal as it stands, and it lands on the primary
	// ahead of the batch that replays after it, which is the order the draw needs anyway.
	if (batch.recording) {
		return;
	}
	batch.flushing = true;
	auto& pool = context.GetDrawWorkerPool();
	pool.EndSecondary(batch.worker);

	// Staged work is recorded here, before the render pass opens and therefore before any draw in
	// this batch executes. Resolve only stages it; this is the serial phase that applies it, in
	// the order it was requested.
	// Order matters: clears transition images to TransferDst and write them, so they run before
	// the transitions that put those images into their sampled layout for the draws.
	context.GetTextureCache().FlushPendingClears();
	context.GetTextureCache().FlushDeferredTouches();
	context.GetTextureCache().FlushDeferredTracks();
	context.GetRenderExecutor().FlushPendingTransitions(context.GetCommandScheduler().Current());
	context.GetBufferCache().FlushPendingUploads();

	auto& scheduler = context.GetCommandScheduler();
	// The render pass opens only now, so every barrier the batch's draws needed was emitted on the
	// primary while no pass was active - which is where Vulkan requires them.
	scheduler.BeginRendering(ResumeStateFor(batch.rendering), true);
	g_last_flushed_state = batch.rendering;
	g_have_last_flushed  = true;
	auto primary = scheduler.Current().Handle();
	primary.executeCommands(1, &batch.buffer);
	scheduler.EndRendering();

	// Applies the batch's retired buffers and merges the workers' LRU touches, on this thread.
	context.GetBufferCache().EndBatch();

	batch.open     = false;
	batch.draws    = 0;
	batch.buffer   = nullptr;
	batch.flushing = false;
}

} // namespace Libs::Graphics
