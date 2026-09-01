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
	vk_buffer.setViewport(0, 1, &viewport);

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
	vk_buffer.setScissor(0, 1, &scissor);

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
	vk_buffer.setLineWidth(line_width);

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	vk_buffer.setDepthBiasEnable(depth_bias_enable ? VK_TRUE : VK_FALSE);
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor =
		    ConvertPolygonOffsetConstantFactor(guest_constant_factor, poly_offset, depth.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		vk_buffer.setDepthBias(constant_factor, poly_offset.clamp, slope_factor);
	}

	if (depth.stencil_test_enable) {
		vk_buffer.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
		                                depth.stencil_dynamic_front.compareMask);
		vk_buffer.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
		                                depth.stencil_dynamic_back.compareMask);
		vk_buffer.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
		                              depth.stencil_dynamic_front.writeMask);
		vk_buffer.setStencilWriteMask(vk::StencilFaceFlagBits::eBack,
		                              depth.stencil_dynamic_back.writeMask);
		vk_buffer.setStencilReference(vk::StencilFaceFlagBits::eFront,
		                              depth.stencil_dynamic_front.reference);
		vk_buffer.setStencilReference(vk::StencilFaceFlagBits::eBack,
		                              depth.stencil_dynamic_back.reference);
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
static void DrawCensusTick() {
	const auto total = g_draw_census_total.fetch_add(1, std::memory_order_relaxed) + 1;
	if (total % 20000 != 0) {
		return;
	}
	LOGF("DrawCensus: accepted=%" PRIu64 " skip_empty=%" PRIu64 " skip_metadata=%" PRIu64
	     " skip_no_vs=%" PRIu64 " skip_ge=%" PRIu64 "\n",
	     g_draw_accepted.load(std::memory_order_relaxed),
	     g_draw_skip_empty.load(std::memory_order_relaxed),
	     g_draw_skip_metadata.load(std::memory_order_relaxed),
	     g_draw_skip_no_vs.load(std::memory_order_relaxed),
	     g_draw_skip_ge.load(std::memory_order_relaxed));
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
		image.Transit(layout,
		              vk::AccessFlagBits2::eColorAttachmentRead |
		                  vk::AccessFlagBits2::eColorAttachmentWrite,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
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
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
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

static void RefreshShaders(CommandBuffer& buffer, const DrawCallInfo& draw, bool log_phases,
                           DrawRenderState& state) {
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
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
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
                                         bool set_bind_debug, bool set_auto_debug) {
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
	auto bindings = PrepareGraphicsBindings(state.vs_input_info.stage, state.ps_input_info.stage,
	                                        state.ps_active);
	auto vertex_bindings = PrepareVertexBuffers(submit_id, buffer, draw, state.vs_input_info);
	auto index_binding   = PrepareIndexBuffer(buffer, index_source);
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
	state.rendering =
	    AcquireRenderTargets(buffer, state.color_info, state.color_count, state.depth_info);
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
	auto& pipeline = m_context.GetPipelineCache().CreateGraphicsPipeline(
	    std::span {state.color_info, state.color_count}, state.depth_info, state.vs_input_info, buffer,
	    state.ps_active ? &state.ps_input_info : nullptr, topology, primitive_restart_enable,
	    state.vertex_program, state.pixel_program, draw.index_count);

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
	CommitVertexBuffers(vk_buffer, vertex_bindings);
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
	               std::span {descriptor_stages.data(), descriptor_stage_count});
	CommitIndexBuffer(vk_buffer, index_binding);

	SetGraphicsDynamicParams(buffer, vk_buffer, state.color_info, state.color_count,
	                         state.depth_info,
	                         state.ps_active ? state.ps_input_info.mrt_output_mask : 0u, draw.index_count);

	LogDrawPhase(draw.name, "BeginRendering");
	if (set_auto_debug) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x400u);
	}
	m_context.GetCommandScheduler().BeginRendering(state.rendering);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	if (set_auto_debug) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x500u);
	}
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
		DrawCensusTick();
		EmitDrawPrimitives(ucfg, vk_buffer, state.vs_input_info, draw, emit);
	}

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
                               int32_t vertex_offset_add, uint32_t first_instance) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopPendingOperations();
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
	if (!PrepareDrawRenderState(submit_id, buffer, draw, render_target_slice_offset, true, state)) {
		ResetBindings();
		return;
	}

	RefreshShaders(buffer, draw, true, state);

	LogDrawStateIfNeeded(buffer, draw, state, true, false, index_type_and_size, index_addr);

	const auto vertex_offset =
	    ResolveVertexOffset(ucfg.GetIndexOffset(), state.vs_input_info) + vertex_offset_add;

	DrawEmitInfo emit {};
	emit.indexed       = true;
	emit.vertex_offset = vertex_offset;

	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart, true, true, false);
	ResetBindings();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, uint32_t index_count,
                              uint32_t flags, uint32_t render_target_slice_offset,
                              uint32_t instance_count, uint32_t first_vertex,
                              uint32_t first_instance) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopPendingOperations();
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

} // namespace Libs::Graphics
