#include "common/emulatorConfig.h"

#include "common/assert.h"

#include <algorithm>
#include <memory>

namespace Config {

static std::unique_ptr<ConfigOptions> g_config;

void Initialize() {
	EXIT_IF(g_config != nullptr);

	g_config = std::make_unique<ConfigOptions>();
}

void Shutdown() {
	g_config.reset();
}

void Load(const ConfigOptions& cfg) {
	EXIT_IF(g_config == nullptr);
	EXIT_IF(cfg.user_name.empty() || cfg.user_name.size() > MAX_USER_NAME_LENGTH);
	EXIT_IF(!IsConfiguredUserIdValid(cfg.user_id));

	*g_config = cfg;
}

uint32_t GetScreenWidth() {
	return g_config->screen_width;
}

uint32_t GetScreenHeight() {
	return g_config->screen_height;
}

const std::string& GetUserName() {
	return g_config->user_name;
}

int32_t GetUserId() {
	return g_config->user_id;
}

PresentMode GetPresentMode() {
	return g_config->present_mode;
}

bool FullscreenEnabled() {
	return g_config->fullscreen_enabled;
}

uint32_t GetVblankFrequency() {
	return std::clamp(g_config->vblank_frequency, 30u, 360u);
}

uint32_t GetConsoleLanguage() {
	return g_config->console_language;
}

bool VulkanValidationEnabled() {
	return g_config->vulkan_validation_enabled;
}

bool ShaderValidationEnabled() {
	return g_config->shader_validation_enabled;
}

ShaderOptimizationType GetShaderOptimizationType() {
	return g_config->shader_optimization_type;
}

ShaderLogDirection GetShaderLogDirection() {
	return g_config->shader_log_direction;
}

std::filesystem::path GetShaderLogFolder() {
	return g_config->shader_log_folder;
}

bool CommandBufferDumpEnabled() {
	return g_config->command_buffer_dump_enabled;
}

std::filesystem::path GetCommandBufferDumpFolder() {
	return g_config->command_buffer_dump_folder;
}

bool GraphicsDebugDumpEnabled() {
	return g_config->graphics_debug_dump_enabled;
}

OutputDirection GetPrintfDirection() {
	return g_config->printf_direction;
}

std::filesystem::path GetPrintfOutputFile() {
	return g_config->printf_output_file;
}

ProfilerDirection GetProfilerDirection() {
	return g_config->profiler_direction;
}

bool SpirvDebugPrintfEnabled() {
	return g_config->spirv_debug_printf_enabled;
}

bool GpuAssistedValidationEnabled() {
	return g_config->gpu_assisted_validation_enabled && g_config->vulkan_validation_enabled;
}

bool RenderDocEnabled() {
	return g_config->renderdoc_enabled;
}

bool ReadbackLinearImagesEnabled() {
	return g_config->readback_linear_images;
}

bool HdrProbeEnabled() {
	return g_config->hdr_probe_enabled;
}

uint32_t HdrProbeInterval() {
	return g_config->hdr_probe_interval;
}

bool ShouldSkipPixelShader(uint64_t ps_addr) {
	if (g_config->skip_ps.empty() || ps_addr == 0) {
		return false;
	}
	return std::find(g_config->skip_ps.begin(), g_config->skip_ps.end(), ps_addr) !=
	       g_config->skip_ps.end();
}

bool SkipDistantLayer() {
	return g_config->skip_distant_layer;
}

bool SkipBackdropPass() {
	return g_config->skip_backdrop_pass;
}

bool RealOcclusionQueries() {
	return g_config->real_occlusion_queries;
}

bool FixCollapsedDepthCompare() {
	return g_config->fix_collapsed_depth_compare;
}

bool FixDegenerateViewportZ() {
	return g_config->fix_degenerate_viewport_z;
}

bool SuppressBandPass() {
	return g_config->suppress_band_pass;
}

bool DepthClearPerFrame() {
	return g_config->depth_clear_per_frame;
}

bool DepthClearOnce() {
	return g_config->depth_clear_once;
}

bool FixInvertedDepthCompare() {
	return g_config->fix_inverted_depth_compare;
}

bool SkipSceneSoftTransparent() {
	return g_config->skip_scene_soft_transparent;
}

bool WaitImageReadback() {
	return g_config->wait_image_readback;
}

bool RetryTransientMapFaults() {
	return g_config->retry_transient_map_faults;
}

bool ForceDepthAlways() {
	return g_config->force_depth_always;
}

bool MaskUnwrittenMrt() {
	return g_config->mask_unwritten_mrt;
}

uint32_t HdrProbeStart() {
	return g_config->hdr_probe_start;
}

float HdrExportClamp() {
	return g_config->hdr_export_clamp;
}

bool PlayGoHackEnabled() {
	return g_config->playgo_hack_enabled;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool RedZoneProtectionEnabled() {
	return g_config->red_zone_protection_enabled;
}
#endif

const Keymap& GetKeymap() {
	return g_config->keymap;
}

} // namespace Config
