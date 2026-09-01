#ifndef KYTY_COMMON_EMULATOR_CONFIG_H_
#define KYTY_COMMON_EMULATOR_CONFIG_H_

#include "common/common.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace Config {

void Initialize();
void Shutdown();

struct Lifecycle {
	static constexpr const char* name       = "Config";
	static constexpr auto        initialize = Config::Initialize;
	static constexpr auto        shutdown   = Config::Shutdown;
};

enum class ShaderOptimizationType { None, Size, Performance };

enum class ShaderLogDirection { Silent, Console, File };

enum class ProfilerDirection { None, Network };

enum class OutputDirection { Silent, Console, File };

enum class PresentMode { Fifo, Mailbox, Immediate };

using Keymap = std::vector<std::string>;

constexpr uint32_t DEFAULT_CONSOLE_LANGUAGE = 1;
constexpr uint32_t MAX_CONSOLE_LANGUAGE     = 29;
constexpr std::size_t MAX_USER_NAME_LENGTH = 16;
constexpr int32_t DEFAULT_USER_ID           = 1000;

constexpr bool IsConfiguredUserIdValid(int32_t user_id) {
	constexpr int32_t USER_ID_EVERYONE = 0xfe;
	constexpr int32_t USER_ID_SYSTEM   = 0xff;
	return user_id >= 0 && user_id != USER_ID_EVERYONE && user_id != USER_ID_SYSTEM;
}

struct ConfigOptions {
	uint32_t               screen_width                = 1280;
	uint32_t               screen_height               = 720;
	std::string            user_name                   = "Kyty";
	int32_t                user_id                     = DEFAULT_USER_ID;
	PresentMode            present_mode                = PresentMode::Fifo;
	bool                   fullscreen_enabled          = false;
	uint32_t               vblank_frequency            = 60;
	uint32_t               console_language            = DEFAULT_CONSOLE_LANGUAGE;
	bool                   vulkan_validation_enabled   = false;
	bool                   shader_validation_enabled   = false;
	ShaderOptimizationType shader_optimization_type    = ShaderOptimizationType::None;
	ShaderLogDirection     shader_log_direction        = ShaderLogDirection::Silent;
	std::filesystem::path  shader_log_folder           = "_Shaders";
	bool                   command_buffer_dump_enabled = false;
	std::filesystem::path  command_buffer_dump_folder  = "_Buffers";
	bool                   graphics_debug_dump_enabled = false;
	OutputDirection        printf_direction            = OutputDirection::Silent;
	std::filesystem::path  printf_output_file          = "_kyty.txt";
	ProfilerDirection      profiler_direction          = ProfilerDirection::None;
	bool                   spirv_debug_printf_enabled  = false;
	bool                   gpu_assisted_validation_enabled = false;
	bool                   renderdoc_enabled           = false;
	bool                   readback_linear_images      = false;
	bool                   hdr_probe_enabled           = false;
	uint32_t               hdr_probe_interval          = 60;
	// 0 disables. ALWAYS launch with --hdr-clamp 1; defaulting it here breaks the graphics
	// export tests, which assert exact unclamped values.
	float                  hdr_export_clamp            = 0.0f;
	uint32_t               hdr_probe_start             = 0;
	bool                   mask_unwritten_mrt          = false;
	bool                   force_depth_always          = false;
	bool                   depth_clear_once            = true;
	// Mirror a LESS-family depth compare on a read-only draw when the buffer's depth writers have
	// established it as reversed-Z. Off by default: it changes visibility of real geometry.
	bool                   fix_inverted_depth_compare  = false;
	// Diagnostic: drop blended depth-read-only draws using a LESS-family compare, to identify
	// whether those draws are the props that render on top.
	bool                   skip_scene_soft_transparent = false;
	// Wait for a GPU image download to land in guest memory before returning, so the CPU cannot
	// read stale contents of a GPU-produced image (a NaN source for the HDR runaway).
	bool                   wait_image_readback         = true;
	// Retry a fault on memory our own map says is committed, instead of aborting. Covers the
	// non-atomic window while guest direct memory is being remapped.
	bool                   retry_transient_map_faults  = true;
	// Multiplier on adaptive-trigger force. 1.0 passes the title's request through unchanged;
	// higher makes a subtle effect easier to feel.
	float                  trigger_strength            = 1.0f;
	// Scales where an adaptive-trigger effect starts. 1.0 is the title's request; 0 makes every
	// effect begin at the top of the pull, which reads as constant pressure.
	float                  trigger_position_scale      = 1.0f;
	// Which trigger a title's first effect slot refers to. Sony's ordering is not documented here,
	// so this makes it testable instead of assumed.
	bool                   trigger_swap                = false;
	bool                   depth_clear_per_frame       = true;
	bool                   suppress_band_pass          = false;
	bool                   fix_degenerate_viewport_z   = false;
	bool                   fix_collapsed_depth_compare = false;
	bool                   real_occlusion_queries      = false;
	bool                   skip_backdrop_pass          = false;
	bool                   skip_distant_layer          = false;
	std::vector<uint64_t>  skip_ps;
	bool                   playgo_hack_enabled         = false;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	bool red_zone_protection_enabled = false;
#endif
	Keymap keymap;
};

void Load(const ConfigOptions& cfg);

uint32_t GetScreenWidth();
uint32_t GetScreenHeight();
const std::string& GetUserName();
int32_t  GetUserId();
PresentMode GetPresentMode();
bool     FullscreenEnabled();
uint32_t GetVblankFrequency();
uint32_t GetConsoleLanguage();
bool     VulkanValidationEnabled();

bool                   ShaderValidationEnabled();
ShaderOptimizationType GetShaderOptimizationType();
ShaderLogDirection     GetShaderLogDirection();
std::filesystem::path  GetShaderLogFolder();

bool                  CommandBufferDumpEnabled();
std::filesystem::path GetCommandBufferDumpFolder();

bool GraphicsDebugDumpEnabled();

OutputDirection       GetPrintfDirection();
std::filesystem::path GetPrintfOutputFile();

ProfilerDirection GetProfilerDirection();

bool SpirvDebugPrintfEnabled();

bool GpuAssistedValidationEnabled();

bool RenderDocEnabled();
bool ReadbackLinearImagesEnabled();
bool HdrProbeEnabled();
uint32_t HdrProbeInterval();
float HdrExportClamp();
uint32_t HdrProbeStart();
bool MaskUnwrittenMrt();
bool ForceDepthAlways();
bool DepthClearOnce();
bool FixInvertedDepthCompare();
bool SkipSceneSoftTransparent();
bool WaitImageReadback();
bool RetryTransientMapFaults();
float TriggerStrength();
float TriggerPositionScale();
bool  TriggerSwap();
bool DepthClearPerFrame();
bool SuppressBandPass();
bool FixDegenerateViewportZ();
bool FixCollapsedDepthCompare();
bool RealOcclusionQueries();
bool SkipBackdropPass();
bool SkipDistantLayer();
bool ShouldSkipPixelShader(uint64_t ps_addr);
bool PlayGoHackEnabled();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool RedZoneProtectionEnabled();
#endif

const Keymap& GetKeymap();

} // namespace Config

#endif /* KYTY_COMMON_EMULATOR_CONFIG_H_ */
