#include "common/emulatorConfig.h"

#include "common/assert.h"

#include <algorithm>
#include <atomic>
#include <memory>

namespace Config {

static std::unique_ptr<ConfigOptions> g_config;

// Toggled from the window thread and read on the audio thread, so it lives outside the config
// struct: everything else in there is written once at startup and never changes.
static std::atomic<bool> g_pad_speaker_muted {false};

// Per-draw redundancy filters. Toggled from the window thread (F6) and read on the render
// thread every draw, so they are atomics rather than plain config fields.
static std::atomic<bool> g_dyn_state_cache {true};
// Per-draw register validation and the descriptor cache are toggled live by the F7 harness,
// so they are atomics rather than plain config fields.
static std::atomic<bool> g_hw_check {true};
static std::atomic<bool> g_draw_profile {false};
static std::atomic<bool> g_secondary_record {false};
static std::atomic<uint32_t> g_draw_workers {1};
static std::atomic<bool> g_draw_queue {false};
static std::atomic<bool> g_defer_uploads {false};
static std::atomic<bool> g_defer_transitions {false};
static std::atomic<bool> g_parallel_resolution {false};
static std::atomic<bool> g_test_parallel_bindings {false};
static std::atomic<bool>     g_frame_pipelining {false};
static std::atomic<uint32_t> g_pipeline_depth {4};
static std::atomic<bool>     g_log_ui_draws {false};
static std::atomic<bool>     g_coalesce_eop_flush {false};
static std::atomic<bool> g_cache_descriptors {false};
static std::atomic<bool> g_pipeline_memo {true};
static std::atomic<bool> g_buffer_dedup {true};

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
	g_pad_speaker_muted.store(cfg.pad_speaker_muted, std::memory_order_relaxed);
	g_dyn_state_cache.store(cfg.dyn_state_cache, std::memory_order_relaxed);
	g_hw_check.store(cfg.hw_check, std::memory_order_relaxed);
	g_draw_profile.store(cfg.draw_profile, std::memory_order_relaxed);
	g_secondary_record.store(cfg.secondary_record, std::memory_order_relaxed);
	g_draw_workers.store(cfg.draw_workers, std::memory_order_relaxed);
	g_draw_queue.store(cfg.draw_queue, std::memory_order_relaxed);
	g_defer_uploads.store(cfg.defer_uploads, std::memory_order_relaxed);
	g_defer_transitions.store(cfg.defer_transitions, std::memory_order_relaxed);
	g_parallel_resolution.store(cfg.parallel_resolution, std::memory_order_relaxed);
	g_test_parallel_bindings.store(cfg.test_parallel_bindings, std::memory_order_relaxed);
	g_frame_pipelining.store(cfg.frame_pipelining, std::memory_order_relaxed);
	g_pipeline_depth.store(cfg.pipeline_depth, std::memory_order_relaxed);
	g_log_ui_draws.store(cfg.log_ui_draws, std::memory_order_relaxed);
	g_coalesce_eop_flush.store(cfg.coalesce_eop_flush, std::memory_order_relaxed);
	g_cache_descriptors.store(cfg.cache_descriptors, std::memory_order_relaxed);
	g_pipeline_memo.store(cfg.pipeline_memo, std::memory_order_relaxed);
	g_buffer_dedup.store(cfg.buffer_dedup, std::memory_order_relaxed);
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

static std::atomic<bool>   g_show_fps_overlay {true};
static std::atomic<double> g_current_fps {0.0};
static std::atomic<bool>   g_real_occlusion {false};
static std::atomic<double> g_fps_sum {0.0};
static std::atomic<uint64_t> g_fps_samples {0};

void ResetFpsAverage() {
	g_fps_sum.store(0.0, std::memory_order_relaxed);
	g_fps_samples.store(0, std::memory_order_relaxed);
}

bool ShowFpsOverlay() {
	return g_show_fps_overlay.load(std::memory_order_relaxed);
}

void SetShowFpsOverlay(bool show) {
	g_show_fps_overlay.store(show, std::memory_order_relaxed);
}

double CurrentFps() {
	return g_current_fps.load(std::memory_order_relaxed);
}

uint64_t FpsSampleCount() {
	return g_fps_samples.load(std::memory_order_relaxed);
}

double AverageFps() {
	const auto n = g_fps_samples.load(std::memory_order_relaxed);
	if (n == 0) {
		return 0.0;
	}
	return g_fps_sum.load(std::memory_order_relaxed) / static_cast<double>(n);
}

void SetCurrentFps(double fps) {
	g_current_fps.store(fps, std::memory_order_relaxed);
	// Running mean over the whole session. The first sample lands while the title is still
	// loading, so skip anything implausible rather than dragging the average down forever.
	if (fps > 0.0) {
		g_fps_sum.store(g_fps_sum.load(std::memory_order_relaxed) + fps, std::memory_order_relaxed);
		g_fps_samples.fetch_add(1, std::memory_order_relaxed);
	}
}

bool CacheDescriptors() {
	return g_cache_descriptors.load(std::memory_order_relaxed);
}

void SetCacheDescriptors(bool enabled) {
	g_cache_descriptors.store(enabled, std::memory_order_relaxed);
}

bool HwCheckEnabled() {
	return g_hw_check.load(std::memory_order_relaxed);
}

bool DrawProfileEnabled() {
	return g_draw_profile.load(std::memory_order_relaxed);
}

bool SecondaryRecordEnabled() {
	return g_secondary_record.load(std::memory_order_relaxed);
}

uint32_t DrawWorkerCount() {
	return g_draw_workers.load(std::memory_order_relaxed);
}

bool DrawQueueEnabled() {
	return g_draw_queue.load(std::memory_order_relaxed);
}

bool DeferUploadsEnabled() {
	return g_defer_uploads.load(std::memory_order_relaxed);
}

bool DeferTransitionsEnabled() {
	return g_defer_transitions.load(std::memory_order_relaxed);
}

bool ParallelResolutionEnabled() {
	return g_parallel_resolution.load(std::memory_order_relaxed);
}

bool TestParallelBindingsEnabled() {
	return g_test_parallel_bindings.load(std::memory_order_relaxed);
}

bool FramePipeliningEnabled() {
	return g_frame_pipelining.load(std::memory_order_relaxed);
}

uint32_t PipelineDepth() {
	return g_pipeline_depth.load(std::memory_order_relaxed);
}

bool LogUiDrawsEnabled() {
	return g_log_ui_draws.load(std::memory_order_relaxed);
}

bool CoalesceEopFlushEnabled() {
	return g_coalesce_eop_flush.load(std::memory_order_relaxed);
}

void SetHwCheck(bool enabled) {
	g_hw_check.store(enabled, std::memory_order_relaxed);
}

bool ParallelResolveEnabled() {
	return g_config->parallel_resolve;
}

bool DynStateCacheEnabled() {
	return g_dyn_state_cache.load(std::memory_order_relaxed);
}

bool PipelineMemoEnabled() {
	return g_pipeline_memo.load(std::memory_order_relaxed);
}

bool BufferDedupEnabled() {
	return g_buffer_dedup.load(std::memory_order_relaxed);
}

void SetPerDrawFilters(bool dyn_state, bool pipeline, bool dedup) {
	g_dyn_state_cache.store(dyn_state, std::memory_order_relaxed);
	g_pipeline_memo.store(pipeline, std::memory_order_relaxed);
	g_buffer_dedup.store(dedup, std::memory_order_relaxed);
}

bool DccClearOnSample() {
	return g_config->dcc_clear_on_sample;
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

bool ShouldSkipPixelShaderChksum(uint64_t chksum) {
	const auto& list = g_config->skip_ps_chksum;
	if (list.empty() || chksum == 0) {
		return false;
	}
	return std::find(list.begin(), list.end(), chksum) != list.end();
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

float TriggerStrength() {
	return g_config->trigger_strength;
}

float TriggerPositionScale() {
	return g_config->trigger_position_scale;
}

bool TriggerSwap() {
	return g_config->trigger_swap;
}

bool PadSpeakerBluetooth() {
	return g_config->pad_speaker_bluetooth;
}

uint32_t PadSpeakerVolume() {
	return g_config->pad_speaker_volume;
}

uint32_t PadSpeakerRate() {
	return g_config->pad_speaker_rate;
}

uint32_t PadSpeakerPreroll() {
	return g_config->pad_speaker_preroll;
}

bool PadSpeakerMuted() {
	return g_pad_speaker_muted.load(std::memory_order_relaxed);
}

void SetPadSpeakerMuted(bool muted) {
	g_pad_speaker_muted.store(muted, std::memory_order_relaxed);
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
