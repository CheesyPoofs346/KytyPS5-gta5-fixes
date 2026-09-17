#include <cstdio>
#include "common/emulatorConfig.h"

#include "common/assert.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
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
static std::atomic<bool>     g_light_partial_flush {false};
static std::atomic<uint32_t> g_eop_flush_interval {1};
static std::atomic<uint32_t> g_stream_repeat_threshold {0};
static std::atomic<bool>     g_stream_read_census {false};
static std::atomic<uint32_t> g_warmup_frames {0};
static std::atomic<bool>     g_backing_fast_path {false};
static std::atomic<bool>     g_texture_single_walk {false};
static std::atomic<bool>     g_backing_lock_sample {false};
static std::atomic<bool>     g_batch_census {false};
static std::atomic<bool>     g_worker_resolve_images {false};
static std::atomic<bool>     g_compiled_srt {false};
static std::atomic<bool>     g_dma_census {false};
static std::atomic<uint32_t> g_blocked_poll_us {0};
static std::atomic<uint32_t> g_blocked_poll_tries {0};
static std::atomic<bool>     g_buffer_census {false};
static std::atomic<bool>     g_image_census {false};
static std::atomic<bool>     g_buffer_growth {false};
static std::atomic<bool> g_cache_descriptors {false};
static std::atomic<bool> g_pipeline_memo {true};
static std::atomic<bool> g_buffer_dedup {true};
static std::atomic<bool> g_binding_publish {false};

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
	g_light_partial_flush.store(cfg.light_partial_flush, std::memory_order_relaxed);
	g_eop_flush_interval.store(cfg.eop_flush_interval, std::memory_order_relaxed);
	g_stream_repeat_threshold.store(cfg.stream_repeat_threshold, std::memory_order_relaxed);
	g_stream_read_census.store(cfg.stream_read_census, std::memory_order_relaxed);
	g_warmup_frames.store(cfg.warmup_frames, std::memory_order_relaxed);
	g_backing_fast_path.store(cfg.backing_fast_path, std::memory_order_relaxed);
	g_texture_single_walk.store(cfg.texture_single_walk, std::memory_order_relaxed);
	g_backing_lock_sample.store(cfg.backing_lock_sample, std::memory_order_relaxed);
	g_batch_census.store(cfg.batch_census, std::memory_order_relaxed);
	g_worker_resolve_images.store(cfg.worker_resolve_images, std::memory_order_relaxed);
	g_compiled_srt.store(cfg.compiled_srt, std::memory_order_relaxed);
	g_dma_census.store(cfg.dma_census, std::memory_order_relaxed);
	g_blocked_poll_us.store(cfg.blocked_poll_us, std::memory_order_relaxed);
	g_blocked_poll_tries.store(cfg.blocked_poll_tries, std::memory_order_relaxed);
	g_buffer_census.store(cfg.buffer_census, std::memory_order_relaxed);
	g_image_census.store(cfg.image_census, std::memory_order_relaxed);
	g_buffer_growth.store(cfg.buffer_growth, std::memory_order_relaxed);
	g_cache_descriptors.store(cfg.cache_descriptors, std::memory_order_relaxed);
	g_pipeline_memo.store(cfg.pipeline_memo, std::memory_order_relaxed);
	g_buffer_dedup.store(cfg.buffer_dedup, std::memory_order_relaxed);
	g_binding_publish.store(cfg.binding_publish, std::memory_order_relaxed);
	LogEffectiveSettings();
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

uint32_t EopFlushInterval() {
	return g_eop_flush_interval.load(std::memory_order_relaxed);
}

bool BackingFastPath() {
	return g_backing_fast_path.load(std::memory_order_relaxed);
}

bool TextureSingleWalkEnabled() {
	return g_texture_single_walk.load(std::memory_order_relaxed);
}

bool BufferGrowthEnabled() {
	return g_buffer_growth.load(std::memory_order_relaxed);
}

bool BufferCensusEnabled() {
	return g_buffer_census.load(std::memory_order_relaxed);
}

bool ImageCensusEnabled() {
	return g_image_census.load(std::memory_order_relaxed);
}

uint32_t BlockedPollMicros() {
	return g_blocked_poll_us.load(std::memory_order_relaxed);
}

uint32_t BlockedPollTries() {
	return g_blocked_poll_tries.load(std::memory_order_relaxed);
}

bool DmaCensusEnabled() {
	return g_dma_census.load(std::memory_order_relaxed);
}

bool CompiledSrtEnabled() {
	return g_compiled_srt.load(std::memory_order_relaxed);
}

void SetCompiledSrtForTest(bool enabled) {
	g_compiled_srt.store(enabled, std::memory_order_relaxed);
}

bool WorkerResolveImages() {
	return g_worker_resolve_images.load(std::memory_order_relaxed);
}

bool BatchCensusEnabled() {
	return g_batch_census.load(std::memory_order_relaxed);
}

bool BackingLockSample() {
	return g_backing_lock_sample.load(std::memory_order_relaxed);
}

uint32_t WarmupFrames() {
	return g_warmup_frames.load(std::memory_order_relaxed);
}

uint32_t StreamRepeatThreshold() {
	return g_stream_repeat_threshold.load(std::memory_order_relaxed);
}

bool StreamReadCensusEnabled() {
	return g_stream_read_census.load(std::memory_order_relaxed);
}

bool LightPartialFlushEnabled() {
	return g_light_partial_flush.load(std::memory_order_relaxed);
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

void SetBindingPublishForTest(bool enabled) {
	g_binding_publish.store(enabled, std::memory_order_relaxed);
}

bool BindingPublishEnabled() {
	return g_binding_publish.load(std::memory_order_relaxed);
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

bool ParsePixelShaderChksum(const char* text, uint64_t& chksum) {
	// strtoull skips whitespace and accepts a sign ("-1" becomes 2^64-1), so require a digit first.
	if (text == nullptr || text[0] < '0' || text[0] > '9') {
		return false;
	}
	// A leading zero followed by a digit would be read as octal.
	if (text[0] == '0' && text[1] >= '0' && text[1] <= '9') {
		return false;
	}
	errno            = 0;
	char*      end   = nullptr;
	const auto value = std::strtoull(text, &end, 0);
	if (errno != 0 || end == text || *end != '\0' || value == 0 || value > UINT32_MAX) {
		return false;
	}
	chksum = value;
	return true;
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


void LogEffectiveSettings() {
	std::printf("EffectiveSettings:\n");
	std::printf("  %-28s %s\n", "draw_profile", g_draw_profile.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "secondary_record", g_secondary_record.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "draw_queue", g_draw_queue.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "parallel_resolution", g_parallel_resolution.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "test_parallel_bindings", g_test_parallel_bindings.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "frame_pipelining", g_frame_pipelining.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "coalesce_eop_flush", g_coalesce_eop_flush.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "light_partial_flush", g_light_partial_flush.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "pipeline_memo", g_pipeline_memo.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "buffer_dedup", g_buffer_dedup.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "dyn_state_cache", g_dyn_state_cache.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "hw_check", g_hw_check.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "defer_uploads", g_defer_uploads.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "defer_transitions", g_defer_transitions.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "cache_descriptors", g_cache_descriptors.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "log_ui_draws", g_log_ui_draws.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "show_fps_overlay", g_show_fps_overlay.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %u\n", "draw_workers", g_draw_workers.load(std::memory_order_relaxed));
	std::printf("  %-28s %u\n", "pipeline_depth", g_pipeline_depth.load(std::memory_order_relaxed));
	std::printf("  %-28s %u\n", "eop_flush_interval", g_eop_flush_interval.load(std::memory_order_relaxed));
	std::printf("  %-28s %u\n", "stream_repeat_threshold", g_stream_repeat_threshold.load(std::memory_order_relaxed));
	std::printf("  %-28s %s\n", "stream_read_census", g_stream_read_census.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %u\n", "warmup_frames", g_warmup_frames.load(std::memory_order_relaxed));
	std::printf("  %-28s %s\n", "backing_fast_path", g_backing_fast_path.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "texture_single_walk", g_texture_single_walk.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "backing_lock_sample", g_backing_lock_sample.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "batch_census", g_batch_census.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "worker_resolve_images", g_worker_resolve_images.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "compiled_srt", g_compiled_srt.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "dma_census", g_dma_census.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %u\n", "blocked_poll_us", g_blocked_poll_us.load(std::memory_order_relaxed));
	std::printf("  %-28s %u\n", "blocked_poll_tries", g_blocked_poll_tries.load(std::memory_order_relaxed));
	std::printf("  %-28s %s\n", "buffer_census", g_buffer_census.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "image_census", g_image_census.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "buffer_growth", g_buffer_growth.load(std::memory_order_relaxed) ? "true" : "false");
	std::printf("  %-28s %s\n", "binding_publish", g_binding_publish.load(std::memory_order_relaxed) ? "true" : "false");
	std::fflush(stdout);
}

} // namespace Config
