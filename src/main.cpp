#include "common/common.h"
#include "common/dateTime.h"
#include "common/debug.h"
#include "common/file.h"
#include "common/magicEnum.h"
#include "common/platform/sysDbg.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "emulator.h"
#include "kytyGitVersion.h"

#include <charconv>
#include <cstdio>
#include <fmt/format.h>

using namespace Common;
using namespace Emulator;

static std::string GetBuildString() {
	Date date = Date::FromMacros(std::string(__DATE__));

#if KYTY_BUILD == KYTY_BUILD_DEBUG
	std::string type = "Debug";
#elif KYTY_BUILD == KYTY_BUILD_RELEASE
	std::string type = "Release";
#else
	std::string type = "????";
#endif

	std::string compiler =
	    Debug::GetCompiler() + "-" + Debug::GetLinker() + "-" + Debug::GetBitness();

	std::string str =
	    fmt::format("{}, {}, ver = {}, git = {}, date = {}", type.c_str(), compiler.c_str(),
	                KYTY_VERSION, KYTY_GIT_VERSION, date.ToString().c_str());

	return str;
}

static void PrintUsage() {
	::printf("%s\n", GetBuildString().c_str());
	::printf("kyty_emulator --game <dir|elf> [options]\n\n");
	::printf("Options:\n");
	::printf("  --game <dir|elf>                     Game directory or ELF to load.\n");
	::printf("  --game-patch <json>                  ETAHen cheat file.\n");
	::printf("  --screen-width <num>                 Window width. Default: 1280.\n");
	::printf("  --screen-height <num>                Window height. Default: 720.\n");
	::printf(
	    "  --user-name <name>                   Local user name (1-16 bytes). Default: Kyty.\n");
	::printf("  --user-id <num>                      Local user ID. Default: %d.\n",
	         Config::DEFAULT_USER_ID);
	::printf(
	    "  --present-mode <value>               Fifo, Mailbox, or Immediate. Default: Fifo.\n");
	::printf("  --fullscreen                         Run in borderless desktop fullscreen.\n");
	::printf("  --vblank-frequency <num>             Virtual vblank frequency. Default: 60.\n");
	::printf("  --console-language <0-29>            Console language. Default: 1 (English US).\n");
	::printf("  --vulkan-validation <true|false>     Enable Vulkan validation.\n");
	::printf("  --gpu-assisted-validation <t|f>      Bounds-check shader accesses on the GPU.\n"
	         "                                       Implies --vulkan-validation; very slow.\n");
	::printf("  --shader-validation <true|false>     Enable shader validation.\n");
	::printf("  --shader-optimization-type <value>   None, Size, or Performance.\n");
	::printf("  --shader-log-direction <value>       Silent, Console, or File.\n");
	::printf("  --shader-log-folder <path>           Shader log output folder.\n");
	::printf("  --command-buffer-dump <true|false>   Enable command buffer dumps.\n");
	::printf("  --command-buffer-dump-folder <path>  Command buffer dump folder.\n");
	::printf("  --graphics-debug-dump <true|false>   Enable graphics debug dumps.\n");
	::printf("  --printf-direction <value>           Silent, Console, or File.\n");
	::printf("  --printf-output-file <path>          Guest printf output file.\n");
	::printf("  --profiler-direction <value>         None or Network.\n");
	::printf("  --spirv-debug-printf <true|false>    Enable SPIR-V debug printf.\n");
	::printf(
	    "  --readback-linear-images <true|false> Read back writable linear images on submit.\n");
	::printf("  --hdr-probe <true|false>             Log max/mean of every float color target.\n");
	::printf("  --hdr-probe-interval <num>           Probe every Nth frame. Default: 60.\n");
	::printf("  --skip-distant-layer <t|f>           Drop the variable-height distant-scenery\n"
	         "                                       strip that shows up as the horizon band.\n");
	::printf("  --skip-backdrop-pass <t|f>           Diagnostic: drop the collapsed-viewport-Z\n"
	         "                                       backdrop/sky draws entirely.\n");
	::printf("  --real-occlusion-queries <t|f>       Use real GPU occlusion queries instead of a\n"
	         "                                       synthetic constant. Default true.\n");
	::printf("  --fix-collapsed-depth-compare <t|f>  Let collapsed-viewport-Z MESHES through the\n"
	         "                                       depth test instead of rejecting them.\n");
	::printf("  --fix-degenerate-viewport-z <t|f>    Substitute [0,1] when a depth-WRITING draw\n"
	         "                                       has a collapsed viewport Z range.\n");
	::printf("  --suppress-band-pass <t|f>           Scissor out the horizon/sky band pass drawn\n"
	         "                                       into the presentation framebuffer.\n");
	::printf("  --depth-clear-per-frame <t|f>        Allow only ONE depth clear per buffer per\n"
	         "                                       frame, so a Z-prepass is not wiped mid-frame.\n");
	::printf("  --skip-scene-soft-transparent <t|f>  Drop blended LESS-compare scene draws (diagnostic)\n");
	::printf("  --fix-inverted-depth-compare <t|f>   Mirror LESS depth tests on reversed-Z\n");
	::printf("  --depth-clear-once <t|f>             Materialize a depth clear once per clear\n"
	         "                                       episode. Default true; false re-clears per\n"
	         "                                       render pass as before.\n");
	::printf("  --force-depth-always <t|f>           Diagnostic: force depth compare to ALWAYS,\n"
	         "                                       so nothing can be depth-rejected.\n");
	::printf("  --mask-unwritten-mrt <t|f>          Mask off MRT attachments the pixel shader\n"
	         "                                       never writes. Experimental; default off.\n");
	::printf("  --hdr-clamp <float>                  Clamp pixel-shader colour exports to this\n"
	         "                                       value on float targets. 0 disables.\n");
	::printf("  --playgo-hack                       Use the supplied PlayGo stub fallback.\n");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	::printf("  --redzone                            Protect the guest SysV red zone.\n");
#endif
	::printf("  --keymap <Control=Input>             DualSense mapping; may be repeated.\n");
	::printf("  --rd                                 Enable RenderDoc capture.\n");
}

static bool NextArg(int argc, char* argv[], int& index, std::string& out) {
	if (index + 1 >= argc) {
		return false;
	}

	index++;
	out = argv[index];
	return true;
}

static bool ParseBool(const std::string& value, bool& out) {
	if (Common::EqualNoCase(value, "true") || value == "1" || Common::EqualNoCase(value, "yes") ||
	    Common::EqualNoCase(value, "on")) {
		out = true;
		return true;
	}

	if (Common::EqualNoCase(value, "false") || value == "0" || Common::EqualNoCase(value, "no") ||
	    Common::EqualNoCase(value, "off")) {
		out = false;
		return true;
	}

	return false;
}

template <typename E>
static bool ParseEnum(const std::string& value, E& out) {
	auto enum_value = magic_enum::enum_cast<E>(value.c_str());
	if (!enum_value.has_value()) {
		return false;
	}

	out = enum_value.value();
	return true;
}

static bool ParseConsoleLanguage(const std::string& value, uint32_t& out) {
	uint32_t language = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), language);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    language > Config::MAX_CONSOLE_LANGUAGE) {
		return false;
	}
	out = language;
	return true;
}

static bool ParseUserId(const std::string& value, int32_t& out) {
	int32_t user_id   = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), user_id);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    !Config::IsConfiguredUserIdValid(user_id)) {
		return false;
	}
	out = user_id;
	return true;
}

static bool ParseArgs(int argc, char* argv[], RunOptions& options, bool& show_help) {
	show_help = false;

	for (int i = 1; i < argc; i++) {
		std::string arg = std::string(argv[i]);
		std::string value;

		if (arg == "--help" || arg == "-h") {
			show_help = true;
			continue;
		}

		if (arg == "--rd") {
			options.config.renderdoc_enabled = true;
			continue;
		}

		if (arg == "--fullscreen") {
			options.config.fullscreen_enabled = true;
			continue;
		}

		if (arg == "--playgo-hack") {
			options.config.playgo_hack_enabled = true;
			continue;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (arg == "--redzone") {
			options.config.red_zone_protection_enabled = true;
			continue;
		}
#endif

		if (!Common::StartsWith(arg, "--")) {
			::printf("game input must be provided with --game\n");
			return false;
		}

		if (!NextArg(argc, argv, i, value)) {
			::printf("missing value for %s\n", arg.c_str());
			return false;
		}

		if (arg == "--game") {
			if (!options.app0_dir.empty()) {
				::printf("--game can only be specified once\n");
				return false;
			}

			value = Common::FixFilenameSlash(value);
			if (Common::File::IsDirectoryExisting(value)) {
				options.app0_dir = value;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::File::IsFileExisting(value)) {
				options.app0_dir = Common::DirectoryWithoutFilename(value);
				if (options.app0_dir.empty()) {
					options.app0_dir = ".";
				}
				options.elf = "/app0/" + Common::FilenameWithoutDirectory(value);
			} else {
				::printf("--game must point to an existing directory or ELF: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--game-patch") {
			if (!options.game_patch.empty()) {
				::printf("--game-patch can only be specified once\n");
				return false;
			}
			value = Common::FixFilenameSlash(value);
			if (!Common::File::IsFileExisting(value)) {
				::printf("--game-patch must point to an existing file: %s\n", value.c_str());
				return false;
			}
			options.game_patch = value;
		} else if (arg == "--screen-width") {
			options.config.screen_width = static_cast<uint32_t>(Common::ToInt32(value));
		} else if (arg == "--screen-height") {
			options.config.screen_height = static_cast<uint32_t>(Common::ToInt32(value));
		} else if (arg == "--user-name") {
			if (value.empty() || value.size() > Config::MAX_USER_NAME_LENGTH) {
				::printf("invalid user name: must contain 1-%zu bytes\n",
				         Config::MAX_USER_NAME_LENGTH);
				return false;
			}
			options.config.user_name = value;
		} else if (arg == "--user-id") {
			if (!ParseUserId(value, options.config.user_id)) {
				::printf("invalid user ID: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--present-mode") {
			if (!ParseEnum(value, options.config.present_mode)) {
				::printf("invalid present mode: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--vblank-frequency") {
			const int32_t vblank_frequency = Common::ToInt32(value);
			options.config.vblank_frequency =
			    static_cast<uint32_t>(vblank_frequency < 0 ? 0 : vblank_frequency);
		} else if (arg == "--console-language") {
			if (!ParseConsoleLanguage(value, options.config.console_language)) {
				::printf("invalid console language: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--vulkan-validation") {
			if (!ParseBool(value, options.config.vulkan_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--gpu-assisted-validation") {
			if (!ParseBool(value, options.config.gpu_assisted_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-validation") {
			if (!ParseBool(value, options.config.shader_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-optimization-type") {
			if (!ParseEnum(value, options.config.shader_optimization_type)) {
				::printf("invalid shader optimization type: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-direction") {
			if (!ParseEnum(value, options.config.shader_log_direction)) {
				::printf("invalid shader log direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-folder") {
			options.config.shader_log_folder = value;
		} else if (arg == "--command-buffer-dump") {
			if (!ParseBool(value, options.config.command_buffer_dump_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--command-buffer-dump-folder") {
			options.config.command_buffer_dump_folder = value;
		} else if (arg == "--graphics-debug-dump") {
			if (!ParseBool(value, options.config.graphics_debug_dump_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--printf-direction") {
			if (!ParseEnum(value, options.config.printf_direction)) {
				::printf("invalid printf direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--printf-output-file") {
			options.config.printf_output_file = value;
		} else if (arg == "--profiler-direction") {
			if (!ParseEnum(value, options.config.profiler_direction)) {
				::printf("invalid profiler direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--spirv-debug-printf") {
			if (!ParseBool(value, options.config.spirv_debug_printf_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--readback-linear-images") {
			if (!ParseBool(value, options.config.readback_linear_images)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--hdr-probe") {
			if (!ParseBool(value, options.config.hdr_probe_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--hdr-probe-interval") {
			const int32_t hdr_interval = Common::ToInt32(value);
			options.config.hdr_probe_interval =
			    static_cast<uint32_t>(hdr_interval < 1 ? 1 : hdr_interval);
		} else if (arg == "--hdr-probe-start") {
			const int32_t hdr_start = Common::ToInt32(value);
			options.config.hdr_probe_start = static_cast<uint32_t>(hdr_start < 0 ? 0 : hdr_start);
		} else if (arg == "--skip-ps-chksum") {
			options.config.skip_ps_chksum.push_back(std::strtoull(value.c_str(), nullptr, 0));
		} else if (arg == "--gpu-timestamps") {
			if (!ParseBool(value, options.config.gpu_timestamps)) {
				::printf("invalid boolean for --gpu-timestamps\n");
				return false;
			}
		} else if (arg == "--gpu-timestamps-ps-chksum") {
			uint64_t chksum = 0;
			if (!Config::ParseShaderChksum32(value.c_str(), chksum)) {
				::printf("invalid pixel shader checksum for %s: %s (expected a nonzero 32-bit value)\n",
				         arg.c_str(), value.c_str());
				return false;
			}
			options.config.gpu_timestamps_ps_chksum.push_back(chksum);
		} else if (arg == "--cache-descriptors") {
			if (!ParseBool(value, options.config.cache_descriptors)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--parallel-resolution") {
			if (!ParseBool(value, options.config.parallel_resolution)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--eop-flush-interval") {
			options.config.eop_flush_interval =
			    static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 0));
			if (options.config.eop_flush_interval == 0) {
				options.config.eop_flush_interval = 1;
			}
		} else if (arg == "--binding-publish") {
			if (!ParseBool(value, options.config.binding_publish)) {
				::printf("invalid boolean for --binding-publish\n");
				return false;
			}
		} else if (arg == "--buffer-growth") {
			if (!ParseBool(value, options.config.buffer_growth)) {
				::printf("invalid boolean for --buffer-growth\n");
				return false;
			}
		} else if (arg == "--image-census") {
			if (!ParseBool(value, options.config.image_census)) {
				::printf("invalid boolean for --image-census\n");
				return false;
			}
		} else if (arg == "--buffer-census") {
			if (!ParseBool(value, options.config.buffer_census)) {
				::printf("invalid boolean for --buffer-census\n");
				return false;
			}
		} else if (arg == "--blocked-poll-us") {
			options.config.blocked_poll_us = static_cast<uint32_t>(strtoul(value.c_str(), nullptr, 10));
		} else if (arg == "--blocked-poll-tries") {
			options.config.blocked_poll_tries = static_cast<uint32_t>(strtoul(value.c_str(), nullptr, 10));
		} else if (arg == "--dma-census") {
			if (!ParseBool(value, options.config.dma_census)) {
				::printf("invalid boolean for --dma-census\n");
				return false;
			}
		} else if (arg == "--compiled-srt") {
			if (!ParseBool(value, options.config.compiled_srt)) {
				::printf("invalid boolean for --compiled-srt\n");
				return false;
			}
		} else if (arg == "--worker-resolve-images") {
			if (!ParseBool(value, options.config.worker_resolve_images)) {
				::printf("invalid boolean for --worker-resolve-images\n");
				return false;
			}
		} else if (arg == "--batch-census") {
			if (!ParseBool(value, options.config.batch_census)) {
				::printf("invalid boolean for --batch-census\n");
				return false;
			}
		} else if (arg == "--backing-fast-path") {
			if (!ParseBool(value, options.config.backing_fast_path)) {
				::printf("invalid boolean for --backing-fast-path\n");
				return false;
			}
		} else if (arg == "--texture-single-walk") {
			if (!ParseBool(value, options.config.texture_single_walk)) {
				::printf("invalid boolean for --texture-single-walk\n");
				return false;
			}
		} else if (arg == "--backing-lock-sample") {
			if (!ParseBool(value, options.config.backing_lock_sample)) {
				::printf("invalid boolean for --backing-lock-sample\n");
				return false;
			}
		} else if (arg == "--warmup-frames") {
			options.config.warmup_frames =
			    static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 0));
		} else if (arg == "--stream-repeat-threshold") {
			options.config.stream_repeat_threshold =
			    static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 0));
		} else if (arg == "--stream-read-census") {
			if (!ParseBool(value, options.config.stream_read_census)) {
				::printf("invalid boolean for --stream-read-census\n");
				return false;
			}
		} else if (arg == "--light-partial-flush") {
			if (!ParseBool(value, options.config.light_partial_flush)) {
				::printf("invalid boolean for --light-partial-flush\n");
				return false;
			}
		} else if (arg == "--coalesce-eop-flush") {
			if (!ParseBool(value, options.config.coalesce_eop_flush)) {
				::printf("invalid boolean for --coalesce-eop-flush\n");
				return false;
			}
		} else if (arg == "--log-ui-draws") {
			if (!ParseBool(value, options.config.log_ui_draws)) {
				::printf("invalid boolean for %s\n", arg.c_str());
				return false;
			}
		} else if (arg == "--frame-pipelining") {
			if (!ParseBool(value, options.config.frame_pipelining)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--pipeline-depth") {
			options.config.pipeline_depth =
			    static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 0));
			if (options.config.pipeline_depth == 0) {
				::printf("--pipeline-depth must be at least 1\n");
				return false;
			}
		} else if (arg == "--test-parallel-bindings") {
			if (!ParseBool(value, options.config.test_parallel_bindings)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--defer-transitions") {
			if (!ParseBool(value, options.config.defer_transitions)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--defer-uploads") {
			if (!ParseBool(value, options.config.defer_uploads)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--draw-queue") {
			if (!ParseBool(value, options.config.draw_queue)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--draw-workers") {
			options.config.draw_workers =
			    static_cast<uint32_t>(std::max(1, std::atoi(value.c_str())));
		} else if (arg == "--secondary-record") {
			if (!ParseBool(value, options.config.secondary_record)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--draw-profile") {
			if (!ParseBool(value, options.config.draw_profile)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--hw-check") {
			if (!ParseBool(value, options.config.hw_check)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--dyn-state-cache") {
			if (!ParseBool(value, options.config.dyn_state_cache)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--pipeline-memo") {
			if (!ParseBool(value, options.config.pipeline_memo)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--buffer-dedup") {
			if (!ParseBool(value, options.config.buffer_dedup)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--parallel-resolve") {
			if (!ParseBool(value, options.config.parallel_resolve)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--skip-ps") {
			options.config.skip_ps.push_back(std::strtoull(value.c_str(), nullptr, 0));
		} else if (arg == "--skip-distant-layer") {
			if (!ParseBool(value, options.config.skip_distant_layer)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--skip-backdrop-pass") {
			if (!ParseBool(value, options.config.skip_backdrop_pass)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--real-occlusion-queries") {
			if (!ParseBool(value, options.config.real_occlusion_queries)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--fix-collapsed-depth-compare") {
			if (!ParseBool(value, options.config.fix_collapsed_depth_compare)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--fix-degenerate-viewport-z") {
			if (!ParseBool(value, options.config.fix_degenerate_viewport_z)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--suppress-band-pass") {
			if (!ParseBool(value, options.config.suppress_band_pass)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--depth-clear-per-frame") {
			if (!ParseBool(value, options.config.depth_clear_per_frame)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--pad-speaker-preroll") {
			options.config.pad_speaker_preroll =
			    static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 0));
		} else if (arg == "--pad-speaker-rate") {
			options.config.pad_speaker_rate =
			    static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 0));
			if (options.config.pad_speaker_rate < 8000) {
				::printf("invalid pad speaker rate: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--pad-speaker-volume") {
			options.config.pad_speaker_volume =
			    static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 0)) & 0xffu;
		} else if (arg == "--pad-speaker-muted") {
			if (!ParseBool(value, options.config.pad_speaker_muted)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--pad-speaker-bluetooth") {
			if (!ParseBool(value, options.config.pad_speaker_bluetooth)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--trigger-swap") {
			if (!ParseBool(value, options.config.trigger_swap)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--trigger-position-scale") {
			options.config.trigger_position_scale = std::strtof(value.c_str(), nullptr);
			if (options.config.trigger_position_scale < 0.0f) {
				::printf("invalid trigger position scale: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--trigger-strength") {
			options.config.trigger_strength = std::strtof(value.c_str(), nullptr);
			if (!(options.config.trigger_strength > 0.0f)) {
				::printf("invalid trigger strength: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--retry-transient-map-faults") {
			if (!ParseBool(value, options.config.retry_transient_map_faults)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--wait-image-readback") {
			if (!ParseBool(value, options.config.wait_image_readback)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--skip-scene-soft-transparent") {
			if (!ParseBool(value, options.config.skip_scene_soft_transparent)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--fix-inverted-depth-compare") {
			if (!ParseBool(value, options.config.fix_inverted_depth_compare)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--depth-clear-once") {
			if (!ParseBool(value, options.config.depth_clear_once)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--force-depth-always") {
			if (!ParseBool(value, options.config.force_depth_always)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--mask-unwritten-mrt") {
			if (!ParseBool(value, options.config.mask_unwritten_mrt)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--hdr-clamp") {
			options.config.hdr_export_clamp = std::strtof(value.c_str(), nullptr);
		} else if (arg == "--keymap") {
			const auto split = value.find('=');
			if (split == std::string::npos || split == 0 || split + 1 == value.size()) {
				::printf("invalid keymap: %s\n", value.c_str());
				return false;
			}
			options.config.keymap.push_back(value);
		} else {
			::printf("unknown option: %s\n", arg.c_str());
			return false;
		}
	}

	if (options.config.gpu_assisted_validation_enabled) {
		options.config.vulkan_validation_enabled = true;
	}

	return show_help || (!options.app0_dir.empty() && !options.elf.empty());
}

int main(int argc, char* argv[]) {
	VirtualMemory::Init();
	InitializeThreads();

	RunOptions options;
	bool       show_help = false;

	if (argc < 2) {
		PrintUsage();
		return 0;
	}

	if (!ParseArgs(argc, argv, options, show_help)) {
		PrintUsage();
		return 1;
	}

	if (show_help) {
		PrintUsage();
		return 0;
	}

	Run(options);

	return 0;
}
