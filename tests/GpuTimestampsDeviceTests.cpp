// Real-device tests for the --gpu-timestamps diagnostic (graphics/host_gpu/renderer/gpuTimestamps.h).
//
// Startup mirrors tests/BufferCacheDeviceHarness.cpp: device selection needs a surface, so a hidden
// 64x64 SDL window is opened. No game, no guest ELF, no PM4, no draws. Timestamp pairs are written
// into the scheduler's primary command buffer OUTSIDE a render pass with no work between them, so
// the measured durations are NOT draw costs; these tests cover support detection, asynchronous
// completion, query reset/reuse, exhaustion, buffer-boundary handling and identity aggregation.
//
// Waiting in this file (FlushAndWait, polling PopPendingOperations) is test-side only; the
// diagnostic itself never waits.
//
// SKIP (exit 2) only when there is no video driver or Vulkan loader.

#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <SDL_vulkan.h>

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "kernel/memory.h"

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/gpuTimestamps.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/window/windowInternal.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

// Validation errors are otherwise only logged (non-fatal ones never stop the process), so a test
// that "passes" while the layer reports misuse would prove nothing. Counted here and checked at the
// end. The mutation that removes the host query reset is caught only through this counter.
std::atomic<uint32_t> g_validation_errors {0};

VKAPI_ATTR vk::Bool32 VKAPI_CALL CountValidationErrors(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT /*types*/,
    const vk::DebugUtilsMessengerCallbackDataEXT* data, void* /*user*/) {
	if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError) {
		const auto n = g_validation_errors.fetch_add(1) + 1;
		if (n <= 3 && data != nullptr && data->pMessageIdName != nullptr) {
			std::printf("         validation error #%u: %s\n", n, data->pMessageIdName);
		}
	}
	return VK_FALSE;
}

bool Check(const char* test, bool ok, const std::string& detail) {
	std::printf("%s %-58s %s\n", ok ? "[ok]  " : "[FAIL]", test, detail.c_str());
	if (!ok) {
		g_failures++;
	}
	return ok;
}

std::string Counters(const GpuTimestampCounters& c) {
	char text[256];
	std::snprintf(text, sizeof(text),
	              "recorded=%llu completed=%llu unavailable=%llu unended=%llu exhausted=%llu "
	              "read_errors=%llu batches=%llu",
	              static_cast<unsigned long long>(c.recorded),
	              static_cast<unsigned long long>(c.completed),
	              static_cast<unsigned long long>(c.unavailable),
	              static_cast<unsigned long long>(c.unended),
	              static_cast<unsigned long long>(c.exhausted),
	              static_cast<unsigned long long>(c.read_errors),
	              static_cast<unsigned long long>(c.batches));
	return text;
}

GpuTimestampIdentity Id(uint32_t ps, uint32_t vs) {
	GpuTimestampIdentity id;
	id.ps_chksum  = ps;
	id.vs_chksum  = vs;
	id.ps_program = 0x1000ull + ps;
	id.vs_program = 0x2000ull + vs;
	return id;
}

// Submit and let the scheduler observe completion, then run deferred operations.
void Complete(CommandScheduler& scheduler) {
	scheduler.FlushAndWait();
	scheduler.PopPendingOperations();
}

} // namespace

int main() {
	std::setvbuf(stdout, nullptr, _IONBF, 0);
	std::printf("GPU timestamp diagnostic device tests:\n\n");

	// Parser (ported from the ps-hash-draw-skip diagnostic). Pure; runs before the device.
	{
		struct Case {
			const char* text;
			bool        ok;
			uint64_t    value;
		};
		const Case cases[] = {
		    {"0x1234abcd", true, 0x1234abcdull}, {"0xFFFFFFFF", true, 0xFFFFFFFFull},
		    {"305419896", true, 305419896ull},   {"0", false, 0},
		    {"0x0", false, 0},                   {"0x100000000", false, 0},
		    {"-1", false, 0},                    {" 1", false, 0},
		    {"+1", false, 0},                    {"0123", false, 0},
		    {"0x12z", false, 0},                 {"", false, 0},
		};
		int bad = 0;
		for (const auto& c: cases) {
			uint64_t   v  = 0;
			const bool ok = Config::ParseShaderChksum32(c.text, v);
			if (ok != c.ok || (ok && v != c.value)) {
				std::printf("         parser mismatch for \"%s\"\n", c.text);
				bad++;
			}
		}
		uint64_t v = 0;
		if (Config::ParseShaderChksum32(nullptr, v)) {
			bad++;
		}
		Check("checksum parser accepts only nonzero 32-bit values", bad == 0,
		      std::to_string(sizeof(cases) / sizeof(cases[0]) + 1) + " cases");
	}

	Common::VirtualMemory::Init();
	Common::InitializeThreads();
	Config::Initialize();
	{
		Config::ConfigOptions options;
		options.user_name                 = "harness";
		options.user_id                   = 1;
		options.screen_width              = 64;
		options.screen_height             = 64;
		options.vulkan_validation_enabled = true;
		options.gpu_timestamps            = true;
		options.gpu_timestamps_ps_chksum  = {0x11111111ull};
		// Console logging so validation-layer messages (non-fatal ones go through LOGF) are visible
		// in this test's output instead of silently dropped.
		options.printf_direction = Config::OutputDirection::Console;
		Config::Load(options);
	}
	Log::Initialize();
	Libs::LibKernel::Memory::Initialize();

	SDL_SetMainReady();
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		std::printf("[skip]  no SDL video driver available: %s\n", SDL_GetError());
		return 2;
	}
	if (SDL_Vulkan_LoadLibrary(nullptr) != 0) {
		std::printf("[skip]  no Vulkan loader available: %s\n", SDL_GetError());
		return 2;
	}

	auto* context_storage = new WindowContext();
	auto& context         = *context_storage;
	context.window = SDL_CreateWindow("kyty gpu timestamps", SDL_WINDOWPOS_UNDEFINED,
	                                  SDL_WINDOWPOS_UNDEFINED, 64, 64,
	                                  SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
	if (context.window == nullptr) {
		std::printf("[FAIL]  could not create the test window: %s\n", SDL_GetError());
		return 1;
	}
	context.graphic_ctx.screen_width  = 64;
	context.graphic_ctx.screen_height = 64;
	context.CreateVulkan();
	if (context.graphic_ctx.device == nullptr) {
		std::printf("[FAIL]  CreateVulkan produced no device\n");
		return 1;
	}
	auto& graphics = context.graphic_ctx;
	std::printf("device: %s\n\n", graphics.GetPhysicalDeviceProperties().deviceName.data());

	vk::DebugUtilsMessengerEXT error_counter = nullptr;
	{
		vk::DebugUtilsMessengerCreateInfoEXT info {};
		info.sType           = vk::StructureType::eDebugUtilsMessengerCreateInfoEXT;
		info.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
		info.messageType     = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
		                   vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;
		info.pfnUserCallback = CountValidationErrors;
		const bool created =
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDebugUtilsMessengerEXT != nullptr &&
		    graphics.instance.createDebugUtilsMessengerEXT(&info, nullptr, &error_counter) ==
		        vk::Result::eSuccess;
		if (!Check("validation error counter installed", created && Config::VulkanValidationEnabled(),
		           "a pass without validation would not cover query reset rules")) {
			return 1;
		}
	}

	auto& production = context.render_context->GetGpuTimestamps();
	{
		char detail[256];
		std::snprintf(detail, sizeof(detail),
		              "timestampValidBits=%u timestampPeriod=%.4f hostQueryReset=%d status=\"%s\"",
		              graphics.timestamp_valid_bits,
		              static_cast<double>(graphics.physical_device_properties.limits.timestampPeriod),
		              graphics.host_query_reset_enabled ? 1 : 0, production.StatusText());
		const bool ok = Check("device support detected, RenderContext instance active",
		                      graphics.timestamp_valid_bits > 0 && graphics.host_query_reset_enabled &&
		                          production.NanosecondsPerTick() > 0.0 && production.Active(),
		                      detail);
		if (!ok) {
			std::printf("\ncannot continue without timestamp support\n");
			return 1;
		}
	}
	Check("RenderContext selection taken from --gpu-timestamps-ps-chksum",
	      production.Selects(0x11111111u) && !production.Selects(0x22222222u), "0x11111111 only");

	static HW::Context    registers {};
	static HW::UserConfig user_config {};
	static HW::Shader     shaders {};
	auto&                 scheduler = context.render_context->GetCommandScheduler();
	scheduler.Begin(registers, user_config, shaders);
	const auto command = [&] { return scheduler.Current().Handle(); };

	// Disabled instance is inert.
	{
		GpuTimestamps off;
		off.Initialize(graphics, scheduler, false, 8, 0);
		const auto token = off.Begin(command(), Id(1, 2));
		off.End(command(), token);
		off.NoteSkippedSecondary();
		const auto c = off.GetCounters();
		Check("disabled instance records nothing", !off.Active() && !token.Valid() &&
		                                               c.recorded == 0 && off.FreePairs() == 0,
		      std::string("status=\"") + off.StatusText() + "\"");
	}

	// Asynchronous completion: nothing is read at record time.
	{
		GpuTimestamps ts;
		ts.Initialize(graphics, scheduler, true, 16, 0);
		const auto token = ts.Begin(command(), Id(0xAAAA0001u, 0xBBBB0001u));
		ts.End(command(), token);
		const auto before = ts.GetCounters();
		Check("pair recorded, result not read at record time",
		      token.Valid() && before.recorded == 1 && before.completed == 0 &&
		          ts.FreePairs() == 15,
		      Counters(before));

		// Non-waiting submit, then poll the scheduler's normal deferred-operation path.
		scheduler.Flush();
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
		uint32_t   polls    = 0;
		while (ts.GetCounters().completed == 0 && std::chrono::steady_clock::now() < deadline) {
			scheduler.PopPendingOperations();
			polls++;
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		const auto after = ts.GetCounters();
		const auto stats = ts.SnapshotStats();
		char       detail[256];
		std::snprintf(detail, sizeof(detail), "%s polls=%u ns=%.1f", Counters(after).c_str(), polls,
		              stats.empty() ? -1.0 : stats[0].second.total_ns);
		Check("deferred read completes after submit, pair released",
		      after.completed == 1 && after.unavailable == 0 && after.read_errors == 0 &&
		          ts.FreePairs() == 16 && stats.size() == 1 && stats[0].second.count == 1 &&
		          stats[0].first == Id(0xAAAA0001u, 0xBBBB0001u),
		      detail);
		ts.Shutdown();
	}

	// Identity aggregation across several buffers.
	{
		GpuTimestamps ts;
		ts.Initialize(graphics, scheduler, true, 64, 0);
		for (int buffer = 0; buffer < 3; buffer++) {
			for (int i = 0; i < 3; i++) {
				ts.End(command(), ts.Begin(command(), Id(0xA, 0x1)));
			}
			for (int i = 0; i < 2; i++) {
				ts.End(command(), ts.Begin(command(), Id(0xB, 0x2)));
			}
			Complete(scheduler);
		}
		const auto c     = ts.GetCounters();
		uint64_t   a     = 0;
		uint64_t   b     = 0;
		size_t     other = 0;
		for (const auto& [id, st]: ts.SnapshotStats()) {
			if (id == Id(0xA, 0x1)) {
				a = st.count;
			} else if (id == Id(0xB, 0x2)) {
				b = st.count;
			} else {
				other++;
			}
		}
		Check("samples aggregated by exact identity, one read per buffer",
		      a == 9 && b == 6 && other == 0 && c.completed == 15 && c.batches == 3,
		      Counters(c) + " A=" + std::to_string(a) + " B=" + std::to_string(b));
		ts.Shutdown();
	}

	// Exhaustion is explicit and does not disturb recording.
	{
		GpuTimestamps ts;
		ts.Initialize(graphics, scheduler, true, 4, 0);
		std::vector<GpuTimestamps::Token> tokens;
		for (int i = 0; i < 6; i++) {
			tokens.push_back(ts.Begin(command(), Id(0xC, 0x3)));
		}
		for (const auto& token: tokens) {
			ts.End(command(), token);
		}
		const auto mid   = ts.GetCounters();
		int        valid = 0;
		for (const auto& token: tokens) {
			valid += token.Valid() ? 1 : 0;
		}
		Complete(scheduler);
		const auto c = ts.GetCounters();
		Check("pool exhaustion counted, excess draws untimed, pairs return",
		      valid == 4 && mid.exhausted == 2 && c.completed == 4 && ts.FreePairs() == 4,
		      Counters(c));
		ts.Shutdown();
	}

	// Reset/reuse: many more pairs than the pool holds, across many buffers.
	{
		GpuTimestamps ts;
		ts.Initialize(graphics, scheduler, true, 4, 0);
		for (int cycle = 0; cycle < 50; cycle++) {
			for (int i = 0; i < 4; i++) {
				ts.End(command(), ts.Begin(command(), Id(0xD, static_cast<uint32_t>(cycle % 2))));
			}
			Complete(scheduler);
		}
		const auto c = ts.GetCounters();
		Check("200 pairs through a 4-pair pool via host reset and reuse",
		      c.recorded == 200 && c.completed == 200 && c.exhausted == 0 && c.unavailable == 0 &&
		          c.read_errors == 0 && ts.FreePairs() == 4,
		      Counters(c));
		ts.Shutdown();
	}

	// Begin without End (an early return between them).
	{
		GpuTimestamps ts;
		ts.Initialize(graphics, scheduler, true, 4, 0);
		(void)ts.Begin(command(), Id(0xE, 0x4));
		Complete(scheduler);
		const auto c = ts.GetCounters();
		Check("unended pair freed and not reported", c.unended == 1 && c.completed == 0 &&
		                                                 ts.FreePairs() == 4 &&
		                                                 ts.SnapshotStats().empty(),
		      Counters(c));
		ts.Shutdown();
	}

	// A submit between Begin and End must not pair timestamps across buffers.
	{
		GpuTimestamps ts;
		ts.Initialize(graphics, scheduler, true, 4, 0);
		const auto token = ts.Begin(command(), Id(0xF, 0x5));
		scheduler.FlushAndWait();
		ts.End(command(), token);
		scheduler.PopPendingOperations();
		Complete(scheduler);
		const auto c = ts.GetCounters();
		Check("End after a buffer boundary writes nothing, pair unended",
		      c.unended == 1 && c.completed == 0 && ts.FreePairs() == 4, Counters(c));
		ts.Shutdown();
	}

	// Owner shut down while a read is outstanding: the pool must outlive the recorded buffer.
	{
		auto* ts = new GpuTimestamps();
		ts->Initialize(graphics, scheduler, true, 4, 0);
		ts->End(command(), ts->Begin(command(), Id(0x10, 0x6)));
		ts->Shutdown();
		delete ts;
		Complete(scheduler);
		Check("shutdown with an outstanding read is safe", true,
		      "no crash; validation output (if any) is printed above");
	}

	// Periodic report path.
	{
		GpuTimestamps ts;
		ts.Initialize(graphics, scheduler, true, 8, 1);
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
		ts.End(command(), ts.Begin(command(), Id(0x12345678u, 0x9abcdef0u)));
		Complete(scheduler);
		Check("report emitted (see 'GpuTimestamps report' above)", ts.GetCounters().completed == 1,
		      Counters(ts.GetCounters()));
		ts.Shutdown();
	}

	scheduler.FlushAndWait();
	scheduler.Finish();
	Check("no Vulkan validation errors during the diagnostic tests", g_validation_errors.load() == 0,
	      std::to_string(g_validation_errors.load()) + " error message(s)");
	if (error_counter != nullptr) {
		graphics.instance.destroyDebugUtilsMessengerEXT(error_counter, nullptr);
	}
	if (graphics.device != nullptr && graphics.device.waitIdle() != vk::Result::eSuccess) {
		Check("orderly shutdown", false, "vkDeviceWaitIdle failed");
	}
	SDL_Window* window = context.window;
	delete context_storage;
	if (window != nullptr) {
		SDL_DestroyWindow(window);
	}
	SDL_Quit();

	if (g_failures != 0) {
		std::printf("\n%d check(s) FAILED\n", g_failures);
		return 1;
	}
	std::printf("\nall GPU timestamp checks passed\n");
	return 0;
}
