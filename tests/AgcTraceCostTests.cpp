// What did the per-draw AGC LOGF calls actually cost?
//
// LOGF is gated only on Log::IsSilent(), which is FALSE in normal runs, so the fmt::sprintf ran
// on every call. Almo7aya 63dd391 replaced ~36 of them in agc.cpp with AgcTrace, gated on
// GraphicsDebugDumpEnabled() which is off in performance runs.
//
// This measures ONE call of a representative removed format against the gated replacement. It
// does NOT measure how many times per frame the guest calls those functions - no counter covers
// agc.cpp - so it gives a per-call cost, not a frame cost.

#include <cinttypes>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fmt/printf.h>
#include <string>

namespace {

// Verbatim shape of the formats removed from the per-draw register setters in agc.cpp.
volatile uint64_t g_sink = 0;
volatile bool     g_debug_dump_enabled = false;   // off in performance runs

double TimeFormat(size_t iterations) {
	const auto start = std::chrono::steady_clock::now();
	for (size_t i = 0; i < iterations; i++) {
		auto s = fmt::sprintf("\t cmd      = 0x%016" PRIx64 "\n"
		                      "\t regs     = 0x%016" PRIx64 "\n"
		                      "\t num_regs = 0x%08" PRIx32 "\n",
		                      static_cast<uint64_t>(i), static_cast<uint64_t>(i * 7u),
		                      static_cast<uint32_t>(i & 0xffffu));
		g_sink += s.size();   // consume so it cannot be optimised away
	}
	return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
	           .count() /
	       static_cast<double>(iterations);
}

double TimeGated(size_t iterations) {
	const auto start = std::chrono::steady_clock::now();
	for (size_t i = 0; i < iterations; i++) {
		// The replacement's fast path: a relaxed flag load and a branch, no formatting.
		// A volatile bool stands in for Config::GraphicsDebugDumpEnabled() so this benchmark
		// does not need the config subsystem initialised; the shape (load + predictable
		// not-taken branch) is what is being timed.
		if (g_debug_dump_enabled) {
			auto s = fmt::sprintf("unused %d", static_cast<int>(i));
			g_sink += s.size();
		}
		g_sink += 1;
	}
	return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
	           .count() /
	       static_cast<double>(iterations);
}

} // namespace

int main() {
	constexpr size_t kIterations = 200000;
	// Warm both paths before timing.
	TimeFormat(2000);
	TimeGated(2000);

	const auto formatted = TimeFormat(kIterations);
	const auto gated     = TimeGated(kIterations);

	std::printf("AgcTrace cost, per call:\n");
	std::printf("  formatting LOGF (old path)   %8.1f ns\n", formatted);
	std::printf("  gated AgcTrace (new path)    %8.1f ns\n", gated);
	std::printf("  saved per call               %8.1f ns\n", formatted - gated);
	std::printf("\n");
	std::printf("  agc.cpp lost ~36 such sites. Cost per frame depends on how often the guest\n");
	std::printf("  calls those register setters, which NO existing counter measures - so this\n");
	std::printf("  is a per-call figure only, not a frame-time claim.\n");
	std::printf("  For scale, at a hypothetical 10k calls/frame that would be %.2f ms/frame.\n",
	            (formatted - gated) * 10000.0 / 1e6);
	if (g_sink == 0) {
		std::printf("");
	}
	return 0;
}
