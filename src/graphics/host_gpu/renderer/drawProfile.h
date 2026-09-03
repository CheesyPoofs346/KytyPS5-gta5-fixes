#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_DRAWPROFILE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_DRAWPROFILE_H_

#include "common/emulatorConfig.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>

namespace Libs::Graphics {

// The Tracy breakdown accounts for ~11.5 of the ~16 us a draw costs, and a stage without a zone
// is invisible. These phases tile the whole of DrawIndex so the remainder is printed as a number
// rather than inferred. Phases marked "of which" nest inside the phase above them and are
// excluded from the sum, so nesting does not double-count.
//
// rdtsc rather than steady_clock: a clock pair costs ~31 ns, which across a dozen phases would be
// most of what is being measured. Cycles convert to time once at print time against wall-clock
// elapsed over the measured span, so no TSC frequency is assumed.
enum class DrawPhase : uint32_t {
	Preamble,
	RenderState,
	RefreshShaders,
	Bindings,
	BindPrepare,        // of which
	BindFindBuffers,    // of which
	BindRebindBuffers,  // of which
	BindRebindImages,   // of which
	VertexIndex,
	RenderTargets,
	Pipeline,
	Commit,
	DynState,
	BeginRendering,
	Emit,
	Total,
	Count,
};

inline bool DrawPhaseIsChild(DrawPhase phase) {
	switch (phase) {
		case DrawPhase::BindPrepare:
		case DrawPhase::BindFindBuffers:
		case DrawPhase::BindRebindBuffers:
		case DrawPhase::BindRebindImages: return true;
		default: return false;
	}
}

inline const char* DrawPhaseName(DrawPhase phase) {
	switch (phase) {
		case DrawPhase::Preamble: return "preamble+checks";
		case DrawPhase::RenderState: return "PrepareDrawRenderState";
		case DrawPhase::RefreshShaders: return "RefreshShaders";
		case DrawPhase::Bindings: return "PrepareGraphicsBindings";
		case DrawPhase::BindPrepare: return "  of which PrepareBindings";
		case DrawPhase::BindFindBuffers: return "  of which FindBuffers";
		case DrawPhase::BindRebindBuffers: return "  of which RebindBuffers";
		case DrawPhase::BindRebindImages: return "  of which RebindImages";
		case DrawPhase::VertexIndex: return "vertex+index buffers";
		case DrawPhase::RenderTargets: return "AcquireRenderTargets";
		case DrawPhase::Pipeline: return "CreateGraphicsPipeline";
		case DrawPhase::Commit: return "Commit vb/desc/ib";
		case DrawPhase::DynState: return "SetGraphicsDynamicParams";
		case DrawPhase::BeginRendering: return "BeginRendering+bind";
		case DrawPhase::Emit: return "EmitDrawPrimitives";
		case DrawPhase::Total: return "TOTAL DrawIndex";
		default: return "?";
	}
}

struct DrawProfileState {
	std::array<uint64_t, static_cast<size_t>(DrawPhase::Count)> cycles {};
	uint64_t                              draws   = 0;
	bool                                  active  = false;
	bool                                  started = false;
	std::chrono::steady_clock::time_point wall_start {};
	uint64_t                              tsc_start = 0;
};

inline thread_local DrawProfileState g_draw_profile;

inline uint64_t DrawProfileReadCycles() {
	return __builtin_ia32_rdtsc();
}

class DrawPhaseTimer {
public:
	explicit DrawPhaseTimer(DrawPhase phase): m_phase(phase) {
		if (g_draw_profile.active) {
			m_start = DrawProfileReadCycles();
		}
	}

	~DrawPhaseTimer() { Stop(); }

	// Phases interleave with declarations that outlive them, so they cannot all be plain scopes.
	void Stop() {
		if (g_draw_profile.active && !m_stopped) {
			g_draw_profile.cycles[static_cast<size_t>(m_phase)] +=
			    DrawProfileReadCycles() - m_start;
			m_stopped = true;
		}
	}

	DrawPhaseTimer(const DrawPhaseTimer&)            = delete;
	DrawPhaseTimer& operator=(const DrawPhaseTimer&) = delete;
	DrawPhaseTimer(DrawPhaseTimer&&)                 = delete;
	DrawPhaseTimer& operator=(DrawPhaseTimer&&)      = delete;

private:
	DrawPhase m_phase;
	uint64_t  m_start   = 0;
	bool      m_stopped = false;
};

// Called once per draw, at the top of DrawIndex/DrawAuto, before any phase timer.
inline void DrawProfileBeginDraw() {
	auto& profile  = g_draw_profile;
	profile.active = Config::DrawProfileEnabled();
	if (!profile.active) {
		return;
	}
	if (!profile.started) {
		profile.started    = true;
		profile.wall_start = std::chrono::steady_clock::now();
		profile.tsc_start  = DrawProfileReadCycles();
	}
}

inline void DrawProfileEndDraw() {
	auto& profile = g_draw_profile;
	if (!profile.active) {
		return;
	}
	profile.draws++;
	if (profile.draws % 200000 != 0) {
		return;
	}
	const auto wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
	                         std::chrono::steady_clock::now() - profile.wall_start)
	                         .count();
	const auto tsc = DrawProfileReadCycles() - profile.tsc_start;
	if (tsc == 0 || wall_ns <= 0) {
		return;
	}
	const double ns_per_cycle = static_cast<double>(wall_ns) / static_cast<double>(tsc);
	const double draws        = static_cast<double>(profile.draws);
	std::printf("DrawProfile: %llu draws, %.2f GHz effective\n",
	            static_cast<unsigned long long>(profile.draws), 1.0 / ns_per_cycle);
	double accounted = 0.0;
	for (uint32_t i = 0; i < static_cast<uint32_t>(DrawPhase::Count); i++) {
		const auto   phase = static_cast<DrawPhase>(i);
		const double ns    = static_cast<double>(profile.cycles[i]) * ns_per_cycle / draws;
		if (phase != DrawPhase::Total && !DrawPhaseIsChild(phase)) {
			accounted += ns;
		}
		std::printf("  %-28s %8.3f us/draw\n", DrawPhaseName(phase), ns / 1000.0);
	}
	const double total =
	    static_cast<double>(profile.cycles[static_cast<size_t>(DrawPhase::Total)]) * ns_per_cycle /
	    draws;
	std::printf("  %-28s %8.3f us/draw  (%.1f%% of total)\n", "sum of phases", accounted / 1000.0,
	            total > 0.0 ? 100.0 * accounted / total : 0.0);
	std::printf("  %-28s %8.3f us/draw\n", "UNACCOUNTED", (total - accounted) / 1000.0);
	std::fflush(stdout);
}

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_DRAWPROFILE_H_ */
