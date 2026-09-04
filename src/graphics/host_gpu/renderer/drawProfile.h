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
	PendingOps,          // of which
	RenderState,
	RefreshShaders,
	ShaderParams,        // of which
	ShaderLookup,        // of which, takes the program-cache mutex
	ShaderMaterialize,   // of which
	SrtEvaluate,         // of which, the actual expression walk
	SrtValidate,         // of which, the ungated ValidateResourceSpecialization
	SrtSnapshot,         // of which, snapshot construction + make_shared
	ApplyOutputs,        // of which, ApplyVertex/PixelOutputs
	Bindings,
	BindPrepare,        // of which
	BindFindBuffers,    // of which
	BindClampRange,     // of which, inside FindBuffers
	BindRebindBuffers,  // of which
	BindNativeBuffers,  // of which, inside RebindBuffers: the per-buffer resolve loop
	BindNativeUpload,   // of which, inside RebindBuffers: the two stream-buffer uploads
	BindRebindImages,   // of which
	VertexIndex,
	RenderTargets,
	Pipeline,
	Commit,
	DynState,
	BeginRendering,
	Emit,
	// The five regions inside DrawIndex that no timer covered. Together they were UNACCOUNTED:
	// 2.135 us/draw, 29% of TOTAL DrawIndex - larger than any single named phase.
	ExecEntry,           // ExecutePreparedDraw entry -> Bindings: probes, logging, user config
	Probes,              // between VertexIndex and Pipeline: HDR + depth diagnostic blocks
	BatchOpen,           // the secondary batch open/flush region - CONTAINS FlushSecondaryBatch
	Teardown,            // after Emit: ReturnPooledBindingStorage, ResetBindings
	Pm4NonDraw,          // outside DrawIndex entirely: every other PM4 packet
	Snapshot,            // outside DrawIndex: per-draw register snapshot for workers
	Total,
	Count,
};

inline bool DrawPhaseIsChild(DrawPhase phase) {
	switch (phase) {
		case DrawPhase::PendingOps:
		case DrawPhase::ShaderParams:
		case DrawPhase::ShaderLookup:
		case DrawPhase::ShaderMaterialize:
		case DrawPhase::SrtEvaluate:
		case DrawPhase::SrtValidate:
		case DrawPhase::SrtSnapshot:
		case DrawPhase::ApplyOutputs:
		case DrawPhase::BindPrepare:
		case DrawPhase::BindFindBuffers:
		case DrawPhase::BindClampRange:
		case DrawPhase::BindRebindBuffers:
		case DrawPhase::BindNativeBuffers:
		case DrawPhase::BindNativeUpload:
		case DrawPhase::BindRebindImages: return true;
		default: return false;
	}
}

inline const char* DrawPhaseName(DrawPhase phase) {
	switch (phase) {
		case DrawPhase::Preamble: return "preamble+checks";
		case DrawPhase::PendingOps: return "  of which PopPendingOperations";
		case DrawPhase::RenderState: return "PrepareDrawRenderState";
		case DrawPhase::RefreshShaders: return "RefreshShaders";
		case DrawPhase::ShaderParams: return "  of which PrepareProgram";
		case DrawPhase::ShaderLookup: return "  of which cache lookup (locked)";
		case DrawPhase::ShaderMaterialize: return "  of which SRT materialize";
		case DrawPhase::SrtEvaluate: return "    of which expression walk";
		case DrawPhase::SrtValidate: return "    of which Validate (ungated)";
		case DrawPhase::SrtSnapshot: return "    of which snapshot+make_shared";
		case DrawPhase::ApplyOutputs: return "    of which ApplyOutputs";
		case DrawPhase::Bindings: return "PrepareGraphicsBindings";
		case DrawPhase::BindPrepare: return "  of which PrepareBindings";
		case DrawPhase::BindFindBuffers: return "  of which FindBuffers";
		case DrawPhase::BindClampRange: return "    of which ClampRangeSize";
		case DrawPhase::BindRebindBuffers: return "  of which RebindBuffers";
		case DrawPhase::BindNativeBuffers: return "    of which buffer loop";
		case DrawPhase::BindNativeUpload: return "    of which NativeUpload x2";
		case DrawPhase::BindRebindImages: return "  of which RebindImages";
		case DrawPhase::VertexIndex: return "vertex+index buffers";
		case DrawPhase::RenderTargets: return "AcquireRenderTargets";
		case DrawPhase::Pipeline: return "CreateGraphicsPipeline";
		case DrawPhase::Commit: return "Commit vb/desc/ib";
		case DrawPhase::DynState: return "SetGraphicsDynamicParams";
		case DrawPhase::BeginRendering: return "BeginRendering+bind";
		case DrawPhase::Emit: return "EmitDrawPrimitives";
		case DrawPhase::ExecEntry: return "exec entry (probes+log)";
		case DrawPhase::Probes: return "HDR/depth probe blocks";
		case DrawPhase::BatchOpen: return "batch open + FlushSecondaryBatch";
		case DrawPhase::Teardown: return "teardown (ResetBindings)";
		case DrawPhase::Pm4NonDraw: return "PM4 non-draw packets";
		case DrawPhase::Snapshot: return "register snapshot";
		case DrawPhase::Total: return "TOTAL DrawIndex";
		default: return "?";
	}
}

struct DrawProfileState {
	std::array<uint64_t, static_cast<size_t>(DrawPhase::Count)> cycles {};
	uint64_t                              draws   = 0;
	uint64_t                              buffers = 0;   // resolved buffer descriptors
	// The ceiling on any "skip it when the state has not changed" design is how often the state
	// actually does not change. Measure that rather than assume it.
	uint64_t                              same_shader_pair = 0;
	uint64_t                              pm4_packets      = 0;
	uint64_t                              snapshots_built  = 0;
	uint64_t                              prev_vs_addr     = 0;
	uint64_t                              prev_vs_chksum   = 0;
	uint64_t                              prev_ps_addr     = 0;
	uint64_t                              prev_ps_chksum   = 0;
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
		// Latch whether profiling was on at construction. Reading g_draw_profile.active again in
		// Stop() is wrong: a timer opened before the first draw of a frame (active still false)
		// would then subtract a start of 0 from a full TSC value and poison the accumulator with
		// ~10^13 cycles. That is what made the PM4 phase report 670 us/draw.
		m_active = g_draw_profile.active;
		if (m_active) {
			m_start = DrawProfileReadCycles();
		}
	}

	~DrawPhaseTimer() { Stop(); }

	// Phases interleave with declarations that outlive them, so they cannot all be plain scopes.
	void Stop() {
		if (m_active && !m_stopped) {
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
	bool      m_active  = false;
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

// Called once per draw with the identity of both bound shaders.
inline void DrawProfileNoteShaders(uint64_t vs_addr, uint64_t vs_chksum, uint64_t ps_addr,
                                   uint64_t ps_chksum) {
	auto& profile = g_draw_profile;
	if (!profile.active) {
		return;
	}
	if (vs_addr == profile.prev_vs_addr && vs_chksum == profile.prev_vs_chksum &&
	    ps_addr == profile.prev_ps_addr && ps_chksum == profile.prev_ps_chksum) {
		profile.same_shader_pair++;
	}
	profile.prev_vs_addr   = vs_addr;
	profile.prev_vs_chksum = vs_chksum;
	profile.prev_ps_addr   = ps_addr;
	profile.prev_ps_chksum = ps_chksum;
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
	std::printf("DrawProfile: %llu draws, %.2f GHz effective, %.1f buffers/draw\n",
	            static_cast<unsigned long long>(profile.draws), 1.0 / ns_per_cycle,
	            static_cast<double>(profile.buffers) / draws);
	std::printf("  %-28s %7.1f%% of draws reuse the previous draw's shader pair\n", "state reuse",
	            100.0 * static_cast<double>(profile.same_shader_pair) / draws);
	std::printf("  %-28s %7.1f non-draw PM4 packets per draw\n", "packet rate",
	            static_cast<double>(profile.pm4_packets) / draws);
	std::printf("  %-28s %7.2f snapshots per draw (rest share by pointer)\n", "snapshot rate",
	            static_cast<double>(profile.snapshots_built) / draws);
	double accounted = 0.0;
	for (uint32_t i = 0; i < static_cast<uint32_t>(DrawPhase::Count); i++) {
		const auto   phase = static_cast<DrawPhase>(i);
		const double ns    = static_cast<double>(profile.cycles[i]) * ns_per_cycle / draws;
		if (phase != DrawPhase::Total && phase != DrawPhase::Pm4NonDraw &&
		    phase != DrawPhase::Snapshot && !DrawPhaseIsChild(phase)) {
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
