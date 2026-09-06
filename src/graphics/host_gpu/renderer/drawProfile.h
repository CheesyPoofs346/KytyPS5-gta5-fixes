#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_DRAWPROFILE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_DRAWPROFILE_H_

#include "common/emulatorConfig.h"

#include <atomic>
#include <array>
#include <vector>
#include <mutex>
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
	// Inside the buffer loop: 1.108 us across 5.5 buffers is ~200 ns each, and these three are
	// what ObtainBuffer does on a dedup miss. If the tracker queries dominate, this is the same
	// region-lock contention as VirtualRanges and a cross-draw cache is the wrong fix.
	ObtTracker,         // of which, IsRegionGpuModified + IsRegionCpuModified
	ObtFindBuffer,      // of which, the page-table lookup
	ObtSynchronize,     // of which, SynchronizeBuffer
	// The 1.355 us/draw that the buffer loop's other children do not account for. 97.6% of buffers
	// take ObtainBuffer's stream fast path, whose Map + TryReadBacking + Commit is a guest->stream
	// memcpy that no timer covered. The dedup scan in front of it was untimed too.
	// The whole stream-upload path, NOT a memcpy: alignment, buffer selection, Map (which
	// contains WaitPendingOperations, a GPU wait), TryReadBacking (the actual copy) and Commit
	// (which contains vmaFlushAllocation). Split below because the copy was assumed to dominate.
	ObtStreamCopy,
	ObtStreamMap,       // of which, stream.Map (includes the GPU wait)
	ObtStreamRead,      // of which, TryReadBacking - the actual guest->host memcpy
	ObtStreamCommit,    // of which, stream.Commit (includes vmaFlushAllocation)
	BindDedupScan,      // of which, FindDrawBufferCache
	// Phase 2c is the only parallel phase that scales negatively (16.1 -> 18.0 -> 22.8% at 2/4/8
	// workers) while phase 2a scales cleanly (16.7 -> 11.5 -> 9.0). The difference is that 2c takes
	// a shared_mutex per resource and 2a takes no lock at all. A shared_mutex reader acquire is an
	// atomic RMW on one counter, so it ping-pongs a cache line between cores. These time the
	// ACQUIRE only, not the work under the lock.
	TexLockAcquire,     // of which, FindTexture's shared_lock on the texture cache
	BufLockAcquire,     // of which, FindBuffer's shared_lock on the page table
	// The caller thread participates in ParallelFor as runner 0, so MustStageForWorker() is false
	// for it and it takes the EXCLUSIVE branch - a writer among readers, blocking every worker.
	TexLockExclusive,   // of which, FindTexture's exclusive lock (main thread / caller)
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
		case DrawPhase::ObtTracker:
		case DrawPhase::ObtFindBuffer:
		case DrawPhase::ObtSynchronize:
		case DrawPhase::ObtStreamMap:
		case DrawPhase::ObtStreamRead:
		case DrawPhase::ObtStreamCommit:
		case DrawPhase::ObtStreamCopy:
		case DrawPhase::BindDedupScan:
		case DrawPhase::TexLockAcquire:
		case DrawPhase::BufLockAcquire:
		case DrawPhase::TexLockExclusive:
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
		case DrawPhase::ObtTracker: return "      of which tracker queries";
		case DrawPhase::ObtFindBuffer: return "      of which FindBuffer";
		case DrawPhase::ObtSynchronize: return "      of which SynchronizeBuffer";
		case DrawPhase::ObtStreamCopy: return "      of which stream-upload path";
		case DrawPhase::ObtStreamMap: return "        of which Map (incl GPU wait)";
		case DrawPhase::ObtStreamRead: return "        of which TryReadBacking (memcpy)";
		case DrawPhase::ObtStreamCommit: return "        of which Commit (incl flush)";
		case DrawPhase::BindDedupScan: return "      of which dedup scan";
		case DrawPhase::TexLockAcquire: return "      of which texture lock acquire";
		case DrawPhase::BufLockAcquire: return "      of which buffer lock acquire";
		case DrawPhase::TexLockExclusive: return "      of which texture lock EXCLUSIVE";
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
// The draw phase profile is thread_local and was previously reported per thread on that
// thread's own cumulative draw count since process start. For attribution we need every
// thread's state, identified, and scoped to the route - so each thread registers itself and
// the reporter walks the registry.
struct DrawProfileRegistryEntry {
	DrawProfileState* state     = nullptr;
	uint32_t          thread_id = 0;
	// Route baselines, captured at ROUTE_START.
	std::array<uint64_t, static_cast<size_t>(DrawPhase::Count)> base_cycles {};
	uint64_t                                                    base_draws = 0;
};

inline std::mutex                            g_draw_profile_registry_lock;
inline std::vector<DrawProfileRegistryEntry> g_draw_profile_registry;

void DrawProfileRegister();
void DrawProfileSnapshotRoute();
void DrawProfileReportRoute();



// Section A diagnostic: separate execution from blocking, per role.
//
// The draw phase profile is thread_local and reported per thread on that thread's own draw
// count, so phases from different threads can never be added and a block is a cumulative mean
// since process start (it starts huge during boot and decays - do not compare two runs at
// different cumulative draw counts). These counters are deliberately different: process-wide
// relaxed atomics in nanoseconds, so PM4 execution, worker execution and explicit waits can be
// compared against one wall-clock route.
enum class CaptureBucket : size_t {
	Pm4Exec,          // inside PM4 packet handlers, excluding nested waits
	WorkerExec,       // inside draw-worker job execution
	WaitIdle,         // GuestGpu::WaitForIdle
	WaitPipeline,     // GuestGpu::WaitForPipelineDepth
	WaitWorkers,      // caller blocked on the worker pool
	WaitStream,       // stream ring waiting on GPU completion
	// PREDICATE STATES OBSERVED IMMEDIATELY BEFORE WAITING, from the loop
	//   while (m_processing || !m_commands.empty() || m_submission_count != 0)
	// These predicates CAN ALL BE TRUE AT ONCE. Attributing each slice to one of them by a
	// fixed priority makes the accounting non-overlapping and summable, but it does NOT
	// establish independent causes: a slice charged to ..commands may equally have been
	// blocked on submissions. Read these as 'what was observed true before this wait', then
	// trace the work that keeps the selected predicate active before choosing any fix.
	// They are NESTED INSIDE WaitIdle - never add them to it.
	WaitIdleProcessing,
	WaitIdleCommands,
	WaitIdleSubmissions,
	Count
};

inline std::array<std::atomic<uint64_t>, static_cast<size_t>(CaptureBucket::Count)>    g_capture_ns {};
inline std::array<std::atomic<uint64_t>, static_cast<size_t>(CaptureBucket::Count)>    g_capture_calls {};

// Guest flips (R_FLIP). A submission boundary is not a flip: correlate the two before
// calling a submission interval a frame time, and never claim displayed FPS from either.
inline std::atomic<uint64_t> g_guest_flips {0};
// Section E: anything recorded between two draws that a single host call cannot span.
// Identical draw-state keys are NOT sufficient - a barrier, rendering restart, dispatch, copy,
// blit, query or conditional-rendering boundary can sit between two otherwise identical draws.
// Bumped at those sites and folded into the census key, so such a pair can never be merged.
inline std::atomic<uint64_t> g_batch_boundary {0};

// Which recording activity ended the run. Without this a dominant boundary-op break reason is
// unattributable, and the census cannot say whether the split was necessary or incidental.
enum class BoundarySource : size_t {
	Rendering,        // BeginRendering / EndRendering
	GlobalBarrier,    // CommandProcessor::EmitGlobalBarrier
	PipelineBarrier,  // vkCmdPipelineBarrier(2) recorded between draws
	Clear,            // clearAttachments
	Dispatch,         // compute
	Transfer,         // copy / blit / resolve
	ExecuteCommands,  // secondary boundary: state does not carry across
	Count,
};

inline std::array<std::atomic<uint64_t>, static_cast<size_t>(BoundarySource::Count)>
    g_boundary_by_source {};

inline const char* BoundarySourceName(BoundarySource s) {
	switch (s) {
		case BoundarySource::Rendering: return "rendering-restart";
		case BoundarySource::GlobalBarrier: return "global-barrier";
		case BoundarySource::PipelineBarrier: return "pipeline-barrier";
		case BoundarySource::Clear: return "clear-attachments";
		case BoundarySource::Dispatch: return "dispatch";
		case BoundarySource::Transfer: return "copy/blit/resolve";
		case BoundarySource::ExecuteCommands: return "execute-commands";
		default: return "?";
	}
}

inline void NoteBatchBoundary(BoundarySource source) {
	g_batch_boundary.fetch_add(1, std::memory_order_relaxed);
	g_boundary_by_source[static_cast<size_t>(source)].fetch_add(1, std::memory_order_relaxed);
}


// Timers in flight when a route marker is consumed span the boundary: they began before it
// and their whole duration lands in the delta. Counted so the contamination is visible
// rather than silently folded into the route figure.
inline std::array<std::atomic<int64_t>, static_cast<size_t>(CaptureBucket::Count)>
    g_capture_inflight {};
inline std::array<uint64_t, static_cast<size_t>(CaptureBucket::Count)> g_capture_base_ns {};
inline std::array<uint64_t, static_cast<size_t>(CaptureBucket::Count)> g_capture_base_calls {};
inline std::array<int64_t, static_cast<size_t>(CaptureBucket::Count)> g_capture_base_inflight {};

// Thread CPU time (kernel+user) for the CALLING thread. Elapsed time alone cannot separate
// computation from blocking inside a function; CPU time can, because for one thread over one
// interval, elapsed - cpu is time the thread was not running. That subtraction is valid
// precisely because both sides are the same thread and the same interval, with no overlap.
// Defined out-of-line so the Windows headers stay out of this header.
uint64_t DrawProfileThreadCpuNs();

// Elapsed and CPU are kept in separate arrays; never mix them into one bucket.
inline std::array<std::atomic<uint64_t>, static_cast<size_t>(CaptureBucket::Count)>
    g_capture_cpu_ns {};
inline std::array<uint64_t, static_cast<size_t>(CaptureBucket::Count)> g_capture_base_cpu_ns {};

// Per-worker elapsed time. Aggregate worker time divided by worker count is an AVERAGE and
// says nothing about balance, so the distribution is recorded per runner and reported as
// min/median/max across runners.
inline constexpr size_t kCaptureMaxRunners = 33;
inline std::array<std::atomic<uint64_t>, kCaptureMaxRunners> g_runner_exec_ns {};
inline std::array<uint64_t, kCaptureMaxRunners>              g_runner_base_ns {};

inline bool g_capture_base_taken = false;
// Bumped when the baseline is taken. A timer stamps the generation at construction; if it
// completes under a newer generation it began BEFORE ROUTE_START, so its whole duration
// lands in the route delta even though only part of it belongs to the route. That
// contribution is accumulated separately and reported, because a delta containing it is not
// strictly route-contained and must not be used for precise percentages.
inline std::atomic<uint64_t> g_capture_generation {0};
inline std::array<std::atomic<uint64_t>, static_cast<size_t>(CaptureBucket::Count)>
    g_capture_spanning_ns {};
inline std::array<std::atomic<uint64_t>, static_cast<size_t>(CaptureBucket::Count)>
    g_capture_spanning_calls {};

// Snapshot at ROUTE_START so every bucket is reported as a route-scoped delta instead of a
// process-cumulative total contaminated by boot.
inline void CaptureSnapshotBaseline() {
	for (size_t i = 0; i < static_cast<size_t>(CaptureBucket::Count); ++i) {
		g_capture_base_ns[i]       = g_capture_ns[i].load(std::memory_order_relaxed);
		g_capture_base_calls[i]    = g_capture_calls[i].load(std::memory_order_relaxed);
		g_capture_base_inflight[i] = g_capture_inflight[i].load(std::memory_order_relaxed);
		g_capture_base_cpu_ns[i]   = g_capture_cpu_ns[i].load(std::memory_order_relaxed);
	}
	for (size_t i = 0; i < kCaptureMaxRunners; ++i) {
		g_runner_base_ns[i] = g_runner_exec_ns[i].load(std::memory_order_relaxed);
	}
	g_capture_base_taken = true;
	g_capture_generation.fetch_add(1, std::memory_order_relaxed);
}


inline void CaptureAccount(CaptureBucket bucket, uint64_t ns) {
	g_capture_ns[static_cast<size_t>(bucket)].fetch_add(ns, std::memory_order_relaxed);
	g_capture_calls[static_cast<size_t>(bucket)].fetch_add(1, std::memory_order_relaxed);
}

// As above, but for a timer that stamped `gen` at construction: if the generation has moved
// on, the interval straddles the baseline and its contribution is tracked separately too.
inline void CaptureAccountStamped(CaptureBucket bucket, uint64_t ns, uint64_t gen) {
	CaptureAccount(bucket, ns);
	if (gen != g_capture_generation.load(std::memory_order_relaxed)) {
		g_capture_spanning_ns[static_cast<size_t>(bucket)].fetch_add(ns,
		                                                             std::memory_order_relaxed);
		g_capture_spanning_calls[static_cast<size_t>(bucket)].fetch_add(
		    1, std::memory_order_relaxed);
	}
}

// Always on: one steady_clock pair per event, and the events are per-frame or per-packet, not
// per-draw-phase. Measured cost is far below the wait durations being recorded.
class CaptureTimer {
public:
	explicit CaptureTimer(CaptureBucket bucket)
	    : m_bucket(bucket), m_start(std::chrono::steady_clock::now()) {
		g_capture_inflight[static_cast<size_t>(bucket)].fetch_add(1, std::memory_order_relaxed);
		m_gen = g_capture_generation.load(std::memory_order_relaxed);
	}
	~CaptureTimer() {
		g_capture_inflight[static_cast<size_t>(m_bucket)].fetch_sub(1, std::memory_order_relaxed);
		CaptureAccountStamped(m_bucket,
		                      static_cast<uint64_t>(
		                          std::chrono::duration_cast<std::chrono::nanoseconds>(
		                              std::chrono::steady_clock::now() - m_start)
		                              .count()),
		                      m_gen);
	}
	CaptureTimer(const CaptureTimer&)            = delete;
	CaptureTimer& operator=(const CaptureTimer&) = delete;

private:
	CaptureBucket                         m_bucket;
	uint64_t                              m_gen = 0;
	std::chrono::steady_clock::time_point m_start;
};

// A bucket with no instrumentation site is UNMEASURED, which is not the same as measured-zero.
// wait-stream has no site wired yet and must never be reported as 0 ms.
inline bool CaptureBucketWired(CaptureBucket bucket) {
	return true;
}

// For recursive functions. ProcessPm4 recurses through ProcessIndirectBuffer, and a plain
// CaptureTimer re-counted the enclosing time on every nested call - measured ~33x inflation.
// Only the outermost activation on a thread records.
//
// NOTE: outermost elapsed time still INCLUDES any blocking done inside. It is wall time spent
// in the call, NOT exclusive CPU execution, and must never be labelled as such.
class CaptureOutermostTimer {
public:
	explicit CaptureOutermostTimer(CaptureBucket bucket) : m_bucket(bucket) {
		m_outermost = (t_depth[static_cast<size_t>(bucket)]++ == 0);
		if (m_outermost) {
			m_start = std::chrono::steady_clock::now();
			g_capture_inflight[static_cast<size_t>(bucket)].fetch_add(1,
			                                                          std::memory_order_relaxed);
			m_gen = g_capture_generation.load(std::memory_order_relaxed);
			m_cpu_start = DrawProfileThreadCpuNs();
		}
	}
	~CaptureOutermostTimer() {
		--t_depth[static_cast<size_t>(m_bucket)];
		if (!m_outermost) {
			return;
		}
		g_capture_inflight[static_cast<size_t>(m_bucket)].fetch_sub(1, std::memory_order_relaxed);
		const auto cpu_end = DrawProfileThreadCpuNs();
		g_capture_cpu_ns[static_cast<size_t>(m_bucket)].fetch_add(
		    cpu_end - std::min(cpu_end, m_cpu_start), std::memory_order_relaxed);
		CaptureAccountStamped(m_bucket,
		                      static_cast<uint64_t>(
		                          std::chrono::duration_cast<std::chrono::nanoseconds>(
		                              std::chrono::steady_clock::now() - m_start)
		                              .count()),
		                      m_gen);
	}
	CaptureOutermostTimer(const CaptureOutermostTimer&)            = delete;
	CaptureOutermostTimer& operator=(const CaptureOutermostTimer&) = delete;

private:
	static inline thread_local std::array<uint32_t, static_cast<size_t>(CaptureBucket::Count)>
	    t_depth {};
	CaptureBucket                         m_bucket;
	uint64_t                              m_gen       = 0;
	uint64_t                              m_cpu_start = 0;
	bool                                  m_outermost = false;
	std::chrono::steady_clock::time_point m_start;
};

inline const char* CaptureBucketName(CaptureBucket bucket) {
	switch (bucket) {
		case CaptureBucket::Pm4Exec: return "pm4-exec";
		case CaptureBucket::WorkerExec: return "worker-exec";
		case CaptureBucket::WaitIdle: return "wait-idle";
		case CaptureBucket::WaitPipeline: return "wait-pipeline";
		case CaptureBucket::WaitWorkers: return "wait-workers";
		case CaptureBucket::WaitStream: return "wait-stream";
		case CaptureBucket::WaitIdleProcessing: return "  ..processing";
		case CaptureBucket::WaitIdleCommands: return "  ..commands";
		case CaptureBucket::WaitIdleSubmissions: return "  ..submissions";
		default: return "?";
	}
}


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
	DrawProfileRegister();
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
