#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMESTAMPS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMESTAMPS_H_

// DIAGNOSTIC ONLY (--gpu-timestamps, default off). Brackets selected primary-buffer draw commands
// with a pair of Vulkan timestamp queries and aggregates the GPU interval by exact shader identity.
//
// Placement: vkCmdWriteTimestamp(TOP_OF_PIPELINE) immediately before EmitDrawPrimitives and
// vkCmdWriteTimestamp(BOTTOM_OF_PIPELINE) immediately after it, in the same primary command buffer
// and inside the already-open render pass. The interval therefore covers the GPU executing that
// one draw command plus whatever the driver defers to it (lazily applied state, descriptor or
// pipeline work bound just before). It is NOT the CPU cost of preparing or recording the draw, and
// it does not include barriers, uploads or render-pass load/store work recorded outside the pair.
// Tile-based or deferring drivers may move work out of the interval entirely.
//
// Intervals from different draws can overlap in time on the GPU timeline (asynchronous
// execution), so summed durations are "GPU time attributed by these brackets", not independent,
// additive costs. Reports label them that way.
//
// Synchronisation: nothing here waits. Results are read by a CommandScheduler deferred operation
// tagged with the tick of the command buffer that wrote them, which runs only once the scheduler
// already knows that tick completed; the read uses WITH_AVAILABILITY and never WAIT. Queries are
// reset on the host (hostQueryReset) only after that completion, then returned to the free list.
// If no pair is free the draw is simply not timed and `exhausted` is incremented.
//
// Query-pool and deferred-read pattern follows OcclusionQueries (graphics/guest_gpu); the
// checksum parser is ported from the ps-hash-draw-skip diagnostic.

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;

struct GpuTimestampIdentity {
	uint32_t ps_chksum  = 0; // ps_regs.chksum as declared by the guest (0 = none declared)
	uint32_t vs_chksum  = 0; // gs_regs.chksum as declared by the guest
	uint64_t ps_program = 0; // host ShaderProgram id (MixId of hash, stage, permutation)
	uint64_t vs_program = 0;

	bool operator==(const GpuTimestampIdentity& o) const noexcept {
		return ps_chksum == o.ps_chksum && vs_chksum == o.vs_chksum &&
		       ps_program == o.ps_program && vs_program == o.vs_program;
	}
};

struct GpuTimestampStats {
	static constexpr uint32_t Buckets = 48; // log2(ns) histogram, bucket i holds [2^i, 2^(i+1))

	uint64_t count    = 0;
	double   total_ns = 0.0;
	double   min_ns   = 0.0;
	double   max_ns   = 0.0;
	uint64_t histogram[Buckets] {};

	void               Add(double ns);
	// Lower bound of the histogram bucket holding the median; a coarse (factor-of-two) estimate.
	[[nodiscard]] double MedianBucketNs() const;
};

struct GpuTimestampCounters {
	uint64_t recorded          = 0; // pairs whose TOP timestamp was written
	uint64_t completed         = 0; // pairs read back with both values available
	uint64_t unavailable       = 0; // tick completed but a value was not available
	uint64_t unended           = 0; // Begin without End (early return); freed, not reported
	uint64_t exhausted         = 0; // selected draws not timed because no pair was free
	uint64_t skipped_secondary = 0; // selected draws on secondary / worker recording, not timed
	uint64_t read_errors       = 0; // getQueryPoolResults returned neither SUCCESS nor NOT_READY
	uint64_t batches           = 0; // deferred read operations scheduled
};

class GpuTimestamps {
public:
	static constexpr uint32_t DefaultPairCapacity = 2048;

	struct Token {
		uint32_t first = UINT32_MAX;
		uint64_t tick  = 0;
		[[nodiscard]] bool Valid() const noexcept { return first != UINT32_MAX; }
	};

	GpuTimestamps() = default;
	~GpuTimestamps();
	KYTY_CLASS_NO_COPY(GpuTimestamps);

	// `enabled` false leaves the diagnostic inert: no pool, every Begin returns an invalid token.
	// report_interval_ms 0 disables the periodic stdout report (tests read SnapshotStats instead).
	void Initialize(GraphicContext& graphics, CommandScheduler& scheduler, bool enabled,
	                uint32_t pair_capacity = DefaultPairCapacity, uint32_t report_interval_ms = 5000);
	// Releases the owner's reference. The pool itself is destroyed once no deferred read still
	// references it, so this is safe with reads outstanding.
	void Shutdown();

	[[nodiscard]] bool        Active() const noexcept { return m_active; }
	[[nodiscard]] const char* StatusText() const noexcept { return m_status.c_str(); }
	[[nodiscard]] double      NanosecondsPerTick() const noexcept { return m_ns_per_tick; }
	[[nodiscard]] uint32_t    ValidBits() const noexcept { return m_valid_bits; }

	// Empty selection = every eligible draw. Values are 32-bit guest pixel shader checksums.
	void               SetSelection(std::vector<uint64_t> ps_chksums);
	[[nodiscard]] bool Selects(uint32_t ps_chksum) const;

	// Begin/End must be recorded into the SAME command buffer with no submit in between.
	Token Begin(vk::CommandBuffer command, const GpuTimestampIdentity& identity);
	void  End(vk::CommandBuffer command, const Token& token);
	void  NoteSkippedSecondary();

	[[nodiscard]] GpuTimestampCounters GetCounters() const;
	[[nodiscard]] uint32_t             FreePairs() const;
	[[nodiscard]] std::vector<std::pair<GpuTimestampIdentity, GpuTimestampStats>>
	SnapshotStats() const;

	struct State;

private:
	std::shared_ptr<State> m_state;
	CommandScheduler*      m_scheduler   = nullptr;
	bool                   m_active      = false;
	double                 m_ns_per_tick = 0.0;
	uint32_t               m_valid_bits  = 0;
	std::string            m_status      = "off";
	std::vector<uint32_t>  m_selection;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMESTAMPS_H_
