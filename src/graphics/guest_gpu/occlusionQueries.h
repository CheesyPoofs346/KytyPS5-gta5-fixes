#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_OCCLUSIONQUERIES_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_OCCLUSIONQUERIES_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;

// Real GPU occlusion queries for the guest's ZPASS_DONE counter dumps.
//
// The guest issues the dump event twice, one qword apart: once for the begin counters and once for
// the end counters, and derives the visible sample count as (end - begin). Publishing a synthetic
// constant for that difference cannot be correct, because the guest weighs the count against how
// much of the screen the object should cover - a fixed answer is right at one camera angle and
// wrong at the next, which shows up as geometry appearing and disappearing as the view moves.
//
// STATUS: does NOT work yet, and is default-OFF. The guest's begin and end dumps are far apart and
// the scheduler flushes the command buffer between them; a Vulkan occlusion query must begin and
// end in the SAME command buffer, so every query returns 0 samples. Reporting 0 means "fully
// occluded", which makes the guest cull MORE - strictly worse than the synthetic constant. Making
// this work needs the query to survive a flush (re-open on the new command buffer and sum), or a
// guarantee that no flush happens between the paired dumps.
//
// So the begin dump starts a real VK_QUERY_TYPE_OCCLUSION query and the end dump finishes it. The
// result is not ready at record time, so the end slots keep their not-ready state until the submit
// completes; bit 63 is the guest's ready flag, and it polls for it.
class OcclusionQueries {
public:
	OcclusionQueries() = default;
	KYTY_CLASS_NO_COPY(OcclusionQueries);

	// True once a pool exists and the feature is enabled.
	[[nodiscard]] bool Enabled() const noexcept { return m_pool != nullptr; }

	void Initialize(GraphicContext& graphics, CommandScheduler& scheduler);
	void Shutdown();

	// Handles one ZPASS_DONE dump. Returns false when the caller should fall back to the synthetic
	// path (feature disabled, or the pool is exhausted for this frame).
	[[nodiscard]] bool Dump(uint64_t event_address);

private:
	static constexpr uint32_t QueryCount  = 512;
	static constexpr uint32_t MaxSegments = 32;

	// A query cannot span a command buffer, but the guest's begin/end dumps routinely do, so the
	// count is accumulated across per-buffer segments.
	void OpenSegment();
	void CloseSegment();

	GraphicContext*       m_graphics  = nullptr;
	CommandScheduler*     m_scheduler = nullptr;
	vk::QueryPool         m_pool      = nullptr;
	uint32_t              m_next      = 0;
	bool                  m_counting  = false;
	bool                  m_overflowed = false;
	uint32_t              m_open_index = UINT32_MAX;
	std::vector<uint32_t> m_segments;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_GUEST_GPU_OCCLUSIONQUERIES_H_ */
