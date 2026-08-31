#include "graphics/guest_gpu/occlusionQueries.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"

#include <atomic>

namespace Libs::Graphics {

namespace {

// Bit 63 is the guest's "result ready" flag; the low bits carry the counter.
constexpr uint64_t ReadyBit = 1ull << 63u;

void WriteAllDbSlots(uint64_t event_address, uint64_t value) {
	auto* results = reinterpret_cast<volatile uint64_t*>(event_address);
	// Each dump writes ONE value per DB at a 16-byte stride; the begin and end dumps target
	// addresses one qword apart, so writing both halves here would clobber the other pass.
	for (uint32_t db = 0; db < 16u; db++) {
		results[db * 2u] = value;
	}
}

} // namespace

void OcclusionQueries::Initialize(GraphicContext& graphics, CommandScheduler& scheduler) {
	m_graphics  = &graphics;
	m_scheduler = &scheduler;

	vk::QueryPoolCreateInfo info {};
	info.sType      = vk::StructureType::eQueryPoolCreateInfo;
	info.queryType  = vk::QueryType::eOcclusion;
	info.queryCount = QueryCount;

	if (graphics.device.createQueryPool(&info, nullptr, &m_pool) != vk::Result::eSuccess) {
		m_pool = nullptr;
		LOGF("OcclusionQueries: failed to create the query pool; using synthetic results\n");
		return;
	}

	// A query cannot span a command buffer, but the guest's begin and end dumps routinely do.
	// Close the open segment before each submit and start a fresh one on the next buffer; the
	// segments are summed when the guest's end dump arrives.
	scheduler.SetBufferBoundaryHooks([this] { CloseSegment(); }, [this] { OpenSegment(); });
	LOGF("OcclusionQueries: enabled with %" PRIu32 " queries\n", QueryCount);
}

void OcclusionQueries::Shutdown() {
	if (m_scheduler != nullptr) {
		m_scheduler->SetBufferBoundaryHooks(nullptr, nullptr);
	}
	if (m_pool != nullptr && m_graphics != nullptr) {
		m_graphics->device.destroyQueryPool(m_pool, nullptr);
	}
	m_pool = nullptr;
}

void OcclusionQueries::OpenSegment() {
	if (!m_counting || m_pool == nullptr || m_scheduler == nullptr || !m_scheduler->Active()) {
		return;
	}
	if (m_segments.size() >= MaxSegments || m_next >= QueryCount) {
		// Out of room: stop counting and let the end dump fall back to "visible".
		m_overflowed = true;
		return;
	}
	const auto index   = m_next++;
	auto       command = m_scheduler->Current().Handle();
	command.resetQueryPool(m_pool, index, 1);
	command.beginQuery(m_pool, index, vk::QueryControlFlags {});
	m_open_index = index;
}

void OcclusionQueries::CloseSegment() {
	if (m_open_index == UINT32_MAX || m_pool == nullptr || m_scheduler == nullptr ||
	    !m_scheduler->Active()) {
		return;
	}
	m_scheduler->Current().Handle().endQuery(m_pool, m_open_index);
	m_segments.push_back(m_open_index);
	m_open_index = UINT32_MAX;
}

bool OcclusionQueries::Dump(uint64_t event_address) {
	// Recording a query needs a live command buffer. The event can arrive with no active
	// scheduler (unit tests drive the processor directly), so fall back rather than assert.
	if (m_pool == nullptr || m_scheduler == nullptr || !m_scheduler->Active()) {
		return false;
	}

	// A query must begin and end outside a render pass instance for the draws between them to be
	// counted without violating the same-render-pass rule.
	m_scheduler->EndRendering();

	if (!m_counting) {
		// Begin dump: start counting. The guest subtracts this baseline, so publish it now.
		m_segments.clear();
		m_overflowed = false;
		m_counting   = true;
		OpenSegment();
		WriteAllDbSlots(event_address, ReadyBit);
		return true;
	}

	// End dump: close the final segment. Results are not available until the submit completes, so
	// leave the end slots not-ready and fill them in from the deferred callback.
	CloseSegment();
	m_counting = false;
	WriteAllDbSlots(event_address, 0);

	auto*      graphics = m_graphics;
	auto       pool     = m_pool;
	auto       segments = m_segments;
	const bool lost     = m_overflowed || segments.empty();
	m_segments.clear();

	m_scheduler->DeferOperation([graphics, pool, segments, lost, event_address] {
		// Fall back to "visible" rather than "occluded": dropping geometry the guest would have
		// drawn is far more damaging than drawing something it would have culled.
		constexpr uint64_t AssumeVisible = 0x00100000ull;
		uint64_t           samples       = 0;
		bool               ok            = !lost;
		if (ok) {
			for (const auto index: segments) {
				uint64_t   part   = 0;
				const auto result = graphics->device.getQueryPoolResults(
				    pool, index, 1, sizeof(part), &part, sizeof(part),
				    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
				if (result != vk::Result::eSuccess) {
					ok = false;
					break;
				}
				samples += part;
			}
		}
		if (!ok) {
			samples = AssumeVisible;
		}
		static std::atomic<uint32_t> q_log {0};
		if (q_log.fetch_add(1, std::memory_order_relaxed) < 30) {
			LOGF("OcclusionResult: segments=%u ok=%d samples=%" PRIu64 "\n",
			     static_cast<uint32_t>(segments.size()), static_cast<int>(ok), samples);
		}
		WriteAllDbSlots(event_address, ReadyBit | (samples & (ReadyBit - 1u)));
	});
	return true;
}

} // namespace Libs::Graphics
