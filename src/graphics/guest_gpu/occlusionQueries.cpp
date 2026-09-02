#include "graphics/guest_gpu/occlusionQueries.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"

#include <atomic>
#include <cstdio>

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

	m_free.clear();
	m_free.reserve(QueryCount);
	// Reverse order so the first acquisitions hand out 0, 1, 2 ... which keeps captures readable.
	for (uint32_t i = QueryCount; i > 0; i--) {
		m_free.push_back(i - 1);
	}
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

bool OcclusionQueries::AcquireIndex(uint32_t& out) {
	std::lock_guard lock(m_free_mutex);
	if (m_free.empty()) {
		return false;
	}
	out = m_free.back();
	m_free.pop_back();
	return true;
}

void OcclusionQueries::ReleaseIndices(const std::vector<uint32_t>& indices) {
	if (indices.empty()) {
		return;
	}
	std::lock_guard lock(m_free_mutex);
	m_free.insert(m_free.end(), indices.begin(), indices.end());
}

void OcclusionQueries::OpenSegment() {
	if (!m_counting || m_pool == nullptr || m_scheduler == nullptr || !m_scheduler->Active()) {
		return;
	}
	if (m_segments.size() >= MaxSegments) {
		// Out of room: stop counting and let the end dump fall back to "visible".
		m_overflowed = true;
		return;
	}
	uint32_t index = 0;
	if (!AcquireIndex(index)) {
		// Every slot is still awaiting its deferred read. Reusing one here would reset a query
		// whose result has not been consumed, so report an overflow and let the end dump fall
		// back to "visible" rather than publish a corrupted count.
		m_overflowed = true;
		return;
	}
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

void OcclusionQueries::CloseForSubmit() {
	if (m_open_index == UINT32_MAX || m_pool == nullptr || m_scheduler == nullptr) {
		m_reopen_after_submit = false;
		return;
	}
	// End it here so the begin/end pair stays inside one command buffer; the samples counted so
	// far are kept as their own segment and Dump() sums every segment.
	m_scheduler->Current().Handle().endQuery(m_pool, m_open_index);
	m_segments.push_back(m_open_index);
	m_open_index          = UINT32_MAX;
	m_reopen_after_submit = m_counting;
}

void OcclusionQueries::ReopenAfterSubmit() {
	if (!m_reopen_after_submit) {
		return;
	}
	m_reopen_after_submit = false;
	OpenSegment();
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
		// Any leftover segments still own their indices; hand them back rather than leaking.
		ReleaseIndices(m_segments);
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

	auto* self = this;
	m_scheduler->DeferOperation([self, graphics, pool, segments, lost, event_address] {
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
		// Health counters: the whole point of the index-ownership fix is that `ok` should now be
		// true almost always. If fallbacks stay high the fix did not work and the guest is still
		// being told "visible" for everything.
		static std::atomic<uint64_t> q_total {0};
		static std::atomic<uint64_t> q_ok {0};
		static std::atomic<uint64_t> q_zero {0};
		const auto seen = q_total.fetch_add(1, std::memory_order_relaxed) + 1;
		if (ok) {
			q_ok.fetch_add(1, std::memory_order_relaxed);
			if (samples == 0) {
				q_zero.fetch_add(1, std::memory_order_relaxed);
			}
		}
		if (seen % 2000 == 0) {
			const auto good = q_ok.load(std::memory_order_relaxed);
			std::printf("OcclusionHealth: total=%llu ok=%llu (%.1f%%) zero_of_ok=%llu (%.1f%%)\n",
			     static_cast<unsigned long long>(seen), static_cast<unsigned long long>(good),
			     100.0 * static_cast<double>(good) / static_cast<double>(seen),
			     static_cast<unsigned long long>(q_zero.load(std::memory_order_relaxed)),
			     good == 0 ? 0.0
			               : 100.0 * static_cast<double>(q_zero.load(std::memory_order_relaxed)) /
			                     static_cast<double>(good));
		}
		static std::atomic<uint32_t> q_log {0};
		if (q_log.fetch_add(1, std::memory_order_relaxed) < 30) {
			LOGF("OcclusionResult: segments=%u ok=%d samples=%" PRIu64 "\n",
			     static_cast<uint32_t>(segments.size()), static_cast<int>(ok), samples);
		}
		WriteAllDbSlots(event_address, ReadyBit | (samples & (ReadyBit - 1u)));
		// The results have been read, so the slots can be handed out again. Releasing only here
		// is the whole point: an index stays owned until its value has actually been consumed.
		self->ReleaseIndices(segments);
	});
	return true;
}

} // namespace Libs::Graphics
