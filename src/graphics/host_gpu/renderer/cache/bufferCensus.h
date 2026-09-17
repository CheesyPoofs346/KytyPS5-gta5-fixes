#ifndef INCLUDE_KYTY_GRAPHICS_RENDERER_CACHE_BUFFERCENSUS_H_
#define INCLUDE_KYTY_GRAPHICS_RENDERER_CACHE_BUFFERCENSUS_H_

// Diagnostic for one question: does this workload repeatedly recreate overlapping buffers?
//
// BufferCache::CreateBuffer never grows a buffer in place. A request reaching past the edge of an
// existing buffer widens the range to the union of every overlapping buffer, allocates a NEW
// buffer for the union, copies each old buffer into it, and deletes the originals. If the guest
// walks a region in small steps, that is a full reallocate-and-copy per step.
//
// The buckets below deliberately mirror the trigger in upstream 74a78f3 "grow repeatedly joined
// buffer ranges", which pads a replacement by StreamLeapSize once a range has been joined
// repeatedly. That commit keys on DIRECTION - it tests `expands_left` and `expands_right`
// separately and pads whichever side grew - and it accumulates its StreamScore across ALL
// overlaps in the join, not just a single one. A census that only recognised single-overlap
// rightward extension would miss most of what that mechanism acts on.
//
// Off by default; nothing is counted and nothing is printed when off.

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>

namespace Libs::Graphics {

// Classified by which way the union grew past what was asked for, because that is what the
// upstream mechanism keys on. Multi-overlap joins are NOT a separate kind - they are counted
// orthogonally in multi_overlap, since upstream treats a multi-buffer join as growth too.
enum class BufferCreateKind {
	// No overlapping buffer. NOTE: this does NOT mean "unavoidable". A fresh create may be the
	// first time a range is seen, OR a re-creation of a range that was evicted by the LRU or
	// merged away earlier. Those are separated by the recently-deleted probe below, within the
	// window that probe remembers - beyond it, they are indistinguishable here.
	Fresh,
	// NAMING, and it is not intuitive: these are upstream's own predicates, computed per overlap
	// against the RUNNING range exactly as ResolveOverlaps does -
	//     expands_left  = buffer_begin < begin
	//     expands_right = buffer_end   > end
	// evaluated BEFORE that buffer is folded in. They describe which side the OLD buffer sticks
	// out on, which is NOT the same as the direction the guest is walking: a guest walking upward
	// asks for a range starting inside a buffer that begins below it, so that join is
	// ExpandsLeft. Upstream pads whichever of these fired, so counting them is what says which
	// padding branch would trigger here.
	ExpandsLeft,
	ExpandsRight,
	ExpandsBoth,
	// Overlapping buffers exist but the union is exactly the requested range, so nothing was
	// widened. Reaching CreateBuffer at all means the page-table lookup did not find an in-bounds
	// owner; kept as its own bucket rather than silently folded into growth.
	Contained,
};

inline BufferCreateKind ClassifyBufferCreate(uint64_t overlap_count, bool expands_left,
                                             bool expands_right) {
	if (overlap_count == 0) {
		return BufferCreateKind::Fresh;
	}
	if (expands_left && expands_right) {
		return BufferCreateKind::ExpandsBoth;
	}
	if (expands_left) {
		return BufferCreateKind::ExpandsLeft;
	}
	if (expands_right) {
		return BufferCreateKind::ExpandsRight;
	}
	return BufferCreateKind::Contained;
}

// Why a buffer went away, so a later fresh create over the same address can say which.
enum class BufferDeleteReason { Merged, Evicted };

// A fixed window of recent deletions. Purpose: tell "first time this range was ever seen" apart
// from "this range was here, went away, and is being rebuilt". Without it, a Fresh-dominated
// result would be misread as proof that the creates are unavoidable.
//
// Bounded and lossy ON PURPOSE: a circular buffer of the last kSlots deletions, scanned linearly.
// A hit is exact (real overlap with a real recorded range). A MISS is not proof the range is new -
// it may simply have fallen out of the window. Report hits as a lower bound, never as a total.
class RecentDeletions {
public:
	static constexpr size_t kSlots = 512;

	void Note(uint64_t begin, uint64_t end, BufferDeleteReason reason) noexcept {
		auto& slot = m_slots[m_cursor];
		slot       = {begin, end, reason};
		m_cursor   = (m_cursor + 1) % kSlots;
		if (m_recorded < kSlots) {
			m_recorded++;
		}
	}

	// Returns whether any recorded range overlaps [begin, end), and if so how that buffer died.
	[[nodiscard]] bool Find(uint64_t begin, uint64_t end,
	                        BufferDeleteReason* reason) const noexcept {
		for (size_t i = 0; i < m_recorded; i++) {
			const auto& slot = m_slots[i];
			if (slot.end > begin && slot.begin < end) {
				if (reason != nullptr) {
					*reason = slot.reason;
				}
				return true;
			}
		}
		return false;
	}

	void Reset() noexcept {
		m_cursor   = 0;
		m_recorded = 0;
	}

private:
	struct Slot {
		uint64_t           begin  = 0;
		uint64_t           end    = 0;
		BufferDeleteReason reason = BufferDeleteReason::Merged;
	};
	std::array<Slot, kSlots> m_slots {};
	size_t                   m_cursor   = 0;
	size_t                   m_recorded = 0;
};

struct BufferCensusStats {
	// Absolute counts. A share alone cannot size the problem: a handful of very expensive joins
	// can dominate the cost while being a rounding error in the create count, so every bucket
	// carries its bytes alongside its count.
	std::atomic<uint64_t> creates {0};
	std::atomic<uint64_t> fresh {0};
	std::atomic<uint64_t> expands_right {0};
	std::atomic<uint64_t> expands_left {0};
	std::atomic<uint64_t> expands_both {0};
	std::atomic<uint64_t> contained {0};

	// Bytes copied out of doomed buffers into their replacement - the cost of not growing in
	// place. Total, and the part attributable to growth of any direction.
	std::atomic<uint64_t> copied_bytes {0};
	std::atomic<uint64_t> growth_copied_bytes {0};
	// Allocated bytes of the replacement buffers, growth creates only, with the bytes they
	// replaced. Their ratio is how much slack a single join buys.
	std::atomic<uint64_t> growth_old_bytes {0};
	std::atomic<uint64_t> growth_new_bytes {0};

	// Joins folding several buffers at once. Counted orthogonally to the direction buckets
	// because upstream's StreamScore accumulates over every overlap in the join.
	std::atomic<uint64_t> multi_overlap {0};
	std::atomic<uint64_t> multi_overlap_copied_bytes {0};
	std::atomic<uint64_t> max_overlap_count {0};

	// Growth-prototype accounting, meaningful only with --buffer-growth on.
	std::atomic<uint64_t> leaps {0};
	std::atomic<uint64_t> pad_bytes {0};
	// Peak of BufferCache::m_total_used_memory, sampled at each create. Padding inflates this and
	// the GC thresholds key on it, so the A/B has to show what the prototype retains.
	std::atomic<uint64_t> peak_buffer_bytes {0};
	std::atomic<uint64_t> live_buffers_at_peak {0};

	// Fresh creates that landed on a range deleted recently. Lower bounds, not totals.
	std::atomic<uint64_t> fresh_after_evict {0};
	std::atomic<uint64_t> fresh_after_merge {0};

	void Reset() {
		for (auto* counter: {&creates, &fresh, &expands_right, &expands_left, &expands_both,
		                     &contained,
		                     &copied_bytes, &growth_copied_bytes, &growth_old_bytes,
		                     &growth_new_bytes, &multi_overlap, &multi_overlap_copied_bytes,
		                     &max_overlap_count, &leaps, &pad_bytes, &peak_buffer_bytes,
		                     &live_buffers_at_peak, &fresh_after_evict, &fresh_after_merge}) {
			counter->store(0);
		}
	}
};

inline BufferCensusStats g_buffer_census {};
inline RecentDeletions   g_recent_deletions {};

// Snapshot taken when the route starts, so boot and loading can be subtracted from the totals.
// Buffer creation during startup and streaming-in is not the same population as creation during
// gameplay, and a total that mixes them cannot answer the question this census exists for.
inline BufferCensusStats g_buffer_census_at_route_start {};
inline std::atomic<bool> g_buffer_census_route_started {false};

// Sampled at each create. Monotonic max, so it survives the periodic reports.
inline void NoteBufferMemory(BufferCensusStats& stats, uint64_t total_bytes, uint64_t live,
                             uint64_t leaps, uint64_t pad_bytes) {
	if (leaps != 0) {
		stats.leaps.fetch_add(leaps, std::memory_order_relaxed);
		stats.pad_bytes.fetch_add(pad_bytes, std::memory_order_relaxed);
	}
	auto peak = stats.peak_buffer_bytes.load(std::memory_order_relaxed);
	while (total_bytes > peak && !stats.peak_buffer_bytes.compare_exchange_weak(
	                                 peak, total_bytes, std::memory_order_relaxed)) {
	}
	if (total_bytes >= stats.peak_buffer_bytes.load(std::memory_order_relaxed)) {
		stats.live_buffers_at_peak.store(live, std::memory_order_relaxed);
	}
}

inline void NoteBufferDelete(uint64_t begin, uint64_t size, BufferDeleteReason reason,
                             RecentDeletions& deletions = g_recent_deletions) {
	deletions.Note(begin, begin + size, reason);
}

inline void AccumulateBufferCreate(BufferCensusStats& stats, uint64_t overlap_count,
                                   uint64_t overlap_bytes, bool expands_left, bool expands_right,
                                   uint64_t requested_begin, uint64_t requested_end,
                                   uint64_t union_size,
                                   RecentDeletions& deletions = g_recent_deletions) {
	const auto kind = ClassifyBufferCreate(overlap_count, expands_left, expands_right);

	stats.creates.fetch_add(1, std::memory_order_relaxed);
	stats.copied_bytes.fetch_add(overlap_bytes, std::memory_order_relaxed);

	if (overlap_count > 1) {
		stats.multi_overlap.fetch_add(1, std::memory_order_relaxed);
		stats.multi_overlap_copied_bytes.fetch_add(overlap_bytes, std::memory_order_relaxed);
	}
	auto observed_max = stats.max_overlap_count.load(std::memory_order_relaxed);
	while (overlap_count > observed_max &&
	       !stats.max_overlap_count.compare_exchange_weak(observed_max, overlap_count,
	                                                      std::memory_order_relaxed)) {
	}

	switch (kind) {
		case BufferCreateKind::Fresh: {
			stats.fresh.fetch_add(1, std::memory_order_relaxed);
			BufferDeleteReason reason = BufferDeleteReason::Merged;
			if (deletions.Find(requested_begin, requested_end, &reason)) {
				(reason == BufferDeleteReason::Evicted ? stats.fresh_after_evict
				                                       : stats.fresh_after_merge)
				    .fetch_add(1, std::memory_order_relaxed);
			}
			return;
		}
		case BufferCreateKind::Contained:
			stats.contained.fetch_add(1, std::memory_order_relaxed);
			return;
		case BufferCreateKind::ExpandsRight:
			stats.expands_right.fetch_add(1, std::memory_order_relaxed);
			break;
		case BufferCreateKind::ExpandsLeft:
			stats.expands_left.fetch_add(1, std::memory_order_relaxed);
			break;
		case BufferCreateKind::ExpandsBoth:
			stats.expands_both.fetch_add(1, std::memory_order_relaxed);
			break;
	}
	stats.growth_copied_bytes.fetch_add(overlap_bytes, std::memory_order_relaxed);
	stats.growth_old_bytes.fetch_add(overlap_bytes, std::memory_order_relaxed);
	stats.growth_new_bytes.fetch_add(union_size, std::memory_order_relaxed);
}

// Copies the current totals aside. Called when FrameStats consumes ROUTE_START.
inline void MarkBufferCensusRouteStart(BufferCensusStats& stats = g_buffer_census,
                                       BufferCensusStats& baseline =
                                           g_buffer_census_at_route_start) {
	const auto copy = [](std::atomic<uint64_t>& to, const std::atomic<uint64_t>& from) {
		to.store(from.load(std::memory_order_relaxed), std::memory_order_relaxed);
	};
	copy(baseline.creates, stats.creates);
	copy(baseline.fresh, stats.fresh);
	copy(baseline.expands_left, stats.expands_left);
	copy(baseline.expands_right, stats.expands_right);
	copy(baseline.expands_both, stats.expands_both);
	copy(baseline.contained, stats.contained);
	copy(baseline.copied_bytes, stats.copied_bytes);
	copy(baseline.growth_copied_bytes, stats.growth_copied_bytes);
	copy(baseline.growth_old_bytes, stats.growth_old_bytes);
	copy(baseline.growth_new_bytes, stats.growth_new_bytes);
	copy(baseline.multi_overlap, stats.multi_overlap);
	copy(baseline.multi_overlap_copied_bytes, stats.multi_overlap_copied_bytes);
	copy(baseline.fresh_after_evict, stats.fresh_after_evict);
	copy(baseline.fresh_after_merge, stats.fresh_after_merge);
	g_buffer_census_route_started.store(true, std::memory_order_relaxed);
}

// Printed every kBufferCensusInterval creates, and once more at exit regardless of the count.
inline constexpr uint64_t kBufferCensusInterval = 2000;

// One block of numbers over a chosen span. `since` is nullptr for whole-process totals, or the
// route-start baseline for gameplay only.
inline void PrintBufferCensusBlock(const char* label, const BufferCensusStats& stats,
                                   const BufferCensusStats* since, std::FILE* out = stdout) {
	const auto value = [&since](const std::atomic<uint64_t>& c,
	                            const std::atomic<uint64_t>& base) -> uint64_t {
		const auto now = c.load(std::memory_order_relaxed);
		if (since == nullptr) {
			return now;
		}
		const auto then = base.load(std::memory_order_relaxed);
		return now >= then ? now - then : 0;
	};
	const auto zero = BufferCensusStats {};
	const auto& b   = since != nullptr ? *since : zero;

	const auto creates = value(stats.creates, b.creates);
	if (creates == 0) {
		std::fprintf(out, "BufferCensus[%s]: no buffer creations\n", label);
		return;
	}
	const auto mb = [](uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); };

	const auto left   = value(stats.expands_left, b.expands_left);
	const auto right  = value(stats.expands_right, b.expands_right);
	const auto both   = value(stats.expands_both, b.expands_both);
	const auto growth = left + right + both;
	const auto old_b  = value(stats.growth_old_bytes, b.growth_old_bytes);
	const auto new_b  = value(stats.growth_new_bytes, b.growth_new_bytes);

	// Absolute first. A share cannot size this: a few very large joins can be a rounding error
	// in the count and all of the bytes.
	std::fprintf(out, "BufferCensus[%s]: creates=%llu copied=%.1f MB | growth: %llu creates, %.1f MB "
	            "copied (expands-left %llu right %llu both %llu), slack x%.2f\n",
	            label, static_cast<unsigned long long>(creates),
	            mb(value(stats.copied_bytes, b.copied_bytes)),
	            static_cast<unsigned long long>(growth),
	            mb(value(stats.growth_copied_bytes, b.growth_copied_bytes)),
	            static_cast<unsigned long long>(left), static_cast<unsigned long long>(right),
	            static_cast<unsigned long long>(both),
	            old_b > 0 ? static_cast<double>(new_b) / static_cast<double>(old_b) : 0.0);
	// The deletion-window figures are evidence that a buffer covering that range was deleted
	// earlier - nothing more. They do not show the deletion caused the recreation, and a MISS is
	// not evidence of absence: the range may have fallen out of the window, or no matching
	// deletion may ever have happened. Both readings stay open.
	std::fprintf(out,
	             "                  fresh=%llu (>=%llu over a range evicted earlier, >=%llu over a "
	            "range merged earlier; %zu-entry window) | contained=%llu | multi-overlap=%llu "
	            "(%.1f MB, max %llu joined)\n",
	            static_cast<unsigned long long>(value(stats.fresh, b.fresh)),
	            static_cast<unsigned long long>(value(stats.fresh_after_evict, b.fresh_after_evict)),
	            static_cast<unsigned long long>(value(stats.fresh_after_merge, b.fresh_after_merge)),
	            RecentDeletions::kSlots,
	            static_cast<unsigned long long>(value(stats.contained, b.contained)),
	            static_cast<unsigned long long>(value(stats.multi_overlap, b.multi_overlap)),
	            mb(value(stats.multi_overlap_copied_bytes, b.multi_overlap_copied_bytes)),
	            static_cast<unsigned long long>(
	                stats.max_overlap_count.load(std::memory_order_relaxed)));
	// Padding raises m_total_used_memory, and the GC thresholds key on that, so a leap can
	// provoke eviction and therefore MORE creates. Retained memory is reported for that reason,
	// not as a footnote.
	std::fprintf(out,
	             "                  leaps=%llu padding=%.1f MB | peak buffer memory %.1f MB "
	             "across %llu live buffers" "\n",
	             static_cast<unsigned long long>(value(stats.leaps, b.leaps)),
	             mb(value(stats.pad_bytes, b.pad_bytes)),
	             mb(stats.peak_buffer_bytes.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(
	                 stats.live_buffers_at_peak.load(std::memory_order_relaxed)));
	std::fflush(out);
}

// Whole-process totals plus, when the route was reached, gameplay on its own. Printed at exit
// even if fewer than kBufferCensusInterval creates happened - a short run must still report.
inline std::atomic<bool> g_buffer_census_reported {false};

// Printed from ~BufferCache, with std::atexit kept only as a backstop. The first capture relied
// on atexit alone and produced NO final summary at all, even though the renderer plainly tore
// down - the Vulkan pipeline cache saved itself in the same shutdown. Whatever ends the process
// does not run atexit handlers, so the report has to hang off a destructor that observably runs.
// The flag makes whichever path fires first the one that prints.
inline void ReportBufferCensusFinal(BufferCensusStats& stats = g_buffer_census,
                                    std::FILE* out = stdout) {
	if (g_buffer_census_reported.exchange(true, std::memory_order_relaxed)) {
		return;
	}
	std::fprintf(out, "\n=== BufferCensus FINAL ===\n");
	PrintBufferCensusBlock("all", stats, nullptr, out);
	if (g_buffer_census_route_started.load(std::memory_order_relaxed)) {
		// Boot and loading create buffers under a completely different pattern from gameplay,
		// so the route-scoped block is the one that answers the question.
		PrintBufferCensusBlock("route only", stats, &g_buffer_census_at_route_start, out);
	} else {
		std::fprintf(out, "BufferCensus[route only]: ROUTE_START never consumed; totals above include "
		            "boot and loading\n");
	}
	std::fprintf(out, "NOTE: copied bytes measure ACTIVITY, not frame-time cost. Attributing hitches to "
	            "this needs timing evidence these counters do not carry.\n");
	std::fflush(out);
}

inline void ReportBufferCensus(BufferCensusStats& stats = g_buffer_census) {
	const auto n = stats.creates.load(std::memory_order_relaxed);
	if (n == 0 || (n % kBufferCensusInterval) != 0) {
		return;
	}
	PrintBufferCensusBlock(g_buffer_census_route_started.load(std::memory_order_relaxed)
	                           ? "running, route active"
	                           : "running, pre-route",
	                       stats, nullptr);
}

} // namespace Libs::Graphics

#endif // INCLUDE_KYTY_GRAPHICS_RENDERER_CACHE_BUFFERCENSUS_H_
