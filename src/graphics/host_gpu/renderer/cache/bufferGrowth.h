#ifndef INCLUDE_KYTY_GRAPHICS_RENDERER_CACHE_BUFFERGROWTH_H_
#define INCLUDE_KYTY_GRAPHICS_RENDERER_CACHE_BUFFERGROWTH_H_

// Port of upstream 74a78f3 "renderer: grow repeatedly joined buffer ranges", default OFF.
//
// Measured motivation (capture 2026-09-08): 31.8% of buffer creations were joins that widened an
// existing range, and those joins accounted for 96.9% of all bytes copied between buffers.
// Upstream's answer is to over-allocate a replacement once a range has been joined repeatedly, so
// later requests land in bounds and never reach CreateBuffer at all.
//
// The selection is templated on the map so it can be driven by a fake store offline. Without that
// the leap could only be tested by launching the game, and the failure mode that matters most -
// a padded range silently swallowing a buffer that was never joined - is precisely a
// range-arithmetic bug.

#include <cstdint>
#include <iterator>

namespace Libs::Graphics {

struct BufferGrowthRange {
	uint64_t begin = 0;
	uint64_t end   = 0;
	int      score = 0;
};

struct BufferGrowthConfig {
	bool enabled = false;
	// Accumulated join score across the overlaps before padding is applied. Upstream: 16.
	int threshold = 16;
	// How far to over-allocate on the side that fired. Upstream: CACHING_PAGESIZE * 128.
	uint64_t leap_size = 0;
	// Hard bounds. Upstream clamps the right side to the page table's address space and refuses
	// to move the left side below two pages.
	uint64_t address_space_size = 0;
	uint64_t min_address        = 0;
};

template <class Iterator>
struct BufferOverlapPlan {
	Iterator first {};
	Iterator last {};
	uint64_t begin = 0;
	uint64_t end   = 0;
	// True when padding was applied. Upstream stops accumulating the join score into the
	// replacement in this case, so a leaped buffer must re-earn the next leap rather than
	// inheriting a score that already triggered one.
	bool leaped = false;
	// Diagnostics for the A/B; not used by the algorithm.
	// Bytes added purely by padding, i.e. beyond the union the old code would have produced.
	// Padding raises m_total_used_memory, which drives the GC thresholds, so a leap can provoke
	// eviction and therefore MORE creates. That feedback is why this is measured, not assumed.
	uint64_t pad_bytes     = 0;
	uint64_t overlap_count = 0;
	uint64_t overlap_bytes = 0;
	bool     expands_left  = false;
	bool     expands_right = false;
};

// `buffers` is an ordered map of range-start -> value; `range(it)` yields that entry's
// [begin, end) and its accumulated join score. Entries must be non-overlapping and ordered, which
// is the invariant BufferCache maintains by construction.
//
// Returns the half-open iterator span of entries to fold in, and the range the replacement must
// cover. With config.enabled false this is exactly the previous behaviour: the union of the
// request and every overlapping entry.
template <class Map, class GetRange>
BufferOverlapPlan<typename Map::iterator> ResolveOverlapsWithLeap(Map& buffers, GetRange range,
                                                                  uint64_t requested_begin,
                                                                  uint64_t requested_end,
                                                                  const BufferGrowthConfig& config) {
	using Iterator = typename Map::iterator;

	// The first entry that could overlap `address`: lower_bound, stepped back one when the
	// preceding entry still covers it.
	const auto find_first = [&](uint64_t address) -> Iterator {
		auto first = buffers.lower_bound(address);
		if (first != buffers.begin()) {
			const auto previous = std::prev(first);
			if (range(previous).end > address) {
				first = previous;
			}
		}
		return first;
	};

	BufferOverlapPlan<Iterator> plan;
	plan.begin = requested_begin;
	plan.end   = requested_end;
	plan.first = find_first(plan.begin);

	int  score = 0;
	auto last  = plan.first;
	// `plan.end` is the loop bound AND is extended by the leap, so entries brought into range by
	// a rightward pad are folded in by the same loop rather than being left overlapping the
	// replacement. That is load-bearing, not incidental.
	for (; last != buffers.end() && last->first < plan.end; ++last) {
		const auto entry         = range(last);
		const bool expands_left  = entry.begin < plan.begin;
		const bool expands_right = entry.end > plan.end;

		plan.expands_left |= expands_left;
		plan.expands_right |= expands_right;

		plan.begin = entry.begin < plan.begin ? entry.begin : plan.begin;
		plan.end   = entry.end > plan.end ? entry.end : plan.end;

		if (!config.enabled || plan.leaped) {
			continue;
		}
		score += entry.score;
		if (score <= config.threshold) {
			continue;
		}
		plan.leaped = true;
		if (expands_right) {
			const auto room = config.address_space_size > plan.end
			                      ? config.address_space_size - plan.end
			                      : uint64_t {0};
			const auto added = config.leap_size < room ? config.leap_size : room;
			plan.end += added;
			plan.pad_bytes += added;
		}
		if (expands_left && plan.begin > config.min_address) {
			const auto room = plan.begin - config.min_address;
			const auto added = config.leap_size < room ? config.leap_size : room;
			plan.begin -= added;
			plan.pad_bytes += added;
			// Padding left can uncover entries below the previous start; re-anchor so they are
			// folded in instead of being left overlapping the replacement.
			plan.first = find_first(plan.begin);
			if (plan.first != buffers.end() && plan.first->first < plan.begin) {
				plan.begin = plan.first->first;
			}
		}
	}
	plan.last = last;

	// Counted over the FINAL span, not accumulated in the loop above. A leftward pad re-anchors
	// `first` behind the cursor, so entries it picks up are never visited by the forward loop -
	// accumulating there undercounted exactly the joins the padding causes. The A/B reports these
	// numbers, so they have to match what the caller will actually copy and retire.
	for (auto it = plan.first; it != plan.last; ++it) {
		const auto entry = range(it);
		plan.overlap_count++;
		plan.overlap_bytes += entry.end - entry.begin;
	}
	return plan;
}

} // namespace Libs::Graphics

#endif // INCLUDE_KYTY_GRAPHICS_RENDERER_CACHE_BUFFERGROWTH_H_
