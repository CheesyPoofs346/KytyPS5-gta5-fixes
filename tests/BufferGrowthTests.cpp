// Offline checks for the buffer-growth prototype (port of upstream 74a78f3), default OFF.
//
// The selection algorithm is driven here by a fake store holding the same invariant BufferCache
// maintains: ordered, non-overlapping ranges. That lets the range arithmetic - the part where a
// padded range can silently swallow a buffer nobody joined - be tested without a GPU.
//
// WHAT THESE COVER: the replacement range, which entries get folded in, page-table consistency
// after padding, the join-score lifecycle, and that OFF is byte-identical to the old behaviour.
//
// WHAT THEY DO NOT COVER, and no offline test here can: the actual vkCmdCopyBuffer, the memory
// tracker's dirty bits, and GPU retirement ordering. Those need a device. What IS established
// about them is structural and stated where each is checked below.

#include "common/emulatorConfig.h"
#include "graphics/host_gpu/renderer/cache/bufferGrowth.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

void Check(const char* test, bool ok, const std::string& message) {
	if (!ok) {
		std::printf("[FAIL]    %-46s %s\n", test, message.c_str());
		g_failures++;
	}
}
void Pass(const char* test, const std::string& detail) {
	std::printf("[ok]      %-46s %s\n", test, detail.c_str());
}

constexpr uint64_t kPage = 4096;

struct FakeBuffer {
	uint64_t             size  = 0;
	int                  score = 0;
	std::vector<uint8_t> bytes;   // stands in for device memory, so a join can be checked
};

// Ordered, non-overlapping - the invariant BufferCache keeps by construction.
using FakeStore = std::map<uint64_t, FakeBuffer>;

auto RangeOf(FakeStore& store) {
	return [&store](FakeStore::iterator it) {
		return BufferGrowthRange {it->first, it->first + it->second.size, it->second.score};
	};
}

BufferGrowthConfig Config(bool enabled, int threshold = 16, uint64_t leap_pages = 128) {
	BufferGrowthConfig c {};
	c.enabled            = enabled;
	c.threshold          = threshold;
	c.leap_size          = kPage * leap_pages;
	c.address_space_size = uint64_t {1} << 40u;
	c.min_address        = kPage * 2;
	return c;
}

void Insert(FakeStore& store, uint64_t begin, uint64_t size, int score, uint8_t fill) {
	FakeBuffer b;
	b.size  = size;
	b.score = score;
	b.bytes.assign(static_cast<size_t>(size), fill);
	store[begin] = std::move(b);
}

// Reproduces what CreateBuffer does with the plan: allocate the replacement, copy each joined
// buffer to its own offset, carry the score, retire the originals.
FakeBuffer ApplyPlan(FakeStore& store, const BufferOverlapPlan<FakeStore::iterator>& plan) {
	FakeBuffer replacement;
	replacement.size = plan.end - plan.begin;
	replacement.bytes.assign(static_cast<size_t>(replacement.size), 0);
	for (auto it = plan.first; it != plan.last;) {
		const auto current = it++;
		const auto offset  = current->first - plan.begin;
		for (uint64_t i = 0; i < current->second.size; i++) {
			replacement.bytes[static_cast<size_t>(offset + i)] = current->second.bytes[static_cast<size_t>(i)];
		}
		if (!plan.leaped) {
			replacement.score += current->second.score + 1;
		}
		store.erase(current);
	}
	store[plan.begin] = replacement;
	return store[plan.begin];
}

// --------------------------------------------------------------------------------------------

void TestDefaultOff() {
	const char* name = "growth is off by default";
	Check(name, !::Config::BufferGrowthEnabled(), "buffer_growth defaulted to true");
	if (!::Config::BufferGrowthEnabled()) {
		Pass(name, "default false; the join range is the plain union");
	}
}

// With the flag off the plan must be exactly the old union - no padding, same entries folded.
void TestOffMatchesPreviousBehaviour() {
	const char* name = "OFF reproduces the plain union";
	FakeStore   store;
	Insert(store, 0x10000, 4 * kPage, 99, 0xAA);   // score far past the threshold
	Insert(store, 0x20000, 4 * kPage, 99, 0xBB);

	const auto plan = ResolveOverlapsWithLeap(store, RangeOf(store), 0x11000, 0x21000,
	                                          Config(/*enabled=*/false));
	Check(name, !plan.leaped, "padding applied with the feature off");
	Check(name, plan.begin == 0x10000, "begin=" + std::to_string(plan.begin));
	Check(name, plan.end == 0x20000 + 4 * kPage, "end=" + std::to_string(plan.end));
	Check(name, plan.overlap_count == 2, "overlaps=" + std::to_string(plan.overlap_count));
	if (!plan.leaped && plan.begin == 0x10000 && plan.overlap_count == 2) {
		Pass(name, "union only, despite scores of 99 - the flag really gates it");
	}
}

void TestBelowThresholdDoesNotLeap() {
	const char* name = "score below the threshold does not pad";
	FakeStore   store;
	Insert(store, 0x10000, 4 * kPage, 3, 0xAA);
	const auto plan = ResolveOverlapsWithLeap(store, RangeOf(store), 0x11000, 0x15000,
	                                          Config(/*enabled=*/true));
	Check(name, !plan.leaped, "padded at score 3 with threshold 16");
	Check(name, plan.end == 0x15000, "end moved without a leap: " + std::to_string(plan.end));
	if (!plan.leaped) {
		Pass(name, "score 3 vs threshold 16, range is the plain union");
	}
}

// The whole point: a range joined repeatedly gets over-allocated.
void TestLeapPadsTheSideThatFired() {
	const char* name = "a repeatedly joined range is padded";
	FakeStore   store;
	// One buffer whose end is above the request -> expands_right fires.
	Insert(store, 0x100000, 16 * kPage, 20, 0xAA);
	const auto plan = ResolveOverlapsWithLeap(store, RangeOf(store), 0x100000, 0x104000,
	                                          Config(/*enabled=*/true));
	Check(name, plan.leaped, "no padding despite score 20 > threshold 16");
	Check(name, plan.expands_right, "expands_right did not fire");
	const auto expected_end = 0x100000 + 16 * kPage + 128 * kPage;
	Check(name, plan.end == expected_end,
	      "end=" + std::to_string(plan.end) + ", expected " + std::to_string(expected_end));
	Check(name, plan.begin == 0x100000, "left side moved when only the right fired");
	if (plan.leaped && plan.end == expected_end && plan.begin == 0x100000) {
		Pass(name, "padded 128 pages on the right only");
	}
}

// The failure mode that matters most: padding must not leave another buffer overlapping the
// replacement, or the page table would map one range to two buffers.
void TestPaddingSwallowsUncoveredNeighbours() {
	const char* name = "padding folds in what it now covers";
	FakeStore   store;
	Insert(store, 0x100000, 16 * kPage, 20, 0xAA);   // triggers the leap, expands right
	// Sits well beyond the request but inside the 128-page pad.
	Insert(store, 0x100000 + 80 * kPage, 4 * kPage, 0, 0xCC);

	const auto plan = ResolveOverlapsWithLeap(store, RangeOf(store), 0x100000, 0x104000,
	                                          Config(/*enabled=*/true));
	Check(name, plan.leaped, "did not leap");
	Check(name, plan.overlap_count == 2,
	      "neighbour inside the padded range was NOT folded in: overlaps=" +
	          std::to_string(plan.overlap_count));

	// And it must actually be removed from the store, not left overlapping.
	const auto replacement = ApplyPlan(store, plan);
	Check(name, store.size() == 1, "store still holds " + std::to_string(store.size()) +
	                                   " entries; ranges now overlap");
	// Its bytes must survive at the right offset.
	const auto offset = 80 * kPage;
	Check(name, replacement.bytes[static_cast<size_t>(offset)] == 0xCC,
	      "neighbour contents lost during the join");
	if (plan.overlap_count == 2 && store.size() == 1 &&
	    replacement.bytes[static_cast<size_t>(offset)] == 0xCC) {
		Pass(name, "neighbour 80 pages away folded in, contents intact at its offset");
	}
}

void TestLeftPaddingReanchors() {
	const char* name = "left padding re-anchors the first entry";
	FakeStore   store;
	// A buffer starting below the request -> expands_left fires.
	Insert(store, 0x200000, 16 * kPage, 20, 0xAA);
	// Below that, inside the 128-page left pad.
	Insert(store, 0x200000 - 100 * kPage, 4 * kPage, 0, 0xDD);

	const auto plan = ResolveOverlapsWithLeap(store, RangeOf(store), 0x204000, 0x205000,
	                                          Config(/*enabled=*/true));
	Check(name, plan.leaped, "did not leap");
	Check(name, plan.expands_left, "expands_left did not fire");
	Check(name, plan.begin <= 0x200000 - 100 * kPage,
	      "begin=" + std::to_string(plan.begin) + " did not reach the uncovered neighbour");
	Check(name, plan.overlap_count == 2,
	      "left neighbour not folded in: overlaps=" + std::to_string(plan.overlap_count));

	const auto replacement = ApplyPlan(store, plan);
	Check(name, store.size() == 1, "left neighbour left overlapping the replacement");
	if (plan.leaped && plan.overlap_count == 2 && store.size() == 1) {
		Pass(name, "first re-anchored below the pad, neighbour folded in");
	}
}

// Data preservation across a join: every joined buffer's bytes land at its own offset.
void TestJoinPreservesData() {
	const char* name = "joined data lands at the right offsets";
	FakeStore   store;
	Insert(store, 0x300000, 2 * kPage, 0, 0x11);
	Insert(store, 0x300000 + 4 * kPage, 2 * kPage, 0, 0x22);

	const auto plan = ResolveOverlapsWithLeap(store, RangeOf(store), 0x300000,
	                                          0x300000 + 6 * kPage, Config(/*enabled=*/true));
	const auto replacement = ApplyPlan(store, plan);
	const bool first_ok  = replacement.bytes[0] == 0x11 &&
	                      replacement.bytes[static_cast<size_t>(2 * kPage - 1)] == 0x11;
	const bool gap_ok    = replacement.bytes[static_cast<size_t>(2 * kPage)] == 0;
	const bool second_ok = replacement.bytes[static_cast<size_t>(4 * kPage)] == 0x22 &&
	                       replacement.bytes[static_cast<size_t>(6 * kPage - 1)] == 0x22;
	Check(name, first_ok, "first buffer contents wrong");
	Check(name, gap_ok, "the untouched gap was overwritten");
	Check(name, second_ok, "second buffer contents wrong or misaligned");
	if (first_ok && gap_ok && second_ok) {
		Pass(name, "both payloads intact, the gap between them left untouched");
	}
}

// Score lifecycle: it climbs across joins, and the create that leaps resets it so the padded
// buffer cannot immediately trigger another leap.
void TestScoreLifecycle() {
	const char* name = "join score climbs, then resets on a leap";
	FakeStore   store;
	Insert(store, 0x400000, 4 * kPage, 0, 0xAA);

	int  observed_score = 0;
	bool leaped_once    = false;
	for (int i = 0; i < 30 && !leaped_once; i++) {
		const auto it    = store.begin();
		const auto begin = it->first;
		const auto end   = begin + it->second.size;
		const auto plan  = ResolveOverlapsWithLeap(store, RangeOf(store), begin, end + kPage,
		                                           Config(/*enabled=*/true));
		const auto replacement = ApplyPlan(store, plan);
		observed_score         = replacement.score;
		leaped_once            = plan.leaped;
	}
	Check(name, leaped_once, "never leaped after 30 successive joins");
	Check(name, observed_score == 0,
	      "score after the leaping create=" + std::to_string(observed_score) + ", expected 0");
	if (leaped_once && observed_score == 0) {
		Pass(name, "leaped within 30 joins and reset to 0, so the next leap must be re-earned");
	}
}

void TestLeapRespectsBounds() {
	const char* name = "padding respects address-space bounds";
	// Right edge: a range near the top of the address space must not pad past it.
	{
		FakeStore  store;
		auto       config = Config(/*enabled=*/true);
		const auto top    = config.address_space_size;
		Insert(store, top - 8 * kPage, 4 * kPage, 20, 0xAA);
		const auto plan = ResolveOverlapsWithLeap(store, RangeOf(store), top - 8 * kPage,
		                                          top - 6 * kPage, config);
		Check(name, plan.end <= config.address_space_size,
		      "end=" + std::to_string(plan.end) + " past the address space");
	}
	// Left edge: must not pad below the minimum.
	{
		FakeStore  store;
		auto       config = Config(/*enabled=*/true);
		Insert(store, kPage * 3, 4 * kPage, 20, 0xAA);
		const auto plan =
		    ResolveOverlapsWithLeap(store, RangeOf(store), kPage * 5, kPage * 6, config);
		Check(name, plan.begin >= config.min_address,
		      "begin=" + std::to_string(plan.begin) + " below the minimum");
	}
	if (g_failures == 0) {
		Pass(name, "clamped at both edges rather than wrapping");
	}
}

// Empty store and a request touching nothing: the degenerate cases that kill looks-right code.
void TestEmptyAndNoOverlap() {
	const char* name = "empty store and no-overlap requests";
	{
		FakeStore  store;
		const auto plan = ResolveOverlapsWithLeap(store, RangeOf(store), 0x1000, 0x2000,
		                                          Config(/*enabled=*/true));
		Check(name, plan.overlap_count == 0, "overlaps found in an empty store");
		Check(name, plan.begin == 0x1000 && plan.end == 0x2000, "empty store changed the range");
		Check(name, plan.first == plan.last, "non-empty join span in an empty store");
	}
	{
		FakeStore store;
		Insert(store, 0x900000, 4 * kPage, 99, 0xAA);
		const auto plan = ResolveOverlapsWithLeap(store, RangeOf(store), 0x100000, 0x101000,
		                                          Config(/*enabled=*/true));
		Check(name, plan.overlap_count == 0, "unrelated buffer was joined");
		Check(name, !plan.leaped, "leaped with nothing to join");
		Check(name, plan.begin == 0x100000 && plan.end == 0x101000, "range widened with no overlap");
	}
	if (g_failures == 0) {
		Pass(name, "no overlap, no padding, no widening");
	}
}

// Ordering and non-overlap must hold after a sequence of random-ish joins, since every later
// lookup depends on that invariant.
void TestStoreInvariantHolds() {
	const char* name = "store stays ordered and non-overlapping";
	FakeStore   store;
	for (int i = 0; i < 12; i++) {
		Insert(store, 0x500000 + static_cast<uint64_t>(i) * 32 * kPage, 8 * kPage, i, 0xA0);
	}
	for (int i = 0; i < 40; i++) {
		const uint64_t begin = 0x500000 + static_cast<uint64_t>((i * 7) % 12) * 32 * kPage;
		const auto     plan  = ResolveOverlapsWithLeap(store, RangeOf(store), begin,
		                                               begin + 10 * kPage, Config(true));
		ApplyPlan(store, plan);
	}
	uint64_t previous_end = 0;
	bool     ok           = true;
	for (const auto& [begin, buffer]: store) {
		if (begin < previous_end) {
			ok = false;
			Check(name, false,
			      "overlap at " + std::to_string(begin) + " after " + std::to_string(previous_end));
			break;
		}
		previous_end = begin + buffer.size;
	}
	if (ok) {
		Pass(name, std::to_string(store.size()) + " entries after 40 joins, none overlapping");
	}
}

} // namespace

int main() {
	std::printf("Buffer growth (upstream 74a78f3 port):\n\n");
	TestDefaultOff();
	TestOffMatchesPreviousBehaviour();
	TestBelowThresholdDoesNotLeap();
	TestLeapPadsTheSideThatFired();
	TestPaddingSwallowsUncoveredNeighbours();
	TestLeftPaddingReanchors();
	TestJoinPreservesData();
	TestScoreLifecycle();
	TestLeapRespectsBounds();
	TestEmptyAndNoOverlap();
	TestStoreInvariantHolds();
	if (g_failures != 0) {
		std::printf("\n%d buffer growth check(s) FAILED\n", g_failures);
		return 1;
	}
	std::printf("\nall buffer growth checks passed\n");
	std::printf("\nNOT covered offline: the real vkCmdCopyBuffer, memory-tracker dirty bits, and\n"
	            "GPU retirement ordering. Retirement is unchanged - joined buffers still go\n"
	            "through DeleteBuffer, which defers destruction while a batch is open - but that\n"
	            "is a structural argument, not a test.\n");
	return 0;
}
