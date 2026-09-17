// Offline checks for the buffer-creation census.
//
// The census exists to answer one question with a run: does this workload repeatedly recreate
// overlapping buffers? These checks verify the classifier and the counters say what they claim,
// so a capture is not wasted on a miscounted category.
//
// The buckets mirror upstream 74a78f3 "grow repeatedly joined buffer ranges", which keys on
// DIRECTION (expands_left / expands_right, padding whichever side grew) and accumulates its
// StreamScore across every overlap in a join. So leftward growth and multi-buffer growth are
// first-class here, not excluded.
//
// NOT tested here: that CreateBuffer is reached in a real frame, or what the real distribution
// looks like. That is what the capture is for.

#include "common/emulatorConfig.h"
#include "graphics/host_gpu/renderer/cache/bufferCensus.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

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

const char* KindName(BufferCreateKind kind) {
	switch (kind) {
		case BufferCreateKind::Fresh: return "Fresh";
		case BufferCreateKind::ExpandsRight: return "ExpandsRight";
		case BufferCreateKind::ExpandsLeft: return "ExpandsLeft";
		case BufferCreateKind::ExpandsBoth: return "ExpandsBoth";
		case BufferCreateKind::Contained: return "Contained";
	}
	return "?";
}

void ExpectKind(const char* test, BufferCreateKind got, BufferCreateKind want,
                const char* scenario) {
	Check(test, got == want,
	      std::string(scenario) + ": got " + KindName(got) + ", want " + KindName(want));
}

void TestDefaultOff() {
	const char* name = "census is off by default";
	Check(name, !Config::BufferCensusEnabled(), "buffer_census defaulted to true");
	if (!Config::BufferCensusEnabled()) {
		Pass(name, "default false, so nothing is counted or printed unless asked for");
	}
}

// Direction, because that is the trigger upstream acts on.
void TestClassification() {
	const char* name = "growth direction is classified";
	ExpectKind(name, ClassifyBufferCreate(0, false, false), BufferCreateKind::Fresh,
	           "no overlapping buffer");
	ExpectKind(name, ClassifyBufferCreate(1, false, true), BufferCreateKind::ExpandsRight,
	           "old buffer ends past the running end");
	ExpectKind(name, ClassifyBufferCreate(1, true, false), BufferCreateKind::ExpandsLeft,
	           "old buffer starts below the running begin");
	ExpectKind(name, ClassifyBufferCreate(1, true, true), BufferCreateKind::ExpandsBoth,
	           "old buffer sticks out both sides");
	ExpectKind(name, ClassifyBufferCreate(1, false, false), BufferCreateKind::Contained,
	           "overlap exists but nothing widened");
	// Multi-overlap is orthogonal: a 3-buffer join that only grew right is still GrowRight, and
	// the multi_overlap counter records the fan-in separately.
	ExpectKind(name, ClassifyBufferCreate(3, false, true), BufferCreateKind::ExpandsRight,
	           "three buffers joined, expanding right");
	ExpectKind(name, ClassifyBufferCreate(3, true, true), BufferCreateKind::ExpandsBoth,
	           "three buffers joined, expanding both ways");
	if (g_failures == 0) {
		Pass(name, "fresh / left / right / both / contained, single and multi overlap");
	}
}

// A guest walking a region UPWARD in small steps.
void TestRightwardGrowth() {
	const char*       name = "upward walk registers as a join";
	BufferCensusStats stats;
	RecentDeletions   deletions;
	uint64_t          begin = 0x100000;
	uint64_t          size  = 0x1000;

	// The guest walks UPWARD: each request starts inside the buffer it already has and runs past
	// its end. The old buffer therefore BEGINS BELOW the request, so the upstream predicate that
	// fires is expands_left. This is the naming trap the header warns about, and modelling it the
	// intuitive way is what made the first version of this test fail.
	constexpr int kSteps = 16;
	uint64_t      copied = 0;
	for (int i = 0; i < kSteps; i++) {
		const uint64_t requested_begin = begin + size - 0x100;
		const uint64_t requested_end   = begin + size + 0x1000;
		const bool     expands_left    = begin < requested_begin;
		const bool     expands_right   = begin + size > requested_end;
		AccumulateBufferCreate(stats, 1, size, expands_left, expands_right, requested_begin,
		                       requested_end, requested_end - begin, deletions);
		copied += size;
		size = requested_end - begin;
	}
	Check(name, stats.expands_left.load() == kSteps,
	      "expands_left=" + std::to_string(stats.expands_left.load()));
	Check(name, stats.expands_right.load() == 0 && stats.fresh.load() == 0,
	      "upward walk leaked into another bucket");
	Check(name, stats.growth_copied_bytes.load() == copied,
	      "growth copied bytes=" + std::to_string(stats.growth_copied_bytes.load()));
	if (stats.expands_left.load() == kSteps && stats.growth_copied_bytes.load() == copied) {
		Pass(name, std::to_string(kSteps) + " growths copied " + std::to_string(copied / 1024) +
		               " KB to reach a " + std::to_string(size / 1024) + " KB buffer");
	}
}

// The same walk DOWNWARD. Upstream pads this side too, so it must not be filed as something else.
void TestLeftwardGrowth() {
	const char*       name = "downward walk registers as a join";
	BufferCensusStats stats;
	RecentDeletions   deletions;
	uint64_t          begin = 0x400000;
	uint64_t          end   = 0x401000;

	// The mirror image: walking DOWNWARD, the old buffer ENDS ABOVE the request, so expands_right
	// is what fires. Both walks must land in growth, and neither in Fresh.
	constexpr int kSteps = 12;
	uint64_t      copied = 0;
	for (int i = 0; i < kSteps; i++) {
		const uint64_t requested_begin = begin - 0x1000;
		const uint64_t requested_end   = begin + 0x100;
		const bool     expands_left    = begin < requested_begin;
		const bool     expands_right   = end > requested_end;
		AccumulateBufferCreate(stats, 1, end - begin, expands_left, expands_right, requested_begin,
		                       requested_end, end - requested_begin, deletions);
		copied += end - begin;
		begin = requested_begin;
	}
	Check(name, stats.expands_right.load() == kSteps,
	      "expands_right=" + std::to_string(stats.expands_right.load()));
	Check(name, stats.expands_left.load() == 0, "downward walk misfiled");
	Check(name, stats.growth_copied_bytes.load() == copied,
	      "growth copied bytes=" + std::to_string(stats.growth_copied_bytes.load()));
	if (stats.expands_right.load() == kSteps && stats.expands_left.load() == 0) {
		Pass(name, std::to_string(kSteps) + " downward growths, " +
		               std::to_string(copied / 1024) + " KB copied, none misfiled");
	}
}

// A request landing in the gap between two buffers joins both and grows both ways.
void TestMultiBufferGrowth() {
	const char*       name = "multi-buffer joins are recorded";
	BufferCensusStats stats;
	RecentDeletions   deletions;
	// Three buffers joined; union runs from below the request to above it.
	AccumulateBufferCreate(stats, 3, 0x6000, /*expands_left=*/true, /*expands_right=*/true,
	                       0x201000, 0x202000, 0x6000, deletions);
	Check(name, stats.expands_both.load() == 1, "not classified as ExpandsBoth");
	Check(name, stats.multi_overlap.load() == 1,
	      "multi_overlap=" + std::to_string(stats.multi_overlap.load()));
	Check(name, stats.multi_overlap_copied_bytes.load() == 0x6000, "multi-overlap bytes wrong");
	Check(name, stats.max_overlap_count.load() == 3,
	      "max_overlap_count=" + std::to_string(stats.max_overlap_count.load()));
	Check(name, stats.growth_copied_bytes.load() == 0x6000,
	      "multi-buffer join not attributed to growth");
	if (g_failures == 0) {
		Pass(name, "3-buffer join counted as ExpandsBoth AND as multi-overlap, bytes in both");
	}
}

// Absolute bytes, not just counts: a few expensive joins can dominate the cost while being a
// rounding error in the create count.
void TestSharesDoNotHideCost() {
	const char*       name = "rare expensive growth stays visible";
	BufferCensusStats stats;
	RecentDeletions   deletions;
	for (int i = 0; i < 999; i++) {
		const uint64_t at = 0x800000 + static_cast<uint64_t>(i) * 0x10000;
		AccumulateBufferCreate(stats, 0, 0, false, false, at, at + 0x1000, 0x1000, deletions);
	}
	// One growth copying 256 MB - 0.1% of creates, but all of the copying.
	constexpr uint64_t kBig = 256ull * 1024 * 1024;
	AccumulateBufferCreate(stats, 1, kBig, /*expands_left=*/false, /*expands_right=*/true,
	                       0x10000000, 0x10001000, 0x10000000, deletions);
	const auto growth_share = 100.0 * static_cast<double>(stats.expands_right.load()) /
	                          static_cast<double>(stats.creates.load());
	Check(name, stats.growth_copied_bytes.load() == kBig, "growth bytes lost");
	Check(name, stats.copied_bytes.load() == kBig, "total copied bytes lost");
	Check(name, growth_share < 1.0, "test did not actually build a rare-growth case");
	if (stats.growth_copied_bytes.load() == kBig && growth_share < 1.0) {
		Pass(name, "growth is 0.1% of creates but 100% of the 256 MB copied");
	}
}

// Fresh does not mean unavoidable: a range can be evicted and rebuilt.
void TestFreshAfterDeletionIsDistinguished() {
	const char*       name = "fresh-after-deletion is distinguished";
	BufferCensusStats stats;
	RecentDeletions   deletions;

	NoteBufferDelete(0x300000, 0x2000, BufferDeleteReason::Evicted, deletions);
	NoteBufferDelete(0x500000, 0x2000, BufferDeleteReason::Merged, deletions);

	// rebuilt after evict, rebuilt after merge, genuinely new
	AccumulateBufferCreate(stats, 0, 0, false, false, 0x300000, 0x301000, 0x1000, deletions);
	AccumulateBufferCreate(stats, 0, 0, false, false, 0x500800, 0x501000, 0x800, deletions);
	AccumulateBufferCreate(stats, 0, 0, false, false, 0x900000, 0x901000, 0x1000, deletions);

	Check(name, stats.fresh.load() == 3, "fresh=" + std::to_string(stats.fresh.load()));
	Check(name, stats.fresh_after_evict.load() == 1,
	      "fresh_after_evict=" + std::to_string(stats.fresh_after_evict.load()));
	Check(name, stats.fresh_after_merge.load() == 1,
	      "fresh_after_merge=" + std::to_string(stats.fresh_after_merge.load()));
	if (stats.fresh.load() == 3 && stats.fresh_after_evict.load() == 1 &&
	    stats.fresh_after_merge.load() == 1) {
		Pass(name, "2 of 3 fresh creates identified as rebuilds, 1 genuinely new");
	}
}

// The deletion window is bounded, so a hit is exact but a miss is not proof of novelty.
void TestDeletionWindowIsALowerBound() {
	const char*     name = "deletion window is a lower bound";
	RecentDeletions deletions;
	// Overflow the window, then look for the very first range recorded.
	for (size_t i = 0; i < RecentDeletions::kSlots + 8; i++) {
		NoteBufferDelete(0x1000000 + i * 0x10000, 0x1000, BufferDeleteReason::Evicted, deletions);
	}
	const bool found_first = deletions.Find(0x1000000, 0x1001000, nullptr);
	const bool found_last =
	    deletions.Find(0x1000000 + (RecentDeletions::kSlots + 7) * 0x10000,
	                   0x1000000 + (RecentDeletions::kSlots + 7) * 0x10000 + 0x1000, nullptr);
	Check(name, !found_first, "evicted-out entry still reported; window is not bounded");
	Check(name, found_last, "most recent deletion was not found");
	if (!found_first && found_last) {
		Pass(name, "oldest entries fall out, so a miss cannot be read as 'never existed'");
	}
}

// The periodic line only fires every kBufferCensusInterval creates. A capture that ends with
// fewer than that must still report, or a short run yields nothing at all. Verified by capturing
// the actual output through the sink parameter - an earlier version redirected stdout with
// freopen, which silently swallowed every other test line when the suite ran under a pipe.
void TestFinalSummaryOnShortRun() {
	const char* name = "short run still prints a final summary";
	BufferCensusStats stats;
	g_buffer_census_route_started.store(false);
	// Well under the 2000-create reporting interval.
	for (int i = 0; i < 7; i++) {
		const uint64_t at = 0x40000 + static_cast<uint64_t>(i) * 0x10000;
		AccumulateBufferCreate(stats, 0, 0, false, false, at, at + 0x1000, 0x1000);
	}

	std::FILE* sink = std::tmpfile();
	if (sink == nullptr) {
		Check(name, false, "could not open a temporary file for the capture");
		return;
	}
	ReportBufferCensusFinal(stats, sink);
	std::rewind(sink);
	char   buffer[4096];
	size_t read  = std::fread(buffer, 1, sizeof(buffer) - 1, sink);
	buffer[read] = ' ';
	std::fclose(sink);
	const std::string captured(buffer);

	const bool has_header  = captured.find("BufferCensus FINAL") != std::string::npos;
	const bool has_creates = captured.find("creates=7") != std::string::npos;
	const bool has_note    = captured.find("not frame-time cost") != std::string::npos;
	const bool says_boot   = captured.find("ROUTE_START never consumed") != std::string::npos;
	Check(name, has_header, "no FINAL header in captured output");
	Check(name, has_creates, "7 creates were not reported");
	Check(name, has_note, "activity-vs-cost caveat missing from the summary");
	Check(name, says_boot, "did not state the totals include boot when no route ran");
	if (has_header && has_creates && has_note && says_boot) {
		Pass(name, "7 creates reported at exit despite the 2000-create interval");
	}
}

// With a route baseline set, the final report must also give gameplay on its own.
void TestRouteScopedSummary() {
	const char*       name = "gameplay is reported apart from boot";
	BufferCensusStats stats;
	BufferCensusStats baseline;
	g_buffer_census_route_started.store(false);

	for (int i = 0; i < 5; i++) {   // boot
		const uint64_t at = 0x40000 + static_cast<uint64_t>(i) * 0x10000;
		AccumulateBufferCreate(stats, 0, 0, false, false, at, at + 0x1000, 0x1000);
	}
	MarkBufferCensusRouteStart(stats, baseline);
	for (int i = 0; i < 3; i++) {   // gameplay
		const uint64_t at = 0x900000 + static_cast<uint64_t>(i) * 0x10000;
		AccumulateBufferCreate(stats, 0, 0, false, false, at, at + 0x1000, 0x1000);
	}

	std::FILE* sink = std::tmpfile();
	if (sink == nullptr) {
		Check(name, false, "could not open a temporary file for the capture");
		return;
	}
	PrintBufferCensusBlock("all", stats, nullptr, sink);
	PrintBufferCensusBlock("route only", stats, &baseline, sink);
	std::rewind(sink);
	char   buffer[4096];
	size_t read  = std::fread(buffer, 1, sizeof(buffer) - 1, sink);
	buffer[read] = ' ';
	std::fclose(sink);
	const std::string captured(buffer);

	const bool all_total   = captured.find("[all]: creates=8") != std::string::npos;
	const bool route_only  = captured.find("[route only]: creates=3") != std::string::npos;
	Check(name, all_total, "whole-process total was not 8");
	Check(name, route_only, "route-scoped count was not 3; boot leaked into gameplay");
	if (all_total && route_only) {
		Pass(name, "8 creates overall, 3 after ROUTE_START");
	}
}

} // namespace

int main() {
	std::printf("Buffer creation census:\n\n");
	TestFinalSummaryOnShortRun();
	TestRouteScopedSummary();
	TestDefaultOff();
	TestClassification();
	TestRightwardGrowth();
	TestLeftwardGrowth();
	TestMultiBufferGrowth();
	TestSharesDoNotHideCost();
	TestFreshAfterDeletionIsDistinguished();
	TestDeletionWindowIsALowerBound();
	if (g_failures != 0) {
		std::printf("\n%d buffer census check(s) FAILED\n", g_failures);
		return 1;
	}
	std::printf("\nall buffer census checks passed\n");
	return 0;
}
