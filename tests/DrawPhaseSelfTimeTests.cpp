// Focused tests for DrawPhaseTimer's exclusive (self) time accounting.
//
// The accumulator determines nesting from ACTUAL TIMER LIFETIMES at run time, not from the
// DrawPhaseIsChild() table, which describes intent and was measurably wrong. These tests pin the
// three shapes that matter: nested, recursive (same id re-entered), and repeated (same id used
// several times in sequence).

#include "common/emulatorConfig.h"
#include "graphics/host_gpu/renderer/drawProfile.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace {

using Libs::Graphics::DrawPhase;
using Libs::Graphics::DrawPhaseTimer;
using Libs::Graphics::g_draw_profile;

int g_failures = 0;

void Check(const char* test, bool ok, const std::string& message) {
	if (!ok) {
		std::printf("[FAIL]    %-44s %s\n", test, message.c_str());
		g_failures++;
	}
}

uint64_t Incl(DrawPhase p) {
	return g_draw_profile.cycles[static_cast<size_t>(p)];
}
uint64_t Self(DrawPhase p) {
	return g_draw_profile.self_cycles[static_cast<size_t>(p)];
}

void Reset() {
	g_draw_profile.cycles       = {};
	g_draw_profile.self_cycles  = {};
	g_draw_profile.child_cycles = 0;
	g_draw_profile.active       = true;
}

// Burn a bounded amount of real time so the cycle deltas are non-zero and ordered.
void Spin(uint64_t cycles) {
	const auto start = Libs::Graphics::DrawProfileReadCycles();
	while (Libs::Graphics::DrawProfileReadCycles() - start < cycles) {
	}
}

constexpr uint64_t kUnit = 200000;

// A zone containing another zone: the outer's SELF must exclude the inner, while both inclusive
// totals stay whole.
void TestNested() {
	const char* test = "Nested: outer SELF excludes inner";
	Reset();
	{
		DrawPhaseTimer outer(DrawPhase::Preamble);
		Spin(kUnit);
		{
			DrawPhaseTimer inner(DrawPhase::RenderState);
			Spin(kUnit * 3);
		}
		Spin(kUnit);
	}
	Check(test, Incl(DrawPhase::Preamble) > Incl(DrawPhase::RenderState),
	      "outer inclusive should contain inner");
	Check(test, Self(DrawPhase::RenderState) == Incl(DrawPhase::RenderState),
	      "a leaf zone's self must equal its inclusive");
	Check(test, Self(DrawPhase::Preamble) < Incl(DrawPhase::Preamble),
	      "outer self must be less than outer inclusive");
	// Outer self is roughly the two 1-unit spins; inner is 3 units. Self must be well under it.
	Check(test, Self(DrawPhase::Preamble) < Incl(DrawPhase::RenderState),
	      "outer self should be smaller than the inner zone it contains");
	std::printf("[host]    %-44s ok\n", test);
}

// The same phase id re-entered inside itself. Inclusive double counts by construction; the
// guarded subtraction must keep self sane and never wrap.
void TestRecursive() {
	const char* test = "Recursive: same id nested in itself";
	Reset();
	{
		DrawPhaseTimer outer(DrawPhase::Bindings);
		Spin(kUnit);
		{
			DrawPhaseTimer inner(DrawPhase::Bindings);
			Spin(kUnit * 2);
		}
		Spin(kUnit);
	}
	const auto self = Self(DrawPhase::Bindings);
	const auto incl = Incl(DrawPhase::Bindings);
	Check(test, self <= incl, "self must never exceed inclusive");
	Check(test, self < (uint64_t {1} << 62u), "self must not have wrapped");
	// Inner contributes its whole elapsed as self; outer contributes only the part outside it.
	Check(test, self > 0, "recursive self must still accumulate");
	std::printf("[host]    %-44s ok  (incl=%llu self=%llu)\n", test,
	            static_cast<unsigned long long>(incl), static_cast<unsigned long long>(self));
}

// The same id opened and closed several times in sequence, not nested. Every occurrence must add
// to both totals, and self must equal inclusive because nothing is nested inside.
void TestRepeated() {
	const char* test = "Repeated: same id in sequence";
	Reset();
	for (int i = 0; i < 4; i++) {
		DrawPhaseTimer timer(DrawPhase::VertexIndex);
		Spin(kUnit);
	}
	Check(test, Self(DrawPhase::VertexIndex) == Incl(DrawPhase::VertexIndex),
	      "sequential leaf zones: self must equal inclusive");
	Check(test, Incl(DrawPhase::VertexIndex) > kUnit * 3,
	      "all four occurrences must accumulate");
	std::printf("[host]    %-44s ok\n", test);
}

// Sibling zones inside one parent: the parent's self must exclude BOTH.
void TestTwoChildren() {
	const char* test = "Two siblings: parent excludes both";
	Reset();
	{
		DrawPhaseTimer parent(DrawPhase::Commit);
		{
			DrawPhaseTimer a(DrawPhase::DynState);
			Spin(kUnit * 2);
		}
		{
			DrawPhaseTimer b(DrawPhase::Emit);
			Spin(kUnit * 2);
		}
		Spin(kUnit);
	}
	const auto children = Incl(DrawPhase::DynState) + Incl(DrawPhase::Emit);
	Check(test, Self(DrawPhase::Commit) + children <= Incl(DrawPhase::Commit) + kUnit,
	      "parent self plus children should reconstruct parent inclusive");
	Check(test, Self(DrawPhase::Commit) < children, "parent self must be under both children");
	std::printf("[host]    %-44s ok\n", test);
}

// Stop() called explicitly before the destructor must not double count.
void TestExplicitStop() {
	const char* test = "Explicit Stop then destructor";
	Reset();
	{
		DrawPhaseTimer timer(DrawPhase::Pipeline);
		Spin(kUnit);
		timer.Stop();
		Spin(kUnit * 4);   // must NOT be attributed
	}
	Check(test, Self(DrawPhase::Pipeline) == Incl(DrawPhase::Pipeline), "leaf self == inclusive");
	Check(test, Incl(DrawPhase::Pipeline) < kUnit * 3,
	      "time after Stop() must not be counted");
	std::printf("[host]    %-44s ok\n", test);
}

// Inactive profiling must leave both accumulators untouched.
void TestInactive() {
	const char* test = "Inactive: nothing accumulates";
	Reset();
	g_draw_profile.active = false;
	{
		DrawPhaseTimer timer(DrawPhase::Snapshot);
		Spin(kUnit);
	}
	Check(test, Incl(DrawPhase::Snapshot) == 0 && Self(DrawPhase::Snapshot) == 0,
	      "no accumulation while inactive");
	g_draw_profile.active = true;
	std::printf("[host]    %-44s ok\n", test);
}

} // namespace

int main() {
	TestNested();
	TestRecursive();
	TestRepeated();
	TestTwoChildren();
	TestExplicitStop();
	TestInactive();
	if (g_failures != 0) {
		std::printf("%d draw-phase self-time check(s) FAILED\n", g_failures);
		return 1;
	}
	std::printf("all draw-phase self-time checks passed\n");
	return 0;
}
