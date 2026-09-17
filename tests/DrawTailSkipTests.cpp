// Focused combined-path check for --skip-ps-chksum + --gpu-timestamps.
//
// EXECUTES: the production RunPreparedDrawTail template (drawTail.h) that ExecutePreparedDraw calls,
// the production selection decision Config::ShouldSkipPixelShaderChksum, and the production skip
// bookkeeping RecordPsChksumSkip / PsChksumSkipCount (the counter DrawCensus prints).
// The tail's operations are counting stand-ins here. NOT executed: the real PreparedDrawTail
// operations (EmitDrawPrimitives, GpuTimestamps::Begin/End gating, CommandScheduler::EndRendering,
// ShaderWriteBarrier, HasShaderBufferWrites) and any Vulkan recording - those remain covered by
// source inspection (check_combined_skip_timestamps.py) and, for timestamps, gpu_timestamps_device_tests.

#include "common/emulatorConfig.h"
#include "graphics/host_gpu/renderer/drawTail.h"

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(const char* test, bool ok, const std::string& message) {
	std::printf("%s %-58s %s\n", ok ? "[ok]  " : "[FAIL]", test, message.c_str());
	if (!ok) {
		g_failures++;
	}
}

void LoadSkipList(std::initializer_list<uint64_t> list) {
	Config::ConfigOptions cfg;
	cfg.skip_ps_chksum.assign(list.begin(), list.end());
	Config::Load(cfg);
}

struct CountingTail {
	uint32_t                 write_stages = 0; // nonzero: the program writes storage buffers
	std::vector<std::string> calls;

	void CountChksumSkip() {
		Libs::Graphics::RecordPsChksumSkip("draw_tail_skip_tests");
		calls.emplace_back("skip-count");
	}
	int BeginTimestamp() {
		calls.emplace_back("timestamp-begin");
		return 42;
	}
	void EmitDraw() { calls.emplace_back("draw"); }
	void EndTimestamp(int token) { calls.emplace_back(token == 42 ? "timestamp-end" : "timestamp-end-wrong-token"); }
	uint32_t BeginTeardown() {
		calls.emplace_back("teardown");
		return write_stages;
	}
	void EndRendering() { calls.emplace_back("end-rendering"); }
	void WriteBarrier(uint32_t stages) {
		calls.emplace_back(stages == write_stages ? "write-barrier" : "write-barrier-wrong-stages");
	}
};

std::string Join(const std::vector<std::string>& calls) {
	std::string text;
	for (const auto& call: calls) {
		text += (text.empty() ? "" : ",") + call;
	}
	return text;
}

// Mirrors ExecutePreparedDraw: decision from the real config, address filter off.
std::vector<std::string> RunDraw(uint32_t ps_chksum, uint32_t write_stages, bool skip_by_address = false) {
	CountingTail tail;
	tail.write_stages         = write_stages;
	const bool skip_ps_chksum = Config::ShouldSkipPixelShaderChksum(ps_chksum);
	const bool emits          = !skip_ps_chksum && !skip_by_address;
	Libs::Graphics::RunPreparedDrawTail(tail, skip_ps_chksum, emits);
	return tail.calls;
}

constexpr uint32_t kSelected = 0x69c80e2au;

} // namespace

int main() {
	Config::Initialize();
	using Calls = std::vector<std::string>;

	// Default off: nothing selected, the draw is timestamped and emitted, teardown runs.
	LoadSkipList({});
	{
		const auto before = Libs::Graphics::PsChksumSkipCount();
		const auto calls  = RunDraw(kSelected, 0x80u);
		Check("default off: selected-looking draw is emitted and timed",
		      calls == Calls {"timestamp-begin", "draw", "timestamp-end", "teardown", "end-rendering",
		                      "write-barrier"},
		      Join(calls));
		Check("default off: skip count unchanged", Libs::Graphics::PsChksumSkipCount() == before,
		      std::to_string(Libs::Graphics::PsChksumSkipCount() - before));
	}

	LoadSkipList({kSelected});
	// Selected draw whose program writes storage buffers.
	{
		const auto before = Libs::Graphics::PsChksumSkipCount();
		const auto calls  = RunDraw(kSelected, 0x80u);
		Check("selected: skip counted, no draw, no timestamp queries, teardown + barrier",
		      calls == Calls {"skip-count", "teardown", "end-rendering", "write-barrier"}, Join(calls));
		Check("selected: skip count +1 exactly", Libs::Graphics::PsChksumSkipCount() == before + 1,
		      std::to_string(Libs::Graphics::PsChksumSkipCount() - before));
	}
	// Selected draw without storage writes: teardown still reached, nothing further required.
	{
		const auto calls = RunDraw(kSelected, 0u);
		Check("selected, no storage writes: teardown reached, no render-pass end/barrier",
		      calls == Calls {"skip-count", "teardown"}, Join(calls));
	}
	// Neighbouring checksum with the list set: emitted and timed.
	{
		const auto before = Libs::Graphics::PsChksumSkipCount();
		const auto calls  = RunDraw(kSelected ^ 1u, 0x80u);
		Check("unselected neighbour: emitted and timed, teardown + barrier",
		      calls == Calls {"timestamp-begin", "draw", "timestamp-end", "teardown", "end-rendering",
		                      "write-barrier"},
		      Join(calls));
		Check("unselected neighbour: skip count unchanged", Libs::Graphics::PsChksumSkipCount() == before,
		      std::to_string(Libs::Graphics::PsChksumSkipCount() - before));
	}
	// Address filter path (--skip-ps): no draw, not counted as a checksum skip, teardown runs.
	{
		const auto before = Libs::Graphics::PsChksumSkipCount();
		const auto calls  = RunDraw(kSelected ^ 1u, 0x80u, true);
		Check("address filter: no draw, no timestamps, not a checksum skip, teardown runs",
		      calls == Calls {"teardown", "end-rendering", "write-barrier"} &&
		          Libs::Graphics::PsChksumSkipCount() == before,
		      Join(calls));
	}
	// Many selected draws: the counter is exact.
	{
		const auto before = Libs::Graphics::PsChksumSkipCount();
		uint64_t   draws  = 0;
		for (uint32_t i = 0; i < 25000u; i++) {
			for (const auto& call: RunDraw(kSelected, i % 2u == 0u ? 0x80u : 0u)) {
				draws += call == "draw" ? 1u : 0u;
			}
		}
		Check("25,000 selected draws: count exact, no draw emitted",
		      Libs::Graphics::PsChksumSkipCount() == before + 25000u && draws == 0u,
		      std::to_string(Libs::Graphics::PsChksumSkipCount() - before));
	}

	LoadSkipList({});
	Config::Shutdown();
	if (g_failures != 0) {
		std::printf("%d draw tail check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("draw tail skip tests passed\n");
	return 0;
}
