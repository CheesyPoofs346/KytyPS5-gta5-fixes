// Differential tests: interpreted vs compiled SRT materialization.
//
// The compiled path (selective port of Almo7aya 0713aeb) replaces the recursive interpreter's
// per-node memo hash and probe with a flat, post-ordered op list over a dense slot array. It must
// be observationally identical to the interpreter:
//
//   * same descriptor values and same flattened SRT,
//   * the SAME NUMBER OF GUEST MEMORY READS - a flat op list must not evaluate operands the
//     interpreter would skip. (Our interpreter is already eager: SelectU32 calls ternary(), which
//     evaluates all three operands, so no lazy divergence is expected. This test pins that.)
//   * identical behaviour on failure, falling back rather than aborting.
//
// The runtime "verify first N evaluations" safeguard is NOT the correctness argument; these
// offline tests are.

#include "common/emulatorConfig.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <array>
#include <memory>
#include <span>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderType;

int g_failures = 0;

void Check(const char* test, bool ok, const std::string& message) {
	if (!ok) {
		std::printf("[FAIL]    %-52s %s\n", test, message.c_str());
		g_failures++;
	}
}

struct Fixture {
	Program program;
	Block*  block = nullptr;

	explicit Fixture(ShaderType stage = ShaderType::Compute) {
		program.stage           = stage;
		program.user_data_count = 64;
		block                   = AddBlock();
	}

	Block* AddBlock() {
		auto  storage = std::make_unique<Block>();
		auto* result  = storage.get();
		program.block_storage.push_back(std::move(storage));
		program.blocks.push_back(result);
		program.block_info.push_back({.id = static_cast<uint32_t>(program.block_info.size())});
		return result;
	}

	Value Emit(ValueOpcode opcode, std::initializer_list<Value> args, uint64_t flags = 0) {
		auto& inst = block->AppendNewInst(opcode, args, flags);
		return Value(&inst);
	}
	template <typename T>
	Value Emit(ValueOpcode opcode, std::initializer_list<Value> args, T flags) {
		uint64_t bits = 0;
		std::memcpy(&bits, &flags, sizeof(flags));
		return Emit(opcode, args, bits);
	}
	Value UserData(uint32_t index) {
		return Emit(ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(index))});
	}
	MemoryFlags AddMemory(MemoryInfo memory, uint32_t pc) {
		const auto index = static_cast<uint32_t>(program.memory_info.size());
		program.memory_info.push_back(memory);
		return {index, pc};
	}
	Value Buffer(std::array<Value, 4> dwords, uint32_t pc = 0) {
		return Emit(ValueOpcode::GetBufferResource, {dwords[0], dwords[1], dwords[2], dwords[3]},
		            MemoryFlags {0, pc});
	}
	bool PlanAndTrack(std::string& error) {
		return BuildSrtPlan(program, &error) && TrackResources(program, &error);
	}
};

// Guest memory stand-in that COUNTS reads, so the two evaluators can be compared on memory
// traffic and not only on results.
struct TestMemory {
	uint64_t                base = 0x1000;
	std::array<uint32_t, 16> words {};
	uint32_t                reads      = 0;
	uint32_t                fail_after = UINT32_MAX;
};

bool ReadTestMemory(void* userdata, uint64_t address, uint32_t* value) {
	auto* memory = static_cast<TestMemory*>(userdata);
	if (memory->reads >= memory->fail_after) {
		return false;
	}
	if (address < memory->base) {
		return false;
	}
	const auto index = (address - memory->base) / sizeof(uint32_t);
	if (index >= memory->words.size()) {
		return false;
	}
	memory->reads++;
	*value = memory->words[index];
	return true;
}

struct Outcome {
	bool                         ok = false;
	std::vector<DescriptorValue> results;
	std::vector<uint32_t>        flat;
	uint32_t                     reads = 0;
};

Outcome Evaluate(const Program& program, std::span<const DescriptorSourceRequest> requests,
                 TestMemory& memory, const std::vector<uint32_t>& user_data, bool compiled) {
	// The flag is what selects the path; both arms run the identical public entry point.
	Config::SetCompiledSrtForTest(compiled);
	memory.reads = 0;
	SrtRuntime runtime {};
	runtime.user_data                  = user_data;
	runtime.read_memory                = ReadTestMemory;
	runtime.read_specialization_memory = ReadTestMemory;
	runtime.userdata                   = &memory;

	Outcome     out;
	std::string error;
	out.ok    = EvaluateRuntimeSources(program, requests, runtime, out.results, out.flat, {},
                                    &error);
	out.reads = memory.reads;
	return out;
}

bool Same(const Outcome& a, const Outcome& b) {
	return a.ok == b.ok && a.results == b.results && a.flat == b.flat;
}

// ---------------------------------------------------------------- cases
//
// A descriptor becomes a tracked source only when it is CONSUMED by a load, so every case ends
// with a LoadBufferU32 over the descriptor - the pattern the existing resource-tracking tests use.

Value Address(Fixture& f, Value low, Value high, uint32_t pc) {
	return f.Emit(ValueOpcode::GetAddressResource, {low, high}, MemoryFlags {0, pc});
}

void Consume(Fixture& f, Value descriptor, uint32_t pc) {
	MemoryInfo memory;
	memory.kind = ResourceKind::Buffer;
	f.Emit(ValueOpcode::LoadBufferU32,
	       {descriptor, Value(0u), Value(0u), Value(0u), Value(true)}, f.AddMemory(memory, pc));
}

// Pure arithmetic over user data: exercises the integer/bitwise ops with no guest reads.
Fixture MakeArithmetic() {
	Fixture    f;
	const auto sum = f.Emit(ValueOpcode::IAdd32, {f.UserData(0), Value(0x40u)});
	const auto sh  = f.Emit(ValueOpcode::ShiftLeftLogical32, {sum, Value(2u)});
	const auto msk = f.Emit(ValueOpcode::BitwiseAnd32, {sh, Value(0xffffu)});
	const auto clamped = f.Emit(ValueOpcode::UMin32, {msk, Value(0x100u)});
	Consume(f, f.Buffer({Value(0u), Value(0u), Value(64u), clamped}, 0x330), 0x330);
	return f;
}

// Select over two arithmetic operands. Our interpreter evaluates all three eagerly (ternary()),
// so a flat op list must produce the same result and the same read count.
Fixture MakeSelect() {
	Fixture    f;
	const auto a    = f.Emit(ValueOpcode::IAdd32, {f.UserData(0), Value(0x11u)});
	const auto b    = f.Emit(ValueOpcode::IAdd32, {f.UserData(1), Value(0x22u)});
	const auto cond = f.Emit(ValueOpcode::IEqual32, {f.UserData(2), Value(1u)});
	const auto pick = f.Emit(ValueOpcode::SelectU32, {cond, a, b});
	Consume(f, f.Buffer({Value(0u), Value(0u), Value(64u), pick}, 0x340), 0x340);
	return f;
}

// A guest memory read feeding the descriptor, and a Select whose UNTAKEN branch is a second read.
// This is the case that would expose a lazy-vs-eager divergence as a read-count mismatch.
Fixture MakeMemorySelect() {
	Fixture    f;
	MemoryInfo word;
	// LoadAddressU32 is an ADDRESS operation, so its MemoryInfo must carry an address kind
	// (IsAddressResourceKind), not Buffer.
	word.kind          = ResourceKind::ScalarAddress;
	const auto address = Address(f, f.UserData(0), Value(0u), 0x350);
	const auto read0   = f.Emit(ValueOpcode::LoadAddressU32,
	                            {address, Value(0u), Value(0u), Value(true)},
	                            f.AddMemory(word, 0x350));
	const auto read1   = f.Emit(ValueOpcode::LoadAddressU32,
	                            {address, Value(4u), Value(0u), Value(true)},
	                            f.AddMemory(word, 0x354));
	const auto cond    = f.Emit(ValueOpcode::IEqual32, {f.UserData(2), Value(1u)});
	const auto pick    = f.Emit(ValueOpcode::SelectU32, {cond, read0, read1});
	Consume(f, f.Buffer({Value(0u), Value(0u), Value(64u), pick}, 0x358), 0x358);
	return f;
}

void RunCase(const char* name, Fixture&& fixture, uint32_t user_data_2) {
	std::string error;
	if (!fixture.PlanAndTrack(error)) {
		std::printf("[skip]    %-52s plan failed: %s\n", name, error.c_str());
		return;
	}
	std::vector<DescriptorSourceRequest> requests;
	for (const auto& buffer: fixture.program.info.buffers) {
		requests.push_back({buffer.source, 0});
	}
	if (requests.empty()) {
		std::printf("[skip]    %-52s no descriptor sources\n", name);
		return;
	}

	TestMemory memory;
	for (uint32_t i = 0; i < memory.words.size(); i++) {
		memory.words[i] = 0xA0000000u + i;
	}
	const std::vector<uint32_t> user_data {0x1000u, 0x0u, user_data_2, 0x30u};

	const auto interpreted = Evaluate(fixture.program, requests, memory, user_data, false);

	// Pass 1: verification is still armed, so the compiled path ALSO runs the interpreter as a
	// reference. Results must match; read counts are expected to be doubled by that reference
	// evaluation, so they are not compared here.
	const auto before          = CompiledSrtUseCount();
	const auto verifies_before = CompiledSrtVerifyCount();
	const auto verified        = Evaluate(fixture.program, requests, memory, user_data, true);
	// Verification is interpreter-first: the INTERPRETER serves this evaluation and the compiled
	// evaluator is replayed against its captured inputs, so the verify counter moves and the use
	// counter does not. Without this the test would pass trivially if nothing ran at all.
	Check(name, CompiledSrtVerifyCount() - verifies_before == 1,
	      "compiled path was never verified (silently ineligible)");
	Check(name, CompiledSrtUseCount() - before == 0,
	      "a verified evaluation was miscounted as a compiled use");
	Check(name, Same(interpreted, verified), "compiled result differs while verifying");

	// Pass 2: steady state. Retire the verification budget so the compiled evaluator runs alone -
	// this is the configuration that actually ships, and the only one whose memory traffic is
	// comparable to the interpreter's.
	fixture.program.compiled_state->verify_left.store(0);
	const auto steady = Evaluate(fixture.program, requests, memory, user_data, true);
	Check(name, CompiledSrtUseCount() - before == 1,
	      "compiled path fell back once verification was retired");
	Check(name, Same(interpreted, steady), "compiled result differs from interpreted");
	Check(name, interpreted.reads == steady.reads,
	      "guest read count differs: interpreted " + std::to_string(interpreted.reads) +
	          " vs compiled " + std::to_string(steady.reads));

	// Changed guest memory: both must observe the change identically.
	memory.words[0] = 0xDEADBEEFu;
	memory.words[1] = 0xFEEDFACEu;
	const auto interpreted2 = Evaluate(fixture.program, requests, memory, user_data, false);
	const auto compiled2    = Evaluate(fixture.program, requests, memory, user_data, true);
	Check(name, Same(interpreted2, compiled2), "differs after guest memory changed");

	// Failing reads must fall back, never abort, and agree.
	memory.fail_after       = 0;
	const auto interpreted3 = Evaluate(fixture.program, requests, memory, user_data, false);
	const auto compiled3    = Evaluate(fixture.program, requests, memory, user_data, true);
	Check(name, interpreted3.ok == compiled3.ok, "failure behaviour differs on unreadable memory");
	memory.fail_after = UINT32_MAX;

	std::printf("[host]    %-52s ok  (reads=%u, sources=%zu)\n", name, interpreted.reads,
	            requests.size());
}

} // namespace

int main() {
	RunCase("arithmetic-only descriptor", MakeArithmetic(), 1);
	RunCase("select over arithmetic, taken", MakeSelect(), 1);
	RunCase("select over arithmetic, untaken", MakeSelect(), 0);
	RunCase("select over guest reads, taken", MakeMemorySelect(), 1);
	RunCase("select over guest reads, untaken", MakeMemorySelect(), 0);

	if (g_failures != 0) {
		std::printf("\n%d compiled-SRT differential check(s) FAILED\n", g_failures);
		return 1;
	}
	std::printf("\nall compiled-SRT differential checks passed\n");
	return 0;
}
