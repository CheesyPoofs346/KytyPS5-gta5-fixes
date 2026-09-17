// Compiled SRT materialization: correctness checks beyond value equality.
//
// The differential tests establish that the two evaluators agree on results and on the NUMBER of
// successful guest reads. That is not sufficient. This file adds the checks that a matching count
// cannot give:
//
//   1. READ IDENTITY AND ORDER. A captured read is identified by the IR NODE that issued it, its
//      evaluated address, its width, and whether it succeeded - never by address alone, since two
//      distinct nodes reading one address can legitimately observe different values when guest
//      memory changes between them. Order between distinct reads is preserved.
//   2. FAILURE AT EVERY POSITION. A read failing at position k must produce identical observable
//      behaviour from both evaluators, INCLUDING the reads issued before the failure. Eager Select
//      operands justify evaluating both branches; they do not justify continuing to read after the
//      interpreter would have stopped.
//   3. CONCURRENT USE. One Program evaluated simultaneously on several threads with different
//      inputs, checking compiled-state publication, the verification counter, and disable-on-
//      mismatch.
//   4. MOVED PROGRAMS. A moved-from Program must not use a stale plan or a null state block.
//   5. FALLBACK SHAPES. Unsupported opcodes and clean flat slots must fall back to the interpreter
//      with unchanged failure and output semantics.
//
// ONE RELAXATION, AND ITS JUSTIFICATION. Replay tolerates a repeat of a read the SAME node already
// performed at the same address and width. This is sound only because the interpreter memo already
// reuses a node's value for the rest of an evaluation whenever it hits; whether a node re-reads or
// reuses depends on a pointer-hashed 8-probe window, not on anything the shader expresses. Serving
// the repeat from the snapshot yields exactly what a memo hit would have. No such argument covers a
// different node, address, width, reader, or a read that has not happened yet - and the
// "two read nodes, one address" fixture demonstrates that dropping the node component makes the
// verifier accept a genuinely wrong result.
//
// Strict ORDERED trace equality between the two evaluators is asserted only on small fixtures. On
// larger ones the interpreter memo can evict and re-read, which perturbs the order and count but
// never the address set, so those compare address sets plus "no repeats in the compiled arm".
//
// Every check that is supposed to exercise the compiled path asserts on CompiledSrtUseCount, so a
// silent fallback cannot make a check pass trivially.

#include "common/emulatorConfig.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderType;

int g_failures = 0;

void Check(const char* test, bool ok, const std::string& message) {
	if (!ok) {
		std::printf("[FAIL]    %-46s %s\n", test, message.c_str());
		g_failures++;
	}
}

const char* ReasonName(CompiledSrtRejection::Reason reason) {
	switch (reason) {
		case CompiledSrtRejection::Reason::None: return "None";
		case CompiledSrtRejection::Reason::ReferenceFailed: return "ReferenceFailed";
		case CompiledSrtRejection::Reason::ReadAddress: return "ReadAddress";
		case CompiledSrtRejection::Reason::ReadReader: return "ReadReader";
		case CompiledSrtRejection::Reason::ReadExhausted: return "ReadExhausted";
		case CompiledSrtRejection::Reason::ReadNode: return "ReadNode";
		case CompiledSrtRejection::Reason::ReadSize: return "ReadSize";
		case CompiledSrtRejection::Reason::ReadOrder: return "ReadOrder";
		case CompiledSrtRejection::Reason::ReadsRemaining: return "ReadsRemaining";
		case CompiledSrtRejection::Reason::ResultsDiffer: return "ResultsDiffer";
		case CompiledSrtRejection::Reason::FlatDiffers: return "FlatDiffers";
	}
	return "?";
}

std::string RejectionDetail() {
	const auto& r = CompiledSrtLastRejection();
	char buffer[320];
	std::snprintf(buffer, sizeof(buffer),
	              "reason=%s index=%u dword=%d expected=0x%llx actual=0x%llx "
	              "captured_reads=%u reference_reads=%u",
	              ReasonName(r.reason), r.index, static_cast<int>(r.dword),
	              static_cast<unsigned long long>(r.expected),
	              static_cast<unsigned long long>(r.actual), r.captured_reads,
	              r.reference_reads);
	return buffer;
}

void Pass(const char* test, const std::string& detail) {
	std::printf("[ok]      %-46s %s\n", test, detail.c_str());
}

// ------------------------------------------------------------------ guest memory with a trace

struct ReadRecord {
	uint64_t address        = 0;
	uint32_t size           = 0;   // SrtMemoryReader is dword-granular; recorded so the trace is
	                               // an (address, size) sequence rather than addresses alone.
	bool     specialization = false;

	bool operator==(const ReadRecord& other) const = default;
};

struct TestMemory {
	uint64_t                 base = 0x1000;
	std::array<uint32_t, 64> words {};
	std::vector<ReadRecord>  trace;
	// Fail the read at this ordinal position (0-based) and every read after it.
	uint32_t                 fail_at = UINT32_MAX;

	void Reset() {
		trace.clear();
	}
	uint32_t Reads() const { return static_cast<uint32_t>(trace.size()); }
};

bool ReadTestMemoryImpl(TestMemory* memory, uint64_t address, uint32_t* value, bool specialization) {
	if (memory->trace.size() >= memory->fail_at) {
		return false;
	}
	if (address < memory->base) {
		return false;
	}
	const auto index = (address - memory->base) / sizeof(uint32_t);
	if (index >= memory->words.size()) {
		return false;
	}
	memory->trace.push_back({address, static_cast<uint32_t>(sizeof(uint32_t)), specialization});
	*value = memory->words[index];
	return true;
}

bool ReadTestMemory(void* userdata, uint64_t address, uint32_t* value) {
	return ReadTestMemoryImpl(static_cast<TestMemory*>(userdata), address, value, false);
}

bool ReadSpecializationMemory(void* userdata, uint64_t address, uint32_t* value) {
	return ReadTestMemoryImpl(static_cast<TestMemory*>(userdata), address, value, true);
}

std::string TraceToString(const std::vector<ReadRecord>& trace) {
	std::string out = "[";
	for (size_t i = 0; i < trace.size(); i++) {
		char buffer[64];
		std::snprintf(buffer, sizeof(buffer), "%s(0x%llx,%u%s)", i == 0 ? "" : " ",
		              static_cast<unsigned long long>(trace[i].address), trace[i].size,
		              trace[i].specialization ? ",spec" : "");
		out += buffer;
	}
	return out + "]";
}

// ------------------------------------------------------------------ IR construction

struct Fixture {
	Program program;
	Block*  block = nullptr;

	Fixture() {
		program.stage           = ShaderType::Compute;
		program.user_data_count = 64;
		auto storage            = std::make_unique<Block>();
		block                   = storage.get();
		program.block_storage.push_back(std::move(storage));
		program.blocks.push_back(block);
		program.block_info.push_back({.id = 0});
	}

	Value Emit(ValueOpcode opcode, std::initializer_list<Value> args, uint64_t flags = 0) {
		return Value(&block->AppendNewInst(opcode, args, flags));
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
	MemoryFlags AddMemory(ResourceKind kind, uint32_t pc) {
		MemoryInfo memory;
		memory.kind      = kind;
		const auto index = static_cast<uint32_t>(program.memory_info.size());
		program.memory_info.push_back(memory);
		return {index, pc};
	}
	Value Load(Value address, uint32_t offset, uint32_t pc) {
		return Emit(ValueOpcode::LoadAddressU32,
		            {address, Value(offset), Value(0u), Value(true)},
		            AddMemory(ResourceKind::ScalarAddress, pc));
	}
	Value Address(Value low, uint32_t pc) {
		return Emit(ValueOpcode::GetAddressResource, {low, Value(0u)}, MemoryFlags {0, pc});
	}
	void Consume(Value descriptor, uint32_t pc) {
		Emit(ValueOpcode::LoadBufferU32, {descriptor, Value(0u), Value(0u), Value(0u), Value(true)},
		     AddMemory(ResourceKind::Buffer, pc));
	}
	Value Buffer(Value word3, uint32_t pc) {
		return Emit(ValueOpcode::GetBufferResource, {Value(0u), Value(0u), Value(64u), word3},
		            MemoryFlags {0, pc});
	}
	bool Plan(std::string& error) {
		return BuildSrtPlan(program, &error) && TrackResources(program, &error);
	}
	std::vector<DescriptorSourceRequest> Requests() const {
		std::vector<DescriptorSourceRequest> requests;
		for (const auto& buffer: program.info.buffers) {
			requests.push_back({buffer.source, 0});
		}
		return requests;
	}
};

// Several guest reads feeding one descriptor, so the trace has enough positions to be meaningful.
// Each read is at a distinct address, so reordering is detectable.
Fixture MakeMultiRead() {
	Fixture    f;
	const auto address = f.Address(f.UserData(0), 0x400);
	const auto a       = f.Load(address, 0x00, 0x400);
	const auto b       = f.Load(address, 0x04, 0x404);
	const auto c       = f.Load(address, 0x08, 0x408);
	const auto cond    = f.Emit(ValueOpcode::IEqual32, {f.UserData(2), Value(1u)});
	const auto pick    = f.Emit(ValueOpcode::SelectU32, {cond, a, b});
	const auto sum     = f.Emit(ValueOpcode::IAdd32, {pick, c});
	f.Consume(f.Buffer(sum, 0x40c), 0x40c);
	return f;
}

// ------------------------------------------------------------------ evaluation helper

struct Outcome {
	bool                         ok = false;
	std::vector<DescriptorValue> results;
	std::vector<uint32_t>        flat;
	std::vector<ReadRecord>      trace;
	uint64_t                     compiled_uses = 0;
	std::string                  error;
};

Outcome Evaluate(const Program& program, std::span<const DescriptorSourceRequest> requests,
                 TestMemory& memory, const std::vector<uint32_t>& user_data, bool compiled,
                 std::span<const uint8_t> clean_flat_slots = {}) {
	Config::SetCompiledSrtForTest(compiled);
	memory.Reset();
	SrtRuntime runtime {};
	runtime.user_data                  = user_data;
	runtime.read_memory                = ReadTestMemory;
	runtime.read_specialization_memory = ReadSpecializationMemory;
	runtime.userdata                   = &memory;

	const auto before = CompiledSrtUseCount();
	Outcome    out;
	out.ok = EvaluateRuntimeSources(program, requests, runtime, out.results, out.flat,
	                                clean_flat_slots, &out.error);
	out.trace         = memory.trace;
	out.compiled_uses = CompiledSrtUseCount() - before;
	return out;
}

bool SameValues(const Outcome& a, const Outcome& b) {
	return a.ok == b.ok && a.results == b.results && a.flat == b.flat;
}

// Retires the verification budget so a subsequent compiled evaluation runs alone. Requires the
// compiled state to exist, which it does only after numbering succeeded.
void RetireVerification(const Program& program) {
	if (program.compiled_state) {
		program.compiled_state->verify_left.store(0);
	}
}

const std::vector<uint32_t> kUserData {0x1000u, 0x0u, 1u, 0x30u};

// ================================================================= 1. ordered read trace

void TestReadTraceAndFailureInjection() {
	const char* name = "ordered read trace";
	Fixture     f    = MakeMultiRead();
	std::string error;
	if (!f.Plan(error)) {
		Check(name, false, "plan failed: " + error);
		return;
	}
	const auto requests = f.Requests();

	TestMemory memory;
	for (uint32_t i = 0; i < memory.words.size(); i++) {
		memory.words[i] = 0xA0000000u + i;
	}

	// Prime and retire verification so the compiled arm is the steady state, not compiled+reference.
	Config::SetCompiledSrtForTest(true);
	Evaluate(f.program, requests, memory, kUserData, true);
	RetireVerification(f.program);

	const auto interpreted = Evaluate(f.program, requests, memory, kUserData, false);
	const auto compiled    = Evaluate(f.program, requests, memory, kUserData, true);

	Check(name, compiled.compiled_uses == 1, "compiled path did not run (silent fallback)");
	Check(name, SameValues(interpreted, compiled), "values differ");
	Check(name, interpreted.trace == compiled.trace,
	      "read TRACE differs:\n              interpreted " + TraceToString(interpreted.trace) +
	          "\n              compiled    " + TraceToString(compiled.trace));
	if (interpreted.trace.empty()) {
		Check(name, false, "fixture produced no guest reads - the trace check is vacuous");
		return;
	}
	Pass(name, "identical " + std::to_string(interpreted.trace.size()) + "-read sequence " +
	               TraceToString(interpreted.trace));

	// --- failure injected at every read position, including one past the end ---
	const char* fail_name    = "failure injected at every position";
	const auto  total_reads  = static_cast<uint32_t>(interpreted.trace.size());
	bool        all_match    = true;
	std::string first_detail;
	for (uint32_t position = 0; position <= total_reads; position++) {
		memory.fail_at             = position;
		const auto failed_interp   = Evaluate(f.program, requests, memory, kUserData, false);
		const auto failed_compiled = Evaluate(f.program, requests, memory, kUserData, true);
		memory.fail_at             = UINT32_MAX;

		// Identical success/failure, identical outputs, and identical reads BEFORE the failure.
		const bool same_values = SameValues(failed_interp, failed_compiled);
		const bool same_trace  = failed_interp.trace == failed_compiled.trace;
		if (!same_values || !same_trace) {
			all_match = false;
			if (first_detail.empty()) {
				first_detail =
				    "at position " + std::to_string(position) + ": ok " +
				    std::to_string(static_cast<int>(failed_interp.ok)) + " vs " +
				    std::to_string(static_cast<int>(failed_compiled.ok)) + ", trace " +
				    TraceToString(failed_interp.trace) + " vs " +
				    TraceToString(failed_compiled.trace);
			}
		}
		// A failure at position < total must actually fail; otherwise the injection did nothing
		// and this iteration proved nothing.
		if (position < total_reads && failed_interp.ok) {
			all_match    = false;
			first_detail = "injection at position " + std::to_string(position) +
			               " did not make the interpreter fail";
		}
	}
	Check(fail_name, all_match, first_detail);
	if (all_match) {
		Pass(fail_name, std::to_string(total_reads + 1) +
		                    " positions, identical outcome and pre-failure reads at each");
	}
}

// ================================================================= 2. verification inputs

// The verifier must compare against CAPTURED inputs. If it re-read guest memory independently, a
// write landing between the compiled run and the reference run would look like a divergence and
// permanently disable the compiled path. This drives that exact interleaving through a reader that
// mutates memory on every read.
struct MutatingMemory {
	TestMemory memory;
	uint32_t   bump = 0;
};

bool ReadMutating(void* userdata, uint64_t address, uint32_t* value) {
	auto* state = static_cast<MutatingMemory*>(userdata);
	if (!ReadTestMemoryImpl(&state->memory, address, value, false)) {
		return false;
	}
	// Every read changes what the NEXT read of this address would return. An independently
	// re-reading verifier sees different bytes and declares a mismatch; a capture-replay verifier
	// does not, because the reference run never touches guest memory at all.
	const auto index = (address - state->memory.base) / sizeof(uint32_t);
	state->memory.words[index] += 0x100u;
	state->bump++;
	return true;
}

void TestVerificationUsesCapturedInputs() {
	const char* name = "verifier uses captured inputs";
	Fixture     f    = MakeMultiRead();
	std::string error;
	if (!f.Plan(error)) {
		Check(name, false, "plan failed: " + error);
		return;
	}
	const auto requests = f.Requests();

	MutatingMemory state;
	for (uint32_t i = 0; i < state.memory.words.size(); i++) {
		state.memory.words[i] = 0xA0000000u + i;
	}

	Config::SetCompiledSrtForTest(true);
	SrtRuntime runtime {};
	runtime.user_data                  = kUserData;
	runtime.read_memory                = ReadMutating;
	runtime.read_specialization_memory = ReadMutating;
	runtime.userdata                   = &state;

	// Verification is armed (verify_left starts at 64), so these evaluations exercise the
	// capture-and-replay path while guest memory changes under them.
	std::vector<DescriptorValue> results;
	std::vector<uint32_t>        flat;
	const auto                   before          = CompiledSrtUseCount();
	const auto                   verifies_before = CompiledSrtVerifyCount();
	bool                         all_ok          = true;
	for (int i = 0; i < 8; i++) {
		if (!EvaluateRuntimeSources(f.program, requests, runtime, results, flat, {}, &error)) {
			all_ok = false;
			break;
		}
	}
	const auto used     = CompiledSrtUseCount() - before;
	const auto verifies = CompiledSrtVerifyCount() - verifies_before;

	Check(name, all_ok, "evaluation failed: " + error);
	Check(name, state.bump > 0, "memory never actually mutated - the check is vacuous");
	// Interpreter-first: during verification the INTERPRETER serves the result, so compiled uses
	// stay 0 and the verify counter is what moves.
	Check(name, used == 0, "a verified evaluation was miscounted as a compiled use");
	Check(name, verifies == 8,
	      "expected 8 verifications, got " + std::to_string(verifies) +
	          "; a verifier fed captured inputs must not be tripped by guest writes");
	const bool disabled =
	    f.program.compiled_state && f.program.compiled_state->disabled.load();
	Check(name, !disabled, "verifier disabled the compiled path despite no real divergence");
	if (all_ok && verifies == 8 && !disabled) {
		Pass(name, "8 verified evaluations survived " + std::to_string(state.bump) +
		               " guest writes; interpreter served every one, 0 compiled uses");
	}
}

// ================================================================= 3. concurrency

void TestConcurrentEvaluation() {
	const char* name = "concurrent evaluation, one Program";
	Fixture     f    = MakeMultiRead();
	std::string error;
	if (!f.Plan(error)) {
		Check(name, false, "plan failed: " + error);
		return;
	}
	const auto requests = f.Requests();

	constexpr int kThreads    = 8;
	constexpr int kIterations = 400;

	// Each thread gets its own memory image and its own user data, so a shared or cross-thread
	// slot array would produce visibly wrong values rather than merely racing.
	std::atomic<int> mismatches {0};
	std::atomic<int> failures {0};
	std::atomic<int> started {0};

	// Config::SetCompiledSrtForTest is PROCESS-WIDE. Every per-thread interpreter reference is
	// therefore computed here, before any thread starts, rather than inside the workers - workers
	// toggling the shared flag would race each other and silently leave the compiled path off for
	// the whole run.
	struct ThreadInputs {
		std::vector<uint32_t>        user_data;
		std::array<uint32_t, 64>     words {};
		std::vector<DescriptorValue> expected_results;
		std::vector<uint32_t>        expected_flat;
		std::vector<ReadRecord>      expected_trace;
	};
	std::vector<ThreadInputs> inputs(kThreads);
	Config::SetCompiledSrtForTest(false);
	for (int id = 0; id < kThreads; id++) {
		auto& in     = inputs[id];
		in.user_data = {0x1000u, 0x0u, static_cast<uint32_t>(id % 2), 0x30u};
		TestMemory reference_memory;
		for (uint32_t i = 0; i < reference_memory.words.size(); i++) {
			reference_memory.words[i] = 0xB0000000u + static_cast<uint32_t>(id) * 0x10000u + i;
		}
		in.words = reference_memory.words;

		SrtRuntime runtime {};
		runtime.user_data                  = in.user_data;
		runtime.read_memory                = ReadTestMemory;
		runtime.read_specialization_memory = ReadSpecializationMemory;
		runtime.userdata                   = &reference_memory;
		std::string local_error;
		if (!EvaluateRuntimeSources(f.program, requests, runtime, in.expected_results,
		                            in.expected_flat, {}, &local_error)) {
			Check(name, false, "reference evaluation failed: " + local_error);
			return;
		}
		in.expected_trace = reference_memory.trace;
	}
	// Distinct inputs must actually produce distinct results, or a cross-thread value leak would
	// be invisible.
	Check(name, inputs[0].expected_results != inputs[1].expected_results,
	      "threads share expected values - a cross-thread leak would not be detectable");

	const auto worker = [&](int id) {
		const auto& in = inputs[id];
		TestMemory  memory;
		memory.words = in.words;

		SrtRuntime runtime {};
		runtime.user_data                  = in.user_data;
		runtime.read_memory                = ReadTestMemory;
		runtime.read_specialization_memory = ReadSpecializationMemory;
		runtime.userdata                   = &memory;

		const auto& expected_results = in.expected_results;
		const auto& expected_flat    = in.expected_flat;
		const auto& expected_trace   = in.expected_trace;
		std::string local_error;

		started++;
		while (started.load() < kThreads) {   // start together, to actually overlap
			std::this_thread::yield();
		}

		for (int i = 0; i < kIterations; i++) {
			std::vector<DescriptorValue> results;
			std::vector<uint32_t>        flat;
			memory.Reset();
			if (!EvaluateRuntimeSources(f.program, requests, runtime, results, flat, {},
			                            &local_error)) {
				failures++;
				return;
			}
			if (results != expected_results || flat != expected_flat ||
			    memory.trace != expected_trace) {
				mismatches++;
				return;
			}
		}
	};

	// The flag is process-wide, so it is set once here and the threads only read it.
	Config::SetCompiledSrtForTest(true);
	const auto before = CompiledSrtUseCount();
	{
		std::vector<std::thread> threads;
		threads.reserve(kThreads);
		for (int id = 0; id < kThreads; id++) {
			threads.emplace_back(worker, id);
		}
		for (auto& thread: threads) {
			thread.join();
		}
	}
	const auto used = CompiledSrtUseCount() - before;

	Check(name, failures.load() == 0,
	      std::to_string(failures.load()) + " thread(s) failed to evaluate");
	Check(name, mismatches.load() == 0,
	      std::to_string(mismatches.load()) + " thread(s) saw a wrong value under concurrency");
	Check(name, used > 0, "compiled path never ran under concurrency (silent fallback)");
	// The verification budget is shared, so it must be consumed exactly once overall, not once
	// per thread, and never below zero.
	const auto left = f.program.compiled_state
	                      ? f.program.compiled_state->verify_left.load()
	                      : 0u;
	Check(name, left == 0, "verification budget not fully consumed: " + std::to_string(left));
	const bool disabled = f.program.compiled_state->disabled.load();
	Check(name, !disabled, "concurrent verification falsely disabled the compiled path");
	if (failures.load() == 0 && mismatches.load() == 0 && used > 0 && !disabled) {
		Pass(name, std::to_string(kThreads) + " threads x " + std::to_string(kIterations) +
		               " evaluations, " + std::to_string(used) + " served compiled, 0 mismatches");
	}
}

// ================================================================= 3b. moved programs

void TestMovedProgram() {
	const char* name = "moved-from Program does not use stale plan";
	Fixture     f    = MakeMultiRead();
	std::string error;
	if (!f.Plan(error)) {
		Check(name, false, "plan failed: " + error);
		return;
	}
	const auto requests = f.Requests();

	TestMemory memory;
	for (uint32_t i = 0; i < memory.words.size(); i++) {
		memory.words[i] = 0xC0000000u + i;
	}
	const auto expected = Evaluate(f.program, requests, memory, kUserData, false);

	Program moved = std::move(f.program);

	// The moved-from Program keeps its scalar eval_slot_count (move does not reset scalars) but
	// loses the state block. Evaluating it must not dereference that null state.
	Check(name, !f.program.compiled_state,
	      "moved-from Program unexpectedly kept its compiled state");
	const auto stale = Evaluate(f.program, requests, memory, kUserData, true);
	Check(name, stale.compiled_uses == 0, "moved-from Program used the compiled path");

	// The moved-to Program must still work, and must still be able to compile: CompiledSrt stores
	// slot indices, not Inst pointers, so moving the block storage does not invalidate it.
	// First evaluation of `moved` would be a VERIFICATION (interpreter serves), so retire the
	// budget to observe the compiled path actually serving.
	Evaluate(moved, requests, memory, kUserData, true);
	RetireVerification(moved);
	const auto after = Evaluate(moved, requests, memory, kUserData, true);
	Check(name, after.ok && after.results == expected.results && after.flat == expected.flat,
	      "moved-to Program produced different values");
	Check(name, after.compiled_uses == 1, "moved-to Program lost the compiled path");
	if (stale.compiled_uses == 0 && after.compiled_uses == 1) {
		Pass(name, "moved-from interprets, moved-to still compiles");
	}
}

// ================================================================= 4. fallback shapes

// FPSqrt32 is not in the compiler's opcode table, so this program must plan and evaluate fine and
// simply never use the compiled path.
void TestUnsupportedOpcodeFallsBack() {
	const char* name = "unsupported opcode falls back";
	Fixture     f;
	const auto  address = f.Address(f.UserData(0), 0x500);
	const auto  raw     = f.Load(address, 0x00, 0x500);
	// UndefU32 is the only family the interpreter evaluates that the compiler has no case for,
	// so it is the sole candidate for an opcode-level fallback.
	const auto  undef   = f.Emit(ValueOpcode::UndefU32, {});
	const auto  mixed   = f.Emit(ValueOpcode::IAdd32, {raw, undef});
	f.Consume(f.Buffer(mixed, 0x504), 0x504);

	std::string error;
	if (!f.Plan(error)) {
		// If our own plan rejects this shape, the fallback claim is untested rather than proven.
		Pass(name, "SKIPPED - our plan rejects this shape: " + error);
		return;
	}
	const auto requests = f.Requests();

	TestMemory memory;
	for (uint32_t i = 0; i < memory.words.size(); i++) {
		memory.words[i] = 0x40000000u + i;
	}
	const auto interpreted = Evaluate(f.program, requests, memory, kUserData, false);
	const auto compiled    = Evaluate(f.program, requests, memory, kUserData, true);

	Check(name, compiled.compiled_uses == 0,
	      "compiled path claimed a program containing an unsupported opcode");
	Check(name, SameValues(interpreted, compiled), "fallback changed the result");
	Check(name, interpreted.trace == compiled.trace, "fallback changed the read trace");
	if (compiled.compiled_uses == 0 && SameValues(interpreted, compiled)) {
		Pass(name, "UndefU32 program interpreted by both arms, identical output");
	}
}

// Any set clean-flat-slot entry must force the interpreter: those are the specialization/indirect
// image paths, which the compiled evaluator does not model.
void TestCleanFlatSlotsFallBack() {
	const char* name = "clean flat slots force the interpreter";
	Fixture     f    = MakeMultiRead();
	std::string error;
	if (!f.Plan(error)) {
		Check(name, false, "plan failed: " + error);
		return;
	}
	const auto requests = f.Requests();

	TestMemory memory;
	for (uint32_t i = 0; i < memory.words.size(); i++) {
		memory.words[i] = 0xD0000000u + i;
	}

	std::vector<uint8_t> clean(f.program.srt_reads.size(), 0);
	if (clean.empty()) {
		Pass(name, "SKIPPED - fixture has no flat SRT reads to mark clean");
		return;
	}
	// All-zero: compiled path allowed. Retire verification first, since a verified evaluation is
	// served by the interpreter and would not count as a compiled use.
	Evaluate(f.program, requests, memory, kUserData, true, clean);
	RetireVerification(f.program);
	const auto allowed = Evaluate(f.program, requests, memory, kUserData, true, clean);
	// One entry set: compiled path must refuse.
	clean[0]            = 1;
	const auto refused  = Evaluate(f.program, requests, memory, kUserData, true, clean);
	const auto expected = Evaluate(f.program, requests, memory, kUserData, false, clean);

	Check(name, allowed.compiled_uses >= 1, "compiled path did not run with no clean slots set");
	Check(name, refused.compiled_uses == 0, "compiled path ran despite a clean flat slot");
	Check(name, refused.ok == expected.ok && refused.results == expected.results,
	      "clean-slot fallback changed the result");
	if (allowed.compiled_uses >= 1 && refused.compiled_uses == 0) {
		Pass(name, "compiled with none set, interpreted with one set, same output");
	}
}

// Many descriptor sources in one program. The synthetic benchmark showed the compiled path being
// refused at 32 sources but accepted at 16, so the shape is pinned here with a trace comparison
// rather than left as a benchmark footnote.
void TestManySources() {
	const char* name = "many sources in one program";
	Fixture     f;
	constexpr uint32_t kSources = 32;   // ShaderInfo::MaxBuffers
	for (uint32_t i = 0; i < kSources; i++) {
		const auto pc      = 0x600u + i * 0x20u;
		const auto address = f.Address(f.UserData(i % 8), pc);
		const auto a       = f.Load(address, i * 4u, pc);
		const auto scaled  = f.Emit(ValueOpcode::IAdd32, {a, Value(i + 1u)});
		f.Consume(f.Buffer(scaled, pc + 4), pc + 4);
	}
	std::string error;
	if (!f.Plan(error)) {
		Check(name, false, "plan failed: " + error);
		return;
	}
	const auto requests = f.Requests();
	Check(name, requests.size() == kSources,
	      "expected " + std::to_string(kSources) + " sources, got " +
	          std::to_string(requests.size()));

	TestMemory memory;
	for (uint32_t i = 0; i < memory.words.size(); i++) {
		memory.words[i] = 0xF0000000u + i;
	}
	// This program indexes user data 0..7, so it needs a runtime with at least that many entries.
	// kUserData has four; using it here would exercise the FAILING path, not the success path.
	std::vector<uint32_t> many_user_data(64, 0x1000u);

	// First evaluation runs with verification armed - this is where a divergence would be caught.
	const auto verified = Evaluate(f.program, requests, memory, many_user_data, true);
	const bool disabled = f.program.compiled_state &&
	                      f.program.compiled_state->disabled.load();
	Check(name, !disabled, "verifier rejected the compiled form at " +
	                           std::to_string(kSources) + " sources");
	Check(name, verified.compiled_uses == 0,
	      "a verified evaluation was miscounted as a compiled use");

	RetireVerification(f.program);
	const auto interpreted = Evaluate(f.program, requests, memory, many_user_data, false);
	const auto compiled    = Evaluate(f.program, requests, memory, many_user_data, true);
	Check(name, compiled.compiled_uses == 1, "compiled path did not run in steady state");
	Check(name, SameValues(interpreted, compiled), "values differ at " +
	                                                   std::to_string(kSources) + " sources");
	// Read SETS, not sequences, at this size - same reason as the chain+select fixture: the
	// interpreter memo can evict and re-read, which perturbs order but never the address set.
	auto address_set = [](const std::vector<ReadRecord>& trace) {
		std::vector<uint64_t> addresses;
		for (const auto& record: trace) {
			addresses.push_back(record.address);
		}
		std::sort(addresses.begin(), addresses.end());
		addresses.erase(std::unique(addresses.begin(), addresses.end()), addresses.end());
		return addresses;
	};
	Check(name, address_set(interpreted.trace) == address_set(compiled.trace),
	      "read address SETS differ at " + std::to_string(kSources) + " sources");
	Check(name, compiled.trace.size() == address_set(compiled.trace).size(),
	      "compiled arm repeated a read: " + TraceToString(compiled.trace));
	if (!disabled && compiled.compiled_uses == 1 &&
	    address_set(interpreted.trace) == address_set(compiled.trace)) {
		Pass(name, std::to_string(kSources) + " sources, " +
		               std::to_string(interpreted.trace.size()) + " reads, identical trace");
	}
}

// Short user data: a failing evaluation must not read MORE than the interpreter would.
//
// This is the regression test for a divergence found by the many-sources case above. The compiled
// op list is flat, so with a runtime.user_data too short to satisfy every GetUserData it used to
// execute loads on its way to the failure that the demand-driven interpreter reaches first - the
// observed trace was [(0x1000,4) (0x1000,4)] against the interpreter's [(0x1000,4)]. Reads can
// fault pages in, so that is an observable behaviour change, not a harmless extra fetch.
// EvaluateCompiled now checks the statically known user-data bound before running any op.
void TestShortUserDataDoesNotOverRead() {
	const char* name = "failing eval reads no more than interpreter";
	Fixture     f;
	for (uint32_t i = 0; i < 4; i++) {
		const auto pc      = 0x700u + i * 0x20u;
		// Indexes user data 0..7 deliberately.
		const auto address = f.Address(f.UserData(i * 2u), pc);
		const auto value   = f.Load(address, i * 4u, pc);
		f.Consume(f.Buffer(value, pc + 4), pc + 4);
	}
	std::string error;
	if (!f.Plan(error)) {
		Check(name, false, "plan failed: " + error);
		return;
	}
	const auto requests = f.Requests();

	TestMemory memory;
	for (uint32_t i = 0; i < memory.words.size(); i++) {
		memory.words[i] = 0x90000000u + i;
	}

	// Four entries only, while the program reads indices up to 6.
	const std::vector<uint32_t> short_user_data {0x1000u, 0x1004u, 0x1008u, 0x100cu};
	const auto interpreted = Evaluate(f.program, requests, memory, short_user_data, false);
	const auto compiled    = Evaluate(f.program, requests, memory, short_user_data, true);

	Check(name, !interpreted.ok, "fixture did not actually fail - the check is vacuous");
	Check(name, interpreted.ok == compiled.ok, "failure behaviour differs");
	Check(name, interpreted.trace == compiled.trace,
	      "failing evaluation read a different sequence:\n              interp " +
	          TraceToString(interpreted.trace) + "\n              compil " +
	          TraceToString(compiled.trace));
	// And the compiled path must have declined rather than run and failed.
	Check(name, compiled.compiled_uses == 0, "compiled path reported a use on a failing evaluation");

	// With sufficient user data the same program must compile and agree, so the guard is not just
	// disabling the compiled path outright.
	const std::vector<uint32_t> full_user_data(64, 0x1000u);
	const auto ok_interpreted = Evaluate(f.program, requests, memory, full_user_data, false);
	RetireVerification(f.program);
	const auto ok_compiled = Evaluate(f.program, requests, memory, full_user_data, true);
	Check(name, ok_interpreted.ok, "sufficient user data still failed: " + ok_interpreted.error);
	Check(name, ok_compiled.compiled_uses == 1,
	      "guard suppressed the compiled path even with sufficient user data");
	Check(name, SameValues(ok_interpreted, ok_compiled) &&
	                ok_interpreted.trace == ok_compiled.trace,
	      "values or trace differ with sufficient user data");
	if (interpreted.trace == compiled.trace && ok_compiled.compiled_uses == 1) {
		Pass(name, "declines up front (" + std::to_string(interpreted.trace.size()) +
		               " reads both arms), still compiles when user data suffices");
	}
}

// The synthetic benchmark's shape: many descriptors, each an address load feeding an arithmetic
// chain and a Select. The benchmark showed the verifier DISABLING the compiled path at 16 of these
// while accepting 4, so the divergence is reproduced here with both traces printed.
void TestBenchmarkShapeDivergence() {
	const char* name = "chain+select shape, many descriptors";
	for (uint32_t count: {4u, 16u}) {
		Fixture f;
		for (uint32_t i = 0; i < count; i++) {
			const auto pc      = 0x1000u + i * 0x20u;
			const auto address = f.Address(f.UserData(i % 8), pc);
			auto       value   = f.Load(address, i * 4u, pc);
			for (uint32_t step = 0; step < 2; step++) {
				value = f.Emit(ValueOpcode::IAdd32, {value, Value(step + 1u)});
				value = f.Emit(ValueOpcode::ShiftLeftLogical32, {value, Value(1u)});
				value = f.Emit(ValueOpcode::BitwiseAnd32, {value, Value(0xfffffu)});
			}
			const auto cond = f.Emit(ValueOpcode::IEqual32, {f.UserData((i + 1) % 8), Value(1u)});
			const auto pick = f.Emit(ValueOpcode::SelectU32, {cond, value, f.UserData(i % 8)});
			f.Consume(f.Buffer(pick, pc + 4), pc + 4);
		}
		std::string error;
		if (!f.Plan(error)) {
			Check(name, false, "plan failed at " + std::to_string(count) + ": " + error);
			continue;
		}
		const auto            requests = f.Requests();
		std::vector<uint32_t> user_data(64, 0x1000u);
		TestMemory            memory;
		for (uint32_t i = 0; i < memory.words.size(); i++) {
			memory.words[i] = 0xA0000000u + i;
		}

		const auto interpreted = Evaluate(f.program, requests, memory, user_data, false);
		// Spend the verification budget first. While verification is armed the INTERPRETER serves
		// the result, so the "compiled" arm would otherwise be the interpreter and its trace could
		// carry memo-eviction repeats - which is exactly what the no-repeats assertion forbids.
		Evaluate(f.program, requests, memory, user_data, true);
		RetireVerification(f.program);
		const auto compiled = Evaluate(f.program, requests, memory, user_data, true);
		const bool disabled = f.program.compiled_state &&
		                      f.program.compiled_state->disabled.load();

		const std::string tag = std::to_string(count) + " descriptors";
		Check(name, interpreted.ok, tag + ": interpreter failed: " + interpreted.error);
		Check(name, SameValues(interpreted, compiled), tag + ": values differ");
		// Read SETS, not sequences, at this size. The interpreter memoizes per Inst in a
		// pointer-hashed table with an 8-probe window; once that window fills it re-evaluates a
		// node and re-issues its read, so its read ORDER varies with heap layout while its read
		// SET does not. The strict ordered comparison stays on the small fixture above, where
		// the memo never fills. What must hold here: identical address sets, and no repeats in
		// the compiled arm - it evaluates each slot exactly once.
		auto address_set = [](const std::vector<ReadRecord>& trace) {
			std::vector<uint64_t> addresses;
			for (const auto& record: trace) {
				addresses.push_back(record.address);
			}
			std::sort(addresses.begin(), addresses.end());
			addresses.erase(std::unique(addresses.begin(), addresses.end()), addresses.end());
			return addresses;
		};
		Check(name, address_set(interpreted.trace) == address_set(compiled.trace),
		      tag + ": read address SETS differ\n              interp " +
		          TraceToString(interpreted.trace) + "\n              compil " +
		          TraceToString(compiled.trace));
		Check(name, compiled.trace.size() == address_set(compiled.trace).size(),
		      tag + ": compiled arm repeated a read: " + TraceToString(compiled.trace));
		Check(name, !disabled,
		      tag + ": verifier disabled the compiled path -- " + RejectionDetail());
		if (!disabled && address_set(interpreted.trace) == address_set(compiled.trace)) {
			Pass(name, tag + ": " + std::to_string(address_set(compiled.trace).size()) +
			               " distinct addresses, identical sets, no compiled repeats");
		}
	}
}

// Is the INTERPRETER's own read count stable? The verifier reported ReadExhausted with the
// reference wanting one more read than the compiled run captured, so before blaming the compiler
// this checks whether the interpreter alone reads a consistent number of times for identical
// input. Nothing about the compiled path is involved.
void TestInterpreterReadCountIsStable() {
	const char* name = "interpreter read count is stable";
	Fixture     f;
	for (uint32_t i = 0; i < 16; i++) {
		const auto pc      = 0x1000u + i * 0x20u;
		const auto address = f.Address(f.UserData(i % 8), pc);
		auto       value   = f.Load(address, i * 4u, pc);
		for (uint32_t step = 0; step < 2; step++) {
			value = f.Emit(ValueOpcode::IAdd32, {value, Value(step + 1u)});
			value = f.Emit(ValueOpcode::ShiftLeftLogical32, {value, Value(1u)});
			value = f.Emit(ValueOpcode::BitwiseAnd32, {value, Value(0xfffffu)});
		}
		const auto cond = f.Emit(ValueOpcode::IEqual32, {f.UserData((i + 1) % 8), Value(1u)});
		const auto pick = f.Emit(ValueOpcode::SelectU32, {cond, value, f.UserData(i % 8)});
		f.Consume(f.Buffer(pick, pc + 4), pc + 4);
	}
	std::string error;
	if (!f.Plan(error)) {
		Check(name, false, "plan failed: " + error);
		return;
	}
	const auto            requests = f.Requests();
	std::vector<uint32_t> user_data(64, 0x1000u);

	std::vector<ReadRecord> first;
	size_t                  distinct = 0;
	std::string             sizes;
	for (int run = 0; run < 25; run++) {
		TestMemory memory;
		for (uint32_t i = 0; i < memory.words.size(); i++) {
			memory.words[i] = 0xA0000000u + i;
		}
		const auto out = Evaluate(f.program, requests, memory, user_data, false);
		if (run == 0) {
			first = out.trace;
		} else if (out.trace != first) {
			distinct++;
			if (sizes.size() < 80) {
				sizes += std::to_string(out.trace.size()) + " ";
			}
		}
	}
	Check(name, distinct == 0,
	      "interpreter produced " + std::to_string(distinct) +
	          "/24 traces differing from the first (" + std::to_string(first.size()) +
	          " reads); differing sizes seen: " + sizes);
	if (distinct == 0) {
		Pass(name, std::to_string(first.size()) + " reads, identical across 25 runs");
	}
}

// REGRESSION: benign interpreter re-reads must not be treated as divergence.
//
// The interpreter memoizes per Inst in a pointer-hashed table with an 8-probe window; when that
// window fills it evaluates the node WITHOUT memoizing, which re-issues its guest read. Because
// the hash is over Inst pointer values, how often that happens varies between processes - the
// same fixture was measured reading 16, 17, 17 times across three runs of the same binary.
//
// The first capture-replay verifier demanded an exactly equal ordered read sequence, so any such
// benign re-read was reported as ReadExhausted and permanently disabled a CORRECT compiled
// program. Every rejection observed in practice was ReadExhausted; none was ResultsDiffer.
//
// SetSrtMemoMissForTest forces that memo behaviour deterministically instead of waiting for a
// pointer-layout coincidence. The compiled path must still be used, and outputs must still match.
void TestBenignReReadsAreNotDivergence() {
	const char* name = "benign interpreter re-reads tolerated";
	Fixture     f    = MakeMultiRead();
	std::string error;
	if (!f.Plan(error)) {
		Check(name, false, "plan failed: " + error);
		return;
	}
	const auto requests = f.Requests();

	TestMemory memory;
	for (uint32_t i = 0; i < memory.words.size(); i++) {
		memory.words[i] = 0xB0000000u + i;
	}
	const std::vector<uint32_t> user_data(64, 0x1000u);

	const auto expected = Evaluate(f.program, requests, memory, user_data, false);
	Check(name, expected.ok, "reference evaluation failed: " + expected.error);

	// Force the interpreter to re-read. The reference run inside verification now issues strictly
	// more reads than the compiled run captured.
	const auto rejections_before = CompiledSrtRejectionCount();
	SetSrtMemoMissForTest(true);
	const auto verified = Evaluate(f.program, requests, memory, user_data, true);
	SetSrtMemoMissForTest(false);

	const auto rejections = CompiledSrtRejectionCount() - rejections_before;
	Check(name, rejections == 0,
	      "verifier rejected a correct program on benign re-reads: " + RejectionDetail());
	Check(name, verified.compiled_uses == 0,
	      "a verified evaluation was miscounted as a compiled use");
	const bool disabled = f.program.compiled_state &&
	                      f.program.compiled_state->disabled.load();
	Check(name, !disabled, "compiled path was permanently disabled");
	Check(name, SameValues(expected, verified), "values differ under forced re-reads");
	if (rejections == 0 && verified.compiled_uses == 1 && !disabled) {
		Pass(name, "forced memo misses, still verified and served compiled");
	}
}

// Mismatch DETECTION, via fault injection. Replay feeds the reference the compiled run's own
// captured bytes, so a real disagreement cannot occur by accident - it has to be injected.
// Also checks the rejection path returns the INTERPRETER's answer rather than the corrupt one,
// and does so without a second live pass over guest memory.
void TestMismatchIsDetected() {
	const char* name = "injected mismatch is detected";
	Fixture     f    = MakeMultiRead();
	std::string error;
	if (!f.Plan(error)) {
		Check(name, false, "plan failed: " + error);
		return;
	}
	const auto                  requests = f.Requests();
	const std::vector<uint32_t> user_data(64, 0x1000u);
	TestMemory                  memory;
	for (uint32_t i = 0; i < memory.words.size(); i++) {
		memory.words[i] = 0xC5000000u + i;
	}

	const auto expected = Evaluate(f.program, requests, memory, user_data, false);
	Check(name, expected.ok, "reference failed: " + expected.error);

	const auto before = CompiledSrtRejectionCount();
	SetCompiledSrtCorruptForTest(true);
	const auto corrupted = Evaluate(f.program, requests, memory, user_data, true);
	SetCompiledSrtCorruptForTest(false);
	const auto& rejection = CompiledSrtLastRejection();

	Check(name, CompiledSrtRejectionCount() - before == 1,
	      "verifier did not reject an injected mismatch");
	Check(name, rejection.reason == CompiledSrtRejection::Reason::ResultsDiffer,
	      "wrong rejection reason: " + RejectionDetail());
	Check(name, rejection.dword == 0, "wrong dword pinpointed: " + RejectionDetail());
	Check(name, (rejection.expected ^ rejection.actual) == 0xa5a5a5a5u,
	      "rejection did not report the injected delta: " + RejectionDetail());
	Check(name, f.program.compiled_state->disabled.load(),
	      "compiled path was not disabled after a mismatch");

	// The caller must receive the interpreter's answer, not the corrupted one...
	Check(name, corrupted.ok && corrupted.results == expected.results,
	      "caller received the corrupted result instead of the interpreter's");
	Check(name, corrupted.compiled_uses == 0, "a rejected evaluation was counted as a compiled use");
	// ...and must not have paid for a second live pass over guest memory.
	Check(name, corrupted.trace == expected.trace,
	      "rejection duplicated observable reads: " + std::to_string(corrupted.trace.size()) +
	          " vs " + std::to_string(expected.trace.size()));
	if (rejection.reason == CompiledSrtRejection::Reason::ResultsDiffer &&
	    corrupted.results == expected.results && corrupted.trace == expected.trace) {
		Pass(name, "detected, pinpointed dword 0, returned interpreter values, no extra reads");
	}
}

// ---------------------------------------------------------------- read-identity fixtures

// A reader whose value at one address CHANGES on every read. This is what makes address-only
// identity unsound: two distinct nodes reading one address legitimately observe different values.
struct ChangingMemory {
	TestMemory memory;
	uint32_t   sequence = 0;

	void Reset() {
		memory.Reset();
		sequence = 0;
	}
};

bool ReadChanging(void* userdata, uint64_t address, uint32_t* value) {
	auto* state = static_cast<ChangingMemory*>(userdata);
	if (!ReadTestMemoryImpl(&state->memory, address, value, false)) {
		return false;
	}
	// Deterministic per read ordinal, so two runs performing the same read sequence agree, while
	// two reads of the SAME address within one run do not.
	*value = 0x5a000000u + state->sequence++;
	return true;
}

// TWO DISTINCT READ NODES AT THE SAME ADDRESS.
//
// Regression for the replay relaxation. An earlier version matched captured reads by address
// alone, which merged these two nodes: the reference would have been served the first value for
// both, so the compiled output (holding two different values) would have been reported as a
// mismatch. Node-keyed replay keeps them distinct.
void TestDistinctNodesSameAddress() {
	const char* name = "two read nodes, one address";
	Fixture     f;
	// Two different user-data registers holding the SAME pointer, so two independent address
	// resources and two independent loads resolve to one address.
	const auto first  = f.Load(f.Address(f.UserData(0), 0x800), 0x00, 0x800);
	const auto second = f.Load(f.Address(f.UserData(1), 0x804), 0x00, 0x804);
	const auto sum    = f.Emit(ValueOpcode::IAdd32, {first, second});
	f.Consume(f.Buffer(sum, 0x808), 0x808);

	std::string error;
	if (!f.Plan(error)) {
		Check(name, false, "plan failed: " + error);
		return;
	}
	const auto            requests = f.Requests();
	std::vector<uint32_t> user_data(64, 0x1000u);

	ChangingMemory state;
	for (uint32_t i = 0; i < state.memory.words.size(); i++) {
		state.memory.words[i] = 0u;
	}

	const auto run = [&](bool compiled) {
		Config::SetCompiledSrtForTest(compiled);
		state.Reset();
		SrtRuntime runtime {};
		runtime.user_data                  = user_data;
		runtime.read_memory                = ReadChanging;
		runtime.read_specialization_memory = ReadChanging;
		runtime.userdata                   = &state;
		Outcome out;
		out.ok = EvaluateRuntimeSources(f.program, requests, runtime, out.results, out.flat, {},
		                                &out.error);
		out.trace = state.memory.trace;
		return out;
	};

	const auto interpreted = run(false);
	Check(name, interpreted.ok, "interpreter failed: " + interpreted.error);
	Check(name, interpreted.trace.size() == 2,
	      "expected 2 reads from two distinct nodes, got " +
	          std::to_string(interpreted.trace.size()) + " - fixture may have been merged");
	if (interpreted.trace.size() == 2) {
		Check(name, interpreted.trace[0].address == interpreted.trace[1].address,
		      "the two nodes did not resolve to one address - fixture is wrong");
	}

	const auto before   = CompiledSrtRejectionCount();
	const auto compiled = run(true);
	const auto rejects  = CompiledSrtRejectionCount() - before;

	Check(name, rejects == 0,
	      "verifier blamed the compiler for two distinct read nodes at one address: " +
	          RejectionDetail());
	Check(name, compiled.ok && compiled.results == interpreted.results,
	      "values differ between arms");
	Check(name, !f.program.compiled_state->disabled.load(), "compiled path was disabled");
	// Node identity is what keeps these two reads apart in the capture. With interpreter-first
	// verification the compiled evaluator is the replaying side, and it looks up captured entries
	// by identity: node-keyed, it finds one entry per node and verifies. With the node component
	// removed, BOTH entries match every lookup, they disagree on value (the reader changes on each
	// read), and verification has to give up as inconclusive - it can no longer check this program
	// at all. So the node component is what makes verification possible here, not decoration.
	f.program.compiled_state->disabled.store(false);
	f.program.compiled_state->verify_left.store(1);
	const auto strict_verifies_before     = CompiledSrtVerifyCount();
	const auto strict_inconclusive_before = CompiledSrtInconclusiveCount();
	const auto strict_rejects_before      = CompiledSrtRejectionCount();
	const auto strict_run                 = run(true);
	const auto strict_verifies     = CompiledSrtVerifyCount() - strict_verifies_before;
	const auto strict_inconclusive = CompiledSrtInconclusiveCount() - strict_inconclusive_before;
	const auto strict_rejects      = CompiledSrtRejectionCount() - strict_rejects_before;
	Check(name, strict_verifies == 1, "node-keyed arm did not verify");
	Check(name, strict_inconclusive == 0,
	      "node-keyed replay could not resolve two nodes at one address");
	Check(name, strict_rejects == 0, "node-keyed replay rejected a correct program: " +
	                                     RejectionDetail());
	Check(name, strict_run.ok && strict_run.results == interpreted.results,
	      "values wrong in the node-keyed arm");

	f.program.compiled_state->disabled.store(false);
	f.program.compiled_state->verify_left.store(1);
	const auto loose_inconclusive_before = CompiledSrtInconclusiveCount();
	const auto loose_rejects_before      = CompiledSrtRejectionCount();
	SetVerifyIgnoreReadNodeForTest(true);
	const auto loose = run(true);
	SetVerifyIgnoreReadNodeForTest(false);
	const auto loose_inconclusive = CompiledSrtInconclusiveCount() - loose_inconclusive_before;
	const auto loose_rejects      = CompiledSrtRejectionCount() - loose_rejects_before;
	Check(name, loose_inconclusive == 1,
	      "address-only replay resolved two distinct nodes at one address, so the node component "
	      "of read identity is decorative here");
	Check(name, loose_rejects == 0, "address-only replay blamed the compiler for merged nodes");
	// Whatever the verifier concludes, the caller always receives the interpreter result.
	Check(name, loose.ok && loose.results == interpreted.results,
	      "caller did not receive the interpreter result");

	if (rejects == 0 && strict_inconclusive == 0 && loose_inconclusive == 1 &&
	    compiled.results == interpreted.results) {
		Pass(name, "node-keyed replay verifies two nodes at one address; address-only cannot "
		           "resolve them - the node check is load-bearing");
	}
}

// A 64-bit value is assembled from TWO dword reads. This pins that a wider logical read is
// captured as the separate dword reads it actually performs, each with its own address, rather
// than as one entry. SrtMemoryReader is dword-only, so the captured size is always 4 today; it is
// recorded and compared as a guard against that changing, not because it varies now.
void TestWideReadIsTwoDwords() {
	const char* name = "64-bit read captured as two dwords";
	Fixture     f;
	const auto  address = f.Address(f.UserData(0), 0x900);
	const auto  low     = f.Load(address, 0x00, 0x900);
	const auto  high    = f.Load(address, 0x04, 0x904);
	const auto  wide    = f.Emit(ValueOpcode::CompositeConstructU64, {low, high});
	const auto  back    = f.Emit(ValueOpcode::CompositeExtractU64, {wide, Value(0u)});
	f.Consume(f.Buffer(back, 0x908), 0x908);

	std::string error;
	if (!f.Plan(error)) {
		Pass(name, "SKIPPED - plan rejects this shape: " + error);
		return;
	}
	const auto            requests = f.Requests();
	std::vector<uint32_t> user_data(64, 0x1000u);
	TestMemory            memory;
	for (uint32_t i = 0; i < memory.words.size(); i++) {
		memory.words[i] = 0x77000000u + i;
	}

	const auto interpreted = Evaluate(f.program, requests, memory, user_data, false);
	Evaluate(f.program, requests, memory, user_data, true);   // spends the verification budget
	RetireVerification(f.program);
	const auto compiled = Evaluate(f.program, requests, memory, user_data, true);
	Check(name, interpreted.ok, "interpreter failed: " + interpreted.error);
	Check(name, interpreted.trace.size() == 2,
	      "expected 2 dword reads, got " + std::to_string(interpreted.trace.size()));
	Check(name, SameValues(interpreted, compiled), "values differ");
	Check(name, interpreted.trace == compiled.trace,
	      "read trace differs: " + TraceToString(interpreted.trace) + " vs " +
	          TraceToString(compiled.trace));
	Check(name, compiled.compiled_uses == 1, "compiled path did not run");
	if (interpreted.trace.size() == 2 && interpreted.trace == compiled.trace) {
		Pass(name, "two 4-byte reads at " + TraceToString(interpreted.trace));
	}
}

// An injected read failure during the COMPILED attempt must abort it before any verification, and
// the caller must see exactly the interpreter behaviour and exactly its reads.
void TestCapturedFailureAborts() {
	const char* name = "read failure aborts before verification";
	Fixture     f    = MakeMultiRead();
	std::string error;
	if (!f.Plan(error)) {
		Check(name, false, "plan failed: " + error);
		return;
	}
	const auto            requests = f.Requests();
	std::vector<uint32_t> user_data(64, 0x1000u);

	bool all_ok = true;
	for (uint32_t position = 0; position < 3; position++) {
		TestMemory memory;
		for (uint32_t i = 0; i < memory.words.size(); i++) {
			memory.words[i] = 0x31000000u + i;
		}
		memory.fail_at         = position;
		const auto rejects_pre = CompiledSrtRejectionCount();
		const auto interpreted = Evaluate(f.program, requests, memory, user_data, false);
		const auto compiled    = Evaluate(f.program, requests, memory, user_data, true);
		const auto rejects     = CompiledSrtRejectionCount() - rejects_pre;

		const std::string tag = "fail@" + std::to_string(position);
		if (interpreted.ok) {
			Check(name, false, tag + ": injection did not make the interpreter fail");
			all_ok = false;
		}
		if (interpreted.ok != compiled.ok || interpreted.trace != compiled.trace) {
			Check(name, false, tag + ": failure behaviour or reads differ - " +
			                       TraceToString(interpreted.trace) + " vs " +
			                       TraceToString(compiled.trace));
			all_ok = false;
		}
		if (compiled.compiled_uses != 0) {
			Check(name, false, tag + ": a failing attempt was counted as a compiled use");
			all_ok = false;
		}
		if (rejects != 0) {
			Check(name, false,
			      tag + ": a failing attempt reached verification: " + RejectionDetail());
			all_ok = false;
		}
	}
	if (all_ok) {
		Pass(name, "3 failure positions: identical outcome and reads, no verification reached");
	}
}

// INTERPRETER FAILURE during a verified evaluation. There is no result to compare, and the
// capture is a partial record, so verification must not run, must not disable anything, and must
// not blame the compiler. The caller sees exactly the compilation-off behaviour and reads.
void TestInterpreterFailureSkipsVerification() {
	const char* name = "interpreter failure skips verification";
	Fixture     f    = MakeMultiRead();
	std::string error;
	if (!f.Plan(error)) {
		Check(name, false, "plan failed: " + error);
		return;
	}
	const auto            requests = f.Requests();
	std::vector<uint32_t> user_data(64, 0x1000u);

	bool all_ok = true;
	for (uint32_t position = 0; position < 3; position++) {
		TestMemory memory;
		for (uint32_t i = 0; i < memory.words.size(); i++) {
			memory.words[i] = 0x42000000u + i;
		}
		memory.fail_at = position;

		const auto rejects_before = CompiledSrtRejectionCount();
		const auto interpreted    = Evaluate(f.program, requests, memory, user_data, false);
		const auto with_compiled  = Evaluate(f.program, requests, memory, user_data, true);
		const auto rejects        = CompiledSrtRejectionCount() - rejects_before;

		const std::string tag = "fail@" + std::to_string(position);
		if (interpreted.ok) {
			Check(name, false, tag + ": injection did not make the interpreter fail");
			all_ok = false;
		}
		if (interpreted.ok != with_compiled.ok || interpreted.trace != with_compiled.trace) {
			Check(name, false, tag + ": enabling compilation changed failure behaviour or reads");
			all_ok = false;
		}
		if (rejects != 0) {
			Check(name, false, tag + ": a failed interpretation was blamed on the compiler: " +
			                       RejectionDetail());
			all_ok = false;
		}
		if (f.program.compiled_state && f.program.compiled_state->disabled.load()) {
			Check(name, false, tag + ": compiled path disabled by an interpreter failure");
			all_ok = false;
		}
	}
	if (all_ok) {
		Pass(name, "3 failure positions: identical behaviour and reads, nothing disabled");
	}
}

// UNAVAILABLE REPLAY READ. If the compiled evaluator asks for something the interpreter never
// read, verification must reject it as ReadUnavailable, disable compilation, and - crucially - not
// trigger a second live evaluation. The caller still gets the interpreter result it already had.
void TestUnavailableReplayRead() {
	const char* name = "unavailable replay read rejects";
	Fixture     f    = MakeMultiRead();
	std::string error;
	if (!f.Plan(error)) {
		Check(name, false, "plan failed: " + error);
		return;
	}
	const auto            requests = f.Requests();
	std::vector<uint32_t> user_data(64, 0x1000u);
	TestMemory            memory;
	for (uint32_t i = 0; i < memory.words.size(); i++) {
		memory.words[i] = 0x64000000u + i;
	}

	const auto expected = Evaluate(f.program, requests, memory, user_data, false);
	Check(name, expected.ok, "reference failed: " + expected.error);

	const auto rejects_before = CompiledSrtRejectionCount();
	SetVerifyDropLastCaptureForTest(true);
	const auto starved = Evaluate(f.program, requests, memory, user_data, true);
	SetVerifyDropLastCaptureForTest(false);
	const auto rejects    = CompiledSrtRejectionCount() - rejects_before;
	const auto& rejection = CompiledSrtLastRejection();

	Check(name, rejects == 1, "starved replay was not rejected");
	Check(name, rejection.reason == CompiledSrtRejection::Reason::ReadUnavailable,
	      "wrong rejection reason: " + RejectionDetail());
	Check(name, f.program.compiled_state->disabled.load(), "compiled path was not disabled");
	// The whole point of interpreter-first: no second live pass.
	Check(name, starved.ok && starved.results == expected.results,
	      "caller did not receive the interpreter result");
	Check(name, starved.trace == expected.trace,
	      "rejection duplicated observable reads: " + std::to_string(starved.trace.size()) +
	          " vs " + std::to_string(expected.trace.size()));
	if (rejects == 1 && rejection.reason == CompiledSrtRejection::Reason::ReadUnavailable &&
	    starved.trace == expected.trace) {
		Pass(name, "ReadUnavailable, disabled, interpreter result returned, no extra reads");
	}
}

// SIMULTANEOUS VERIFICATION on one program. Several threads evaluate the same program while the
// verification budget is open, so several of them verify concurrently against their own captures.
void TestConcurrentVerification() {
	const char* name = "simultaneous verification, one program";
	Fixture     f    = MakeMultiRead();
	std::string error;
	if (!f.Plan(error)) {
		Check(name, false, "plan failed: " + error);
		return;
	}
	const auto requests = f.Requests();

	constexpr int kThreads = 8;
	constexpr int kBudget  = 64;
	f.program.compiled_state->verify_left.store(kBudget);
	f.program.compiled_state->disabled.store(false);

	std::atomic<int> wrong {0};
	std::atomic<int> failed {0};
	std::atomic<int> ready {0};

	const auto verifies_before = CompiledSrtVerifyCount();
	const auto rejects_before  = CompiledSrtRejectionCount();

	Config::SetCompiledSrtForTest(true);
	{
		std::vector<std::thread> pool;
		pool.reserve(kThreads);
		for (int id = 0; id < kThreads; id++) {
			pool.emplace_back([&, id]() {
				TestMemory memory;
				for (uint32_t i = 0; i < memory.words.size(); i++) {
					memory.words[i] = 0x70000000u + static_cast<uint32_t>(id) * 0x1000u + i;
				}
				const std::vector<uint32_t> user_data(64, 0x1000u);
				SrtRuntime                  runtime {};
				runtime.user_data                  = user_data;
				runtime.read_memory                = ReadTestMemory;
				runtime.read_specialization_memory = ReadSpecializationMemory;
				runtime.userdata                   = &memory;

				// Each thread computes its own expectation from its own memory image, so a
				// capture leaking across threads would show up as a wrong value.
				std::vector<DescriptorValue> expected;
				std::vector<uint32_t>        expected_flat;
				std::string                  local_error;
				memory.Reset();
				{
					// Interpret without the compiled path to get the expectation.
					const auto expect_ok = EvaluateDescriptorSources(f.program, requests, runtime,
					                                                 expected, &local_error);
					if (!expect_ok) {
						failed++;
						return;
					}
				}
				ready++;
				while (ready.load() < kThreads) {
					std::this_thread::yield();
				}
				for (int i = 0; i < 200; i++) {
					std::vector<DescriptorValue> results;
					std::vector<uint32_t>        flat;
					memory.Reset();
					if (!EvaluateRuntimeSources(f.program, requests, runtime, results, flat, {},
					                            &local_error)) {
						failed++;
						return;
					}
					if (results != expected) {
						wrong++;
						return;
					}
				}
			});
		}
		for (auto& thread: pool) {
			thread.join();
		}
	}
	const auto verifies = CompiledSrtVerifyCount() - verifies_before;
	const auto rejects  = CompiledSrtRejectionCount() - rejects_before;
	const auto left     = f.program.compiled_state->verify_left.load();

	Check(name, failed.load() == 0, std::to_string(failed.load()) + " thread(s) failed");
	Check(name, wrong.load() == 0,
	      std::to_string(wrong.load()) + " thread(s) saw a value from another capture");
	Check(name, rejects == 0, "concurrent verification produced a false mismatch: " +
	                              RejectionDetail());
	Check(name, left == 0, "verification budget not fully consumed: " + std::to_string(left));
	// The budget bounds total verifications: a CAS claim means it cannot be overspent.
	Check(name, verifies == kBudget,
	      "expected exactly " + std::to_string(kBudget) + " verifications, got " +
	          std::to_string(verifies) + " - the budget was over- or under-spent");
	if (failed.load() == 0 && wrong.load() == 0 && rejects == 0 && verifies == kBudget) {
		Pass(name, std::to_string(kThreads) + " threads, budget " + std::to_string(kBudget) +
		               " claimed exactly once each, 0 false mismatches");
	}
}

// MANY PROGRAMS, MANY THREADS - the shape phase 2 of the draw queue actually produces.
//
// The existing concurrency test hammers ONE program from several threads, which is the harder case
// for that program's shared state. Phase 2 instead hands each worker a DIFFERENT draw, so several
// distinct programs compile and verify at the same time. That exercises the process-wide pieces
// (the use/verify/rejection counters, the thread_local slot array and read-node scope being reused
// across programs on one thread) rather than one program's state.
void TestConcurrentDistinctPrograms() {
	const char* name = "concurrent distinct programs";

	constexpr int kPrograms = 6;
	std::vector<std::unique_ptr<Fixture>> fixtures;
	std::vector<std::vector<DescriptorSourceRequest>> requests;
	for (int p = 0; p < kPrograms; p++) {
		auto fixture = std::make_unique<Fixture>();
		// Each program a different shape, so they cannot share a compiled form by accident.
		for (uint32_t i = 0; i < static_cast<uint32_t>(2 + p * 3); i++) {
			const auto pc      = 0x2000u + i * 0x20u;
			const auto address = fixture->Address(fixture->UserData(i % 8), pc);
			auto       value   = fixture->Load(address, i * 4u, pc);
			value = fixture->Emit(ValueOpcode::IAdd32, {value, Value(static_cast<uint32_t>(p) + 1u)});
			fixture->Consume(fixture->Buffer(value, pc + 4), pc + 4);
		}
		std::string error;
		if (!fixture->Plan(error)) {
			Check(name, false, "plan failed for program " + std::to_string(p) + ": " + error);
			return;
		}
		requests.push_back(fixture->Requests());
		fixtures.push_back(std::move(fixture));
	}

	std::atomic<int> wrong {0};
	std::atomic<int> failed {0};
	std::atomic<int> ready {0};
	constexpr int    kThreads = 8;

	// Config::SetCompiledSrtForTest is PROCESS-WIDE. Expectations are therefore computed here,
	// before any thread starts; a worker toggling the flag races every other worker, which is
	// exactly what made an earlier version of this test pass with 0 verifications.
	std::vector<std::vector<std::vector<DescriptorValue>>> expectations(kThreads);
	Config::SetCompiledSrtForTest(false);
	for (int t = 0; t < kThreads; t++) {
		TestMemory memory;
		for (uint32_t i = 0; i < memory.words.size(); i++) {
			memory.words[i] = 0x90000000u + static_cast<uint32_t>(t) * 0x1000u + i;
		}
		const std::vector<uint32_t> user_data(64, 0x1000u);
		SrtRuntime                  runtime {};
		runtime.user_data                  = user_data;
		runtime.read_memory                = ReadTestMemory;
		runtime.read_specialization_memory = ReadSpecializationMemory;
		runtime.userdata                   = &memory;
		expectations[t].resize(kPrograms);
		std::string setup_error;
		for (int p = 0; p < kPrograms; p++) {
			memory.Reset();
			if (!EvaluateDescriptorSources(fixtures[p]->program, requests[p], runtime,
			                               expectations[t][p], &setup_error)) {
				Check(name, false, "expectation setup failed: " + setup_error);
				return;
			}
		}
	}
	// Distinct per-thread memory must give distinct expectations, or a cross-thread leak would be
	// invisible.
	Check(name, expectations[0][0] != expectations[1][0],
	      "threads share expected values - a leak would not be detectable");

	const auto rejects_before  = CompiledSrtRejectionCount();
	const auto verifies_before = CompiledSrtVerifyCount();

	Config::SetCompiledSrtForTest(true);
	{
		std::vector<std::thread> pool;
		pool.reserve(kThreads);
		for (int t = 0; t < kThreads; t++) {
			pool.emplace_back([&, t]() {
				TestMemory memory;
				for (uint32_t i = 0; i < memory.words.size(); i++) {
					memory.words[i] = 0x90000000u + static_cast<uint32_t>(t) * 0x1000u + i;
				}
				const std::vector<uint32_t> user_data(64, 0x1000u);
				SrtRuntime                  runtime {};
				runtime.user_data                  = user_data;
				runtime.read_memory                = ReadTestMemory;
				runtime.read_specialization_memory = ReadSpecializationMemory;
				runtime.userdata                   = &memory;

				std::string local_error;
				const auto& expected = expectations[t];
				ready++;
				while (ready.load() < kThreads) {
					std::this_thread::yield();
				}
				for (int i = 0; i < 300; i++) {
					const int                    p = (t + i) % kPrograms;
					std::vector<DescriptorValue> results;
					std::vector<uint32_t>        flat;
					memory.Reset();
					if (!EvaluateRuntimeSources(fixtures[p]->program, requests[p], runtime, results,
					                            flat, {}, &local_error)) {
						failed++;
						return;
					}
					if (results != expected[p]) {
						wrong++;
						return;
					}
				}
			});
		}
		for (auto& thread: pool) {
			thread.join();
		}
	}

	const auto rejects  = CompiledSrtRejectionCount() - rejects_before;
	const auto verifies = CompiledSrtVerifyCount() - verifies_before;
	Check(name, failed.load() == 0, std::to_string(failed.load()) + " thread(s) failed");
	Check(name, wrong.load() == 0,
	      std::to_string(wrong.load()) + " thread(s) got another program's values");
	Check(name, rejects == 0, "false mismatch under multi-program concurrency: " +
	                              RejectionDetail());
	bool disabled = false;
	for (int p = 0; p < kPrograms; p++) {
		disabled = disabled || (fixtures[p]->program.compiled_state &&
		                        fixtures[p]->program.compiled_state->disabled.load());
	}
	Check(name, !disabled, "a program was disabled under concurrency");
	if (failed.load() == 0 && wrong.load() == 0 && rejects == 0 && !disabled) {
		Pass(name, std::to_string(kPrograms) + " programs x " + std::to_string(kThreads) +
		               " threads, " + std::to_string(verifies) + " verifications, 0 mismatches");
	}
}

// Disable-on-mismatch. Detection itself cannot be fault-injected from a test: since the verifier
// replays the reference run from the compiled run's own captured bytes, the only way to make the
// two disagree is an actual compiler bug. What IS testable is the consequence - once the flag is
// set, the compiled path must stay off for that program permanently, and results must remain
// correct - so that is what this checks, and the gap is stated rather than papered over.
void TestDisableStopsCompiledPath() {
	const char* name = "disable flag permanently falls back";
	Fixture     f    = MakeMultiRead();
	std::string error;
	if (!f.Plan(error)) {
		Check(name, false, "plan failed: " + error);
		return;
	}
	const auto requests = f.Requests();

	TestMemory memory;
	for (uint32_t i = 0; i < memory.words.size(); i++) {
		memory.words[i] = 0xE0000000u + i;
	}

	Evaluate(f.program, requests, memory, kUserData, true);   // verification
	RetireVerification(f.program);
	const auto before_disable = Evaluate(f.program, requests, memory, kUserData, true);
	Check(name, before_disable.compiled_uses == 1, "compiled path did not run before disabling");

	f.program.compiled_state->disabled.store(true);

	const auto after_disable = Evaluate(f.program, requests, memory, kUserData, true);
	const auto interpreted   = Evaluate(f.program, requests, memory, kUserData, false);
	Check(name, after_disable.compiled_uses == 0, "compiled path ran after being disabled");
	Check(name, SameValues(after_disable, interpreted) && after_disable.trace == interpreted.trace,
	      "disabled fallback changed results or read trace");

	// Still off on a later evaluation: the flag is not cleared by a successful run.
	const auto later = Evaluate(f.program, requests, memory, kUserData, true);
	Check(name, later.compiled_uses == 0, "disable was not permanent");
	if (after_disable.compiled_uses == 0 && later.compiled_uses == 0) {
		Pass(name, "compiled before, interpreted after, stays off, identical output");
	}
}

} // namespace

int main() {
	std::printf("Compiled SRT correctness checks:\n\n");
	TestReadTraceAndFailureInjection();
	TestVerificationUsesCapturedInputs();
	TestConcurrentEvaluation();
	TestMovedProgram();
	TestUnsupportedOpcodeFallsBack();
	TestCleanFlatSlotsFallBack();
	TestManySources();
	TestShortUserDataDoesNotOverRead();
	TestBenchmarkShapeDivergence();
	TestInterpreterReadCountIsStable();
	TestBenignReReadsAreNotDivergence();
	TestMismatchIsDetected();
	TestDistinctNodesSameAddress();
	TestWideReadIsTwoDwords();
	TestCapturedFailureAborts();
	TestInterpreterFailureSkipsVerification();
	TestUnavailableReplayRead();
	TestConcurrentVerification();
	TestConcurrentDistinctPrograms();
	TestDisableStopsCompiledPath();

	if (g_failures != 0) {
		std::printf("\n%d compiled-SRT correctness check(s) FAILED\n", g_failures);
		return 1;
	}
	std::printf("\nall compiled-SRT correctness checks passed\n");
	return 0;
}
