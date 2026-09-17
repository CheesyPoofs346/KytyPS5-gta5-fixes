// Focused tests for allocations in the compiled SRT evaluator (EvaluateCompiled).
//
// Drives the real public entry point, EvaluateRuntimeSources, with the compiled path enabled and its
// verification budget retired (the steady state), and counts REAL heap allocations by replacing
// global operator new/delete. Also checks that outputs stay exactly equal to the interpreter's, that
// reused output buffers carry no stale state when output sizes change, that a failing evaluation
// leaves the caller's outputs untouched, and prints a hash of the guest-read sequence so read order
// can be compared across builds. Synthetic IR (the benchmark's shapes); not captured from GTA V.

#include "common/emulatorConfig.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <vector>

namespace {
thread_local uint64_t t_allocations = 0;
} // namespace

void* operator new(size_t size) {
	t_allocations++;
	if (void* p = std::malloc(size == 0 ? 1 : size)) {
		return p;
	}
	throw std::bad_alloc();
}
void* operator new[](size_t size) { return ::operator new(size); }
void  operator delete(void* p) noexcept { std::free(p); }
void  operator delete[](void* p) noexcept { std::free(p); }
void  operator delete(void* p, size_t) noexcept { std::free(p); }
void  operator delete[](void* p, size_t) noexcept { std::free(p); }

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderType;

int g_failures = 0;

void Check(const char* test, bool ok, const std::string& message) {
	if (!ok) {
		std::printf("[FAIL]    %-58s %s\n", test, message.c_str());
		g_failures++;
	}
}

struct TestMemory {
	uint64_t                 base  = 0x1000;
	uint32_t                 limit = 64;   // words readable; lower it to make reads fail
	std::array<uint32_t, 64> words {};
	std::vector<uint64_t>    trace;
	bool                     record = false;
};

bool ReadTestMemory(void* userdata, uint64_t address, uint32_t* value) {
	auto* memory = static_cast<TestMemory*>(userdata);
	if (memory->record) {
		memory->trace.push_back(address);
	}
	if (address < memory->base) {
		return false;
	}
	const auto index = (address - memory->base) / sizeof(uint32_t);
	if (index >= memory->limit || index >= memory->words.size()) {
		return false;
	}
	*value = memory->words[index];
	return true;
}

struct Builder {
	Program program;
	Block*  block = nullptr;

	Builder() {
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
};

// Same descriptor shape as CompiledSrtBenchmark: guest read + arithmetic chain + select.
void AddDescriptor(Builder& b, uint32_t index, uint32_t chain_length) {
	const auto pc      = 0x1000u + index * 0x20u;
	const auto address = b.Emit(ValueOpcode::GetAddressResource, {b.UserData(index % 8), Value(0u)},
	                            MemoryFlags {0, pc});
	auto       value   = b.Emit(ValueOpcode::LoadAddressU32,
	                            {address, Value(index * 4u), Value(0u), Value(true)},
	                            b.AddMemory(ResourceKind::ScalarAddress, pc));
	for (uint32_t step = 0; step < chain_length; step++) {
		value = b.Emit(ValueOpcode::IAdd32, {value, Value(step + 1u)});
		value = b.Emit(ValueOpcode::ShiftLeftLogical32, {value, Value(1u)});
		value = b.Emit(ValueOpcode::BitwiseAnd32, {value, Value(0xfffffu)});
	}
	const auto cond = b.Emit(ValueOpcode::IEqual32, {b.UserData((index + 1) % 8), Value(1u)});
	const auto pick = b.Emit(ValueOpcode::SelectU32, {cond, value, b.UserData(index % 8)});
	const auto descriptor = b.Emit(ValueOpcode::GetBufferResource,
	                               {Value(0u), Value(0u), Value(64u), pick}, MemoryFlags {0, pc});
	b.Emit(ValueOpcode::LoadBufferU32, {descriptor, Value(0u), Value(0u), Value(0u), Value(true)},
	       b.AddMemory(ResourceKind::Buffer, pc));
}

struct Fixture {
	std::unique_ptr<Builder>             builder;
	std::vector<DescriptorSourceRequest> requests;
	uint32_t                             descriptors = 0;
};

std::unique_ptr<Fixture> MakeFixture(uint32_t descriptors) {
	auto fixture         = std::make_unique<Fixture>();
	fixture->builder     = std::make_unique<Builder>();
	fixture->descriptors = descriptors;
	for (uint32_t i = 0; i < descriptors; i++) {
		AddDescriptor(*fixture->builder, i, 2);
	}
	std::string error;
	if (!BuildSrtPlan(fixture->builder->program, &error) ||
	    !TrackResources(fixture->builder->program, &error)) {
		std::printf("fixture %u: plan failed: %s\n", descriptors, error.c_str());
		return nullptr;
	}
	for (const auto& buffer: fixture->builder->program.info.buffers) {
		fixture->requests.push_back({buffer.source, 0});
	}
	return fixture;
}

struct Env {
	TestMemory            memory;
	std::vector<uint32_t> user_data = std::vector<uint32_t>(64, 0x1000u);
	SrtRuntime            runtime {};

	Env() {
		for (uint32_t i = 0; i < memory.words.size(); i++) {
			memory.words[i] = 0xA0000000u + i;
		}
		runtime.user_data                  = user_data;
		runtime.read_memory                = ReadTestMemory;
		runtime.read_specialization_memory = ReadTestMemory;
		runtime.userdata                   = &memory;
	}
};

bool Evaluate(const Fixture& f, Env& env, bool compiled, std::span<const DescriptorSourceRequest> requests,
              std::vector<DescriptorValue>& results, std::vector<uint32_t>& flat, std::string& error) {
	Config::SetCompiledSrtForTest(compiled);
	return EvaluateRuntimeSources(f.builder->program, requests, env.runtime, results, flat, {}, &error);
}

// Prime the compiled program and retire verification so the compiled path serves evaluations.
void Retire(const Fixture& f, Env& env) {
	std::vector<DescriptorValue> results;
	std::vector<uint32_t>        flat;
	std::string                  error;
	(void)Evaluate(f, env, true, f.requests, results, flat, error);
	if (f.builder->program.compiled_state) {
		f.builder->program.compiled_state->verify_left.store(0);
	}
}

void TestSteadyStateAllocations(const Fixture& f, Env& env) {
	const std::string test = "Allocations: steady state, " + std::to_string(f.descriptors) + " descriptors";
	Retire(f, env);
	std::vector<DescriptorValue> results;
	std::vector<uint32_t>        flat;
	std::string                  error;
	for (int i = 0; i < 300; i++) {
		(void)Evaluate(f, env, true, f.requests, results, flat, error);
	}
	constexpr int iterations = 1000;
	const auto    uses       = CompiledSrtUseCount();
	const auto    before     = t_allocations;
	bool          ok         = true;
	for (int i = 0; i < iterations; i++) {
		ok = Evaluate(f, env, true, f.requests, results, flat, error) && ok;
	}
	const auto allocations = t_allocations - before;
	const auto served      = CompiledSrtUseCount() - uses;
	std::printf("[info]    %-58s allocations=%llu over %d evaluations (%.3f/eval), compiled served=%llu, "
	            "flat=%zu results=%zu\n",
	            test.c_str(), static_cast<unsigned long long>(allocations), iterations,
	            static_cast<double>(allocations) / iterations, static_cast<unsigned long long>(served),
	            flat.size(), results.size());
	Check(test.c_str(), ok, "evaluation failed: " + error);
	Check(test.c_str(), served == iterations, "compiled path did not serve every evaluation");
	Check(test.c_str(), allocations == 0, std::to_string(allocations) + " allocations after warmup");
}

void TestMatchesInterpreter(const Fixture& f, Env& env) {
	const std::string test = "Outputs: compiled == interpreter, " + std::to_string(f.descriptors) + " descriptors";
	Retire(f, env);
	std::vector<DescriptorValue> compiled_results, interpreted_results;
	std::vector<uint32_t>        compiled_flat, interpreted_flat;
	std::string                  error;
	const bool a = Evaluate(f, env, true, f.requests, compiled_results, compiled_flat, error);
	const bool b = Evaluate(f, env, false, f.requests, interpreted_results, interpreted_flat, error);
	Check(test.c_str(), a && b, "evaluation failed: " + error);
	Check(test.c_str(), compiled_results == interpreted_results, "results differ");
	Check(test.c_str(), compiled_flat == interpreted_flat, "flat differs");
	Check(test.c_str(), compiled_results.size() == f.requests.size(), "results size");
}

void TestChangingSizes(const std::vector<Fixture*>& fixtures, Env& env) {
	const char* test = "Reuse: outputs rebuilt exactly when sizes change";
	std::vector<std::vector<DescriptorValue>> reference_results(fixtures.size());
	std::vector<std::vector<uint32_t>>        reference_flat(fixtures.size());
	std::string                               error;
	for (size_t i = 0; i < fixtures.size(); i++) {
		Retire(*fixtures[i], env);
		(void)Evaluate(*fixtures[i], env, false, fixtures[i]->requests, reference_results[i],
		               reference_flat[i], error);
	}
	std::vector<DescriptorValue> results;
	std::vector<uint32_t>        flat;
	const size_t                 order[] = {2, 0, 1, 0, 2, 1, 2, 0};
	for (const auto index: order) {
		const auto& f = *fixtures[index];
		if (!Evaluate(f, env, true, f.requests, results, flat, error)) {
			Check(test, false, "evaluation failed: " + error);
			continue;
		}
		Check(test, results == reference_results[index],
		      "results differ after switching to " + std::to_string(f.descriptors) + " descriptors");
		Check(test, flat == reference_flat[index],
		      "flat differs after switching to " + std::to_string(f.descriptors) + " descriptors");
	}
}

void TestZeroSources(const Fixture& f, Env& env) {
	const char* test = "Zero sources: empty results, flat still produced";
	Retire(f, env);
	std::vector<DescriptorValue> compiled_results(3), interpreted_results(3);
	std::vector<uint32_t>        compiled_flat, interpreted_flat;
	std::string                  error;
	const std::span<const DescriptorSourceRequest> none {};
	const bool a = Evaluate(f, env, true, none, compiled_results, compiled_flat, error);
	const bool b = Evaluate(f, env, false, none, interpreted_results, interpreted_flat, error);
	Check(test, a && b, "evaluation failed: " + error);
	Check(test, compiled_results.empty() && interpreted_results.empty(), "results not empty");
	Check(test, compiled_flat == interpreted_flat, "flat differs");
}

void TestFailureLeavesOutputsUntouched(const Fixture& f, Env& env) {
	const char* test = "Failure: caller outputs untouched";
	Retire(f, env);
	DescriptorValue sentinel {};
	sentinel.dword_count = 99;
	sentinel.dwords.fill(0xdeadbeefu);
	std::vector<DescriptorValue> results(2, sentinel);
	std::vector<uint32_t>        flat(5, 0xfeedfaceu);
	const auto                   results_before = results;
	const auto                   flat_before    = flat;
	const auto                   saved_limit    = env.memory.limit;
	env.memory.limit                            = 1;   // most guest reads now fail
	std::string error;
	const bool  ok = Evaluate(f, env, true, f.requests, results, flat, error);
	env.memory.limit = saved_limit;
	Check(test, !ok, "evaluation unexpectedly succeeded");
	Check(test, results == results_before, "results modified on failure");
	Check(test, flat == flat_before, "flat modified on failure");
}

void TestReadTrace(const Fixture& f, Env& env) {
	Retire(f, env);
	std::vector<DescriptorValue> results;
	std::vector<uint32_t>        flat;
	std::string                  error;
	env.memory.trace.clear();
	env.memory.record = true;
	const bool ok     = Evaluate(f, env, true, f.requests, results, flat, error);
	env.memory.record = false;
	uint64_t hash = 1469598103934665603ull;
	for (const auto address: env.memory.trace) {
		hash = (hash ^ address) * 1099511628211ull;
	}
	std::printf("[info]    Read trace (compiled, %u descriptors): ok=%d reads=%zu fnv1a=0x%016llx\n",
	            f.descriptors, ok ? 1 : 0, env.memory.trace.size(), static_cast<unsigned long long>(hash));
	Check("Read trace: evaluation succeeds", ok, error);
}

void TestCorruptionInjection(Env& env) {
	const char* test = "Fault injection: corrupted compiled output is rejected";
	auto        f    = MakeFixture(8);
	if (!f) {
		Check(test, false, "fixture failed");
		return;
	}
	std::vector<DescriptorValue> reference_results, results;
	std::vector<uint32_t>        reference_flat, flat;
	std::string                  error;
	(void)Evaluate(*f, env, false, f->requests, reference_results, reference_flat, error);
	const auto rejections = CompiledSrtRejectionCount();
	SetCompiledSrtCorruptForTest(true);
	const bool ok = Evaluate(*f, env, true, f->requests, results, flat, error);   // verification pass
	SetCompiledSrtCorruptForTest(false);
	Check(test, ok, "evaluation failed: " + error);
	Check(test, CompiledSrtRejectionCount() > rejections, "verification did not reject corrupted output");
	Check(test, results == reference_results && flat == reference_flat,
	      "caller received something other than the interpreter's result");
}

} // namespace

int main() {
	Env env;
	auto one   = MakeFixture(1);
	auto eight = MakeFixture(8);
	auto many  = MakeFixture(32);
	if (!one || !eight || !many) {
		std::printf("fixture construction failed\n");
		return 1;
	}
	for (auto* f: {one.get(), eight.get(), many.get()}) {
		TestSteadyStateAllocations(*f, env);
		TestMatchesInterpreter(*f, env);
		TestReadTrace(*f, env);
	}
	TestChangingSizes({one.get(), eight.get(), many.get()}, env);
	TestZeroSources(*eight, env);
	TestFailureLeavesOutputsUntouched(*eight, env);
	TestCorruptionInjection(env);
	if (g_failures != 0) {
		std::printf("%d compiled SRT allocation check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("compiled SRT allocation tests passed\n");
	return 0;
}
