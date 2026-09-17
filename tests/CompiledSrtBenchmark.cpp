// Interpreter vs compiled SRT evaluator: how long does ONE runtime evaluation take?
//
// This measures only the evaluator. It does not measure a frame, and it does not claim a frame-time
// gain: how much of a real frame this accounts for depends on how many descriptor sources each draw
// evaluates and how many draws re-evaluate rather than hit an upstream cache, neither of which this
// benchmark observes. Treat the output as a per-evaluation cost difference and nothing more.
//
// Both arms call the same public entry point; only Config::compiled_srt differs. The compiled arm
// has its verification budget retired first, so it measures the steady state rather than the
// compiled+reference double evaluation that runs for the first 64 evaluations of a program.

#include "common/emulatorConfig.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <atomic>
#include <thread>
#include <vector>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderType;

struct TestMemory {
	uint64_t                 base = 0x1000;
	std::array<uint32_t, 64> words {};
};

bool ReadTestMemory(void* userdata, uint64_t address, uint32_t* value) {
	auto* memory = static_cast<TestMemory*>(userdata);
	if (address < memory->base) {
		return false;
	}
	const auto index = (address - memory->base) / sizeof(uint32_t);
	if (index >= memory->words.size()) {
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

// A descriptor whose address dwords come from a guest read plus a short arithmetic chain - closer
// to the shape the walker actually sees than a single constant would be.
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
	b.Emit(ValueOpcode::LoadBufferU32,
	       {descriptor, Value(0u), Value(0u), Value(0u), Value(true)},
	       b.AddMemory(ResourceKind::Buffer, pc));
}

double TimeArm(const Program& program, std::span<const DescriptorSourceRequest> requests,
               const SrtRuntime& runtime, bool compiled, size_t iterations) {
	Config::SetCompiledSrtForTest(compiled);
	std::vector<DescriptorValue> results;
	std::vector<uint32_t>        flat;
	std::string                  error;

	const auto start = std::chrono::steady_clock::now();
	for (size_t i = 0; i < iterations; i++) {
		if (!EvaluateRuntimeSources(program, requests, runtime, results, flat, {}, &error)) {
			std::printf("evaluation FAILED in %s arm: %s\n", compiled ? "compiled" : "interpreted",
			            error.c_str());
			return -1.0;
		}
	}
	const auto elapsed = std::chrono::steady_clock::now() - start;
	return std::chrono::duration<double, std::nano>(elapsed).count() /
	       static_cast<double>(iterations);
}

void RunSize(uint32_t descriptors, uint32_t chain_length, size_t iterations) {
	Builder b;
	for (uint32_t i = 0; i < descriptors; i++) {
		AddDescriptor(b, i, chain_length);
	}
	std::string error;
	if (!BuildSrtPlan(b.program, &error) || !TrackResources(b.program, &error)) {
		std::printf("  %3u descriptors: plan failed: %s\n", descriptors, error.c_str());
		return;
	}
	std::vector<DescriptorSourceRequest> requests;
	for (const auto& buffer: b.program.info.buffers) {
		requests.push_back({buffer.source, 0});
	}

	TestMemory memory;
	for (uint32_t i = 0; i < memory.words.size(); i++) {
		memory.words[i] = 0xA0000000u + i;
	}
	// SrtRuntime::user_data is a span, so its backing storage must outlive the timed loop.
	const std::vector<uint32_t> user_data(64, 0x1000u);
	SrtRuntime                  runtime {};
	runtime.user_data                  = user_data;
	runtime.read_memory                = ReadTestMemory;
	runtime.read_specialization_memory = ReadTestMemory;
	runtime.userdata                   = &memory;

	// Prime the compiled program and retire verification, so the timed loop is the steady state.
	Config::SetCompiledSrtForTest(true);
	std::vector<DescriptorValue> warm_results;
	std::vector<uint32_t>        warm_flat;
	EvaluateRuntimeSources(b.program, requests, runtime, warm_results, warm_flat, {}, &error);
	if (b.program.compiled_state) {
		b.program.compiled_state->verify_left.store(0);
	}

	// Warm both arms, and confirm the compiled arm is not silently falling back - otherwise the
	// two figures below would be the same code measured twice.
	const auto uses_before = CompiledSrtUseCount();
	TimeArm(b.program, requests, runtime, true, 200);
	TimeArm(b.program, requests, runtime, false, 200);
	const bool compiled_ran = CompiledSrtUseCount() > uses_before;

	const auto interpreted = TimeArm(b.program, requests, runtime, false, iterations);
	const auto compiled    = TimeArm(b.program, requests, runtime, true, iterations);
	if (interpreted < 0 || compiled < 0) {
		return;
	}
	if (!compiled_ran) {
		std::printf("  %3u descriptors: compiled path never ran (fell back) - ignore the ratio\n",
		            descriptors);
	}
	std::printf("  %3u descriptors, %2u-op chains: %3zu sources, %4u slots | interpreted %8.0f ns"
	            " | compiled %8.0f ns | %5.2fx\n",
	            descriptors, chain_length * 3, requests.size(), b.program.eval_slot_count,
	            interpreted, compiled, interpreted / compiled);
}

// Concurrent throughput. The single-threaded figures cannot show whether the compiled path scales
// with the draw workers: the interpreter memoizes per evaluation, the compiled evaluator uses a
// thread_local slot array, and until this port the compiled state was reached through a
// process-wide mutex. This measures aggregate evaluations/second at several thread counts.
void RunConcurrent(uint32_t descriptors, uint32_t chain_length, int threads, size_t per_thread) {
	Builder b;
	for (uint32_t i = 0; i < descriptors; i++) {
		AddDescriptor(b, i, chain_length);
	}
	std::string error;
	if (!BuildSrtPlan(b.program, &error) || !TrackResources(b.program, &error)) {
		std::printf("  %d threads: plan failed: %s\n", threads, error.c_str());
		return;
	}
	std::vector<DescriptorSourceRequest> requests;
	for (const auto& buffer: b.program.info.buffers) {
		requests.push_back({buffer.source, 0});
	}

	const auto measure = [&](bool compiled) {
		Config::SetCompiledSrtForTest(compiled);
		std::atomic<int> ready {0};
		const auto       run = [&]() {
            TestMemory memory;
            for (uint32_t i = 0; i < memory.words.size(); i++) {
                memory.words[i] = 0xA0000000u + i;
            }
            const std::vector<uint32_t> user_data(64, 0x1000u);
            SrtRuntime                  runtime {};
            runtime.user_data                  = user_data;
            runtime.read_memory                = ReadTestMemory;
            runtime.read_specialization_memory = ReadTestMemory;
            runtime.userdata                   = &memory;
            std::vector<DescriptorValue> results;
            std::vector<uint32_t>        flat;
            std::string                  local_error;
            ready++;
            while (ready.load() < threads) {
                std::this_thread::yield();
            }
            for (size_t i = 0; i < per_thread; i++) {
                EvaluateRuntimeSources(b.program, requests, runtime, results, flat, {},
                                       &local_error);
            }
		};
		const auto               start = std::chrono::steady_clock::now();
		std::vector<std::thread> pool;
		pool.reserve(threads);
		for (int t = 0; t < threads; t++) {
			pool.emplace_back(run);
		}
		for (auto& thread: pool) {
			thread.join();
		}
		const auto elapsed = std::chrono::steady_clock::now() - start;
		return std::chrono::duration<double, std::nano>(elapsed).count() /
		       static_cast<double>(per_thread * static_cast<size_t>(threads));
	};

	// Prime, then retire verification, so the timed loop is the steady state.
	Config::SetCompiledSrtForTest(true);
	{
		TestMemory                  memory;
		const std::vector<uint32_t> user_data(64, 0x1000u);
		SrtRuntime                  runtime {};
		runtime.user_data                  = user_data;
		runtime.read_memory                = ReadTestMemory;
		runtime.read_specialization_memory = ReadTestMemory;
		runtime.userdata                   = &memory;
		std::vector<DescriptorValue> results;
		std::vector<uint32_t>        flat;
		EvaluateRuntimeSources(b.program, requests, runtime, results, flat, {}, &error);
	}
	if (b.program.compiled_state) {
		b.program.compiled_state->verify_left.store(0);
	}

	const auto uses_before  = CompiledSrtUseCount();
	const auto compiled     = measure(true);
	const bool compiled_ran = CompiledSrtUseCount() > uses_before;
	const auto interpreted  = measure(false);

	std::printf("  %2d threads: interpreted %8.0f ns/eval | compiled %8.0f ns/eval | %5.2fx%s\n",
	            threads, interpreted, compiled, interpreted / compiled,
	            compiled_ran ? "" : "   (COMPILED PATH DID NOT RUN - ignore)");
}

} // namespace

int main() {
	std::printf("SYNTHETIC SRT evaluator benchmark. The IR here is hand-built, not captured from\n"
	            "GTA V, so the shapes and sizes are chosen, not observed. Treat every number below\n"
	            "as a per-evaluation cost on synthetic input.\n\n");

	std::printf("Single-threaded, per full evaluation of all descriptor sources + flat SRT:\n\n");
	RunSize(1, 2, 20000);
	RunSize(4, 2, 20000);
	RunSize(16, 2, 10000);
	// ShaderInfo::MaxBuffers is 32, so 32 is the largest a single program can reach.
	RunSize(32, 2, 4000);
	RunSize(32, 8, 2000);

	std::printf("\nConcurrent, 16 descriptors with 6-op chains, one shared Program:\n\n");
	RunConcurrent(16, 2, 1, 8000);
	RunConcurrent(16, 2, 2, 8000);
	RunConcurrent(16, 2, 4, 8000);
	RunConcurrent(16, 2, 8, 8000);

	std::printf("\n  Per-evaluation cost on synthetic IR only. Frame impact depends on evaluations\n"
	            "  per frame in the real workload, which this benchmark does not measure.\n");
	return 0;
}
