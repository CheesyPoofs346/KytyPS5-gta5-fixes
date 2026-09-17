// Equivalence tests and an operation-level benchmark for ValidateMaterializedResourceSpecialization.
//
// ShaderMaterializeStageRuntime used to run ValidateResourceSpecialization right after
// MaterializeResources, repeating the ValidateResourceSnapshot that MaterializeResources had just
// passed on the same content. These tests check that the specialization-only entry makes exactly the
// same decision, with the same error text, as the full check on materialized snapshots (including
// every single-dword descriptor mutation), and time the old and new stage sequences side by side.
// Synthetic IR with chosen shapes; not captured from GTA V. Allocations are counted by replacing
// global operator new/delete.

#include "common/emulatorConfig.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
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

void Check(const std::string& test, bool ok, const std::string& message) {
	if (!ok) {
		std::printf("[FAIL]    %-56s %s\n", test.c_str(), message.c_str());
		g_failures++;
	}
}

struct TestMemory {
	uint64_t                  base = 0x1000;
	std::array<uint32_t, 256> words {};
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

// Fixture shape copied from SnapshotAllocationBenchmark: pixel stage, buffers + sampled images.
struct Fixture {
	Program program;
	Block*  block = nullptr;

	Fixture() {
		program.stage           = ShaderType::Pixel;
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
	MemoryFlags AddMemory(MemoryInfo memory, uint32_t pc) {
		const auto index = static_cast<uint32_t>(program.memory_info.size());
		program.memory_info.push_back(memory);
		return {index, pc};
	}
	MemoryFlags AddMemory(ResourceKind kind, uint32_t pc) {
		MemoryInfo memory;
		memory.kind = kind;
		return AddMemory(memory, pc);
	}
	void AddBuffer(uint32_t index) {
		const auto pc      = 0x1000u + index * 0x40u;
		const auto address = Emit(ValueOpcode::GetAddressResource, {UserData(index % 8), Value(0u)},
		                          MemoryFlags {0, pc});
		const auto word    = Emit(ValueOpcode::LoadAddressU32,
		                          {address, Value(index * 4u), Value(0u), Value(true)},
		                          AddMemory(ResourceKind::ScalarAddress, pc));
		const auto sum     = Emit(ValueOpcode::IAdd32, {word, Value(index + 1u)});
		const auto buffer  = Emit(ValueOpcode::GetBufferResource,
		                          {Value(0u), Value(0u), Value(64u), sum}, MemoryFlags {0, pc});
		Emit(ValueOpcode::LoadBufferU32, {buffer, Value(0u), Value(0u), Value(0u), Value(true)},
		     AddMemory(ResourceKind::Buffer, pc));
	}
	Value ImageAddress() {
		return Emit(ValueOpcode::MakeImageAddress,
		            {Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
		             Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u)});
	}
	void AddSampledImage(uint32_t index) {
		const auto pc    = 0x4000u + index * 0x40u;
		const auto d     = Value(0u);
		const auto image = Emit(ValueOpcode::GetImageResource,
		                        {Value(index + 1u), d, Value(0x2000u), Value(0x1000u), d, d, d, d},
		                        MemoryFlags {0, pc});
		const auto sampler = Emit(ValueOpcode::GetSamplerResource, {d, d, d, d}, MemoryFlags {0, pc});
		MemoryInfo sample;
		sample.kind            = ResourceKind::Image;
		sample.image_dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D;
		const auto sampled = Emit(ValueOpcode::ImageSampleRaw, {image, sampler, ImageAddress()},
		                          AddMemory(sample, pc));
		const auto component = Emit(ValueOpcode::CompositeExtractU32x4, {sampled, Value(0u)});
		Emit(ValueOpcode::ReferenceU32, {component});
	}
};

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

struct Shape {
	const char* name;
	uint32_t    buffers;
	uint32_t    images;
};

// Plans, materializes once, specializes, then materializes again so the returned snapshot is the
// kind ShaderMaterializeStageRuntime validates.
std::unique_ptr<Fixture> BuildSpecialized(const Shape& shape, Env& env, ResourceSnapshot& snapshot,
                                          std::string& error) {
	auto fixture = std::make_unique<Fixture>();
	for (uint32_t i = 0; i < shape.buffers; i++) {
		fixture->AddBuffer(i);
	}
	for (uint32_t i = 0; i < shape.images; i++) {
		fixture->AddSampledImage(i);
	}
	if (!BuildSrtPlan(fixture->program, &error) || !TrackResources(fixture->program, &error)) {
		error = "plan: " + error;
		return nullptr;
	}
	ResourceSnapshot first;
	if (!MaterializeResources(fixture->program, env.runtime, first, &error)) {
		error = "first materialization: " + error;
		return nullptr;
	}
	if (!SpecializeResources(fixture->program, first, &error)) {
		error = "specialize: " + error;
		return nullptr;
	}
	if (!MaterializeResources(fixture->program, env.runtime, snapshot, &error)) {
		error = "second materialization: " + error;
		return nullptr;
	}
	return fixture;
}

void TestEquivalence(const Shape& shape, const Fixture& fixture, const ResourceSnapshot& snapshot) {
	const std::string test = std::string("Equivalence: ") + shape.name;
	std::string       full_error, materialized_error;
	const bool full         = ValidateResourceSpecialization(fixture.program, snapshot, &full_error);
	const bool materialized = ValidateMaterializedResourceSpecialization(fixture.program, snapshot,
	                                                                     &materialized_error);
	Check(test, full, "materialized specialized snapshot rejected by full check: " + full_error);
	Check(test, full == materialized && full_error == materialized_error,
	      "unmutated decision differs: " + full_error + " | " + materialized_error);

	uint32_t mutations = 0, rejected = 0, snapshot_rejected = 0;
	const auto mutate = [&](auto member, size_t index, uint32_t dword) {
		ResourceSnapshot copy = snapshot;
		auto&            value = (copy.*member)[index];
		value.dwords[dword] ^= 0xffffffffu;
		std::string snapshot_error;
		if (!ValidateResourceSnapshot(fixture.program, copy, &snapshot_error)) {
			snapshot_rejected++;   // outside the entry's precondition; not compared
			return;
		}
		mutations++;
		std::string a, b;
		const bool  full_ok = ValidateResourceSpecialization(fixture.program, copy, &a);
		const bool  mat_ok  = ValidateMaterializedResourceSpecialization(fixture.program, copy, &b);
		if (!full_ok) {
			rejected++;
		}
		if (full_ok != mat_ok || a != b) {
			Check(test, false, "decision differs after mutating dword " + std::to_string(dword) + ": '" +
			                       a + "' vs '" + b + "'");
		}
	};
	for (size_t i = 0; i < snapshot.buffers.size(); i++) {
		for (uint32_t d = 0; d < 8; d++) {
			mutate(&ResourceSnapshot::buffers, i, d);
		}
	}
	for (size_t i = 0; i < snapshot.images.size(); i++) {
		for (uint32_t d = 0; d < 8; d++) {
			mutate(&ResourceSnapshot::images, i, d);
		}
	}
	for (size_t i = 0; i < snapshot.samplers.size(); i++) {
		for (uint32_t d = 0; d < 8; d++) {
			mutate(&ResourceSnapshot::samplers, i, d);
		}
	}
	std::printf("[info]    %-56s compared mutations=%u (rejected by both=%u), outside precondition=%u\n",
	            test.c_str(), mutations, rejected, snapshot_rejected);
	Check(test, shape.buffers + shape.images == 0 || mutations > 0, "no mutations compared");
}

double Median(std::vector<double> values) {
	std::sort(values.begin(), values.end());
	return values[values.size() / 2];
}

void Benchmark(const Shape& shape, const Fixture& fixture, Env& env) {
	constexpr int rounds     = 7;
	constexpr int iterations = 20000;
	std::vector<double> old_sequence, new_sequence, snapshot_only;
	uint64_t            snapshot_allocations = 0;
	std::string         error;
	ResourceSnapshot    snapshot;
	for (int warm = 0; warm < 256; warm++) {
		(void)MaterializeResources(fixture.program, env.runtime, snapshot, &error);
		(void)ValidateResourceSpecialization(fixture.program, snapshot, &error);
		(void)ValidateMaterializedResourceSpecialization(fixture.program, snapshot, &error);
	}
	bool ok = true;
	for (int round = 0; round < rounds; round++) {
		auto start = std::chrono::steady_clock::now();
		for (int i = 0; i < iterations; i++) {
			ok = MaterializeResources(fixture.program, env.runtime, snapshot, &error) &&
			     ValidateResourceSpecialization(fixture.program, snapshot, &error) && ok;
		}
		old_sequence.push_back(
		    std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() /
		    iterations);

		start = std::chrono::steady_clock::now();
		for (int i = 0; i < iterations; i++) {
			ok = MaterializeResources(fixture.program, env.runtime, snapshot, &error) &&
			     ValidateMaterializedResourceSpecialization(fixture.program, snapshot, &error) && ok;
		}
		new_sequence.push_back(
		    std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() /
		    iterations);

		const auto allocations_before = t_allocations;
		start                         = std::chrono::steady_clock::now();
		for (int i = 0; i < iterations; i++) {
			ok = ValidateResourceSnapshot(fixture.program, snapshot, &error) && ok;
		}
		snapshot_only.push_back(
		    std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() /
		    iterations);
		snapshot_allocations += t_allocations - allocations_before;
	}
	Check(std::string("Benchmark: ") + shape.name, ok, "a timed validation failed: " + error);
	const auto old_ns = Median(old_sequence), new_ns = Median(new_sequence), snap_ns = Median(snapshot_only);
	std::printf("[bench]   %-22s buffers=%zu images=%zu | materialize+full %8.1f ns | materialize+specialization-only "
	            "%8.1f ns | delta %7.1f ns | ValidateResourceSnapshot alone %7.1f ns, %.2f allocations/call\n",
	            shape.name, snapshot.buffers.size(), snapshot.images.size(), old_ns, new_ns, old_ns - new_ns,
	            snap_ns, static_cast<double>(snapshot_allocations) / (rounds * iterations));
}

} // namespace

int main() {
	const Shape shapes[] = {{"small (4 buf, 2 img)", 4, 2}, {"medium (16 buf, 8 img)", 16, 8},
	                        {"buffers only (8 buf)", 8, 0}};
	for (const auto& shape: shapes) {
		Env              env;
		ResourceSnapshot snapshot;
		std::string      error;
		auto             fixture = BuildSpecialized(shape, env, snapshot, error);
		if (!fixture) {
			Check(std::string("Fixture: ") + shape.name, false, error);
			continue;
		}
		TestEquivalence(shape, *fixture, snapshot);
		Benchmark(shape, *fixture, env);
	}
	std::printf("Operation-level timings on synthetic IR; not a frame-time or FPS measurement.\n");
	if (g_failures != 0) {
		std::printf("%d materialized specialization check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("materialized specialization tests passed\n");
	return 0;
}
