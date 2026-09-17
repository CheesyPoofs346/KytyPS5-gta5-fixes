// Contract tests for ShaderMaterializeStageRuntime after the duplicate snapshot validation was removed.
//
// The stage runtime now runs only the specialization checks after MaterializeResources, relying on
// MaterializeResources having validated exactly the snapshot it returns. These tests drive the REAL
// stage-runtime entry point (not the validators directly) through its production memory path: it
// never sets SrtRuntime::read_memory, so guest reads are direct host-memory reads. A fixed low
// VirtualAlloc region stands in for guest memory, and the buffer-only fixture reads at user-data
// addresses inside it. Cases: success (published resources equal an independent materialization and
// pass the FULL specialization check), materialization failure (untracked program; invalid buffer
// image alias caught by ValidateResourceSnapshot inside MaterializeResources), and specialization
// rejection (stale descriptor format). Every failure must leave the prior stage untouched and report
// the same error as the full checks. Synthetic IR; not captured from GTA V.

#include "common/emulatorConfig.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/shader.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <windows.h>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderMaterializeStageRuntime;
using Libs::Graphics::ShaderStageRuntime;
using Libs::Graphics::ShaderType;

int g_failures = 0;

void Check(const std::string& test, bool ok, const std::string& message) {
	if (!ok) {
		std::printf("[FAIL]    %-60s %s\n", test.c_str(), message.c_str());
		g_failures++;
	}
}

struct GuestRegion {
	uint64_t base = 0;
	size_t   size = 0x10000;
};

GuestRegion AllocateLowRegion() {
	GuestRegion region;
	for (uint64_t candidate = 0x30000000ull; candidate < 0x80000000ull; candidate += 0x01000000ull) {
		auto* p = VirtualAlloc(reinterpret_cast<LPVOID>(candidate), region.size,
		                       MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
		if (p != nullptr) {
			region.base = reinterpret_cast<uint64_t>(p);
			auto* words = static_cast<uint32_t*>(p);
			for (size_t i = 0; i < region.size / sizeof(uint32_t); i++) {
				words[i] = 0xA0000000u + static_cast<uint32_t>(i);
			}
			return region;
		}
	}
	return region;
}

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
	MemoryFlags AddMemory(ResourceKind kind, uint32_t pc) {
		MemoryInfo memory;
		memory.kind      = kind;
		const auto index = static_cast<uint32_t>(program.memory_info.size());
		program.memory_info.push_back(memory);
		return {index, pc};
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
};

enum class Prepare { Specialized, Untracked, InvalidAlias, StaleFormat };

// Builds a program and returns it as shared_ptr<const Program> the way the stage runtime receives it.
std::shared_ptr<const Program> MakeProgram(Prepare prepare, std::span<const uint32_t> user_data,
                                           std::string& error) {
	auto fixture = std::make_shared<Fixture>();
	for (uint32_t i = 0; i < 6; i++) {
		fixture->AddBuffer(i);
	}
	if (!BuildSrtPlan(fixture->program, &error)) {
		return nullptr;
	}
	if (prepare == Prepare::Untracked) {
		return std::shared_ptr<const Program>(fixture, &fixture->program);
	}
	if (!TrackResources(fixture->program, &error)) {
		return nullptr;
	}
	SrtRuntime runtime {};
	runtime.user_data = user_data;
	ResourceSnapshot first;
	if (!MaterializeResources(fixture->program, runtime, first, &error) ||
	    !SpecializeResources(fixture->program, first, &error)) {
		return nullptr;
	}
	if (prepare == Prepare::InvalidAlias) {
		fixture->program.info.buffers[0].image_alias = static_cast<uint32_t>(fixture->program.info.images.size() + 5);
	} else if (prepare == Prepare::StaleFormat) {
		fixture->program.info.buffers[0].descriptor_format =
		    static_cast<decltype(fixture->program.info.buffers[0].descriptor_format)>(
		        static_cast<uint32_t>(fixture->program.info.buffers[0].descriptor_format) + 1u);
	}
	return std::shared_ptr<const Program>(fixture, &fixture->program);
}

// Independent reference: MaterializeResources on the same program and memory path, plus the FULL checks.
struct Reference {
	bool             materialized = false;
	ResourceSnapshot snapshot;
	std::string      materialize_error;
	bool             full_ok = false;
	std::string      full_error;
};

Reference Independent(const Program& program, std::span<const uint32_t> user_data) {
	Reference  ref;
	SrtRuntime runtime {};
	runtime.user_data  = user_data;
	ref.materialized   = MaterializeResources(program, runtime, ref.snapshot, &ref.materialize_error);
	if (ref.materialized) {
		ref.full_ok = ValidateResourceSpecialization(program, ref.snapshot, &ref.full_error);
	}
	return ref;
}

bool SameSnapshot(const ResourceSnapshot& a, const ResourceSnapshot& b) {
	return a.buffers == b.buffers && a.images == b.images && a.samplers == b.samplers &&
	       a.flattened_srt == b.flattened_srt && a.user_data == b.user_data &&
	       a.indirect_images.size() == b.indirect_images.size();
}

void RunCase(const char* name, Prepare prepare, bool expect_success, const ShaderStageRuntime& prior,
             std::span<const uint32_t> user_data, ShaderStageRuntime* published_out) {
	const std::string test = std::string("Stage runtime: ") + name;
	std::string       build_error;
	auto              program = MakeProgram(prepare, user_data, build_error);
	if (!program) {
		Check(test, false, "fixture failed: " + build_error);
		return;
	}
	const auto reference = Independent(*program, user_data);

	ShaderStageRuntime stage = prior;
	std::string        error;
	const bool ok = ShaderMaterializeStageRuntime(program, user_data, 0, stage, &error, nullptr, nullptr);

	Check(test, ok == expect_success, std::string("returned ") + (ok ? "true" : "false") + ", error: " + error);
	if (ok) {
		Check(test, reference.materialized && reference.full_ok,
		      "published a stage the full checks reject: " + reference.materialize_error + reference.full_error);
		Check(test, stage.program == program && stage.resources != nullptr, "stage not published");
		if (stage.resources) {
			Check(test, SameSnapshot(*stage.resources, reference.snapshot),
			      "published resources differ from an independent materialization");
			std::string full_error;
			Check(test, ValidateResourceSpecialization(*program, *stage.resources, &full_error),
			      "published resources fail the full check: " + full_error);
		}
		if (published_out != nullptr) {
			*published_out = stage;
		}
	} else {
		Check(test, stage.program == prior.program && stage.resources == prior.resources,
		      "prior stage was modified on failure");
		const auto& expected = reference.materialized ? reference.full_error : reference.materialize_error;
		Check(test, !expected.empty() && error == expected,
		      "error '" + error + "' differs from the full checks' '" + expected + "'");
		if (prepare == Prepare::InvalidAlias || prepare == Prepare::Untracked) {
			Check(test, !reference.materialized, "expected MaterializeResources itself to fail");
		}
		if (prepare == Prepare::StaleFormat) {
			Check(test, reference.materialized && !reference.full_ok,
			      "expected materialization to succeed and the specialization check to reject");
		}
	}
	std::printf("[info]    %-60s ok=%d error='%s'\n", test.c_str(), ok ? 1 : 0, error.c_str());
}

} // namespace

int main() {
	const auto region = AllocateLowRegion();
	if (region.base == 0) {
		std::printf("could not allocate a low guest-memory region\n");
		return 1;
	}
	std::vector<uint32_t> user_data(64, static_cast<uint32_t>(region.base));
	std::printf("[info]    guest stand-in region at 0x%llx\n", static_cast<unsigned long long>(region.base));

	ShaderStageRuntime published;
	RunCase("success publishes a fully valid snapshot", Prepare::Specialized, true, {}, user_data, &published);
	if (!published) {
		Check("Stage runtime: prior stage for failure cases", false, "success case did not publish");
		return 1;
	}
	RunCase("untracked program fails in materialization", Prepare::Untracked, false, published, user_data, nullptr);
	RunCase("invalid image alias fails in materialization", Prepare::InvalidAlias, false, published, user_data, nullptr);
	RunCase("stale descriptor format rejected by specialization", Prepare::StaleFormat, false, published, user_data, nullptr);

	if (g_failures != 0) {
		std::printf("%d stage runtime check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("stage runtime contract tests passed\n");
	return 0;
}
