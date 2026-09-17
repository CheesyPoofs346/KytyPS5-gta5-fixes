// What does one ResourceSnapshot actually cost to construct, publish and destroy?
//
// The claim this exists to test is that per-draw snapshot allocation churn is worth attacking. That
// claim was previously an ESTIMATE derived from "ResourceSnapshot holds six containers, so expect
// about six allocations". That inference is not sound: containers may be empty (an empty vector
// allocates nothing), sizes vary per stage shape, make_shared collapses the control block into the
// object allocation, and moved-from scratch buffers hand over storage rather than allocating.
//
// So this counts REAL allocations by replacing global operator new/delete, over the actual path:
//
//     MaterializeResources(...)                                   construction
//     make_shared<const ResourceSnapshot>(std::move(snapshot))    publication
//     shared_ptr release                                          destruction
//
// Stage shapes are CHOSEN, not observed from GTA V - no per-shader resource histogram exists for
// the game, so these are a spread rather than a workload model, and are labelled as such.
//
// The batch case matters separately: holding many snapshots alive at once denies the allocator the
// immediate free-list reuse it gets when each snapshot dies before the next is built, which is
// closer to what a real frame does with snapshots retained by in-flight draws.

#include "common/emulatorConfig.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

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

// ---------------------------------------------------------------- allocation counting

namespace {
struct AllocCounters {
	uint64_t allocs = 0;
	uint64_t frees  = 0;
	uint64_t bytes  = 0;
};

// Not atomic and not thread-safe on purpose: this benchmark is single-threaded, and making these
// atomic would add cost to the very path being measured.
AllocCounters g_counters;
bool          g_counting = false;

void CountAlloc(size_t size) {
	if (g_counting) {
		g_counters.allocs++;
		g_counters.bytes += size;
	}
}
void CountFree() {
	if (g_counting) {
		g_counters.frees++;
	}
}
} // namespace

void* operator new(size_t size) {
	CountAlloc(size);
	if (void* p = std::malloc(size == 0 ? 1 : size)) {
		return p;
	}
	throw std::bad_alloc();
}
void* operator new[](size_t size) { return ::operator new(size); }
void  operator delete(void* p) noexcept {
    if (p != nullptr) {
        CountFree();
    }
    std::free(p);
}
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete(void* p, size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, size_t) noexcept { ::operator delete(p); }

// ---------------------------------------------------------------- fixtures

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderType;

int g_failures = 0;

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
		const auto address = Emit(ValueOpcode::GetAddressResource,
		                          {UserData(index % 8), Value(0u)}, MemoryFlags {0, pc});
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
		const auto pc = 0x4000u + index * 0x40u;
		const auto d  = Value(0u);
		const auto image =
		    Emit(ValueOpcode::GetImageResource,
		         {Value(index + 1u), d, Value(0x2000u), Value(0x1000u), d, d, d, d},
		         MemoryFlags {0, pc});
		const auto sampler = Emit(ValueOpcode::GetSamplerResource, {d, d, d, d},
		                          MemoryFlags {0, pc});
		MemoryInfo sample;
		sample.kind            = ResourceKind::Image;
		sample.image_dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D;
		const auto sampled     = Emit(ValueOpcode::ImageSampleRaw, {image, sampler, ImageAddress()},
		                              AddMemory(sample, pc));
		const auto component = Emit(ValueOpcode::CompositeExtractU32x4, {sampled, Value(0u)});
		Emit(ValueOpcode::ReferenceU32, {component});
	}

	bool Plan(std::string& error) {
		return BuildSrtPlan(program, &error) && TrackResources(program, &error);
	}
};

struct Shape {
	const char* name;
	uint32_t    buffers;
	uint32_t    images;   // each adds one image + one sampler
};

std::unique_ptr<Fixture> Build(const Shape& shape, std::string& error) {
	auto fixture = std::make_unique<Fixture>();
	for (uint32_t i = 0; i < shape.buffers; i++) {
		fixture->AddBuffer(i);
	}
	for (uint32_t i = 0; i < shape.images; i++) {
		fixture->AddSampledImage(i);
	}
	if (!fixture->Plan(error)) {
		return nullptr;
	}
	return fixture;
}

// ---------------------------------------------------------------- measurement

struct Result {
	double   ns_per_snapshot = 0;
	double   allocs          = 0;
	double   frees           = 0;
	double   bytes           = 0;
	uint32_t buffers         = 0;
	uint32_t images          = 0;
	uint32_t samplers        = 0;
	bool     ok              = false;
};

// batch = how many snapshots are held alive simultaneously before releasing them all.
Result Measure(Fixture& fixture, size_t iterations, size_t batch) {
	Result out;
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

	std::string error;
	// Warm up: first call populates the thread_local scratch pools inside MaterializeResources,
	// which would otherwise be counted as snapshot cost.
	for (int i = 0; i < 64; i++) {
		ResourceSnapshot warm;
		if (!MaterializeResources(fixture.program, runtime, warm, &error)) {
			std::printf("  MaterializeResources failed: %s\n", error.c_str());
			return out;
		}
		auto published = std::make_shared<const ResourceSnapshot>(std::move(warm));
		out.buffers    = static_cast<uint32_t>(published->buffers.size());
		out.images     = static_cast<uint32_t>(published->images.size());
		out.samplers   = static_cast<uint32_t>(published->samplers.size());
	}

	// The holder is allocated and sized OUTSIDE the counted region so its own storage is not
	// attributed to the snapshots.
	std::vector<std::shared_ptr<const ResourceSnapshot>> held;
	held.reserve(batch);

	g_counters = {};
	g_counting = true;
	const auto start = std::chrono::steady_clock::now();
	for (size_t i = 0; i < iterations; i++) {
		ResourceSnapshot snapshot;
		if (!MaterializeResources(fixture.program, runtime, snapshot, &error)) {
			g_counting = false;
			std::printf("  MaterializeResources failed mid-run\n");
			return out;
		}
		held.push_back(std::make_shared<const ResourceSnapshot>(std::move(snapshot)));
		if (held.size() == batch) {
			held.clear();   // destruction of the whole batch
		}
	}
	held.clear();
	const auto elapsed = std::chrono::steady_clock::now() - start;
	g_counting         = false;

	out.ns_per_snapshot =
	    std::chrono::duration<double, std::nano>(elapsed).count() / static_cast<double>(iterations);
	out.allocs = static_cast<double>(g_counters.allocs) / static_cast<double>(iterations);
	out.frees  = static_cast<double>(g_counters.frees) / static_cast<double>(iterations);
	out.bytes  = static_cast<double>(g_counters.bytes) / static_cast<double>(iterations);
	out.ok     = true;
	return out;
}


// Isolates the SRT EVALUATION from the snapshot assembly around it.
//
// MaterializeResources does both: it evaluates the descriptor sources (the compiled path) and then
// assembles a ResourceSnapshot from the results. Attributing the whole per-snapshot cost to
// allocation churn would be wrong, so this arm runs only the evaluation, through the same
// thread_local pooled vectors the real caller uses, and allocates no snapshot at all.
Result MeasureEvaluationOnly(Fixture& fixture, size_t iterations) {
	Result     out;
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

	// Same request set the real caller builds, and the same pooled destinations.
	static thread_local std::vector<DescriptorSourceRequest> requests;
	static thread_local std::vector<DescriptorValue>         values;
	static thread_local std::vector<uint32_t>                flattened;
	static thread_local std::vector<uint8_t>                 clean;
	requests.clear();
	for (const auto& buffer: fixture.program.info.buffers) {
		requests.push_back({buffer.source, buffer.first_use_pc});
	}
	// These fixtures contain no indirect images, so every image contributes a request. The real
	// caller skips indirect ones; that branch is not exercised here.
	for (const auto& image: fixture.program.info.images) {
		requests.push_back({image.source, image.first_use_pc});
	}
	for (const auto& sampler: fixture.program.info.samplers) {
		requests.push_back({sampler.source, sampler.first_use_pc});
	}
	clean.assign(fixture.program.srt_reads.size(), 0);

	std::string error;
	for (int i = 0; i < 64; i++) {
		values.clear();
		flattened.clear();
		if (!EvaluateRuntimeSources(fixture.program, requests, runtime, values, flattened, clean,
		                            &error)) {
			std::printf("  evaluation failed: %s\n", error.c_str());
			return out;
		}
	}

	g_counters       = {};
	g_counting       = true;
	const auto start = std::chrono::steady_clock::now();
	for (size_t i = 0; i < iterations; i++) {
		values.clear();
		flattened.clear();
		if (!EvaluateRuntimeSources(fixture.program, requests, runtime, values, flattened, clean,
		                            &error)) {
			g_counting = false;
			return out;
		}
	}
	const auto elapsed = std::chrono::steady_clock::now() - start;
	g_counting         = false;

	out.ns_per_snapshot =
	    std::chrono::duration<double, std::nano>(elapsed).count() / static_cast<double>(iterations);
	out.allocs = static_cast<double>(g_counters.allocs) / static_cast<double>(iterations);
	out.frees  = static_cast<double>(g_counters.frees) / static_cast<double>(iterations);
	out.bytes  = static_cast<double>(g_counters.bytes) / static_cast<double>(iterations);
	out.ok     = true;
	return out;
}


// How much of the snapshot cost is the ALLOCATOR, as opposed to the copying that any design still
// has to do?
//
// "snapshot ns" above is an upper bound: it includes assembling and copying descriptor values,
// which neither pooling nor a single-arena layout removes. Only the malloc/free traffic is
// removable. This replays exactly the observed pattern - 8 snapshot-attributable allocations of the
// observed size distribution, allocated together and freed together, at a chosen retention depth -
// and nothing else. It is the ceiling on what any allocation-reduction design can win.
double MeasureAllocatorOnly(size_t iterations, size_t batch) {
	const size_t sizes[] = {160, 208, 208, 96, 256, 64, 32, 48};
	const size_t per     = std::size(sizes);

	// Bookkeeping is allocated ONCE, outside the timed region. An earlier version pushed a fresh
	// std::vector per iteration inside the loop, which allocated more than the pattern it was
	// meant to isolate and reported a ceiling higher than the whole snapshot cost.
	std::vector<void*> live(batch * per, nullptr);

	size_t     held  = 0;
	const auto start = std::chrono::steady_clock::now();
	for (size_t i = 0; i < iterations; i++) {
		for (size_t s = 0; s < per; s++) {
			live[held * per + s] = std::malloc(sizes[s]);
		}
		if (++held == batch) {
			for (void*& p: live) {
				std::free(p);
				p = nullptr;
			}
			held = 0;
		}
	}
	for (size_t i = 0; i < held * per; i++) {
		std::free(live[i]);
	}
	const auto elapsed = std::chrono::steady_clock::now() - start;
	return std::chrono::duration<double, std::nano>(elapsed).count() /
	       static_cast<double>(iterations);
}

} // namespace

int main() {
	Config::SetCompiledSrtForTest(true);

	std::printf("ResourceSnapshot: construct + publish (make_shared) + destroy, per snapshot.\n");
	std::printf("Allocations are COUNTED via replaced global operator new/delete, not inferred.\n");
	std::printf("Stage shapes are chosen, not measured from GTA V.\n\n");

	const Shape shapes[] = {
	    {"empty", 0, 0},
	    {"tiny (1 buf)", 1, 0},
	    {"small (4 buf, 2 img)", 4, 2},
	    {"medium (8 buf, 4 img)", 8, 4},
	    {"large (24 buf, 8 img)", 24, 8},
	};

	std::printf("%-24s %8s %8s %8s | %7s %7s %9s | %9s\n", "shape", "buffers", "images", "samplers",
	            "allocs", "frees", "bytes", "ns/snap");
	std::printf("%s\n", std::string(96, '-').c_str());

	for (const auto& shape: shapes) {
		std::string error;
		auto        fixture = Build(shape, error);
		if (!fixture) {
			std::printf("%-24s SKIPPED - plan failed: %s\n", shape.name, error.c_str());
			continue;
		}
		const auto r = Measure(*fixture, 20000, 1);
		if (!r.ok) {
			g_failures++;
			continue;
		}
		std::printf("%-24s %8u %8u %8u | %7.2f %7.2f %9.0f | %9.0f\n", shape.name, r.buffers,
		            r.images, r.samplers, r.allocs, r.frees, r.bytes, r.ns_per_snapshot);
	}

	std::printf("\nBatch retention (medium shape): many snapshots alive at once, so the allocator\n"
	            "cannot immediately reuse the block that the previous snapshot just freed.\n\n");
	std::printf("%-24s %7s %7s %9s | %9s\n", "batch (held alive)", "allocs", "frees", "bytes",
	            "ns/snap");
	std::printf("%s\n", std::string(66, '-').c_str());
	{
		std::string error;
		auto        fixture = Build({"medium", 8, 4}, error);
		if (!fixture) {
			std::printf("SKIPPED - plan failed: %s\n", error.c_str());
			g_failures++;
		} else {
			for (size_t batch: {size_t {1}, size_t {8}, size_t {64}, size_t {512}}) {
				const auto r = Measure(*fixture, 20000, batch);
				if (!r.ok) {
					g_failures++;
					continue;
				}
				std::printf("%-24zu %7.2f %7.2f %9.0f | %9.0f\n", batch, r.allocs, r.frees, r.bytes,
				            r.ns_per_snapshot);
			}
		}
	}

	std::printf("\nDecomposition: how much of the per-snapshot cost is the SRT evaluation itself\n"
	            "(which compiled materialization already optimised) versus the snapshot assembly,\n"
	            "publication and destruction around it? Only the latter is what a pool could save.\n\n");
	std::printf("%-24s %7s %7s | %9s %9s %9s\n", "shape", "allocs", "eval", "full ns", "eval ns",
	            "snapshot ns");
	std::printf("%s\n", std::string(74, '-').c_str());
	for (const auto& shape: shapes) {
		std::string error;
		auto        fixture = Build(shape, error);
		if (!fixture) {
			continue;
		}
		const auto full = Measure(*fixture, 20000, 1);
		const auto eval = MeasureEvaluationOnly(*fixture, 20000);
		if (!full.ok || !eval.ok) {
			g_failures++;
			continue;
		}
		std::printf("%-24s %7.2f %7.2f | %9.0f %9.0f %9.0f\n", shape.name, full.allocs, eval.allocs,
		            full.ns_per_snapshot, eval.ns_per_snapshot,
		            full.ns_per_snapshot - eval.ns_per_snapshot);
	}

	std::printf("\nSynthetic 8-malloc replay - NOT a valid ceiling. Reported for what it rules out:\n"
	            "it costs MORE per snapshot than the whole measured snapshot-attributable time\n"
	            "above, so the real assembly is already cheaper than a naive malloc of the same\n"
	            "count (vector growth reuse, size classes, and flattened_srt being MOVED in\n"
	            "rather than allocated). Use the measured snapshot ns column as the bound.\n\n");
	std::printf("%-24s %12s\n", "retention depth", "ns/snapshot");
	std::printf("%s\n", std::string(38, '-').c_str());
	for (size_t batch: {size_t {1}, size_t {8}, size_t {64}, size_t {256}}) {
		std::printf("%-24zu %12.0f\n", batch, MeasureAllocatorOnly(200000, batch));
	}

	std::printf("\nns/snap is construction+publication+destruction. A draw does this twice (vertex\n"
	            "and pixel). Frame impact is snapshots/frame x cost and is NOT measured here.\n"
	            "'snapshot ns' is the upper bound on what perfect recycling could remove, and even\n"
	            "that includes copying values into the snapshot, which recycling does not avoid.\n");
	return g_failures == 0 ? 0 : 1;
}
