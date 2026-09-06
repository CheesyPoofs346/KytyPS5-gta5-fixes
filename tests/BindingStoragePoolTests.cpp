// Correctness + allocation benchmark for the PreparedBindings storage pool.
//
// Exercises the REAL TakePooledStorage / ReturnPooledBindingStorage, not a re-implementation.
//
// Why this matters: the pool was sized for SERIAL draws, where one stage object is live at a
// time. Batched acquisition (DrawBatchQueue) builds every draw's bindings before recording
// consumes any, so up to 2 * kMaxBatchDraws stage objects coexist while the pool retains 8.
//
// Shape is taken from preserved gameplay logs, not invented:
//   draws/batch    73.9   (DrainCensus, armA-preserved.log)
//   buffers/draw   6.1    (DrawProfile, self-preserved.log)  -> ~3 per stage
//   image slots    1.82/draw (WorkerImages 22.0M slots / DrawCensus 12.06M draws) -> ~1 per stage
// user_data and flattened_srt sizes are NOT measured anywhere available, so they are set to
// plausible small values and flagged as the weakest part of the model.

#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

namespace {

// ---- exact allocation accounting -------------------------------------------------------
std::atomic<uint64_t> g_allocs {0};
std::atomic<uint64_t> g_bytes {0};
bool                  g_counting = false;

} // namespace

void* operator new(std::size_t size) {
	if (g_counting) {
		g_allocs.fetch_add(1, std::memory_order_relaxed);
		g_bytes.fetch_add(size, std::memory_order_relaxed);
	}
	if (void* p = std::malloc(size == 0 ? 1 : size)) {
		return p;
	}
	throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

void Check(const char* test, bool ok, const std::string& message) {
	if (!ok) {
		std::printf("[FAIL]    %-46s %s\n", test, message.c_str());
		g_failures++;
	}
}

// Measured per-stage shape. Kept in one place so the numbers above are traceable.
struct StageShape {
	size_t buffers   = 3;   // 6.1/draw across two stages
	size_t images    = 1;   // 1.82/draw across two stages
	size_t samplers  = 1;
	size_t flat_srt  = 24;  // NOT measured - see header comment
	size_t user_data = 24;  // NOT measured - see header comment
};

// Fill a PreparedBindings the way PrepareBindings/FindBuffers do: take pooled storage, then
// grow each vector to the stage's size. Only the growth can allocate.
void FillStage(PreparedBindings& prepared, const StageShape& shape) {
	TakePooledStorage(prepared);
	prepared.resources.buffers.resize(shape.buffers);
	prepared.resources.images.resize(shape.images);
	prepared.resources.samplers.resize(shape.samplers);
	prepared.buffer_ids.resize(shape.buffers);
	prepared.buffer_descriptors.resize(shape.buffers);
	prepared.buffer_sizes.resize(shape.buffers);
	prepared.flattened_srt.resize(shape.flat_srt);
	prepared.user_data.resize(shape.user_data);
}

// One drain: acquisition builds every stage object for the batch, THEN recording consumes them
// one at a time and returns their storage. That ordering is the whole point - it is what the
// serial-sized pool does not anticipate.
uint64_t RunDrain(size_t draws, const StageShape& shape, uint64_t* out_bytes) {
	std::vector<PreparedBindings> live;
	live.resize(draws * 2);   // vertex + pixel

	g_allocs.store(0, std::memory_order_relaxed);
	g_bytes.store(0, std::memory_order_relaxed);
	g_counting = true;

	for (auto& stage: live) {   // phase 2b: acquisition, all stages live at once
		FillStage(stage, shape);
	}
	for (auto& stage: live) {   // phase 3: recording consumes and returns
		ReturnPooledBindingStorage(stage);
	}

	g_counting = false;
	if (out_bytes != nullptr) {
		*out_bytes = g_bytes.load(std::memory_order_relaxed);
	}
	return g_allocs.load(std::memory_order_relaxed);
}

double TimeDrains(size_t draws, const StageShape& shape, size_t iterations) {
	// Warm the pool first so steady state is measured, not first touch.
	RunDrain(draws, shape, nullptr);
	const auto start = std::chrono::steady_clock::now();
	uint64_t   sink  = 0;
	for (size_t i = 0; i < iterations; i++) {
		sink += RunDrain(draws, shape, nullptr);
	}
	const auto elapsed = std::chrono::duration<double, std::milli>(
	                         std::chrono::steady_clock::now() - start)
	                         .count();
	// Consume the result so the work cannot be optimised away.
	if (sink == 0xdeadbeefu) {
		std::printf("");
	}
	return elapsed / static_cast<double>(iterations);
}

// ---- correctness -----------------------------------------------------------------------

void TestRecycledStorageIsEmptyButCapable() {
	const char* test = "Recycled storage is empty with capacity";
	ClearPreparedBindingsPool();
	StageShape shape;
	{
		PreparedBindings p;
		FillStage(p, shape);
		ReturnPooledBindingStorage(p);
	}
	PreparedBindings next;
	TakePooledStorage(next);
	Check(test, next.resources.buffers.empty() && next.buffer_ids.empty(),
	      "recycled vectors must be logically empty");
	Check(test, next.resources.buffers.capacity() >= shape.buffers,
	      "recycled vectors must retain capacity");
	std::printf("[host]    %-46s ok\n", test);
}

void TestReturnedObjectIsLeftEmpty() {
	const char* test = "Returned object keeps value semantics";
	ClearPreparedBindingsPool();
	PreparedBindings p;
	FillStage(p, StageShape {});
	ReturnPooledBindingStorage(p);
	Check(test, p.resources.buffers.empty() && p.user_data.empty(),
	      "the source must be moved-from, never aliased by the pool");
	std::printf("[host]    %-46s ok\n", test);
}

void TestLiveObjectsAreIndependent() {
	const char* test = "Concurrently live stages are independent";
	ClearPreparedBindingsPool();
	StageShape shape;
	std::vector<PreparedBindings> live(64);
	for (auto& s: live) {
		FillStage(s, shape);
	}
	for (size_t i = 0; i < live.size(); i++) {
		live[i].buffer_sizes[0] = i + 1;
	}
	bool distinct = true;
	for (size_t i = 0; i < live.size(); i++) {
		if (live[i].buffer_sizes[0] != i + 1) {
			distinct = false;
		}
		for (size_t j = i + 1; j < live.size(); j++) {
			if (live[i].buffer_sizes.data() == live[j].buffer_sizes.data()) {
				distinct = false;   // two live objects sharing one buffer
			}
		}
	}
	Check(test, distinct, "no two live stage objects may share storage");
	std::printf("[host]    %-46s ok\n", test);
}

void TestPoolIsBounded() {
	const char* test = "Pool retention is bounded";
	ClearPreparedBindingsPool();
	StageShape shape;
	for (size_t i = 0; i < 4096; i++) {
		PreparedBindings p;
		FillStage(p, shape);
		ReturnPooledBindingStorage(p);
	}
	Check(test, PreparedBindingsPoolSize() <= PreparedBindingsPoolCap(),
	      "pool must not grow past its cap");
	std::printf("[host]    %-46s ok  (size=%zu cap=%zu)\n", test, PreparedBindingsPoolSize(),
	            PreparedBindingsPoolCap());
}

void TestWarmupThenSteadyState() {
	const char* test = "Steady state allocates less than cold";
	ClearPreparedBindingsPool();
	StageShape shape;
	const auto cold  = RunDrain(8, shape, nullptr);
	const auto warm  = RunDrain(8, shape, nullptr);
	Check(test, warm <= cold, "a warmed pool must not allocate more than a cold one");
	std::printf("[host]    %-46s ok  (cold=%llu warm=%llu)\n", test,
	            static_cast<unsigned long long>(cold), static_cast<unsigned long long>(warm));
}

} // namespace

int main() {
	TestRecycledStorageIsEmptyButCapable();
	TestReturnedObjectIsLeftEmpty();
	TestLiveObjectsAreIndependent();
	TestPoolIsBounded();
	TestWarmupThenSteadyState();

	std::printf("\n=== allocation benchmark: pool cap = %zu ===\n", PreparedBindingsPoolCap());
	std::printf("  %-8s %-12s %-14s %-14s %s\n", "draws", "stage objs", "allocs/drain",
	            "bytes/drain", "ms/drain");
	const StageShape shape;
	for (const size_t draws: {size_t {1}, size_t {8}, size_t {32}, size_t {64}, size_t {74},
	                          size_t {256}}) {
		ClearPreparedBindingsPool();
		RunDrain(draws, shape, nullptr);   // warm
		uint64_t   bytes  = 0;
		const auto allocs = RunDrain(draws, shape, &bytes);
		const auto ms     = TimeDrains(draws, shape, draws >= 64 ? 200 : 2000);
		std::printf("  %-8zu %-12zu %-14llu %-14llu %.4f\n", draws, draws * 2,
		            static_cast<unsigned long long>(allocs),
		            static_cast<unsigned long long>(bytes), ms);
	}

	if (g_failures != 0) {
		std::printf("\n%d binding-storage check(s) FAILED\n", g_failures);
		return 1;
	}
	std::printf("\nall binding-storage checks passed\n");
	return 0;
}
