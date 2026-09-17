// Is walk-only parallelism worth dispatching, now that compiled SRT made the walk cheaper?
//
// Phase 2 of the draw-queue drain hands each draw's resource walk (MaterializeResources) to
// DrawWorkerPool::ParallelFor. That pool needs a Vulkan device, so it cannot be built offline; the
// dispatch here is a FAITHFUL REPLICA of its algorithm - same mutex + condition_variable wake,
// same atomic work-stealing counter, same "caller participates as runner 0", same
// kInlineThreshold - not the pool itself. Dispatch cost is therefore representative, not identical.
//
// What the real pool assumes, from its own comment: "an item of work here is the shader resource
// walk at roughly 3.4 us" and a wake costs "tens of microseconds". That 3.4 us is an
// interpreter-era figure. The compiled-ON diagnostic puts SrtEvaluate at 1131 ns/draw, so the item
// is now smaller and the batch needed to amortise a wake is correspondingly larger. This measures
// where the crossover actually falls.
//
// Outputs are compared item-by-item between the serial and parallel arms; a timing result from
// arms that disagree would be meaningless.

#include "common/emulatorConfig.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderType;

int g_failures = 0;

// ---------------------------------------------------------------- replica of DrawWorkerPool

// Mirrors DrawWorkerPool::ParallelFor: caller participates as runner 0, workers steal items from
// one atomic counter, a generation counter plus condition_variable wakes them, and batches at or
// below the inline threshold never touch a worker at all.
class DispatchReplica {
public:
	static constexpr uint32_t kInlineThreshold = 16;

	explicit DispatchReplica(uint32_t worker_count) {
		for (uint32_t i = 0; i < worker_count; i++) {
			m_threads.emplace_back([this, i]() { Run(i + 1); });
		}
	}
	~DispatchReplica() {
		{
			std::lock_guard lock(m_mutex);
			m_quit = true;
			m_generation++;
		}
		m_work_available.notify_all();
		for (auto& thread: m_threads) {
			thread.join();
		}
	}

	void ParallelFor(uint32_t count, const std::function<void(uint32_t, uint32_t)>& body) {
		if (count == 0) {
			return;
		}
		if (count <= kInlineThreshold || m_threads.empty()) {
			m_inline_batches++;
			for (uint32_t i = 0; i < count; i++) {
				body(i, 0);
			}
			return;
		}
		m_parallel_batches++;
		{
			std::lock_guard lock(m_mutex);
			m_body  = &body;
			m_count = count;
			m_next.store(0, std::memory_order_relaxed);
			m_outstanding.store(static_cast<uint32_t>(m_threads.size()) + 1,
			                    std::memory_order_release);
			m_generation++;
		}
		m_work_available.notify_all();

		for (;;) {
			const auto item = m_next.fetch_add(1, std::memory_order_relaxed);
			if (item >= count) {
				break;
			}
			body(item, 0);
		}
		if (m_outstanding.fetch_sub(1, std::memory_order_acq_rel) != 1) {
			std::unique_lock lock(m_done_mutex);
			m_done.wait(lock, [this]() { return m_outstanding.load(std::memory_order_acquire) == 0; });
		}
	}

	uint64_t InlineBatches() const { return m_inline_batches; }
	uint64_t ParallelBatches() const { return m_parallel_batches; }

private:
	void Run(uint32_t worker) {
		uint64_t seen = 0;
		for (;;) {
			const std::function<void(uint32_t, uint32_t)>* body = nullptr;
			uint32_t                                       count = 0;
			{
				std::unique_lock lock(m_mutex);
				m_work_available.wait(lock, [this, &seen]() { return m_generation != seen; });
				seen = m_generation;
				if (m_quit) {
					return;
				}
				body  = m_body;
				count = m_count;
			}
			if (body == nullptr) {
				continue;
			}
			for (;;) {
				const auto item = m_next.fetch_add(1, std::memory_order_relaxed);
				if (item >= count) {
					break;
				}
				(*body)(item, worker);
			}
			if (m_outstanding.fetch_sub(1, std::memory_order_acq_rel) == 1) {
				std::lock_guard lock(m_done_mutex);
				m_done.notify_all();
			}
		}
	}

	std::vector<std::thread>                       m_threads;
	std::mutex                                     m_mutex;
	std::condition_variable                        m_work_available;
	std::mutex                                     m_done_mutex;
	std::condition_variable                        m_done;
	const std::function<void(uint32_t, uint32_t)>* m_body = nullptr;
	uint32_t                                       m_count = 0;
	std::atomic<uint32_t>                          m_next {0};
	std::atomic<uint32_t>                          m_outstanding {0};
	uint64_t                                       m_generation = 0;
	bool                                           m_quit       = false;
	uint64_t                                       m_inline_batches   = 0;
	uint64_t                                       m_parallel_batches = 0;
};

// ---------------------------------------------------------------- fixtures

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

struct Builder {
	Program program;
	Block*  block = nullptr;

	Builder() {
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
	void AddBuffer(uint32_t index, uint32_t salt) {
		const auto pc      = 0x1000u + index * 0x40u;
		const auto address = Emit(ValueOpcode::GetAddressResource,
		                          {UserData(index % 8), Value(0u)}, MemoryFlags {0, pc});
		const auto word    = Emit(ValueOpcode::LoadAddressU32,
		                          {address, Value(index * 4u), Value(0u), Value(true)},
		                          AddMemory(ResourceKind::ScalarAddress, pc));
		const auto sum     = Emit(ValueOpcode::IAdd32, {word, Value(index + salt + 1u)});
		const auto buffer  = Emit(ValueOpcode::GetBufferResource,
		                          {Value(0u), Value(0u), Value(64u), sum}, MemoryFlags {0, pc});
		Emit(ValueOpcode::LoadBufferU32, {buffer, Value(0u), Value(0u), Value(0u), Value(true)},
		     AddMemory(ResourceKind::Buffer, pc));
	}
};

// One "item" is one draw's resource walk: a program plus the runtime it is resolved against.
struct Item {
	std::unique_ptr<Builder>             builder;
	std::vector<DescriptorSourceRequest> requests;
	TestMemory                           memory;
	std::vector<uint32_t>                user_data;
};

std::vector<Item> BuildBatch(uint32_t count, uint32_t buffers_per_program) {
	std::vector<Item> items;
	items.reserve(count);
	for (uint32_t i = 0; i < count; i++) {
		Item item;
		item.builder = std::make_unique<Builder>();
		for (uint32_t b = 0; b < buffers_per_program; b++) {
			item.builder->AddBuffer(b, i % 4);
		}
		std::string error;
		if (!BuildSrtPlan(item.builder->program, &error) ||
		    !TrackResources(item.builder->program, &error)) {
			std::printf("  plan failed: %s\n", error.c_str());
			return {};
		}
		for (const auto& buffer: item.builder->program.info.buffers) {
			item.requests.push_back({buffer.source, buffer.first_use_pc});
		}
		for (uint32_t w = 0; w < item.memory.words.size(); w++) {
			item.memory.words[w] = 0xA0000000u + i * 0x100u + w;
		}
		item.user_data.assign(64, 0x1000u);
		items.push_back(std::move(item));
	}
	return items;
}

SrtRuntime RuntimeFor(Item& item) {
	SrtRuntime runtime {};
	runtime.user_data                  = item.user_data;
	runtime.read_memory                = ReadTestMemory;
	runtime.read_specialization_memory = ReadTestMemory;
	runtime.userdata                   = &item.memory;
	return runtime;
}

bool SameSnapshot(const ResourceSnapshot& a, const ResourceSnapshot& b) {
	return a.buffers == b.buffers && a.images == b.images && a.samplers == b.samplers &&
	       a.flattened_srt == b.flattened_srt && a.user_data == b.user_data;
}

// ---------------------------------------------------------------- measurement

struct Arm {
	double   ns_per_item = 0;
	uint64_t parallel_batches = 0;
	bool     ok = false;
};

Arm RunArm(std::vector<Item>& items, std::vector<ResourceSnapshot>& out, DispatchReplica* pool,
           size_t repeats) {
	Arm arm;
	const auto count = static_cast<uint32_t>(items.size());
	out.assign(count, ResourceSnapshot {});

	const auto body = [&items, &out](uint32_t index, uint32_t) {
		auto        runtime = RuntimeFor(items[index]);
		std::string error;
		MaterializeResources(items[index].builder->program, runtime, out[index], &error);
	};

	// Warm: first call per program compiles the SRT and spends verification budget.
	for (int i = 0; i < 2; i++) {
		if (pool != nullptr) {
			pool->ParallelFor(count, body);
		} else {
			for (uint32_t j = 0; j < count; j++) {
				body(j, 0);
			}
		}
	}
	for (auto& item: items) {
		if (item.builder->program.compiled_state) {
			item.builder->program.compiled_state->verify_left.store(0);
		}
	}

	const auto before = pool != nullptr ? pool->ParallelBatches() : 0;
	const auto start  = std::chrono::steady_clock::now();
	for (size_t r = 0; r < repeats; r++) {
		if (pool != nullptr) {
			pool->ParallelFor(count, body);
		} else {
			for (uint32_t j = 0; j < count; j++) {
				body(j, 0);
			}
		}
	}
	const auto elapsed = std::chrono::steady_clock::now() - start;
	arm.parallel_batches = pool != nullptr ? pool->ParallelBatches() - before : 0;
	arm.ns_per_item = std::chrono::duration<double, std::nano>(elapsed).count() /
	                  static_cast<double>(repeats * count);
	arm.ok = true;
	return arm;
}

} // namespace

int main() {
	Config::SetCompiledSrtForTest(true);

	std::printf("Phase-2 walk parallelism with compiled SRT: serial vs 2 workers.\n");
	std::printf("Dispatch is a faithful REPLICA of DrawWorkerPool::ParallelFor (same wake, same\n");
	std::printf("work-stealing, caller as runner 0, kInlineThreshold=%u), not the pool itself.\n\n",
	            DispatchReplica::kInlineThreshold);

	DispatchReplica pool(2);

	std::printf("%-8s %-10s %12s %12s %8s %9s %s\n", "batch", "buffers", "serial ns", "2wkr ns",
	            "speedup", "dispatch", "note");
	std::printf("%s\n", std::string(86, '-').c_str());

	for (const uint32_t batch: {4u, 16u, 17u, 32u, 64u, 128u, 256u}) {
		auto items = BuildBatch(batch, 8);
		if (items.empty()) {
			g_failures++;
			continue;
		}
		std::vector<ResourceSnapshot> serial_out;
		std::vector<ResourceSnapshot> parallel_out;

		const size_t repeats = std::max<size_t>(1, 200000u / batch);
		const auto   serial   = RunArm(items, serial_out, nullptr, repeats);
		const auto   parallel = RunArm(items, parallel_out, &pool, repeats);
		if (!serial.ok || !parallel.ok) {
			g_failures++;
			continue;
		}

		bool identical = serial_out.size() == parallel_out.size();
		for (size_t i = 0; identical && i < serial_out.size(); i++) {
			identical = SameSnapshot(serial_out[i], parallel_out[i]);
		}
		if (!identical) {
			std::printf("%-8u OUTPUTS DIFFER - timing not reported\n", batch);
			g_failures++;
			continue;
		}

		const char* note = parallel.parallel_batches == 0
		                       ? "inline: never dispatched, arms are the same code"
		                       : "";
		std::printf("%-8u %-10u %12.1f %12.1f %7.2fx %9llu %s\n", batch, 8u, serial.ns_per_item,
		            parallel.ns_per_item, serial.ns_per_item / parallel.ns_per_item,
		            static_cast<unsigned long long>(parallel.parallel_batches), note);
	}

	std::printf("\nOutputs were compared item-by-item in every row reported above.\n");
	std::printf("Batches of %u or fewer never reach a worker by design, so those rows measure the\n"
	            "same code twice. The real drain census recorded 43%% of drains at 16 or fewer.\n",
	            DispatchReplica::kInlineThreshold);
	return g_failures == 0 ? 0 : 1;
}
