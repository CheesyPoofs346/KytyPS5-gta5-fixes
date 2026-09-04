#include "graphics/host_gpu/renderer/drawBatchQueue.h"

#include "common/emulatorConfig.h"

#include <cstdio>
#include "graphics/host_gpu/renderer/drawWorkerContext.h"
#include "graphics/host_gpu/renderer/drawWorkerPool.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

namespace Libs::Graphics {

// This type is constructed, moved and stored once per draw, so its size is load-bearing: it was
// 10496 bytes when PreparedShaders was embedded, which cost ~3.2 us/draw on its own.
static_assert(sizeof(QueuedDraw) <= 128, "QueuedDraw must stay small - see m_prepared");

// Defined in textureCache.cpp; declared here rather than in the public header because it is a
// diagnostic, not part of the cache's interface.
void ReportHashCensus();
namespace {

// ParallelFor pays a fixed dispatch cost per batch, so its value depends entirely on how many
// draws a batch actually carries. If batches are being cut to a handful of draws the dispatch
// cannot pay for itself, and that is measured here rather than assumed.
struct DrainCensus {
	uint64_t drains        = 0;
	uint64_t draws         = 0;
	uint64_t prepare_cyc   = 0;   // phase 1, serial lookup
	uint64_t parallel_cyc  = 0;   // phase 2, the walk across workers
	uint64_t record_cyc    = 0;   // phase 3, serial recording
	uint64_t buckets[6]    = {};  // 1, 2-4, 5-16, 17-64, 65-255, 256
};

thread_local DrainCensus t_drain;

void NoteBatchSize(size_t size) {
	const auto bucket = size <= 1     ? 0
	                    : size <= 4   ? 1
	                    : size <= 16  ? 2
	                    : size <= 64  ? 3
	                    : size <= 255 ? 4
	                                  : 5;
	t_drain.buckets[bucket]++;
}


void ReportDrainCensus() {
	if (t_drain.drains % 2000 != 0) {
		return;
	}
	const auto drains = static_cast<double>(t_drain.drains);
	const auto total  = static_cast<double>(t_drain.prepare_cyc + t_drain.parallel_cyc +
                                           t_drain.record_cyc);
	std::printf("DrainCensus: drains=%llu draws/batch=%.1f | phase1_prepare=%.1f%% "
	            "phase2_walk=%.1f%% phase3_record=%.1f%% | sizes 1:%llu 2-4:%llu 5-16:%llu "
	            "17-64:%llu 65-255:%llu 256:%llu\n",
	            static_cast<unsigned long long>(t_drain.drains),
	            static_cast<double>(t_drain.draws) / drains,
	            total > 0 ? 100.0 * static_cast<double>(t_drain.prepare_cyc) / total : 0.0,
	            total > 0 ? 100.0 * static_cast<double>(t_drain.parallel_cyc) / total : 0.0,
	            total > 0 ? 100.0 * static_cast<double>(t_drain.record_cyc) / total : 0.0,
	            static_cast<unsigned long long>(t_drain.buckets[0]),
	            static_cast<unsigned long long>(t_drain.buckets[1]),
	            static_cast<unsigned long long>(t_drain.buckets[2]),
	            static_cast<unsigned long long>(t_drain.buckets[3]),
	            static_cast<unsigned long long>(t_drain.buckets[4]),
	            static_cast<unsigned long long>(t_drain.buckets[5]));
	std::fflush(stdout);
}

} // namespace

void DrawBatchQueue::Drain(RenderExecutor& executor, CommandBuffer& buffer) {
	if (m_draws.empty()) {
		return;
	}
	m_batches++;
	t_drain.drains++;
	t_drain.draws += m_draws.size();
	NoteBatchSize(m_draws.size());

	// Cleared before translating so a draw that triggers a flush mid-drain cannot re-enter and
	// translate these entries twice.
	std::vector<QueuedDraw> draws;
	draws.swap(m_draws);

	// const_cast is safe here and only here: the draw path reads its register state and never
	// writes it - checked across every GetRegisters/GetShaders/GetUserConfig use - which is what
	// lets consecutive draws share one immutable snapshot by pointer.
	const auto bind = [&buffer](const QueuedDraw& draw) {
		auto& snapshot = const_cast<DrawStateSnapshot&>(*draw.snapshot);
		return buffer.SwapRegisterView(CommandBuffer::RegisterView {
		    &snapshot.context, &snapshot.user_config, &snapshot.shaders});
	};

	// Opt-in. Off, the drain is a plain serial walk of the queue and the main-thread path is
	// byte-for-byte what it was before any of this existed.
	const auto workers = Config::ParallelResolutionEnabled() ? Config::DrawWorkerCount() : 1u;
	if (workers > 1) {
		// Phase 1, serial: locate both permutations per draw. The lookup takes the program-cache
		// mutex, so it cannot be part of the parallel phase.
		// Sized only on the parallel path, and kept across drains so the ~10 KB entries are
		// allocated once rather than per batch.
		if (m_prepared.size() < draws.size()) {
			m_prepared.resize(draws.size());
		}
		const auto prepare_start = __builtin_ia32_rdtsc();
		for (size_t i = 0; i < draws.size(); i++) {
			const auto previous = bind(draws[i]);
			m_prepared[i] = PreparedShaders {};
			executor.PrepareQueuedShaders(buffer, m_prepared[i]);
			buffer.SwapRegisterView(previous);
		}
		t_drain.prepare_cyc += __builtin_ia32_rdtsc() - prepare_start;

		// Phase 2, parallel: the resource walk. 3.4 us/draw and the only phase in a draw that
		// touches no command buffer, no image layout and no cache - which is exactly why this one
		// can be handed out while recording still cannot.
		//
		// No register view is bound here on purpose: the walk reads what PrepareQueuedShaders
		// already captured into prepared, not the live registers, so workers never race the view
		// the serial phases swap.
		const auto parallel_start = __builtin_ia32_rdtsc();
		auto& pool = buffer.GetContext().GetDrawWorkerPool();
		auto& prepared = m_prepared;
		pool.ParallelFor(static_cast<uint32_t>(draws.size()),
		                 [&prepared, &executor](uint32_t index, uint32_t worker) {
			                 ScopedDrawWorker slot(worker);
			                 executor.ResolveQueuedShaders(prepared[index]);
		                 });
		t_drain.parallel_cyc += __builtin_ia32_rdtsc() - parallel_start;
	}

	// Phase 3, serial: record, in guest order. A draw whose permutation needed compiling arrives
	// with prepared.valid false and resolves inline, exactly as it did before.
	const auto record_start = __builtin_ia32_rdtsc();
	const bool have_prepared = workers > 1 && m_prepared.size() >= draws.size();
	for (size_t i = 0; i < draws.size(); i++) {
		auto&      draw     = draws[i];
		const auto previous = bind(draw);
		const auto* prepared =
		    have_prepared && m_prepared[i].valid ? &m_prepared[i] : nullptr;
		executor.DrawIndex(draw.submit_id, buffer, draw.index_type_and_size, draw.index_count,
		                   draw.index_addr, draw.flags, draw.type, draw.instance_count,
		                   draw.render_target_slice_offset, draw.vertex_offset_add,
		                   draw.first_instance, prepared);
		buffer.SwapRegisterView(previous);
	}
	t_drain.record_cyc += __builtin_ia32_rdtsc() - record_start;
	ReportDrainCensus();
	if (t_drain.drains % 2000 == 0) {
		ReportHashCensus();
	}
}

} // namespace Libs::Graphics
