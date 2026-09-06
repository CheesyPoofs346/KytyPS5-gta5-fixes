#include <chrono>
#include "graphics/host_gpu/renderer/drawBatchQueue.h"

#include "common/emulatorConfig.h"

#include <atomic>
#include <cstdio>
#include "graphics/host_gpu/renderer/drawWorkerContext.h"
#include "graphics/host_gpu/renderer/drawWorkerPool.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

namespace Libs::Graphics {

// This type is constructed, moved and stored once per draw, so its size is load-bearing: it was
// 10496 bytes when PreparedShaders was embedded, which cost ~3.2 us/draw on its own.
static_assert(sizeof(QueuedDraw) <= 128, "QueuedDraw must stay small - see m_prepared");

namespace {

// ParallelFor pays a fixed dispatch cost per batch, so its value depends entirely on how many
// draws a batch actually carries. If batches are being cut to a handful of draws the dispatch
// cannot pay for itself, and that is measured here rather than assumed.
struct DrainCensus {
	uint64_t drains        = 0;
	uint64_t draws         = 0;
	uint64_t prepare_cyc   = 0;   // phase 1, serial lookup
	uint64_t parallel_cyc  = 0;   // phase 2a, the SRT walk across workers
	uint64_t acquire_cyc   = 0;   // phase 2b, serial acquisition (creates what is missing)
	uint64_t bind_cyc      = 0;   // phase 2c, parallel binding (every lookup hits)
	uint64_t record_cyc    = 0;   // phase 3, serial recording
	uint64_t buckets[6]    = {};  // 1, 2-4, 5-16, 17-64, 65-255, 256
	// Wall anchor so phases can be reported in ABSOLUTE ms. Percentages alone are unusable for
	// an A/B: their denominator is the sum of the phases, which the change itself moves.
	std::chrono::steady_clock::time_point wall_start {};
	uint64_t                              tsc_start = 0;
};

thread_local DrainCensus t_drain;

// Not part of DrainCensus: that is thread_local and the increments happen on worker threads,
// whose copies the main thread's report would never see.
std::atomic<uint64_t> g_bailouts {0};

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
	// Convert cycles to time against measured wall elapsed, assuming no TSC frequency.
	const auto now_tsc = __builtin_ia32_rdtsc();
	const auto wall_ns = t_drain.wall_start.time_since_epoch().count() == 0
	                         ? 0.0
	                         : std::chrono::duration<double, std::nano>(
	                               std::chrono::steady_clock::now() - t_drain.wall_start)
	                               .count();
	const auto tsc_span     = static_cast<double>(now_tsc - t_drain.tsc_start);
	const auto ns_per_cycle = (tsc_span > 0.0 && wall_ns > 0.0) ? wall_ns / tsc_span : 0.0;
	const auto ms           = [&](uint64_t cyc) {
        return static_cast<double>(cyc) * ns_per_cycle / 1e6;
	};
	std::printf("DrainMs[window %.2f s]: phase1=%.1f phase2a=%.1f phase2b_ACQUIRE=%.1f "
	            "phase2c=%.1f phase3_RECORD=%.1f ms | draws=%llu\n",
	            wall_ns / 1e9, ms(t_drain.prepare_cyc), ms(t_drain.parallel_cyc),
	            ms(t_drain.acquire_cyc), ms(t_drain.bind_cyc), ms(t_drain.record_cyc),
	            static_cast<unsigned long long>(t_drain.draws));
	const auto drains = static_cast<double>(t_drain.drains);
	const auto total  = static_cast<double>(t_drain.prepare_cyc + t_drain.parallel_cyc +
                                           t_drain.acquire_cyc + t_drain.bind_cyc +
                                           t_drain.record_cyc);
	std::printf("DrainCensus: drains=%llu draws/batch=%.1f | phase1_prepare=%.1f%% "
	            "phase2a_walk=%.1f%% phase2b_acquire=%.1f%% phase2c_bind=%.1f%% "
	            "phase3_record=%.1f%% | sizes 1:%llu 2-4:%llu 5-16:%llu "
	            "17-64:%llu 65-255:%llu 256:%llu | bailouts=%llu\n",
	            static_cast<unsigned long long>(t_drain.drains),
	            static_cast<double>(t_drain.draws) / drains,
	            total > 0 ? 100.0 * static_cast<double>(t_drain.prepare_cyc) / total : 0.0,
	            total > 0 ? 100.0 * static_cast<double>(t_drain.parallel_cyc) / total : 0.0,
	            total > 0 ? 100.0 * static_cast<double>(t_drain.acquire_cyc) / total : 0.0,
	            total > 0 ? 100.0 * static_cast<double>(t_drain.bind_cyc) / total : 0.0,
	            total > 0 ? 100.0 * static_cast<double>(t_drain.record_cyc) / total : 0.0,
	            static_cast<unsigned long long>(t_drain.buckets[0]),
	            static_cast<unsigned long long>(t_drain.buckets[1]),
	            static_cast<unsigned long long>(t_drain.buckets[2]),
	            static_cast<unsigned long long>(t_drain.buckets[3]),
	            static_cast<unsigned long long>(t_drain.buckets[4]),
	            static_cast<unsigned long long>(t_drain.buckets[5]),
	            static_cast<unsigned long long>(g_bailouts.load(std::memory_order_relaxed)));
	std::fflush(stdout);
}

} // namespace

void DrawBatchQueue::Drain(RenderExecutor& executor, CommandBuffer& buffer) {
	if (t_drain.wall_start.time_since_epoch().count() == 0) {
		t_drain.wall_start = std::chrono::steady_clock::now();
		t_drain.tsc_start  = __builtin_ia32_rdtsc();
	}
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
			                 // Cleared first: the flag is per-thread and a slot resolves many
			                 // draws, so a bail-out left set by an earlier item would silently
			                 // invalidate this one.
			                 (void)TakeWorkerBailout();
			                 executor.ResolveQueuedShaders(prepared[index]);
			                 if (TakeWorkerBailout()) {
				                 // The walk reached work only the main thread may do. Whatever it
				                 // produced is discarded rather than patched up - phase 3 resolves
				                 // this draw inline, which is the same path an uncompiled
				                 // permutation already takes.
				                 prepared[index].valid = false;
				                 prepared[index].bindings_valid = false;
				                 g_bailouts.fetch_add(1, std::memory_order_relaxed);
			                 } else {
				                 // Phase 2a2: resolve this draw's images here rather than in the
				                 // serial acquire phase. Resolution only; a miss tickets one slot
				                 // and leaves the rest usable, so unlike the bail-out above this
				                 // never discards the draw.
				                 executor.PreResolveQueuedImages(prepared[index]);
			                 }
		                 });
		t_drain.parallel_cyc += __builtin_ia32_rdtsc() - parallel_start;

		if (Config::TestParallelBindingsEnabled()) {
			// Phase 2b, SERIAL: acquisition. Creates every buffer and image the batch needs.
			// PrepareBindings reaches FindImage and FindBuffers reaches CreateBuffer, both of which
			// allocate and record into the primary - which is what aborted three earlier attempts
			// to run the whole resolve on workers.
			const auto acquire_start = __builtin_ia32_rdtsc();
			for (size_t i = 0; i < draws.size(); i++) {
				const auto previous = bind(draws[i]);
				executor.AcquireQueuedBindings(m_prepared[i]);
				buffer.SwapRegisterView(previous);
			}
			t_drain.acquire_cyc += __builtin_ia32_rdtsc() - acquire_start;

			// Phase 2c, PARALLEL: binding. Every lookup now hits, so nothing here can allocate.
			// The creation tripwires stay armed: one firing means acquisition missed a route.
			const auto bind_start = __builtin_ia32_rdtsc();
			pool.ParallelFor(static_cast<uint32_t>(draws.size()),
			                 [&prepared, &executor](uint32_t index, uint32_t worker) {
				                 ScopedDrawWorker slot(worker);
				                 (void)TakeWorkerBailout();
				                 executor.BindQueuedResources(prepared[index]);
				                 if (TakeWorkerBailout()) {
					                 prepared[index].valid          = false;
					                 prepared[index].bindings_valid = false;
					                 g_bailouts.fetch_add(1, std::memory_order_relaxed);
				                 }
			                 });
			t_drain.bind_cyc += __builtin_ia32_rdtsc() - bind_start;
		}
	}

	// Phase 3, serial: record, in guest order. A draw whose permutation needed compiling arrives
	// with prepared.valid false and resolves inline, exactly as it did before.
	const auto record_start = __builtin_ia32_rdtsc();
	const bool have_prepared = workers > 1 && m_prepared.size() >= draws.size();
	for (size_t i = 0; i < draws.size(); i++) {
		auto&      draw     = draws[i];
		const auto previous = bind(draw);
		auto* prepared = have_prepared && m_prepared[i].valid ? &m_prepared[i] : nullptr;
		executor.DrawIndex(draw.submit_id, buffer, draw.index_type_and_size, draw.index_count,
		                   draw.index_addr, draw.flags, draw.type, draw.instance_count,
		                   draw.render_target_slice_offset, draw.vertex_offset_add,
		                   draw.first_instance, prepared);
		buffer.SwapRegisterView(previous);
	}
	t_drain.record_cyc += __builtin_ia32_rdtsc() - record_start;
	ReportDrainCensus();
	if (t_drain.drains % 2000 == 0) {
		ReportRecordingCensus();
	}
}

} // namespace Libs::Graphics
