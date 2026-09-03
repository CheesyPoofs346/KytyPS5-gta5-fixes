#include "graphics/host_gpu/renderer/drawBatchQueue.h"

#include "common/emulatorConfig.h"
#include "graphics/host_gpu/renderer/drawWorkerContext.h"
#include "graphics/host_gpu/renderer/drawWorkerPool.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

namespace Libs::Graphics {

void DrawBatchQueue::Drain(RenderExecutor& executor, CommandBuffer& buffer) {
	if (m_draws.empty()) {
		return;
	}
	m_batches++;

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

	const auto workers = Config::DrawWorkerCount();
	if (workers > 1) {
		// Phase 1, serial: locate both permutations per draw. The lookup takes the program-cache
		// mutex, so it cannot be part of the parallel phase.
		for (auto& draw: draws) {
			const auto previous = bind(draw);
			executor.PrepareQueuedShaders(buffer, draw.prepared);
			buffer.SwapRegisterView(previous);
		}

		// Phase 2, parallel: the resource walk. 3.4 us/draw and the only phase in a draw that
		// touches no command buffer, no image layout and no cache - which is exactly why this one
		// can be handed out while recording still cannot.
		//
		// No register view is bound here on purpose: the walk reads what PrepareQueuedShaders
		// already captured into prepared, not the live registers, so workers never race the view
		// the serial phases swap.
		auto& pool = buffer.GetContext().GetDrawWorkerPool(workers);
		pool.ParallelFor(static_cast<uint32_t>(draws.size()),
		                 [&draws, &executor](uint32_t index, uint32_t worker) {
			                 ScopedDrawWorker slot(worker);
			                 executor.ResolveQueuedShaders(draws[index].prepared);
		                 });
	}

	// Phase 3, serial: record, in guest order. A draw whose permutation needed compiling arrives
	// with prepared.valid false and resolves inline, exactly as it did before.
	for (auto& draw: draws) {
		const auto previous = bind(draw);
		executor.DrawIndex(draw.submit_id, buffer, draw.index_type_and_size, draw.index_count,
		                   draw.index_addr, draw.flags, draw.type, draw.instance_count,
		                   draw.render_target_slice_offset, draw.vertex_offset_add,
		                   draw.first_instance, draw.prepared.valid ? &draw.prepared : nullptr);
		buffer.SwapRegisterView(previous);
	}
}

} // namespace Libs::Graphics
