#include "graphics/host_gpu/renderer/drawBatchQueue.h"

#include "graphics/host_gpu/renderer/render.h"

namespace Libs::Graphics {

void DrawBatchQueue::Drain(RenderExecutor& executor, CommandBuffer& buffer) {
	if (m_draws.empty()) {
		return;
	}
	m_batches++;

	// Draining is in guest order, and the queue is cleared before translating so that a draw which
	// triggers a flush mid-drain cannot re-enter and translate these entries twice.
	std::vector<QueuedDraw> draws;
	draws.swap(m_draws);

	for (auto& draw: draws) {
		// const_cast is safe here and only here: the draw path reads its register state and never
		// writes it - checked across every GetRegisters/GetShaders/GetUserConfig use - which is
		// what lets consecutive draws share one immutable snapshot by pointer.
		auto& snapshot = const_cast<DrawStateSnapshot&>(*draw.snapshot);
		const auto previous = buffer.SwapRegisterView(
		    CommandBuffer::RegisterView {&snapshot.context, &snapshot.user_config,
		                                 &snapshot.shaders});
		executor.DrawIndex(draw.submit_id, buffer, draw.index_type_and_size, draw.index_count,
		                   draw.index_addr, draw.flags, draw.type, draw.instance_count,
		                   draw.render_target_slice_offset, draw.vertex_offset_add,
		                   draw.first_instance);
		buffer.SwapRegisterView(previous);
	}
}

} // namespace Libs::Graphics
