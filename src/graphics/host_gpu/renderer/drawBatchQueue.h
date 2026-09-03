#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWBATCHQUEUE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWBATCHQUEUE_H_

#include "graphics/host_gpu/renderer/drawStateSnapshot.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Libs::Graphics {

class CommandBuffer;
class RenderExecutor;

// A draw held for later translation, with the register state it must be translated against.
//
// The snapshot is the whole point: the packet loop keeps writing HW::Context as it ingests, so a
// draw translated later must not read the live registers - by then they describe a later draw.
struct QueuedDraw {
	std::shared_ptr<const DrawStateSnapshot> snapshot;
	uint64_t                                 submit_id                  = 0;
	uint32_t                                 index_type_and_size        = 0;
	uint32_t                                 index_count                = 0;
	const void*                              index_addr                 = nullptr;
	uint32_t                                 flags                      = 0;
	uint32_t                                 type                       = 0;
	uint32_t                                 instance_count             = 0;
	uint32_t                                 render_target_slice_offset = 0;
	int32_t                                  vertex_offset_add          = 0;
	uint32_t                                 first_instance             = 0;
};

// Collects consecutive draws so they can be translated as a group instead of one at a time.
//
// Draining is strictly in guest order. Blending and depth depend on it, and nothing here reorders
// draws - the batch exists to give the translator a group to work on, not to change what is drawn.
class DrawBatchQueue {
public:
	// Capacity is a boundary of its own so a long unbroken run of draws cannot grow the queue
	// without bound, and so the GPU is not left idle while the CPU accumulates a whole frame.
	static constexpr size_t kMaxBatchDraws = 256;

	[[nodiscard]] bool   Empty() const noexcept { return m_draws.empty(); }
	[[nodiscard]] size_t Size() const noexcept { return m_draws.size(); }
	[[nodiscard]] bool   Full() const noexcept { return m_draws.size() >= kMaxBatchDraws; }

	void Push(QueuedDraw&& draw) { m_draws.push_back(std::move(draw)); }

	// Translates every queued draw and clears the queue. Each draw is translated against its own
	// snapshot: the command buffer's register view is repointed for the duration and restored
	// afterwards, so nothing outside the drain observes the swap.
	void Drain(RenderExecutor& executor, CommandBuffer& buffer);

	[[nodiscard]] uint64_t DrawsQueued() const noexcept { return m_queued; }
	[[nodiscard]] uint64_t BatchesDrained() const noexcept { return m_batches; }

private:
	std::vector<QueuedDraw> m_draws;
	uint64_t                m_queued  = 0;
	uint64_t                m_batches = 0;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWBATCHQUEUE_H_ */
