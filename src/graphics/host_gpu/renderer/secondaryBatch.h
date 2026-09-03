#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SECONDARYBATCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SECONDARYBATCH_H_

namespace Libs::Graphics {

class RenderContext;

// Ends the open secondary batch, replays it into the primary inside its render pass, and closes
// the pass. Safe to call when nothing is open.
//
// Anything that touches the render targets outside the draw stream - a resolve, a copy, a submit,
// a compute dispatch - must call this first, because those belong on the primary outside a render
// pass and the batch has not opened one yet.
void FlushSecondaryBatch(RenderContext& context);

[[nodiscard]] bool SecondaryBatchOpen() noexcept;

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SECONDARYBATCH_H_ */
