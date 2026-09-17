#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWTAIL_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWTAIL_H_

// Post-pipeline-bind tail of a prepared draw, shared by the two default-off diagnostics:
//   --skip-ps-chksum  : a selected draw is counted exactly and its draw command is omitted;
//   --gpu-timestamps  : only an emitted draw command is bracketed by timestamp queries.
// The teardown (render-pass end + shader-write barrier when the program writes storage buffers)
// runs for every draw, skipped or not.
//
// One template so this ordering is executed by draw_tail_skip_tests with counting operations;
// ExecutePreparedDraw instantiates it with PreparedDrawTail (renderDraw.cpp), which performs the
// real Vulkan recording.

#include <cstdint>

namespace Libs::Graphics {

// Exact skip bookkeeping for --skip-ps-chksum: increments the counter reported as
// "DrawCensus ... skip_ps_chksum=", logs the phase and ticks the draw census.
void     RecordPsChksumSkip(const char* draw_name);
uint64_t PsChksumSkipCount();

// Tail requirements: CountChksumSkip(), BeginTimestamp() -> token, EmitDraw(),
// EndTimestamp(token), BeginTeardown() -> stage flags (convertible to bool), EndRendering(),
// WriteBarrier(stages).
template <typename Tail>
void RunPreparedDrawTail(Tail& tail, bool skip_ps_chksum, bool emits) {
	if (skip_ps_chksum) {
		tail.CountChksumSkip();
	} else if (emits) {
		const auto token = tail.BeginTimestamp();
		tail.EmitDraw();
		tail.EndTimestamp(token);
	}
	const auto stages = tail.BeginTeardown();
	if (stages) {
		tail.EndRendering();
		tail.WriteBarrier(stages);
	}
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWTAIL_H_
