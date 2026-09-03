#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWSTATESNAPSHOT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWSTATESNAPSHOT_H_

#include "graphics/guest_gpu/hardwareContext.h"

#include <cstdint>
#include <memory>

namespace Libs::Graphics {

// An immutable copy of everything a draw reads out of the guest register file.
//
// Worker threads cannot read HW::Context directly: PM4 packets keep writing it as the stream is
// ingested, so by the time a worker resolves draw N the registers may already describe draw N+3.
// A snapshot is the decoupling their plan calls for - the worker's inputs stop moving.
struct DrawStateSnapshot {
	HW::Context    context;
	HW::UserConfig user_config;
	HW::Shader     shaders;
};

// Snapshots are shared, not copied per draw. Copying the register file for every draw would cost
// more than the parallelism buys, so a snapshot is only built when a register write has actually
// happened since the last one - consecutive draws that change nothing share the same object by
// pointer. The measured shader-pair reuse rate is 56-60%, so most draws are expected to share.
class DrawStateSnapshotCache {
public:
	// Called by the packet loop whenever a register-writing packet is dispatched.
	void MarkRegistersDirty() noexcept { m_register_version++; }

	[[nodiscard]] uint64_t RegisterVersion() const noexcept { return m_register_version; }

	// Returns the snapshot describing the current register state, building one only if a register
	// write has landed since the last call. Never null.
	const std::shared_ptr<const DrawStateSnapshot>& Acquire(const HW::Context&    context,
	                                                        const HW::UserConfig& user_config,
	                                                        const HW::Shader&     shaders);

	[[nodiscard]] uint64_t SnapshotsBuilt() const noexcept { return m_built; }
	[[nodiscard]] uint64_t SnapshotsReused() const noexcept { return m_reused; }

private:
	std::shared_ptr<const DrawStateSnapshot> m_current;
	uint64_t                                 m_register_version  = 1;
	uint64_t                                 m_snapshot_version  = 0;
	uint64_t                                 m_built             = 0;
	uint64_t                                 m_reused            = 0;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWSTATESNAPSHOT_H_ */
