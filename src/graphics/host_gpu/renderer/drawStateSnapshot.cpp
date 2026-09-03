#include "graphics/host_gpu/renderer/drawStateSnapshot.h"

#include <cstdio>

namespace Libs::Graphics {

const std::shared_ptr<const DrawStateSnapshot>& DrawStateSnapshotCache::Acquire(
    const HW::Context& context, const HW::UserConfig& user_config, const HW::Shader& shaders) {
	if (m_current != nullptr && m_snapshot_version == m_register_version) {
		m_reused++;
		return m_current;
	}

	// Size is the tax the whole parallel design pays per register change, so state it once rather
	// than leave it to be guessed at.
	static bool reported = false;
	if (!reported) {
		reported = true;
		std::printf("DrawStateSnapshot: %zu bytes per snapshot (context %zu, user_config %zu, "
		            "shaders %zu)\n",
		            sizeof(DrawStateSnapshot), sizeof(HW::Context), sizeof(HW::UserConfig),
		            sizeof(HW::Shader));
		std::fflush(stdout);
	}

	auto snapshot         = std::make_shared<DrawStateSnapshot>();
	snapshot->context     = context;
	snapshot->user_config = user_config;
	snapshot->shaders     = shaders;

	m_current          = std::move(snapshot);
	m_snapshot_version = m_register_version;
	m_built++;
	return m_current;
}

} // namespace Libs::Graphics
