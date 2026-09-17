#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include "graphics/host_gpu/renderer/drawProfile.h"

#include <utility>

namespace Libs::Graphics {

bool ShaderMaterializeStageRuntime(std::shared_ptr<const ShaderRecompiler::IR::Program> program,
                                   std::span<const uint32_t> user_data, uint64_t shader_base,
                                   ShaderStageRuntime& stage, std::string* error,
                                   ShaderSpecializationMemoryReader read_specialization_memory,
                                   void*                            read_memory_data) {
	if (program == nullptr) {
		if (error != nullptr) {
			*error = "missing native shader plan";
		}
		return false;
	}
	ShaderRecompiler::IR::SrtRuntime runtime;
	runtime.user_data                  = user_data;
	runtime.shader_base                = shader_base;
	runtime.read_specialization_memory = read_specialization_memory;
	runtime.userdata                   = read_memory_data;
	ShaderRecompiler::IR::ResourceSnapshot snapshot;
	{
		DrawPhaseTimer evaluate_timer(DrawPhase::SrtEvaluate);
		if (!ShaderRecompiler::IR::MaterializeResources(*program, runtime, snapshot, error)) {
			return false;
		}
	}
	{
		// MaterializeResources returned true only after ValidateResourceSnapshot passed on exactly
		// this content, so only the specialization checks remain to run here. The full copy in
		// PrepareBindings is behind --hw-check; this one never was.
		DrawPhaseTimer validate_timer(DrawPhase::SrtValidate);
		if (!ShaderRecompiler::IR::ValidateMaterializedResourceSpecialization(*program, snapshot,
		                                                                      error)) {
			return false;
		}
	}
	DrawPhaseTimer snapshot_timer(DrawPhase::SrtSnapshot);
	auto resources =
	    std::make_shared<const ShaderRecompiler::IR::ResourceSnapshot>(std::move(snapshot));
	stage = {std::move(program), std::move(resources)};
	return true;
}

} // namespace Libs::Graphics
