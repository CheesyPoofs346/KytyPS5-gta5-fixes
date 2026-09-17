// Focused tests for the diagnostic --pipeline-names identity strings and reuse check.
//
// Offline and device-free: covers the name format (declared checksum, code-hash fallback, no pixel
// stage, permutation index, zero padding), the identity comparison used on pipeline reuse, the
// config default, and the ShaderParams default. Whether the Vulkan object actually receives the name
// is a run-time property checked in a GPU tool, not here; where names are set is checked by
// check_pipeline_names_structure.py.

#include "common/emulatorConfig.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/shaderCompiler.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace {

using Libs::Graphics::FormatGraphicsPipelineName;
using Libs::Graphics::GraphicsPipelineIdentityMatches;
using Libs::Graphics::GraphicsPipelineNameMismatches;
using Libs::Graphics::PipelineCache;
using Libs::Graphics::ShaderParams;
using Libs::Graphics::ShaderProgram;

int g_failures = 0;

void Check(const char* test, bool ok, const std::string& message) {
	if (!ok) {
		std::printf("[FAIL]    %-50s %s\n", test, message.c_str());
		g_failures++;
	}
}

ShaderProgram Program(uint64_t id, uint64_t hash, uint32_t permutation, bool declared) {
	ShaderProgram program {};
	program.id                  = id;
	program.guest_hash          = hash;
	program.permutation         = permutation;
	program.guest_hash_declared = declared;
	return program;
}

PipelineCache::GraphicsPipeline Stored(const ShaderProgram& vs, const ShaderProgram* ps) {
	PipelineCache::GraphicsPipeline pipeline {};
	pipeline.vs_shader_id   = vs.id;
	pipeline.ps_shader_id   = ps != nullptr ? ps->id : 0;
	pipeline.vs_guest_hash  = vs.guest_hash;
	pipeline.vs_permutation = vs.permutation;
	pipeline.ps_guest_hash  = ps != nullptr ? ps->guest_hash : 0;
	pipeline.ps_permutation = ps != nullptr ? ps->permutation : 0;
	return pipeline;
}

void TestDeclaredName() {
	const char* test = "Name: declared checksums, permutations, program ids";
	const auto  vs   = Program(0x11, 0x2222aaaa, 0, true);
	const auto  ps   = Program(0x33, 0x1234abcd, 2, true);
	const auto  name = FormatGraphicsPipelineName(vs, &ps);
	Check(test,
	      name == "Kyty.GfxPipeline[ps=0x1234abcd#2 vs=0x2222aaaa#0 vs_prog=0x0000000000000011 "
	              "ps_prog=0x0000000000000033]",
	      name);
}

void TestZeroPaddedChecksum() {
	const char* test = "Name: checksum keeps 8 hex digits";
	const auto  vs   = Program(1, 0x0000beef, 0, true);
	const auto  ps   = Program(2, 0x0000abcd, 0, true);
	const auto  name = FormatGraphicsPipelineName(vs, &ps);
	Check(test, name.find("ps=0x0000abcd#0 vs=0x0000beef#0") != std::string::npos, name);
}

void TestFallbackName() {
	const char* test = "Name: code-hash fallback is marked, not a checksum";
	const auto  vs   = Program(5, 0x0123456789abcdefull, 1, false);
	const auto  ps   = Program(6, 0xfedcba9876543210ull, 3, false);
	const auto  name = FormatGraphicsPipelineName(vs, &ps);
	Check(test, name.find("ps=code:0xfedcba9876543210#3") != std::string::npos, name);
	Check(test, name.find("vs=code:0x0123456789abcdef#1") != std::string::npos, name);
}

void TestNoPixelStage() {
	const char* test = "Name: pipeline without a pixel stage";
	const auto  vs   = Program(7, 0x00c0ffee, 4, true);
	const auto  name = FormatGraphicsPipelineName(vs, nullptr);
	Check(test,
	      name == "Kyty.GfxPipeline[ps=none vs=0x00c0ffee#4 vs_prog=0x0000000000000007 "
	              "ps_prog=0x0000000000000000]",
	      name);
}

void TestIdentityMatches() {
	const char* test   = "Reuse: identity comparison";
	const auto  vs     = Program(0x11, 0x2222aaaa, 0, true);
	const auto  ps     = Program(0x33, 0x1234abcd, 2, true);
	const auto  stored = Stored(vs, &ps);
	Check(test, GraphicsPipelineIdentityMatches(stored, vs, &ps), "identical identity rejected");

	auto other_vs_hash = vs;
	other_vs_hash.guest_hash ^= 1u;
	Check(test, !GraphicsPipelineIdentityMatches(stored, other_vs_hash, &ps), "vs hash change accepted");
	auto other_vs_perm = vs;
	other_vs_perm.permutation++;
	Check(test, !GraphicsPipelineIdentityMatches(stored, other_vs_perm, &ps), "vs permutation change accepted");
	auto other_ps_hash = ps;
	other_ps_hash.guest_hash ^= 1u;
	Check(test, !GraphicsPipelineIdentityMatches(stored, vs, &other_ps_hash), "ps hash change accepted");
	auto other_ps_perm = ps;
	other_ps_perm.permutation++;
	Check(test, !GraphicsPipelineIdentityMatches(stored, vs, &other_ps_perm), "ps permutation change accepted");
	Check(test, !GraphicsPipelineIdentityMatches(stored, vs, nullptr), "missing pixel stage accepted");

	const auto no_ps = Stored(vs, nullptr);
	Check(test, GraphicsPipelineIdentityMatches(no_ps, vs, nullptr), "no-pixel identity rejected");
	Check(test, !GraphicsPipelineIdentityMatches(no_ps, vs, &ps), "added pixel stage accepted");
}

void TestNoMismatchesOffline() {
	const char* test = "Reuse: mismatch counter untouched offline";
	Check(test, GraphicsPipelineNameMismatches() == 0,
	      std::to_string(GraphicsPipelineNameMismatches()));
}

void TestConfigDefault() {
	const char*           test = "Config: --pipeline-names defaults off";
	Config::ConfigOptions cfg;
	Check(test, !cfg.pipeline_names, "ConfigOptions default is true");
	Config::Load(cfg);
	Check(test, !Config::PipelineNamesEnabled(), "loaded default enabled");
	cfg.pipeline_names = true;
	Config::Load(cfg);
	Check(test, Config::PipelineNamesEnabled(), "explicit true not applied");
	Config::Load(Config::ConfigOptions {});
	Check(test, !Config::PipelineNamesEnabled(), "reload default left it enabled");
}

void TestShaderParamsDefault() {
	const char*        test = "ShaderParams: hash_declared defaults false";
	const ShaderParams params {};
	Check(test, !params.hash_declared, "default is true");
}

} // namespace

int main() {
	Config::Initialize();
	TestDeclaredName();
	TestZeroPaddedChecksum();
	TestFallbackName();
	TestNoPixelStage();
	TestIdentityMatches();
	TestNoMismatchesOffline();
	TestConfigDefault();
	TestShaderParamsDefault();
	Config::Shutdown();
	if (g_failures != 0) {
		std::printf("%d pipeline name check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("pipeline name tests passed\n");
	return 0;
}
