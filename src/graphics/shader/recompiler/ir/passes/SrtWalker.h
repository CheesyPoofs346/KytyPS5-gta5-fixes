#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, uint32_t* value);

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base = 0;
	SrtMemoryReader           read_memory = nullptr;
	void*                     userdata    = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
};

struct DescriptorSourceRequest {
	uint32_t source = 0;
	uint32_t use_pc = 0;
};

// Collects reachable ReadConst values. Immediate offsets receive compact flat-buffer slots;
// dynamic offsets remain explicit and are never assigned a fake slot.
bool BuildSrtPlan(Program& program, std::string* error);

// Test/diagnostic: how many evaluations the COMPILED path actually served. Without this a
// differential test cannot tell a genuine match from a silent fallback to the interpreter.
// Why the runtime verifier last rejected a compiled evaluation. Diagnostic only: without this a
// mismatch is indistinguishable from a missing read, and the combined trace hides which came
// first. Thread-local, overwritten on every rejection.
struct CompiledSrtRejection {
	enum class Reason : uint8_t {
		None,
		ReferenceFailed,    // the reference interpreter itself failed on the captured inputs
		ReadAddress,        // reference asked for a different address than was captured
		ReadReader,         // reference used the other reader (memory vs specialization)
		ReadExhausted,      // reference wanted MORE reads than the compiled run made
		ReadUnavailable,    // compiled side wanted a read the interpreter never performed
		ReadNode,           // same address and width, but a DIFFERENT IR node issued it
		ReadSize,           // same node and address, different width
		ReadOrder,          // a captured read, but requested out of order
		ReadsRemaining,     // reference made FEWER reads than the compiled run made
		ResultsDiffer,      // descriptor values differ
		FlatDiffers,        // flattened SRT differs
	};

	Reason   reason   = Reason::None;
	uint32_t index    = 0;   // result index, flat index, or replay position
	uint32_t dword    = 0;   // first differing dword within a descriptor
	uint64_t expected = 0;   // reference / captured
	uint64_t actual   = 0;   // compiled / requested
	uint32_t captured_reads  = 0;
	uint32_t reference_reads = 0;
};

// TEST-ONLY fault-injection seam. Forces the interpreter's memo to miss on every node, so it
// re-evaluates shared subexpressions and re-issues their guest reads. That is the same observable
// behaviour the real memo produces when its 8-probe window fills, which depends on Inst pointer
// values and therefore varies between processes. Making it deterministic is the only way to
// regression-test the verifier's tolerance of benign re-reads.
// TEST-ONLY: corrupt the compiled evaluator's first output dword, so the verifier has a genuine
// divergence to detect. Without this, replay makes real disagreement impossible to provoke - the
// reference is fed the compiled run's own captured bytes - and "detection works" would be an
// untested claim.
// TEST-ONLY: make verification match captured reads by address alone, ignoring which IR
// node issued them - the behaviour of an earlier, too-loose replay. Used to prove the
// node component of read identity is load-bearing: with it enabled, a fixture with two
// distinct read nodes at one address must be REJECTED.
void SetVerifyIgnoreReadNodeForTest(bool ignore);

// TEST-ONLY: discard the final captured read before the compiled evaluator is replayed, so the
// replay asks for a read that is not in the capture. That path is otherwise unreachable from a
// test, because the compiled evaluator normally reads a subset of what the interpreter read.
void SetVerifyDropLastCaptureForTest(bool drop);

void SetCompiledSrtCorruptForTest(bool corrupt);

void SetSrtMemoMissForTest(bool force_miss);

[[nodiscard]] const CompiledSrtRejection& CompiledSrtLastRejection();
[[nodiscard]] uint64_t                    CompiledSrtRejectionCount();
// Evaluations that were VERIFIED: the interpreter produced the returned result and the
// compiled evaluator was replayed against its captured inputs. Deliberately separate from
// CompiledSrtUseCount, which counts only evaluations the compiled path actually served.
[[nodiscard]] uint64_t                    CompiledSrtVerifyCount();
// Verifications abandoned because the interpreter itself observed one node reading one address
// twice with DIFFERENT values - guest memory moved under it, so no single-valued replay can
// reproduce its result. Neither evaluator is at fault and nothing is disabled.
[[nodiscard]] uint64_t                    CompiledSrtInconclusiveCount();
// Print the coverage counters. Called periodically from both evaluation paths.
void CompiledSrtReport();

[[nodiscard]] uint64_t CompiledSrtUseCount();
// Numbers the descriptor-reachable value subgraph so the compiled evaluator has a dense slot
// space. Must run AFTER TrackResources, which is what produces descriptor_sources and
// srt_reads - the roots of the walk. Purely additive: on any shortfall eval_slot_count stays
// 0 and the compiled path is simply not used.
void NumberEvalSlots(Program& program);

bool ValidateRuntimeValue(const Program& program, Value value, std::string& reason);

bool EvaluateDescriptorSource(const Program& program, uint32_t source, uint32_t use_pc,
                              const SrtRuntime& runtime, DescriptorValue& result,
                              std::string* error);

// Evaluates one runtime snapshot transactionally. Scalar values and ReadConst results shared by
// several descriptors are memoized once across the batch.
bool EvaluateDescriptorSources(const Program&                           program,
                               std::span<const DescriptorSourceRequest> requests,
                               const SrtRuntime& runtime, std::vector<DescriptorValue>& results,
                               std::string* error);

// Evaluates descriptor sources and the flattened immediate SRT with one memoized scalar walk.
// On failure neither destination is changed.
bool EvaluateRuntimeSources(const Program&                           program,
                            std::span<const DescriptorSourceRequest> requests,
                            const SrtRuntime& runtime, std::vector<DescriptorValue>& results,
                            std::vector<uint32_t>& flat, std::span<const uint8_t> clean_flat_slots,
                            std::string* error);

bool WalkSrt(const Program& program, const SrtRuntime& runtime, std::vector<uint32_t>& flat,
             std::string* error);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
