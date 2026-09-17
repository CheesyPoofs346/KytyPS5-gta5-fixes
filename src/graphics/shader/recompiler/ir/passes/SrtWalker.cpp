#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"

#include "common/emulatorConfig.h"

#include <algorithm>
#include <span>
#include <array>
#include <bit>
#include <atomic>
#include <cstdio>
#include <memory>
#include <cstring>
#include <fmt/format.h>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

// Whether compiling SRT expression graphs into flat fetch lists is worth building turns on this
// split. If the ~27 ns per resolution is mostly guest memory reads, a compiled list still
// performs those reads and saves almost nothing; if it is mostly graph walking, it is the single
// biggest item left. The comment above claims the walk dominates - measure it rather than trust it.
thread_local uint64_t t_srt_reads       = 0;
// TEST-ONLY: see SetSrtMemoMissForTest.
thread_local bool     t_force_memo_miss = false;
// Identity of the IR node currently issuing a guest read, as its dense eval_slot. Both
// evaluators publish it so the verifier can tell a memo-eviction repeat of ONE node apart
// from two DISTINCT nodes that happen to read the same address - values at one address are
// not interchangeable across nodes when guest memory can change mid-evaluation.
constexpr uint32_t   kNoReadNode  = 0xffffffffu;
thread_local uint32_t t_read_node = kNoReadNode;

struct ReadNodeScope {
	explicit ReadNodeScope(uint32_t node): m_saved(t_read_node) { t_read_node = node; }
	~ReadNodeScope() { t_read_node = m_saved; }
	ReadNodeScope(const ReadNodeScope&)            = delete;
	ReadNodeScope& operator=(const ReadNodeScope&) = delete;

private:
	uint32_t m_saved;
};
// TEST-ONLY: see SetCompiledSrtCorruptForTest.
thread_local bool     t_corrupt_compiled = false;
// TEST-ONLY: see SetVerifyIgnoreReadNodeForTest.
thread_local bool     t_verify_ignore_read_node = false;
// TEST-ONLY: see SetVerifyDropLastCaptureForTest.
thread_local bool     t_verify_drop_last_capture = false;
thread_local uint64_t t_srt_read_cycles = 0;
thread_local uint64_t t_srt_nodes       = 0;
thread_local uint64_t t_srt_calls       = 0;
thread_local uint64_t t_srt_resolutions = 0;
thread_local uint64_t t_compiled_report = 0;
// nodes counts unique nodes (memo misses). If the DAG is heavily shared, the interpreter is
// entered far more often than that, and a compiled form - which evaluates each node exactly once
// - wins by removing revisits, not just by being a cheaper per-node loop. These separate the two.
thread_local uint64_t t_srt_entries    = 0;   // every EvaluateWide entry
thread_local uint64_t t_srt_immediates = 0;   // entries that were literals
thread_local uint64_t t_srt_hits       = 0;   // entries served by the memo


constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "vertex";
		case ShaderType::Pixel: return "pixel";
		case ShaderType::Fetch: return "fetch";
		case ShaderType::Compute: return "compute";
		default: return "unknown";
	}
}

std::string Diagnostic(const Program& program, uint32_t pc, const std::string& message) {
	return fmt::format("shader SRT: hash=0x{:016x} stage={} pc=0x{:08x} {}", program.shader_hash,
	                   StageName(program.stage), pc, message);
}

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const Program& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto kind = values.memory_info[index].kind;
	return (op == ValueOpcode::LoadAddressU32 && kind == ResourceKind::ScalarAddress) ||
	       (op == ValueOpcode::ReadConstBuffer && kind == ResourceKind::ScalarBuffer);
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeIntegerOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UMin32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const Program& program): m_program(program) {}

	bool Run(Value value, std::string& reason) { return Validate(value, reason); }

private:
	bool Validate(Value value, std::string& reason) {
		value            = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64: return true;
				default:
					reason = fmt::format("contains a non-integer immediate of type {}",
					                     TypeName(value.GetType()));
					return false;
			}
		}
		if (!m_visiting.insert(inst).second) {
			reason = fmt::format("contains a cyclic {} value", ValueOpcodeName(inst->GetOpcode()));
			return false;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU8 ||
		    op == ValueOpcode::UndefU16 || op == ValueOpcode::UndefU32 ||
		    op == ValueOpcode::UndefU64 || op == ValueOpcode::Void) {
			reason = fmt::format("contains {}", ValueOpcodeName(op));
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				reason = "contains a malformed GetUserData";
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				reason = fmt::format("references unavailable user SGPR {}", reg);
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				reason = "contains a malformed GetShaderBase";
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::Phi) {
			auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				// Different descriptors per branch. Accept one leaf rather than failing the
				// shader: aborting the emulator over a legal shader is the worse outcome.
				invariant = ResolveFirstPhiLeaf(m_program, value);
				if (invariant.IsEmpty()) {
					reason = "contains a control-dependent phi";
					return finish(false);
				}
			}
			return finish(Validate(invariant, reason));
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				reason = "contains a malformed GetSrtResource";
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				reason = "contains a malformed flattened SRT read";
				return finish(false);
			}
		} else if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				reason = fmt::format("contains a non-scalar {}", ValueOpcodeName(op));
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				reason = "contains an invalid U64 component index";
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u ||
			    (source->GetOpcode() != ValueOpcode::CompositeConstructU32x2 &&
			     source->GetOpcode() != ValueOpcode::IAddCarry32)) {
				reason = "contains an unsupported composite runtime source";
				return finish(false);
			}
		}
		if (op == ValueOpcode::GetBufferResource || op == ValueOpcode::GetImageResource ||
		    op == ValueOpcode::GetSamplerResource || op == ValueOpcode::GetAddressResource) {
			const size_t expected = op == ValueOpcode::GetBufferResource    ? 4u
			                        : op == ValueOpcode::GetImageResource   ? 8u
			                        : op == ValueOpcode::GetSamplerResource ? 4u
			                                                                : 2u;
			if (inst->NumArgs() != expected) {
				reason = fmt::format("contains a malformed {}", ValueOpcodeName(op));
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeIntegerOp(op)) {
			reason =
			    fmt::format("contains unsupported or control-dependent {}", ValueOpcodeName(op));
			return finish(false);
		}
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			if (!Validate(inst->Arg(index), reason)) {
				return finish(false);
			}
		}
		return finish(true);
	}

	const Program&                  m_program;
	std::unordered_set<const Inst*> m_visiting;
};

class PlanBuilder {
public:
	explicit PlanBuilder(Program& program): m_program(program) {}

	bool Run(std::string* error) {
		m_program.srt_reads.clear();
		m_program.dynamic_reads.clear();
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				const auto op = inst.GetOpcode();
				if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
					const auto flags = inst.Flags<MemoryFlags>();
					if (flags.index < m_program.memory_info.size()) {
						const auto kind       = m_program.memory_info[flags.index].kind;
						const bool crosswired = (op == ValueOpcode::LoadAddressU32 &&
						                         kind == ResourceKind::ScalarBuffer) ||
						                        (op == ValueOpcode::ReadConstBuffer &&
						                         kind == ResourceKind::ScalarAddress);
						if (crosswired) {
							return Fail(flags.pc, error,
							            fmt::format("{} has incompatible scalar memory metadata",
							                        ValueOpcodeName(op)));
						}
					}
				}
				if (IsDescriptorHandle(inst.GetOpcode())) {
					for (size_t index = 0; index < inst.NumArgs(); index++) {
						if (!Collect(inst.Arg(index), 0, error)) {
							return false;
						}
					}
				}
			}
		}
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				std::string reason;
				if (inst.GetOpcode() == ValueOpcode::LoadAddressU32 && IsRawRead(m_program, inst) &&
				    inst.Arg(1).Resolve().IsImmediate() &&
				    ValidateRuntimeValue(m_program, Value(&inst), reason) &&
				    !Collect(Value(&inst), inst.Flags<MemoryFlags>().pc, error)) {
					return false;
				}
			}
		}
		PatchReads();
		return true;
	}

private:
	struct Patch {
		Inst*    inst = nullptr;
		uint32_t slot = 0;
		bool     keep = false;
	};

	bool Fail(uint32_t pc, std::string* error, const std::string& message) const {
		return ShaderError::Fail(error, Diagnostic(m_program, pc, message));
	}

	bool Collect(Value value, uint32_t use_pc, std::string* error) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			return true;
		}
		auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			return Fail(use_pc, error, "invalid typed planning value");
		}
		const auto cycle = std::ranges::find(m_visiting, inst);
		if (cycle != m_visiting.end()) {
			const auto contains_phi = std::any_of(cycle, m_visiting.end(), [](const Inst* value) {
				return value->GetOpcode() == ValueOpcode::Phi;
			});
			if (contains_phi) {
				return true;
			}
			return Fail(use_pc, error,
			            fmt::format("cyclic typed planning value {} without a phi",
			                        ValueOpcodeName(inst->GetOpcode())));
		}
		if (std::ranges::find(m_visited, inst) != m_visited.end()) {
			return true;
		}
		m_visiting.push_back(inst);
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			if (!Collect(inst->Arg(index), use_pc, error)) {
				return false;
			}
		}
		m_visiting.pop_back();
		m_visited.push_back(inst);
		if (!IsRawRead(m_program, *inst)) {
			return true;
		}
		const auto offset = inst->Arg(1).Resolve();
		if (!offset.IsImmediate() || offset.GetType() != Type::U32) {
			if (std::ranges::find(m_program.dynamic_reads, value) == m_program.dynamic_reads.end()) {
				m_program.dynamic_reads.push_back(value);
			}
			return true;
		}
		use_pc = inst->Flags<MemoryFlags>().pc;
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); slot++) {
			if (EquivalentValue(m_program, value, m_program.srt_reads[slot].value)) {
				m_patches.push_back({inst, slot, false});
				return true;
			}
		}
		const auto slot = static_cast<uint32_t>(m_program.srt_reads.size());
		m_program.srt_reads.push_back({value, slot, use_pc});
		m_patches.push_back({inst, slot, true});
		return true;
	}

	void PatchReads() {
		for (const auto& patch: m_patches) {
			auto* block = patch.inst->Parent();
			auto& list  = block->Instructions();
			auto  where =
			    std::ranges::find_if(list, [&](const Inst& inst) { return &inst == patch.inst; });
			const auto resource =
			    Value(&*block->PrependNewInst(where, ValueOpcode::GetSrtResource));
			const auto flat = Value(&*block->PrependNewInst(where, ValueOpcode::ReadConst,
			                                                {resource, Value(patch.slot)}));
			const auto uses = patch.inst->Uses();
			for (const auto& use: uses) {
				use.user->SetArg(use.operand, flat);
			}
			for (auto& info: m_program.block_info) {
				if (info.condition.Resolve() == Value(patch.inst)) {
					info.condition = flat;
				}
				if (info.indirect_target.Resolve() == Value(patch.inst)) {
					info.indirect_target = flat;
				}
			}
			if (patch.keep) {
				const auto memory = patch.inst->Flags<MemoryFlags>().index;
				if (memory < m_program.memory_info.size()) {
					m_program.memory_info[memory].planning_only = true;
				}
				block->AppendNewInst(ValueOpcode::ReferenceU32, {Value(patch.inst)});
			}
		}
	}

	Program&           m_program;
	std::vector<Inst*> m_visiting;
	std::vector<Inst*> m_visited;
	std::vector<Patch> m_patches;
};

class Evaluator {
public:
	Evaluator(const Program& program, const SrtRuntime& runtime,
	          std::span<const uint8_t> clean_flat_slots = {}, Evaluator* clean_evaluator = nullptr)
	    : m_program(program), m_runtime(runtime),
	      m_clean_flat_slots(clean_flat_slots), m_clean_evaluator(clean_evaluator) {
		m_arena = AcquireArena();
		if (m_arena == nullptr) {
			m_owned_arena             = std::make_unique<MemoArena>();
			m_owned_arena->generation = 1;
			m_arena                   = m_owned_arena.get();
		}
		m_generation = m_arena->generation;
	}

	~Evaluator() {
		if (m_owned_arena == nullptr && m_arena != nullptr) {
			m_arena->in_use = false;
		}
	}

	Evaluator(const Evaluator&)            = delete;
	Evaluator& operator=(const Evaluator&) = delete;
	Evaluator(Evaluator&&)                 = delete;
	Evaluator& operator=(Evaluator&&)      = delete;

	void SetUsePc(uint32_t pc) { m_use_pc = pc; }

	// --- Dependency recording for cheap re-validation ---
	//
	// Resolving a descriptor costs ~370ns, almost all of it walking the expression tree; the
	// memory it reads is only a handful of dwords (~20ns). So record exactly what an evaluation
	// depended on - which user-data slots, and which (address, value) pairs it read - and the
	// next draw can re-read just those and compare. If they match, the result is provably
	// identical and the whole walk is skipped.
	// An input is either a user-data slot or a memory dword. The previous representation kept
	// slots in a bitmask and their values in append order, then compared them by ascending bit
	// index - so any expression that read its slots out of order compared value A against slot B
	// and reported a false change. Storing the slot alongside its value removes that coupling.
	static constexpr uint32_t kNoSlot            = 0xffffffffu;
	static constexpr size_t   kMaxRecordedInputs = 24;
	// Bounds how much a deeply shared expression graph can replay into the log.
	static constexpr size_t   kMaxLogEntries     = 4096;

	struct DepEntry {
		uint64_t address = 0;        // memory address; 0 when this entry is a user-data slot
		uint32_t value   = 0;
		uint32_t slot    = kNoSlot;  // user-data slot; kNoSlot when this entry is a memory read
	};

	struct Dependencies {
		DepEntry inputs[kMaxRecordedInputs] {};
		uint32_t count      = 0;
		bool     overflowed = false;   // too many inputs, or an input we cannot re-read
	};

	// Only the descriptor cache needs the log, so the default path records nothing at all.
	void SetRecording(bool record) { m_record = record; }

	void BeginRecording() {
		m_request_start    = static_cast<uint32_t>(m_log.size());
		m_request_overflow = m_overflow_count;
	}

	// Collapses this request's slice of the log into a dependency set.
	Dependencies RecordedDependencies() const {
		Dependencies deps;
		if (m_overflow_count != m_request_overflow) {
			deps.overflowed = true;
			return deps;
		}
		for (size_t i = m_request_start; i < m_log.size(); i++) {
			const auto& entry = m_log[i];
			bool        duplicate = false;
			for (uint32_t d = 0; d < deps.count; d++) {
				if (deps.inputs[d].slot == entry.slot && deps.inputs[d].address == entry.address) {
					duplicate = true;
					break;
				}
			}
			if (duplicate) {
				continue;
			}
			if (deps.count >= kMaxRecordedInputs) {
				deps.overflowed = true;
				return deps;
			}
			deps.inputs[deps.count++] = entry;
		}
		return deps;
	}

	bool Evaluate(Value value, uint32_t& result, std::string* error) {
		uint64_t wide = 0;
		if (!EvaluateWide(value, wide, error)) {
			return false;
		}
		result = static_cast<uint32_t>(wide);
		return true;
	}

private:
	bool Fail(std::string* error, const std::string& message) const {
		return ShaderError::Fail(error, Diagnostic(m_program, m_use_pc, message));
	}

	bool EvaluateWide(Value value, uint64_t& result, std::string* error) {
		t_srt_entries++;
		value = value.Resolve();
		if (value.IsImmediate()) {
			t_srt_immediates++;
			switch (value.GetType()) {
				case Type::U1: result = value.U1(); return true;
				case Type::U8: result = value.U8(); return true;
				case Type::U16: result = value.U16(); return true;
				case Type::U32: result = value.U32(); return true;
				case Type::U64: result = value.U64(); return true;
				default: return Fail(error, "non-integer immediate in runtime expression");
			}
		}
		auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			return Fail(error, "invalid typed runtime value");
		}
		uint64_t   memo       = 0;
		size_t     slot_index = kNoSlot2;
		const auto status     = MemoLookup(inst, memo, slot_index);
		if (status == 1) {
			t_srt_hits++;
			result = memo;
			return true;
		}
		if (status == -1) {
			return Fail(error, "cyclic typed runtime value");
		}
		t_srt_nodes++;
		MemoMarkVisiting(inst, slot_index);
		const auto log_start      = static_cast<uint32_t>(m_log.size());
		const auto overflow_start = m_overflow_count;
		uint64_t   out            = 0;
		if (!EvaluateInst(*inst, out, error)) {
			return false;
		}
		MemoInsert(inst, out, log_start, overflow_start, slot_index);
		result = out;
		return true;
	}

	bool Arg(const Inst& inst, size_t index, uint64_t& result, std::string* error) {
		return EvaluateWide(inst.Arg(index), result, error);
	}

	bool EvaluatePhi(const Inst& inst, uint64_t& result, std::string* error) {
		auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
		if (value.IsEmpty()) {
			value = ResolveFirstPhiLeaf(m_program, Value(const_cast<Inst*>(&inst)));
		}
		return !value.IsEmpty() ? EvaluateWide(value, result, error)
		                        : Fail(error, "typed phi has runtime-dependent values");
	}

	bool EvaluateExtract(const Inst& inst, uint64_t& result, std::string* error) {
		const auto index = inst.Arg(1).Resolve();
		if (!index.IsImmediate() || index.GetType() != Type::U32) {
			return Fail(error, "dynamic composite extract in runtime expression");
		}
		const auto component = index.U32();
		if (component >= 2u) {
			return Fail(error, "unsupported composite runtime source");
		}
		if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
			uint64_t packed = 0;
			if (!Arg(inst, 0, packed, error)) {
				return false;
			}
			result = static_cast<uint32_t>(packed >> (component * 32u));
			return true;
		}
		const auto* source = inst.Arg(0).ResolveInstruction();
		if (source == nullptr) {
			return Fail(error, "unsupported composite runtime source");
		}
		if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
			return EvaluateWide(source->Arg(component), result, error);
		}
		if (source->GetOpcode() == ValueOpcode::IAddCarry32) {
			uint64_t lhs = 0;
			uint64_t rhs = 0;
			if (!Arg(*source, 0, lhs, error) || !Arg(*source, 1, rhs, error)) {
				return false;
			}
			const auto sum =
			    static_cast<uint64_t>(static_cast<uint32_t>(lhs)) + static_cast<uint32_t>(rhs);
			result =
			    component == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
			return true;
		}
		return Fail(error, "unsupported composite runtime source");
	}

	bool EvaluateRawRead(const Inst& inst, uint64_t& result, std::string* error) {
		const auto flags = inst.Flags<MemoryFlags>();
		if (flags.index >= m_program.memory_info.size()) {
			return Fail(error, "raw scalar read has invalid metadata");
		}
		const auto& mem    = m_program.memory_info[flags.index];
		const auto* handle = inst.Arg(0).ResolveInstruction();
		if (handle == nullptr) {
			return Fail(error, "raw scalar read has no descriptor handle");
		}
		uint64_t low    = 0;
		uint64_t high   = 0;
		uint64_t offset = 0;
		if (!Arg(*handle, 0, low, error) || !Arg(*handle, 1, high, error) ||
		    !Arg(inst, 1, offset, error)) {
			return false;
		}
		const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
		const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
		uint64_t   address   = 0;
		if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer) {
			uint64_t records = 0;
			uint64_t word3   = 0;
			if (handle->NumArgs() != 4u || !Arg(*handle, 2, records, error) ||
			    !Arg(*handle, 3, word3, error)) {
				return Fail(error, "constant-buffer descriptor has invalid width");
			}
			if (immediate < 0) {
				return Fail(error, "constant-buffer read has a negative immediate offset");
			}
			const auto byte_offset =
			    static_cast<uint64_t>(immediate) + static_cast<uint32_t>(offset);
			const auto aligned = byte_offset & ~uint64_t {3};
			const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
			const auto size = stride == 0u
			                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
			                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
			if (aligned > size || size - aligned < sizeof(uint32_t)) {
				return Fail(
				    error, fmt::format("constant-buffer offset {} exceeds size {}", aligned, size));
			}
			address = ((base & ~uint64_t {3}) + byte_offset) & ~uint64_t {3};
		} else {
			const auto relative = (immediate & ~int64_t {3}) +
			                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
			if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
				return Fail(error, "raw scalar read is outside the 48-bit address space");
			}
		}
		uint32_t word = 0;
		if (m_runtime.read_memory != nullptr) {
			const ReadNodeScope read_node {inst.GetEvalSlot()};
			if (!m_runtime.read_memory(m_runtime.userdata, address, &word)) {
				return Fail(error, fmt::format("constant read failed at 0x{:016x}", address));
			}
			// This read is not logged below, so revalidation could never see it change.
			// Anything reaching this path must never be cached.
			MarkOverflow();
		} else {
			// The rdtsc pair that used to bracket this read answered its question - reads are
			// ~4% of the walk - and cost ~500 cycles a call to keep asking it.
			std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
			t_srt_reads++;
			LogInput(address, word, kNoSlot);
		}
		result = word;
		return true;
	}

	bool EvaluateInst(const Inst& inst, uint64_t& result, std::string* error) {
		uint64_t   a       = 0;
		uint64_t   b       = 0;
		uint64_t   c       = 0;
		const auto binary  = [&]() { return Arg(inst, 0, a, error) && Arg(inst, 1, b, error); };
		const auto ternary = [&]() {
			return Arg(inst, 0, a, error) && Arg(inst, 1, b, error) && Arg(inst, 2, c, error);
		};
		switch (inst.GetOpcode()) {
			case ValueOpcode::GetUserData: {
				const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
				if (reg < m_program.user_data_base ||
				    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
					return Fail(error, fmt::format("user SGPR {} is unavailable", reg));
				}
				const auto slot = reg - m_program.user_data_base;
				result          = m_runtime.user_data[slot];
				LogInput(0, static_cast<uint32_t>(result), slot);
				return true;
			}
			case ValueOpcode::GetShaderBase: result = m_runtime.shader_base; return true;
			case ValueOpcode::Phi: return EvaluatePhi(inst, result, error);
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2: return EvaluateExtract(inst, result, error);
			case ValueOpcode::CompositeConstructU64:
				if (!binary()) {
					return false;
				}
				result = static_cast<uint32_t>(a) |
				         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
				return true;
			case ValueOpcode::ReadConst: {
				const auto slot = inst.Arg(1).Resolve();
				if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
				    slot.U32() >= m_program.srt_reads.size()) {
					return Fail(error, "invalid flattened SRT index");
				}
				if (slot.U32() < m_clean_flat_slots.size() &&
				    m_clean_flat_slots[slot.U32()] != 0u && m_clean_evaluator != nullptr) {
					// The clean evaluator records dependencies into its own m_deps and reads
					// through read_specialization_memory, so this subtree is invisible to our
					// revalidation. Refuse to cache rather than cache a partial key.
					MarkOverflow();
					m_clean_evaluator->SetUsePc(m_use_pc);
					return m_clean_evaluator->EvaluateWide(m_program.srt_reads[slot.U32()].value,
					                                       result, error);
				}
				return EvaluateWide(m_program.srt_reads[slot.U32()].value, result, error);
			}
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer:
				if (IsRawRead(m_program, inst)) {
					return EvaluateRawRead(inst, result, error);
				}
				break;
			case ValueOpcode::IAdd32:
				if (binary()) {
					result = static_cast<uint32_t>(a + b);
					return true;
				}
				return false;
			case ValueOpcode::IAdd64:
				if (binary()) {
					result = a + b;
					return true;
				}
				return false;
			case ValueOpcode::ISub32:
				if (binary()) {
					result = static_cast<uint32_t>(a - b);
					return true;
				}
				return false;
			case ValueOpcode::ISub64:
				if (binary()) {
					result = a - b;
					return true;
				}
				return false;
			case ValueOpcode::IMul32:
				if (binary()) {
					result = static_cast<uint32_t>(a * b);
					return true;
				}
				return false;
			case ValueOpcode::IMul64:
				if (binary()) {
					result = a * b;
					return true;
				}
				return false;
			case ValueOpcode::UMin32:
				if (binary()) {
					result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
					return true;
				}
				return false;
			case ValueOpcode::BitwiseAnd32:
				if (binary()) {
					result = static_cast<uint32_t>(a & b);
					return true;
				}
				return false;
			case ValueOpcode::BitwiseAnd64:
				if (binary()) {
					result = a & b;
					return true;
				}
				return false;
			case ValueOpcode::BitwiseOr32:
				if (binary()) {
					result = static_cast<uint32_t>(a | b);
					return true;
				}
				return false;
			case ValueOpcode::BitwiseXor32:
				if (binary()) {
					result = static_cast<uint32_t>(a ^ b);
					return true;
				}
				return false;
			case ValueOpcode::BitwiseNot32:
				if (Arg(inst, 0, a, error)) {
					result = ~static_cast<uint32_t>(a);
					return true;
				}
				return false;
			case ValueOpcode::ShiftLeftLogical32:
				if (binary()) {
					result = static_cast<uint32_t>(a) << (b & 31u);
					return true;
				}
				return false;
			case ValueOpcode::ShiftLeftLogical64:
				if (binary()) {
					result = a << (b & 63u);
					return true;
				}
				return false;
			case ValueOpcode::ShiftRightLogical32:
				if (binary()) {
					result = static_cast<uint32_t>(a) >> (b & 31u);
					return true;
				}
				return false;
			case ValueOpcode::ShiftRightLogical64:
				if (binary()) {
					result = a >> (b & 63u);
					return true;
				}
				return false;
			case ValueOpcode::ShiftRightArithmetic32:
				if (binary()) {
					result = static_cast<uint32_t>(
					    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
					return true;
				}
				return false;
			case ValueOpcode::ShiftRightArithmetic64:
				if (binary()) {
					result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
					return true;
				}
				return false;
			case ValueOpcode::BitFieldUExtract:
				if (ternary()) {
					const auto offset = static_cast<uint32_t>(b);
					const auto width  = static_cast<uint32_t>(c);
					if (offset > 32u || width > 32u - offset) {
						return Fail(error, "invalid unsigned bit-field range");
					}
					const auto mask = width == 32u  ? UINT32_MAX
					                  : width == 0u ? 0u
					                                : (uint32_t {1} << width) - 1u;
					result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
					return true;
				}
				return false;
			case ValueOpcode::BitFieldSExtract:
				if (ternary()) {
					const auto offset = static_cast<uint32_t>(b);
					const auto width  = static_cast<uint32_t>(c);
					if (offset > 32u || width > 32u - offset) {
						return Fail(error, "invalid signed bit-field range");
					}
					if (width == 0u) {
						result = 0;
						return true;
					}
					const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
					auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
					if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
						bits |= ~mask;
					}
					result = bits;
					return true;
				}
				return false;
			case ValueOpcode::BitFieldInsert: {
				uint64_t d = 0;
				if (!ternary() || !Arg(inst, 3, d, error)) {
					return false;
				}
				const auto offset = static_cast<uint32_t>(c);
				const auto width  = static_cast<uint32_t>(d);
				if (offset > 32u || width > 32u - offset) {
					return Fail(error, "invalid inserted bit-field range");
				}
				if (width == 0u) {
					result = static_cast<uint32_t>(a);
					return true;
				}
				const auto mask =
				    width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
				result = (static_cast<uint32_t>(a) & ~mask) |
				         ((static_cast<uint32_t>(b) << offset) & mask);
				return true;
			}
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectU1:
				if (ternary()) {
					result = a != 0u ? b : c;
					return true;
				}
				return false;
			case ValueOpcode::IEqual32:
				if (binary()) {
					result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::INotEqual32:
				if (binary()) {
					result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::ULessThan32:
				if (binary()) {
					result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::UGreaterThan32:
				if (binary()) {
					result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::LogicalAnd:
				if (binary()) {
					result = (a != 0u) && (b != 0u);
					return true;
				}
				return false;
			case ValueOpcode::LogicalOr:
				if (binary()) {
					result = (a != 0u) || (b != 0u);
					return true;
				}
				return false;
			case ValueOpcode::LogicalXor:
				if (binary()) {
					result = (a != 0u) != (b != 0u);
					return true;
				}
				return false;
			case ValueOpcode::LogicalNot:
				if (Arg(inst, 0, a, error)) {
					result = a == 0u;
					return true;
				}
				return false;
			case ValueOpcode::UndefU1:
			case ValueOpcode::UndefU8:
			case ValueOpcode::UndefU16:
			case ValueOpcode::UndefU32:
			case ValueOpcode::UndefU64: return Fail(error, "undefined typed runtime value");
			default: break;
		}
		return Fail(error, fmt::format("unsupported typed runtime opcode {}",
		                               ValueOpcodeName(inst.GetOpcode())));
	}

	// Append-only record of every input read during this call. A subtree owns the slice it
	// appended, which is what lets a memo hit hand its inputs to the expression above it.
	bool                                      m_record = false;
	std::vector<DepEntry>                     m_log;
	std::vector<DepEntry>                     m_replay_scratch;
	uint32_t                                  m_request_start    = 0;
	uint32_t                                  m_overflow_count   = 0;
	uint32_t                                  m_request_overflow = 0;
	const Program&                            m_program;
	const SrtRuntime&                         m_runtime;
	std::span<const uint8_t>                  m_clean_flat_slots;
	Evaluator*                                m_clean_evaluator = nullptr;
	uint32_t                                  m_use_pc          = 0;
	// Open-addressed memo, 256 slots x 16 bytes = 4KB, stays in L1. The previous
	// unordered_map cost two hash operations per node (find + emplace) and an expression has
	// tens of nodes, so this was a large share of the ~370ns spent per descriptor.
	//
	// Scoped to a single evaluation: cleared per Evaluator, never reused across draws. An
	// earlier attempt at cross-draw reuse resolved stale descriptors and rendered light coronas
	// as red blobs - do not reintroduce that without per-input validation.
	static constexpr size_t kMemoBits   = 8;
	static constexpr size_t kMemoSize   = size_t {1} << kMemoBits;
	// Two are live at once (the evaluator and its clean counterpart); the rest is headroom.
	static constexpr size_t kMemoArenas = 4;

	void MarkOverflow() { m_overflow_count++; }

	void LogInput(uint64_t address, uint32_t value, uint32_t slot) {
		if (!m_record) {
			return;
		}
		if (m_log.size() >= kMaxLogEntries) {
			MarkOverflow();
			return;
		}
		m_log.push_back(DepEntry {address, value, slot});
	}

	struct MemoSlot {
		const Inst* key = nullptr;
		uint64_t    value = 0;
		// Slots belong to one evaluation; any other stamp reads as an empty slot.
		uint32_t    generation = 0;
		// The slice of the log this subtree appended, replayed on a hit.
		uint32_t    log_start  = 0;
		uint16_t    log_count  = 0;
		bool        overflowed = false;
		// "Currently being evaluated" lives here rather than in a separate vector: the old cycle
		// check was a linear scan of m_visiting on EVERY node, and expressions are 10-20 deep.
		bool        done = false;
	};

	// Clearing the table cost a memset of the whole array per Evaluator, twice per draw, on the
	// hottest path in the renderer. The arenas live for the thread instead and a generation stamp
	// retires the previous evaluation in O(1) - semantically identical to a cleared table, with
	// no cross-evaluation reuse, which is the thing that previously rendered coronas as red blobs.
	struct MemoArena {
		std::array<MemoSlot, kMemoSize> slots {};
		uint32_t                        generation = 0;
		bool                            in_use     = false;
	};

	static MemoArena* AcquireArena() {
		static thread_local std::array<MemoArena, kMemoArenas> arenas;
		for (auto& arena: arenas) {
			if (!arena.in_use) {
				arena.in_use = true;
				arena.generation++;
				if (arena.generation == 0) {
					// Wrapped: a surviving stamp of 0 would alias this evaluation. Clear once.
					arena.slots.fill(MemoSlot {});
					arena.generation = 1;
				}
				return &arena;
			}
		}
		return nullptr;
	}

	std::unique_ptr<MemoArena> m_owned_arena;   // only if every arena was already taken
	MemoArena*                 m_arena      = nullptr;
	uint32_t                   m_generation = 0;

	static size_t MemoIndex(const Inst* inst) {
		auto h = reinterpret_cast<uintptr_t>(inst);
		h ^= h >> 29;
		h *= 0xbf58476d1ce4e5b9ull;
		return static_cast<size_t>(h >> 33) & (kMemoSize - 1);
	}

	// Returns: 1 = resolved (out set), 0 = not present, -1 = currently being evaluated (cycle).
	static constexpr size_t kNoSlot2 = ~size_t {0};

	// Returns where the key belongs as well as what was found: every node used to hash and probe
	// three times - once to look up, once to mark visiting, once to insert - and the second two
	// were re-deriving a slot the first had already found.
	int MemoLookup(const Inst* inst, uint64_t& out, size_t& slot_index) {
		auto& slots = m_arena->slots;
		slot_index  = kNoSlot2;
		if (t_force_memo_miss) {
			return 0;   // test-only: behave as if the probe window were full
		}
		for (size_t i = MemoIndex(inst), probe = 0; probe < 8; probe++, i = (i + 1) & (kMemoSize - 1)) {
			auto& slot = slots[i];
			if (slot.generation != m_generation) {
				slot_index = i;   // first free slot: this is where an insert would go
				return 0;
			}
			if (slot.key == inst) {
				slot_index = i;
				if (!slot.done) {
					return -1;
				}
				out = slot.value;
				ReplayMemo(slot);
				return 1;
			}
		}
		return 0;   // table full after 8 probes: evaluate without memoising, as before
	}

	// The slot was already located by MemoLookup. Claiming it with our key means nested
	// evaluation cannot take it: an insert only writes a slot whose generation is stale or whose
	// key already matches, and ours is neither.
	void MemoMarkVisiting(const Inst* inst, size_t slot_index) {
		if (slot_index == kNoSlot2) {
			return;
		}
		auto& slot      = m_arena->slots[slot_index];
		slot            = MemoSlot {};
		slot.generation = m_generation;
		slot.key        = inst;
	}

	// A memo hit performs no reads, so without this the enclosing expression would record
	// nothing for the subtree it just reused - and a cache entry built from that window would
	// revalidate as unchanged no matter what those inputs did.
	void ReplayMemo(const MemoSlot& slot) {
		if (!m_record) {
			return;
		}
		if (slot.overflowed) {
			MarkOverflow();
			return;
		}
		if (slot.log_count == 0) {
			return;
		}
		// The generation stamp is what guarantees this window belongs to the current evaluation
		// and therefore indexes the current log. Check it anyway: getting that wrong reads other
		// memory rather than merely missing the cache.
		if (static_cast<size_t>(slot.log_start) + slot.log_count > m_log.size()) {
			MarkOverflow();
			return;
		}
		if (m_log.size() + slot.log_count > kMaxLogEntries) {
			MarkOverflow();
			return;
		}
		m_replay_scratch.assign(m_log.begin() + slot.log_start,
		                        m_log.begin() + slot.log_start + slot.log_count);
		m_log.insert(m_log.end(), m_replay_scratch.begin(), m_replay_scratch.end());
	}

	void MemoInsert(const Inst* inst, uint64_t value, uint32_t log_start, uint32_t overflow_start,
	                size_t slot_index) {
		if (slot_index != kNoSlot2) {
			auto&      slot = m_arena->slots[slot_index];
			const auto span = m_log.size() - log_start;
			slot.generation = m_generation;
			slot.key        = inst;
			slot.value      = value;
			slot.done       = true;
			slot.log_start  = log_start;
			slot.log_count  = static_cast<uint16_t>(span);
			slot.overflowed = m_overflow_count != overflow_start || span > kMaxLogEntries;
			return;
		}
		// Full after 8 probes: skip memoising. The value is simply recomputed; never wrong.
	}
	std::vector<const Inst*>                  m_visiting;
};

const DescriptorSource* Source(const Program& program, uint32_t source) {
	if (source >= program.descriptor_sources.size()) {
		return nullptr;
	}
	return &program.descriptor_sources[source];
}

namespace {

// Descriptor result cache with cheap re-validation.
//
// A cached entry stores exactly what its evaluation depended on. Validating it costs a handful
// of dword reads and compares (~30-80ns) instead of re-walking the expression tree (~370ns).
// Unlike a plain user-data key, this also covers descriptors that dereference guest memory -
// which is the majority - because the recorded (address, value) pairs prove the memory it
// actually read has not changed.
//
// Correctness: if every recorded user-data slot and every recorded memory location still holds
// the value it held when the result was computed, the expression is a pure function of those
// inputs and must produce the same result. Anything that read more locations than we track is
// marked overflowed and never cached.
struct DescriptorEntry {
	const void*              program      = nullptr;
	uint64_t                 program_hash = 0;   // raw pointers get recycled; hash does not
	uint32_t                 source       = 0;
	uint32_t                 dword_count  = 0;   // guards against a mismatched entry
	Evaluator::Dependencies  deps;
	DescriptorValue          value;
	bool                     valid = false;
};

// 256 entries x ~460 B ~= 118 KB. An earlier memo in this project measured worse at 24 KB
// than at 6 KB once it left L1, so size this deliberately rather than making it large.
constexpr size_t kDescCacheBits = 8;
constexpr size_t kDescCacheSize = size_t {1} << kDescCacheBits;

std::array<DescriptorEntry, kDescCacheSize>& DescriptorCache() {
	static thread_local std::array<DescriptorEntry, kDescCacheSize> cache;
	return cache;
}

// Hit rate decides whether this cache is worth its memory at all; without it, enabling the
// flag and reading a frame time cannot tell a working cache from a cache that never hits.
std::atomic<uint64_t> g_desc_lookups {0};
std::atomic<uint64_t> g_desc_hits {0};
std::atomic<uint64_t> g_desc_uncacheable {0};

void DescriptorCacheTick(bool hit, bool uncacheable) {
	const auto total = g_desc_lookups.fetch_add(1, std::memory_order_relaxed) + 1;
	if (hit) {
		g_desc_hits.fetch_add(1, std::memory_order_relaxed);
	}
	if (uncacheable) {
		g_desc_uncacheable.fetch_add(1, std::memory_order_relaxed);
	}
	if (total % 200000 == 0) {
		const auto hits = g_desc_hits.load(std::memory_order_relaxed);
		const auto over = g_desc_uncacheable.load(std::memory_order_relaxed);
		std::printf("DescriptorCacheCensus: lookups=%llu hits=%llu (%.1f%%) uncacheable=%llu "
		            "(%.1f%%)\n",
		            static_cast<unsigned long long>(total),
		            static_cast<unsigned long long>(hits),
		            100.0 * static_cast<double>(hits) / static_cast<double>(total),
		            static_cast<unsigned long long>(over),
		            100.0 * static_cast<double>(over) / static_cast<double>(total));
	}
}

size_t DescriptorSlot(const void* program, uint32_t source) {
	uint64_t h = reinterpret_cast<uintptr_t>(program) ^ (uint64_t {source} * 0x9e3779b97f4a7c15ull);
	h ^= h >> 29;
	h *= 0xbf58476d1ce4e5b9ull;
	return static_cast<size_t>(h >> 33) & (kDescCacheSize - 1);
}

// Re-read the recorded inputs and compare. Cheap: a few dwords.
bool DependenciesStillHold(const Evaluator::Dependencies& deps,
                           std::span<const uint32_t> user_data) {
	if (deps.overflowed) {
		return false;
	}
	for (uint32_t i = 0; i < deps.count; i++) {
		const auto& input = deps.inputs[i];
		if (input.slot != Evaluator::kNoSlot) {
			if (input.slot >= user_data.size() || user_data[input.slot] != input.value) {
				return false;
			}
		} else {
			uint32_t current = 0;
			std::memcpy(&current, reinterpret_cast<const void*>(input.address), sizeof(current));
			if (current != input.value) {
				return false;
			}
		}
	}
	return true;
}

} // namespace

} // namespace

// The compiled-SRT machinery is IR-scoped, not in the anonymous namespace: Program and
// CompiledSrtState name these types, and defining them anonymously made them distinct,
// unrelated types.
// Creates the state on first use. A single process-wide mutex is enough: this runs once per
// program, not per draw.
// The state block is allocated once by NumberEvalSlots, single-threaded, at shader-compile
// time. Doing it lazily here would need a lock on EVERY evaluation - a process-wide mutex
// on the exact path the draw workers run in parallel, which is the opposite of the point.
// Callers must check eval_slot_count first; it is 0 whenever this was not allocated.
CompiledSrtState& CompiledState(const Program& program) {
	return *program.compiled_state;
}

// Flat evaluation program for a Program. The interpreter above walks the value graph
// recursively with a memo per node; that costs tens of nanoseconds per visit and runs for every
// draw. The graph is static per plan, so it is compiled once into post-ordered ops over a slot
// array (Inst::GetEvalSlot for values, extra slots for immediates) and executed in one loop.
// Anything the compiler does not understand leaves the plan (or that descriptor source) on the
// interpreter. Results are cross-checked against the interpreter for the first evaluations of
// each plan; a mismatch disables the compiled path for that plan.
struct CompiledSrt {
	enum class Op : uint8_t {
		Const,
		UserData,
		ShaderBase,
		Alias,
		ExtractU64,
		CarryLow,
		CarryHigh,
		ConstructU64,
		RawReadAddress,
		RawReadBuffer,
		IAdd32,
		IAdd64,
		ISub32,
		ISub64,
		IMul32,
		IMul64,
		UMin32,
		ConvertF32U32,
		ConvertU32F32,
		FPMul32,
		FPTrunc32,
		FPIsNan32,
		FPOrdLessThanEqual32,
		FPOrdGreaterThanEqual32,
		BitwiseAnd32,
		BitwiseAnd64,
		BitwiseOr32,
		BitwiseXor32,
		BitwiseNot32,
		ShiftLeftLogical32,
		ShiftLeftLogical64,
		ShiftRightLogical32,
		ShiftRightLogical64,
		ShiftRightArithmetic32,
		ShiftRightArithmetic64,
		BitFieldUExtract,
		BitFieldSExtract,
		BitFieldInsert,
		Select,
		IEqual32,
		INotEqual32,
		ULessThan32,
		UGreaterThan32,
		LogicalAnd,
		LogicalOr,
		LogicalXor,
		LogicalNot,
	};

	struct Instr {
		Op       op  = Op::Const;
		uint32_t out = 0;
		uint32_t a   = 0;
		uint32_t b   = 0;
		uint32_t c   = 0;
		uint32_t d   = 0;
		uint64_t imm = 0;
	};

	static constexpr uint32_t Invalid = UINT32_MAX;

	std::vector<Instr>                   code;
	std::vector<std::array<uint32_t, 8>> source_slots;
	std::vector<uint8_t>                 source_compiled;
	std::vector<uint32_t>                read_slots;
	uint32_t                             slot_count = 0;
	// One past the highest user-data index any op reads. Checked BEFORE the op list runs so a
	// too-short runtime.user_data makes the compiled path decline up front instead of issuing
	// guest reads on its way to a failure the interpreter reaches earlier.
	uint32_t                             user_data_needed = 0;
	bool                                 flat_ok    = false;
	bool                                 ok         = false;
};

namespace {

class SrtCompiler {
public:
	using Op = CompiledSrt::Op;
	static constexpr uint32_t Invalid = CompiledSrt::Invalid;

	explicit SrtCompiler(const Program& program, CompiledSrt& out,
	                     std::span<const uint8_t> clean_flat_slots)
	    : m_program(program), m_out(out), m_clean_flat_slots(clean_flat_slots) {
		m_state.assign(program.eval_slot_count, 0);
		m_out.slot_count = program.eval_slot_count;
	}

	void Run() {
		if (m_program.eval_slot_count == 0) {
			m_out.ok = false;
			return;
		}
		m_out.source_slots.resize(m_program.descriptor_sources.size());
		m_out.source_compiled.assign(m_program.descriptor_sources.size(), 0);
		for (size_t index = 0; index < m_program.descriptor_sources.size(); index++) {
			const auto& source   = m_program.descriptor_sources[index];
			bool        compiled = source.dword_count <= 8;
			for (uint32_t dword = 0; compiled && dword < source.dword_count; dword++) {
				const auto slot = Compile(source.dwords[dword]);
				compiled        = slot != Invalid;
				m_out.source_slots[index][dword] = slot;
			}
			m_out.source_compiled[index] = compiled ? 1u : 0u;
		}
		m_out.read_slots.resize(m_program.srt_reads.size());
		m_out.flat_ok = true;
		for (size_t index = 0; index < m_program.srt_reads.size(); index++) {
			const auto slot          = Compile(m_program.srt_reads[index].value);
			m_out.read_slots[index] = slot;
			if (slot == Invalid) {
				m_out.flat_ok = false;
			}
		}
		m_out.ok = true;
	}

private:
	uint32_t NewSlot() { return m_out.slot_count++; }

	void Emit(Op op, uint32_t out, uint32_t a = 0, uint32_t b = 0, uint32_t c = 0, uint32_t d = 0,
	          uint64_t imm = 0) {
		m_out.code.push_back({op, out, a, b, c, d, imm});
	}

	uint32_t Compile(Value value) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			uint64_t bits = 0;
			switch (value.GetType()) {
				case Type::U1: bits = value.U1(); break;
				case Type::U8: bits = value.U8(); break;
				case Type::U16: bits = value.U16(); break;
				case Type::U32: bits = value.U32(); break;
				case Type::U64: bits = value.U64(); break;
				case Type::F32: bits = std::bit_cast<uint32_t>(value.F32Value()); break;
				default: return Invalid;
			}
			const auto slot = NewSlot();
			Emit(Op::Const, slot, 0, 0, 0, 0, bits);
			return slot;
		}
		auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			return Invalid;
		}
		const auto slot = inst->GetEvalSlot();
		if (slot >= m_program.eval_slot_count) {
			return Invalid;
		}
		if (m_state[slot] == 2) {
			return slot;
		}
		if (m_state[slot] == 1) {
			return Invalid;
		}
		m_state[slot] = 1;
		if (!CompileInst(*inst, slot)) {
			return Invalid;
		}
		m_state[slot] = 2;
		return slot;
	}

	bool Unary(const Inst& inst, Op op, uint32_t out) {
		const auto a = Compile(inst.Arg(0));
		if (a == Invalid) {
			return false;
		}
		Emit(op, out, a);
		return true;
	}

	bool Binary(const Inst& inst, Op op, uint32_t out) {
		const auto a = Compile(inst.Arg(0));
		const auto b = a == Invalid ? Invalid : Compile(inst.Arg(1));
		if (b == Invalid) {
			return false;
		}
		Emit(op, out, a, b);
		return true;
	}

	bool Ternary(const Inst& inst, Op op, uint32_t out) {
		const auto a = Compile(inst.Arg(0));
		const auto b = a == Invalid ? Invalid : Compile(inst.Arg(1));
		const auto c = b == Invalid ? Invalid : Compile(inst.Arg(2));
		if (c == Invalid) {
			return false;
		}
		Emit(op, out, a, b, c);
		return true;
	}

	bool CompileInst(const Inst& inst, uint32_t out) {
		switch (inst.GetOpcode()) {
			case ValueOpcode::GetUserData: {
				if (inst.NumArgs() != 1 || inst.Arg(0).GetType() != Type::ScalarReg) {
					return false;
				}
				const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
				if (reg < m_program.user_data_base) {
					return false;
				}
				const auto index    = reg - m_program.user_data_base;
				m_out.user_data_needed = std::max(m_out.user_data_needed, index + 1);
				Emit(Op::UserData, out, 0, 0, 0, 0, index);
				return true;
			}
			case ValueOpcode::GetShaderBase: Emit(Op::ShaderBase, out); return true;
			case ValueOpcode::Phi: {
				const auto invariant =
				    ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
				if (invariant.IsEmpty()) {
					return false;
				}
				const auto a = Compile(invariant);
				if (a == Invalid) {
					return false;
				}
				Emit(Op::Alias, out, a);
				return true;
			}
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32: return Unary(inst, Op::Alias, out);
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2: {
				const auto index = inst.Arg(1).Resolve();
				if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
					return false;
				}
				const auto component = index.U32();
				if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
					const auto a = Compile(inst.Arg(0));
					if (a == Invalid) {
						return false;
					}
					Emit(Op::ExtractU64, out, a, 0, 0, 0, component);
					return true;
				}
				const auto* source = inst.Arg(0).ResolveInstruction();
				if (source == nullptr) {
					return false;
				}
				if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
					if (source->NumArgs() < 2) {
						return false;
					}
					const auto a = Compile(source->Arg(component));
					if (a == Invalid) {
						return false;
					}
					Emit(Op::Alias, out, a);
					return true;
				}
				if (source->GetOpcode() == ValueOpcode::IAddCarry32) {
					if (source->NumArgs() < 2) {
						return false;
					}
					const auto a = Compile(source->Arg(0));
					const auto b = a == Invalid ? Invalid : Compile(source->Arg(1));
					if (b == Invalid) {
						return false;
					}
					Emit(component == 0u ? Op::CarryLow : Op::CarryHigh, out, a, b);
					return true;
				}
				return false;
			}
			case ValueOpcode::CompositeConstructU64: return Binary(inst, Op::ConstructU64, out);
			case ValueOpcode::ReadConst: {
				const auto slot = inst.Arg(1).Resolve();
				if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
				    slot.U32() >= m_program.srt_reads.size()) {
					return false;
				}
				if (slot.U32() < m_clean_flat_slots.size() &&
				    m_clean_flat_slots[slot.U32()] != 0u) {
					return false;
				}
				const auto a = Compile(m_program.srt_reads[slot.U32()].value);
				if (a == Invalid) {
					return false;
				}
				Emit(Op::Alias, out, a);
				return true;
			}
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer: {
				if (!IsRawRead(m_program, inst) || inst.NumArgs() < 2) {
					return false;
				}
				const auto  flags  = inst.Flags<MemoryFlags>();
				const auto& mem    = m_program.memory_info[flags.index];
				const auto* handle = inst.Arg(0).ResolveInstruction();
				if (handle == nullptr || handle->NumArgs() < 2) {
					return false;
				}
				const auto low    = Compile(handle->Arg(0));
				const auto high   = low == Invalid ? Invalid : Compile(handle->Arg(1));
				const auto offset = high == Invalid ? Invalid : Compile(inst.Arg(1));
				if (offset == Invalid) {
					return false;
				}
				if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer) {
					if (handle->NumArgs() != 4u ||
					    static_cast<int64_t>(static_cast<int32_t>(mem.offset)) < 0) {
						return false;
					}
					const auto records = Compile(handle->Arg(2));
					const auto word3   = records == Invalid ? Invalid : Compile(handle->Arg(3));
					if (word3 == Invalid) {
						return false;
					}
					Emit(Op::RawReadBuffer, out, low, high, offset, records, mem.offset);
					return true;
				}
				Emit(Op::RawReadAddress, out, low, high, offset, 0, mem.offset);
				return true;
			}
			case ValueOpcode::IAdd32: return Binary(inst, Op::IAdd32, out);
			case ValueOpcode::IAdd64: return Binary(inst, Op::IAdd64, out);
			case ValueOpcode::ISub32: return Binary(inst, Op::ISub32, out);
			case ValueOpcode::ISub64: return Binary(inst, Op::ISub64, out);
			case ValueOpcode::IMul32: return Binary(inst, Op::IMul32, out);
			case ValueOpcode::IMul64: return Binary(inst, Op::IMul64, out);
			case ValueOpcode::UMin32: return Binary(inst, Op::UMin32, out);
			case ValueOpcode::ConvertF32U32: return Unary(inst, Op::ConvertF32U32, out);
			case ValueOpcode::ConvertU32F32: return Unary(inst, Op::ConvertU32F32, out);
			case ValueOpcode::FPMul32: return Binary(inst, Op::FPMul32, out);
			case ValueOpcode::FPTrunc32: return Unary(inst, Op::FPTrunc32, out);
			case ValueOpcode::FPIsNan32: return Unary(inst, Op::FPIsNan32, out);
			case ValueOpcode::FPOrdLessThanEqual32:
				return Binary(inst, Op::FPOrdLessThanEqual32, out);
			case ValueOpcode::FPOrdGreaterThanEqual32:
				return Binary(inst, Op::FPOrdGreaterThanEqual32, out);
			case ValueOpcode::BitwiseAnd32: return Binary(inst, Op::BitwiseAnd32, out);
			case ValueOpcode::BitwiseAnd64: return Binary(inst, Op::BitwiseAnd64, out);
			case ValueOpcode::BitwiseOr32: return Binary(inst, Op::BitwiseOr32, out);
			case ValueOpcode::BitwiseXor32: return Binary(inst, Op::BitwiseXor32, out);
			case ValueOpcode::BitwiseNot32: return Unary(inst, Op::BitwiseNot32, out);
			case ValueOpcode::ShiftLeftLogical32: return Binary(inst, Op::ShiftLeftLogical32, out);
			case ValueOpcode::ShiftLeftLogical64: return Binary(inst, Op::ShiftLeftLogical64, out);
			case ValueOpcode::ShiftRightLogical32:
				return Binary(inst, Op::ShiftRightLogical32, out);
			case ValueOpcode::ShiftRightLogical64:
				return Binary(inst, Op::ShiftRightLogical64, out);
			case ValueOpcode::ShiftRightArithmetic32:
				return Binary(inst, Op::ShiftRightArithmetic32, out);
			case ValueOpcode::ShiftRightArithmetic64:
				return Binary(inst, Op::ShiftRightArithmetic64, out);
			case ValueOpcode::BitFieldUExtract: return Ternary(inst, Op::BitFieldUExtract, out);
			case ValueOpcode::BitFieldSExtract: return Ternary(inst, Op::BitFieldSExtract, out);
			case ValueOpcode::BitFieldInsert: {
				if (inst.NumArgs() < 4) {
					return false;
				}
				const auto a = Compile(inst.Arg(0));
				const auto b = a == Invalid ? Invalid : Compile(inst.Arg(1));
				const auto c = b == Invalid ? Invalid : Compile(inst.Arg(2));
				const auto d = c == Invalid ? Invalid : Compile(inst.Arg(3));
				if (d == Invalid) {
					return false;
				}
				Emit(Op::BitFieldInsert, out, a, b, c, d);
				return true;
			}
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectF32: return Ternary(inst, Op::Select, out);
			case ValueOpcode::IEqual32: return Binary(inst, Op::IEqual32, out);
			case ValueOpcode::INotEqual32: return Binary(inst, Op::INotEqual32, out);
			case ValueOpcode::ULessThan32: return Binary(inst, Op::ULessThan32, out);
			case ValueOpcode::UGreaterThan32: return Binary(inst, Op::UGreaterThan32, out);
			case ValueOpcode::LogicalAnd: return Binary(inst, Op::LogicalAnd, out);
			case ValueOpcode::LogicalOr: return Binary(inst, Op::LogicalOr, out);
			case ValueOpcode::LogicalXor: return Binary(inst, Op::LogicalXor, out);
			case ValueOpcode::LogicalNot: return Unary(inst, Op::LogicalNot, out);
			default: return false;
		}
	}

	const Program&           m_program;
	std::span<const uint8_t> m_clean_flat_slots;
	CompiledSrt&         m_out;
	std::vector<uint8_t> m_state;
};

bool RunCompiled(const CompiledSrt& compiled, const SrtRuntime& runtime,
                 std::vector<uint64_t>& slots) {
	using Op = CompiledSrt::Op;
	slots.resize(compiled.slot_count);
	auto* s = slots.data();
	for (const auto& i: compiled.code) {
		uint64_t& out = s[i.out];
		switch (i.op) {
			case Op::Const: out = i.imm; break;
			case Op::UserData:
				if (i.imm >= runtime.user_data.size()) {
					return false;
				}
				out = runtime.user_data[i.imm];
				break;
			case Op::ShaderBase: out = runtime.shader_base; break;
			case Op::Alias: out = s[i.a]; break;
			case Op::ExtractU64: out = static_cast<uint32_t>(s[i.a] >> (i.imm * 32u)); break;
			case Op::CarryLow:
			case Op::CarryHigh: {
				const auto sum = static_cast<uint64_t>(static_cast<uint32_t>(s[i.a])) +
				                 static_cast<uint32_t>(s[i.b]);
				out = i.op == Op::CarryLow ? static_cast<uint32_t>(sum)
				                           : static_cast<uint32_t>(sum >> 32u);
				break;
			}
			case Op::ConstructU64:
				out = static_cast<uint32_t>(s[i.a]) |
				      (static_cast<uint64_t>(static_cast<uint32_t>(s[i.b])) << 32u);
				break;
			case Op::RawReadAddress:
			case Op::RawReadBuffer: {
				const auto low       = s[i.a];
				const auto high      = s[i.b];
				const auto offset    = s[i.c];
				const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
				const auto immediate = static_cast<int64_t>(static_cast<int32_t>(i.imm));
				uint64_t   address   = 0;
				if (i.op == Op::RawReadBuffer) {
					const auto records = s[i.d];
					const auto byte_offset =
					    static_cast<uint64_t>(immediate) + static_cast<uint32_t>(offset);
					const auto aligned = byte_offset & ~uint64_t {3};
					const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
					const auto size    = stride == 0u
					                         ? static_cast<uint64_t>(static_cast<uint32_t>(records))
					                         : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
					if (aligned > size || size - aligned < sizeof(uint32_t)) {
						return false;
					}
					address = ((base & ~uint64_t {3}) + byte_offset) & ~uint64_t {3};
				} else {
					const auto relative = (immediate & ~int64_t {3}) +
					                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
					if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
						return false;
					}
				}
				uint32_t word = 0;
				if (runtime.read_memory != nullptr) {
					// i.out is the read node's eval_slot: Compile() returns the slot it numbered.
					const ReadNodeScope read_node {i.out};
					if (!runtime.read_memory(runtime.userdata, address, &word)) {
						return false;
					}
				} else {
					std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
				}
				out = word;
				break;
			}
			case Op::IAdd32: out = static_cast<uint32_t>(s[i.a] + s[i.b]); break;
			case Op::IAdd64: out = s[i.a] + s[i.b]; break;
			case Op::ISub32: out = static_cast<uint32_t>(s[i.a] - s[i.b]); break;
			case Op::ISub64: out = s[i.a] - s[i.b]; break;
			case Op::IMul32: out = static_cast<uint32_t>(s[i.a] * s[i.b]); break;
			case Op::IMul64: out = s[i.a] * s[i.b]; break;
			case Op::UMin32:
				out = std::min(static_cast<uint32_t>(s[i.a]), static_cast<uint32_t>(s[i.b]));
				break;
			case Op::ConvertF32U32:
				out = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(s[i.a])));
				break;
			case Op::ConvertU32F32: {
				const auto value = std::bit_cast<float>(static_cast<uint32_t>(s[i.a]));
				if (!std::isfinite(value) || value < 0.0f || static_cast<double>(value) > UINT32_MAX) {
					return false;
				}
				out = static_cast<uint32_t>(value);
				break;
			}
			case Op::FPMul32:
				out = std::bit_cast<uint32_t>(std::bit_cast<float>(static_cast<uint32_t>(s[i.a])) *
				                              std::bit_cast<float>(static_cast<uint32_t>(s[i.b])));
				break;
			case Op::FPTrunc32:
				out = std::bit_cast<uint32_t>(
				    std::trunc(std::bit_cast<float>(static_cast<uint32_t>(s[i.a]))));
				break;
			case Op::FPIsNan32:
				out = std::isnan(std::bit_cast<float>(static_cast<uint32_t>(s[i.a]))) ? 1u : 0u;
				break;
			case Op::FPOrdLessThanEqual32:
				out = std::bit_cast<float>(static_cast<uint32_t>(s[i.a])) <=
				              std::bit_cast<float>(static_cast<uint32_t>(s[i.b]))
				          ? 1u
				          : 0u;
				break;
			case Op::FPOrdGreaterThanEqual32:
				out = std::bit_cast<float>(static_cast<uint32_t>(s[i.a])) >=
				              std::bit_cast<float>(static_cast<uint32_t>(s[i.b]))
				          ? 1u
				          : 0u;
				break;
			case Op::BitwiseAnd32: out = static_cast<uint32_t>(s[i.a] & s[i.b]); break;
			case Op::BitwiseAnd64: out = s[i.a] & s[i.b]; break;
			case Op::BitwiseOr32: out = static_cast<uint32_t>(s[i.a] | s[i.b]); break;
			case Op::BitwiseXor32: out = static_cast<uint32_t>(s[i.a] ^ s[i.b]); break;
			case Op::BitwiseNot32: out = ~static_cast<uint32_t>(s[i.a]); break;
			case Op::ShiftLeftLogical32:
				out = static_cast<uint32_t>(s[i.a]) << (s[i.b] & 31u);
				break;
			case Op::ShiftLeftLogical64: out = s[i.a] << (s[i.b] & 63u); break;
			case Op::ShiftRightLogical32:
				out = static_cast<uint32_t>(s[i.a]) >> (s[i.b] & 31u);
				break;
			case Op::ShiftRightLogical64: out = s[i.a] >> (s[i.b] & 63u); break;
			case Op::ShiftRightArithmetic32:
				out = static_cast<uint32_t>(std::bit_cast<int32_t>(static_cast<uint32_t>(s[i.a])) >>
				                            (s[i.b] & 31u));
				break;
			case Op::ShiftRightArithmetic64:
				out = static_cast<uint64_t>(std::bit_cast<int64_t>(s[i.a]) >> (s[i.b] & 63u));
				break;
			case Op::BitFieldUExtract: {
				const auto offset = static_cast<uint32_t>(s[i.b]);
				const auto width  = static_cast<uint32_t>(s[i.c]);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				const auto mask = width == 32u ? UINT32_MAX : width == 0u ? 0u : (uint32_t {1} << width) - 1u;
				out = width == 0u ? 0u : (static_cast<uint32_t>(s[i.a]) >> offset) & mask;
				break;
			}
			case Op::BitFieldSExtract: {
				const auto offset = static_cast<uint32_t>(s[i.b]);
				const auto width  = static_cast<uint32_t>(s[i.c]);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				if (width == 0u) {
					out = 0;
					break;
				}
				const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
				auto       bits = (static_cast<uint32_t>(s[i.a]) >> offset) & mask;
				if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
					bits |= ~mask;
				}
				out = bits;
				break;
			}
			case Op::BitFieldInsert: {
				const auto offset = static_cast<uint32_t>(s[i.c]);
				const auto width  = static_cast<uint32_t>(s[i.d]);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				if (width == 0u) {
					out = static_cast<uint32_t>(s[i.a]);
					break;
				}
				const auto mask = width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
				out = (static_cast<uint32_t>(s[i.a]) & ~mask) |
				      ((static_cast<uint32_t>(s[i.b]) << offset) & mask);
				break;
			}
			case Op::Select: out = s[i.a] != 0u ? s[i.b] : s[i.c]; break;
			case Op::IEqual32:
				out = static_cast<uint32_t>(s[i.a]) == static_cast<uint32_t>(s[i.b]) ? 1u : 0u;
				break;
			case Op::INotEqual32:
				out = static_cast<uint32_t>(s[i.a]) != static_cast<uint32_t>(s[i.b]) ? 1u : 0u;
				break;
			case Op::ULessThan32:
				out = static_cast<uint32_t>(s[i.a]) < static_cast<uint32_t>(s[i.b]) ? 1u : 0u;
				break;
			case Op::UGreaterThan32:
				out = static_cast<uint32_t>(s[i.a]) > static_cast<uint32_t>(s[i.b]) ? 1u : 0u;
				break;
			case Op::LogicalAnd: out = (s[i.a] != 0u) && (s[i.b] != 0u) ? 1u : 0u; break;
			case Op::LogicalOr: out = (s[i.a] != 0u) || (s[i.b] != 0u) ? 1u : 0u; break;
			case Op::LogicalXor: out = (s[i.a] != 0u) != (s[i.b] != 0u) ? 1u : 0u; break;
			case Op::LogicalNot: out = s[i.a] == 0u ? 1u : 0u; break;
		}
	}
	return true;
}

// Runs the compiled program for the requested sources. Returns false when the plan (or one of
// the requested sources) is not compiled or evaluation failed; the caller then interprets.
bool EvaluateCompiled(const Program& program, std::span<const uint32_t> sources,
                      const SrtRuntime& runtime, std::vector<DescriptorValue>& results,
                      std::vector<uint32_t>& flat, bool evaluate_flat,
                      std::span<const uint8_t> clean_flat_slots) {
	if (std::ranges::any_of(clean_flat_slots, [](uint8_t clean) { return clean != 0u; })) {
		return false;
	}
	// Their version wrote compiled_srt_attempted/compiled_srt without synchronisation. Draw
	// workers evaluate concurrently here, so the build happens once under a mutex and the result
	// is read through a local shared_ptr copy that keeps it alive for this evaluation.
	auto&                              state = CompiledState(program);
	std::shared_ptr<const CompiledSrt> held;
	{
		std::lock_guard lock(state.mutex);
		if (!state.attempted) {
			state.attempted = true;
			auto built      = std::make_shared<CompiledSrt>();
			SrtCompiler(program, *built, clean_flat_slots).Run();
			if (built->ok) {
				state.compiled = std::move(built);
			}
		}
		held = state.compiled;
	}
	const auto* compiled = held.get();
	if (compiled == nullptr || !compiled->ok || (evaluate_flat && !compiled->flat_ok)) {
		return false;
	}
	for (const auto source_index: sources) {
		if (source_index >= compiled->source_compiled.size() ||
		    compiled->source_compiled[source_index] == 0u) {
			return false;
		}
	}
	// Decline BEFORE running any op if the runtime cannot satisfy every user-data index this
	// program reads. The op list is flat, so it would otherwise issue guest reads on its way to
	// a failure the demand-driven interpreter reaches without them - an observable difference in
	// the guest read pattern on failing evaluations, since a read can fault a page in.
	if (runtime.user_data.size() < compiled->user_data_needed) {
		return false;
	}
	thread_local std::vector<uint64_t> slots;
	if (!RunCompiled(*compiled, runtime, slots)) {
		return false;
	}
	// Every check that can fail runs before an output is touched, so a declined evaluation still
	// leaves both destinations exactly as the caller passed them. The outputs are then rebuilt in
	// place: both callers pass buffers they keep for reuse, and building locals here and
	// move-assigning them over those buffers threw the retained capacity away - two allocations and
	// two frees on every evaluation. The loops write through local pointers: this build disables
	// strict aliasing, so writing through the vectors themselves makes the compiler reload their data
	// pointers on every iteration, which cost more than the allocations on wide programs.
	const auto*  reads      = program.srt_reads.data();
	const size_t read_count = program.srt_reads.size();
	const auto*  values     = slots.data();
	if (evaluate_flat) {
		for (size_t index = 0; index < read_count; index++) {
			if (reads[index].flat_offset >= read_count) {
				return false;
			}
		}
	}
	results.resize(sources.size());
	auto* result_out = results.data();
	for (size_t out = 0; out < sources.size(); out++) {
		const auto      source_index = sources[out];
		const auto&     source       = program.descriptor_sources[source_index];
		const auto*     dword_slots  = compiled->source_slots[source_index].data();
		DescriptorValue value;
		value.dword_count = source.dword_count;
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			value.dwords[dword] = static_cast<uint32_t>(values[dword_slots[dword]]);
		}
		if (t_corrupt_compiled && value.dword_count > 0) {
			value.dwords[0] ^= 0xa5a5a5a5u;   // TEST-ONLY fault injection
		}
		result_out[out] = value;
	}
	if (evaluate_flat) {
		flat.assign(read_count, 0u);
		auto*       flat_out   = flat.data();
		const auto* read_slots = compiled->read_slots.data();
		for (size_t index = 0; index < read_count; index++) {
			flat_out[reads[index].flat_offset] = static_cast<uint32_t>(values[read_slots[index]]);
		}
	}
	return true;
}

bool EvaluateRuntimeSourcesInterpreted(const Program& program,
                                       std::span<const uint32_t> sources, const SrtRuntime& runtime,
                                       std::vector<DescriptorValue>& results,
                                       std::vector<uint32_t>& flat, bool evaluate_flat,
                                       std::span<const uint8_t> clean_flat_slots);


} // namespace

// Everything below stays in namespace IR: our EvaluateRuntimeSources et al are DEFINITIONS of
// the header's declarations, and putting them in an anonymous namespace made them separate
// overloads instead.
bool EvaluateRuntimeSourcesImpl(const Program& program,
                                std::span<const DescriptorSourceRequest> requests,
                                const SrtRuntime& runtime,
                                std::vector<DescriptorValue>& results,
                                std::vector<uint32_t>& flat, bool evaluate_flat,
                                std::span<const uint8_t> clean_flat_slots,
                                std::string* error, bool allow_compiled);

// Dense eval-slot numbering over the descriptor-reachable value subgraph.
//
// Their version gets this for free: ExtractResourcePlan clones exactly that subgraph into
// plan.value_storage and numbers it. We number in place over our Program instead, walking from
// the same roots (descriptor source dwords, indirect-image selectors, and srt_reads), so the flat
// slot array stays proportional to the descriptors rather than to the whole shader.
//
// Runs ONCE, after our fallible BuildSrtPlan has already succeeded. It never fails the plan: if
// numbering cannot complete, eval_slot_count stays 0 and the compiled path is simply not used.
void NumberEvalSlots(Program& program) {
	if (!program.srt_plan_complete || !program.resource_tracking_complete) {
		return;
	}
	uint32_t                     next = 0;
	std::vector<const Inst*>     stack;
	std::unordered_set<const Inst*> seen;

	const auto push = [&](Value value) {
		value = value.Resolve();
		if (auto* inst = value.TryInstruction(); inst != nullptr) {
			stack.push_back(inst);
		}
	};
	for (const auto& source: program.descriptor_sources) {
		for (uint32_t i = 0; i < source.dword_count && i < source.dwords.size(); i++) {
			push(source.dwords[i]);
		}
	}
	for (const auto& read: program.srt_reads) {
		push(read.value);
	}
	while (!stack.empty()) {
		const auto* inst = stack.back();
		stack.pop_back();
		if (!seen.insert(inst).second) {
			continue;
		}
		const_cast<Inst*>(inst)->SetEvalSlot(next++);
		for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
			push(inst->Arg(arg));
		}
	}
	if (next == 0) {
		return;
	}
	program.compiled_state  = std::make_shared<CompiledSrtState>();
	program.eval_slot_count = next;
}


// Externally visible from here: the header declares CompiledSrtUseCount, so it must not sit
// in the file's anonymous namespace.
std::atomic<uint64_t> g_compiled_uses {0};

thread_local CompiledSrtRejection t_last_rejection {};
std::atomic<uint64_t> g_compiled_rejections {0};
std::atomic<uint64_t> g_compiled_verifies {0};
std::atomic<uint64_t> g_compiled_inconclusive {0};
std::atomic<uint64_t> g_compiled_disabled_programs {0};
std::atomic<uint64_t> g_compiled_declined {0};

// Gate shared by the steady-state path and verification. Checks the state block itself, not just
// eval_slot_count: Program's move constructor moves compiled_state (leaving the source null) while
// leaving the source's eval_slot_count set, so a moved-from Program must not pass this.
bool CompiledSrtEligible(const Program& program) {
	return Config::CompiledSrtEnabled() && program.srt_plan_complete &&
	       program.eval_slot_count != 0 && static_cast<bool>(program.compiled_state);
}

const CompiledSrtRejection& CompiledSrtLastRejection() {
	return t_last_rejection;
}

void CompiledSrtReport() {
	std::printf("CompiledSrt: uses=%llu verified=%llu rejected=%llu inconclusive=%llu declined=%llu disabled_programs=%llu\n",
	            static_cast<unsigned long long>(g_compiled_uses.load()),
	            static_cast<unsigned long long>(g_compiled_verifies.load()),
	            static_cast<unsigned long long>(g_compiled_rejections.load()),
	            static_cast<unsigned long long>(g_compiled_inconclusive.load()),
	            static_cast<unsigned long long>(g_compiled_declined.load()),
	            static_cast<unsigned long long>(g_compiled_disabled_programs.load()));
	std::fflush(stdout);
}

uint64_t CompiledSrtInconclusiveCount() {
	return g_compiled_inconclusive.load(std::memory_order_relaxed);
}

uint64_t CompiledSrtVerifyCount() {
	return g_compiled_verifies.load(std::memory_order_relaxed);
}

uint64_t CompiledSrtRejectionCount() {
	return g_compiled_rejections.load(std::memory_order_relaxed);
}

void SetVerifyDropLastCaptureForTest(bool drop) {
	t_verify_drop_last_capture = drop;
}

void SetVerifyIgnoreReadNodeForTest(bool ignore) {
	t_verify_ignore_read_node = ignore;
}

void SetCompiledSrtCorruptForTest(bool corrupt) {
	t_corrupt_compiled = corrupt;
}

void SetSrtMemoMissForTest(bool force_miss) {
	t_force_memo_miss = force_miss;
}

uint64_t CompiledSrtUseCount() {
	return g_compiled_uses.load(std::memory_order_relaxed);
}

namespace {

// Verification input capture.
//
// The first version of this verifier ran the reference interpreter against the SAME live
// SrtRuntime, so it independently re-read mutable guest memory. That is wrong twice over: a benign
// guest write landing between the two evaluations is indistinguishable from a compiler divergence
// (and would permanently disable the compiled path), and it doubles the guest reads a safeguard is
// supposed to be transparent to.
//
// Instead the compiled run's reads are captured in order, and the interpreter is replayed against
// that capture. Both evaluators then see byte-identical inputs, so any surviving difference is a
// real divergence - and because replay is ordered and exact, a difference in WHICH addresses are
// read, in WHAT ORDER, or HOW MANY is caught too, not just a difference in the final values.
struct VerifyCapture {
	// A captured read is identified by its IR NODE, its evaluated address, its width, and whether
	// it succeeded - not by address alone. Address-only identity would let two distinct read nodes
	// at one address be treated as interchangeable, which they are not: guest memory can change
	// between them, so they may legitimately observe different values.
	struct Read {
		uint32_t node           = kNoReadNode;   // eval_slot of the IR node issuing the read
		uint64_t address        = 0;
		uint32_t size           = 0;             // dwords are the only width this reader exposes
		uint32_t value          = 0;
		bool     specialization = false;         // which reader: memory vs specialization memory
		bool     ok             = false;         // did the underlying read succeed
		bool     served         = false;         // did the reference ask for this read

		[[nodiscard]] bool SameIdentity(uint32_t other_node, uint64_t other_address,
		                                uint32_t other_size, bool other_specialization) const {
			return (t_verify_ignore_read_node || node == other_node) &&
			       address == other_address && size == other_size &&
			       specialization == other_specialization;
		}
	};

	SrtMemoryReader   read_memory                = nullptr;
	SrtMemoryReader   read_specialization_memory = nullptr;
	void*             userdata                   = nullptr;
	std::vector<Read> trace;
	size_t            cursor     = 0;   // next UNSERVED capture entry, in order
	size_t            served     = 0;   // total reads served to the reference
	bool              diverged   = false;
	// The interpreter read one node at one address twice and got DIFFERENT values, so the inputs
	// were not stable for the duration of its own evaluation. Its result then depends on which
	// consumer saw which value, and no single-valued replay can reproduce it. Verification is
	// abandoned without penalty rather than blamed on the compiler.
	bool              inconclusive = false;
	CompiledSrtRejection::Reason divergence = CompiledSrtRejection::Reason::None;
	uint64_t          want_address = 0;
	uint64_t          got_address  = 0;
};

bool CaptureOne(VerifyCapture* capture, SrtMemoryReader reader, uint64_t address,
                bool specialization, uint32_t* value) {
	const bool ok = reader != nullptr && reader(capture->userdata, address, value);
	capture->trace.push_back({t_read_node, address, static_cast<uint32_t>(sizeof(uint32_t)),
	                          ok ? *value : 0u, specialization, ok, false});
	return ok;
}

bool CaptureRead(void* userdata, uint64_t address, uint32_t* value) {
	auto* capture = static_cast<VerifyCapture*>(userdata);
	return CaptureOne(capture, capture->read_memory, address, false, value);
}

bool CaptureSpecializationRead(void* userdata, uint64_t address, uint32_t* value) {
	auto* capture = static_cast<VerifyCapture*>(userdata);
	return CaptureOne(capture, capture->read_specialization_memory, address, true, value);
}

// Replay preserves ORDER between distinct reads and tolerates only one specific relaxation: a
// repeat of a read this same node already performed at the same address and width.
//
// That relaxation is sound for exactly one reason, and it is worth stating rather than assuming:
// the interpreter's own memo already reuses a node's value for the rest of an evaluation when the
// memo hits. Whether a given node re-reads or reuses is therefore already unobservable to the
// program's semantics - it depends on a pointer-hashed 8-probe window, not on anything the shader
// expresses. Serving the repeat from the snapshot yields exactly what a memo HIT would have
// yielded. No such argument covers a different node, a different address, a different width, or a
// read that has not happened yet, so none of those are tolerated.
bool ReplayOne(VerifyCapture* capture, uint64_t address, bool specialization, uint32_t* value) {
	const auto size = static_cast<uint32_t>(sizeof(uint32_t));
	const auto node = t_read_node;

	// In-order match against the next unserved entry: the normal case.
	if (capture->cursor < capture->trace.size()) {
		auto& next = capture->trace[capture->cursor];
		if (next.SameIdentity(node, address, size, specialization)) {
			next.served = true;
			capture->cursor++;
			capture->served++;
			while (capture->cursor < capture->trace.size() &&
			       capture->trace[capture->cursor].served) {
				capture->cursor++;
			}
			if (!next.ok) {
				return false;   // the captured read FAILED; replay reproduces the failure
			}
			*value = next.value;
			return true;
		}
	}

	// Otherwise: only a repeat of an ALREADY-SERVED read by the same node is acceptable.
	for (auto& record: capture->trace) {
		if (!record.served || !record.SameIdentity(node, address, size, specialization)) {
			continue;
		}
		capture->served++;
		if (!record.ok) {
			return false;
		}
		*value = record.value;
		return true;
	}

	// Anything else is a genuine divergence. Classify it as precisely as the capture allows.
	capture->diverged    = true;
	capture->got_address = address;
	capture->want_address =
	    capture->cursor < capture->trace.size() ? capture->trace[capture->cursor].address : 0;
	if (capture->cursor >= capture->trace.size()) {
		capture->divergence = CompiledSrtRejection::Reason::ReadExhausted;
	} else {
		const auto& next = capture->trace[capture->cursor];
		if (next.address != address) {
			// Either a different address entirely, or the right one out of order.
			const bool later = std::ranges::any_of(capture->trace, [&](const auto& r) {
				return !r.served && r.SameIdentity(node, address, size, specialization);
			});
			capture->divergence = later ? CompiledSrtRejection::Reason::ReadOrder
			                            : CompiledSrtRejection::Reason::ReadAddress;
		} else if (next.specialization != specialization) {
			capture->divergence = CompiledSrtRejection::Reason::ReadReader;
		} else if (next.size != size) {
			capture->divergence = CompiledSrtRejection::Reason::ReadSize;
		} else {
			capture->divergence = CompiledSrtRejection::Reason::ReadNode;
		}
	}
	return false;
}

// Replay for the COMPILED evaluator, against inputs captured from the interpreter.
//
// Direction matters. The interpreter now runs first and live, so the capture is the authoritative
// record of what this evaluation actually read. The compiled evaluator is replayed against it and
// never touches guest memory, which is why a mismatch cannot cost a second live pass.
//
// The compiled evaluator computes each slot exactly once, so it never repeats a read; the
// interpreter may, when its memo evicts. Repeats therefore appear in the CAPTURE, not in the
// requests, and are resolved here: entries sharing (node, address, width, reader) must agree on
// their value, otherwise the interpreter observed unstable memory and verification is abandoned.
bool ReplayForCompiled(VerifyCapture* capture, uint64_t address, bool specialization,
                       uint32_t* value) {
	const auto size = static_cast<uint32_t>(sizeof(uint32_t));
	const auto node = t_read_node;

	const VerifyCapture::Read* found = nullptr;
	for (auto& record: capture->trace) {
		if (!record.SameIdentity(node, address, size, specialization)) {
			continue;
		}
		if (found != nullptr && (found->value != record.value || found->ok != record.ok)) {
			capture->inconclusive = true;
			return false;
		}
		record.served = true;
		found         = &record;
	}
	if (found == nullptr) {
		// The compiled evaluator wants a read the interpreter never performed.
		capture->diverged    = true;
		capture->divergence  = CompiledSrtRejection::Reason::ReadUnavailable;
		capture->got_address = address;
		return false;
	}
	capture->served++;
	if (!found->ok) {
		return false;
	}
	*value = found->value;
	return true;
}

bool ReplayForCompiledRead(void* userdata, uint64_t address, uint32_t* value) {
	return ReplayForCompiled(static_cast<VerifyCapture*>(userdata), address, false, value);
}

bool ReplayForCompiledSpecializationRead(void* userdata, uint64_t address, uint32_t* value) {
	return ReplayForCompiled(static_cast<VerifyCapture*>(userdata), address, true, value);
}

bool ReplayRead(void* userdata, uint64_t address, uint32_t* value) {
	return ReplayOne(static_cast<VerifyCapture*>(userdata), address, false, value);
}

bool ReplaySpecializationRead(void* userdata, uint64_t address, uint32_t* value) {
	return ReplayOne(static_cast<VerifyCapture*>(userdata), address, true, value);
}

} // namespace

// Gate + verification wrapper. Returns true only when the compiled path produced a result that
// is being used; every other outcome leaves the caller to interpret.
// Steady-state compiled evaluation. NO verification happens here any more: verification is
// interpreter-first (see VerifyCompiledAgainstCapture), so by the time this runs the program has
// already been checked or was never eligible.
bool TryEvaluateCompiled(const Program& program, std::span<const DescriptorSourceRequest> requests,
                         const SrtRuntime& runtime, std::vector<DescriptorValue>& results,
                         std::vector<uint32_t>& flat, bool evaluate_flat,
                         std::span<const uint8_t> clean_flat_slots) {
	if (!CompiledSrtEligible(program)) {
		return false;
	}
	auto& state = CompiledState(program);
	if (state.disabled.load(std::memory_order_relaxed) ||
	    state.verify_left.load(std::memory_order_relaxed) > 0) {
		return false;
	}
	static thread_local std::vector<uint32_t> sources;
	sources.clear();
	sources.reserve(requests.size());
	for (const auto& request: requests) {
		sources.push_back(request.source);
	}
	static thread_local std::vector<DescriptorValue> compiled_results;
	static thread_local std::vector<uint32_t>        compiled_flat;
	compiled_results.clear();
	compiled_flat.clear();
	if (!EvaluateCompiled(program, sources, runtime, compiled_results, compiled_flat, evaluate_flat,
	                      clean_flat_slots)) {
		return false;
	}
	results = compiled_results;
	if (evaluate_flat) {
		flat = compiled_flat;
	}
	// Report from HERE as well as the interpreter path. The interpreter-side report never fires
	// once the compiled path is serving evaluations, because this function returns before
	// reaching it - which is exactly the run where coverage most needs measuring.
	const auto uses = g_compiled_uses.fetch_add(1, std::memory_order_relaxed) + 1;
	if (uses % 5000000 == 0) {
		CompiledSrtReport();
	}
	return true;
}

// INTERPRETER-FIRST VERIFICATION.
//
// The interpreter has already run against live guest memory through a capture shim, and its result
// is what the caller receives no matter what happens here. The compiled evaluator is then replayed
// against those captured inputs and compared.
//
// The point of this ordering: a mismatch can never trigger a second live evaluation, because the
// live one already happened and its answer is already in hand. Guest reads during a verified
// evaluation are exactly the reads the interpreter would have made with compilation switched off -
// not "bounded extra reads", none.
void VerifyCompiledAgainstCapture(const Program& program,
                                  std::span<const DescriptorSourceRequest> requests,
                                  const SrtRuntime& runtime, VerifyCapture& capture,
                                  const std::vector<DescriptorValue>& interpreted_results,
                                  const std::vector<uint32_t>& interpreted_flat, bool evaluate_flat,
                                  std::span<const uint8_t> clean_flat_slots) {
	auto& state = CompiledState(program);

	static thread_local std::vector<uint32_t> sources;
	sources.clear();
	sources.reserve(requests.size());
	for (const auto& request: requests) {
		sources.push_back(request.source);
	}

	SrtRuntime replay_runtime  = runtime;
	replay_runtime.userdata    = &capture;
	replay_runtime.read_memory = runtime.read_memory == nullptr ? nullptr : ReplayForCompiledRead;
	replay_runtime.read_specialization_memory =
	    runtime.read_specialization_memory == nullptr ? nullptr
	                                                  : ReplayForCompiledSpecializationRead;

	if (t_verify_drop_last_capture && !capture.trace.empty()) {
		capture.trace.pop_back();   // TEST-ONLY fault injection
	}

	static thread_local std::vector<DescriptorValue> compiled_results;
	static thread_local std::vector<uint32_t>        compiled_flat;
	compiled_results.clear();
	compiled_flat.clear();
	const bool produced = EvaluateCompiled(program, sources, replay_runtime, compiled_results,
	                                       compiled_flat, evaluate_flat, clean_flat_slots);

	if (capture.inconclusive) {
		g_compiled_inconclusive.fetch_add(1, std::memory_order_relaxed);
		// Guest memory moved under the interpreter itself. Neither evaluator is at fault and no
		// conclusion is available, so the budget entry is spent and nothing is disabled.
		return;
	}
	if (!produced && !capture.diverged) {
		// The compiled path simply declined this program (unsupported shape, uncompiled source,
		// insufficient user data). That is not a mismatch; it would decline in steady state too.
		g_compiled_declined.fetch_add(1, std::memory_order_relaxed);
		return;
	}

	auto& rejection           = t_last_rejection;
	rejection                 = {};
	rejection.captured_reads  = static_cast<uint32_t>(capture.trace.size());
	rejection.reference_reads = static_cast<uint32_t>(capture.served);

	if (capture.diverged) {
		rejection.reason   = capture.divergence;
		rejection.expected = capture.want_address;
		rejection.actual   = capture.got_address;
	} else if (interpreted_results != compiled_results) {
		rejection.reason = CompiledSrtRejection::Reason::ResultsDiffer;
		for (size_t i = 0; i < compiled_results.size(); i++) {
			if (i >= interpreted_results.size()) {
				rejection.index = static_cast<uint32_t>(i);
				break;
			}
			if (interpreted_results[i] == compiled_results[i]) {
				continue;
			}
			rejection.index = static_cast<uint32_t>(i);
			if (interpreted_results[i].dword_count != compiled_results[i].dword_count) {
				rejection.dword    = UINT32_MAX;
				rejection.expected = interpreted_results[i].dword_count;
				rejection.actual   = compiled_results[i].dword_count;
				break;
			}
			for (uint32_t d = 0; d < interpreted_results[i].dwords.size(); d++) {
				if (interpreted_results[i].dwords[d] != compiled_results[i].dwords[d]) {
					rejection.dword    = d;
					rejection.expected = interpreted_results[i].dwords[d];
					rejection.actual   = compiled_results[i].dwords[d];
					break;
				}
			}
			break;
		}
	} else if (evaluate_flat && interpreted_flat != compiled_flat) {
		rejection.reason = CompiledSrtRejection::Reason::FlatDiffers;
		for (size_t i = 0; i < compiled_flat.size(); i++) {
			if (i >= interpreted_flat.size() || interpreted_flat[i] != compiled_flat[i]) {
				rejection.index    = static_cast<uint32_t>(i);
				rejection.expected = i < interpreted_flat.size() ? interpreted_flat[i] : 0;
				rejection.actual   = compiled_flat[i];
				break;
			}
		}
	}

	if (rejection.reason != CompiledSrtRejection::Reason::None) {
		g_compiled_rejections.fetch_add(1, std::memory_order_relaxed);
		// Disable only. The caller already holds the interpreter result; there is nothing to
		// recompute and no live memory to touch again.
		if (!state.disabled.exchange(true, std::memory_order_relaxed)) {
			g_compiled_disabled_programs.fetch_add(1, std::memory_order_relaxed);
		}
		std::printf("CompiledSrt: DISABLED hash=0x%016llx reason=%d index=%u dword=%d exp=0x%llx act=0x%llx reads=%u\n",
		            static_cast<unsigned long long>(program.shader_hash),
		            static_cast<int>(rejection.reason), rejection.index,
		            static_cast<int>(rejection.dword),
		            static_cast<unsigned long long>(rejection.expected),
		            static_cast<unsigned long long>(rejection.actual), rejection.captured_reads);
		std::fflush(stdout);
	}
}

bool EvaluateRuntimeSourcesImpl(const Program&                           program,
                                std::span<const DescriptorSourceRequest> requests,
                                const SrtRuntime& runtime, std::vector<DescriptorValue>& results,
                                std::vector<uint32_t>& flat, bool evaluate_flat,
                                std::span<const uint8_t> clean_flat_slots, std::string* error,
                                bool allow_compiled) {
	// Compiled path first, when enabled. It is attempted only after our own plan succeeded, and
	// EVERY failure inside - unsupported opcode, failed compile, uncompiled source, clean flat
	// slots, verification mismatch - returns false and leaves the interpreter below to run
	// unchanged. Nothing here can fail the plan or abort.
	if (allow_compiled && program.srt_plan_complete &&
	    TryEvaluateCompiled(program, requests, runtime, results, flat, evaluate_flat,
	                        clean_flat_slots)) {
		return true;
	}

	// Verification is INTERPRETER-FIRST. A budget slot is claimed with a CAS before evaluating, so
	// concurrent evaluations of one program cannot both spend the same entry; whoever claims one
	// runs the interpreter below through a capture shim and then replays the compiled evaluator
	// against those captured bytes. The interpreter's result is returned either way, which is why
	// a mismatch never causes a second live pass over guest memory.
	bool          verifying = false;
	VerifyCapture capture;
	SrtRuntime    capture_runtime = runtime;
	if (allow_compiled && CompiledSrtEligible(program) &&
	    !CompiledState(program).disabled.load(std::memory_order_relaxed)) {
		auto& state = CompiledState(program);
		for (auto left = state.verify_left.load(std::memory_order_relaxed); left > 0;) {
			if (state.verify_left.compare_exchange_weak(left, left - 1,
			                                            std::memory_order_relaxed)) {
				verifying = true;
				break;
			}
		}
		if (verifying) {
			capture.read_memory                = runtime.read_memory;
			capture.read_specialization_memory = runtime.read_specialization_memory;
			capture.userdata                   = runtime.userdata;
			capture_runtime.userdata           = &capture;
			capture_runtime.read_memory = runtime.read_memory == nullptr ? nullptr : CaptureRead;
			capture_runtime.read_specialization_memory =
			    runtime.read_specialization_memory == nullptr ? nullptr
			                                                  : CaptureSpecializationRead;
		}
	}
	// Everything below reads guest memory through this. It is the caller's runtime except while
	// verifying, when it is the same readers wrapped in a recorder.
	const SrtRuntime& live = verifying ? capture_runtime : runtime;
	if (!program.srt_plan_complete) {
		if (error != nullptr) {
			*error = Diagnostic(program, 0, "typed SRT plan is not ready");
		}
		return false;
	}
	if (std::ranges::any_of(clean_flat_slots, [](uint8_t clean) { return clean != 0u; }) &&
	    live.read_specialization_memory == nullptr) {
		if (error != nullptr) {
			*error = Diagnostic(program, 0, "clean flattened SRT read has no memory reader");
		}
		return false;
	}
	SrtRuntime clean_runtime  = live;
	clean_runtime.read_memory = live.read_specialization_memory;
	const bool                   use_cache = Config::CacheDescriptors();
	Evaluator                    clean_evaluator(program, clean_runtime);
	Evaluator                    evaluator(program, live, clean_flat_slots, &clean_evaluator);
	evaluator.SetRecording(use_cache);
	// Pooled rather than local: this runs twice a draw, and a fresh vector here is a malloc and
	// a free every time. The transactional guarantee is unchanged - the destination is still only
	// written once everything has succeeded.
	static thread_local std::vector<DescriptorValue> evaluated;
	evaluated.clear();
	evaluated.reserve(requests.size());
	for (const auto& request: requests) {
		const auto* source = Source(program, request.source);
		if (source == nullptr) {
			if (error != nullptr) {
				*error = Diagnostic(program, request.use_pc,
				                    fmt::format("invalid descriptor source {}", request.source));
			}
			return false;
		}
		evaluator.SetUsePc(request.use_pc);
		DescriptorValue value;
		value.dword_count = source->dword_count;

		auto& slot = DescriptorCache()[DescriptorSlot(&program, request.source)];
		if (use_cache && slot.valid && slot.program == &program &&
		    slot.program_hash == program.shader_hash && slot.source == request.source &&
		    slot.dword_count == source->dword_count &&
		    DependenciesStillHold(slot.deps, runtime.user_data)) {
			DescriptorCacheTick(true, false);
			evaluated.push_back(slot.value);
			continue;
		}

		t_srt_resolutions++;
		evaluator.BeginRecording();
		for (uint32_t index = 0; index < source->dword_count; index++) {
			if (!evaluator.Evaluate(source->dwords[index], value.dwords[index], error)) {
				return false;
			}
		}
		if (use_cache) {
			const auto deps = evaluator.RecordedDependencies();
			DescriptorCacheTick(false, deps.overflowed);
			if (!deps.overflowed) {
				slot.program      = &program;
				slot.program_hash = program.shader_hash;
				slot.source       = request.source;
				slot.dword_count  = source->dword_count;
				slot.deps         = deps;
				slot.value        = value;
				slot.valid        = true;
			} else if (slot.program == &program && slot.source == request.source) {
				slot.valid = false;
			}
		}
		evaluated.push_back(value);
	}
	static thread_local std::vector<uint32_t> flattened;
	flattened.clear();
	if (evaluate_flat) {
		flattened.resize(program.srt_reads.size());
		for (const auto& read: program.srt_reads) {
			const bool clean    = read.flat_offset < clean_flat_slots.size() &&
			                      clean_flat_slots[read.flat_offset] != 0u;
			auto&      selected = clean ? clean_evaluator : evaluator;
			selected.SetUsePc(read.use_pc);
			if (read.flat_offset >= flattened.size() ||
			    !selected.Evaluate(read.value, flattened[read.flat_offset], error)) {
				return false;
			}
		}
	}
	if (Config::CompiledSrtEnabled() && ++t_compiled_report % 200000 == 0) {
		std::printf("CompiledSrt: uses=%llu verified=%llu rejected=%llu inconclusive=%llu declined=%llu disabled_programs=%llu\n",
		            static_cast<unsigned long long>(g_compiled_uses.load()),
		            static_cast<unsigned long long>(g_compiled_verifies.load()),
		            static_cast<unsigned long long>(g_compiled_rejections.load()),
		            static_cast<unsigned long long>(g_compiled_inconclusive.load()),
		            static_cast<unsigned long long>(g_compiled_declined.load()),
		            static_cast<unsigned long long>(g_compiled_disabled_programs.load()));
		std::fflush(stdout);
	}
	if (++t_srt_calls % 100000 == 0) {
		std::printf("SrtWork: per call - resolutions=%.1f nodes=%.1f entries=%.1f (imm=%.1f "
		            "memo_hit=%.1f) reads=%.1f\n",
		            static_cast<double>(t_srt_resolutions) / 100000.0,
		            static_cast<double>(t_srt_nodes) / 100000.0,
		            static_cast<double>(t_srt_entries) / 100000.0,
		            static_cast<double>(t_srt_immediates) / 100000.0,
		            static_cast<double>(t_srt_hits) / 100000.0,
		            static_cast<double>(t_srt_reads) / 100000.0);
		std::fflush(stdout);
		t_srt_nodes       = 0;
		t_srt_entries     = 0;
		t_srt_immediates  = 0;
		t_srt_hits        = 0;
		t_srt_reads       = 0;
		t_srt_read_cycles = 0;
		t_srt_resolutions = 0;
	}
	// The interpreter has succeeded and its answer is final. Only now is the compiled evaluator
	// replayed against what the interpreter actually read, so a disagreement costs a disable and
	// nothing else.
	if (verifying) {
		g_compiled_verifies.fetch_add(1, std::memory_order_relaxed);
		VerifyCompiledAgainstCapture(program, requests, runtime, capture, evaluated, flattened,
		                             evaluate_flat, clean_flat_slots);
	}
	// assign, not move: moving would hand the pool's buffer away and allocate a fresh one next
	// draw, and it would also throw away the destination's capacity. Both sides stay warm.
	results.assign(evaluated.begin(), evaluated.end());
	if (evaluate_flat) {
		flat.assign(flattened.begin(), flattened.end());
	}
	return true;
}


bool ValidateRuntimeValue(const Program& program, Value value, std::string& reason) {
	return RuntimeValidator(program).Run(value, reason);
}

bool BuildSrtPlan(Program& program, std::string* error) {
	if (program.resource_tracking_complete) {
		if (error != nullptr) {
			*error = "cannot rebuild SRT after resource tracking";
		}
		return false;
	}
	program.srt_plan_complete = false;
	if (!PlanBuilder(program).Run(error)) {
		return false;
	}
	program.srt_plan_complete = true;
	return true;
}

bool EvaluateDescriptorSource(const Program& program, uint32_t source, uint32_t use_pc,
                              const SrtRuntime& runtime, DescriptorValue& result,
                              std::string* error) {
	const DescriptorSourceRequest request {source, use_pc};
	std::vector<DescriptorValue>  results;
	if (!EvaluateDescriptorSources(program, std::span {&request, 1}, runtime, results, error)) {
		return false;
	}
	result = results.front();
	return true;
}

bool EvaluateDescriptorSources(const Program&                           program,
                               std::span<const DescriptorSourceRequest> requests,
                               const SrtRuntime& runtime, std::vector<DescriptorValue>& results,
                               std::string* error) {
	std::vector<uint32_t> ignored;
	return EvaluateRuntimeSourcesImpl(program, requests, runtime, results, ignored, false, {},
	                                  error, /*allow_compiled=*/true);
}

bool EvaluateRuntimeSources(const Program&                           program,
                            std::span<const DescriptorSourceRequest> requests,
                            const SrtRuntime& runtime, std::vector<DescriptorValue>& results,
                            std::vector<uint32_t>& flat, std::span<const uint8_t> clean_flat_slots,
                            std::string* error) {
	return EvaluateRuntimeSourcesImpl(program, requests, runtime, results, flat, true,
	                                  clean_flat_slots, error, /*allow_compiled=*/true);
}

bool WalkSrt(const Program& program, const SrtRuntime& runtime, std::vector<uint32_t>& flat,
             std::string* error) {
	std::vector<DescriptorValue> ignored;
	return EvaluateRuntimeSources(program, {}, runtime, ignored, flat, {}, error);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
