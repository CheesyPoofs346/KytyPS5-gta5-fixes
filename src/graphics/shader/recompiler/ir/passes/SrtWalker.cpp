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
thread_local uint64_t t_srt_read_cycles = 0;
thread_local uint64_t t_srt_nodes       = 0;
thread_local uint64_t t_srt_calls       = 0;
thread_local uint64_t t_srt_resolutions = 0;


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
		value = value.Resolve();
		if (value.IsImmediate()) {
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

bool EvaluateRuntimeSourcesImpl(const Program&                           program,
                                std::span<const DescriptorSourceRequest> requests,
                                const SrtRuntime& runtime, std::vector<DescriptorValue>& results,
                                std::vector<uint32_t>& flat, bool evaluate_flat,
                                std::span<const uint8_t> clean_flat_slots, std::string* error) {
	if (!program.srt_plan_complete) {
		if (error != nullptr) {
			*error = Diagnostic(program, 0, "typed SRT plan is not ready");
		}
		return false;
	}
	if (std::ranges::any_of(clean_flat_slots, [](uint8_t clean) { return clean != 0u; }) &&
	    runtime.read_specialization_memory == nullptr) {
		if (error != nullptr) {
			*error = Diagnostic(program, 0, "clean flattened SRT read has no memory reader");
		}
		return false;
	}
	SrtRuntime clean_runtime  = runtime;
	clean_runtime.read_memory = runtime.read_specialization_memory;
	const bool                   use_cache = Config::CacheDescriptors();
	Evaluator                    clean_evaluator(program, clean_runtime);
	Evaluator                    evaluator(program, runtime, clean_flat_slots, &clean_evaluator);
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
	if (++t_srt_calls % 100000 == 0) {
		std::printf("SrtWork: per call - resolutions=%.1f nodes=%.1f reads=%.1f read=%.0f cyc "
		            "(%.1f cyc/read), walk cycles are the rest\n",
		            static_cast<double>(t_srt_resolutions) / 100000.0,
		            static_cast<double>(t_srt_nodes) / 100000.0,
		            static_cast<double>(t_srt_reads) / 100000.0,
		            static_cast<double>(t_srt_read_cycles) / 100000.0,
		            t_srt_reads != 0 ? static_cast<double>(t_srt_read_cycles) /
		                                   static_cast<double>(t_srt_reads)
		                             : 0.0);
		std::fflush(stdout);
		t_srt_nodes       = 0;
		t_srt_reads       = 0;
		t_srt_read_cycles = 0;
		t_srt_resolutions = 0;
	}
	// assign, not move: moving would hand the pool's buffer away and allocate a fresh one next
	// draw, and it would also throw away the destination's capacity. Both sides stay warm.
	results.assign(evaluated.begin(), evaluated.end());
	if (evaluate_flat) {
		flat.assign(flattened.begin(), flattened.end());
	}
	return true;
}

} // namespace

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
	                                  error);
}

bool EvaluateRuntimeSources(const Program&                           program,
                            std::span<const DescriptorSourceRequest> requests,
                            const SrtRuntime& runtime, std::vector<DescriptorValue>& results,
                            std::vector<uint32_t>& flat, std::span<const uint8_t> clean_flat_slots,
                            std::string* error) {
	return EvaluateRuntimeSourcesImpl(program, requests, runtime, results, flat, true,
	                                  clean_flat_slots, error);
}

bool WalkSrt(const Program& program, const SrtRuntime& runtime, std::vector<uint32_t>& flat,
             std::string* error) {
	std::vector<DescriptorValue> ignored;
	return EvaluateRuntimeSources(program, {}, runtime, ignored, flat, {}, error);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
