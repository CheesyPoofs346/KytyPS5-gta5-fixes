#include <chrono>
#include "graphics/guest_gpu/graphicsRun.h"

#include "graphics/host_gpu/renderer/drawProfile.h"
#include "graphics/host_gpu/renderer/drawStateSnapshot.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/guest_gpu/occlusionQueries.h"
#include "graphics/presentation/renderDoc.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <thread>
#include <vector>

namespace Libs::Graphics {

// Frame thread census.
//
// Every measurement so far has been INSIDE a draw. None of it explains a 16-thread machine sitting
// at 20-30%. This measures the frame at the thread level instead: how long the guest thread is
// blocked in WaitForIdle, and how long the GPU thread is actually processing versus idle.
//
// The architecture makes the question sharp. Submission::commands is a std::span into GUEST
// memory, not a copy, so the guest may not reuse its command buffer until the GPU thread is done
// reading it - which is what Done() -> WaitForIdle() enforces, once per frame (m_done_num is the
// frame counter the whole profiler uses). Guest and GPU thread therefore alternate instead of
// pipelining. What that costs depends entirely on the split this prints.
namespace {

// Windowed, exclusive handler cost. NOP's custom subcommands get separate buckets.
// Draws drained inside a handler are already charged to DrawPhase::Total.
struct Pm4OpcodeProfile {
	struct Bucket {
		uint64_t cycles = 0;
		uint64_t calls = 0;
	};
	std::array<Bucket, 256 + Pm4::R_NUM> buckets {};
	std::array<Bucket, 4> work {};
	uint64_t packets = 0;
	uint64_t start_draws = 0;
	uint64_t start_tsc = 0;
	std::chrono::steady_clock::time_point start_wall {};

	void Begin() {
		if (start_tsc == 0) {
			start_wall = std::chrono::steady_clock::now();
			start_tsc = DrawProfileReadCycles();
			start_draws = g_draw_profile.draws;
		}
	}

	void Add(uint32_t opcode, uint32_t header, uint64_t cycles) {
		const auto key = opcode == Pm4::IT_NOP ? 256 + KYTY_PM4_R(header) : opcode;
		buckets[key].cycles += cycles;
		buckets[key].calls++;
		if (++packets < 1000000) {
			return;
		}
		const auto elapsed_tsc = DrawProfileReadCycles() - start_tsc;
		const double elapsed_ns = std::chrono::duration<double, std::nano>(
		    std::chrono::steady_clock::now() - start_wall).count();
		const double ns_per_cycle = elapsed_tsc == 0 ? 0.0 : elapsed_ns / elapsed_tsc;
		const auto draws = g_draw_profile.draws - start_draws;
		uint64_t total_cycles = 0;
		for (const auto& bucket : buckets) {
			total_cycles += bucket.cycles;
		}
		std::printf("Pm4Opcodes: %llu packets, %llu draws, %.2f s window, %.2f ms exclusive handlers\n",
		            static_cast<unsigned long long>(packets), static_cast<unsigned long long>(draws),
		            elapsed_ns / 1e9, total_cycles * ns_per_cycle / 1e6);
		for (uint32_t key = 0; key < buckets.size(); ++key) {
			const auto& b = buckets[key];
			if (b.calls == 0) {
				continue;
			}
			std::printf("  %s=0x%02x calls=%llu us/packet=%.3f us/draw=%.3f share=%.1f%%\n",
			            key >= 256 ? "NOP.R" : "opcode", key >= 256 ? key - 256 : key,
			            static_cast<unsigned long long>(b.calls), b.cycles * ns_per_cycle / b.calls / 1e3,
			            draws == 0 ? 0.0 : b.cycles * ns_per_cycle / draws / 1e3,
			            total_cycles == 0 ? 0.0 : 100.0 * b.cycles / total_cycles);
		}
		std::fflush(stdout);
		constexpr const char* names[] = {"drain-minus-draws", "barrier-record", "scheduler-flush", "label-write"};
		for (size_t i = 0; i < work.size(); ++i) {
			const auto& b = work[i];
			std::printf("Pm4Work: %s calls=%llu ms=%.3f us/draw=%.3f\n", names[i],
			            static_cast<unsigned long long>(b.calls), b.cycles * ns_per_cycle / 1e6,
			            draws == 0 ? 0.0 : b.cycles * ns_per_cycle / draws / 1e3);
		}
		std::fflush(stdout);
		*this = {};
	}
};

thread_local Pm4OpcodeProfile g_pm4_opcode_profile;

struct Pm4WorkTimer {
	uint32_t kind;
	bool active = g_draw_profile.active;
	uint64_t draws_before = g_draw_profile.cycles[static_cast<size_t>(DrawPhase::Total)];
	uint64_t start = active ? DrawProfileReadCycles() : 0;

	~Pm4WorkTimer() {
		if (active) {
			const auto elapsed = DrawProfileReadCycles() - start;
			const auto nested = g_draw_profile.cycles[static_cast<size_t>(DrawPhase::Total)] - draws_before;
			auto& bucket = g_pm4_opcode_profile.work[kind];
			bucket.cycles += elapsed - std::min(elapsed, nested);
			bucket.calls++;
		}
	}
};

template <typename T>
void WriteProfiledLabel(T* destination, const T& value) {
	Pm4WorkTimer timer {3};
	std::memcpy(destination, &value, sizeof(value));
}

std::atomic<uint64_t> g_guest_blocked_ns {0};   // guest thread inside WaitForIdle
std::atomic<uint64_t> g_gpu_busy_ns {0};        // GPU thread inside Process()
std::atomic<uint64_t> g_census_frames {0};
// When every queued submission is blocked - DE/CE interleaving that cannot progress - the GPU
// thread sleeps 100us and retries. That is a poll, not a wait on an event, so if it fires often it
// is pure dead time on the one thread that does all the work.
// Which IT_EVENT_WRITE types actually fire, and how many reach EmitGlobalBarrier. 51933 packets
// over 158816 draws at 1.734 us each; the type histogram says whether the partial flushes are the
// bulk of it or whether the cache writeback/invalidate events are.
std::array<std::atomic<uint64_t>, 64> g_event_write_types {};
std::atomic<uint64_t>                 g_event_write_total {0};

void NoteEventWrite(uint32_t event_type) {
	if (event_type < g_event_write_types.size()) {
		g_event_write_types[event_type].fetch_add(1, std::memory_order_relaxed);
	}
	const auto n = g_event_write_total.fetch_add(1, std::memory_order_relaxed) + 1;
	if (n % 200000 != 0) {
		return;
	}
	std::printf("EventWrite: total=%llu", static_cast<unsigned long long>(n));
	for (uint32_t i = 0; i < g_event_write_types.size(); i++) {
		const auto c = g_event_write_types[i].load(std::memory_order_relaxed);
		if (c > n / 100) {
			const char* name = i == 0x07   ? "CsPartialFlush"
			                   : i == 0x0f ? "GsPartialFlush"
			                   : i == 0x10 ? "PsPartialFlush"
			                   : i == 0x16 ? "CbDbWritebackInv"
			                   : i == 0x31 ? "CbDataWritebackInv"
			                   : i == 0x2a ? "DbDataWritebackInv"
			                   : i == 0x2c ? "DbMetaWritebackInv"
			                   : i == 0x2e ? "CbMetaWritebackInv"
			                               : "other";
			std::printf(" | 0x%02x %s=%llu (%.1f%%)", i, name,
			            static_cast<unsigned long long>(c),
			            100.0 * static_cast<double>(c) / static_cast<double>(n));
		}
	}
	std::printf("\n");
	std::fflush(stdout);
}

std::atomic<uint64_t> g_blocked_polls {0};
std::atomic<uint64_t> g_blocked_poll_ns {0};
std::chrono::steady_clock::time_point g_census_start {};

void ReportFrameThreadCensus(int frame) {
	if (frame % 300 != 0) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (g_census_start.time_since_epoch().count() == 0) {
		g_census_start = now;
		return;
	}
	const auto wall_ns =
	    static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - g_census_start)
	                            .count());
	const auto frames  = static_cast<double>(g_census_frames.exchange(0, std::memory_order_relaxed));
	const auto blocked = static_cast<double>(g_guest_blocked_ns.exchange(0, std::memory_order_relaxed));
	const auto busy    = static_cast<double>(g_gpu_busy_ns.exchange(0, std::memory_order_relaxed));
	if (frames > 0.0 && wall_ns > 0.0) {
		std::printf("FrameThreads: %.1f ms/frame | guest blocked %.1f ms (%.0f%%) | gpu busy %.1f ms "
		            "(%.0f%%) | overlap headroom %.1f ms\n",
		            wall_ns / frames / 1e6, blocked / frames / 1e6, 100.0 * blocked / wall_ns,
		            busy / frames / 1e6, 100.0 * busy / wall_ns,
		            (wall_ns - busy) / frames / 1e6);
		const auto polls    = g_blocked_polls.exchange(0, std::memory_order_relaxed);
		const auto poll_ns  = g_blocked_poll_ns.exchange(0, std::memory_order_relaxed);
		std::printf("  blocked-queue polls: %.1f/frame costing %.2f ms/frame (%.0f%% of wall)\n",
		            static_cast<double>(polls) / frames,
		            static_cast<double>(poll_ns) / frames / 1e6,
		            100.0 * static_cast<double>(poll_ns) / wall_ns);
		std::fflush(stdout);
	}
	g_census_start = now;
}

} // namespace


static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static thread_local bool              g_gpu_mutex_owned   = false;
static thread_local bool              g_gpu_thread        = false;
static thread_local GuestGpu*         g_gpu_state         = nullptr;

class GpuMutexLock final {
public:
	explicit GpuMutexLock(Common::Mutex& mutex): m_mutex(mutex) {
		if (g_gpu_mutex_owned) {
			EXIT("recursive GPU mutex acquisition\n");
		}
		g_gpu_mutex_owned = true;
		m_mutex.Lock();
	}
	~GpuMutexLock() {
		if (!g_gpu_mutex_owned) {
			EXIT("invalid GPU mutex release\n");
		}
		m_mutex.Unlock();
		g_gpu_mutex_owned = false;
	}

private:
	Common::Mutex& m_mutex;
};

static bool GraphicsRunDebugDumpEnabled() {
	return Config::GraphicsDebugDumpEnabled() &&
	       Config::GetPrintfDirection() != Config::OutputDirection::Silent;
}

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	GraphicsInitJmpTables();
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_thread = std::jthread(ThreadRun, this);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::ProcessCommands() {
	EXIT_IF(!IsGpuThread());
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		command();
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}

void GuestGpu::Submit(std::span<const uint32_t> draw_commands,
                      std::span<const uint32_t> constant_commands) {
	if (draw_commands.empty()) {
		return;
	}
	GpuMutexLock lock(m_submission_mutex);
	Submission   submission;
	submission.type              = SubmissionType::Graphics;
	submission.queue_id          = 0;
	submission.commands          = draw_commands;
	submission.constant_commands = constant_commands;
	submission.reset_processor   = m_graphics_done;
	m_graphics_done              = false;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());
	GpuMutexLock lock(m_submission_mutex);

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Submission submission;
	submission.type     = SubmissionType::Compute;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	GpuMutexLock lock(m_submission_mutex);
	Submission   submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.reset_processor = m_graphics_done;
	submission.flip_request_id = request_id;
	m_graphics_done            = false;
	Enqueue(std::move(submission));
}

void GuestGpu::Done() {
	GpuMutexLock lock(m_submission_mutex);
	if (!IsGpuThread()) {
		if (Config::FramePipeliningEnabled()) {
			// The drain existed only to protect the guest's command buffer, which Enqueue now
			// copies. What remains is backpressure: let the guest run ahead, but bounded.
			WaitForPipelineDepth(Config::PipelineDepth());
		} else {
			WaitForIdle();
		}
	}
	m_graphics_done = true;
	const auto frame = ++m_done_num;
	g_census_frames.fetch_add(1, std::memory_order_relaxed);
	ReportFrameThreadCensus(frame);
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	EXIT_IF(queue_id >= QueueCount);
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
	}
	return *processor;
}

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_saved_ctx.Reset();
	m_context_state_pushed             = false;
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_user_data_marker                 = HW::UserSgprType::Unknown;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;

	std::memset(m_const_ram, 0, sizeof(m_const_ram));
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_context_state_pushed);
			m_ctx                  = m_saved_ctx;
			m_saved_ctx            = {};
			m_context_state_pushed = false;
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			m_ctx.Reset();
			break;
		default: EXIT("unknown context state operation: %u\n", static_cast<uint32_t>(operation));
	}
}

void CommandProcessor::BufferInit() {
	GetScheduler().Begin(m_ctx, m_ucfg, m_sh_ctx);
}

void CommandProcessor::BufferFlush() {
	DrainQueuedDraws();
	Pm4WorkTimer timer {2};
	GetScheduler().Flush();
}

void CommandProcessor::BufferFlushAndWait() {
	DrainQueuedDraws();
	GetScheduler().FlushAndWait();
}

// Queued draws must be translated before the command buffer they belong to is submitted, or the
// frame ships without them. Called from the command processor, never from the renderer's flush
// path: translation re-enters buffer and texture resolution, and that path already holds the
// memory tracker's region lock.
void CommandProcessor::DrainQueuedDraws() {
	if (!Config::DrawQueueEnabled() || !GetScheduler().Active()) {
		return;
	}
	Pm4WorkTimer timer {0};
	m_renderer.GetRenderExecutor().DrainDrawQueue(CurrentBuffer());
}

void CommandProcessor::BufferWait() {
	BufferInit();
	GetScheduler().Finish();
}

void CommandProcessor::ResetDeCe() {
	m_de_count    = 0;
	m_ce_count    = 0;
	m_ce_complete = false;
}

void CommandProcessor::WaitCe() {
	if (m_ce_count <= m_de_count && !m_ce_complete) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitDeDiff(uint32_t diff) {
	EXIT_IF(m_de_count > m_ce_count);
	if (m_ce_count - m_de_count >= diff) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4();
	}
}

void CommandProcessor::IncrementDe() {
	m_de_count++;
}

void CommandProcessor::IncrementCe() {
	m_ce_count++;
}

void CommandProcessor::WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num) {
	memcpy(m_const_ram + offset / 4, src, static_cast<size_t>(dw_num) * 4);
}

void CommandProcessor::DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num) {
	memcpy(dst, m_const_ram + offset / 4, static_cast<size_t>(dw_num) * 4);
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll,
                                  uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	(void)poll;
	if (!TestWaitRegMemValue(*addr, ref, mask, func)) {
		SuspendPm4();
	}
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t, uint32_t);

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}

	if (write_one_address) {
		for (uint32_t i = 0; i < dw_num; i++) {
			dst[0] = src[i];
		}
	} else {
		memcpy(dst, src, static_cast<size_t>(dw_num) * sizeof(uint32_t));
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto value = Sync::ReadReferenceClock();
	std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
	LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64 " size=%u\n",
	     dst_address, value, num_bytes);
}

void CommandProcessor::DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
                               uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint8_t  src_cache_policy,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
                               uint8_t wait_for_previous, uint8_t write_confirm,
                               uint8_t block_engine) {
	EXIT_NOT_IMPLEMENTED(engine > 1);
	if (num_bytes == 0) {
		return;
	}
	EXIT_NOT_IMPLEMENTED(dst_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(src_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(wait_for_previous > 1);
	EXIT_NOT_IMPLEMENTED(write_confirm > 1);
	EXIT_NOT_IMPLEMENTED(block_engine > 1);
	if (static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	auto& buffer_cache = GetGpuResources().GetBufferCache();
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}
	bool src_gds = false;
	if (!decode_gds(src_sel, src_gds)) {
		EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
	}
	if (src_gds && dst_gds) {
		EXIT("unsupported dmaData GDS-to-GDS copy\n");
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
}

void GuestGpu::Enqueue(Submission submission) {
	EXIT_IF(submission.queue_id >= QueueCount);
	if (Config::FramePipeliningEnabled()) {
		// Take ownership of the PM4 stream so the guest may reuse its command buffer the moment
		// this returns. ~640 KB a frame at typical draw counts, against the per-frame drain it
		// replaces. Done once here because every submission path funnels through Enqueue.
		if (!submission.commands.empty()) {
			auto owned = std::make_shared<const std::vector<uint32_t>>(submission.commands.begin(),
			                                                          submission.commands.end());
			submission.commands       = *owned;
			submission.owned_commands = std::move(owned);
		}
		if (!submission.constant_commands.empty()) {
			auto owned = std::make_shared<const std::vector<uint32_t>>(
			    submission.constant_commands.begin(), submission.constant_commands.end());
			submission.constant_commands       = *owned;
			submission.owned_constant_commands = std::move(owned);
		}
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_available.Signal();
}

void GuestGpu::WaitForPipelineDepth(uint32_t max_pending) {
	const auto        start = std::chrono::steady_clock::now();
	Common::LockGuard lock(m_queue_mutex);
	while (m_submission_count > max_pending) {
		m_idle.Wait(&m_queue_mutex);
	}
	g_guest_blocked_ns.fetch_add(
	    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                              std::chrono::steady_clock::now() - start)
	                              .count()),
	    std::memory_order_relaxed);
}

void GuestGpu::WaitForIdle() {
	const auto        start = std::chrono::steady_clock::now();
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
	g_guest_blocked_ns.fetch_add(
	    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                              std::chrono::steady_clock::now() - start)
	                              .count()),
	    std::memory_order_relaxed);
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	g_gpu_thread = true;
	g_gpu_state  = gpu;

	for (;;) {
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         has_submission = false;
		bool                         should_stop    = false;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
			}
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
			} else {
				int selected_queue = -1;
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (!gpu->m_queues[id].empty() && !gpu->m_queues[id].front().blocked) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0) {
					gpu->m_processing = false;
					const auto poll_start = std::chrono::steady_clock::now();
					gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					g_blocked_polls.fetch_add(1, std::memory_order_relaxed);
					g_blocked_poll_ns.fetch_add(
					    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
					                              std::chrono::steady_clock::now() - poll_start)
					                              .count()),
					    std::memory_order_relaxed);
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				}
				auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
				submission  = std::move(queue.front());
				queue.pop_front();
				gpu->m_submission_count--;
				gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
				gpu->m_processing = true;
				has_submission    = true;
			}
		}
		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			return;
		}

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			command();

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		EXIT_IF(!has_submission);
		const auto process_start = std::chrono::steady_clock::now();
		const bool complete      = gpu->Process(submission);
		g_gpu_busy_ns.fetch_add(
		    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                              std::chrono::steady_clock::now() - process_start)
		                              .count()),
		    std::memory_order_relaxed);

		Common::LockGuard lock(gpu->m_queue_mutex);
		if (!complete) {
			submission.blocked = true;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
		gpu->m_processing = false;
		// Signalled unconditionally: WaitForIdle waits on full idle and WaitForPipelineDepth on a
		// queue bound, and each re-checks its own predicate, so an extra wake is free but a
		// missing one hangs the guest.
		gpu->m_idle.SignalAll();
	}
}

bool GuestGpu::Process(Submission& submission) {
	const bool first_slice = !submission.started;
	if (first_slice && RenderDocCaptureRequested()) {
		Common::LockGuard render_lock(m_renderer.GetMutex());
		RenderDocStartCapture();
	}
	auto& cp = GetProcessor(submission.queue_id);

	if (first_slice && submission.reset_processor) {
		cp.Reset();
	}

	if (first_slice) {
		submission.started = true;
		cp.SetSubmitId(++m_submit_id);
		cp.ResetDeCe();
		cp.SetFlip({});
	}

	cp.BufferInit();
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Graphics: {
			bool progressed = false;
			submission.constant_complete |= submission.constant_commands.empty();
			for (;;) {
				bool round_progress = false;
				if (!submission.constant_complete) {
					submission.constant_complete =
					    cp.Process(submission.constant_execution, submission.constant_commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.constant_execution.MadeProgress();
				}
				cp.SetCeComplete(submission.constant_complete);
				if (!submission.command_complete) {
					submission.command_complete =
					    cp.Process(submission.command_execution, submission.commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.command_execution.MadeProgress();
				}
				progressed |= round_progress;
				complete = submission.command_complete && submission.constant_complete;
				if (complete || !round_progress) {
					break;
				}
			}
			if (progressed) {
				if (complete) {
					m_renderer.GetGpuResources().RunGarbageCollector();
				}
				cp.BufferFlush();
			} else if (complete) {
				m_renderer.GetGpuResources().RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::Compute: {
			const auto      num_dw = static_cast<uint32_t>(submission.commands.size());
			const auto*     buffer = submission.commands.data();
			static uint32_t compute_batch_log_count = 0;
			if (first_slice && num_dw <= 128 && compute_batch_log_count++ < 32) {
				LOGF("compute direct batch: data=0x%016" PRIx64 ", num_dw=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(buffer), num_dw);
				for (uint32_t i = 0; i < std::min<uint32_t>(num_dw, 16); i++) {
					LOGF("\t compute[%02" PRIu32 "] = 0x%08" PRIx32 "\n", i, buffer[i]);
				}
			}
			if (first_slice) {
				GraphicsDbgDumpDcb("cc", num_dw, buffer);
			}
			complete = cp.Process(submission.command_execution, submission.commands) ==
			           Pm4ProcessResult::Complete;
			if (submission.command_execution.MadeProgress()) {
				if (complete) {
					m_renderer.GetGpuResources().RunGarbageCollector();
				}
				cp.BufferFlush();
			} else if (complete) {
				m_renderer.GetGpuResources().RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::FlipPreparation:
			m_renderer.GetGpuResources().RunGarbageCollector();
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
	}

	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;

	struct ExecutionScope {
		ExecutionScope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~ExecutionScope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}

		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} execution_scope(*this, execution);

	ProcessPm4(execution, 0);
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands) {
	EXIT_IF(g_current_execution == nullptr);
	if (commands.empty()) {
		return;
	}
	auto&      execution  = *g_current_execution;
	const auto stop_depth = execution.m_buffer_stack.size();
	execution.m_buffer_stack.push_back({commands});
	ProcessPm4(execution, stop_depth);
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspended = true;
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution, size_t stop_depth) {
	while (execution.m_buffer_stack.size() > stop_depth) {
		if (g_gpu_state != nullptr) {
			g_gpu_state->ProcessCommands();
		}
		const auto buffer_index = execution.m_buffer_stack.size() - 1;
		auto&      cursor       = execution.m_buffer_stack[buffer_index];
		EXIT_IF(cursor.offset_dw > cursor.commands.size());
		if (cursor.deferred_advance_dw != 0) {
			EXIT_IF(cursor.deferred_advance_dw > cursor.commands.size() - cursor.offset_dw);
			cursor.offset_dw += cursor.deferred_advance_dw;
			cursor.deferred_advance_dw = 0;
			execution.m_made_progress  = true;
			continue;
		}
		if (cursor.offset_dw == cursor.commands.size()) {
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];
		const auto        opcode        = (packet_header >> 8u) & 0xffu;
		EXIT_NOT_IMPLEMENTED(remaining_dw > total_dw);

		if (packet_header == 0x80000000u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		// How much is predication actually culling? If eligible packets are many but skips
		// are ~0, the guest is trying to cull and the emulator draws anyway - which shows up
		// directly as inflated draws/frame.
		if ((packet_header & 1u) != 0) {
			static std::atomic<uint64_t> pred_eligible {0};
			static std::atomic<uint64_t> pred_skipped {0};
			const auto seen = pred_eligible.fetch_add(1, std::memory_order_relaxed) + 1;
			if (ShouldSkipPredicatedPackets()) {
				pred_skipped.fetch_add(1, std::memory_order_relaxed);
			}
			if (seen % 20000 == 0) {
				const auto sk = pred_skipped.load(std::memory_order_relaxed);
				std::printf("PredicationHealth: eligible=%llu skipped=%llu (%.2f%%)\n",
					            static_cast<unsigned long long>(seen),
					            static_cast<unsigned long long>(sk),
					            100.0 * static_cast<double>(sk) / static_cast<double>(seen));
			}
		}
		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.fetch_add(1) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			if (opcode == Pm4::IT_NOP && KYTY_PM4_R(packet_header) == Pm4::R_RELEASE_MEM &&
			    packet_dw >= 7) {
				static std::atomic<uint32_t> log_count {0};
				if (log_count.fetch_add(1) < 128) {
					const auto dst = packet[3] | (static_cast<uint64_t>(packet[4]) << 32u);
					const auto val = packet[5] | (static_cast<uint64_t>(packet[6]) << 32u);
					LOGF("\t predicated skip: R_RELEASE_MEM dst=0x%016" PRIx64
					     ", value=0x%016" PRIx64 ", action=0x%08" PRIx32
					     ", gcr/data/int=0x%08" PRIx32 "\n",
					     dst, val, packet[1], packet[2]);
				}
			}
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		auto handler = g_cp_op_func[opcode];

		if (handler == nullptr) {
			const auto offset = total_dw - remaining_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n",
			     total_dw - remaining_dw, packet_header);
		}

		// ~3 us/draw of frame time sits outside DrawIndex and has never been profiled. Draw and
		// dispatch packets are timed by their own phases, so only the rest is charged here:
		// register writes, waits, sync, everything between one draw and the next.
		// IT_INDIRECT_BUFFER recurses into another command buffer, so its timer would span every
		// packet inside it while those packets are timed too. That nesting is what made this phase
		// report 670 us/draw. Excluded along with draws, which have their own phases.
		const bool packet_is_draw =
		    opcode == Pm4::IT_INDIRECT_BUFFER || opcode == Pm4::IT_INDIRECT_BUFFER_CNST ||
		    opcode == Pm4::IT_DRAW_INDEX_2 || opcode == Pm4::IT_DRAW_INDEX_OFFSET_2 ||
		    opcode == Pm4::IT_DRAW_INDEX_AUTO || opcode == Pm4::IT_DRAW_INDIRECT ||
		    opcode == Pm4::IT_DRAW_INDEX_INDIRECT || opcode == Pm4::IT_DRAW_INDIRECT_MULTI ||
		    opcode == Pm4::IT_DRAW_INDEX_INDIRECT_MULTI || opcode == Pm4::IT_DISPATCH_DIRECT ||
		    opcode == Pm4::IT_DISPATCH_INDIRECT;
		// Every path that writes the guest register file. A worker resolving a draw must not read
		// registers that a later packet has already moved, so a draw's state is snapshotted
		// whenever one of these has landed since the previous snapshot.
		const bool packet_writes_registers =
		    opcode == Pm4::IT_SET_CONTEXT_REG || opcode == Pm4::IT_SET_SH_REG ||
		    opcode == Pm4::IT_SET_UCONFIG_REG || opcode == Pm4::IT_SET_UCONFIG_REG_INDEX ||
		    opcode == Pm4::IT_SET_CONTEXT_REG_INDIRECT || opcode == Pm4::IT_SET_SH_REG_INDIRECT ||
		    opcode == Pm4::IT_SET_UCONFIG_REG_INDIRECT || opcode == Pm4::IT_CLEAR_STATE;

		uint32_t packet_dw = 0;
		if (packet_is_draw) {
			// Built here, before the draw runs, because this is the last point at which the
			// registers still describe THIS draw. Behind the profile flag until workers consume
			// it, so it costs nothing by default.
			if (g_draw_profile.active && GetScheduler().Active()) {
				DrawPhaseTimer snapshot_timer(DrawPhase::Snapshot);
				auto&          buffer = CurrentBuffer();
				if (!buffer.IsInvalid()) {
					const auto before = m_snapshot_cache.SnapshotsBuilt();
					m_snapshot_cache.Acquire(buffer.GetRegisters(), buffer.GetUserConfig(),
					                         buffer.GetShaders());
					g_draw_profile.snapshots_built += m_snapshot_cache.SnapshotsBuilt() - before;
				}
			}
			packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		} else if (g_draw_profile.active) {
			g_pm4_opcode_profile.Begin();
			const auto total_before =
			    g_draw_profile.cycles[static_cast<size_t>(DrawPhase::Total)];
			auto& non_draw = g_draw_profile.cycles[static_cast<size_t>(DrawPhase::Pm4NonDraw)];
			const auto non_draw_before = non_draw;
			const auto start = DrawProfileReadCycles();
			packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
			const auto elapsed = DrawProfileReadCycles() - start;
			// Subtract both nested draws and recursively measured non-draw handlers.
			const auto nested =
			    g_draw_profile.cycles[static_cast<size_t>(DrawPhase::Total)] - total_before +
			    non_draw - non_draw_before;
			const auto exclusive = elapsed - std::min(nested, elapsed);
			non_draw += exclusive;
			g_draw_profile.pm4_packets++;
			g_pm4_opcode_profile.Add(opcode, packet_header, exclusive);
		} else {
			packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		}
		if (packet_writes_registers) {
			m_snapshot_cache.MarkRegistersDirty();
		}
		EXIT_IF(packet_dw > remaining_dw);
		if (execution.m_suspended) {
			if (execution.m_buffer_stack.size() > buffer_index + 1) {
				execution.m_buffer_stack[buffer_index].deferred_advance_dw = packet_dw;
			}
			return;
		}
		EXIT_IF(execution.m_buffer_stack.size() != buffer_index + 1);
		execution.m_buffer_stack[buffer_index].offset_dw += packet_dw;
		execution.m_made_progress = true;
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances = num_instances;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address, uint32_t count_in_dwords) {
	if (wait_op != 0) {
		BufferFlushAndWait();
	}

	(void)count_in_dwords;

	switch (op) {
		case 0x00: {
			m_predicate_skip = false;
		} break;
		case 0x03: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);

			auto value = *reinterpret_cast<const volatile uint64_t*>(address);

			switch (condition) {
				case 0x00: m_predicate_skip = (value != 0); break;
				case 0x01: m_predicate_skip = (value == 0); break;
				default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
			}
			static std::atomic<uint64_t> pred_calls {0};
			static std::atomic<uint64_t> pred_skip_true {0};
			const auto calls = pred_calls.fetch_add(1, std::memory_order_relaxed) + 1;
			if (m_predicate_skip) {
				pred_skip_true.fetch_add(1, std::memory_order_relaxed);
			}
			if (calls % 2000 == 0) {
				// The occlusion path always sets bit 63 as a "ready" flag. If every value here has
				// bit 63 set, this comparison is decided by that flag and not by the sample count.
				const auto st = pred_skip_true.load(std::memory_order_relaxed);
				std::printf("PredicationSet: calls=%llu skip_true=%llu (%.1f%%) value=0x%016llx"
					            " condition=%u\n",
					            static_cast<unsigned long long>(calls),
					            static_cast<unsigned long long>(st),
					            100.0 * static_cast<double>(st) / static_cast<double>(calls),
					            static_cast<unsigned long long>(value), condition);
			}
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1) < 128) {
				LOGF("\t bool predication: addr=0x%016" PRIx64 ", value=0x%016" PRIx64
				     ", condition=%" PRIu32 ", skip=%u, wait_op=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(address), value, condition,
				     m_predicate_skip ? 1u : 0u, wait_op);
			}
		} break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
}

void CommandProcessor::DrawIndex(uint32_t index_count, const void* index_addr, uint32_t flags,
                                 uint32_t type, uint32_t instance_count, const void* object_ids,
                                 uint32_t render_target_slice_offset, int32_t vertex_offset_add,
                                 uint32_t first_instance) {
	CheckBuffer();

	if (instance_count == 0) {
		instance_count = m_num_instances;
	}
	if (object_ids != nullptr) {
		LOGF("\t draw indexed multi-instanced objectIds = 0x%016" PRIx64 "\n",
		     reinterpret_cast<uint64_t>(object_ids));
	}
	if (render_target_slice_offset != 0) {
		LOGF("\t draw render target slice offset = %" PRIu32 "\n", render_target_slice_offset);
	}
	if (vertex_offset_add != 0 || first_instance != 0) {
		LOGF("\t draw indexed offsets: vertex_offset_add = %" PRId32 ", first_instance = %" PRIu32
		     "\n",
		     vertex_offset_add, first_instance);
	}
	auto& executor = m_renderer.GetRenderExecutor();
	if (Config::DrawQueueEnabled()) {
		// Queued against the register state as it stands now. The packet loop keeps writing those
		// registers, so a draw translated later must carry its own copy or it would be translated
		// against a later draw's state.
		QueuedDraw queued {};
		queued.snapshot = m_snapshot_cache.Acquire(CurrentBuffer().GetRegisters(),
		                                           CurrentBuffer().GetUserConfig(),
		                                           CurrentBuffer().GetShaders());
		queued.submit_id                  = m_submit_id;
		queued.index_type_and_size        = m_index_type_and_size;
		queued.index_count                = index_count;
		queued.index_addr                 = index_addr;
		queued.flags                      = flags;
		queued.type                       = type;
		queued.instance_count             = instance_count;
		queued.render_target_slice_offset = render_target_slice_offset;
		queued.vertex_offset_add          = vertex_offset_add;
		queued.first_instance             = first_instance;
		if (executor.EnqueueDrawIndex(std::move(queued))) {
			// Capacity is a boundary of its own: it bounds the queue and keeps the GPU fed rather
			// than letting the CPU accumulate a whole frame before anything is recorded.
			executor.DrainDrawQueue(CurrentBuffer());
		}
		return;
	}
	executor.DrawIndex(m_submit_id, CurrentBuffer(), m_index_type_and_size, index_count,
	                   index_addr, flags, type, instance_count, render_target_slice_offset,
	                   vertex_offset_add, first_instance);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count,
                                       uint32_t flags) {
	CheckBuffer();

	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(index_offset) * index_size);

	m_renderer.GetRenderExecutor().DrawIndex(m_submit_id, CurrentBuffer(), m_index_type_and_size,
	                                         index_count, index_addr, flags, 1, m_num_instances);
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, uint32_t draw_initiator, bool indexed) {
	struct DrawIndirectArgs {
		uint32_t vertex_count_per_instance;
		uint32_t instance_count;
		uint32_t start_vertex_location;
		uint32_t start_instance_location;
	};
	struct DrawIndexedIndirectArgs {
		uint32_t index_count_per_instance;
		uint32_t instance_count;
		uint32_t start_index_location;
		uint32_t base_vertex_location;
		uint32_t start_instance_location;
	};

	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != 2u);
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	const auto* args_addr =
	    reinterpret_cast<const void*>(m_draw_indirect_args_base_addr + data_offset);

	if (!indexed) {
		DrawIndirectArgs args {};
		std::memcpy(&args, args_addr, sizeof(args));
		if (args.instance_count != 1u || args.start_vertex_location != 0u ||
		    args.start_instance_location != 0u) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1) < 64) {
				LOGF("\t warning: partial DrawIndirect args: vertex_count=%" PRIu32
				     ", instance_count=%" PRIu32 ", start_vertex=%" PRIu32
				     ", start_instance=%" PRIu32 "\n",
				     args.vertex_count_per_instance, args.instance_count,
				     args.start_vertex_location, args.start_instance_location);
			}
		}
		m_num_instances = args.instance_count;
		SubmitNonIndexedDraw(args.vertex_count_per_instance, 0, 0, args.start_vertex_location,
		                     args.start_instance_location);
		return;
	}

	DrawIndexedIndirectArgs args {};
	std::memcpy(&args, args_addr, sizeof(args));
	if (args.base_vertex_location != 0u || args.start_instance_location != 0u) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 64) {
			LOGF("\t warning: partial DrawIndexIndirect args: index_count=%" PRIu32
			     ", instance_count=%" PRIu32 ", start_index=%" PRIu32 ", base_vertex=%" PRIu32
			     ", start_instance=%" PRIu32 "\n",
			     args.index_count_per_instance, args.instance_count, args.start_index_location,
			     args.base_vertex_location, args.start_instance_location);
		}
	}

	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(args.start_index_location) * index_size);

	const uint32_t index_count =
	    (m_index_buffer_size != 0 ? std::min(args.index_count_per_instance, m_index_buffer_size)
	                              : args.index_count_per_instance);
	if (GraphicsRunDebugDumpEnabled() && index_count != args.index_count_per_instance) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
			LOGF("\t DrawIndexIndirect: clamped index_count from %" PRIu32 " to %" PRIu32
			     " using INDEX_BUFFER_SIZE\n",
			     args.index_count_per_instance, index_count);
		}
	}

	m_num_instances = args.instance_count;
	DrawIndex(index_count, index_addr, 0, 1, args.instance_count, nullptr, 0,
	          static_cast<int32_t>(args.base_vertex_location), args.start_instance_location);
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, uint32_t draw_initiator,
                                         bool indexed) {
	struct DrawIndirectArgs {
		uint32_t vertex_count_per_instance;
		uint32_t instance_count;
		uint32_t start_vertex_location;
		uint32_t start_instance_location;
	};
	struct DrawIndexedIndirectArgs {
		uint32_t index_count_per_instance;
		uint32_t instance_count;
		uint32_t start_index_location;
		uint32_t base_vertex_location;
		uint32_t start_instance_location;
	};

	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != 2u);
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	uint32_t draw_count = max_count_or_count;
	if (count_addr != nullptr) {
		draw_count = *count_addr;
		if (draw_count > max_count_or_count) {
			draw_count = max_count_or_count;
		}
	}

	// Census: this early return sits UPSTREAM of RenderExecutor::DrawIndex, so a batch dropped
	// here is invisible to the draw census. GPU-driven batches take their count from memory a
	// compute pass writes; if that never reaches the CPU the whole batch silently disappears.
	{
		static std::atomic<uint64_t> multi_total {0};
		static std::atomic<uint64_t> multi_zero {0};
		static std::atomic<uint64_t> multi_draws {0};
		const auto total = multi_total.fetch_add(1, std::memory_order_relaxed) + 1;
		if (draw_count == 0) {
			multi_zero.fetch_add(1, std::memory_order_relaxed);
		} else {
			multi_draws.fetch_add(draw_count, std::memory_order_relaxed);
		}
		if (total == 1 || total % 500 == 0) {
			std::printf("IndirectMultiCensus: calls=%" PRIu64 " zero_count=%" PRIu64 " draws_issued=%" PRIu64
			     " from_count_addr=%d\n",
			     total, multi_zero.load(std::memory_order_relaxed),
			     multi_draws.load(std::memory_order_relaxed),
			     static_cast<int>(count_addr != nullptr));
		}
	}

	if (draw_count == 0) {
		return;
	}

	const auto args_size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	EXIT_NOT_IMPLEMENTED(stride_in_bytes < args_size);

	for (uint32_t i = 0; i < draw_count; i++) {
		const auto args_addr = m_draw_indirect_args_base_addr + data_offset +
		                       static_cast<uint64_t>(i) * stride_in_bytes;

		if (!indexed) {
			auto* args = reinterpret_cast<const DrawIndirectArgs*>(args_addr);
			if (args->instance_count != 1u || args->start_vertex_location != 0u ||
			    args->start_instance_location != 0u) {
				static std::atomic<uint32_t> log_count {0};
				if (log_count.fetch_add(1) < 64) {
					LOGF("\t warning: partial DrawIndirectMulti args[%u]: vertex_count=%" PRIu32
					     ", instance_count=%" PRIu32 ", start_vertex=%" PRIu32
					     ", start_instance=%" PRIu32 "\n",
					     i, args->vertex_count_per_instance, args->instance_count,
					     args->start_vertex_location, args->start_instance_location);
				}
			}
			m_num_instances = args->instance_count;
			SubmitNonIndexedDraw(args->vertex_count_per_instance, 0, 0, args->start_vertex_location,
			                     args->start_instance_location);
			continue;
		}

		auto* args = reinterpret_cast<const DrawIndexedIndirectArgs*>(args_addr);
		if (args->base_vertex_location != 0u || args->start_instance_location != 0u) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1) < 64) {
				LOGF("\t warning: partial DrawIndexIndirectMulti args[%u]: index_count=%" PRIu32
				     ", instance_count=%" PRIu32 ", start_index=%" PRIu32 ", base_vertex=%" PRIu32
				     ", start_instance=%" PRIu32 "\n",
				     i, args->index_count_per_instance, args->instance_count,
				     args->start_index_location, args->base_vertex_location,
				     args->start_instance_location);
			}
		}

		uint64_t index_size = 0;
		switch (m_index_type_and_size) {
			case 0: index_size = 2; break;
			case 1: index_size = 4; break;
			case 2: index_size = 1; break;
			default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
		}

		auto* index_addr = reinterpret_cast<const void*>(
		    m_index_base_addr + static_cast<uint64_t>(args->start_index_location) * index_size);

		const uint32_t index_count =
		    (m_index_buffer_size != 0
		         ? std::min(args->index_count_per_instance, m_index_buffer_size)
		         : args->index_count_per_instance);
		if (GraphicsRunDebugDumpEnabled() && index_count != args->index_count_per_instance) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("\t DrawIndexIndirectMulti: clamped index_count from %" PRIu32 " to %" PRIu32
				     " using INDEX_BUFFER_SIZE\n",
				     args->index_count_per_instance, index_count);
			}
		}

		m_num_instances = args->instance_count;
		DrawIndex(index_count, index_addr, 0, 1, args->instance_count, nullptr, 0,
		          static_cast<int32_t>(args->base_vertex_location), args->start_instance_location);
	}
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));

	uint32_t frame_num = 0;
	// uint32_t local_x   = 1;
	// uint32_t local_y   = 1;
	// uint32_t local_z   = 1;

	{
		CheckBuffer();
		frame_num = m_renderer.GetGpu().GetFrameNum();
		if (GraphicsRunDebugDumpEnabled()) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 1024) {
				const auto& cs = m_sh_ctx.GetCs().cs_regs;
				const auto& oa = m_ucfg.GetGdsOaCounter(m_ucfg.GetGdsOaState().GetIndex());
				LOGF("QueuePoint DispatchDirect: frame=%u submit=%" PRIu64
				     " groups=%ux%ux%u local=%ux%ux%u mode=0x%08" PRIx32 " wave=%u cs=0x%016" PRIx64
				     " oa_index=%u oa_enabled=%s oa_addr=0x%04" PRIx32 " oa_space=0x%08" PRIx32
				     "\n",
				     frame_num, m_submit_id, thread_group_x, thread_group_y, thread_group_z,
				     std::max(cs.num_thread_x, 1u), std::max(cs.num_thread_y, 1u),
				     std::max(cs.num_thread_z, 1u), mode, static_cast<uint32_t>(cs.wave_size),
				     cs.data_addr, m_ucfg.GetGdsOaState().GetIndex(),
				     oa.IsCounterEnabled() ? "true" : "false", oa.GetAddressBytes(),
				     oa.GetSpaceAvailable());
			}
		}

		const auto& cs = m_sh_ctx.GetCs().cs_regs;
		// local_x        = std::max(cs.num_thread_x, 1u);
		// local_y        = std::max(cs.num_thread_y, 1u);
		// local_z        = std::max(cs.num_thread_z, 1u);
		if (cs.wave_size == 64u) {
			static std::atomic_bool logged_wave64_shader {false};
			if (!logged_wave64_shader.exchange(true, std::memory_order_relaxed)) {
				LOGF("warning: executing wave64 compute shader cs=0x%016" PRIx64 "\n",
				     cs.data_addr);
				std::printf("warning: executing wave64 compute shader cs=0x%016" PRIx64 "\n",
				            cs.data_addr);
				std::fflush(stdout);
			}
		}

		m_renderer.GetRenderExecutor().DispatchDirect(m_submit_id, CurrentBuffer(), thread_group_x,
		                                              thread_group_y, thread_group_z, mode);
	}

	/*constexpr uint32_t DispatchInitiatorUseThreadDimensions = 1u << 5u;
	auto               group_count = [](uint32_t threads, uint32_t group_size) {
	    return (threads == 0
	                ? 0u
	                : (threads + std::max(group_size, 1u) - 1u) / std::max(group_size, 1u));
	};

	auto groups_x = thread_group_x;
	auto groups_y = thread_group_y;
	auto groups_z = thread_group_z;
	if ((mode & DispatchInitiatorUseThreadDimensions) != 0) {
	    groups_x = group_count(thread_group_x, local_x);
	    groups_y = group_count(thread_group_y, local_y);
	    groups_z = group_count(thread_group_z, local_z);
	}

	const uint64_t invocations =
	    static_cast<uint64_t>(groups_x) * groups_y * groups_z * local_x * local_y * local_z;
	if (invocations != 0) {
	    BufferFlushAndWait();
	}*/
}

void CommandProcessor::DispatchIndirect(uint32_t data_offset, uint32_t mode) {
	struct DispatchIndirectArgs {
		uint32_t thread_group_x;
		uint32_t thread_group_y;
		uint32_t thread_group_z;
	};

	EXIT_NOT_IMPLEMENTED(m_dispatch_indirect_args_base_addr == 0);

	const auto args_addr = m_dispatch_indirect_args_base_addr + data_offset;
	auto*      args      = reinterpret_cast<const DispatchIndirectArgs*>(args_addr);

	DispatchDirect(args->thread_group_x, args->thread_group_y, args->thread_group_z, mode);
}

void CommandProcessor::DrawIndexAuto(uint32_t index_count, uint32_t flags,
                                     uint32_t render_target_slice_offset) {
	SubmitNonIndexedDraw(index_count, flags, render_target_slice_offset, 0, 0);
}

void CommandProcessor::SubmitNonIndexedDraw(uint32_t vertex_count, uint32_t flags,
                                            uint32_t render_target_slice_offset,
                                            uint32_t first_vertex, uint32_t first_instance) {
	CheckBuffer();

	m_renderer.GetRenderExecutor().DrawAuto(m_submit_id, CurrentBuffer(), vertex_count, flags,
	                                        render_target_slice_offset, m_num_instances,
	                                        first_vertex, first_instance);
}

void CommandProcessor::WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index) {
	BufferFlush();

	m_renderer.GetVideoOut().WaitFlipDone(static_cast<int>(video_out_handle),
	                                      static_cast<int>(display_buffer_index));
}

template <typename T>
void CommandProcessor::WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest,
                                        uint32_t eop_event_type, uint32_t cache_action,
                                        uint32_t event_index, uint32_t event_write_source,
                                        void* dst_gpu_addr, T value, uint32_t interrupt_selector,
                                        uint32_t interrupt_context_id) {
	static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));

	CheckBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		const auto bits      = static_cast<unsigned>(sizeof(T) * 8u);
		const auto log_width = static_cast<int>(sizeof(T) * 2u);

		LOGF("CommandProcessor::WriteAtEndOfPipe%u()\n"
		     "\t cache_policy        = 0x%08" PRIx32 "\n"
		     "\t event_write_dest    = 0x%08" PRIx32 "\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t event_index         = 0x%08" PRIx32 "\n"
		     "\t event_write_source  = 0x%08" PRIx32 "\n"
		     "\t interrupt_selector  = 0x%08" PRIx32 "\n"
		     "\t interrupt_context   = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%0*" PRIx64 "\n",
		     bits, cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
		     event_write_source, interrupt_selector, interrupt_context_id,
		     reinterpret_cast<uint64_t>(dst_gpu_addr), log_width, static_cast<uint64_t>(value));
	}

	EXIT_NOT_IMPLEMENTED(cache_policy != 0x00000000);
	EXIT_NOT_IMPLEMENTED(event_write_dest != 0x00000000);

	bool with_interrupt = false;
	switch (interrupt_selector) {
		case 0x00:
		case 0x03: with_interrupt = false; break;
		case 0x01:
			if (!IsAsyncComputeQueue()) {
				Sync::TriggerEopEventAtEndOfPipe(CurrentBuffer(), m_interrupt_event_id,
				                                 interrupt_context_id);
				return;
			}
			with_interrupt = true;
			break;
		case 0x02: with_interrupt = true; break;
		default: EXIT("unknown interrupt_selector\n");
	}

	auto write32 = [&](bool with_writeback) {
		auto* dst  = static_cast<uint32_t*>(dst_gpu_addr);
		auto  data = static_cast<uint32_t>(value);
		WriteProfiledLabel(dst, data);

		if (with_interrupt) {
			if (with_writeback) {
				Sync::WriteAtEndOfPipeWithInterruptWriteBack32(m_submit_id, CurrentBuffer(), dst,
				                                               data, m_interrupt_event_id,
				                                               interrupt_context_id);
			} else {
				Sync::WriteAtEndOfPipeWithInterrupt32(m_submit_id, CurrentBuffer(), dst, data,
				                                      m_interrupt_event_id, interrupt_context_id);
			}
		} else if (with_writeback) {
			Sync::WriteAtEndOfPipeWithWriteBack32(m_submit_id, CurrentBuffer(), dst, data);
		} else {
			Sync::WriteAtEndOfPipe32(m_submit_id, CurrentBuffer(), dst, data);
		}
	};

	switch (event_write_source) {
		case 0x01:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (eop_event_type == 0x2f && cache_action == 0x00 && event_index == 0x06) {
					auto* dst = static_cast<uint32_t*>(dst_gpu_addr);
					SynchronizeGpu();
					Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(), dst, value & 0xffffu,
					              value >> 16u);
					Sync::WriteAtEndOfPipeGds32(m_submit_id, CurrentBuffer(), dst, value & 0xffffu,
					                            value >> 16u);
					if (with_interrupt) {
						m_renderer.TriggerInterrupt(m_interrupt_event_id, interrupt_context_id);
					}
					return;
				}
			} else if (eop_event_type == 0x04 && cache_action == 0x00 && event_index == 0x05) {
				write32(false);
				return;
			}
			break;
		case 0x02:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (eop_event_type == 0x2f && event_index == 0x06) {
					switch (cache_action) {
						case 0x00: write32(false); return;
						case 0x38: write32(true); return;
						default: break;
					}
				}
			} else {
				auto write64 = [&](bool with_writeback) {
					auto* dst = static_cast<uint64_t*>(dst_gpu_addr);
					WriteProfiledLabel(dst, value);

					if (with_interrupt) {
						if (with_writeback) {
							Sync::WriteAtEndOfPipeWithInterruptWriteBack64(
							    m_submit_id, CurrentBuffer(), dst, value, m_interrupt_event_id,
							    interrupt_context_id);
						} else {
							Sync::WriteAtEndOfPipeWithInterrupt64(m_submit_id, CurrentBuffer(), dst,
							                                      value, m_interrupt_event_id,
							                                      interrupt_context_id);
						}
					} else if (with_writeback) {
						Sync::WriteAtEndOfPipeWithWriteBack64(m_submit_id, CurrentBuffer(), dst,
						                                      value);
					} else {
						Sync::WriteAtEndOfPipe64(m_submit_id, CurrentBuffer(), dst, value);
					}
				};

				switch (cache_action) {
					case 0x00:
						switch (eop_event_type) {
							case 0x04:
								if (event_index == 0x05) {
									write64(false);
									return;
								}
								break;
							case 0x14:
							case 0x28:
								if (event_index == 0x00) {
									write64(false);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
							case 0x2f:
							case 0x30:
								if (event_index == 0x00 && !with_interrupt) {
									write64(false);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x38:
						switch (eop_event_type) {
							case 0x04:
							case 0x14:
							case 0x28:
								if (((eop_event_type == 0x04 || eop_event_type == 0x28) &&
								     event_index == 0x05 && !with_interrupt) ||
								    (event_index == 0x00)) {
									write64(true);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
								if (event_index == 0x00 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							case 0x2f:
								if (event_index == 0x06 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x3b:
						if (eop_event_type == 0x04 && event_index == 0x05 && with_interrupt) {
							write64(true);
							return;
						}
						break;
					default: break;
				}
			}
			break;
		case 0x04:
			if constexpr (sizeof(T) == sizeof(uint64_t)) {
				const auto clock = Sync::ReadReferenceClock();
				auto*      dst   = static_cast<uint64_t*>(dst_gpu_addr);
				WriteProfiledLabel(dst, clock);
				switch (cache_action) {
					case 0x00:
						if ((eop_event_type == 0x04 && event_index == 0x05) ||
						    (eop_event_type == 0x28 && event_index == 0x00)) {
							if (with_interrupt) {
								Sync::WriteAtEndOfPipeWithInterrupt64(
								    m_submit_id, CurrentBuffer(), dst, clock, m_interrupt_event_id,
								    interrupt_context_id);
							} else {
								Sync::WriteAtEndOfPipeClockCounter(m_submit_id, CurrentBuffer(),
								                                   dst, clock);
							}
							return;
						}
						break;
					case 0x38:
						if ((eop_event_type == 0x04 &&
						     (event_index == 0x00 || event_index == 0x05)) ||
						    (eop_event_type == 0x28 && event_index == 0x00)) {
							if (with_interrupt) {
								Sync::WriteAtEndOfPipeWithInterruptWriteBack64(
								    m_submit_id, CurrentBuffer(), dst, clock, m_interrupt_event_id,
								    interrupt_context_id);
							} else {
								Sync::WriteAtEndOfPipeClockCounterWithWriteBack(
								    m_submit_id, CurrentBuffer(), dst, clock);
							}
							return;
						}
						break;
					default: break;
				}
			}
			break;
		default: break;
	}

	EXIT("unknown event type\n");
}

void CommandProcessor::WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint32_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint64_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

// The barrier the guest asked for, without tearing the render pass down. Vulkan allows a memory
// barrier inside a dynamic-rendering pass as long as it does not cross attachment scope, which a
// shader-completion wait does not.
void CommandProcessor::EmitInPassBarrier() {
	CheckBuffer();
	Common::LockGuard lock(m_renderer.GetMutex());
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	// The queued draws still have to be recorded before the barrier they are being synchronised
	// against - that part is not optional. Only the EndRendering is skipped.
	if (Config::DrawQueueEnabled()) {
		m_renderer.GetRenderExecutor().DrainDrawQueue(CurrentBuffer());
	}
	CurrentBuffer().Handle().pipelineBarrier2(dependency);
}

void CommandProcessor::EmitGlobalBarrier() {
	CheckBuffer();

	Common::LockGuard lock(m_renderer.GetMutex());

	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;

	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	// Command-processor level, so no cache lock is held: safe to translate queued draws here.
	// They must be recorded before this barrier, which is what the guest is asking to synchronise
	// against.
	if (Config::DrawQueueEnabled()) {
		Pm4WorkTimer timer {0};
		m_renderer.GetRenderExecutor().DrainDrawQueue(CurrentBuffer());
	}
	Pm4WorkTimer timer {1};
	GetScheduler().EndRendering();
	CurrentBuffer().Handle().pipelineBarrier2(dependency);
}

void CommandProcessor::TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id) {
	CheckBuffer();

	Sync::TriggerEopEventAtEndOfPipe(CurrentBuffer(), m_interrupt_event_id, interrupt_context_id);
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	NoteEventWrite(event_type);
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		//
		// Measured: IT_EVENT_WRITE fires 51933 times across 158816 draws - once every ~3 draws -
		// at 1.734 us each, and EmitGlobalBarrier is what makes it expensive. It drains the batched
		// draw queue, ends the render pass and emits an AllCommands->AllCommands barrier, so the
		// next draw has to reopen the pass. That is what destroys draw batching.
		//
		// On GCN these three are waits for shader completion, not global cache invalidations. The
		// light path keeps the barrier - the guest did ask for a dependency - but leaves the render
		// pass open, which is what the batching actually needs.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010:
			if (Config::LightPartialFlushEnabled()) {
				EmitInPassBarrier();
			} else {
				EmitGlobalBarrier();
			}
			break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			EmitGlobalBarrier();
			break;
		// DbDataWritebackInvalidate, DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		//
		// CbMetadataWritebackInvalidate alone is 42.8% of every IT_EVENT_WRITE, and these three
		// together are 59%. They write back GCN's colour/depth METADATA caches - DCC and CMask -
		// which a Vulkan driver owns and manages itself; there is no guest-visible metadata cache
		// on this side to flush. The memory barrier still goes out, because the guest is expressing
		// a real dependency, but ending the render pass for it is not justified and is what breaks
		// draw batching.
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			if (Config::LightPartialFlushEnabled()) {
				EmitInPassBarrier();
			} else {
				EmitGlobalBarrier();
			}
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			if (Config::RealOcclusionQueries()) {
				if (!m_occlusion.Enabled() && GetScheduler().Active()) {
					m_occlusion.Initialize(m_renderer.GetGraphics(), GetScheduler());
				}
				if (m_occlusion.Dump(event_address)) {
					break;
				}
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are currently treated as "
				            "always visible; GPU usage may be higher and FPS may be lower.\n");
			});

			// Until host occlusion queries are implemented, publish an always-visible result. The
			// PS5 layout contains one interleaved begin/end pair per DB, and bit 63 marks a result
			// ready. The guest issues this event twice at addresses one qword apart - once for the
			// begin counters and once for the end counters - so each dump writes ONE value per DB at a
			// 16-byte stride and the counter advances between them. Writing both halves here would
			// clobber the end slot during the begin pass (see CheckPm4SyntheticOcclusionCounterDump).
			//
			// The advance must be a PLAUSIBLE sample count, not 1. The guest compares the returned
			// count against how much of the screen the queried object covers; a single sample reads as
			// ~100% occluded, so the game culls the object from rendering while still simulating it -
			// producing geometry you can collide with but cannot see. This is roughly a quarter of a
			// 4K frame per DB, comfortably "fully visible" without approaching the 63-bit counter range.
			constexpr uint64_t ready_bit       = 1ull << 63u;
			constexpr uint64_t counter_mask    = ready_bit - 1u;
			// Must EXCEED the largest coverage any object can have, or the guest compares our fixed
			// answer against a bigger expected screen area, concludes the object is mostly occluded
			// and culls it - which shows up as geometry vanishing at particular camera angles. A
			// full 2560x1440 frame is ~3.7M samples, so stay comfortably above that.
			constexpr uint64_t visible_samples = 0x00800000ull;
			auto*              results         = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value           = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			m_synthetic_occlusion_counter =
			    (m_synthetic_occlusion_counter + visible_samples) & counter_mask;
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::Flip() {
	CheckBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}

	auto& command = CurrentBuffer();
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeOnlyFlip(m_submit_id, command, m_flip.handle, m_flip.index,
	                               m_flip.flip_mode, m_flip.flip_arg, request);
	GetScheduler().Flush();
}

void CommandProcessor::Flip(void* dst_gpu_addr, uint32_t value) {
	CheckBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n"
		     "\t dst_gpu_addr = 0x%016" PRIx64 "\n"
		     "\t value        = 0x%08" PRIx32 "\n",
		     reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	std::memcpy(dst_gpu_addr, &value, sizeof(value));
	auto& command = CurrentBuffer();
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeWithFlip32(m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr),
	                                 value, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                 m_flip.flip_arg, request);
	GetScheduler().Flush();
}

void CommandProcessor::FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action,
                                         void* dst_gpu_addr, uint32_t value) {
	CheckBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::FlipWithInterrupt()\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%08" PRIx32 "\n",
		     eop_event_type, cache_action, reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	if (eop_event_type != 0x00000004 || cache_action != 0x00000038) {
		EXIT("unknown event type\n");
	}
	std::memcpy(dst_gpu_addr, &value, sizeof(value));
	auto& command = CurrentBuffer();
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeWithInterruptWriteBackFlip32(
	    m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr), value, m_flip.handle,
	    m_flip.index, m_flip.flip_mode, m_flip.flip_arg, request, m_interrupt_event_id);
	GetScheduler().Flush();
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	CheckBuffer();
	if (g_current_processor != nullptr) {
		EXIT("invalid graphics-thread CPU flip preparation\n");
	}
	struct ProcessorScope {
		explicit ProcessorScope(CommandProcessor& processor) { g_current_processor = &processor; }
		~ProcessorScope() { g_current_processor = nullptr; }
	};
	ProcessorScope processor_scope(*this);

	m_renderer.GetVideoOut().PrepareFlip(request_id, CurrentBuffer());
	GetScheduler().Flush();
	m_renderer.GetVideoOut().CompleteFlip(request_id);
}

void CommandProcessor::SynchronizeGpu() {
	GetScheduler().Finish();
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_thread;
}

} // namespace Libs::Graphics
