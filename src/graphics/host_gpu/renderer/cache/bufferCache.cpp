#include "graphics/host_gpu/renderer/drawWorkerContext.h"
#include "graphics/host_gpu/renderer/drawProfile.h"
#include "graphics/host_gpu/renderer/secondaryBatch.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

// Conditional locks: no-ops until workers are actually running, so the single-threaded path pays
// nothing. Written as guards rather than if-statements at each site so an early return cannot
// leave the lock held.
class MaybeSharedLock {
public:
	MaybeSharedLock(std::shared_mutex& lock, bool engaged): m_lock(lock), m_engaged(engaged) {
		if (m_engaged) {
			m_lock.lock_shared();
		}
	}
	~MaybeSharedLock() {
		if (m_engaged) {
			m_lock.unlock_shared();
		}
	}
	KYTY_CLASS_NO_COPY(MaybeSharedLock);

private:
	std::shared_mutex& m_lock;
	bool               m_engaged;
};

class MaybeUniqueLock {
public:
	MaybeUniqueLock(std::shared_mutex& lock, bool engaged): m_lock(lock), m_engaged(engaged) {
		if (m_engaged) {
			m_lock.lock();
		}
	}
	~MaybeUniqueLock() {
		if (m_engaged) {
			m_lock.unlock();
		}
	}
	KYTY_CLASS_NO_COPY(MaybeUniqueLock);

private:
	std::shared_mutex& m_lock;
	bool               m_engaged;
};

} // namespace


namespace {

// Does the tracker fast path pay for itself? It costs two region traversals per buffer - measured
// 0.670 us/draw over ~11 queries - to decide whether a small read-only buffer can be served from
// the stream ring instead of the normal path. If it rarely fires, the fix is not to make the
// queries cheaper but to stop making them.
std::atomic<uint64_t> g_tracker_gate_reached {0};
std::atomic<uint64_t> g_tracker_gate_taken {0};

void ReportTrackerGate() {
	const auto reached = g_tracker_gate_reached.load(std::memory_order_relaxed);
	if (reached % 500000 != 0) {
		return;
	}
	const auto taken = g_tracker_gate_taken.load(std::memory_order_relaxed);
	std::printf("TrackerGate: reached=%llu taken=%llu (%.1f%%)\n",
	            static_cast<unsigned long long>(reached), static_cast<unsigned long long>(taken),
	            reached > 0 ? 100.0 * static_cast<double>(taken) / static_cast<double>(reached)
	                        : 0.0);
	std::fflush(stdout);
}

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		buffer.CopyFrom(m_scheduler.Current(), m_staging_buffer, offset, buffer.Offset(address),
		                chunk, vk::AccessFlagBits::eHostWrite);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

struct BufferCache::DownloadCopy {
	Buffer*  buffer        = nullptr;
	uint64_t source_offset = 0;
	uint64_t address       = 0;
	uint64_t size          = 0;
};

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;   // caller holds the unique lock
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, pages.first * sizeof(vk::DeviceAddress),
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(pages.first * sizeof(vk::DeviceAddress),
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (buffer.is_deleted) {
		return;
	}
	// Recorded per worker and merged at EndBatch: this runs on every buffer resolution, so a
	// shared LRU here would serialise the workers on what is otherwise a read-only path.
	if (m_concurrent && m_batch_depth != 0) {
		const auto worker = CurrentDrawWorker();
		if (worker < m_deferred_touch.size()) {
			m_deferred_touch[worker].push_back(buffer.lru_id);
			return;
		}
	}
	m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
}

void BufferCache::DeleteBuffer(BufferId id) {
	auto* buffer = m_slot_buffers.try_get(id);
	if (buffer == nullptr || buffer->is_deleted) {
		return;
	}
	Unregister(id);
	// Unregister already marks the buffer deleted and takes it out of the lookup structures, so
	// nothing new can find it. Only the destruction is held back.
	if (m_batch_depth != 0) {
		m_retired_in_batch.push_back(id);
		return;
	}
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

void BufferCache::RecordUpload(vk::Buffer destination, uint64_t destination_size,
                               vk::Buffer source, std::span<const vk::BufferCopy> copies) {
	// Recording, and it ends the render pass to do it. A worker reaching here would record into
	// the primary and close a pass another thread is using, which corrupts silently rather than
	// failing - so it is an assert, not a branch. Workers stage instead; see MustStageForWorker.
	EXIT_IF(MustStageForWorker());
	auto&      command = m_scheduler.Current();
	// Transfers cannot sit inside a render pass.
	command.EndRendering();
	const auto native = command.Handle();

	vk::BufferMemoryBarrier before {};
	before.sType         = vk::StructureType::eBufferMemoryBarrier;
	before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
	                       vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = destination;
	before.offset              = 0;
	before.size                = destination_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 1, &before, 0, nullptr);
	native.copyBuffer(source, destination, static_cast<uint32_t>(copies.size()), copies.data());
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
}

void BufferCache::FlushPendingUploads() {
	if (m_pending_uploads.empty()) {
		return;
	}
	// Each RecordUpload closes the pass itself; the batch opens its own afterwards.
	for (const auto& upload: m_pending_uploads) {
		RecordUpload(upload.destination, upload.destination_size, upload.source, upload.copies);
	}
	m_pending_uploads.clear();
}

void BufferCache::CreateWorkerStreamBuffers(uint32_t worker_count) {
	const auto slots = std::min(worker_count, kMaxDrawWorkers);
	const auto extra = slots > 0 ? slots - 1 : 0;
	if (m_worker_stream_buffers.size() >= extra) {
		return;
	}
	// 16 MiB rather than the main ring's 64: a worker ring serves its share of a frame's draws,
	// not all of them, and running one dry costs a fallback to the normal buffer path rather than
	// anything worse.
	m_deferred_touch.resize(kMaxDrawWorkers);
	constexpr uint64_t kWorkerStreamSize = 16 * MiB;
	m_worker_stream_buffers.reserve(extra);
	while (m_worker_stream_buffers.size() < extra) {
		m_worker_stream_buffers.push_back(std::make_unique<StreamBuffer>(
		    m_graphics, m_scheduler, MemoryUsage::Stream, kWorkerStreamSize));
	}
}

void BufferCache::EndBatch() {
	EXIT_IF(m_batch_depth == 0);
	if (--m_batch_depth != 0) {
		return;
	}
	// Deferred rather than erased directly: the GPU may still be reading these, which is the same
	// reason DeleteBuffer defers outside a batch.
	for (auto& touches: m_deferred_touch) {
		for (const auto lru_id: touches) {
			m_lru_cache.Touch(lru_id, m_gc_tick);
		}
		touches.clear();
	}
	for (const auto id: m_retired_in_batch) {
		if (m_scheduler.Active()) {
			m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
		} else {
			m_slot_buffers.erase(id);
		}
	}
	m_retired_in_batch.clear();
}

std::pair<uint64_t, uint64_t> BufferCache::DownloadEnvelope(const DownloadCopy& copy) {
	if (copy.buffer == nullptr || copy.size == 0 || copy.source_offset > copy.buffer->Size() ||
	    copy.size > copy.buffer->Size() - copy.source_offset) {
		EXIT("BufferCache: invalid download copy\n");
	}
	const auto begin = copy.source_offset & ~uint64_t {3};
	if (copy.source_offset > UINT64_MAX - copy.size ||
	    copy.source_offset + copy.size > UINT64_MAX - 3) {
		EXIT("BufferCache: download copy alignment overflow\n");
	}
	const auto end = (copy.source_offset + copy.size + 3) & ~uint64_t {3};
	if (end > copy.buffer->Size()) {
		EXIT("BufferCache: aligned download copy exceeds its owner\n");
	}
	return {begin, end - begin};
}

void BufferCache::DownloadBufferMemory(std::span<const DownloadCopy> copies) {
	// Reading back what an upload was supposed to have written requires that upload to exist.
	FlushPendingUploads();
	KYTY_PROFILER_FUNCTION();
	std::vector<DownloadCopy> batch;
	batch.reserve(copies.size());
	uint64_t                  packed_size = 0;
	auto&                     download    = m_download_buffer;
	const auto flush = [&] {
		const auto [mapped, base_offset] = download.Map(packed_size, DOWNLOAD_ALIGNMENT);
		EXIT_IF(mapped == nullptr);
		uint64_t cursor = 0;
		for (const auto& copy: batch) {
			const auto [source_begin, envelope_size] = DownloadEnvelope(copy);
			download.CopyFrom(m_scheduler.Current(), *copy.buffer, source_begin, base_offset + cursor,
			                  envelope_size, vk::AccessFlagBits::eMemoryWrite, vk::AccessFlags {},
			                  vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
			                  vk::AccessFlagBits::eHostRead);
			cursor += AlignDownload(envelope_size);
		}
		download.Commit();
		const auto completion_tick = m_scheduler.CurrentTick();
		m_scheduler.Finish();
		m_scheduler.WaitPriorityOperations(completion_tick);
		cursor = 0;
		for (const auto& copy: batch) {
			const auto [source_begin, envelope_size] = DownloadEnvelope(copy);
			const auto offset = cursor + copy.source_offset - source_begin;
			download.Invalidate(base_offset + offset, copy.size);
			Libs::LibKernel::Memory::WriteBacking(copy.address, mapped + offset, copy.size);
			cursor += AlignDownload(envelope_size);
		}
		batch.clear();
		packed_size = 0;
	};
	for (auto copy: copies) {
		while (copy.size != 0) {
			const auto available = download.Size() - packed_size;
			const auto prefix    = copy.source_offset & 3u;
			const auto bytes     = std::min(copy.size, available - prefix);
			DownloadCopy part {copy.buffer, copy.source_offset, copy.address, bytes};
			const auto [source_begin, envelope_size] = DownloadEnvelope(part);
			(void)source_begin;
			packed_size += AlignDownload(envelope_size);
			batch.push_back(part);
			copy.source_offset += bytes;
			copy.address += bytes;
			copy.size -= bytes;
			if (packed_size == download.Size()) {
				flush();
			}
		}
	}
	if (!batch.empty()) {
		flush();
	}
	for (const auto& copy: copies) {
		m_gpu_modified_ranges.Subtract(copy.address, copy.size);
	}
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
	: m_graphics(graphics), m_scheduler(scheduler),
	  m_fault_manager(graphics, scheduler, *this, CACHING_PAGEBITS, CACHING_NUMPAGES),
	  m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
	  m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
	                         BDA_PAGETABLE_SIZE),
	  m_memory_tracker(page_manager),
	  m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
	  m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
	  m_download_buffer(graphics, scheduler, MemoryUsage::Download, 32 * MiB),
	  m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
	  m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0 || size == 0 || vaddr >= TRACKER_ADDRESS_SIZE ||
	    size > TRACKER_ADDRESS_SIZE - vaddr) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	KYTY_PROFILER_FUNCTION();
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	m_scheduler.Context().GetGpu().SendCommandSync(
	    [this, vaddr, size, is_write] { ReadMemoryOnGpu(vaddr, size, is_write); });
}

void BufferCache::ReadMemoryOnGpu(uint64_t vaddr, uint64_t size, bool is_write) {
	FlushPendingUploads();
	// CPU invalidation reaches this point only for a GPU-owned tracker page. Resolve the exact
	// Buffer owner on the GPU thread so the cache index remains single-thread-owned.
	if (is_write && !IsRegionRegistered(vaddr, size)) {
		return;
	}
	std::vector<DownloadCopy> copies;
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "memory invalidation");
	    },
		[&](uint64_t address, uint64_t bytes) noexcept {
		    for (const auto range: m_gpu_modified_ranges.Intersections(address, bytes)) {
			    for (uint64_t copied = 0; copied < range.size;) {
				    const auto copy_address = range.address + copied;
				    const auto* owner = m_page_table.Find(copy_address >> PageTable::kPageBits);
				    if (owner == nullptr || !*owner) {
					    EXIT("BufferCache: invalidation readback has no buffer owner\n");
				    }
				    auto& buffer = m_slot_buffers[*owner];
				    if (!buffer.IsInBounds(copy_address, 1)) {
					    EXIT("BufferCache: invalidation readback is outside its buffer owner\n");
				    }
				    const auto copy_size = std::min(
				        range.size - copied, buffer.CpuAddress() + buffer.Size() - copy_address);
				    copies.push_back(
				        {&buffer, buffer.Offset(copy_address), copy_address, copy_size});
				    copied += copy_size;
			    }
		    }
	    });
	if (copies.empty()) {
		if (!is_write) {
			return;
		}
		// A preceding read fault can consume the last GPU-owned copy after this write invalidation
		// has already chosen to flush. Complete the CPU ownership handoff even though this callback
		// no longer has bytes to download.
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		return;
	}
	DownloadBufferMemory(copies);
	// The enumeration above covered whole dirty pages and every exact interval on them.
	m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (size == 0 || vaddr >= TRACKER_ADDRESS_SIZE || size > TRACKER_ADDRESS_SIZE - vaddr) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	{
		MaybeSharedLock lock(m_page_table_lock, m_concurrent);
		const auto*     owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
		if (owner != nullptr && *owner) {
			auto& buffer = m_slot_buffers[*owner];
			if (buffer.IsInBounds(vaddr, size)) {
				return *owner;
			}
		}
	}
	// Creating mutates the page table and the address index, so it takes the unique lock inside.
	return CreateBuffer(vaddr, size);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	MaybeUniqueLock lock(m_page_table_lock, m_concurrent);
	// NOT assertion-only: the overlap loop below does buffer.CopyFrom(command, ...), recording
	// into the primary. A worker reaching here would record into whatever command buffer happens
	// to be current, corrupting silently rather than failing - so this is an assert, and
	// ObtainBuffer bails out before it can get here.
	EXIT_IF(MustStageForWorker());
	auto& command = m_scheduler.Current();
	EXIT_IF(command.IsInvalid());
	auto       begin = vaddr & ~(CACHING_PAGESIZE - 1);
	auto       end   = (vaddr + size + CACHING_PAGESIZE - 1) & ~(CACHING_PAGESIZE - 1);
	auto       first = m_buffers.lower_bound(begin);
	if (first != m_buffers.begin()) {
		const auto previous = std::prev(first);
		if (const auto& buffer = m_slot_buffers[previous->second];
		    buffer.CpuAddress() + buffer.Size() > begin) {
			first = previous;
		}
	}
	auto last = first;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer = m_slot_buffers[last->second];
		begin              = std::min(begin, buffer.CpuAddress());
		end                = std::max(end, buffer.CpuAddress() + buffer.Size());
	}

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, end - begin);
	auto&      buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", begin, end - begin);
	for (auto overlap = first; overlap != last;) {
		const auto current = overlap++;
		const auto old_id  = current->second;
		const auto& old    = m_slot_buffers[old_id];
		buffer.CopyFrom(command, old, 0, old.CpuAddress() - begin, old.Size());
		DeleteBuffer(old_id);
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
	if (source && !Config::DeferUploadsEnabled() && !MustStageForWorker()) {
		// Immediate path, unchanged and still the default. Deferral is correct only if every
		// consumer of an upload drains the list first, and enumerating those by hand missed one
		// (RenderExecutorColorVolumeDiscovery caught it), so it stays behind a flag until a run
		// can prove the list is complete.
		RecordUpload(buffer.Handle(), buffer.Size(), source, copies);
	} else if (source) {
		// Ordering guard. Uploads are replayed as a block before the batch's draws, so an upload
		// requested after draws are already in the batch would move ahead of them and those draws
		// would see data they must not. Replaying the batch first keeps guest order exact. This is
		// no worse than before: the EndRendering this replaces tore the batch unconditionally.
		if (SecondaryBatchOpen()) {
			FlushSecondaryBatch(m_scheduler.Context());
		}
		PendingBufferUpload upload {};
		upload.source           = source;
		upload.destination      = buffer.Handle();
		upload.destination_size = buffer.Size();
		upload.copies.assign(copies.begin(), copies.end());
		m_pending_uploads.push_back(std::move(upload));
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}

	auto [mapped, base_offset] = m_staging_buffer.Map(total_size, 4);
	if (mapped != nullptr) {
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			if (!Libs::LibKernel::Memory::TryReadGpuUploadMemory(address, mapped + copy.srcOffset,
			                                                     copy.size)) {
				EXIT("BufferCache: unreadable upload source addr=0x%016" PRIx64
				     " size=0x%016" PRIx64 "\n",
				     address, uint64_t(copy.size));
			}
			copy.srcOffset += base_offset;
		}
		m_staging_buffer.Commit();
		return m_staging_buffer.Handle();
	}

	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	for (const auto& copy: copies) {
		const auto address = buffer.CpuAddress() + copy.dstOffset;
		if (!Libs::LibKernel::Memory::TryReadGpuUploadMemory(
		        address, temporary->Mapped().data() + copy.srcOffset, copy.size)) {
			EXIT("BufferCache: unreadable temporary upload source addr=0x%016" PRIx64
			     " size=0x%016" PRIx64 "\n",
			     address, uint64_t(copy.size));
		}
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	// Address validity is a property of the request and always holds.
	if (vaddr == 0 || size == 0 || vaddr >= TRACKER_ADDRESS_SIZE ||
	    size > TRACKER_ADDRESS_SIZE - vaddr) {
		EXIT("BufferCache: invalid buffer request range\n");
	}
	// A recording command buffer is a precondition only for the caller that records inline.
	// SynchronizeBuffer already stages its upload when MustStageForWorker(), and the staged work
	// is replayed on the main thread at the batch flush, so a worker resolving between
	// submissions - where Current() is legitimately invalid - is not an error. Enforcing it there
	// is what aborted the first parallel-bindings run.
	if (!MustStageForWorker() && m_scheduler.Current().IsInvalid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	// A written buffer publishes into m_gpu_modified_ranges below, which is shared and has no
	// per-worker staging. Writable bindings are a small minority, so a worker abandons the draw
	// rather than the cache growing another deferral path for them.
	if (is_written && MustStageForWorker()) {
		RequestWorkerBailout();
		return {nullptr, 0};
	}

	bool tracker_fast_path = false;
	{
		// These query the region bitmaps and can take the region tracking lock, so they are timed
		// apart from the lookup and the sync they gate.
		DrawPhaseTimer tracker_timer(DrawPhase::ObtTracker);
		if (!is_written && size <= CACHING_PAGESIZE) {
			g_tracker_gate_reached.fetch_add(1, std::memory_order_relaxed);
			tracker_fast_path = !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
			                    m_memory_tracker.IsRegionCpuModified(vaddr, size);
			if (tracker_fast_path) {
				g_tracker_gate_taken.fetch_add(1, std::memory_order_relaxed);
			}
			ReportTrackerGate();
		}
	}
	if (tracker_fast_path) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		// Through the selector, not m_stream_buffer directly. A StreamBuffer is bump-allocated
		// from one cursor, so two workers mapping the shared ring are handed overlapping regions
		// and silently overwrite each other's uniform and vertex data - which is exactly the
		// stretched geometry and detached meshes the first attempt produced. Slot 0 resolves back
		// to m_stream_buffer, so the main thread's behaviour is unchanged.
		auto& stream          = GetUtilityBuffer(MemoryUsage::Stream);
		auto [mapped, offset] = stream.Map(size, alignment, false);
		if (mapped != nullptr && Libs::LibKernel::Memory::TryReadBacking(vaddr, mapped, size)) {
			stream.Commit();
			return {&stream, offset};
		}
	}

	auto* buffer = m_slot_buffers.try_get(id);
	{
		DrawPhaseTimer find_timer(DrawPhase::ObtFindBuffer);
		if (buffer == nullptr || buffer->is_deleted || !buffer->IsInBounds(vaddr, size)) {
			// FindBuffer can reach CreateBuffer, which records a CopyFrom into the primary. Bail
			// conservatively whenever the cached id misses, rather than trying to predict whether
			// this particular lookup would create one - a wrong prediction is silent corruption,
			// and the bailout census measured these well under 1% of draws.
			if (MustStageForWorker()) {
				RequestWorkerBailout();
				return {nullptr, 0};
			}
			id     = FindBuffer(vaddr, size);
			buffer = &m_slot_buffers[id];
		}
	}
	TouchBuffer(*buffer);
	{
		DrawPhaseTimer sync_timer(DrawPhase::ObtSynchronize);
		(void)SynchronizeBuffer(*buffer, vaddr, size, is_written, is_texel_buffer);
	}
	if (is_written) {
		m_gpu_modified_ranges.Add(vaddr, size);
	}
	return {buffer, buffer->Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0 || size == 0 || vaddr >= TRACKER_ADDRESS_SIZE ||
	    size > TRACKER_ADDRESS_SIZE - vaddr) {
		EXIT("BufferCache: invalid image source\n");
	}
	auto find_owner = [&]() -> Buffer* {
		const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
		if (owner == nullptr || !*owner) {
			return nullptr;
		}
		auto& buffer = m_slot_buffers[*owner];
		return buffer.IsInBounds(vaddr, size) ? &buffer : nullptr;
	};

	{
		const bool cpu_modified            = m_memory_tracker.IsRegionCpuModified(vaddr, size);
		const bool gpu_modified            = m_memory_tracker.IsRegionGpuModified(vaddr, size);
		const bool has_dirty_buffer_source = m_gpu_modified_ranges.Intersects(vaddr, size);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, vaddr, size,
		                                           "image source");

		auto* owner = find_owner();
		if (has_dirty_buffer_source && owner == nullptr) {
			if (!IsRegionRegistered(vaddr, size)) {
				EXIT("BufferCache: GPU-dirty image source has no native buffer\n");
			}
			owner = &m_slot_buffers[FindBuffer(vaddr, size)];
		}
		if (owner != nullptr && !cpu_modified && (!gpu_modified || has_dirty_buffer_source)) {
			TouchBuffer(*owner);
			return {owner, owner->Offset(vaddr)};
		}
		if (has_dirty_buffer_source && owner == nullptr) {
			EXIT("BufferCache: GPU-dirty image source could not resolve its native owner\n");
		}
	}

	auto [staging, stage_offset] = m_staging_buffer.Map(size, 16);
	if (staging == nullptr || (!Libs::LibKernel::Memory::TryReadBacking(vaddr, staging, size) &&
	                           !Libs::LibKernel::Memory::TryReadPrtBacking(vaddr, staging, size))) {
		EXIT("BufferCache: failed to read mapped guest image backing\n");
	}
	m_staging_buffer.Commit();

	const bool has_dirty_buffer_source = m_gpu_modified_ranges.Intersects(vaddr, size);
	auto*      owner                   = find_owner();
	if (has_dirty_buffer_source && owner == nullptr) {
		EXIT("BufferCache: GPU-dirty image source lost its native owner\n");
	}
	if (owner == nullptr ||
	    (m_memory_tracker.IsRegionGpuModified(vaddr, size) && !has_dirty_buffer_source)) {
		return {&m_staging_buffer, stage_offset};
	}

	TouchBuffer(*owner);
	std::vector<std::pair<uint64_t, uint64_t>> uploads;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, false,
	    [&](uint64_t address, uint64_t upload_size) noexcept {
		    uploads.emplace_back(address, upload_size);
	    },
	    [&]() noexcept {
		    for (const auto& [address, upload_size]: uploads) {
			    owner->CopyFrom(m_scheduler.Current(), m_staging_buffer,
			                    stage_offset + address - vaddr, owner->Offset(address), upload_size,
			                    vk::AccessFlagBits::eHostWrite);
		    }
	    });
	return {owner, owner->Offset(vaddr)};
}

void BufferCache::WriteHostMemory(uint64_t vaddr, std::span<const uint8_t> data) {
	if (vaddr == 0 || data.empty() || data.size() > UINT64_MAX - vaddr) {
		EXIT("BufferCache: invalid host DMA write\n");
	}
	Libs::LibKernel::Memory::WriteBacking(vaddr, data.data(), data.size());

	const auto end = vaddr + data.size();
	for (const auto& [address, id]: m_buffers) {
		auto&      buffer     = m_slot_buffers[id];
		const auto buffer_end = address + buffer.Size();
		const auto begin      = std::max(vaddr, address);
		const auto range_end  = std::min(end, buffer_end);
		if (begin >= range_end) {
			continue;
		}
		WriteDataBuffer(buffer, begin, data.data() + begin - vaddr, range_end - begin);
		TouchBuffer(buffer);
	}
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	{
		const auto region = m_texture_cache.QueryRegion(vaddr, size);
		if (!HasGpuDirtyBytes(vaddr, size) && !region.gpu_image_bytes) {
			if (region.image_bytes) {
				m_texture_cache.InvalidateMemory(vaddr, size);
			}
			std::array<uint32_t, 4096> values;
			values.fill(value);
			const std::span<const uint8_t> bytes {reinterpret_cast<const uint8_t*>(values.data()),
			                                      sizeof(values)};
			for (uint64_t offset = 0; offset < size;) {
				const auto chunk = std::min<uint64_t>(size - offset, bytes.size());
				WriteHostMemory(vaddr + offset, bytes.first(chunk));
				offset += chunk;
			}
			return;
		}
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	const auto id          = FindBuffer(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true, id);
	EXIT_IF(dst == nullptr);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory || dst_memory) {
		const auto src_region =
		    src_memory ? m_texture_cache.QueryRegion(src_vaddr, size) : TextureCache::RegionInfo {};
		const auto dst_region =
		    dst_memory ? m_texture_cache.QueryRegion(dst_vaddr, size) : TextureCache::RegionInfo {};
		if (src_memory && dst_memory && !HasGpuDirtyBytes(src_vaddr, size) &&
		    !HasGpuDirtyBytes(dst_vaddr, size) && !src_region.gpu_image_bytes &&
		    !dst_region.gpu_image_bytes) {
			if (dst_region.image_bytes) {
				m_texture_cache.InvalidateMemory(dst_vaddr, size);
			}
			std::array<uint8_t, 64 * 1024> bytes;
			for (uint64_t offset = 0; offset < size;) {
				const auto chunk = std::min<uint64_t>(size - offset, bytes.size());
				if (!Libs::LibKernel::Memory::TryReadBacking(src_vaddr + offset, bytes.data(),
				                                             chunk)) {
					EXIT("BufferCache: host DMA source has no direct backing\n");
				}
				WriteHostMemory(dst_vaddr + offset, std::span {bytes}.first(chunk));
				offset += chunk;
			}
			return;
		}
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	EXIT_IF(src == nullptr || dst == nullptr);
	if (src == dst && src_offset < dst_offset + size && dst_offset < src_offset + size) {
		EXIT("BufferCache: resolved Vulkan copy ranges overlap\n");
	}
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0 || size == 0 || vaddr >= TRACKER_ADDRESS_SIZE ||
	    size > TRACKER_ADDRESS_SIZE - vaddr) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	std::vector<DownloadCopy> copies;
	size_t                    retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			m_memory_tracker.ForEachDownloadRange<false>(
			    buffer.CpuAddress(), buffer.Size(),
			    [&](uint64_t dirty_address, uint64_t dirty_size) noexcept {
				    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, dirty_address,
				                                           dirty_size, "garbage collection");
			    },
			    [&](uint64_t dirty_address, uint64_t dirty_size) noexcept {
				    m_gpu_modified_ranges.ForEachIntersection(
				        dirty_address, dirty_size, [&](RangeSet::Range range) {
					    copies.push_back({&buffer, range.address - buffer.CpuAddress(),
					                      range.address, range.size});
				        });
				});
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	EXIT_IF(copies.empty());
	DownloadBufferMemory(copies);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		if (m_batch_depth != 0) {
			m_retired_in_batch.push_back(id);
		} else {
			m_slot_buffers.erase(id);
		}
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

} // namespace Libs::Graphics
