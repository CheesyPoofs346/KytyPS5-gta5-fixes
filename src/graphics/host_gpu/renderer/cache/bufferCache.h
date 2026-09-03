#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include <memory>
#include <shared_mutex>
#include "graphics/host_gpu/renderer/drawWorkerContext.h"
#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <map>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = uint64_t {1} << (40 - CACHING_PAGEBITS);
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	// Ownership rule for parallel resolve: while a batch is open, no Buffer is destroyed.
	//
	// CreateBuffer merges overlapping ranges and deletes what it subsumes, and GC deletes by age.
	// Either can free a Buffer that another worker is holding a pointer into - the addresses
	// themselves are stable (SlotVector stores a deque), so destruction is the whole hazard.
	// Deletions that land during a batch are recorded and applied at EndBatch, on one thread.
	void BeginBatch() noexcept { m_batch_depth++; }
	void EndBatch();

	// Locking is off until workers actually run. A shared_lock is ~20 ns and FindBuffer runs about
	// six times a draw, so paying it single-threaded would be a regression for no benefit.
	void SetConcurrent(bool concurrent) noexcept { m_concurrent = concurrent; }

	// An upload the GPU still needs, staged but not yet recorded.
	//
	// Resolve used to record barrier/copyBuffer/barrier straight into the primary, which is both
	// something a worker thread may not do and - now that draws are batched - a tear of the open
	// batch on every upload, because it called EndRendering to leave the render pass.
	struct PendingBufferUpload {
		vk::Buffer                  source;
		vk::Buffer                  destination;
		uint64_t                    destination_size = 0;
		std::vector<vk::BufferCopy> copies;
	};

	// Records every staged upload into the primary, in the order they were requested. Must run on
	// the main thread, outside a render pass, before the draws that read them.
	void FlushPendingUploads();

private:
	// The one place upload commands are recorded, shared by the immediate and deferred paths so
	// they cannot drift apart.
	void RecordUpload(vk::Buffer destination, uint64_t destination_size, vk::Buffer source,
	                  std::span<const vk::BufferCopy> copies);

public:

	[[nodiscard]] bool HasPendingUploads() const noexcept { return !m_pending_uploads.empty(); }

	// Safe to call before any worker exists; idempotent.
	void CreateWorkerStreamBuffers(uint32_t worker_count);
	[[nodiscard]] bool BatchActive() const noexcept { return m_batch_depth != 0; }

	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: {
				// One ring per worker slot. A single bump-allocated ring has one cursor, so two
				// workers mapping concurrently would be handed overlapping regions and would
				// silently corrupt each other's uniform data - no crash, just wrong constants.
				// Slot 0 keeps the original 64 MiB ring, so single-threaded behaviour is
				// unchanged; a worker ring that fills simply returns null from Map and the caller
				// falls through to the normal buffer path.
				const auto worker = CurrentDrawWorker();
				if (worker == 0 || worker > m_worker_stream_buffers.size()) {
					return m_stream_buffer;
				}
				return *m_worker_stream_buffers[worker - 1];
			}
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void               ProcessFaultBuffer();
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	void               RunGarbageCollector();

private:
	friend struct BufferCacheTestAccess;

	struct DownloadCopy;
	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 40, 16>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	static constexpr uint64_t               DOWNLOAD_ALIGNMENT = 64;
	[[nodiscard]] static constexpr uint64_t AlignDownload(uint64_t size) noexcept {
		return (size + DOWNLOAD_ALIGNMENT - 1) & ~(DOWNLOAD_ALIGNMENT - 1);
	}
	[[nodiscard]] static std::pair<uint64_t, uint64_t> DownloadEnvelope(const DownloadCopy& copy);
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void DownloadBufferMemory(std::span<const DownloadCopy> copies);
	void WriteHostMemory(uint64_t vaddr, std::span<const uint8_t> data);
	void ReadMemoryOnGpu(uint64_t vaddr, uint64_t size, bool is_write);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	std::map<uint64_t, BufferId>                      m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	std::vector<std::unique_ptr<StreamBuffer>>         m_worker_stream_buffers;
	TextureCache&                                     m_texture_cache;
	// Guards m_page_table and m_buffers, which Register/Unregister mutate together.
	std::shared_mutex                                 m_page_table_lock;
	// TouchBuffer writes the LRU on every buffer resolution, so shared it would serialise every
	// worker on a pure read path. Each worker records its touches and EndBatch merges them.
	std::vector<std::vector<size_t>>                  m_deferred_touch;
	bool                                              m_concurrent = false;
	std::vector<PendingBufferUpload>                  m_pending_uploads;
	std::vector<BufferId>                             m_retired_in_batch;
	uint32_t                                          m_batch_depth        = 0;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
