#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_

#include "common/assert.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <compare>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class CommandScheduler;
struct ImageTestAccess;

using ImageId = Common::SlotId;

struct CachedImageView {
	ImageViewInfo info;
	vk::ImageView view = nullptr;
};

// Sticky, set-only flags: resolution turns them on and nothing turns them off while it runs.
//
// Atomic rather than deferred. A deferred list is the right tool for a value that accumulates -
// the LRU touch - but these only ever go false to true, so two workers setting the same one race
// benignly in intent and identically in outcome. Making the store atomic gives that intent
// defined behaviour without a merge step or any ordering to get wrong. Relaxed is sufficient:
// nothing is published through these, they are read after the batch by the main thread.
struct ImageUsage {
	std::atomic<bool> texture       = false;
	std::atomic<bool> storage       = false;
	std::atomic<bool> render_target = false;
	std::atomic<bool> depth_target  = false;
	std::atomic<bool> video_out     = false;

	ImageUsage() = default;
	ImageUsage(const ImageUsage& other) noexcept { *this = other; }
	ImageUsage& operator=(const ImageUsage& other) noexcept {
		texture.store(other.texture.load(std::memory_order_relaxed), std::memory_order_relaxed);
		storage.store(other.storage.load(std::memory_order_relaxed), std::memory_order_relaxed);
		render_target.store(other.render_target.load(std::memory_order_relaxed),
		                    std::memory_order_relaxed);
		depth_target.store(other.depth_target.load(std::memory_order_relaxed),
		                   std::memory_order_relaxed);
		video_out.store(other.video_out.load(std::memory_order_relaxed), std::memory_order_relaxed);
		return *this;
	}
};

struct ImageBinding {
	bool is_bound      = false;
	bool is_target     = false;
	bool needs_rebind  = false;
	bool force_general = false;
	bool shader_write  = false;
};

class Image final {
public:
	Image(GraphicContext& graphics, CommandScheduler& scheduler, const ImageInfo& info);
	~Image();
	KYTY_CLASS_NO_COPY(Image);

	[[nodiscard]] vk::ImageView FindView(const ImageViewInfo& view_info);
	void                        AssociateDepth(ImageId image_id) { depth_id = image_id; }
	using Barriers = std::vector<vk::ImageMemoryBarrier2>;
	[[nodiscard]] Barriers GetBarriers(vk::ImageLayout                      destination_layout,
	                                   vk::AccessFlags2                     destination_access,
	                                   vk::PipelineStageFlags2              destination_stage,
	                                   std::optional<ImageSubresourceRange> range);
	void Transit(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
	             std::optional<ImageSubresourceRange> range, vk::CommandBuffer command_buffer);
	void Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	            uint64_t size);
	void Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	              uint64_t size);
	void CopyImage(Image& source);
	void Resolve(Image& source, const ImageSubresourceRange& source_range,
	             const ImageSubresourceRange& destination_range);
	void CopyImageWithBuffer(Image& source, Buffer& buffer);
	void CopyMip(Image& source, uint32_t mip, uint32_t layer);

	void InvalidateCpuWrite(uint64_t vaddr, uint64_t size) {
		if (ImageRangeOverlaps(info.data.address, info.data.size, vaddr, size)) {
			m_cpu_dirty        = true;
			m_maybe_cpu_dirty  = false;
			m_maybe_hash_valid = false;
		} else if (ImagePageRangesOverlap(info.data.address, info.data.size, vaddr, size)) {
			m_maybe_cpu_dirty = true;
		}
	}

	[[nodiscard]] bool IsCpuDirty() const { return m_cpu_dirty || m_maybe_cpu_dirty; }
	[[nodiscard]] bool IsDefinitelyCpuDirty() const { return m_cpu_dirty; }
	[[nodiscard]] bool IsMaybeCpuDirty() const { return m_maybe_cpu_dirty; }
	void               MarkMaybeCpuDirty() {
		if (!m_cpu_dirty) {
			m_maybe_cpu_dirty = true;
		}
	}
	[[nodiscard]] bool NeedsMaybeCpuHash() const {
		return m_maybe_cpu_dirty && !m_maybe_hash_valid;
	}
	void SetMaybeCpuHash(uint64_t hash) {
		if (!NeedsMaybeCpuHash()) {
			EXIT("image cannot initialize maybe-dirty hash\n");
		}
		m_maybe_cpu_hash   = hash;
		m_maybe_hash_valid = true;
	}
	[[nodiscard]] bool ResolveMaybeCpuHash(uint64_t hash) {
		if (!m_maybe_cpu_dirty || !m_maybe_hash_valid || m_cpu_dirty) {
			EXIT("image cannot resolve maybe-dirty hash\n");
		}
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
		m_cpu_dirty |= hash != m_maybe_cpu_hash;
		return m_cpu_dirty;
	}

	void RefreshComplete() {
		if (!IsCpuDirty()) {
			EXIT("clean image cannot complete a refresh\n");
		}
		m_cpu_dirty        = false;
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
	}

	[[nodiscard]] bool IsGpuModified() const noexcept {
		return m_gpu_modified.load(std::memory_order_relaxed);
	}
	// Set from resolution, which will run on workers; cleared only by the main thread outside a
	// batch, so a set can never race a clear.
	void MarkGpuModified() noexcept { m_gpu_modified.store(true, std::memory_order_relaxed); }
	void ClearGpuModified() noexcept { m_gpu_modified.store(false, std::memory_order_relaxed); }

	[[nodiscard]] bool IsBufferModified() const noexcept { return m_buffer_modified; }
	void               MarkBufferModified() noexcept { m_buffer_modified = true; }
	void               ClearBufferModified() noexcept { m_buffer_modified = false; }

	[[nodiscard]] bool Overlaps(uint64_t address, uint64_t size,
	                            bool pages = false) const noexcept {
		return pages ? ImagePageRangesOverlap(info.data.address, info.data.size, address, size)
		             : ImageRangeOverlaps(info.data.address, info.data.size, address, size);
	}
	[[nodiscard]] bool GpuOverlaps(uint64_t address, uint64_t size) const noexcept {
		return IsGpuModified() && Overlaps(address, size);
	}
	[[nodiscard]] bool SafeToDownload() const noexcept {
		return IsGpuModified() && !IsBufferModified() && !IsCpuDirty();
	}
	[[nodiscard]] bool IsTracked() const noexcept { return track_addr != 0 && track_addr_end != 0; }
	[[nodiscard]] uint64_t AccountedSize() const noexcept {
		return backing.image == nullptr ? 0 : (info.data.size + 1023) & ~uint64_t {1023};
	}
	[[nodiscard]] uint64_t HashGuestEdges() const;

	ImageInfo        info;
	VulkanImage      backing;
	std::vector<CachedImageView> views;
	ImageUsage       usage;
	ImageBinding     binding;
	bool             registered     = false;
	bool                         cpu_read_tracked = false;
	mutable uint32_t query_epoch    = 0;
	uint64_t         track_addr     = 0;
	uint64_t         track_addr_end = 0;
	ImageId          depth_id {};
	uint64_t         tick_accessed_last = 0;
	size_t           lru_id             = 0;

private:
	friend struct ImageTestAccess;

	[[nodiscard]] static vk::ImageAspectFlags FullAspectMask(vk::Format format) noexcept;
	[[nodiscard]] static uint32_t             CopyRows(uint64_t row_size, uint32_t rows,
	                                                   uint64_t capacity) noexcept;
	[[nodiscard]] static std::pair<uint32_t, uint32_t>
	SanitizeCopyLayers(const Image& source, const Image& destination, uint32_t depth);

	GraphicContext*   m_graphics         = nullptr;
	CommandScheduler* m_scheduler        = nullptr;
	uint64_t          m_maybe_cpu_hash   = 0;
	bool              m_cpu_dirty        = false;
	bool              m_maybe_cpu_dirty  = false;
	bool              m_maybe_hash_valid = false;
	std::atomic<bool> m_gpu_modified     = false;
	bool              m_buffer_modified  = false;
};

namespace ImageOps {

void                                 Validate(const ImageInfo& info);
[[nodiscard]] Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element);

} // namespace ImageOps

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
