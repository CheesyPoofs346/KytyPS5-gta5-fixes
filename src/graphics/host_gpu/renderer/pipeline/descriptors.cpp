#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/drawBatchQueue.h"   // kMaxBatchDraws

#include "common/assert.h"
#include <xxhash.h>
#include "common/emulatorConfig.h"
#include "graphics/host_gpu/renderer/drawProfile.h"
#include "common/common.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vma.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <fmt/format.h>
#include <limits>
#include <span>
#include <vector>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace Libs::Graphics {

// Per-draw scratch, one copy per recording thread.
//
// Both are cleared and refilled from scratch on every draw and never read across draws, so making
// them thread_local changes no semantics - it only stops eight workers recording into the same
// buffer. Kept as function-local statics rather than RenderExecutor members because a member would
// have to be indexed by worker at every use site; this matches DynState() in renderDraw.cpp.
std::vector<vk::WriteDescriptorSet>& DescriptorWrites() {
	static thread_local std::vector<vk::WriteDescriptorSet> writes;
	return writes;
}

std::array<uint32_t, ShaderRecompiler::IR::NativePushConstantSize / sizeof(uint32_t)>&
PushConstants() {
	static thread_local
	    std::array<uint32_t, ShaderRecompiler::IR::NativePushConstantSize / sizeof(uint32_t)>
	        constants {};
	return constants;
}


namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

bool IsSampledImage(BindingKind kind) {
	switch (kind) {
		case BindingKind::Sampled1D:
		case BindingKind::Sampled1DArray:
		case BindingKind::Sampled2D:
		case BindingKind::Sampled2DArray:
		case BindingKind::Sampled2DMsaa:
		case BindingKind::Sampled2DMsaaArray:
		case BindingKind::Sampled3D:
		case BindingKind::SampledUint1D:
		case BindingKind::SampledUint1DArray:
		case BindingKind::SampledUint2D:
		case BindingKind::SampledUint2DArray:
		case BindingKind::SampledUint2DMsaa:
		case BindingKind::SampledUint2DMsaaArray:
		case BindingKind::SampledUint3D: return true;
		default: return false;
	}
}

bool IsStorageImage(BindingKind kind) {
	switch (kind) {
		case BindingKind::Storage1D:
		case BindingKind::Storage1DArray:
		case BindingKind::Storage2D:
		case BindingKind::Storage2DArray:
		case BindingKind::Storage3D:
		case BindingKind::StorageUint1D:
		case BindingKind::StorageUint1DArray:
		case BindingKind::StorageUint2D:
		case BindingKind::StorageUint2DArray:
		case BindingKind::StorageUint3D:
		case BindingKind::StorageAtomic1D:
		case BindingKind::StorageAtomic1DArray:
		case BindingKind::StorageAtomic2D:
		case BindingKind::StorageAtomic2DArray:
		case BindingKind::StorageAtomic3D: return true;
		default: return false;
	}
}

} // namespace

vk::DescriptorType NativeDescriptorType(BindingKind kind) {
	if (kind == BindingKind::Samplers) {
		return vk::DescriptorType::eSampler;
	}
	if (IsSampledImage(kind)) {
		return vk::DescriptorType::eSampledImage;
	}
	if (IsStorageImage(kind)) {
		return vk::DescriptorType::eStorageImage;
	}
	return vk::DescriptorType::eStorageBuffer;
}

uint32_t NativeDescriptorCount(const ShaderRecompiler::IR::DescriptorBinding& binding) {
	return binding.resources.empty() ? 1u : static_cast<uint32_t>(binding.resources.size());
}

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	const auto view =
	    texture.mip_views.empty()
	        ? (element == 0u ? texture.image_view : vk::ImageView {})
	        : (element < texture.mip_views.size() ? texture.mip_views[element] : vk::ImageView {});
	EXIT_IF(!texture.image_id || view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	return {nullptr, view, texture.layout};
}

static const char* ShaderStageResourceName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "Vertex";
		case ShaderType::Pixel: return "Pixel";
		case ShaderType::Compute: return "Compute";
		default: return "Unknown";
	}
}

static vk::ShaderStageFlags NativeShaderStage(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return vk::ShaderStageFlagBits::eVertex;
		case ShaderType::Pixel: return vk::ShaderStageFlagBits::eFragment;
		case ShaderType::Compute: return vk::ShaderStageFlagBits::eCompute;
		default: EXIT("unknown native shader stage\n");
	}
}

static void BindNullStorageBuffer(RenderContext& context, BufferView& dst) {
	dst.buffer = context.GetBufferCache().GetBuffer(NULL_BUFFER_ID).Handle();
	dst.offset = 0;
	dst.range  = 16;
}

static void CopyNativeDescriptor(const ShaderRecompiler::IR::DescriptorValue& source,
                                 std::span<uint32_t>                          destination) {
	EXIT_IF(source.dword_count != destination.size());
	std::copy_n(source.dwords.begin(), destination.size(), destination.begin());
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1DArray: return Prospero::ImageType::kColor1D;
		case Prospero::ImageType::kColor2DArray:
		case Prospero::ImageType::kColor2DMsaa:
		case Prospero::ImageType::kColor2DMsaaArray: return Prospero::ImageType::kColor2D;
		default: return type;
	}
}

static bool IsMultisampledTexture(Prospero::ImageType type) {
	return type == Prospero::ImageType::kColor2DMsaa ||
	       type == Prospero::ImageType::kColor2DMsaaArray;
}

namespace {

// Cache of buffer resolutions valid for the duration of ONE draw. See the comment at its use
// site for why the scope is exactly one draw and no longer.
// The key must contain EVERY input ObtainBuffer uses, or the cache hands one resource another
// resource's view. An earlier version keyed on address+size alone and ignored `formatted` and
// the BufferId; that is the same omitted-input mistake that made light coronas render as red
// blobs twice before. `written` is not a key field because written buffers are never cached.
struct DrawBufferCacheEntry {
	uint64_t   address       = 0;
	uint64_t   size          = 0;
	bool       formatted     = false;
	BufferId   id {};
	uint32_t   alignment     = 0;
	uint32_t   buffer_offset = 0;
	BufferView view;
	// Whether the cached view points at a throwaway ring allocation. Publication republishes
	// stream entries verbatim and re-derives cache-backed ones from the guest range.
	bool       stream        = false;
};

const DrawBufferCacheEntry* FindDrawBufferCache(uint64_t address, uint64_t size, bool formatted,
                                                BufferId id, uint32_t alignment);

std::vector<DrawBufferCacheEntry>& DrawBufferCache() {
	static thread_local std::vector<DrawBufferCacheEntry> cache;
	return cache;
}

const DrawBufferCacheEntry* FindDrawBufferCache(uint64_t address, uint64_t size, bool formatted,
                                                BufferId id, uint32_t alignment) {
	for (const auto& e: DrawBufferCache()) {
		// alignment participates because buffer_offset is derived from it.
		if (e.address == address && e.size == size && e.formatted == formatted && e.id == id &&
		    e.alignment == alignment) {
			return &e;
		}
	}
	return nullptr;
}

} // namespace

void BeginDrawBufferScope() {
	DrawBufferCache().clear();
}

static BufferView NativeStorageBuffer(RenderContext&                              context,
                                      const ShaderBufferResource&                 descriptor,
                                      const ShaderRecompiler::IR::BufferResource& resource,
                                      ShaderType stage, uint32_t slot, uint32_t& buffer_offset,
                                      BufferId id, uint64_t clamped_size,
                                      PreparedBindings::PublishedBuffer* publish) {
	BufferView result;
	buffer_offset = 0;

	const auto address = descriptor.Base48();
	const auto stride  = descriptor.Stride();
	const auto records = descriptor.NumRecords();
	if (stride != 0 && records > UINT64_MAX / stride) {
		EXIT("storage buffer descriptor footprint overflow\n");
	}
	const auto requested_size = stride != 0 ? static_cast<uint64_t>(stride) * records : records;
	if (address == 0 || requested_size == 0) {
		BindNullStorageBuffer(context, result);
		return result;   // publish->resolved stays false: nothing to re-derive.
	}
	// Already clamped by FindBuffers for this exact descriptor; asking again costs a query that
	// blocks on a contended lock roughly 4% of the time.
	const auto  size      = clamped_size;
	const auto& graphics  = context.GetGraphics();
	const auto  alignment = graphics.StorageMinAlignment();
	if (alignment == 0 ||
	    size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range or device alignment is unsupported\n");
	}
	// Per-draw dedup. ObtainBuffer is not cheap - two locked memory-tracker queries, and for
	// small read-only buffers a memcpy of the guest data into the stream buffer. Vertex and pixel
	// stages frequently bind the SAME buffer, and RebindBuffers runs once per stage, so the
	// identical request is served twice per draw.
	//
	// Scoped to one draw (cleared by BeginDrawBufferScope), because between draws the guest may
	// have written the memory and the answer can legitimately change. Written buffers are never
	// cached - they mutate GPU-side state.
	if (!resource.written && Config::BufferDedupEnabled()) {
		// Timed to close the buffer loop's accounting: 1.355 us/draw sat between the loop and its
		// measured children, and this linear scan was one of the two untimed things inside it.
		DrawPhaseTimer dedup_timer(DrawPhase::BindDedupScan);
		if (const auto* hit = FindDrawBufferCache(address, size, resource.formatted, id,
		                                          static_cast<uint32_t>(alignment))) {
			buffer_offset = hit->buffer_offset;
			// A dedup hit returns a view captured EARLIER in this draw, so it can already be
			// stale. Record its identity anyway: publication re-derives every resolved entry, so
			// a hit cannot bypass final-owner validation.
			if (publish != nullptr) {
				publish->address   = address;
				publish->size      = size;
				publish->alignment = static_cast<uint32_t>(alignment);
				publish->stream    = hit->stream;
				publish->resolved  = true;
			}
			return hit->view;
		}
	}
	auto [buffer, offset] = context.GetBufferCache().ObtainBuffer(address, size, resource.written,
	                                                              resource.formatted, id);
	if (buffer == nullptr) {
		// A worker abandoned the draw inside ObtainBuffer - a written buffer, which publishes into
		// shared state. The half-built view is discarded with the rest of the prepared entry and
		// phase 3 rebuilds the bindings serially.
		return result;
	}
	const auto aligned_offset = offset - offset % alignment;
	const auto adjustment     = offset - aligned_offset;
	const bool stream_backed  = context.GetBufferCache().IsStreamAllocation(buffer);
	if (publish != nullptr) {
		publish->address   = address;
		publish->size      = size;
		publish->alignment = static_cast<uint32_t>(alignment);
		// Stream-backed: a throwaway ring allocation holding a private copy. It is not owned by
		// any guest range and must be republished verbatim, allocation and offset intact.
		publish->stream   = stream_backed;
		publish->resolved = true;
	}
	const auto max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	if (adjustment % sizeof(uint32_t) != 0 || adjustment >= 256 || size > max_range - adjustment) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	buffer_offset            = static_cast<uint32_t>(adjustment);
	result.publish_adjustment = buffer_offset;
	result.buffer            = buffer->Handle();
	result.offset = aligned_offset;
	result.range  = static_cast<vk::DeviceSize>(size + adjustment);
	if (resource.formatted && resource.written) {
		context.GetTextureCache().InvalidateMemoryFromGPU(address, size);
	}
	SetVulkanObjectNameF(
	    graphics.device, result.buffer,
	    "Kyty.{}.StorageBuffer[slot={} guest=0x{:016x} size=0x{:x} access={} formatted={}]",
	    ShaderStageResourceName(stage), slot, address, size,
	    resource.written ? (resource.read ? "ReadWrite" : "Write") : "Read", resource.formatted);
	if (!resource.written && Config::BufferDedupEnabled() && DrawBufferCache().size() < 32) {
		DrawBufferCache().push_back({address, size, resource.formatted, id,
		                             static_cast<uint32_t>(alignment), buffer_offset, result,
		                             stream_backed});
	}
	return result;
}

static bool IsSupportedSampledColorResource(const ShaderRecompiler::IR::ImageResource& resource) {
	bool supported_dimension = false;
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			supported_dimension = true;
			break;
		default: break;
	}
	const bool sampled_kind = resource.kind == ShaderRecompiler::IR::ResourceKind::Image ||
	                          resource.kind == ShaderRecompiler::IR::ResourceKind::ImageUint;
	return sampled_kind && supported_dimension &&
	       resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::None && resource.read &&
	       !resource.written && !resource.atomic && !resource.depth_compare;
}

TargetTextureViewInfo ResolveTargetTextureView(const ShaderRecompiler::IR::ImageResource& resource,
                                               Prospero::ImageType type, uint32_t base_layer,
                                               uint32_t image_layers) {
	switch (type) {
		case Prospero::ImageType::kColor2D:
			return resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D &&
			               base_layer == 0 && image_layers == 1
			           ? TargetTextureViewInfo {vk::ImageViewType::e2D, 0, 1}
			           : TargetTextureViewInfo {};
		case Prospero::ImageType::kCube:
			if (resource.dimension != ShaderRecompiler::Decoder::ImageDimension::Dim2DArray ||
			    base_layer >= image_layers || (image_layers - base_layer) % 6u != 0) {
				return {};
			}
			return {vk::ImageViewType::e2DArray, base_layer, image_layers - base_layer};
		case Prospero::ImageType::kColor2DArray:
			if (resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D &&
			    base_layer == 0 && image_layers == 1) {
				return {vk::ImageViewType::e2D, 0, 1};
			}
			return resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray &&
			               base_layer < image_layers
			           ? TargetTextureViewInfo {vk::ImageViewType::e2DArray, base_layer,
			                                    image_layers - base_layer}
			           : TargetTextureViewInfo {};
		case Prospero::ImageType::kColor2DMsaa:
			return resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa &&
			               base_layer == 0 && image_layers == 1
			           ? TargetTextureViewInfo {vk::ImageViewType::e2D, 0, 1}
			           : TargetTextureViewInfo {};
		case Prospero::ImageType::kColor2DMsaaArray:
			if (resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa &&
			    base_layer == 0 && image_layers == 1) {
				return {vk::ImageViewType::e2D, 0, 1};
			}
			return resource.dimension ==
			                   ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray &&
			               base_layer < image_layers
			           ? TargetTextureViewInfo {vk::ImageViewType::e2DArray, base_layer,
			                                    image_layers - base_layer}
			           : TargetTextureViewInfo {};
		default: return {};
	}
}

bool IsSupportedSampledVideoOutView(const ShaderRecompiler::IR::ImageResource& resource,
                                    const ShaderTextureResource& descriptor, const Image& image) {
	return image.usage.video_out && image.info.resources.layers == 1 &&
	       IsSupportedSampledColorResource(resource) &&
	       resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D &&
	       descriptor.Type() == Prospero::ImageType::kColor2D && descriptor.Depth() == 0 &&
	       descriptor.BaseArray5() == 0;
}

bool IsSupportedDepthTargetDescriptor(const ShaderTextureResource& descriptor, const Image& image,
                                      bool r128) {
	const auto width        = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto height       = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto type         = descriptor.Type();
	const bool multisampled = IsMultisampledTexture(type);
	const auto samples      = multisampled ? 1u << descriptor.LastLevel() : 1u;
	const auto pitch =
	    multisampled ? TileGetDepthPitch(width, image.info.bytes_per_block, descriptor.LastLevel())
	                 : TileGetTexturePitch(descriptor.Format(), width, descriptor.TileMode());
	const bool supported_2d    = type == Prospero::ImageType::kColor2D &&
	                             image.info.resources.layers == 1 && descriptor.Depth() == 0 &&
	                             descriptor.BaseArray5() == 0;
	const bool supported_array = type == Prospero::ImageType::kColor2DArray &&
	                             descriptor.BaseArray5() <= descriptor.Depth() &&
	                             descriptor.Depth() < image.info.resources.layers;
	const bool supported_cube =
	    type == Prospero::ImageType::kCube && width == height && image.info.resources.layers >= 6 &&
	    image.info.resources.layers % 6u == 0 &&
	    static_cast<uint32_t>(descriptor.Depth()) + 1u == image.info.resources.layers &&
	    descriptor.BaseArray5() == 0;
	const bool supported_msaa_2d    = type == Prospero::ImageType::kColor2DMsaa &&
	                                  image.info.resources.layers == 1 && descriptor.Depth() == 0 &&
	                                  descriptor.BaseArray5() == 0;
	const bool supported_msaa_array = type == Prospero::ImageType::kColor2DMsaaArray &&
	                                  descriptor.BaseArray5() <= descriptor.Depth() &&
	                                  descriptor.Depth() < image.info.resources.layers;
	const bool levels_ok =
	    multisampled ? descriptor.BaseLevel() == 0 && descriptor.LastLevel() >= 1 &&
	                       descriptor.LastLevel() <= 3 &&
	                       (r128 || descriptor.MaxMip() == descriptor.LastLevel()) &&
	                       image.info.resources.levels == 1 && image.info.samples == samples
	                 : descriptor.BaseLevel() == 0 && descriptor.LastLevel() == 0 &&
	                       (r128 || descriptor.MaxMip() == 0) && image.info.samples == 1;
	return image.info.IsDepth() && width == image.info.extent.width &&
	       height == image.info.extent.height &&
	       (supported_2d || supported_array || supported_cube || supported_msaa_2d ||
	        supported_msaa_array) &&
	       levels_ok && descriptor.MinLod() == 0 &&
	       descriptor.TileMode() == Prospero::TileMode::kDepth && descriptor.BCSwizzle() == 0 &&
	       (!descriptor.MsaaDepth() || multisampled) && pitch >= width && pitch == image.info.pitch;
}

bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor, const Image& image,
                                     bool r128) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	const bool         common          = (descriptor.fields[1] & field1_reserved_mask) == 0 &&
	                                     (descriptor.fields[2] & field2_reserved_mask) == 0 &&
	                                     descriptor.fields[3] == field3_expected;
	if (r128) {
		return common && descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	const bool full = common && descriptor.fields[4] == field4_expected &&
	                  descriptor.fields[5] == field5_expected;
	if (!full || (descriptor.fields[6] == 0 && descriptor.fields[7] != 0)) {
		return false;
	}
	if (descriptor.fields[6] == 0) {
		return true;
	}
	constexpr uint32_t htile_control = 0x00280000u;
	const uint32_t expected_control  = htile_control | (descriptor.MsaaDepth() ? (1u << 10u) : 0u);
	const auto     metadata_addr     = descriptor.MetaAddr() << 8u;
	return (descriptor.fields[6] & 0x00ffffffu) == expected_control && metadata_addr != 0 &&
	       descriptor.TileMode() == Prospero::TileMode::kDepth &&
	       image.info.tile_mode == Prospero::TileMode::kDepth &&
	       image.info.metadata.kind == ImageMetadataKind::Htile &&
	       image.info.metadata.range.Valid() && image.info.metadata.range.address == metadata_addr;
}

static void ValidateDepthTargetBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                       const ShaderTextureResource& descriptor, const Image* image,
                                       vk::Format view_format, uint64_t size) {
	const bool resource_ok = IsSupportedSampledDepthResource(resource);
	const bool descriptor_ok =
	    image != nullptr && IsSupportedDepthTargetDescriptor(descriptor, *image, resource.r128);
	const bool encoding_ok =
	    image != nullptr && IsSupportedDepthTextureEncoding(descriptor, *image, resource.r128);
	const bool format_ok =
	    image != nullptr && IsSupportedSampledDepthFormat(image->info.pixel_format, view_format);
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return;
	}
	const auto descriptor_pitch =
	    TileGetTexturePitch(descriptor.Format(), static_cast<uint32_t>(descriptor.Width5()) + 1u,
	                        descriptor.TileMode());
	EXIT("unsupported sampled depth target: resource=%d descriptor=%d encoding=%d format=%d "
	     "kind=%u dimension=%u mip_mode=%u read=%d written=%d atomic=%d compare=%d "
	     "guest_format=%u swizzle=0x%03x image_format=%d view_format=%d image_layers=%u "
	     "descriptor_type=%u base_array=%u depth=%u descriptor_pitch=%u target_pitch=%u "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, descriptor_ok, encoding_ok, format_ok, static_cast<uint32_t>(resource.kind),
	     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
	     resource.read, resource.written, resource.atomic, resource.depth_compare,
	     static_cast<uint32_t>(descriptor.Format()), descriptor.DstSelXYZW(),
	     image == nullptr ? static_cast<int>(vk::Format::eUndefined)
	                      : static_cast<int>(image->info.pixel_format),
	     static_cast<int>(view_format), image == nullptr ? 0u : image->info.resources.layers,
	     static_cast<uint32_t>(descriptor.Type()), descriptor.BaseArray5(), descriptor.Depth(),
	     descriptor_pitch, image == nullptr ? 0u : image->info.pitch, descriptor.Base40(), size,
	     descriptor.fields[0], descriptor.fields[1], descriptor.fields[2], descriptor.fields[3],
	     descriptor.fields[4], descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static bool IsSupportedStorageTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                const ShaderTextureResource& descriptor) {
	const auto tile              = descriptor.TileMode();
	const bool is_color_1d       = descriptor.Type() == Prospero::ImageType::kColor1D;
	const bool is_color_1d_array = descriptor.Type() == Prospero::ImageType::kColor1DArray;
	const bool valid_1d_slice =
	    (is_color_1d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_1d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_1d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D &&
	                   descriptor.Height5() == 0 && valid_1d_slice;
	const bool is_1d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray &&
	    is_color_1d_array && descriptor.Height5() == 0 &&
	    descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_color_2d       = descriptor.Type() == Prospero::ImageType::kColor2D;
	const bool is_color_2d_array = descriptor.Type() == Prospero::ImageType::kColor2DArray;
	const bool valid_2d_slice =
	    (is_color_2d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_2d =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D && valid_2d_slice;
	const bool is_cube = resource.cube && descriptor.Type() == Prospero::ImageType::kCube &&
	                     descriptor.Width5() == descriptor.Height5() &&
	                     descriptor.BaseArray5() <= descriptor.Depth() &&
	                     (descriptor.Depth() - descriptor.BaseArray5() + 1u) % 6u == 0;
	const bool is_2d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray &&
	    ((!resource.cube && is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth()) ||
	     is_cube);
	const bool is_3d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	                   descriptor.Type() == Prospero::ImageType::kColor3D &&
	                   descriptor.BaseArray5() == 0;
	TileTextureBlockLayout tile_layout {};
	bool                   supported_tile = false;
	switch (tile) {
		case Prospero::TileMode::kLinear: supported_tile = true; break;
		case Prospero::TileMode::kDepth:
			supported_tile =
			    !resource.read && !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard256B:
			supported_tile =
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, is_3d, tile_layout);
			break;
		case Prospero::TileMode::kRenderTarget:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		default: break;
	}
	const auto swizzle = descriptor.DstSelXYZW();
	const bool supported_swizzle =
	    IsValidImageSwizzle(swizzle) &&
	    (swizzle == DstSel(4, 5, 6, 7) || !resource.read || resource.atomic);
	const auto max_mip = resource.r128 ? descriptor.LastLevel() : descriptor.MaxMip();
	const auto view_last_level =
	    resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::DynamicStorage
	        ? descriptor.LastLevel()
	        : std::min(descriptor.LastLevel(), max_mip);
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d) && supported_tile &&
	       descriptor.BaseLevel() <= view_last_level && view_last_level <= max_mip &&
	       descriptor.MinLod() == 0 && supported_swizzle && descriptor.BCSwizzle() == 0 &&
	       !descriptor.MsaaDepth();
}

static bool IsSupportedStorageTextureEncoding(const ShaderRecompiler::IR::ImageResource& resource,
                                              const ShaderTextureResource& descriptor) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	constexpr uint32_t field5_expected      = 0x00700000u;
	constexpr uint32_t field5_max_mip_mask  = 0x000000f0u;
	const uint32_t     expected_field3 = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     expected_field4 =
	    descriptor.Depth() | (static_cast<uint32_t>(descriptor.BaseArray5()) << 16u);
	const bool common = (descriptor.fields[1] & field1_reserved_mask) == 0 &&
	                    (descriptor.fields[2] & field2_reserved_mask) == 0 &&
	                    descriptor.fields[3] == expected_field3;
	if (resource.r128) {
		return common && descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	return common && descriptor.fields[4] == expected_field4 &&
	       (descriptor.fields[5] & ~field5_max_mip_mask) == field5_expected;
}

void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsSupportedStorageTextureEncoding(resource, descriptor);
	const bool uint_resource =
	    resource.kind == ShaderRecompiler::IR::ResourceKind::StorageImageUint;
	const bool raw_sint_storage = format == Prospero::BufferFormat::k32SInt && uint_resource &&
	                              resource.written && !resource.read && !resource.atomic;
	const bool format_ok =
	    raw_sint_storage || (Prospero::IsSampledTextureFormat(format) &&
	                         uint_resource == Prospero::IsUintTextureFormat(format) &&
	                         (!resource.atomic || format == Prospero::BufferFormat::k32UInt));
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return;
	}
	EXIT("unsupported storage texture: resource=%d descriptor=%d encoding=%d format=%d "
	     "kind=%u dimension=%u mip_mode=%u atomic=%d compare=%d "
	     "base_level=%u last_level=%u max_mip=%u min_lod=%u base_array=%u bc=%u msaa=%d "
	     "depth_tile_bpe=%u swizzle_ok=%d "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " extent=%ux%ux%u type=%u format=%u tile=%u swizzle=0x%03x read=%d written=%d "
	     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, descriptor_ok, encoding_ok, format_ok, static_cast<uint32_t>(resource.kind),
	     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
	     resource.atomic, resource.depth_compare, descriptor.BaseLevel(), descriptor.LastLevel(),
	     descriptor.MaxMip(), descriptor.MinLod(), descriptor.BaseArray5(), descriptor.BCSwizzle(),
	     descriptor.MsaaDepth(), Prospero::RenderTargetBytesPerElement(format),
	     IsValidImageSwizzle(descriptor.DstSelXYZW()), descriptor.Base40(), size,
	     static_cast<uint32_t>(descriptor.Width5()) + 1u,
	     static_cast<uint32_t>(descriptor.Height5()) + 1u,
	     static_cast<uint32_t>(descriptor.Depth()) + 1u, static_cast<uint32_t>(descriptor.Type()),
	     static_cast<uint32_t>(format), static_cast<uint32_t>(descriptor.TileMode()),
	     descriptor.DstSelXYZW(), resource.read, resource.written, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

struct NullImageSpec {
	vk::Format             format;
	Prospero::BufferFormat guest_format;
};

static NullImageSpec NullTextureSpec(const ShaderRecompiler::IR::ImageResource& resource) {
	const bool uint_image = resource.kind == ShaderRecompiler::IR::ResourceKind::ImageUint ||
	                        resource.kind == ShaderRecompiler::IR::ResourceKind::StorageImageUint;
	return uint_image ? NullImageSpec {vk::Format::eR32Uint, Prospero::BufferFormat::k32UInt}
	                  : NullImageSpec {vk::Format::eR32Sfloat, Prospero::BufferFormat::k32Float};
}

static TextureCache::ImageDesc NullTextureDesc(const ShaderRecompiler::IR::ImageResource& resource,
                                               TextureCache::BindingType                  binding) {
	const auto              spec = NullTextureSpec(resource);
	TextureCache::ImageDesc desc {};
	desc.info.pixel_format    = spec.format;
	desc.info.guest_format    = spec.guest_format;
	desc.info.type            = Prospero::ImageType::kColor2D;
	desc.info.extent          = {1, 1, 1};
	desc.info.resources       = {1, 1};
	desc.info.bytes_per_block = 4;
	desc.info.samples         = 1;
	desc.info.mip_layout[0]   = {0, 0, 1, 1};
	desc.view_info.format     = desc.info.pixel_format;
	desc.view_info.type       = vk::ImageViewType::e2D;
	desc.view_info.aspect     = vk::ImageAspectFlagBits::eColor;
	desc.view_info.usage      = binding == TextureCache::BindingType::Storage
	                                ? vk::ImageUsageFlagBits::eStorage
	                                : vk::ImageUsageFlagBits::eSampled;
	desc.type                 = binding;
	return desc;
}

static void PopulateTextureMipLayout(ImageInfo& info) {
	if (info.IsVolume() && info.tile_mode != Prospero::TileMode::kLinear) {
		TileSurfaceLayout            surface {};
		const TileSurfaceDescription description {
		    info.guest_format,  info.tile_mode,    TileSurfaceDimension::Dim3D, info.extent.width,
		    info.extent.height, info.extent.depth, info.resources.levels,       1};
		if (!TileGetTiledTextureLayout(description, surface)) {
			EXIT("unsupported normalized volume texture layout\n");
		}
		for (uint32_t level = 0; level < info.resources.levels; level++) {
			const auto& mip        = surface.mips[level];
			info.mip_layout[level] = {
			    mip.offset,
			    mip.size,
			    mip.padded_width,
			    mip.padded_height,
			};
		}
		return;
	}

	TileSizeOffset levels[16] {};
	TilePaddedSize padded[16] {};
	TileGetTextureSize(info.guest_format, info.extent.width, info.extent.height,
	                   info.resources.levels, info.tile_mode, nullptr, levels, padded);
	for (uint32_t level = 0; level < info.resources.levels; level++) {
		const auto offset =
		    levels[level].src_size != 0 ? levels[level].src_offset : levels[level].offset;
		auto size = static_cast<uint64_t>(levels[level].src_size != 0 ? levels[level].src_size
		                                                              : levels[level].size);
		if (info.IsVolume()) {
			size *= std::max(info.extent.depth >> level, 1u);
		} else {
			size *= info.resources.layers;
		}
		info.mip_layout[level] = {
		    offset,
		    size,
		    padded[level].width != 0 ? padded[level].width : std::max(info.pitch >> level, 1u),
		    padded[level].height != 0 ? padded[level].height
		                              : std::max(info.extent.height >> level, 1u),
		};
	}
}

static ImageViewInfo TextureViewInfo(const ShaderRecompiler::IR::ImageResource& resource,
                                     const ShaderTextureResource& descriptor, vk::Format format,
                                     bool shader_conversion, bool storage, uint32_t view_levels,
                                     uint32_t image_layers) {
	ImageViewInfo view {};
	view.format      = format;
	view.aspect      = vk::ImageAspectFlagBits::eColor;
	view.base_level  = descriptor.BaseLevel();
	view.level_count = view_levels;
	view.usage   = storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled;
	view.mapping = storage || shader_conversion
	                   ? vk::ComponentMapping {}
	                   : TextureGetComponentMapping(descriptor.DstSelXYZW());
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
			view.type       = vk::ImageViewType::e1D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
			view.type       = vk::ImageViewType::e1DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type       = vk::ImageViewType::e2DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			view.type       = vk::ImageViewType::e2D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		default: EXIT("unsupported texture view dimension\n");
	}
	return view;
}

// ---------------------------------------------------------------- texture descriptor decode
//
// Split out of ResolveTexture so the PURE half can be cached without touching the texture cache.
//
// Purity, checked rather than assumed: this reads only `resource` (six fields: dimension, kind,
// mip_mode, r128, read, written) and the eight descriptor dwords. It calls
// TextureGetSurfaceFormatInfo, PopulateTextureMipLayout and TextureViewInfo, none of which read
// Config, renderer context or any mutable global. There is therefore NO cache state, NO generation
// and NO invalidation: a decode result for given inputs is valid for the life of the process, and
// this deliberately does not become another invalidation framework.
//
// Everything cache-dependent stays in ResolveTexture and still runs on every call: FindImage
// (overlap reconciliation, LRU touch, tick_accessed_last, resource creation, PrepareStorage-
// SampledOverlap) and FindTexture (refresh, page-watcher re-arm, DCC clear, view creation).
void DecodeTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                   const ShaderRecompiler::IR::DescriptorValue& value, DecodedTexture& out) {
	// Reset FIRST, before anything else touches `out`.
	//
	// This function writes into a destination the caller may reuse - the flag-off production path
	// decodes into one persistent thread_local scratch - and NOT every field is assigned on every
	// path. `is_null` is set only in the null branch below and never cleared, so without this a
	// real texture decoded after a null one on the same thread inherits is_null = true. The caller
	// then takes the null branch for a real texture: the wrong FindImage overload (exact_format
	// defaulted), no depth_id remap, and none of the depth/storage view validation. Skipping the
	// depth_id remap is what produced the abort
	// "TextureCache: texture requires rediscovery before final acquisition" - FindTextureLocked
	// rejects an id whose image still has depth_id set, which is exactly what the remap resolves.
	//
	// The original code had no such hazard because it built a fresh value-initialised
	// DecodedTexture-equivalent per call. Resetting restores that exactly.
	out = {};
	ShaderTextureResource descriptor;
	CopyNativeDescriptor(value, descriptor.fields);
	const bool storage = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
	}

	if (descriptor.IsNull()) {
		out.is_null = true;
		out.desc    = NullTextureDesc(resource, storage ? TextureCache::BindingType::Storage
		                                                : TextureCache::BindingType::Texture);
		return;
	}

	const auto address      = descriptor.Base40();
	const auto width        = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto height       = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto base_level   = descriptor.BaseLevel();
	const auto last_level   = descriptor.LastLevel();
	const auto type         = TextureType(descriptor);
	const bool multisampled = IsMultisampledTexture(type);
	const auto max_mip      = resource.r128 ? last_level : descriptor.MaxMip();
	const auto levels       = multisampled ? 1u : static_cast<uint32_t>(max_mip) + 1u;
	const bool dynamic_storage =
	    storage && resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::DynamicStorage;
	const auto view_last_level =
	    !multisampled && !dynamic_storage ? std::min(last_level, max_mip) : last_level;
	const auto tile       = descriptor.TileMode();
	const bool depth_tile = tile == Prospero::TileMode::kDepth;
	const bool msaa_tile  = depth_tile || tile == Prospero::TileMode::kRenderTarget;
	const bool msaa_array = type == Prospero::ImageType::kColor2DMsaaArray;
	if ((!multisampled && (base_level > view_last_level || view_last_level >= levels)) ||
	    (multisampled &&
	     (base_level != 0 || last_level == 0 || last_level > 3 || max_mip != last_level ||
	      !msaa_tile || (descriptor.MsaaDepth() && !depth_tile) ||
	      (!msaa_array && (descriptor.Depth() != 0 || descriptor.BaseArray5() != 0))))) {
		EXIT("unsupported texture mip view: base=%u last=%u levels=%u max=%u type=%u tile=%u "
		     "kind=%u dimension=%u mip_mode=%u read=%d written=%d "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     base_level, last_level, levels, descriptor.MaxMip(),
		     static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(tile),
		     static_cast<uint32_t>(resource.kind), static_cast<uint32_t>(resource.dimension),
		     static_cast<uint32_t>(resource.mip_mode), resource.read, resource.written,
		     descriptor.fields[0], descriptor.fields[1], descriptor.fields[2], descriptor.fields[3],
		     descriptor.fields[4], descriptor.fields[5], descriptor.fields[6],
		     descriptor.fields[7]);
	}
	const auto samples = multisampled ? 1u << last_level : 1u;
	const auto view_levels =
	    multisampled ? 1u : static_cast<uint32_t>(view_last_level - base_level) + 1u;
	const auto depth          = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	const auto format         = descriptor.Format();
	const auto surface_format = TextureGetSurfaceFormatInfo(format);
	const bool shader_conversion =
	    surface_format.conversion_format != Prospero::BufferFormat::kInvalid;
	const bool sampled_numeric_class =
	    storage || (Prospero::IsSampledTextureFormat(format) &&
	                (resource.kind == ShaderRecompiler::IR::ResourceKind::ImageUint) ==
	                    Prospero::IsUintTextureFormat(format));
	if (!storage &&
	    (resource.kind == ShaderRecompiler::IR::ResourceKind::Image ||
	     resource.kind == ShaderRecompiler::IR::ResourceKind::ImageUint) &&
	    !sampled_numeric_class) {
		EXIT("sampled image numeric class mismatch: kind=%u format=%u addr=0x%016" PRIx64 "\n",
		     static_cast<uint32_t>(resource.kind), static_cast<uint32_t>(format), address);
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	const auto    image_layers = layered ? depth : 1u;
	uint32_t      pitch        = 0;
	TileSizeAlign size {};
	if (multisampled) {
		const auto bytes = Prospero::NumBytesPerElement(format);
		pitch            = depth_tile ? TileGetDepthPitch(width, bytes, last_level)
		                              : TileGetRenderTargetPitch(width, bytes, last_level);
		if (pitch == 0 || !TileGetRenderTargetSize(width, height, pitch, bytes, size, last_level) ||
		    size.size > UINT32_MAX / image_layers) {
			EXIT("unsupported multisample texture layout\n");
		}
		size.size *= image_layers;
	} else {
		pitch = TileGetTexturePitch(format, width, tile);
		TileGetTextureTotalSize(format, width, height, volume ? depth : image_layers, levels, tile,
		                        volume, size);
	}
	EXIT_NOT_IMPLEMENTED(size.size == 0 || size.align == 0 ||
	                     (address & (static_cast<uint64_t>(size.align) - 1u)) != 0);
	if (storage) {
		ValidateStorageTexture(resource, descriptor, size.size);
	}

	const auto pixel_format        = surface_format.vk_format;
	const auto storage_view_format = storage && format == Prospero::BufferFormat::k32SInt
	                                     ? vk::Format::eR32Uint
	                                     : SrgbStorageViewFormat(pixel_format);
	const auto view_format         = storage && storage_view_format != vk::Format::eUndefined
	                                     ? storage_view_format
	                                     : pixel_format;
	const auto block_bytes         = Prospero::BlockCompressedBytesPerBlock(format);
	auto& desc = out.desc;
	desc.info.data         = {address, size.size};
	desc.info.pixel_format = pixel_format;
	desc.info.guest_format = format;
	desc.info.type         = TextureBaseType(type);
	desc.info.extent       = {width, height, volume ? depth : 1u};
	desc.info.resources    = {levels, image_layers};
	desc.info.pitch        = pitch;
	desc.info.bytes_per_block =
	    block_bytes != 0 ? block_bytes : Prospero::NumBytesPerElement(format);
	desc.info.samples   = samples;
	desc.info.tile_mode = tile;
	if (samples > 1) {
		desc.info.mip_layout[0] = {0, size.size, pitch, height};
	} else {
		PopulateTextureMipLayout(desc.info);
	}
	desc.view_info = TextureViewInfo(resource, descriptor, view_format, shader_conversion, storage,
	                                 view_levels, desc.info.resources.layers);
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;
	out.descriptor        = descriptor;
	out.pixel_format      = pixel_format;
	out.view_format       = view_format;
	out.size_bytes        = size.size;
	out.storage           = storage;
	out.shader_conversion = shader_conversion;
}


// Test seam. Decoding is a pure function of the resource fields and descriptor dwords, so this is
// simply the decode with no wrapper - it exists so the differential test can call it without
// standing up a RenderExecutor.
void DecodeTextureUncachedForTest(const ShaderRecompiler::IR::ImageResource&   resource,
                                  const ShaderRecompiler::IR::DescriptorValue& value,
                                  DecodedTexture&                              out) {
	DecodeTexture(resource, value, out);
}

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value) {
	// A FRESH local per call, deliberately. An earlier version decoded into a persistent
	// thread_local scratch; because `is_null` is written only on the null path, a real texture
	// decoded after a null one inherited it, skipped the depth_id remap, and aborted with
	// "texture requires rediscovery before final acquisition". DecodeTexture also resets `out`,
	// so this is belt and braces on purpose.
	DecodedTexture decoded;
	DecodeTexture(resource, value, decoded);

	auto& texture_cache = m_context.GetTextureCache();
	// COPY, never a reference into the cache entry: FindImage takes ImageDesc& and mutates it
	// (ValidateImageDesc, overlap merging). Handing it the cached object would corrupt the entry
	// for every later hit.
	auto desc = decoded.desc;
	if (decoded.is_null) {
		const auto id = texture_cache.FindImage(desc);
		return {id, nullptr, std::move(desc)};
	}
	const auto& descriptor        = decoded.descriptor;
	const auto  storage           = decoded.storage;
	const auto  pixel_format      = decoded.pixel_format;
	const auto  view_format       = decoded.view_format;
	const auto  shader_conversion = decoded.shader_conversion;
	const auto  size_bytes        = decoded.size_bytes;


	auto id = texture_cache.FindImage(desc, shader_conversion);

	if (!id) {
		// A worker bailed inside FindImage rather than create an image. Return the empty binding
		// immediately: the draw is already flagged for serial retry and its prepared entry will be
		// discarded, but indexing the slot vector with an empty id aborts, which is how this
		// surfaced. Every consumer of image_id below is guarded the same way.
		return {id, nullptr, std::move(desc)};
	}
	auto*      image               = &texture_cache.GetImage(id);
	const bool stencil_association = static_cast<bool>(image->depth_id);
	if (stencil_association) {
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
	} else if (image->info.IsDepth()) {
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		ValidateDepthTargetBinding(resource, descriptor, image, pixel_format, size_bytes);
		(void)SelectSampledDepthView(image->info.pixel_format, pixel_format,
		                             descriptor.DstSelXYZW());
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, view_format, descriptor.DstSelXYZW());
	} else {
		(void)SelectSampledColorView(image->info.pixel_format, pixel_format,
		                             descriptor.DstSelXYZW());
	}
	return {id, nullptr, std::move(desc)};
}

static vk::Sampler NativeSampler(RenderContext&                       context,
                                 const ShaderRecompiler::IR::Program& program, uint32_t index,
                                 const ShaderRecompiler::IR::DescriptorValue& value) {
	ShaderSamplerResource descriptor;
	CopyNativeDescriptor(value, descriptor.fields);
	const bool depth_compare = std::any_of(program.info.sampled_pairs.begin(),
	                                       program.info.sampled_pairs.end(), [&](const auto& pair) {
		                                       return pair.sampler == index &&
		                                              pair.image < program.info.images.size() &&
		                                              program.info.images[pair.image].depth_compare;
	                                       });
	if (!depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
	}
	if (program.info.samplers[index].force_point_filtering) {
		descriptor.SetPointFiltering();
	}
	return context.GetSamplerCache().GetSampler(descriptor);
}

static BufferView NativeUpload(RenderContext& context, std::span<const uint32_t> data) {
	EXIT_IF(data.empty());
	// Assertion only - nothing is recorded into the primary here, and the stream ring this writes
	// to is already per worker. A worker resolving between submissions, where Current() is
	// legitimately invalid, is not an error. Same class as the ObtainBuffer precondition.
	EXIT_IF(!MustStageForWorker() && context.GetCommandScheduler().Current().IsInvalid());
	auto&      buffer = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset = buffer.Copy(data.data(), data.size_bytes(), 256);
	return {.buffer = buffer.Handle(), .offset = offset, .range = data.size_bytes()};
}

void RenderExecutor::TrackImageBinding(ImageId id) {
	EXIT_IF(m_context.GetTextureCache().m_slot_images.try_get(id) == nullptr);
	if (std::ranges::find(m_bound_images, id) == m_bound_images.end()) {
		m_bound_images.push_back(id);
	}
}

void RenderExecutor::BindImage(ImageId id, bool storage) {
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.info.data.Empty()) {
		return;
	}
	if (image.binding.is_bound) {
		image.binding.force_general |= image.binding.shader_write != storage;
	}
	image.binding.is_bound = true;
	image.binding.shader_write |= storage;
	TrackImageBinding(id);
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto& image             = m_context.GetTextureCache().GetImage(id);
	image.binding.is_target = true;
	TrackImageBinding(id);
}

void RenderExecutor::ResetBindings() {
	for (const auto id: m_bound_images) {
		if (auto* image = m_context.GetTextureCache().m_slot_images.try_get(id); image != nullptr) {
			image->binding = {};
		}
	}
	m_bound_images.clear();
}

namespace {

// Vector-buffer recycling for PreparedBindings.
//
// PrepareBindings runs twice per draw and each call allocates ~9 vectors, so ~18 heap
// allocations per draw. The objects themselves keep value semantics (an earlier attempt at
// pooling the whole object let pointers escape and broke tests) - only the heap buffers are
// recycled, by swapping them out before destruction and back in on the next call.
struct BindingStorage {
	std::vector<BufferView>     buffers;
	std::vector<TextureBinding> images;
	std::vector<vk::Sampler>    samplers;
	std::vector<BufferId>       buffer_ids;
	std::vector<ShaderBufferResource> buffer_descriptors;
	std::vector<uint64_t>             buffer_sizes;
	std::vector<uint32_t>       flattened_srt;
	std::vector<uint32_t>       user_data;
};

// Sized for SERIAL draws: one stage object was live at a time. Batched acquisition builds every
// draw's bindings before recording consumes any, so up to 2 * kMaxBatchDraws coexist while the
// pool retained 8 - the rest of every batch allocated fresh and was then dropped on return.
//
// Measured offline against the real functions at the observed 73.9 draws/batch:
//   1120 allocations and 138,880 bytes per drain, 0.0901 ms.
// At 4350 draws/frame that is ~58.9 drains, so ~66,000 allocations and ~5.3 ms per frame.
//
// Cap now covers a full batch of both stages. Steady-state retention measured at 512 slots is
// ~488 KB, which is the price of not re-allocating them every drain.
constexpr size_t kStoragePoolCap = 2 * DrawBatchQueue::kMaxBatchDraws;

// One pathological draw must not pin a huge buffer in the pool forever. A slot whose vectors
// have grown past this is released rather than retained; the common small shapes are unaffected.
constexpr size_t kStoragePoolMaxRetainedElements = 4096;

std::vector<BindingStorage>& StoragePool() {
	static thread_local std::vector<BindingStorage> pool;
	return pool;
}

} // namespace

// Seed a fresh PreparedBindings with recycled buffers (empty, but with capacity).
void TakePooledStorage(PreparedBindings& prepared) {
	auto& pool = StoragePool();
	if (pool.empty()) {
		return;
	}
	auto slot = std::move(pool.back());
	pool.pop_back();
	slot.buffers.clear();
	slot.images.clear();
	slot.samplers.clear();
	slot.buffer_ids.clear();
	slot.buffer_descriptors.clear();
	slot.buffer_sizes.clear();
	slot.flattened_srt.clear();
	slot.user_data.clear();
	prepared.resources.buffers  = std::move(slot.buffers);
	prepared.resources.images   = std::move(slot.images);
	prepared.resources.samplers = std::move(slot.samplers);
	prepared.buffer_ids         = std::move(slot.buffer_ids);
	prepared.buffer_descriptors = std::move(slot.buffer_descriptors);
	prepared.buffer_sizes       = std::move(slot.buffer_sizes);
	prepared.flattened_srt      = std::move(slot.flattened_srt);
	prepared.user_data          = std::move(slot.user_data);
}

size_t PreparedBindingsPoolSize() {
	return StoragePool().size();
}

size_t PreparedBindingsPoolCap() {
	return kStoragePoolCap;
}

void ClearPreparedBindingsPool() {
	StoragePool().clear();
}

void ReturnPooledBindingStorage(PreparedBindings& prepared) {
	auto& pool = StoragePool();
	if (pool.size() >= kStoragePoolCap) {
		return;
	}
	// Oversized buffers are dropped rather than pooled: retention is bounded by cap AND by size.
	const auto too_big = [](size_t capacity) {
		return capacity > kStoragePoolMaxRetainedElements;
	};
	if (too_big(prepared.resources.buffers.capacity()) ||
	    too_big(prepared.resources.images.capacity()) ||
	    too_big(prepared.buffer_descriptors.capacity()) ||
	    too_big(prepared.flattened_srt.capacity()) || too_big(prepared.user_data.capacity())) {
		return;
	}
	BindingStorage slot;
	slot.buffers       = std::move(prepared.resources.buffers);
	slot.images        = std::move(prepared.resources.images);
	slot.samplers      = std::move(prepared.resources.samplers);
	slot.buffer_ids    = std::move(prepared.buffer_ids);
	slot.buffer_descriptors = std::move(prepared.buffer_descriptors);
	slot.buffer_sizes       = std::move(prepared.buffer_sizes);
	slot.flattened_srt = std::move(prepared.flattened_srt);
	slot.user_data     = std::move(prepared.user_data);
	pool.push_back(std::move(slot));
}

// Coverage and cost of the worker image-resolution slice. Thread_local, no atomics, so
// measuring does not perturb what it measures.
struct WorkerImageCensus {
	uint64_t slots        = 0;   // image slots seen by the caller
	uint64_t used         = 0;   // served from a worker resolve
	uint64_t ticketed     = 0;   // worker could not resolve (creation needed): caller resolved
	uint64_t stale        = 0;   // worker resolved, but the cache moved: caller re-resolved
	uint64_t not_attempted = 0;  // no worker pass ran for this draw

	void Report() const {
		if (slots == 0 || (slots % 2000000) != 0) {
			return;
		}
		std::printf("WorkerImages: slots=%llu used=%llu (%.1f%%) ticketed=%llu (%.1f%%) "
		            "stale=%llu (%.1f%%) no-worker-pass=%llu (%.1f%%)\n",
		            static_cast<unsigned long long>(slots),
		            static_cast<unsigned long long>(used), 100.0 * double(used) / double(slots),
		            static_cast<unsigned long long>(ticketed),
		            100.0 * double(ticketed) / double(slots),
		            static_cast<unsigned long long>(stale), 100.0 * double(stale) / double(slots),
		            static_cast<unsigned long long>(not_attempted),
		            100.0 * double(not_attempted) / double(slots));
		std::fflush(stdout);
	}
};

WorkerImageCensus& ImageCensus() {
	static thread_local WorkerImageCensus census;
	return census;
}

// Phase 2a2, on a WORKER. Resolution only.
//
// What is deliberately NOT done here: BindImage (mutates image.binding and pushes to the shared
// m_bound_images), image creation (FindImage bails to the caller at its creation point),
// barriers, and recording. Those stay on the caller in guest order.
//
// A miss ticketed one slot; every other slot's result is retained, so a draw that needs one new
// image does not replay its whole resolution on the caller.
void RenderExecutor::PreResolveQueuedImages(PreparedShaders& prepared) {
	if (!Config::WorkerResolveImages() || !prepared.valid) {
		return;
	}
	auto&      cache      = m_context.GetTextureCache();
	const auto generation = cache.ImageInvalidationGeneration();
	auto       run        = [&](const ShaderStageRuntime& rt,
                        PreparedShaders::PreResolvedImages& out) {
		if (!rt) {
			return;
		}
		const auto& program  = *rt.program;
		const auto& snapshot = *rt.resources;
		const auto  count    = program.info.images.size();
		out.bindings.assign(count, TextureBinding {});
		out.ok.assign(count, 0);
		out.generation = generation;
		out.attempted  = true;
		for (uint32_t i = 0; i < count; i++) {
			// Clear any stale flag first so this slot's outcome is its own.
			(void)TakeWorkerBailout();
			auto binding = ResolveTexture(program.info.images[i], snapshot.images[i]);
			if (TakeWorkerBailout() || !binding.image_id) {
				continue;   // ticketed: the caller resolves this slot
			}
			out.bindings[i] = std::move(binding);
			out.ok[i]       = 1;
		}
	};
	run(prepared.vs_input_info.stage, prepared.vertex_images);
	if (prepared.ps_active) {
		run(prepared.ps_input_info.stage, prepared.pixel_images);
	}
	// The bailout flag must not leak into the phase-2a caller, which reads it to invalidate the
	// whole draw. Ticketing already recorded every miss.
	(void)TakeWorkerBailout();
}

PreparedBindings RenderExecutor::PrepareBindings(const ShaderStageRuntime&                 runtime,
                                                 const PreparedShaders::PreResolvedImages* pre) {
	KYTY_PROFILER_FUNCTION();
	DrawPhaseTimer draw_phase_timer(DrawPhase::BindPrepare);
	EXIT_IF(!runtime);
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	// ShaderMaterializeStageRuntime already validated this exact pair before publishing it,
	// and both halves are immutable shared_ptr<const> from that point on - so this re-runs a
	// decision that cannot have changed. Kept under the same gate as the other per-draw
	// validation rather than deleted outright.
	if (Config::HwCheckEnabled()) {
		std::string error;
		if (!ShaderRecompiler::IR::ValidateResourceSpecialization(program, snapshot, &error)) {
			EXIT("invalid native shader runtime snapshot: %s\n", error.c_str());
		}
	}

	PreparedBindings prepared;
	TakePooledStorage(prepared);
	prepared.program  = runtime.program.get();
	prepared.snapshot = runtime.resources.get();
	auto& descriptors = prepared.resources;
	descriptors.buffers.reserve(program.info.buffers.size());
	descriptors.images.reserve(program.info.images.size());
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		TextureBinding binding;
		{
			DrawPhaseTimer resolve_timer(DrawPhase::BindResolveTexture);
			auto&      census        = ImageCensus();
			auto&      texture_cache = m_context.GetTextureCache();
			const bool have_pre      = pre != nullptr && pre->attempted && i < pre->ok.size();
			census.slots++;
			if (pre == nullptr || !pre->attempted) {
				census.not_attempted++;
			}
			// Revalidation: a worker's handle is trusted only if nothing since could have
			// retired it. Any insertion or free - including one caused by an earlier slot of
			// THIS draw - bumps the generation, and then the handle is re-resolved.
			if (have_pre && pre->ok[i] != 0 &&
			    pre->generation == texture_cache.ImageInvalidationGeneration()) {
				binding = pre->bindings[i];
				census.used++;
			} else {
				if (have_pre && pre->ok[i] != 0) {
					census.stale++;
				} else if (have_pre) {
					census.ticketed++;
				}
				binding = ResolveTexture(program.info.images[i], snapshot.images[i]);
			}
			census.Report();
		}
		// An empty id means a worker bailed rather than create the image; the draw is already
		// flagged for serial retry. Skip it instead of indexing the slot vector with an empty id.
		if (binding.image_id) {
			DrawPhaseTimer bind_timer(DrawPhase::BindBindImage);
			BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
		}
		descriptors.images.push_back(binding);
	}
	descriptors.samplers.reserve(program.info.samplers.size());
	for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
		DrawPhaseTimer sampler_timer(DrawPhase::BindNativeSampler);
		descriptors.samplers.push_back(NativeSampler(m_context, program, i, snapshot.samplers[i]));
	}
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::FlattenedSrt) !=
	    nullptr) {
		prepared.flattened_srt.assign(snapshot.flattened_srt.begin(), snapshot.flattened_srt.end());
	}

	prepared.user_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.bindings.user_data_registers) {
		prepared.user_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.user_data.resize(program.bindings.ShaderDataDwords());
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::Gds) != nullptr) {
		descriptors.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
	return prepared;
}

void RenderExecutor::FindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	// Re-resolving ids invalidates any previous publication.
	prepared.published = false;
	DrawPhaseTimer draw_phase_timer(DrawPhase::BindFindBuffers);
	EXIT_IF(prepared.program == nullptr || prepared.snapshot == nullptr);
	const auto& program  = *prepared.program;
	const auto& snapshot = *prepared.snapshot;
	auto&       cache    = m_context.GetBufferCache();

	prepared.buffer_ids.clear();
	prepared.buffer_ids.reserve(program.info.buffers.size());
	prepared.buffer_descriptors.clear();
	prepared.buffer_descriptors.resize(program.info.buffers.size());
	prepared.buffer_sizes.clear();
	prepared.buffer_sizes.resize(program.info.buffers.size());
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		ShaderBufferResource& descriptor = prepared.buffer_descriptors[i];
		CopyNativeDescriptor(snapshot.buffers[i], descriptor.fields);
		const auto address = descriptor.Base48();
		const auto stride  = descriptor.Stride();
		const auto records = descriptor.NumRecords();
		EXIT_IF(stride != 0 && records > UINT64_MAX / stride);
		const auto requested_size = stride != 0 ? static_cast<uint64_t>(stride) * records : records;
		if (address == 0 || requested_size == 0) {
			prepared.buffer_ids.emplace_back();
			continue;
		}
		uint64_t size = 0;
		{
			// Takes a mutex and binary-searches the guest range table, per buffer per stage per
			// draw. Timed separately to see how much of FindBuffers it actually is.
			DrawPhaseTimer clamp_timer(DrawPhase::BindClampRange);
			size = Libs::LibKernel::Memory::ClampRangeSize(address, requested_size);
		}
		g_draw_profile.buffers++;
		prepared.buffer_sizes[i] = size;
		prepared.buffer_ids.push_back(cache.FindBuffer(address, size));
	}

}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	// Rebinding rebuilds the views and publish records, so anything published earlier is
	// stale from here until PublishBuffers runs again.
	prepared.published = false;
	DrawPhaseTimer draw_phase_timer(DrawPhase::BindRebindBuffers);
	EXIT_IF(prepared.program == nullptr || prepared.snapshot == nullptr);
	const auto& program   = *prepared.program;
	auto&       resources = prepared.resources;
	const auto& layout    = program.bindings;
	EXIT_IF(prepared.buffer_ids.size() != program.info.buffers.size());

	resources.buffers.clear();
	resources.buffers.reserve(program.info.buffers.size());
	EXIT_IF(prepared.user_data.size() != layout.ShaderDataDwords());
	std::fill(prepared.user_data.begin() + layout.memory_offset_dword, prepared.user_data.end(), 0);
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.user_data[dword] |= offset << shift;
	};
	EXIT_IF(prepared.buffer_descriptors.size() != program.info.buffers.size() ||
	        prepared.buffer_sizes.size() != program.info.buffers.size());
	{
		// The per-buffer resolve. Deduped within a draw by FindDrawBufferCache, but the cache is
		// cleared every draw, so a buffer bound by 4000 consecutive draws is resolved 4000 times.
		//
		// KNOWN, PRE-EXISTING, NOT FIXED HERE: this loop captures a handle, offset and device
		// address per buffer, and a later range-changing operation can retire the buffer one of
		// them points at. The old handle stays usable (retirement is deferred past GPU completion)
		// but stops being current. A stabilise-and-refresh loop was tried HERE and removed: it is
		// the wrong place. RebindImages runs AFTER both RebindBuffers calls
		// (ResolveTexture -> FindImage -> InitializeImage -> ObtainBufferForImage -> CreateBuffer),
		// and the packed memory offsets are already committed to a stream buffer at the end of
		// this function, so refreshing here cannot cover either. See StaleBindingTests.cpp.
		DrawPhaseTimer native_buffers_timer(DrawPhase::BindNativeBuffers);
		prepared.buffer_publish.assign(program.info.buffers.size(), {});
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const ShaderBufferResource& descriptor = prepared.buffer_descriptors[i];
			uint32_t buffer_offset = 0;
			resources.buffers.push_back(NativeStorageBuffer(
			    m_context, descriptor, program.info.buffers[i], program.stage, i, buffer_offset,
			    prepared.buffer_ids[i], prepared.buffer_sizes[i],
			    &prepared.buffer_publish[i]));
			pack_memory_offset(i, buffer_offset);
		}
	}
	// The user_data upload used to happen here. It carries the packed memory offsets, so it has
	// to come from the SETTLED state - see PublishBuffers.
}

// Re-derives every cache-backed binding from its guest range and republishes the descriptor, the
// packed offset and the device address from one settled snapshot, then uploads user_data from
// that same snapshot.
//
// Runs after ALL range-changing work for the draw: both stages' buffers AND both stages' images.
// RebindImages reaches CreateBuffer through
// ResolveTexture -> FindImage -> InitializeImage -> ObtainBufferForImage, so a per-stage refresh
// inside RebindBuffers cannot cover it.
//
// This performs NO obtain-side work: no allocation, no join, no synchronize, no upload into a
// buffer, no dirty-state change, no LRU touch. Those effects happened once, in order, during
// RebindBuffers, and must not be repeated.
void RenderExecutor::PublishBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.program == nullptr);
	prepared.published = true;
	const auto& program   = *prepared.program;
	auto&       resources = prepared.resources;
	const auto& layout    = program.bindings;

	if (Config::BindingPublishEnabled() &&
	    prepared.buffer_publish.size() == resources.buffers.size()) {
		auto& cache = m_context.GetBufferCache();
		std::fill(prepared.user_data.begin() + layout.memory_offset_dword,
		          prepared.user_data.end(), 0);
		for (uint32_t i = 0; i < prepared.buffer_publish.size(); i++) {
			const auto& entry = prepared.buffer_publish[i];
			if (!entry.resolved) {
				continue;   // null binding: nothing was resolved for it
			}
			if (entry.stream) {
				// Verbatim. The ring allocation holds this binding's private copy of the guest
				// bytes; re-resolving by address would hand back a cache buffer instead and
				// discard it. Its offset and lifetime are already correct.
				const auto dword = layout.memory_offset_dword + i / 4u;
				const auto shift = (i % 4u) * 8u;
				prepared.user_data[dword] |= resources.buffers[i].publish_adjustment << shift;
				continue;
			}
			const auto owner = cache.FindPublishedOwner(entry.address, entry.size);
			if (owner.first == nullptr) {
				// Never silently allocate and never publish a view we could not confirm:
				// allocating here would itself be a range-changing operation, after the point
				// where everything was supposed to have settled.
				EXIT("binding publication found no owner for guest range 0x%016" PRIx64
				     " size 0x%" PRIx64 "\n",
				     entry.address, entry.size);
			}
			const auto alignment      = entry.alignment != 0 ? entry.alignment : 1;
			const auto aligned_offset = owner.second - owner.second % alignment;
			const auto adjustment     = static_cast<uint32_t>(owner.second - aligned_offset);
			// The same contract NativeStorageBuffer enforces on the obtain side, enforced here
			// too. The packed memory offset is an 8-BIT LANE (adjustment << (i % 4) * 8), so an
			// adjustment of 256 or more does not merely truncate - it spills into the
			// neighbouring binding's lane and silently moves THAT buffer's base. Obtain EXITs on
			// this; publication was packing whatever it computed.
			const auto max_range =
			    m_context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
			if (adjustment % sizeof(uint32_t) != 0 || adjustment >= 256 ||
			    entry.size > max_range - adjustment) {
				EXIT("binding publication produced an unsupported offset adjustment %u for guest "
				     "range 0x%016" PRIx64 " size 0x%" PRIx64 "\n",
				     adjustment, entry.address, entry.size);
			}
			auto&      view           = resources.buffers[i];
			// Does publication ever actually CHANGE a binding? If the owning buffer was not
			// retired, FindPublishedOwner returns the very buffer obtain used at the very same
			// offset, and everything below is a no-op. Counted so one run can say whether this
			// path is inert in practice instead of the question being argued from the source.
			if (Config::BufferCensusEnabled()) {
				static std::atomic<uint64_t> s_examined {0};
				static std::atomic<uint64_t> s_changed {0};
				const bool same = view.buffer == owner.first->Handle() &&
				                  view.offset == aligned_offset &&
				                  view.range == static_cast<vk::DeviceSize>(entry.size + adjustment);
				s_changed.fetch_add(same ? 0 : 1, std::memory_order_relaxed);
				const auto n = s_examined.fetch_add(1, std::memory_order_relaxed) + 1;
				if (n % 200000 == 0) {
					std::printf("PublishCensus: examined=200000 changed=%llu\n",
					            static_cast<unsigned long long>(
					                s_changed.exchange(0, std::memory_order_relaxed)));
					std::fflush(stdout);
				}
			}
			view.buffer               = owner.first->Handle();
			view.offset               = aligned_offset;
			// range MUST be republished with the offset. The descriptor window is
			// [offset, offset + range), and a replacement usually places the guest address at a
			// different alignment remainder, so a stale range leaves the window shifted and
			// mis-sized - the shader then fetches vertex data from the wrong bytes.
			view.range                = static_cast<vk::DeviceSize>(entry.size + adjustment);
			view.publish_adjustment   = adjustment;
			const auto dword          = layout.memory_offset_dword + i / 4u;
			const auto shift          = (i % 4u) * 8u;
			prepared.user_data[dword] |= adjustment << shift;
		}
	}
	{
		// Two unconditional stream-buffer map+memcpy+commit per stage, 256-byte aligned, whether
		// or not the payload changed since the last draw. Uploaded HERE so the offsets it carries
		// match the handles published above.
		DrawPhaseTimer native_upload_timer(DrawPhase::BindNativeUpload);
		if (!prepared.flattened_srt.empty()) {
			resources.flattened_srt = NativeUpload(m_context, prepared.flattened_srt);
		}
		if (ShaderRecompiler::IR::FindBinding(
		        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::UserData) !=
		    nullptr) {
			resources.user_data = NativeUpload(m_context, prepared.user_data);
		}
	}
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	// Image resolution reaches CreateBuffer through ResolveTexture -> FindImage ->
	// InitializeImage -> ObtainBufferForImage, so it can retire a buffer an earlier publish
	// resolved. Clearing only in the buffer methods was not sufficient.
	prepared.published = false;
	DrawPhaseTimer draw_phase_timer(DrawPhase::BindRebindImages);
	EXIT_IF(prepared.program == nullptr || prepared.snapshot == nullptr);
	const auto& program  = *prepared.program;
	const auto& snapshot = *prepared.snapshot;
	auto&       images   = prepared.resources.images;
	EXIT_IF(images.size() != program.info.images.size());
	auto& texture_cache = m_context.GetTextureCache();
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto old_image = texture_cache.m_slot_images.try_get(images[i].image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			images[i] = ResolveTexture(program.info.images[i], snapshot.images[i]);
			if (images[i].image_id) {
				BindImage(images[i].image_id,
				          images[i].desc.type == TextureCache::BindingType::Storage);
			}
		}
	}
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		auto& binding = images[i];
		binding.mip_views.clear();
		// A worker bailed inside FindImage rather than create this image. Everything below indexes
		// the slot vector with the id, so skip the whole entry; the draw is already flagged for
		// serial retry and its bindings are discarded.
		if (!binding.image_id) {
			continue;
		}
		// --image-census: is this image about to be sampled with contents nothing ever wrote?
		//
		// A VkImage whose backing was never uploaded, copied, resolved, cleared or written by a
		// GPU pass has undefined contents; in practice it samples as a flat colour, which is what
		// "the asset went black" looks like. This does not guess at a cause - it answers whether
		// the black surface is an EMPTY IMAGE at all, or a correctly-filled image being shaded
		// black. Those need completely different fixes, and nothing so far distinguishes them.
		if (Config::ImageCensusEnabled()) {
			static std::atomic<uint64_t> s_binds {0};
			static std::atomic<uint64_t> s_empty {0};
			static std::atomic<uint32_t> s_logged {0};
			if (const auto* img = texture_cache.m_slot_images.try_get(binding.image_id);
			    img != nullptr && img->backing.image != nullptr) {
				s_binds.fetch_add(1, std::memory_order_relaxed);
				if (!img->IsHostInitialized()) {
					s_empty.fetch_add(1, std::memory_order_relaxed);
					if (s_logged.fetch_add(1, std::memory_order_relaxed) < 40) {
						std::printf("ImageCensus EMPTY: guest=0x%016llx size=0x%llx fmt=%u "
						            "extent=%ux%ux%u mips=%u layers=%u rt=%d depth=%d storage=%d "
						            "tex=%d cpu_dirty=%d buf_mod=%d gpu_mod=%d\n",
						            static_cast<unsigned long long>(img->info.data.address),
						            static_cast<unsigned long long>(img->info.data.size),
						            static_cast<uint32_t>(img->info.pixel_format),
						            img->info.extent.width, img->info.extent.height,
						            img->info.extent.depth, img->info.resources.levels,
						            img->info.resources.layers,
						            static_cast<int>(img->usage.render_target),
						            static_cast<int>(img->usage.depth_target),
						            static_cast<int>(img->usage.storage),
						            static_cast<int>(img->usage.texture),
						            static_cast<int>(img->IsDefinitelyCpuDirty()),
						            static_cast<int>(img->IsBufferModified()),
						            static_cast<int>(img->IsGpuModified()));
						std::fflush(stdout);
					}
				}
			}
			const auto n = s_binds.load(std::memory_order_relaxed);
			if (n != 0 && n % 200000 == 0) {
				std::printf("ImageCensus: sampled_binds=%llu empty=%llu (%.3f%%)\n",
				            static_cast<unsigned long long>(n),
				            static_cast<unsigned long long>(s_empty.load(std::memory_order_relaxed)),
				            100.0 * static_cast<double>(s_empty.load(std::memory_order_relaxed)) /
				                static_cast<double>(n));
				std::fflush(stdout);
			}
		}
		const auto& resource = program.info.images[i];
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::DynamicStorage) {
			EXIT_IF(resource.mip_count == 0u ||
			        resource.mip_count != binding.desc.view_info.level_count);
			binding.mip_views.reserve(resource.mip_count);
			for (uint32_t mip = 0; mip < resource.mip_count; mip++) {
				auto desc = binding.desc;
				desc.view_info.base_level += mip;
				desc.view_info.level_count = 1;
				binding.mip_views.push_back(texture_cache.FindTexture(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else {
			auto desc = binding.desc;
			if (desc.type == TextureCache::BindingType::Storage) {
				desc.view_info.level_count = 1;
			}
			binding.image_view = texture_cache.FindTexture(binding.image_id, desc);
		}
		auto&      image   = texture_cache.GetImage(binding.image_id);
		const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
		// Sticky set-only flags: only ever raised, so a plain store is the whole operation.
		if (storage) {
			image.usage.storage.store(true, std::memory_order_relaxed);
		} else {
			image.usage.texture.store(true, std::memory_order_relaxed);
		}
	}
}

GraphicsBindings
RenderExecutor::PrepareGraphicsBindings(const ShaderStageRuntime& vertex,
                                        const ShaderStageRuntime& pixel, bool pixel_active) {
	auto bindings = AcquireGraphicsBindings(vertex, pixel, pixel_active);
	BindGraphicsResources(bindings);
	return bindings;
}

// Phase 2b - ACQUISITION. Main thread only.
//
// Everything here can create. PrepareBindings reaches ResolveTexture -> FindImage, which inserts
// and expands images and uploads them; FindBuffers calls FindBuffer -> CreateBuffer, which records
// a CopyFrom into the primary. That is what aborted every earlier attempt to run the whole resolve
// on workers - three times, each on a different route.
//
// The split is not arbitrary: PrepareBindings and FindBuffers ARE the acquisition half, and
// RebindBuffers/RebindImages ARE the binding half, so the existing boundaries already drew the line.
GraphicsBindings RenderExecutor::AcquireGraphicsBindings(
    const ShaderStageRuntime& vertex, const ShaderStageRuntime& pixel, bool pixel_active,
    const PreparedShaders::PreResolvedImages* vertex_pre,
    const PreparedShaders::PreResolvedImages* pixel_pre) {
	EXIT_IF(MustStageForWorker());
	// One draw's worth of buffer resolutions; see BeginDrawBufferScope.
	BeginDrawBufferScope();
	GraphicsBindings bindings {
	    .vertex = PrepareBindings(vertex, vertex_pre),
	};
	if (pixel_active) {
		bindings.pixel.emplace(PrepareBindings(pixel, pixel_pre));
	}
	FindBuffers(bindings.vertex);
	if (bindings.pixel) {
		FindBuffers(*bindings.pixel);
	}
	// PrepareBda mutates shared GPU resource state, so it belongs on this side too. It used to bail
	// a worker out; now it simply runs where it is safe.
	if (bindings.vertex.program->info.uses_dma ||
	    (bindings.pixel && bindings.pixel->program->info.uses_dma)) {
		m_context.GetGpuResources().PrepareBda();
	}
	return bindings;
}

// Phase 2c - BINDING. Safe on a worker.
//
// Every lookup below is guaranteed to hit because acquisition already created what was missing. The
// creation bail-outs inside FindImage, ExpandImage, CreateBuffer and ObtainBuffer stay as
// tripwires: if one fires now, acquisition missed something, and a loud abort naming the line is
// how the three previous routes were found.
// The single entry point both the graphics and compute paths use to finalize bindings.
//
// THREE PASSES OVER THE WHOLE SPAN, in this order, and never stage-at-a-time: finalizing one
// stage completely before starting the next is exactly the cross-stage invalidation bug - the
// second stage's rebinding can retire a buffer the first stage already published.
//
//   1. rebind buffers, every stage
//   2. rebind images,  every stage   (these reach CreateBuffer via ObtainBufferForImage)
//   3. publish,        every stage   (nothing after this changes ranges)
//
// It exists as one call so a caller cannot rebind without publishing. renderCompute did exactly
// that when publication owned the user_data upload, and every dispatch committed with unbound
// descriptors.
void RenderExecutor::FinalizeBindings(std::span<PreparedBindings* const> stages) {
	KYTY_PROFILER_FUNCTION();
	// Entry: no stage in this span carries publication state from an earlier draw.
	for (auto* stage: stages) {
		if (stage != nullptr) {
			stage->published = false;
		}
	}
	for (auto* stage: stages) {
		if (stage != nullptr) {
			RebindBuffers(*stage);
		}
	}
	for (auto* stage: stages) {
		if (stage != nullptr) {
			RebindImages(*stage);
		}
	}
	for (auto* stage: stages) {
		if (stage != nullptr) {
			PublishBuffers(*stage);
		}
	}
}

void RenderExecutor::BindGraphicsResources(GraphicsBindings& bindings) {
	PreparedBindings* stages[2] = {&bindings.vertex,
	                               bindings.pixel ? &*bindings.pixel : nullptr};
	const size_t      count     = bindings.pixel ? 2u : 1u;
	FinalizeBindings(std::span<PreparedBindings* const> {stages, count});
}

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings,
                                    vk::CommandBuffer                  record_target) {
	KYTY_PROFILER_FUNCTION();
	// Publication owns the user_data and flattened_srt uploads, so a stage that reaches commit
	// unpublished has unbound descriptors. Fail here, naming the cause, rather than at the
	// first null view several frames of confusion later.
	for (const auto* stage: prepared_bindings) {
		EXIT_IF(stage != nullptr && stage->program != nullptr && !stage->published);
	}
	auto   vk_buffer        = buffer.Handle();
	// Image layout transitions and barriers must stay on the primary, outside the render pass -
	// a secondary recorded inside one may not contain them. Only the binds follow the draw.
	const auto record = record_target ? record_target : vk_buffer;
	size_t descriptor_count = 0;
	size_t write_count      = 0;
	constexpr auto GraphicsStages =
	    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
	const auto push_constant_stages = pipeline_bind_point == vk::PipelineBindPoint::eGraphics
	                                      ? vk::ShaderStageFlags {GraphicsStages}
	                                      : vk::ShaderStageFlags {vk::ShaderStageFlagBits::eCompute};
	for (const auto* prepared: prepared_bindings) {
		EXIT_IF(prepared == nullptr || prepared->program == nullptr ||
		        prepared->snapshot == nullptr || prepared->committed);
		write_count += prepared->program->bindings.descriptors.size();
		for (const auto& binding: prepared->program->bindings.descriptors) {
			descriptor_count += NativeDescriptorCount(binding);
		}
		const auto shader_stage = NativeShaderStage(prepared->program->stage);
		EXIT_IF((pipeline_bind_point == vk::PipelineBindPoint::eGraphics &&
		         (shader_stage & GraphicsStages) == vk::ShaderStageFlags {}) ||
		        (pipeline_bind_point == vk::PipelineBindPoint::eCompute &&
		         shader_stage != vk::ShaderStageFlagBits::eCompute));
	}
	m_descriptor_buffers.clear();
	m_descriptor_images.clear();
	DescriptorWrites().clear();
	PushConstants().fill(0);
	m_descriptor_buffers.reserve(descriptor_count);
	m_descriptor_images.reserve(descriptor_count);
	DescriptorWrites().reserve(write_count);

	for (auto* prepared: prepared_bindings) {
		const auto& program       = *prepared->program;
		auto&       descriptors   = prepared->resources;
		const auto  shader_stage  = NativeShaderStage(program.stage);
		const auto  shader_stages = ShaderPipelineStages(shader_stage);
		if (descriptors.gds.buffer != nullptr) {
			buffer.EndRendering();
			const auto barrier = MakeGdsDependency(descriptors.gds.buffer);
			NoteBatchBoundary(BoundarySource::PipelineBarrier);
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			        vk::PipelineStageFlagBits::eAllGraphics |
			        vk::PipelineStageFlagBits::eComputeShader,
			    shader_stages, vk::DependencyFlags {}, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			// Empty id: a worker bailed inside FindImage. See RebindImages.
			if (!descriptors.images[i].image_id) {
				continue;
			}
			auto& image   = m_context.GetTextureCache().GetImage(descriptors.images[i].image_id);
			auto& binding = descriptors.images[i];
			const auto&                 view = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;

			// The four cases below decide one target layout and access mask. Computing them
			// first, then either transitioning now or staging for the pre-pass, keeps the two
			// paths from drifting - and lets the descriptor be written with the layout the image
			// will hold, which is what the shader sees either way.
			vk::ImageLayout                     target_layout = vk::ImageLayout::eGeneral;
			vk::AccessFlags2                    target_access {};
			std::optional<ImageSubresourceRange> target_range = range;
			if (image.info.data.Empty()) {
				target_access = storage ? vk::AccessFlagBits2::eShaderRead |
				                              vk::AccessFlagBits2::eShaderWrite
				                        : vk::AccessFlagBits2::eShaderRead;
			} else if ((image.binding.force_general || image.binding.is_target) &&
			           !image.info.IsDepth()) {
				const vk::AccessFlags2 storage_access = image.binding.shader_write
				                                            ? vk::AccessFlagBits2::eShaderWrite
				                                            : vk::AccessFlags2 {};
				target_access = vk::AccessFlagBits2::eShaderRead | storage_access |
				                vk::AccessFlagBits2::eColorAttachmentRead |
				                vk::AccessFlagBits2::eColorAttachmentWrite;
				target_range  = {};
			} else if (storage) {
				target_access = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
			} else {
				target_layout = image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                                     : vk::ImageLayout::eShaderReadOnlyOptimal;
				target_access = vk::AccessFlagBits2::eShaderRead;
			}

			if (Config::DeferTransitionsEnabled() || MustStageForWorker()) {
				m_pending_transitions.push_back(PendingTransition {
				    descriptors.images[i].image_id, target_layout, target_access, target_range});
				binding.layout = target_layout;
			} else {
				image.Transit(target_layout, target_access, target_range, vk_buffer);
				binding.layout = image.backing.state.layout;
			}
		}

		m_image_occurrences.assign(descriptors.images.size(), 0);
		for (const auto& binding: program.bindings.descriptors) {
			vk::WriteDescriptorSet write {};
			write.sType          = vk::StructureType::eWriteDescriptorSet;
			write.dstBinding     = ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind);
			write.descriptorType = NativeDescriptorType(binding.kind);
			write.descriptorCount   = NativeDescriptorCount(binding);
			const auto buffer_start = m_descriptor_buffers.size();
			const auto image_start  = m_descriptor_images.size();
			switch (binding.kind) {
				case BindingKind::Buffers:
					for (const auto resource: binding.resources) {
						const auto& view = descriptors.buffers.at(resource);
						EXIT_IF(view.buffer == nullptr);
						m_descriptor_buffers.emplace_back(view.buffer, view.offset, view.range);
					}
					break;
				case BindingKind::BdaPagetable:
				case BindingKind::FaultBuffer: {
					auto& cache = m_context.GetBufferCache();
					const auto* bda_buffer = binding.kind == BindingKind::BdaPagetable
					                             ? cache.GetBdaPageTableBuffer()
					                             : cache.GetFaultBuffer();
					m_descriptor_buffers.emplace_back(bda_buffer->Handle(), 0, bda_buffer->Size());
					break;
				}
				case BindingKind::FlattenedSrt:
				case BindingKind::UserData:
				case BindingKind::Gds: {
					const auto& view =
					    binding.kind == BindingKind::FlattenedSrt ? descriptors.flattened_srt
					    : binding.kind == BindingKind::UserData   ? descriptors.user_data
					                                              : descriptors.gds;
					EXIT_IF(view.buffer == nullptr);
					m_descriptor_buffers.emplace_back(view.buffer, view.offset, view.range);
					break;
				}
				case BindingKind::Samplers:
					for (const auto resource: binding.resources) {
						const auto sampler = descriptors.samplers.at(resource);
						EXIT_IF(sampler == nullptr);
						m_descriptor_images.emplace_back(sampler, nullptr,
						                                 vk::ImageLayout::eUndefined);
					}
					break;
				default:
					for (const auto resource: binding.resources) {
						m_descriptor_images.push_back(MakeImageInfo(
						    descriptors.images.at(resource), m_image_occurrences.at(resource)++));
					}
					break;
			}
			if (m_descriptor_buffers.size() != buffer_start) {
				write.pBufferInfo = m_descriptor_buffers.data() + buffer_start;
			}
			if (m_descriptor_images.size() != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			DescriptorWrites().push_back(write);
		}
		for (uint32_t i = 0; i < descriptors.images.size(); i++) {
			const auto expected =
			    descriptors.images[i].mip_views.empty()
			        ? 1u
			        : static_cast<uint32_t>(descriptors.images[i].mip_views.size());
			EXIT_IF(m_image_occurrences[i] != expected);
		}

		if (program.bindings.push_constant_size != 0) {
			const auto offset = program.bindings.push_constant_offset;
			EXIT_IF(offset % sizeof(uint32_t) != 0 ||
			        program.bindings.push_constant_size !=
			            prepared->user_data.size() * sizeof(uint32_t) ||
			        offset + program.bindings.push_constant_size >
			            ShaderRecompiler::IR::NativePushConstantSize);
			std::copy(prepared->user_data.begin(), prepared->user_data.end(),
			          PushConstants().begin() + offset / sizeof(uint32_t));
		}
	}
	// Batching census (diagnostic): publish what this draw actually binds, so the recorder can
	// decide whether the next draw could have joined this one. Hashing the descriptor writes and
	// the push constants captures resource SELECTION and per-draw data separately, which is the
	// distinction descriptor indexing turns on.
	if (Config::BatchCensusEnabled()) {
		uint64_t desc_hash = 0;
		for (const auto& w: DescriptorWrites()) {
			desc_hash = XXH3_64bits_withSeed(&w.dstBinding, sizeof(w.dstBinding), desc_hash);
			desc_hash = XXH3_64bits_withSeed(&w.descriptorType, sizeof(w.descriptorType), desc_hash);
			if (w.pBufferInfo != nullptr) {
				desc_hash = XXH3_64bits_withSeed(w.pBufferInfo,
				                                 sizeof(vk::DescriptorBufferInfo) *
				                                     w.descriptorCount,
				                                 desc_hash);
			}
			if (w.pImageInfo != nullptr) {
				desc_hash = XXH3_64bits_withSeed(
				    w.pImageInfo, sizeof(vk::DescriptorImageInfo) * w.descriptorCount, desc_hash);
			}
		}
		const auto push_hash = XXH3_64bits(PushConstants().data(),
		                                   ShaderRecompiler::IR::NativePushConstantSize);
		BatchCensusNoteBindings(desc_hash, push_hash);
	}
	record.pushConstants(pipeline.pipeline_layout, push_constant_stages, 0,
	                     ShaderRecompiler::IR::NativePushConstantSize, PushConstants().data());

	if (!DescriptorWrites().empty()) {
		EXIT_IF(pipeline.descriptor_set_layout == nullptr);
		if (pipeline.uses_push_descriptors) {
			record.pushDescriptorSetKHR(pipeline_bind_point, pipeline.pipeline_layout, 0,
			                            static_cast<uint32_t>(DescriptorWrites().size()),
			                            DescriptorWrites().data());
		} else {
			const auto set = m_context.GetDescriptorHeap().Commit(pipeline.descriptor_set_layout);
			for (auto& write: DescriptorWrites()) {
				write.dstSet = set;
			}
			m_context.GetGraphics().device.updateDescriptorSets(
			    static_cast<uint32_t>(DescriptorWrites().size()), DescriptorWrites().data(), 0,
			    nullptr);
			record.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, 0, 1, &set, 0,
			                          nullptr);
		}
	}
	for (auto* prepared: prepared_bindings) {
		prepared->committed = true;
	}
}

} // namespace Libs::Graphics
