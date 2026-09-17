#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_

#include "common/assert.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <optional>
#include <vector>

namespace Libs::Graphics {

namespace ShaderRecompiler::IR {
struct ResourceSnapshot;
}

struct BufferView {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::DeviceSize range  = VK_WHOLE_SIZE;
	// Byte adjustment folded into the packed memory offsets in user_data. Kept on the view so
	// publication can rewrite both together and they cannot disagree.
	uint32_t       publish_adjustment = 0;
};

// Result of the PURE texture-descriptor decode. Published so the differential test can compare
// cached against uncached decoding without standing up a RenderExecutor.
struct DecodedTexture {
	ShaderTextureResource   descriptor {};
	TextureCache::ImageDesc desc {};
	vk::Format              pixel_format      = vk::Format::eUndefined;
	vk::Format              view_format       = vk::Format::eUndefined;
	uint64_t                size_bytes        = 0;
	bool                    storage           = false;
	bool                    shader_conversion = false;
	bool                    is_null           = false;
};

// Test seam: the pure decode, callable without a RenderExecutor.
void DecodeTextureUncachedForTest(const ShaderRecompiler::IR::ImageResource&   resource,
                                  const ShaderRecompiler::IR::DescriptorValue& value,
                                  DecodedTexture&                              out);

struct TextureBinding {
	ImageId                    image_id;
	vk::ImageView              image_view = nullptr;
	TextureCache::ImageDesc    desc;
	vk::ImageLayout            layout = vk::ImageLayout::eUndefined;
	std::vector<vk::ImageView> mip_views;
};

struct NativeDescriptors {
	std::vector<BufferView>     buffers;
	std::vector<TextureBinding> images;
	std::vector<vk::Sampler>    samplers;
	BufferView                  gds;
	BufferView                  flattened_srt;
	BufferView                  user_data;
};

struct PreparedBindings {
	const ShaderRecompiler::IR::Program*          program  = nullptr;
	const ShaderRecompiler::IR::ResourceSnapshot* snapshot = nullptr;
	NativeDescriptors                             resources;
	std::vector<BufferId>                         buffer_ids;
	// Decoded once in FindBuffers and reused by RebindBuffers, which used to decode the same
	// snapshot dwords a second time for every buffer of every stage of every draw.
	std::vector<ShaderBufferResource>             buffer_descriptors;
	// Clamped against the guest range table by FindBuffers. RebindBuffers used to ask for the
	// same answer a second time, and that query is a contended lock away.
	std::vector<uint64_t>                         buffer_sizes;
	// Identity for the publication step, one entry per buffer resource. A BufferId is exactly
	// what retirement invalidates, so cache-backed bindings are re-derived from the GUEST RANGE.
	// Stream-backed bindings carry no range and are republished verbatim.
	struct PublishedBuffer {
		uint64_t address   = 0;
		uint64_t size      = 0;
		uint32_t alignment = 0;
		bool     stream    = false;
		bool     resolved  = false;
	};
	std::vector<PublishedBuffer>                  buffer_publish;
	std::vector<uint32_t>                         flattened_srt;
	std::vector<uint32_t>                         user_data;
	bool                                          committed = false;
	// Set by PublishBuffers. CommitBindings asserts it: the user_data and flattened_srt uploads
	// live in publication, so a caller that rebinds and commits without publishing leaves those
	// descriptors unbound. That shipped once and died as view.buffer == nullptr at the commit
	// site, which named the symptom and not the cause.
	bool                                          published = false;
};

// Both stages' bindings for one draw.
//
// Lives here rather than nested in RenderExecutor so PreparedShaders can carry it: render.h
// includes drawBatchQueue.h, so a type declared inside RenderExecutor cannot be named by the
// queue without a cycle. It is a pair of PreparedBindings and belongs beside them anyway.
struct GraphicsBindings {
	PreparedBindings                vertex;
	std::optional<PreparedBindings> pixel;
};

// Recycles a PreparedBindings' heap buffers for reuse by the next draw. Safe because only the
// buffers move; the object keeps value semantics.
void ReturnPooledBindingStorage(PreparedBindings& prepared);
// Exposed as the pair of the above so an offline benchmark can exercise the REAL recycling
// path rather than a re-implementation of it.
void TakePooledStorage(PreparedBindings& prepared);
// Test/benchmark accessors for the pool's occupancy and its cap.
[[nodiscard]] size_t PreparedBindingsPoolSize();
[[nodiscard]] size_t PreparedBindingsPoolCap();
void                 ClearPreparedBindingsPool();

// Clears the per-draw buffer resolution cache. MUST be called at the start of every draw: the
// guest can write buffer memory between draws, so a resolution is only valid within one.
void BeginDrawBufferScope();

[[nodiscard]] vk::DescriptorType
NativeDescriptorType(ShaderRecompiler::IR::DescriptorBindingKind kind);
[[nodiscard]] uint32_t
NativeDescriptorCount(const ShaderRecompiler::IR::DescriptorBinding& binding);
[[nodiscard]] vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture,
                                                    uint32_t              element = 0);

template <typename T>
[[nodiscard]] T DecodeNativeDescriptor(const ShaderRecompiler::IR::DescriptorValue& value) {
	static_assert(std::is_trivially_copyable_v<T>);
	static_assert(sizeof(T) % sizeof(uint32_t) == 0);
	T result {};
	EXIT_IF(value.dword_count < sizeof(result) / sizeof(uint32_t));
	std::memcpy(&result, value.dwords.data(), sizeof(result));
	return result;
}

struct TargetTextureViewInfo {
	vk::ImageViewType type        = static_cast<vk::ImageViewType>(VK_IMAGE_VIEW_TYPE_MAX_ENUM);
	uint32_t          base_layer  = 0;
	uint32_t          layer_count = 0;
};

[[nodiscard]] TargetTextureViewInfo
ResolveTargetTextureView(const ShaderRecompiler::IR::ImageResource& resource,
                         Prospero::ImageType type, uint32_t base_layer, uint32_t image_layers);

[[nodiscard]] bool IsSupportedDepthTargetDescriptor(const ShaderTextureResource& descriptor,
                                                    const Image& image, bool r128 = false);
[[nodiscard]] bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor,
                                                   const Image& image, bool r128 = false);
[[nodiscard]] bool
IsSupportedSampledVideoOutView(const ShaderRecompiler::IR::ImageResource& resource,
                               const ShaderTextureResource& descriptor, const Image& image);
void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
