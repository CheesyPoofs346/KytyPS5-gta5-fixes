// Vulkan integration harness for the buffer cache and binding publication.
//
// PURPOSE: reproduce the within-draw stale-binding defect against a REAL device and REAL
// BufferCache, prove the wrong value by GPU readback, and show the publication helper the fix
// uses returns the right one - while a stream-backed binding is left exactly as it was.
//
// NOT strictly headless: device selection runs through WindowContext::CreateVulkan, and
// VulkanFindPhysicalDevice asserts surface != nullptr and picks the queue family by presentation
// support. So a hidden 64x64 SDL window is opened. No game, no guest ELF, no PM4.
//
// SKIP is only for an environment that genuinely cannot support this suite - no video driver, no
// Vulkan loader. Everything past that point is a FAILURE, never a skip.

#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <SDL_vulkan.h>

#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "kernel/memory.h"

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/vertexBufferDescriptor.h"
#include <span>
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/presentation/window/windowInternal.h"

#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

void Check(const char* test, bool ok, const std::string& message) {
	if (!ok) {
		std::printf("[FAIL]    %-46s %s\n", test, message.c_str());
		g_failures++;
	}
}
void Pass(const char* test, const std::string& detail) {
	std::printf("[ok]      %-46s %s\n", test, detail.c_str());
}
// Prints before each step and flushes, so whichever step dies last-printed is the one that
// failed. A cleanup complaint during teardown cannot then be mistaken for the original fault.
void Step(const char* what) {
	std::fprintf(stderr, "STEP: %s\n", what);
	std::fflush(stderr);
}

std::string Hex(uint8_t v) {
	char buffer[8];
	std::snprintf(buffer, sizeof(buffer), "0x%02x", v);
	return buffer;
}

uint64_t MapGuestMemory(uint64_t size) {
	void*     address = nullptr;
	const int result  = Libs::LibKernel::Memory::KernelMapFlexibleMemory(
        // CPU read|write (0x01|0x02) AND GPU read|write (0x10|0x20). Mapping without the GPU
        // bits produced gpu_mode = NoAccess, and the buffer cache needs GPU-visible guest memory.
        &address, static_cast<size_t>(size), 0x01 | 0x02 | 0x10 | 0x20, 0);
	if (result != 0 || address == nullptr) {
		return 0;
	}
	return reinterpret_cast<uint64_t>(address);
}

// Write guest bytes and announce the change through the emulator's own invalidation entry point,
// so the upload is driven by real dirty tracking. Poking the cache directly would make a stale
// readback ambiguous: it could mean missing upload tracking rather than stale publication.
void GuestWrite(uint64_t vaddr, uint64_t size, uint8_t value) {
	// WriteBacking, not a raw memset: guest pages may be reserved rather than committed in the
	// host mapping, and writing through the pointer segfaults. This is the emulator API for
	// putting bytes into guest memory. InvalidateMemory then announces the change through the
	// normal path, so the upload is driven by real dirty tracking.
	std::vector<uint8_t> bytes(static_cast<size_t>(size), value);
	Libs::LibKernel::Memory::WriteBacking(vaddr, bytes.data(), size);
	Libs::LibKernel::Memory::InvalidateMemory(vaddr, size);
}

// One reusable host-visible staging buffer; readback copies into it, submits, and reads byte 0.
struct Readback {
	VulkanBuffer staging;
	void*        mapped = nullptr;

	void Create(GraphicContext& graphics) {
		staging.usage           = vk::BufferUsageFlagBits::eTransferDst;
		staging.memory.property = vk::MemoryPropertyFlagBits::eHostVisible |
		                          vk::MemoryPropertyFlagBits::eHostCoherent;
		staging.memory.preferred_property = staging.memory.property;
		graphics.CreateBuffer(256, staging);
		graphics.MapMemory(staging.memory, mapped);
	}
	void Destroy(GraphicContext& graphics) {
		if (mapped != nullptr) {
			graphics.UnmapMemory(staging.memory);
			mapped = nullptr;
		}
		if (staging.buffer != nullptr) {
			vmaDestroyBuffer(graphics.allocator, static_cast<VkBuffer>(staging.buffer),
			                 staging.memory.allocation);
			staging.buffer            = nullptr;
			staging.memory.allocation = nullptr;
		}
	}

	// Records a copy into the scheduler's current command buffer, submits and waits. The value is
	// what the GPU actually sees through that handle and offset.
	uint8_t ReadByte(WindowContext& context, vk::Buffer source, uint64_t offset) {
		auto&          scheduler = context.render_context->GetCommandScheduler();
		vk::BufferCopy region {};
		region.srcOffset = offset;
		region.dstOffset = 0;
		region.size      = 1;
		scheduler.Current().Handle().copyBuffer(source, staging.buffer, 1, &region);
		scheduler.FlushAndWait();
		return mapped != nullptr ? *static_cast<const uint8_t*>(mapped) : 0xFFu;
	}
};

} // namespace

int main() {
	// Unbuffered: a failure during teardown must not swallow results already produced.
	std::setvbuf(stdout, nullptr, _IONBF, 0);
	std::printf("Buffer cache / binding publication harness:\n\n");

	// The emulator's own startup order, mirroring main() and Run(). It was NOT established that
	// each of these is strictly required - only that this order works.
	Common::VirtualMemory::Init();
	Common::InitializeThreads();
	Config::Initialize();
	{
		Config::ConfigOptions options;
		options.user_name     = "harness";
		options.user_id       = 1;
		options.screen_width  = 64;
		options.screen_height = 64;
		options.vulkan_validation_enabled = true;
		Config::Load(options);
	}
	Libs::LibKernel::Memory::Initialize();

	SDL_SetMainReady();
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		std::printf("[skip]    no SDL video driver available: %s\n", SDL_GetError());
		std::printf("\nenvironment cannot support this suite; skipped, NOT passed\n");
		return 2;
	}
	if (SDL_Vulkan_LoadLibrary(nullptr) != 0) {
		std::printf("[skip]    no Vulkan loader available: %s\n", SDL_GetError());
		std::printf("\nenvironment cannot support this suite; skipped, NOT passed\n");
		return 2;
	}

	// Heap-allocated so shutdown order is explicit, and destroyed - not leaked. Production
	// ownership: the context owns presenter, render context and device, all of which reference
	// the SDL window and surface, so it must die while the window is still alive.
	auto* context_storage = new WindowContext();
	auto& context         = *context_storage;
	context.window        = SDL_CreateWindow("kyty buffer harness", SDL_WINDOWPOS_UNDEFINED,
	                                         SDL_WINDOWPOS_UNDEFINED, 64, 64,
	                                         SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
	if (context.window == nullptr) {
		// Loader and driver are both present by now, so this is a real fault, not an
		// unsupported environment.
		std::printf("[FAIL]    could not create the harness window: %s\n", SDL_GetError());
		return 1;
	}
	context.graphic_ctx.screen_width  = 64;
	context.graphic_ctx.screen_height = 64;
	context.CreateVulkan();
	if (context.graphic_ctx.device == nullptr) {
		std::printf("[FAIL]    CreateVulkan produced no device\n");
		return 1;
	}
	Pass("vulkan device comes up without the game",
	     std::string("device: ") +
	         context.graphic_ctx.GetPhysicalDeviceProperties().deviceName.data());
	Check("validation layers enabled", Config::VulkanValidationEnabled(),
	      "the harness must not silently run this probe without VK_LAYER_KHRONOS_validation");
	if (Config::VulkanValidationEnabled()) {
		Pass("validation layers enabled", "VK_LAYER_KHRONOS_validation requested for this harness");
	}

	// ObtainBuffer records into the scheduler's current command buffer and asserts it is valid.
	static HW::Context    registers {};
	static HW::UserConfig user_config {};
	static HW::Shader     shaders {};
	auto&                 scheduler = context.render_context->GetCommandScheduler();
	Step("scheduler begin");
	scheduler.Begin(registers, user_config, shaders);
	// GPU-dirty readback is owned by GuestGpu's command lane. The old harness deliberately did
	// not start one, which was sufficient for CPU-owned buffer tests but made a later unmap of a
	// GPU-dirty range correctly fail at RenderContext::GetGpu().
	context.render_context->InitializeGpu(nullptr);

	auto&    cache = context.render_context->GetBufferCache();
	Readback readback;
	Step("create readback staging");
	readback.Create(context.graphic_ctx);

	// ------------------------------------------------------------------------------------
	// The regression.

	constexpr uint64_t kGuestSize  = 4ull * 1024 * 1024;
	constexpr uint64_t kCacheSize  = 32 * 1024; // > CACHING_PAGESIZE (16K): never the stream path
	constexpr uint64_t kStreamSize = 256;       // <= CACHING_PAGESIZE: eligible for the stream path

	Step("map guest memory");
	const uint64_t guest = MapGuestMemory(kGuestSize);
	if (guest == 0) {
		std::printf("[FAIL]    could not map guest memory for the regression\n");
		return 1;
	}
	const uint64_t cache_range  = (guest + 0x20000) & ~(BufferCache::CACHING_PAGESIZE - 1);
	const uint64_t stream_range = (guest + 0x200000) & ~(BufferCache::CACHING_PAGESIZE - 1);

	Step("initial backing write, cache range");
	GuestWrite(cache_range, kCacheSize, 0x11);
	Step("initial backing write, stream range");
	GuestWrite(stream_range, kStreamSize, 0x55);

	// 1. Capture a CACHE-BACKED binding, as draw preparation does.
	Step("obtain cache-backed buffer");
	const auto obtained_a = cache.ObtainBuffer(cache_range, kCacheSize, false);
	if (obtained_a.first == nullptr) {
		std::printf("[FAIL]    ObtainBuffer returned no buffer for the cache-backed range\n");
		return 1;
	}
	auto* const             buffer_a         = obtained_a.first;
	const vk::Buffer        captured_handle  = buffer_a->Handle();
	const uint64_t          captured_offset  = obtained_a.second;
	const vk::DeviceAddress captured_address = buffer_a->BufferDeviceAddress();

	// Bind the exact API used by CommitVertexBuffers with two real cache allocations: a
	// descriptor-sized slice of a normal vertex buffer and the real 16-byte null buffer used for
	// an empty slot.  The checks establish the same bounds invariant production enforces before
	// invoking bindVertexBuffers2; validation layers then see the actual Vulkan call.
	{
		const char* name = "vertex bindings use descriptor-sized real allocations";
		auto& null_buffer = cache.GetBuffer(NULL_BUFFER_ID);
		ShaderVertexInputInfo vertex_info {};
		auto& stride_zero = vertex_info.buffers[0];
		stride_zero.stride = 0;
		stride_zero.num_records = 1;
		stride_zero.attr_num = 2;
		stride_zero.attr_indices[0] = 0;
		stride_zero.attr_offsets[0] = 12;
		stride_zero.attr_indices[1] = 1;
		stride_zero.attr_offsets[1] = 32;
		vertex_info.resources[0].fields[3] =
		    (static_cast<uint32_t>(Prospero::BufferFormat::k32Float) << 12u) | (2u << 28u);
		vertex_info.resources[1].fields[3] =
		    (static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float) << 12u) |
		    (2u << 28u);
		const auto descriptor_size = VertexBufferDescriptorSize(stride_zero, vertex_info);
		const auto available_size = captured_offset <= buffer_a->Size()
		                                ? buffer_a->Size() - captured_offset
		                                : 0;
		const auto normal_size = std::min<uint64_t>(descriptor_size, available_size);
		const std::array<vk::Buffer, 2> buffers {buffer_a->Handle(), null_buffer.Handle()};
		const std::array<vk::DeviceSize, 2> offsets {captured_offset, 0};
		const std::array<vk::DeviceSize, 2> sizes {normal_size, null_buffer.Size()};
		Check(name, descriptor_size == 48 && normal_size == descriptor_size &&
		                offsets[0] <= buffer_a->Size() &&
		                sizes[0] <= buffer_a->Size() - offsets[0],
		      "stride-zero OOB=2 descriptor range exceeds its actual Buffer allocation");
		Check(name, sizes[1] != 0 && offsets[1] <= null_buffer.Size() &&
		                sizes[1] <= null_buffer.Size() - offsets[1],
		      "null-slot range exceeds the actual null Buffer allocation");
		if (normal_size != 0 && sizes[1] != 0) {
			scheduler.Current().Handle().bindVertexBuffers2(0, static_cast<uint32_t>(buffers.size()),
			                                                buffers.data(), offsets.data(),
			                                                sizes.data(), nullptr);
			Pass(name, "stride-zero OOB=2 slice and null slot fit their actual allocations before bindVertexBuffers2");
		}
	}

	// 2. Capture a STREAM-BACKED binding: a private copy in a ring allocation, which must survive
	//    the replacement untouched.
	Step("obtain stream-backed buffer");
	const auto obtained_s = cache.ObtainBuffer(stream_range, kStreamSize, false);
	if (obtained_s.first == nullptr) {
		std::printf("[FAIL]    ObtainBuffer returned no buffer for the stream-backed range\n");
		return 1;
	}
	const vk::Buffer stream_handle = obtained_s.first->Handle();
	const uint64_t   stream_offset = obtained_s.second;
	// A stream allocation is not owned by any guest range, so publication finds no owner for it.
	const bool is_stream = cache.FindPublishedOwner(stream_range, kStreamSize).first == nullptr;

	// 3. A later range-changing operation joins the cache-backed buffer and retires it - the same
	//    thing an image resolve does via ObtainBufferForImage -> FindBuffer -> CreateBuffer.
	Step("spanning obtain that retires the first buffer");
	const uint64_t spanning_begin = cache_range - BufferCache::CACHING_PAGESIZE;
	const auto     spanning =
	    cache.ObtainBuffer(spanning_begin, kCacheSize + 2 * BufferCache::CACHING_PAGESIZE, false);
	{
		const char* name    = "a later range change retires the binding";
		const bool  retired = buffer_a->is_deleted;
		Check(name, retired, "the cache-backed buffer was not retired; the case did not set up");
		Check(name, spanning.first != nullptr && spanning.first != buffer_a,
		      "the spanning request did not produce a replacement");
		if (retired) {
			Pass(name, "captured handle now refers to a retired buffer");
		}
	}

	// 4. The guest updates the shared bytes, and the current owner is synchronized.
	Step("update the tracked range");
	GuestWrite(cache_range, kCacheSize, 0x99);
	Step("re-obtain after replacement");
	const auto obtained_now = cache.ObtainBuffer(cache_range, kCacheSize, false);
	if (obtained_now.first == nullptr) {
		std::printf("[FAIL]    no owner after the replacement\n");
		return 1;
	}

	// CONTROL, and it has to come first: if the current owner does NOT show the update, the fault
	// is upload/dirty tracking in this harness rather than stale publication, and every result
	// below would be unattributable.
	{
		const char* name = "control: current owner sees the update";
		const auto  current =
		    readback.ReadByte(context, obtained_now.first->Handle(), obtained_now.second);
		Check(name, current == 0x99,
		      "current owner read " + Hex(current) +
		          ", expected 0x99 - dirty tracking is not delivering the upload, so the stale "
		          "result below cannot be attributed to binding publication");
		if (current == 0x99) {
			Pass(name, "0x99 through the live buffer: upload tracking is working");
		}
	}

	// 5. ORIGINAL PATH: read through the handle captured in step 1, which is what a draw records
	//    when nothing re-resolves the binding.
	{
		const char* name  = "original path publishes a stale value";
		const auto  stale = readback.ReadByte(context, captured_handle, captured_offset);
		Check(name, stale == 0x11,
		      "captured handle read " + Hex(stale) +
		          "; expected the pre-update 0x11. If this is already 0x99 the defect did not "
		          "reproduce and the fix below proves nothing");
		if (stale == 0x11) {
			Pass(name, "read " + Hex(stale) + " where the current value is 0x99");
		}
	}

	// 6. FIXED PATH: publication resolves the guest range to its current owner, using the helper
	//    the production publication step calls. No allocation, no synchronize.
	{
		const char* name  = "fixed path publishes the current value";
		const auto  owner = cache.FindPublishedOwner(cache_range, kCacheSize);
		Check(name, owner.first != nullptr,
		      "publication found no owner; it must fail loudly, not guess");
		if (owner.first != nullptr) {
			const auto fresh = readback.ReadByte(context, owner.first->Handle(), owner.second);
			Check(name, fresh == 0x99, "republished handle read " + Hex(fresh) + ", expected 0x99");
			Check(name, owner.first->Handle() != captured_handle,
			      "publication returned the retired handle");
			Check(name, owner.first->BufferDeviceAddress() != captured_address,
			      "device address was not refreshed");
			if (fresh == 0x99) {
				Pass(name, "handle, offset and device address all resolve to the replacement");
			}
		}
	}

	// 7. The stream-backed binding must be untouched by all of this.
	{
		const char* name  = "stream-backed binding survives unchanged";
		const auto  value = readback.ReadByte(context, stream_handle, stream_offset);
		Check(name, value == 0x55,
		      "stream allocation read " + Hex(value) + ", expected the original 0x55");
		Check(name, is_stream,
		      "the stream range resolved to a cache buffer; this case did not exercise the ring");
		if (value == 0x55 && is_stream) {
			Pass(name, "ring allocation and offset still hold 0x55");
		}
	}

	// 8. Publication refuses a range it cannot resolve rather than inventing one.
	{
		const char* name  = "publication fails loudly on no owner";
		const auto  owner = cache.FindPublishedOwner(guest + kGuestSize - 0x1000, 0x800);
		Check(name, owner.first == nullptr, "publication invented an owner for an unmapped range");
		if (owner.first == nullptr) {
			Pass(name, "returns no owner instead of allocating one");
		}
	}

	Step("destroy readback staging");
	// ------------------------------------------------------------------------------------
	// The PRODUCTION publication path: RenderExecutor::PublishBuffers, driven with a hand-built
	// PreparedBindings. This is the function BindGraphicsResources calls after both stages'
	// buffers and images, so it is the code that actually has to fix the defect - the checks
	// above only established that the lookup primitive returns the right thing.
	{
		auto& executor = context.render_context->GetRenderExecutor();

		ShaderRecompiler::IR::Program program {};
		program.stage = ShaderType::Vertex;
		program.info.buffers.resize(2);
		program.info.buffers[0].read = true;
		program.info.buffers[1].read = true;
		program.bindings.memory_offset_dword = 0;
		program.bindings.memory_offset_count = 2;

		// Two bindings over the SAME ranges the checks above used: index 0 cache-backed, index 1
		// stream-backed. Both must come out of publication correct, and differently.
		// Each arm gets its OWN untouched guest range. Reusing an earlier range meant the
		// spanning request found it already covered, created nothing, retired nothing, and the
		// "stale" handle was simply the live owner - the fix-disabled arm passed for the wrong
		// reason. The retirement is now asserted rather than assumed.
		uint64_t next_range = guest + 0x300000;

		struct Arm {
			PreparedBindings prepared;
			bool             retired = false;
			uint8_t          updated = 0;
		};

		const auto run_publication = [&](bool fix_enabled, uint8_t updated_value) {
			Config::SetBindingPublishForTest(fix_enabled);

			const uint64_t range = next_range;
			// 256 KB apart: 1 MB spacing walked past the end of the 4 MB mapping and the
			// backing write failed fatally.
			next_range += 0x40000;
			GuestWrite(range, kCacheSize, 0x40);

			Arm arm;
			arm.updated          = updated_value;
			arm.prepared.program = &program;
			arm.prepared.user_data.assign(program.bindings.ShaderDataDwords(), 0);
			arm.prepared.buffer_publish.assign(2, {});
			arm.prepared.resources.buffers.assign(2, BufferView {});

			// The obtain phase, recorded exactly as RebindBuffers records it.
			const auto cache_obtained  = cache.ObtainBuffer(range, kCacheSize, false);
			const auto stream_obtained = cache.ObtainBuffer(stream_range, kStreamSize, false);

			arm.prepared.resources.buffers[0].buffer = cache_obtained.first->Handle();
			arm.prepared.resources.buffers[0].offset = cache_obtained.second;
			arm.prepared.buffer_publish[0]           = {range, kCacheSize, 1, false, true};

			arm.prepared.resources.buffers[1].buffer = stream_obtained.first->Handle();
			arm.prepared.resources.buffers[1].offset = stream_obtained.second;
			arm.prepared.buffer_publish[1]           = {
                stream_range, kStreamSize, 1,
                cache.IsStreamAllocation(stream_obtained.first), true};

			// A later range-changing operation retires the cache-backed buffer, exactly as an
			// image resolve does through ObtainBufferForImage -> FindBuffer -> CreateBuffer.
			cache.ObtainBuffer(range - BufferCache::CACHING_PAGESIZE,
			                   kCacheSize + 2 * BufferCache::CACHING_PAGESIZE, false);
			arm.retired = cache_obtained.first->is_deleted;

			// The guest updates the shared bytes, and the live owner takes them.
			GuestWrite(range, kCacheSize, updated_value);
			cache.ObtainBuffer(range, kCacheSize, false);

			executor.PublishBuffers(arm.prepared);
			return arm;
		};

		// Fix DISABLED: publication must leave the stale handle in place, and the readback must
		// show the pre-update byte. If this passes, the test proves nothing.
		{
			const char* name = "production path: stale WITHOUT the fix";
			auto        arm  = run_publication(false, 0xC0);
			Check(name, arm.retired,
			      "the captured buffer was never retired, so nothing could go stale");
			const auto value = readback.ReadByte(context, arm.prepared.resources.buffers[0].buffer,
			                                     arm.prepared.resources.buffers[0].offset);
			Check(name, value != 0xC0,
			      "published handle already showed the current 0xC0 with the fix off; the "
			      "regression would not prove the fix");
			if (arm.retired && value != 0xC0) {
				Pass(name, "published a stale " + Hex(value) + " where the current value is 0xC0");
			}
		}

		// Fix ENABLED: publication must republish onto the live owner.
		{
			const char* name = "production path: correct WITH the fix";
			auto        arm  = run_publication(true, 0xC1);
			Check(name, arm.retired,
			      "the captured buffer was never retired, so the fix had nothing to correct");
			auto&      prepared = arm.prepared;
			const auto value    = readback.ReadByte(context, prepared.resources.buffers[0].buffer,
			                                        prepared.resources.buffers[0].offset);
			Check(name, value == 0xC1,
			      "published handle read " + Hex(value) + ", expected the current 0xC1");

			// The stream-backed binding must be untouched: same allocation, same offset.
			const auto stream_value = readback.ReadByte(context,
			                                            prepared.resources.buffers[1].buffer,
			                                            prepared.resources.buffers[1].offset);
			Check(name, stream_value == 0x55,
			      "stream binding read " + Hex(stream_value) + " after publication, expected 0x55");

			// The uploaded offsets must agree with the published handles.
			const auto owner = cache.FindPublishedOwner(arm.prepared.buffer_publish[0].address,
			                                            kCacheSize);
			Check(name, owner.first != nullptr && prepared.resources.buffers[0].buffer ==
			                                          owner.first->Handle(),
			      "published handle is not the live owner");
			const auto expected_adjustment = prepared.resources.buffers[0].publish_adjustment;
			Check(name, (prepared.user_data[0] & 0xffu) == expected_adjustment,
			      "packed user_data offset disagrees with the published view");
			if (value == 0xC1 && stream_value == 0x55) {
				Pass(name, "republished onto the live owner; stream and packed offsets consistent");
			}
		}
		Config::SetBindingPublishForTest(true);

		// ---------------------------------------------------------------------------------
		// RenderExecutor::FinalizeBindings - the shared helper BOTH the graphics and compute
		// paths call. This is the caller-side regression: publication owns the user_data and
		// flattened_srt uploads, and renderCompute previously rebound and committed without
		// publishing, so every dispatch reached the commit site with those descriptors unbound.
		//
		// COVERAGE, precisely: this drives the real helper and checks what it produces. It does
		// NOT dispatch compute, build a pipeline, or call CommitBindings - the assertion added
		// there is verified by reading, not by running.
		ShaderRecompiler::IR::Program stage_program {};
		stage_program.stage                        = ShaderType::Compute;
		stage_program.info.buffers.resize(1);
		stage_program.info.buffers[0].read         = true;
		stage_program.bindings.memory_offset_dword = 0;
		stage_program.bindings.memory_offset_count = 1;
		// A UserData binding is what makes publication upload user_data at all.
		stage_program.bindings.descriptors.push_back(
		    {ShaderRecompiler::IR::DescriptorBindingKind::UserData, {}});

		// FinalizeBindings calls RebindBuffers, which needs the decoded descriptor arrays
		// FindBuffers would have filled. A zero descriptor resolves to address 0 and is skipped
		// as a null binding, which is enough to exercise the helper's structure and its uploads.
		// RebindBuffers asserts a snapshot; one zeroed descriptor per buffer resource is enough,
		// and resolves to address 0 so the binding is skipped as null.
		static ShaderRecompiler::IR::ResourceSnapshot stage_snapshot;
		stage_snapshot.buffers.assign(1, ShaderRecompiler::IR::DescriptorValue {});

		const auto make_stage = [&](PreparedBindings& prepared) {
			prepared.program  = &stage_program;
			prepared.snapshot = &stage_snapshot;
			prepared.buffer_ids.assign(1, BufferId {});
			prepared.buffer_descriptors.assign(1, ShaderBufferResource {});
			prepared.buffer_sizes.assign(1, 0);
			prepared.user_data.assign(stage_program.bindings.ShaderDataDwords(), 0);
			prepared.flattened_srt.assign(8, 0x5A5A5A5Au);
			prepared.published = true;   // stale value from a previous draw
		};

		{
			// ONE STAGE, compute-shaped: the exact call renderCompute now makes.
			const char*      name = "FinalizeBindings: one compute-shaped stage";
			PreparedBindings only;
			make_stage(only);
			PreparedBindings* stages[1] = {&only};
			executor.FinalizeBindings(std::span<PreparedBindings* const> {stages, 1u});

			Check(name, only.published, "helper did not publish the stage");
			Check(name, only.resources.user_data.buffer != nullptr,
			      "user_data unbound after FinalizeBindings - the compute regression");
			Check(name, only.resources.flattened_srt.buffer != nullptr,
			      "flattened_srt unbound after FinalizeBindings");
			if (only.published && only.resources.user_data.buffer != nullptr &&
			    only.resources.flattened_srt.buffer != nullptr) {
				Pass(name, "published, with user_data and flattened_srt both bound");
			}
		}
		{
			// TWO STAGES, graphics-shaped. Both must come out published and bound; neither may
			// be left finalized against a state the other stage later changed.
			const char*      name = "FinalizeBindings: two graphics-shaped stages";
			PreparedBindings vertex;
			PreparedBindings pixel;
			make_stage(vertex);
			make_stage(pixel);
			PreparedBindings* stages[2] = {&vertex, &pixel};
			executor.FinalizeBindings(std::span<PreparedBindings* const> {stages, 2u});

			Check(name, vertex.published && pixel.published, "a stage was left unpublished");
			Check(name, vertex.resources.user_data.buffer != nullptr &&
			                pixel.resources.user_data.buffer != nullptr,
			      "a stage was left with user_data unbound");
			if (vertex.published && pixel.published) {
				Pass(name, "both stages published and bound");
			}
		}
		{
			// A LATER STAGE REPLACING AN EARLIER STAGE'S BUFFER. Stage 0 resolves a range; stage
			// 1 resolves a spanning range that joins and retires it. Finalizing stage 0
			// completely before starting stage 1 would leave stage 0 pointing at the retired
			// buffer - the three-pass order over the whole span is what prevents that.
			const char* name = "FinalizeBindings: later stage retires earlier buffer";

			const uint64_t shared = next_range;
			next_range += 0x40000;
			GuestWrite(shared, kCacheSize, 0x70);

			PreparedBindings first;
			PreparedBindings second;
			make_stage(first);
			make_stage(second);

			// Publication records built by hand, standing in for what RebindBuffers records:
			// stage 0 over the range, stage 1 over a span that will swallow it.
			const auto first_obtained = cache.ObtainBuffer(shared, kCacheSize, false);
			first.buffer_publish.assign(1, {shared, kCacheSize, 1, false, true});
			first.resources.buffers.assign(1, BufferView {});
			first.resources.buffers[0].buffer = first_obtained.first->Handle();
			first.resources.buffers[0].offset = first_obtained.second;

			// Stage 1's obtain is the range-changing operation.
			const auto second_obtained =
			    cache.ObtainBuffer(shared - BufferCache::CACHING_PAGESIZE,
			                       kCacheSize + 2 * BufferCache::CACHING_PAGESIZE, false);
			second.buffer_publish.assign(
			    1, {shared - BufferCache::CACHING_PAGESIZE,
			        kCacheSize + 2 * BufferCache::CACHING_PAGESIZE, 1, false, true});
			second.resources.buffers.assign(1, BufferView {});
			second.resources.buffers[0].buffer = second_obtained.first->Handle();
			second.resources.buffers[0].offset = second_obtained.second;

			const bool retired = first_obtained.first->is_deleted;
			Check(name, retired, "stage 1 did not retire stage 0's buffer; nothing to prove");

			// Publish both from the settled state, as FinalizeBindings' third pass does.
			executor.PublishBuffers(first);
			executor.PublishBuffers(second);

			const auto owner = cache.FindPublishedOwner(shared, kCacheSize);
			Check(name, owner.first != nullptr, "no owner for the shared range after the join");
			if (owner.first != nullptr) {
				Check(name, first.resources.buffers[0].buffer == owner.first->Handle(),
				      "stage 0 still points at the retired buffer after publication");
				const auto value = readback.ReadByte(context, first.resources.buffers[0].buffer,
				                                     first.resources.buffers[0].offset);
				Check(name, value == 0x70,
				      "stage 0 read " + Hex(value) + " through its republished view, expected 0x70");
				if (retired && first.resources.buffers[0].buffer == owner.first->Handle() &&
				    value == 0x70) {
					Pass(name, "stage 0 republished onto the buffer stage 1 created");
				}
			}
		}
	}

	// ------------------------------------------------------------------------------------
	// Stream-repeat threshold experiment. This intentionally uses ObtainBuffer, dirty tracking,
	// staging uploads and GPU readback from the production cache -- it is not a model of them.
	// The workload is synthetic: it establishes the best case for one unchanged address, not a
	// GTA repeat rate or a gameplay performance claim.
	{
		constexpr uint64_t kRepeatSize  = 256;
		constexpr uint32_t kRepeatCount = 4096;
		constexpr uint64_t kTrackerPage = 4 * 1024;
		const uint64_t repeat_range = (guest + 0x280000) & ~(BufferCache::CACHING_PAGESIZE - 1);

		auto set_threshold = [](uint32_t threshold) {
			Config::ConfigOptions options {};
			options.user_name               = "harness";
			options.user_id                 = 1;
			options.screen_width            = 64;
			options.screen_height           = 64;
			options.stream_repeat_threshold = threshold;
			Config::Load(options);
		};
		auto run_unchanged = [&](uint32_t threshold, uint8_t value) {
			set_threshold(threshold);
			cache.ResetStreamRepeatTestStats();
			GuestWrite(repeat_range, kRepeatSize, value);
			const auto start = std::chrono::steady_clock::now();
			std::pair<Buffer*, uint64_t> last {};
			for (uint32_t i = 0; i < kRepeatCount; ++i) {
				last = cache.ObtainBuffer(repeat_range, kRepeatSize, false);
			}
			const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
			                         std::chrono::steady_clock::now() - start)
			                         .count();
			return std::tuple {cache.GetStreamRepeatTestStats(), last, elapsed};
		};

		const auto [off_stats, off_last, off_us] = run_unchanged(0, 0x3c);
		const auto [one_stats, one_last, one_us] = run_unchanged(1, 0x7e);
		std::printf("[info]    repeat counters: off stream=%llu alloc=%llu sync=%llu upload=%llu; one stream=%llu alloc=%llu sync=%llu upload=%llu\n",
		            static_cast<unsigned long long>(off_stats.stream_bytes),
		            static_cast<unsigned long long>(off_stats.persistent_allocations),
		            static_cast<unsigned long long>(off_stats.persistent_sync_calls),
		            static_cast<unsigned long long>(off_stats.persistent_upload_bytes),
		            static_cast<unsigned long long>(one_stats.stream_bytes),
		            static_cast<unsigned long long>(one_stats.persistent_allocations),
		            static_cast<unsigned long long>(one_stats.persistent_sync_calls),
		            static_cast<unsigned long long>(one_stats.persistent_upload_bytes));
		const char* name = "stream repeat: threshold 0 versus 1, unchanged reads";
		Check(name, off_last.first != nullptr && one_last.first != nullptr,
		      "ObtainBuffer returned no buffer");
		Check(name, off_stats.stream_bytes == kRepeatCount * kRepeatSize,
		      "threshold 0 did not retain the stream path for every unchanged read");
		Check(name, one_stats.stream_bytes == kRepeatSize,
		      "threshold 1 did not leave exactly the first read on the stream path");
		Check(name, one_stats.persistent_allocations == 1 && one_stats.persistent_upload_bytes == kTrackerPage,
		      "threshold 1 did not create one persistent buffer with one tracker-page upload");
		Check(name, off_stats.stream_bytes - one_stats.stream_bytes ==
		                uint64_t {kRepeatCount - 1} * kRepeatSize,
		      "avoided stream bytes do not equal the threshold delta");
		const auto one_value = one_last.first != nullptr ?
		                           readback.ReadByte(context, one_last.first->Handle(), one_last.second) : 0;
		Check(name, one_value == 0x7e, "persistent result read " + Hex(one_value));
		if (g_failures == 0) {
			Pass(name, "stream bytes " + std::to_string(off_stats.stream_bytes) + " -> " +
			                   std::to_string(one_stats.stream_bytes) + ", persistent upload " +
			                   std::to_string(one_stats.persistent_upload_bytes) + ", elapsed us " +
			                   std::to_string(off_us) + " -> " + std::to_string(one_us));
		}

		// A partial CPU write must upload only the dirty span after the persistent transition.
		set_threshold(1);
		cache.ResetStreamRepeatTestStats();
		const uint64_t partial_range = repeat_range + 0x4000;
		GuestWrite(partial_range, kRepeatSize, 0x19);
		cache.ObtainBuffer(partial_range, kRepeatSize, false);
		cache.ObtainBuffer(partial_range, kRepeatSize, false);
		GuestWrite(partial_range + 64, 32, 0xa6);
		const auto partial = cache.ObtainBuffer(partial_range, kRepeatSize, false);
		const auto partial_stats = cache.GetStreamRepeatTestStats();
		std::printf("[info]    partial counters: stream=%llu alloc=%llu sync=%llu upload=%llu\n",
		            static_cast<unsigned long long>(partial_stats.stream_bytes),
		            static_cast<unsigned long long>(partial_stats.persistent_allocations),
		            static_cast<unsigned long long>(partial_stats.persistent_sync_calls),
		            static_cast<unsigned long long>(partial_stats.persistent_upload_bytes));
		const char* partial_name = "stream repeat: partial CPU write after persistent transition";
		Check(partial_name, partial.first != nullptr && !cache.IsStreamAllocation(partial.first),
		      "partial-write read did not use the persistent owner");
		if (partial.first != nullptr) {
			const auto before = readback.ReadByte(context, partial.first->Handle(), partial.second);
			const auto changed = readback.ReadByte(context, partial.first->Handle(), partial.second + 64);
			Check(partial_name, before == 0x19 && changed == 0xa6,
			      "readback " + Hex(before) + "/" + Hex(changed) + ", expected 0x19/0xa6");
		}
		Check(partial_name, partial_stats.persistent_upload_bytes == 2 * kTrackerPage,
		      "persistent upload bytes do not reflect the two 4 KiB dirty tracker pages");
		if (g_failures == 0) {
			Pass(partial_name, "32-byte write preserves data but reuploads its 4 KiB tracker page");
		}

		// Distinct overlapping ranges have distinct tracker keys. Each must transition correctly,
		// and the later dirty page must not leave either stream or persistent data stale.
		set_threshold(1);
		cache.ResetStreamRepeatTestStats();
		const uint64_t overlap_range = repeat_range + 0x8000;
		GuestWrite(overlap_range, 512, 0x42);
		cache.ObtainBuffer(overlap_range, kRepeatSize, false);
		cache.ObtainBuffer(overlap_range, kRepeatSize, false);
		GuestWrite(overlap_range + 128, 128, 0xd4);
		cache.ObtainBuffer(overlap_range + 128, kRepeatSize, false);
		const auto overlap = cache.ObtainBuffer(overlap_range + 128, kRepeatSize, false);
		const auto overlap_stats = cache.GetStreamRepeatTestStats();
		const char* overlap_name = "stream repeat: overlapping ranges retain fresh data";
		Check(overlap_name, overlap.first != nullptr && !cache.IsStreamAllocation(overlap.first),
		      "overlapping range did not transition to its persistent owner");
		if (overlap.first != nullptr) {
			const auto value = readback.ReadByte(context, overlap.first->Handle(), overlap.second);
			Check(overlap_name, value == 0xd4, "overlap read " + Hex(value) + ", expected 0xd4");
		}
		Check(overlap_name, overlap_stats.stream_bytes == 2 * kRepeatSize &&
		                        overlap_stats.persistent_upload_bytes == 2 * kTrackerPage,
		      "overlap arms did not each stream once then upload one dirty tracker page");
		if (g_failures == 0) {
			Pass(overlap_name, "two address keys, two stream copies, two 4 KiB persistent uploads");
		}

		// GPU-dirty ranges are outside the stream fast path. Submit a GPU fill, read it back through
		// BufferCache::ReadMemory (which sends the work through GuestGpu), then verify both the
		// device buffer and guest backing before the mapping is released.
		set_threshold(1);
		const uint64_t gpu_range = repeat_range + 0xc000;
		GuestWrite(gpu_range, kRepeatSize, 0x11);
		const auto gpu_written = cache.ObtainBuffer(gpu_range, kRepeatSize, true);
		cache.FillBuffer(gpu_range, kRepeatSize, 0x5a5a5a5a, false);
		const auto gpu = cache.ObtainBuffer(gpu_range, kRepeatSize, false);
		const auto gpu_owner = cache.FindPublishedOwner(gpu_range, kRepeatSize);
		const char* gpu_name = "stream repeat: GPU-dirty readback and publication";
		Check(gpu_name, gpu_written.first != nullptr && gpu.first != nullptr &&
		                    !cache.IsStreamAllocation(gpu.first),
		      "GPU-dirty range incorrectly used the stream fast path");
		if (gpu.first != nullptr) {
			const auto device_value = readback.ReadByte(context, gpu.first->Handle(), gpu.second);
			Check(gpu_name, device_value == 0x5a,
			      "GPU buffer read " + Hex(device_value) + ", expected 0x5a");
		}
		cache.ReadMemory(gpu_range, kRepeatSize);
		uint8_t guest_value = 0;
		const bool guest_read = Libs::LibKernel::Memory::TryReadBacking(gpu_range, &guest_value, 1);
		Check(gpu_name, guest_read && guest_value == 0x5a &&
		                    !cache.HasGpuDirtyBytes(gpu_range, kRepeatSize) &&
		                    gpu_owner.first == gpu.first,
		      "renderer readback did not clear GPU dirtiness and publish 0x5a to guest backing");
		if (g_failures == 0) {
			Pass(gpu_name, "GuestGpu readback clears dirty state before unmap; publication keeps owner");
		}

		// The tracker is deliberately address-keyed and does not observe VM remapping. Reuse the
		// exact guest address through the real fixed-map API: inherited history may select the
		// persistent path immediately, but the new backing must still be uploaded and visible.
		constexpr uint64_t kRemapSize = 64 * 1024;
		const uint64_t remap_old = MapGuestMemory(kRemapSize);
		const char* remap_name = "stream repeat: address reuse after fixed remap";
		Check(remap_name, remap_old != 0, "could not allocate remap test mapping");
		if (remap_old != 0) {
			set_threshold(1);
			cache.ResetStreamRepeatTestStats();
			GuestWrite(remap_old, kRepeatSize, 0x27);
			cache.ObtainBuffer(remap_old, kRepeatSize, false);
			cache.ObtainBuffer(remap_old, kRepeatSize, false);
			const int unmap_result =
			    Libs::LibKernel::Memory::KernelMunmap(remap_old, static_cast<size_t>(kRemapSize));
			Check(remap_name, unmap_result == 0, "first KernelMunmap returned " + std::to_string(unmap_result));
			void* remap_ptr = reinterpret_cast<void*>(remap_old);
			constexpr int kGuestReadWriteGpuReadWrite = 0x01 | 0x02 | 0x10 | 0x20;
			constexpr int kGuestMapFixed              = 0x10;
			const int remap_result = Libs::LibKernel::Memory::KernelMapFlexibleMemory(
			    &remap_ptr, static_cast<size_t>(kRemapSize), kGuestReadWriteGpuReadWrite, kGuestMapFixed);
			const uint64_t remap_new = reinterpret_cast<uint64_t>(remap_ptr);
			Check(remap_name, remap_result == 0 && remap_new == remap_old,
			      "fixed remap returned " + std::to_string(remap_result) + " at a different address");
			if (remap_result == 0 && remap_new == remap_old) {
				GuestWrite(remap_new, kRepeatSize, 0xbe);
				const auto remapped = cache.ObtainBuffer(remap_new, kRepeatSize, false);
				const auto remap_stats = cache.GetStreamRepeatTestStats();
				Check(remap_name, remapped.first != nullptr && !cache.IsStreamAllocation(remapped.first),
				      "reused address returned a stream allocation instead of the inherited fallback");
				if (remapped.first != nullptr) {
					const auto value = readback.ReadByte(context, remapped.first->Handle(), remapped.second);
					Check(remap_name, value == 0xbe,
					      "remapped backing read " + Hex(value) + ", expected 0xbe");
				}
				Check(remap_name, remap_stats.stream_bytes == kRepeatSize,
				      "remapped address did not inherit its tracker history");
				if (g_failures == 0) {
					Pass(remap_name, "history is address-keyed; remapped backing still reaches GPU correctly");
				}
			}
			if (remap_result == 0) {
				const int final_unmap =
				    Libs::LibKernel::Memory::KernelMunmap(remap_new, static_cast<size_t>(kRemapSize));
				Check(remap_name, final_unmap == 0,
				      "final KernelMunmap returned " + std::to_string(final_unmap));
			}
		}
	}

	readback.Destroy(context.graphic_ctx);

	// Release the guest mapping BEFORE the context is destroyed. The page manager tracks these
	// pages, and tearing it down while they are live is what produced
	// "PageManager destroyed with live page state" - a cleanup-order fault, not a missing
	// startup dependency, and it was masking the results printed above.
	{
		const char* name = "guest mapping released before teardown";
		const int   result =
		    Libs::LibKernel::Memory::KernelMunmap(guest, static_cast<size_t>(kGuestSize));
		Check(name, result == 0, "KernelMunmap returned " + std::to_string(result));
		if (result == 0) {
			Pass(name, "unmapped before the page manager is destroyed");
		}
	}

	// ------------------------------------------------------------------------------------
	// Orderly shutdown. vkDeviceWaitIdle completes submitted GPU work; it does NOT run the
	// scheduler's CPU-side deferred callbacks, so those are drained explicitly and VERIFIED
	// rather than assumed to have happened.
	{
		const char*       name = "deferred retirement callbacks actually run";
		std::atomic<bool> ran {false};
		scheduler.DeferOperation([&ran] { ran.store(true); });
		scheduler.FlushAndWait();
		scheduler.Finish();
		Check(name, ran.load(),
		      "a deferred operation never executed; the retirement path was NOT exercised");
		if (ran.load()) {
			Pass(name, "deferred callback observed after submit and drain");
		}
	}
	context.render_context->ShutdownGpu();
	if (context.graphic_ctx.device != nullptr &&
	    context.graphic_ctx.device.waitIdle() != vk::Result::eSuccess) {
		Check("orderly shutdown", false, "vkDeviceWaitIdle failed");
	}
	SDL_Window* window = context.window;
	delete context_storage;
	if (window != nullptr) {
		SDL_DestroyWindow(window);
	}
	SDL_Quit();

	if (g_failures != 0) {
		std::printf("\n%d harness check(s) FAILED\n", g_failures);
		return 1;
	}
	std::printf("\nall harness checks passed\n");
	return 0;
}
