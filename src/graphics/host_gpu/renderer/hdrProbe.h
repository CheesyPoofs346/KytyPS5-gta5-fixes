#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_HDRPROBE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_HDRPROBE_H_

#include "common/common.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <vector>

namespace Libs::Graphics {

class Image;
class CommandScheduler;

// Diagnostic instrument for the HDR runaway-gain hunt.
//
// As each floating-point color target is bound, this copies a small centre crop of its current
// contents into a host-visible staging buffer. On drain the crops are reduced on the CPU and one
// line per pass is logged with the max and mean channel values. A pass whose input has already
// left a sane range identifies the producer of the runaway, without guessing.
//
// The probe reads the target's contents *before* the pass renders into it, so the reported
// numbers are the values entering that pass.
class HdrProbe {
public:
	HdrProbe() = default;
	KYTY_CLASS_NO_COPY(HdrProbe);

	// True for the float color formats GTA5 uses for its HDR chain.
	[[nodiscard]] static bool IsFloatColorFormat(vk::Format format) noexcept;

	void               Initialize(GraphicContext& graphics, CommandScheduler& scheduler);
	[[nodiscard]] bool Enabled() const noexcept { return m_enabled; }

	// Records a crop copy of `image` when the current frame is armed. Cheap no-op otherwise, so
	// it is safe to call for every bound color target.
	void Capture(Image& image, uint32_t frame, uint32_t base_level, uint32_t base_layer,
	             bool self_bound, uint32_t export_mapping);
	// Captures the current depth attachment before one identified draw. This is intentionally
	// one sample per frame: it is a diagnostics path for a depth-rejection investigation.
	void CaptureDepth(Image& image, uint32_t frame, uint32_t base_layer, uint64_t guest_addr);

	// Waits for recorded work to finish, then reduces and logs every pending crop.

	void Drain();

	// Records which guest pixel shader ran in the pass most recently captured, so a probe line
	// can be traced back to the shader that produced the values.
	void NoteDrawShader(uint64_t ps_addr, uint64_t vs_addr);

	// Records depth state for the pass most recently captured.
	// Records how the depth attachment for this pass is being cleared.
	void NoteDepthClear(bool load_clear, bool meta_clear, float clear_value, bool test_enable,
	                    bool write_enable, uint32_t compare_op);

	void NoteDrawDepth(bool z_enable, bool z_write, uint32_t zfunc, float min_depth,
	                   float max_depth, uint64_t depth_addr);

private:
	struct Sample {
		uint64_t   guest_addr = 0;
		vk::Format format     = vk::Format::eUndefined;
		uint32_t   width      = 0;
		uint32_t   height     = 0;
		uint32_t   crop_w     = 0;
		uint32_t   crop_h     = 0;
		uint32_t   frame      = 0;
		uint32_t   pass       = 0;
		uint64_t   offset     = 0;
		uint64_t   vk_image   = 0;
		bool       depth_sample = false;
		bool       self_bound = false;
		uint32_t   export_map = 0;
		static constexpr uint32_t MaxShaders = 6;
		uint64_t   ps_addr[MaxShaders] = {};
		uint32_t   ps_count   = 0;
		uint32_t   draw_count = 0;
		bool       depth_seen = false;
		bool       z_enable   = false;
		bool       z_write    = false;
		uint32_t   zfunc      = 0;
		float      min_depth  = 0.0f;
		float      max_depth  = 0.0f;
		uint64_t   depth_addr  = 0;
		bool       load_clear  = false;
		bool       meta_clear  = false;
		float      clear_value  = -1.0f;
		bool       rtest        = false;
		bool       rwrite       = false;
		uint32_t   rcompare     = 0;
		uint64_t   vs_addr    = 0;
	};

	// A 64x64 centre crop is plenty to separate "a few units" from "hundreds", and keeps the
	// staging buffer small enough that probing does not perturb what it measures.
	static constexpr uint32_t CropSize  = 64;
	static constexpr uint64_t SlotBytes = static_cast<uint64_t>(CropSize) * CropSize * 16;
	static constexpr uint32_t SlotCount = 64;

	[[nodiscard]] static uint32_t FormatBytesPerPixel(vk::Format format) noexcept;
	[[nodiscard]] bool ArmFrame(uint32_t frame);
	void                          ReduceAndLog(const Sample& sample) const;

	GraphicContext*     m_graphics  = nullptr;
	CommandScheduler*   m_scheduler = nullptr;
	VulkanBuffer        m_staging;
	uint8_t*            m_mapped = nullptr;
	std::vector<Sample> m_pending;
	bool                m_enabled  = false;
	uint32_t            m_interval = 60;
	uint32_t            m_start    = 0;
	uint32_t            m_frame    = UINT32_MAX;
	uint32_t            m_depth_frame = UINT32_MAX;
	uint32_t            m_pass     = 0;
	bool                m_armed    = false;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_HDRPROBE_H_ */
