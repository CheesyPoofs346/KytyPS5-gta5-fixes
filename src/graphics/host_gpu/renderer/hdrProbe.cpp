#include "graphics/host_gpu/renderer/hdrProbe.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/image.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace Libs::Graphics {

namespace {

// IEEE half -> float.
float DecodeHalf(uint16_t bits) {
	const uint32_t sign     = (bits >> 15u) & 0x1u;
	const uint32_t exponent = (bits >> 10u) & 0x1fu;
	const uint32_t mantissa = bits & 0x3ffu;
	float          value    = 0.0f;
	if (exponent == 0) {
		value = std::ldexp(static_cast<float>(mantissa) / 1024.0f, -14);
	} else if (exponent == 31) {
		value = mantissa == 0 ? std::numeric_limits<float>::infinity()
		                      : std::numeric_limits<float>::quiet_NaN();
	} else {
		value = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
		                   static_cast<int>(exponent) - 15);
	}
	return sign != 0 ? -value : value;
}

// Unsigned packed float (no sign bit) as used by B10G11R11: 5 exponent bits, `mantissa_bits`
// mantissa bits, bias 15.
float DecodeUnsignedFloat(uint32_t bits, uint32_t mantissa_bits) {
	const uint32_t mantissa_max = 1u << mantissa_bits;
	const uint32_t mantissa     = bits & (mantissa_max - 1u);
	const uint32_t exponent     = (bits >> mantissa_bits) & 0x1fu;
	if (exponent == 0) {
		return std::ldexp(static_cast<float>(mantissa) / static_cast<float>(mantissa_max), -14);
	}
	if (exponent == 31) {
		return mantissa == 0 ? std::numeric_limits<float>::infinity()
		                     : std::numeric_limits<float>::quiet_NaN();
	}
	return std::ldexp(1.0f + static_cast<float>(mantissa) / static_cast<float>(mantissa_max),
	                  static_cast<int>(exponent) - 15);
}

float ReadF32(const uint8_t* pixel, uint32_t index) {
	float value = 0.0f;
	std::memcpy(&value, pixel + index * sizeof(float), sizeof(float));
	return value;
}

uint16_t ReadU16(const uint8_t* pixel, uint32_t index) {
	uint16_t value = 0;
	std::memcpy(&value, pixel + index * sizeof(uint16_t), sizeof(uint16_t));
	return value;
}

uint32_t ReadU32(const uint8_t* pixel) {
	uint32_t value = 0;
	std::memcpy(&value, pixel, sizeof(uint32_t));
	return value;
}

// Decodes one pixel into up to 4 channels. Returns the channel count written.
uint32_t DecodePixel(vk::Format format, const uint8_t* pixel, std::array<float, 4>& out) {
	switch (format) {
		case vk::Format::eD32SfloatS8Uint:
		case vk::Format::eD32Sfloat:
		case vk::Format::eR32Sfloat: out[0] = ReadF32(pixel, 0); return 1;
		case vk::Format::eR32G32Sfloat:
			out[0] = ReadF32(pixel, 0);
			out[1] = ReadF32(pixel, 1);
			return 2;
		case vk::Format::eR32G32B32A32Sfloat:
			for (uint32_t i = 0; i < 4; i++) {
				out[i] = ReadF32(pixel, i);
			}
			return 4;
		case vk::Format::eR16Sfloat: out[0] = DecodeHalf(ReadU16(pixel, 0)); return 1;
		case vk::Format::eR16G16Sfloat:
			out[0] = DecodeHalf(ReadU16(pixel, 0));
			out[1] = DecodeHalf(ReadU16(pixel, 1));
			return 2;
		case vk::Format::eR16G16B16A16Sfloat:
			for (uint32_t i = 0; i < 4; i++) {
				out[i] = DecodeHalf(ReadU16(pixel, i));
			}
			return 4;
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
			for (uint32_t i = 0; i < 4; i++) {
				out[i] = static_cast<float>(pixel[i]) / 255.0f;
			}
			return 4;
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
			// Report in LOGICAL RGBA order so a channel swap is visible rather than hidden by the
			// storage order.
			out[0] = static_cast<float>(pixel[2]) / 255.0f;
			out[1] = static_cast<float>(pixel[1]) / 255.0f;
			out[2] = static_cast<float>(pixel[0]) / 255.0f;
			out[3] = static_cast<float>(pixel[3]) / 255.0f;
			return 4;
		case vk::Format::eB10G11R11UfloatPack32: {
			const uint32_t packed = ReadU32(pixel);
			out[0]                = DecodeUnsignedFloat(packed & 0x7ffu, 6);
			out[1]                = DecodeUnsignedFloat((packed >> 11u) & 0x7ffu, 6);
			out[2]                = DecodeUnsignedFloat((packed >> 22u) & 0x3ffu, 5);
			return 3;
		}
		default: return 0;
	}
}

} // namespace

bool HdrProbe::IsFloatColorFormat(vk::Format format) noexcept {
	switch (format) {
		case vk::Format::eR16Sfloat:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR32G32B32A32Sfloat:
		case vk::Format::eB10G11R11UfloatPack32:
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb: return true;
		default: return false;
	}
}

uint32_t HdrProbe::FormatBytesPerPixel(vk::Format format) noexcept {
	switch (format) {
		case vk::Format::eD32SfloatS8Uint:
		case vk::Format::eD32Sfloat: return 4;
		case vk::Format::eR16Sfloat: return 2;
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR32Sfloat:
		case vk::Format::eB10G11R11UfloatPack32:
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb: return 4;
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR32G32Sfloat: return 8;
		case vk::Format::eR32G32B32A32Sfloat: return 16;
		default: return 0;
	}
}

void HdrProbe::Initialize(GraphicContext& graphics, CommandScheduler& scheduler) {
	m_enabled = Config::HdrProbeEnabled();
	if (!m_enabled) {
		return;
	}
	m_graphics  = &graphics;
	m_scheduler = &scheduler;
	m_interval  = std::max(Config::HdrProbeInterval(), 1u);
	m_start     = Config::HdrProbeStart();

	m_staging.usage           = vk::BufferUsageFlagBits::eTransferDst;
	m_staging.memory.property = vk::MemoryPropertyFlagBits::eHostVisible |
	                            vk::MemoryPropertyFlagBits::eHostCoherent;
	graphics.CreateBuffer(SlotBytes * SlotCount, m_staging);

	void* mapped = nullptr;
	graphics.MapMemory(m_staging.memory, mapped);
	m_mapped = static_cast<uint8_t*>(mapped);
	m_pending.reserve(SlotCount);

	LOGF("HdrProbe: enabled, probing every %" PRIu32 " frame(s), %" PRIu32 "x%" PRIu32
	     " centre crop, %" PRIu32 " slots\n",
	     m_interval, CropSize, CropSize, SlotCount);
}

bool HdrProbe::ArmFrame(uint32_t frame) {
	if (frame != m_frame) {
		// A frame boundary: publish whatever the previous frame recorded before reusing slots.
		Drain();
		m_frame = frame;
		m_pass  = 0;
		m_armed = frame >= m_start && ((frame - m_start) % m_interval) == 0;
	}
	return m_armed;
}

void HdrProbe::Capture(Image& image, uint32_t frame, uint32_t base_level,
                       uint32_t base_layer, bool self_bound, uint32_t export_mapping) {
	if (!m_enabled || m_mapped == nullptr || !ArmFrame(frame)) {
		return;
	}
	const auto format = image.backing.format;
	if (!IsFloatColorFormat(format)) {
		return;
	}
	const auto bytes_per_pixel = FormatBytesPerPixel(format);
	if (bytes_per_pixel == 0 || image.backing.image == nullptr) {
		return;
	}
	if (m_pending.size() >= SlotCount) {
		// Slot ring exhausted mid-frame: flush so later passes still get recorded.
		Drain();
		m_armed = true;
	}

	const uint32_t width  = std::max(image.backing.extent.width >> base_level, 1u);
	const uint32_t height = std::max(image.backing.extent.height >> base_level, 1u);
	// A horizontal band cannot be seen in a small centre square, so sample a tall narrow column:
	// 4 px wide by up to 1024 rows still fits one slot (4*1024*16 == SlotBytes).
	const uint32_t crop_w = std::min(4u, width);
	const uint32_t crop_h = std::min(1024u, height);

	Sample sample {};
	sample.guest_addr = image.info.data.address;
	sample.vk_image   = reinterpret_cast<uint64_t>(
	    static_cast<VkImage>(image.backing.image));
	sample.self_bound = self_bound;
	sample.export_map = export_mapping;
	sample.format     = format;
	sample.width      = width;
	sample.height     = height;
	sample.crop_w     = crop_w;
	sample.crop_h     = crop_h;
	sample.frame      = frame;
	sample.pass       = m_pass++;
	sample.offset     = static_cast<uint64_t>(m_pending.size()) * SlotBytes;

	vk::BufferImageCopy copy {};
	copy.bufferOffset                    = sample.offset;
	copy.bufferRowLength                 = 0;
	copy.bufferImageHeight               = 0;
	copy.imageSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
	copy.imageSubresource.mipLevel       = base_level;
	copy.imageSubresource.baseArrayLayer = base_layer;
	copy.imageSubresource.layerCount     = 1;
	copy.imageOffset = vk::Offset3D {0, static_cast<int32_t>((height - crop_h) / 2), 0};
	copy.imageExtent = vk::Extent3D {crop_w, crop_h, 1};

	const uint64_t copy_bytes =
	    static_cast<uint64_t>(crop_w) * crop_h * bytes_per_pixel;
	image.Download({&copy, 1}, m_staging.buffer, sample.offset, copy_bytes);
	m_pending.push_back(sample);
}

void HdrProbe::CaptureDepth(Image& image, uint32_t frame, uint32_t base_layer,
                            uint64_t guest_addr) {
	if (!m_enabled || m_mapped == nullptr || !ArmFrame(frame) || m_depth_frame == frame) {
		return;
	}
	const auto format = image.backing.format;
	if ((format != vk::Format::eD32Sfloat && format != vk::Format::eD32SfloatS8Uint) ||
	    image.backing.image == nullptr) {
		return;
	}
	if (m_pending.size() >= SlotCount) {
		Drain();
		m_armed = true;
	}

	const uint32_t width  = std::max(image.backing.extent.width, 1u);
	const uint32_t height = std::max(image.backing.extent.height, 1u);
	// Sample the FULL WIDTH as a short strip rather than a small centre square. A narrow centre
	// column cannot tell "the whole buffer was wiped" from "the centre happens to look at empty
	// space", and that ambiguity produced a wrong conclusion once already. 4 bytes per depth texel,
	// so keep width*rows within one slot.
	const uint32_t crop_w = std::min(width, static_cast<uint32_t>(SlotBytes / (4u * 6u)));
	const uint32_t crop_h = std::min(6u, height);

	Sample sample {};
	sample.guest_addr   = guest_addr;
	sample.vk_image     = reinterpret_cast<uint64_t>(static_cast<VkImage>(image.backing.image));
	sample.format       = format;
	sample.width        = width;
	sample.height       = height;
	sample.crop_w       = crop_w;
	sample.crop_h       = crop_h;
	sample.frame        = frame;
	sample.pass         = m_pass++;
	sample.offset       = static_cast<uint64_t>(m_pending.size()) * SlotBytes;
	sample.depth_sample = true;

	vk::BufferImageCopy copy {};
	copy.bufferOffset                    = sample.offset;
	copy.imageSubresource.aspectMask     = vk::ImageAspectFlagBits::eDepth;
	copy.imageSubresource.mipLevel       = 0;
	copy.imageSubresource.baseArrayLayer = base_layer;
	copy.imageSubresource.layerCount     = 1;
	copy.imageOffset = vk::Offset3D {static_cast<int32_t>((width - crop_w) / 2),
	                                 static_cast<int32_t>((height - crop_h) / 2), 0};
	copy.imageExtent = vk::Extent3D {crop_w, crop_h, 1};

	const uint64_t copy_bytes = static_cast<uint64_t>(crop_w) * crop_h * sizeof(float);
	image.Download({&copy, 1}, m_staging.buffer, sample.offset, copy_bytes);
	m_pending.push_back(sample);
	m_depth_frame = frame;
}

void HdrProbe::NoteDrawShader(uint64_t ps_addr, uint64_t vs_addr) {
	if (!m_enabled || !m_armed || m_pending.empty()) {
		return;
	}
	// Only the first draw of a pass is recorded; that is enough to name the shader that owns it.
	auto& sample = m_pending.back();
	sample.draw_count++;
	if (ps_addr == 0) {
		return;
	}
	for (uint32_t i = 0; i < sample.ps_count; i++) {
		if (sample.ps_addr[i] == ps_addr) {
			return;
		}
	}
	if (sample.ps_count < Sample::MaxShaders) {
		sample.ps_addr[sample.ps_count++] = ps_addr;
	}
}

void HdrProbe::NoteDepthClear(bool load_clear, bool meta_clear, float clear_value,
                              bool test_enable, bool write_enable,
                              uint32_t compare_op) {
	if (!m_enabled || !m_armed || m_pending.empty()) {
		return;
	}
	auto& sample       = m_pending.back();
	sample.load_clear  = load_clear;
	sample.meta_clear  = meta_clear;
	sample.clear_value = clear_value;
	sample.rtest       = test_enable;
	sample.rwrite      = write_enable;
	sample.rcompare    = compare_op;
}

void HdrProbe::NoteDrawDepth(bool z_enable, bool z_write, uint32_t zfunc, float min_depth,
                             float max_depth, uint64_t depth_addr) {
	if (!m_enabled || !m_armed || m_pending.empty()) {
		return;
	}
	auto& sample = m_pending.back();
	if (sample.depth_seen) {
		return;
	}
	sample.depth_seen = true;
	sample.z_enable   = z_enable;
	sample.z_write    = z_write;
	sample.zfunc      = zfunc;
	sample.min_depth  = min_depth;
	sample.max_depth  = max_depth;
	sample.depth_addr = depth_addr;
}

void HdrProbe::ReduceAndLog(const Sample& sample) const {
	const auto bytes_per_pixel = FormatBytesPerPixel(sample.format);
	if (bytes_per_pixel == 0) {
		return;
	}
	const uint8_t* base = m_mapped + sample.offset;

	float    min_value  = std::numeric_limits<float>::infinity();
	float    max_value  = -std::numeric_limits<float>::infinity();
	std::array<double, 3>   channel_sum {};
	std::array<uint64_t, 3> channel_count {};
	double   sum        = 0.0;
	uint64_t counted    = 0;
	uint64_t nonfinite  = 0;
	uint64_t negatives  = 0;

	for (uint32_t y = 0; y < sample.crop_h; y++) {
		for (uint32_t x = 0; x < sample.crop_w; x++) {
			const uint8_t* pixel =
			    base + (static_cast<uint64_t>(y) * sample.crop_w + x) * bytes_per_pixel;
			std::array<float, 4> channels {};
			const auto           count = DecodePixel(sample.format, pixel, channels);
			// Alpha is not part of the luminance question; only look at the colour channels.
			const auto colour_count = std::min(count, 3u);
			for (uint32_t c = 0; c < colour_count; c++) {
				const float value = channels[c];
				if (!std::isfinite(value)) {
					nonfinite++;
					continue;
				}
				if (value < 0.0f) {
					negatives++;
				}
				if (c < channel_sum.size()) {
					channel_sum[c] += static_cast<double>(value);
					channel_count[c]++;
				}
				min_value = std::min(min_value, value);
				max_value = std::max(max_value, value);
				sum += static_cast<double>(value);
				counted++;
			}
		}
	}


	// Vertical profile: the horizon band is a horizontal artifact, so a per-row mean over the crop
	// shows which target actually contains it and at what height, instead of hiding it in a single
	// average over the whole crop.
	if (!sample.depth_sample && sample.crop_h >= 8) {
		std::array<double, 32> band_rows {};
		const uint32_t         rows_per_bucket = std::max(sample.crop_h / 32u, 1u);
		for (uint32_t bucket = 0; bucket < 32; bucket++) {
			double   bucket_sum   = 0.0;
			uint64_t bucket_count = 0;
			for (uint32_t r = 0; r < rows_per_bucket; r++) {
				const uint32_t y = bucket * rows_per_bucket + r;
				if (y >= sample.crop_h) {
					break;
				}
				for (uint32_t x = 0; x < sample.crop_w; x++) {
					const uint8_t* px =
					    base + (static_cast<uint64_t>(y) * sample.crop_w + x) * bytes_per_pixel;
					std::array<float, 4> ch {};
					const auto           n = DecodePixel(sample.format, px, ch);
					for (uint32_t c = 0; c < std::min(n, 3u); c++) {
						if (std::isfinite(ch[c])) {
							bucket_sum += static_cast<double>(ch[c]);
							bucket_count++;
						}
					}
				}
			}
			band_rows[bucket] = bucket_count != 0 ? bucket_sum / bucket_count : 0.0;
		}
		// A horizon band is a NARROW spike: one or two buckets much brighter than BOTH neighbours.
		// A smooth gradient (ceiling to floor) is not a band, so compare locally rather than
		// globally, otherwise ordinary scene shading trips the detector.
		int    spike_at    = -1;
		double spike_ratio = 0.0;
		for (uint32_t i = 1; i + 1 < 32; i++) {
			const double neighbour = std::max(band_rows[i - 1], band_rows[i + 1]);
			if (neighbour > 0.001 && band_rows[i] > neighbour * 1.8 && band_rows[i] > 0.05) {
				const double ratio = band_rows[i] / neighbour;
				if (ratio > spike_ratio) {
					spike_ratio = ratio;
					spike_at    = static_cast<int>(i);
				}
			}
		}
		if (spike_at >= 0) {
			LOGF("BandSpike: frame=%" PRIu32 " pass=%" PRIu32 " addr=0x%010" PRIx64 " %ux%u fmt=%u written=%.1f%%"
			     " bucket=%d/32 ratio=%.2f ps=0x%010" PRIx64 "\n",
			     sample.frame, sample.pass, sample.guest_addr, sample.width, sample.height,
			     static_cast<uint32_t>(sample.format), spike_at, spike_ratio,
			     sample.ps_count > 0 ? sample.ps_addr[0] : 0ull);
		}
	}
	const double mean = counted != 0 ? sum / static_cast<double>(counted) : 0.0;
	if (counted == 0) {
		min_value = 0.0f;
		max_value = 0.0f;
	}
	if (sample.depth_sample) {
				uint64_t nonzero = 0;
		for (uint32_t y = 0; y < sample.crop_h; y++) {
			for (uint32_t x = 0; x < sample.crop_w; x++) {
				std::array<float, 4> ch {};
				const uint8_t* px = base + (static_cast<uint64_t>(y) * sample.crop_w + x) *
				                               bytes_per_pixel;
				if (DecodePixel(sample.format, px, ch) > 0 && ch[0] > 1e-7f) {
					nonzero++;
				}
			}
		}
		const double nonzero_pct =
		    sample.crop_w * sample.crop_h != 0
		        ? 100.0 * static_cast<double>(nonzero) / (sample.crop_w * sample.crop_h)
		        : 0.0;
		LOGF("DepthProbe: frame=%" PRIu32 " addr=0x%010" PRIx64 " %ux%u fmt=%u written=%.1f%%"
		     " min=%.9g mean=%.9g max=%.9g nonfinite=%" PRIu64 "\n",
		     sample.frame, sample.guest_addr, sample.width, sample.height,
		     static_cast<uint32_t>(sample.format), nonzero_pct,
		     static_cast<double>(min_value), mean, static_cast<double>(max_value), nonfinite);
		return;
	}

	char shader_list[128] = {};
	int  written              = 0;
	for (uint32_t i = 0; i < sample.ps_count && written >= 0 && written < 110; i++) {
		written += std::snprintf(shader_list + written, sizeof(shader_list) - written,
		                         "%s0x%010" PRIx64, i == 0 ? "" : ",", sample.ps_addr[i]);
	}

	LOGF("HdrProbe: frame=%" PRIu32 " pass=%" PRIu32 " addr=0x%010" PRIx64 " %ux%u fmt=%u"
	     " max=%.4f mean=%.4f nonfinite=%" PRIu64 " negative=%" PRIu64
	     " draws=%" PRIu32 " z=%d/%d func=%" PRIu32 " depth=[%.3f,%.3f]@0x%010" PRIx64 " dclear=%d/%d=%.3f resolved=%d/%d/%" PRIu32 " rgb=(%.3f,%.3f,%.3f) emap=0x%02" PRIx32 " ps=[%s]\n",
	     sample.frame, sample.pass, sample.guest_addr, sample.width, sample.height,
	     static_cast<uint32_t>(sample.format), static_cast<double>(max_value), mean, nonfinite,
	     negatives, sample.draw_count, static_cast<int>(sample.z_enable),
	     static_cast<int>(sample.z_write), sample.zfunc,
	     static_cast<double>(sample.min_depth), static_cast<double>(sample.max_depth),
	     sample.depth_addr, static_cast<int>(sample.load_clear),
	     static_cast<int>(sample.meta_clear),
	     static_cast<double>(sample.clear_value), static_cast<int>(sample.rtest),
	     static_cast<int>(sample.rwrite), sample.rcompare,
	     channel_count[0] != 0 ? channel_sum[0] / channel_count[0] : 0.0,
	     channel_count[1] != 0 ? channel_sum[1] / channel_count[1] : 0.0,
	     channel_count[2] != 0 ? channel_sum[2] / channel_count[2] : 0.0, sample.export_map,
	     shader_list);
}

void HdrProbe::Drain() {
	if (!m_enabled || m_pending.empty()) {
		return;
	}
	// The copies were recorded into the current command buffer; they must complete before the
	// staging memory holds anything meaningful.
	m_scheduler->FlushAndWait();
	for (const auto& sample: m_pending) {
		ReduceAndLog(sample);
	}
	m_pending.clear();
}

} // namespace Libs::Graphics
