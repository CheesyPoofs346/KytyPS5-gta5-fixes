#include "common/emulatorConfig.h"
#include "libs/padAudio.h"

#include "common/logging/log.h"

#include <SDL_hidapi.h>
#include <SDL_error.h>
#include <opus.h>

#include <algorithm>
#include <cstdio>
#include <chrono>
#include <cstring>

namespace Kyty::Libs::Controller {

namespace {

constexpr uint16_t SonyVendor      = 0x054c;
constexpr uint16_t DualSenseId     = 0x0ce6;
constexpr uint16_t DualSenseEdgeId = 0x0df2;

// The pad checks a CRC32 over an 0xA2 HID header byte followed by the report, same as its other
// Bluetooth output reports.
uint32_t Crc32(const uint8_t* data, size_t size, uint32_t crc) {
	crc = ~crc;
	for (size_t i = 0; i < size; i++) {
		crc ^= data[i];
		for (int bit = 0; bit < 8; bit++) {
			crc = (crc >> 1u) ^ (0xEDB88320u & (0u - (crc & 1u)));
		}
	}
	return ~crc;
}

} // namespace

PadAudioStream::~PadAudioStream() {
	Close();
}

bool PadAudioStream::Open(uint32_t freq, uint32_t channels) {
	if (m_ready) {
		return true;
	}
	if (freq == 0 || channels == 0) {
		return false;
	}
	m_src_freq     = freq;
	m_src_channels = channels;

	for (const auto product: {DualSenseId, DualSenseEdgeId}) {
		m_hid = SDL_hid_open(SonyVendor, product, nullptr);
		if (m_hid != nullptr) {
			break;
		}
	}
	if (m_hid == nullptr) {
		::printf("PadAudio: no DualSense on HID: %s\n", SDL_GetError());
		return false;
	}

	int          error   = OPUS_OK;
	OpusEncoder* encoder = opus_encoder_create(static_cast<opus_int32>(OpusRate),
	                                           static_cast<int>(Channels),
	                                           OPUS_APPLICATION_RESTRICTED_LOWDELAY, &error);
	if (encoder == nullptr || error != OPUS_OK) {
		::printf("PadAudio: opus_encoder_create failed (%d)\n", error);
		SDL_hid_close(m_hid);
		m_hid = nullptr;
		return false;
	}
	// The pad expects a constant 200 bytes per 10 ms frame.
	opus_encoder_ctl(encoder, OPUS_SET_BITRATE(160000));
	opus_encoder_ctl(encoder, OPUS_SET_VBR(0));
	opus_encoder_ctl(encoder, OPUS_SET_SIGNAL(OPUS_SIGNAL_MUSIC));

	m_encoder = encoder;
	m_ready   = true;
	m_running.store(true, std::memory_order_release);
	m_worker = std::thread([this] { Pump(); });
	::printf("PadAudio: streaming to the controller over Bluetooth (source %u Hz, %u ch)\n", freq,
	         channels);
	return true;
}

void PadAudioStream::Close() {
	m_running.store(false, std::memory_order_release);
	if (m_worker.joinable()) {
		m_worker.join();
	}
	if (m_encoder != nullptr) {
		opus_encoder_destroy(static_cast<OpusEncoder*>(m_encoder));
		m_encoder = nullptr;
	}
	if (m_hid != nullptr) {
		SDL_hid_close(m_hid);
		m_hid = nullptr;
	}
	m_pending.clear();
	m_ready = false;
}

void PadAudioStream::SendFrame(const uint8_t* opus, uint32_t size) {
	if (m_hid == nullptr || opus == nullptr || size == 0) {
		return;
	}
	// Report 0x36 carries three sub-packets. The control block enables the speaker and sets its
	// volume without touching the light bar, player LEDs or trigger state; the audio block carries
	// routing tags, a frame counter and the 200-byte Opus frame; the haptic block carries PCM for
	// the motors, left silent here. Layout follows the DualSense web tester.
	uint8_t report[ReportSize] = {};
	report[0]                  = 0x36; // report id
	uint8_t* p                 = report + 1;

	p[0] = static_cast<uint8_t>((m_seq & 0x0fu) << 4u);
	m_seq++;

	// control sub-packet: speaker volume (valid_flag0 bit5) + audio control (bit7)
	p[1]  = 0x90;
	p[2]  = 0x3f;
	p[3]  = 0xa0;
	p[4]  = 0x00;
	p[8]  = static_cast<uint8_t>(Config::PadSpeakerVolume()); // speaker volume
	p[10] = 0x09; // audio control

	// audio sub-packet
	p[66] = 0x91;
	p[67] = 0x07;
	p[68] = 0xfe;
	p[69] = 0x40;
	p[70] = 0x40;
	p[71] = 0x40;
	p[72] = 0x40;
	p[73] = 0x40;
	p[74] = static_cast<uint8_t>(m_frame_counter & 0xffu);
	m_frame_counter++;
	p[75] = 0x93; // route to the speaker
	p[76] = 0xc8; // 200-byte Opus frame
	std::memcpy(p + 77, opus, std::min<uint32_t>(size, OpusFrameBytes));

	// haptic sub-packet, silent
	p[277] = 0x92;
	p[278] = 0x40;

	const uint8_t  header = 0xa2;
	const uint32_t crc    = Crc32(report, ReportSize - sizeof(uint32_t), Crc32(&header, 1, 0));
	std::memcpy(report + ReportSize - sizeof(crc), &crc, sizeof(crc));

	if (SDL_hid_write(m_hid, report, sizeof(report)) < 0) {
		static uint32_t fail_log = 0;
		if (fail_log++ < 5) {
			::printf("PadAudio: HID write failed: %s\n", SDL_GetError());
		}
	}
}

bool PadAudioStream::Active() const {
	const auto last = m_last_audio_ms.load(std::memory_order_acquire);
	if (last == 0) {
		return false;
	}
	const auto now = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(
	        std::chrono::steady_clock::now().time_since_epoch())
	        .count());
	return now - last < 1000;
}

void PadAudioStream::Submit(const void* samples, uint32_t frames, bool is_float) {
	if (!m_ready || samples == nullptr || frames == 0) {
		return;
	}
	// Resample to the rate the pad consumes and force stereo, then hand the block to the worker.
	std::vector<int16_t> block;
	block.reserve(static_cast<size_t>(frames) * 2);
	const double step =
	    static_cast<double>(m_src_freq) / static_cast<double>(Config::PadSpeakerRate());
	double       pos  = m_resample_pos;
	while (pos < static_cast<double>(frames)) {
		const auto index = static_cast<uint32_t>(pos);
		if (index >= frames) {
			break;
		}
		// A float port carries -1..1 samples; reading those as int16 is pure noise, which is what
		// the pad played back.
		// Interpolate between neighbouring samples. Picking the nearest one aliases badly at these
		// ratios and is what made speech sound robotic.
		const auto   next_index = std::min<uint32_t>(index + 1, frames - 1);
		const auto   frac       = static_cast<float>(pos - static_cast<double>(index));
		const auto   fetch      = [&](uint32_t at, uint32_t channel) -> float {
            const auto offset = static_cast<size_t>(at) * m_src_channels +
                                (m_src_channels > 1 ? channel : 0);
            if (is_float) {
                return std::clamp(static_cast<const float*>(samples)[offset], -1.0f, 1.0f);
            }
            return static_cast<float>(static_cast<const int16_t*>(samples)[offset]) / 32768.0f;
		};
		const auto lerp = [&](uint32_t channel) {
			const auto a = fetch(index, channel);
			const auto b = fetch(next_index, channel);
			return std::clamp(a + (b - a) * frac, -1.0f, 1.0f);
		};
		const auto left  = static_cast<int16_t>(lerp(0) * 32767.0f);
		const auto right = static_cast<int16_t>(lerp(1) * 32767.0f);
		block.push_back(left);
		block.push_back(right);
		pos += step;
	}
	m_resample_pos = pos - static_cast<double>(frames);
	if (m_resample_pos < 0.0) {
		m_resample_pos = 0.0;
	}
	if (block.empty()) {
		return;
	}
	// Measure instead of assume: how many source samples per second the title actually supplies,
	// and how many frames per second we emit. The ratio between them is what sets the pitch.
	{
		static uint64_t window_start = 0;
		static uint64_t src_frames   = 0;
		const auto      now_ms       = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
		src_frames += frames;
		if (window_start == 0) {
			window_start = now_ms;
		} else if (now_ms - window_start >= 1000) {
			::printf("PadAudio: source %llu samples/s (port says %u Hz), sent %llu frames/s\n",
			         static_cast<unsigned long long>(src_frames * 1000u / (now_ms - window_start)),
			         m_src_freq,
			         static_cast<unsigned long long>(m_frames_sent * 1000u / (now_ms - window_start)));
			window_start   = now_ms;
			src_frames     = 0;
			m_frames_sent  = 0;
		}
	}
	m_last_audio_ms.store(static_cast<uint64_t>(
	                          std::chrono::duration_cast<std::chrono::milliseconds>(
	                              std::chrono::steady_clock::now().time_since_epoch())
	                              .count()),
	                      std::memory_order_release);
	std::lock_guard lock(m_queue_mutex);
	// Roughly a second of audio is plenty; beyond that the pad is not keeping up and older audio is
	// no longer worth playing.
	if (m_queue.size() > 100) {
		m_queue.pop_front();
	}
	m_queue.push_back(std::move(block));
}

void PadAudioStream::Pump() {
	const size_t frame_values = static_cast<size_t>(FrameSamples) * Channels;
	auto         next         = std::chrono::steady_clock::now();
	while (m_running.load(std::memory_order_acquire)) {
		std::vector<int16_t> block;
		{
			std::lock_guard lock(m_queue_mutex);
			if (!m_queue.empty()) {
				block = std::move(m_queue.front());
				m_queue.pop_front();
			}
		}
		if (!block.empty()) {
			m_pending.insert(m_pending.end(), block.begin(), block.end());
		}
		// The pad expects an unbroken 100 Hz stream: the control sub-packet that keeps its speaker
		// enabled rides along with every frame, so going quiet closes the audio path. Send silence
		// when the title has nothing queued rather than stopping.
		if (!Active()) {
			next = std::chrono::steady_clock::now();
			std::this_thread::sleep_for(std::chrono::milliseconds(20));
			continue;
		}
		// Hold a few frames before starting so bursty submissions do not leave holes mid-sentence.
		// Emitting a silent frame whenever a whole one was not ready is what made speech choppy.
		const size_t PrerollFrames = std::max<size_t>(1, Config::PadSpeakerPreroll());
		if (!m_playing && m_pending.size() < frame_values * PrerollFrames) {
			next = std::chrono::steady_clock::now();
			std::this_thread::sleep_for(std::chrono::milliseconds(2));
			continue;
		}
		m_playing = true;
		if (m_pending.size() < frame_values) {
			// Ran dry: go back to buffering rather than stuttering silence into the stream.
			m_playing = false;
			continue;
		}
		uint8_t              encoded[OpusFrameBytes] = {};
		std::vector<int16_t> frame(frame_values, 0);
		std::copy(m_pending.begin(), m_pending.begin() + static_cast<std::ptrdiff_t>(frame_values),
		          frame.begin());
		m_pending.erase(m_pending.begin(),
		                m_pending.begin() + static_cast<std::ptrdiff_t>(frame_values));
		const auto written =
		    opus_encode(static_cast<OpusEncoder*>(m_encoder), frame.data(),
		                static_cast<int>(FrameSamples), encoded,
		                static_cast<opus_int32>(sizeof(encoded)));
		if (written > 0) {
			SendFrame(encoded, static_cast<uint32_t>(written));
			m_frames_sent++;
		}
		// Sleeping 10ms *after* the work makes each cycle longer than a frame, so the pad starves and
		// plays noise. Hold a deadline instead so frames leave at a true 100 Hz.
		// A frame holds FrameSamples samples that the pad plays at its own rate, so it lasts
		// FrameSamples/rate seconds - about 10.67ms at 45kHz. Sending every 10ms overfeeds it.
		const auto frame_us = static_cast<int64_t>((1000000.0 * FrameSamples) /
		                                           static_cast<double>(Config::PadSpeakerRate()));
		next += std::chrono::microseconds(frame_us);
		const auto now = std::chrono::steady_clock::now();
		if (next > now) {
			std::this_thread::sleep_until(next);
		} else {
			next = now; // fell behind; do not try to catch up in a burst
		}
	}
}

} // namespace Kyty::Libs::Controller
