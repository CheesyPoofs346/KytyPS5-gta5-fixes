#ifndef EMULATOR_SRC_LIBS_PADAUDIO_H_
#define EMULATOR_SRC_LIBS_PADAUDIO_H_

#include "common/common.h"

#include <SDL_hidapi.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace Kyty::Libs::Controller {

// Streams audio to a DualSense over Bluetooth.
//
// Over USB Windows exposes the controller as an audio endpoint and the normal output path covers it.
// Over Bluetooth it does not: the pad consumes 10 ms Opus frames carried in HID output report 0x36,
// so the samples a title writes to its pad-speaker port have to be resampled, encoded and paced out
// by hand. The pad consumes at roughly 45 kHz while the stream is nominally 48 kHz, which is what
// keeps the pitch right.
class PadAudioStream {
public:
	PadAudioStream() = default;
	~PadAudioStream();
	KYTY_CLASS_NO_COPY(PadAudioStream);

	// True once an encoder exists and the pad is reachable over HID.
	[[nodiscard]] bool Ready() const noexcept { return m_ready; }

	// Prepares the encoder and locates the pad. Safe to call repeatedly.
	bool Open(uint32_t freq, uint32_t channels);
	void Close();

	// Feeds interleaved 16-bit PCM. Whole 10 ms frames are encoded and sent; the remainder is kept
	// for the next call.
	void Submit(const void* samples, uint32_t frames, bool is_float);

private:
	static constexpr uint32_t OpusRate        = 48000;
	static constexpr uint32_t PadRate         = 45000;
	static constexpr uint32_t Channels        = 2;
	static constexpr uint32_t FrameSamples    = OpusRate / 100; // 10 ms
	static constexpr uint32_t OpusFrameBytes  = 200;            // 160 kbps CBR at 10 ms
	static constexpr uint32_t ReportSize      = 398;            // 0x36 report including its id

	void SendFrame(const uint8_t* opus, uint32_t size);
	// Encoding and the HID write both block; doing them on the thread a title feeds audio from
	// stalled the game to a couple of frames a second. They run here instead, paced at one 10 ms
	// frame per 10 ms, dropping rather than blocking when the queue runs long.
	void Pump();
	// True while a title is actually feeding audio. Streaming silence forever competes with the
	// rumble reports SDL writes to the same pad, so the stream idles out when nothing is playing.
	[[nodiscard]] bool Active() const;

	void*                m_encoder = nullptr; // OpusEncoder
	SDL_hid_device*      m_hid     = nullptr;
	bool                 m_ready   = false;
	uint32_t             m_seq     = 0;
	uint32_t             m_frame_counter = 0;
	uint32_t             m_src_freq = 0;
	uint32_t             m_src_channels = 0;
	double               m_resample_pos = 0.0;
	std::vector<int16_t> m_pending;      // resampled PCM awaiting a whole frame (worker only)
	std::deque<std::vector<int16_t>> m_queue; // frames handed over from the audio thread
	std::mutex           m_queue_mutex;
	std::thread          m_worker;
	std::atomic<bool>    m_running {false};
	std::atomic<uint64_t> m_last_audio_ms {0};
	std::atomic<uint64_t> m_frames_sent {0};
	bool                 m_playing = false;
};

} // namespace Kyty::Libs::Controller

#endif /* EMULATOR_SRC_LIBS_PADAUDIO_H_ */
