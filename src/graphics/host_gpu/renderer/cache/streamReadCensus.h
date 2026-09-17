#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_STREAMREADCENSUS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_STREAMREADCENSUS_H_

#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdio>

namespace Libs::Graphics {

// Read-only diagnostic for the stream-buffer candidate. It deliberately has a separate
// address+size table from StreamRepeatTracker: enabling it must neither change fallback choices
// nor share their history. One instance is used per resolver thread, avoiding locks and
// cross-worker cache-line traffic while the census is enabled.
struct StreamReadCensusStats {
	uint64_t requests                    = 0;
	uint64_t writable                    = 0;
	uint64_t too_large                   = 0;
	uint64_t gpu_dirty                   = 0;
	uint64_t cpu_clean                   = 0;
	uint64_t eligible                    = 0;
	uint64_t eligible_bytes              = 0;
	uint64_t readable                    = 0;
	uint64_t readable_bytes              = 0;
	uint64_t unreadable                  = 0;
	uint64_t matching_repeats            = 0;
	uint64_t potentially_avoidable_bytes = 0;
	std::array<uint64_t, 16> size_bins {};
};

class StreamReadCensus {
public:
	void ObserveRequest(uint64_t vaddr, uint64_t size, bool is_written, bool is_gpu_dirty,
	                    bool is_cpu_dirty) noexcept {
		m_stats.requests++;
		if (is_written) {
			m_stats.writable++;
			return;
		}
		if (size > kMaxStreamSize) {
			m_stats.too_large++;
			return;
		}
		if (is_gpu_dirty) {
			m_stats.gpu_dirty++;
			return;
		}
		if (!is_cpu_dirty) {
			m_stats.cpu_clean++;
			return;
		}
		m_stats.eligible++;
		m_stats.eligible_bytes += size;
		m_stats.size_bins[SizeBucket(size)]++;
		(void)vaddr;
	}

	// Call exactly once after TryReadBacking for an eligible request. Only successful reads enter
	// the repeat table, so potentially_avoidable_bytes never counts a range that could not stream.
	void ObserveReadResult(uint64_t vaddr, uint64_t size, bool readable) noexcept {
		if (!readable) {
			m_stats.unreadable++;
			return;
		}
		m_stats.readable++;
		m_stats.readable_bytes += size;
		auto& slot = m_seen[Hash(vaddr, size)];
		if (slot.vaddr == vaddr && slot.size == size && slot.seen) {
			m_stats.matching_repeats++;
			m_stats.potentially_avoidable_bytes += size;
		} else {
			slot = {vaddr, size, true};
		}
	}

	[[nodiscard]] bool ShouldReport() noexcept {
		if (m_stats.requests < m_next_report) {
			return false;
		}
		m_next_report += kReportInterval;
		return true;
	}

	void Report() const noexcept {
		std::printf(
		    "StreamReadCensus[req=%" PRIu64 " eligible=%" PRIu64 " eligible_bytes=%" PRIu64
		    " readable=%" PRIu64 " readable_bytes=%" PRIu64 " repeats=%" PRIu64
		    " potential_bytes=%" PRIu64 " fallback(write=%" PRIu64 " large=%" PRIu64
		    " gpu=%" PRIu64 " clean=%" PRIu64 " unreadable=%" PRIu64 ")]\n",
		    m_stats.requests, m_stats.eligible, m_stats.eligible_bytes, m_stats.readable,
		    m_stats.readable_bytes, m_stats.matching_repeats, m_stats.potentially_avoidable_bytes,
		    m_stats.writable, m_stats.too_large, m_stats.gpu_dirty, m_stats.cpu_clean,
		    m_stats.unreadable);
		std::printf("StreamReadCensusSizeBins[2^n..2^(n+1)-1]:");
		for (const auto count: m_stats.size_bins) {
			std::printf(" %" PRIu64, count);
		}
		std::printf("\n");
		std::fflush(stdout);
	}

	[[nodiscard]] const StreamReadCensusStats& Stats() const noexcept { return m_stats; }

private:
	static constexpr uint64_t kMaxStreamSize  = 16 * 1024;
	static constexpr uint64_t kReportInterval = 400000;
	static constexpr size_t   kSlots          = 8192;

	struct SeenSlot {
		uint64_t vaddr = 0;
		uint64_t size  = 0;
		bool     seen  = false;
	};

	[[nodiscard]] static size_t Hash(uint64_t vaddr, uint64_t size) noexcept {
		return static_cast<size_t>(((vaddr >> 4u) ^ (size * 0x9E3779B97F4A7C15ull)) >> 48u) &
		       (kSlots - 1);
	}
	[[nodiscard]] static size_t SizeBucket(uint64_t size) noexcept {
		size_t bucket = 0;
		while (size > 1 && bucket + 1 < 16) {
			size >>= 1u;
			bucket++;
		}
		return bucket;
	}

	StreamReadCensusStats            m_stats {};
	std::array<SeenSlot, kSlots>     m_seen {};
	uint64_t                         m_next_report = kReportInterval;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_STREAMREADCENSUS_H_
