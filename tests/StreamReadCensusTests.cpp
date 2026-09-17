#include "graphics/host_gpu/renderer/cache/streamReadCensus.h"

#include <cstdio>

namespace {

int failures = 0;

void Check(bool value, const char* message) {
	if (!value) {
		std::printf("[FAIL] %s\n", message);
		failures++;
	}
}

} // namespace

int main() {
	Libs::Graphics::StreamReadCensus census;
	census.ObserveRequest(0x1000, 64, true, false, false);   // writable
	census.ObserveRequest(0x1000, 32768, false, false, false); // too large
	census.ObserveRequest(0x1000, 64, false, true, false);   // GPU dirty
	census.ObserveRequest(0x1000, 64, false, false, false);  // CPU clean

	census.ObserveRequest(0x2000, 256, false, false, true);
	census.ObserveReadResult(0x2000, 256, true);
	census.ObserveRequest(0x2000, 256, false, false, true);
	census.ObserveReadResult(0x2000, 256, true);
	census.ObserveRequest(0x2000, 512, false, false, true);
	census.ObserveReadResult(0x2000, 512, true); // Different size: not a repeat of the 256 B key.
	census.ObserveRequest(0x3000, 128, false, false, true);
	census.ObserveReadResult(0x3000, 128, false);

	const auto stats = census.Stats();
	Check(stats.writable == 1 && stats.too_large == 1 && stats.gpu_dirty == 1 && stats.cpu_clean == 1,
	      "gate fallback reasons were not counted independently");
	Check(stats.eligible == 4 && stats.eligible_bytes == 1152,
	      "eligible requests or bytes were miscounted");
	Check(stats.readable == 3 && stats.readable_bytes == 1024 && stats.unreadable == 1,
	      "readable/unreadable outcomes were miscounted");
	Check(stats.matching_repeats == 1 && stats.potentially_avoidable_bytes == 256,
	      "only matching address-and-size repeats may contribute avoidable bytes");
	if (failures != 0) {
		std::printf("%d stream-read census check(s) failed\n", failures);
		return 1;
	}
	std::printf("stream-read census checks passed\n");
	return 0;
}
