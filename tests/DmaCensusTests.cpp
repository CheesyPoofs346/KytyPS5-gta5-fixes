// Offline checks for the DMA census counters.
//
// These exist to stop a gameplay capture being wasted. The compiled-SRT reporter once printed
// nothing for a whole run because its trigger sat on a path the run never took, and the run had to
// be repeated. The counters below are trivial, but "trivial and silently zero" is precisely the
// failure that costs a drive.
//
// What is NOT tested here: that DmaData actually calls DmaCensusAdd in a real frame. That needs the
// emulator running and is exactly what the capture itself will show.

#include "common/emulatorConfig.h"
#include "graphics/host_gpu/renderer/drawProfile.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
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

void TestDefaultOff() {
	// Reads the accessor without Config::Load, which needs Init() and a valid user profile. The
	// atomic behind it is what the emulator sees before any configuration is applied, so this is
	// the real default rather than a re-statement of the struct initialiser.
	const char* name = "census is off by default";
	Check(name, !Config::DmaCensusEnabled(), "dma_census defaulted to true");
	if (!Config::DmaCensusEnabled()) {
		Pass(name, "default false, so the CSV schema is unchanged unless asked for");
	}
}

void TestAccumulateAndTake() {
	const char* name = "accumulate then take resets";
	uint64_t    bytes = 0;
	uint64_t    ns    = 0;
	DmaCensusTake(&bytes, &ns);   // clear whatever a previous test left

	DmaCensusAdd(1000, 50);
	DmaCensusAdd(2500, 70);
	DmaCensusTake(&bytes, &ns);
	Check(name, bytes == 3500, "bytes accumulated to " + std::to_string(bytes) + ", expected 3500");
	Check(name, ns == 120, "ns accumulated to " + std::to_string(ns) + ", expected 120");

	uint64_t bytes2 = 1, ns2 = 1;
	DmaCensusTake(&bytes2, &ns2);
	Check(name, bytes2 == 0 && ns2 == 0, "take did not reset: bytes=" + std::to_string(bytes2) +
	                                         " ns=" + std::to_string(ns2));
	if (bytes == 3500 && ns == 120 && bytes2 == 0 && ns2 == 0) {
		Pass(name, "sums are exact and a take zeroes both counters");
	}
}

// The adds happen on the submission thread while the capture sample is closed elsewhere, so the
// counters are written and read concurrently. This checks no update is lost.
void TestConcurrentAdds() {
	const char* name = "no updates lost under concurrency";
	uint64_t    drain_bytes = 0;
	uint64_t    drain_ns    = 0;
	DmaCensusTake(&drain_bytes, &drain_ns);

	constexpr int kThreads = 8;
	constexpr int kAdds    = 20000;
	{
		std::vector<std::thread> pool;
		for (int t = 0; t < kThreads; t++) {
			pool.emplace_back([]() {
				for (int i = 0; i < kAdds; i++) {
					DmaCensusAdd(7, 3);
				}
			});
		}
		for (auto& thread: pool) {
			thread.join();
		}
	}
	uint64_t bytes = 0, ns = 0;
	DmaCensusTake(&bytes, &ns);
	const uint64_t want_bytes = static_cast<uint64_t>(kThreads) * kAdds * 7;
	const uint64_t want_ns    = static_cast<uint64_t>(kThreads) * kAdds * 3;
	Check(name, bytes == want_bytes,
	      "bytes " + std::to_string(bytes) + " != " + std::to_string(want_bytes));
	Check(name, ns == want_ns, "ns " + std::to_string(ns) + " != " + std::to_string(want_ns));
	if (bytes == want_bytes && ns == want_ns) {
		Pass(name, std::to_string(kThreads) + " threads x " + std::to_string(kAdds) +
		               " adds, exact totals");
	}
}

} // namespace

int main() {
	std::printf("DMA census counters:\n\n");
	TestDefaultOff();
	TestAccumulateAndTake();
	TestConcurrentAdds();
	if (g_failures != 0) {
		std::printf("\n%d DMA census check(s) FAILED\n", g_failures);
		return 1;
	}
	std::printf("\nall DMA census checks passed\n");
	return 0;
}
