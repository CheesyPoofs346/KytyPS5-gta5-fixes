// Focused coverage for the single-containing-range fast path in TryTransferBacking
// (--backing-fast-path). The existing suites exercise none of these paths, so passing them says
// nothing about this change.
//
// Every case is run TWICE on the same build - once with the fast path off, once with it on - and
// the results must agree exactly. Equivalence with the established two-pass path is the property
// that matters; asserting the fast path's behaviour in isolation would only restate its own code.
//
// Cases: contained, spanning two mappings, unmapped, invalid range, and no-partial-copy on
// failure.

#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint64_t SceKernelPageSize       = 0x4000;
constexpr int      SceKernelProtCpuRw      = 0x02;
constexpr int      SceKernelMapFixed       = 0x10;
constexpr int      SceKernelMapNoOverwrite = 0x80;
constexpr int      SceKernelMtypeC         = 3;

int g_failures = 0;

void Fail(const char* test, const std::string& message) {
	std::printf("[FAIL]    %-52s %s\n", test, message.c_str());
	g_failures++;
}

void Check(const char* test, bool value, const std::string& message) {
	if (!value) {
		Fail(test, message);
	}
}

// Runs `body` with the fast path forced off and then on, so both paths see identical input.
template <typename Body>
void BothPaths(const char* test, Body&& body) {
	for (const bool fast: {false, true}) {
		Config::ConfigOptions options {};
		options.backing_fast_path = fast;
		Config::Load(options);
		Check(test, Config::BackingFastPath() == fast, "flag did not take effect");
		body(fast);
	}
}

// A mapped region of two adjacent pages, so a request can be made to span a mapping boundary.
struct MappedPages {
	uint64_t vaddr = 0;
	int64_t  phys  = -1;
	bool     ok    = false;
};

MappedPages MapTwoPages(const char* test) {
	MappedPages out;
	if (Libs::LibKernel::Memory::KernelAllocateDirectMemory(
	        0, Libs::LibKernel::Memory::KernelGetDirectMemorySize(), SceKernelPageSize * 2,
	        SceKernelPageSize, SceKernelMtypeC, &out.phys) != 0) {
		Fail(test, "KernelAllocateDirectMemory failed");
		return out;
	}
	void* addr = nullptr;
	if (Libs::LibKernel::Memory::KernelMapNamedDirectMemory(
	        &addr, SceKernelPageSize * 2, SceKernelProtCpuRw, 0, out.phys, SceKernelPageSize,
	        "fastpath_test") != 0) {
		Fail(test, "KernelMapNamedDirectMemory failed");
		return out;
	}
	out.vaddr = reinterpret_cast<uint64_t>(addr);
	out.ok    = out.vaddr != 0;
	return out;
}

// A request entirely inside one mapping: the fast path's target case.
void TestContainedRange() {
	const char* test = "ContainedRange";
	auto        m    = MapTwoPages(test);
	if (!m.ok) {
		return;
	}

	std::vector<uint8_t> pattern(1024);
	for (size_t i = 0; i < pattern.size(); ++i) {
		pattern[i] = static_cast<uint8_t>(i * 7 + 3);
	}
	Check(test, Libs::LibKernel::Memory::TryWriteBacking(m.vaddr + 64, pattern.data(), 1024),
	      "TryWriteBacking failed");

	std::vector<std::vector<uint8_t>> results;
	BothPaths(test, [&](bool fast) {
		std::vector<uint8_t> got(1024, 0xCD);
		const bool           ok = Libs::LibKernel::Memory::TryReadBacking(m.vaddr + 64, got.data(),
		                                                                 got.size());
		Check(test, ok, std::string("TryReadBacking returned false (fast=") +
		                    (fast ? "true" : "false") + ")");
		results.push_back(std::move(got));
	});
	Check(test, results.size() == 2 && results[0] == results[1],
	      "fast path and two-pass path returned different bytes");
	Check(test, results.size() == 2 && results[1] == pattern, "bytes did not match what was written");

	std::printf("[host]    %-52s ok\n", test);
}

// A request crossing a mapping boundary must fall back to the two-pass path and behave as before.
void TestSpanningRange() {
	const char* test = "SpanningRange";
	auto        m    = MapTwoPages(test);
	if (!m.ok) {
		return;
	}

	// Straddles the page boundary between the two mapped pages.
	const auto           start = m.vaddr + SceKernelPageSize - 512;
	std::vector<uint8_t> pattern(1024);
	for (size_t i = 0; i < pattern.size(); ++i) {
		pattern[i] = static_cast<uint8_t>(255 - (i % 251));
	}
	Check(test, Libs::LibKernel::Memory::TryWriteBacking(start, pattern.data(), pattern.size()),
	      "TryWriteBacking across boundary failed");

	std::vector<std::vector<uint8_t>> results;
	std::vector<bool>                 oks;
	BothPaths(test, [&](bool) {
		std::vector<uint8_t> got(1024, 0xAB);
		oks.push_back(Libs::LibKernel::Memory::TryReadBacking(start, got.data(), got.size()));
		results.push_back(std::move(got));
	});
	Check(test, oks.size() == 2 && oks[0] == oks[1],
	      "fast path and two-pass path disagreed on success for a spanning range");
	Check(test, results.size() == 2 && results[0] == results[1],
	      "fast path and two-pass path returned different bytes for a spanning range");

	std::printf("[host]    %-52s ok\n", test);
}

// An address with no mapping must fail identically on both paths and write nothing.
void TestUnmappedRange() {
	const char*          test = "UnmappedRange";
	constexpr uint64_t   kUnmapped = 0x7f0000000000ull;
	std::vector<bool>    oks;
	std::vector<uint8_t> untouched;

	BothPaths(test, [&](bool) {
		std::vector<uint8_t> got(512, 0x5A);
		const bool ok = Libs::LibKernel::Memory::TryReadBacking(kUnmapped, got.data(), got.size());
		oks.push_back(ok);
		// NO-PARTIAL-COPY: a failed transfer must leave the destination untouched.
		bool all_sentinel = true;
		for (const auto byte: got) {
			if (byte != 0x5A) {
				all_sentinel = false;
				break;
			}
		}
		Check(test, !ok, "TryReadBacking unexpectedly succeeded on an unmapped address");
		Check(test, all_sentinel, "failed transfer wrote bytes into the destination");
		untouched.push_back(all_sentinel ? 1 : 0);
	});
	Check(test, oks.size() == 2 && oks[0] == oks[1],
	      "fast path and two-pass path disagreed on an unmapped address");

	std::printf("[host]    %-52s ok\n", test);
}

// Degenerate and overflowing ranges must be rejected by both paths, with nothing written.
void TestInvalidRanges() {
	const char* test = "InvalidRanges";
	auto        m    = MapTwoPages(test);
	if (!m.ok) {
		return;
	}

	BothPaths(test, [&](bool) {
		uint8_t byte = 0x11;
		Check(test, !Libs::LibKernel::Memory::TryReadBacking(m.vaddr, &byte, 0),
		      "zero-size read was accepted");
		Check(test, byte == 0x11, "zero-size read wrote a byte");

		std::vector<uint8_t> got(64, 0x22);
		// vaddr + size overflows 64 bits.
		Check(test,
		      !Libs::LibKernel::Memory::TryReadBacking(UINT64_MAX - 8, got.data(), got.size()),
		      "overflowing range was accepted");
		bool untouched = true;
		for (const auto b: got) {
			if (b != 0x22) {
				untouched = false;
				break;
			}
		}
		Check(test, untouched, "overflowing range wrote bytes");

		Check(test, !Libs::LibKernel::Memory::TryReadBacking(m.vaddr, nullptr, 16),
		      "null destination was accepted");
	});

	std::printf("[host]    %-52s ok\n", test);
}

// A read running off the end of a mapping must fail and write nothing, on both paths. This is the
// case where a naive fast path would validate only the first entry and then copy anyway.
void TestNoPartialCopyPastMappingEnd() {
	const char* test = "NoPartialCopyPastMappingEnd";
	auto        m    = MapTwoPages(test);
	if (!m.ok) {
		return;
	}

	// Starts inside the mapping and runs past its end.
	const auto        start = m.vaddr + SceKernelPageSize * 2 - 256;
	std::vector<bool> oks;
	BothPaths(test, [&](bool fast) {
		std::vector<uint8_t> got(4096, 0x77);
		const bool ok = Libs::LibKernel::Memory::TryReadBacking(start, got.data(), got.size());
		oks.push_back(ok);
		if (!ok) {
			for (const auto byte: got) {
				if (byte != 0x77) {
					Fail(test, std::string("partial copy on failure (fast=") +
					               (fast ? "true" : "false") + ")");
					break;
				}
			}
		}
	});
	Check(test, oks.size() == 2 && oks[0] == oks[1],
	      "fast path and two-pass path disagreed on a read past the mapping end");

	std::printf("[host]    %-52s ok\n", test);
}

void InitSubsystems() {
	static Common::Subsystems subsystems;
	Common::VirtualMemory::Init();
	Common::InitializeThreads();
	subsystems.Initialize<Config::Lifecycle>();
	subsystems.Initialize<Log::Lifecycle>();
	subsystems.Initialize<Libs::LibKernel::Memory::Lifecycle>();
}

// ---------------------------------------------------------------- throughput

// What does the redundant traversal actually cost?
//
// With the flag OFF, TryTransferBacking validates the whole request with one walk of the mapping
// map and then transfers it with a second walk. With it ON, a request contained in a single
// mapping takes one lookup. Historical logs show that case is essentially universal: a run with
// the flag on recorded fast=29999376 (100.0%) against two-pass=624 (0.002% spanning).
//
// The compiled-ON diagnostic measured TryReadBacking at 7.90% exclusive self-time, 787 ns/draw,
// with the flag OFF - so every one of that run's 86,000,000 transfers paid the second walk.
//
// This measures the difference directly. Transfer sizes are swept because the traversal is a fixed
// cost per call while the copy scales with size: the smaller the transfer, the more the second
// walk matters. Sizes here are CHOSEN, not observed - no per-transfer size histogram exists.
void BenchmarkTransferSizes() {
	const char* test = "Throughput";
	auto        m    = MapTwoPages(test);
	if (!m.ok) {
		return;
	}
	auto* base = reinterpret_cast<uint8_t*>(m.vaddr);
	std::memset(base, 0xab, SceKernelPageSize * 2);

	std::printf("\nTryReadBacking throughput: one mapping, request contained in it.\n");
	std::printf("The flag-off arm walks the map twice; the flag-on arm once.\n\n");
	std::printf("%10s %14s %14s %10s %12s\n", "bytes", "two-pass ns", "fast ns", "saved ns",
	            "speedup");
	std::printf("%s\n", std::string(64, '-').c_str());

	for (const uint64_t size: {uint64_t {64}, uint64_t {256}, uint64_t {1024}, uint64_t {4096},
	                           uint64_t {16384}}) {
		std::vector<uint8_t> dst(static_cast<size_t>(size));
		double               timing[2] = {0.0, 0.0};
		bool                 ok        = true;

		for (int arm = 0; arm < 2; arm++) {
			Config::ConfigOptions options {};
			options.backing_fast_path = arm == 1;
			Config::Load(options);

			// Warm.
			for (int i = 0; i < 1000; i++) {
				ok = ok && Libs::LibKernel::Memory::TryReadBacking(m.vaddr, dst.data(), size);
			}
			constexpr size_t kIterations = 200000;
			const auto       start       = std::chrono::steady_clock::now();
			for (size_t i = 0; i < kIterations; i++) {
				ok = ok && Libs::LibKernel::Memory::TryReadBacking(m.vaddr, dst.data(), size);
			}
			timing[arm] =
			    std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
			        .count() /
			    static_cast<double>(kIterations);
		}
		if (!ok) {
			Fail(test, "TryReadBacking failed during timing");
			return;
		}
		std::printf("%10llu %14.1f %14.1f %10.1f %11.2fx\n",
		            static_cast<unsigned long long>(size), timing[0], timing[1],
		            timing[0] - timing[1], timing[0] / timing[1]);
	}

	std::printf("\nThe saving is per TRANSFER, not per draw or per frame. Converting it needs a\n"
	            "transfers-per-frame figure this benchmark does not measure, and the in-game\n"
	            "7.90%% self-time is a share of one profiled thread, not of frame time.\n");
}

// How much does the redundant traversal cost as the mapping map grows?
//
// FindContainingUnlocked is a std::map::upper_bound - O(log n) over a red-black tree, so its cost
// is pointer chasing that grows with the number of live mappings. The single-mapping benchmark
// above therefore measures the fast path's BEST case for the copy and its WORST case for the
// lookup: with one entry the tree walk is nearly free and the saving disappears into noise.
//
// This maps many regions first, so the second traversal costs what it would in a process with a
// realistic mapping count, and times a small transfer where the copy cannot mask the lookup.
void BenchmarkMappingCount() {
	const char* test = "MappingCount";

	std::printf("\nSame transfer (256 B), increasing live mappings. The flag-off arm walks the\n");
	std::printf("mapping map twice per transfer, the flag-on arm once.\n\n");
	std::printf("%10s %14s %14s %10s %12s\n", "mappings", "two-pass ns", "fast ns", "saved ns",
	            "speedup");
	std::printf("%s\n", std::string(64, '-').c_str());

	std::vector<uint64_t> held;
	constexpr uint64_t    kSize = 256;
	std::vector<uint8_t>  dst(static_cast<size_t>(kSize));

	for (const int target: {1, 64, 512}) {
		while (static_cast<int>(held.size()) < target) {
			auto m = MapTwoPages(test);
			if (!m.ok) {
				Fail(test, "could not map enough regions");
				return;
			}
			held.push_back(m.vaddr);
		}
		// Read from the FIRST mapping every time, so only the map size varies.
		const auto vaddr = held.front();
		std::memset(reinterpret_cast<uint8_t*>(vaddr), 0xcd, static_cast<size_t>(kSize));

		// Arms are INTERLEAVED and repeated, and the median of each is reported. A single
		// A-then-B pass gave physically impossible results here - the fast path measured 0.60x
		// while doing strictly less work - because the per-call difference is a few nanoseconds
		// and drift between two sequential timing windows is the same size.
		constexpr int    kRounds     = 9;
		constexpr size_t kIterations = 300000;
		std::vector<double> samples[2];
		bool                ok = true;
		for (int round = 0; round < kRounds; round++) {
			for (int arm = 0; arm < 2; arm++) {
				Config::ConfigOptions options {};
				options.backing_fast_path = arm == 1;
				Config::Load(options);
				for (int i = 0; i < 2000; i++) {
					ok = ok && Libs::LibKernel::Memory::TryReadBacking(vaddr, dst.data(), kSize);
				}
				const auto start = std::chrono::steady_clock::now();
				for (size_t i = 0; i < kIterations; i++) {
					ok = ok && Libs::LibKernel::Memory::TryReadBacking(vaddr, dst.data(), kSize);
				}
				samples[arm].push_back(
				    std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() -
				                                            start)
				        .count() /
				    static_cast<double>(kIterations));
			}
		}
		double timing[2];
		double spread[2];
		for (int arm = 0; arm < 2; arm++) {
			std::sort(samples[arm].begin(), samples[arm].end());
			timing[arm] = samples[arm][samples[arm].size() / 2];
			spread[arm] = samples[arm].back() - samples[arm].front();
		}
		if (!ok) {
			Fail(test, "TryReadBacking failed during timing");
			return;
		}
		std::printf("%10d %14.1f %14.1f %10.1f %11.2fx   (spread off %.1f / on %.1f ns)\n",
		            static_cast<int>(held.size()), timing[0], timing[1], timing[0] - timing[1],
		            timing[0] / timing[1], spread[0], spread[1]);
	}
}

} // namespace

int main(int /*argc*/, char** /*argv*/) {
	InitSubsystems();

	TestContainedRange();
	TestSpanningRange();
	TestUnmappedRange();
	TestInvalidRanges();
	TestNoPartialCopyPastMappingEnd();

	BenchmarkTransferSizes();
	BenchmarkMappingCount();

	if (g_failures != 0) {
		std::printf("\n%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("\nall backing transfer fast-path checks passed\n");
	return 0;
}
