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

} // namespace

int main(int /*argc*/, char** /*argv*/) {
	InitSubsystems();

	TestContainedRange();
	TestSpanningRange();
	TestUnmappedRange();
	TestInvalidRanges();
	TestNoPartialCopyPastMappingEnd();

	if (g_failures != 0) {
		std::printf("\n%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("\nall backing transfer fast-path checks passed\n");
	return 0;
}
