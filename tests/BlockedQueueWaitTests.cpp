// Offline harness for the blocked-queue wait.
//
// Drives the REAL AwaitBlockedQueues template against a fake producer whose readiness time the
// harness controls, using the REAL Common::Mutex / Common::CondVar so the Windows 1 ms rounding
// in CondVar::WaitFor is the actual behaviour under test rather than a model of it.
//
// EVERYTHING BELOW IS A HARNESS RESULT. The consumer here does no PM4 work, the producer becomes
// ready at a time the harness picks, and there is no guest. These numbers describe the wait
// primitive in isolation. They are not frame times and they do not project to FPS.

#include "common/threads.h"
#include "graphics/guest_gpu/blockedQueueWait.h"

// SDL.h redefines main() to SDL_main; this harness owns its own entry point.
#define SDL_MAIN_HANDLED
#include <SDL.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
#endif

namespace {

using namespace Libs::Graphics;
using Clock = std::chrono::steady_clock;

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

// CPU actually burned by the calling thread. The point of a short poll interval is to trade CPU
// for latency, so a latency number without this is only half the result.
//
// Cycles, not GetThreadTimes: thread times are accounted in scheduler ticks (~15.6 ms here), so
// they read a flat zero for episodes this short. The first version of this harness reported
// exactly that, and a column of zeroes is worse than no column.
uint64_t ThreadCpuCycles() {
#ifdef _WIN32
	ULONG64 cycles = 0;
	if (QueryThreadCycleTime(GetCurrentThread(), &cycles) == 0) {
		return 0;
	}
	return static_cast<uint64_t>(cycles);
#else
	return 0;
#endif
}

// The absolute latencies below are dominated by how coarse this process's timers are, and that is
// NOT the emulator's resolution: the emulator's own blocked-poll counters average ~618 us per
// poll, well under one scheduler tick, so something in that process holds a finer setting.
// Measure it here rather than assuming, so the rows can be read honestly.
struct HostGranularity {
	double condvar_us  = 0.0;
	double sleep100_us = 0.0;
	double sleep50_us  = 0.0;
};

HostGranularity MeasureHostGranularity() {
	HostGranularity g;
	Common::Mutex   mutex;
	Common::CondVar cv;
	constexpr int   kN = 5;

	const auto median_us = [](std::vector<int64_t> v) {
		std::sort(v.begin(), v.end());
		return static_cast<double>(v[v.size() / 2]) / 1000.0;
	};

	std::vector<int64_t> cv_ns;
	for (int i = 0; i < kN; i++) {
		mutex.Lock();
		const auto start = Clock::now();
		cv.WaitFor(&mutex, 100); // the exact call the scheduler makes
		cv_ns.push_back(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
		mutex.Unlock();
	}
	g.condvar_us = median_us(cv_ns);

	for (uint32_t us: {100u, 50u}) {
		std::vector<int64_t> ns;
		for (int i = 0; i < kN; i++) {
			const auto start = Clock::now();
			Common::Thread::SleepMicro(us);
			ns.push_back(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
		}
		(us == 100 ? g.sleep100_us : g.sleep50_us) = median_us(ns);
	}
	return g;
}

// The fake scheduler. Readiness stands in for "the guest wrote the value the blocked packet is
// comparing against"; ClearBlocked stands in for clearing the sticky blocked flags so the packet
// is retried. Both are read and written under the same mutex the real backend uses.
struct FakeScheduler {
	Common::Mutex   mutex;
	Common::CondVar cv;

	bool     ready         = false;
	bool     stopping      = false;
	uint64_t clear_calls   = 0;
	uint64_t runnable_reads = 0;
	// Whether the producer signals the condition variable when it becomes ready. The real
	// producer for these waits (a guest memory write) does not, which is the case that matters.
	bool notify_on_ready = false;

	Clock::time_point ready_at {};

	bool Runnable() {
		runnable_reads++;
		return ready;
	}
	bool Stopping() const { return stopping; }
	void ClearBlocked() { clear_calls++; }

	BlockedWaitWake TimedWait(uint32_t micros) {
		return cv.WaitFor(&mutex, micros) ? BlockedWaitWake::BeforeTimeout
		                                  : BlockedWaitWake::TimedOut;
	}
	void PauseUnlocked(uint32_t micros) {
		mutex.Unlock();
		Common::Thread::SleepMicro(micros);
		mutex.Lock();
	}
};

struct EpisodeResult {
	int64_t  latency_ns    = 0; // readiness -> consumer resumed
	uint64_t consumer_cpu_cycles = 0;
	uint64_t timed_waits   = 0;
	uint64_t short_pauses  = 0;
	uint64_t timed_out     = 0;
	uint64_t woke_early    = 0;
	uint64_t polls         = 0;
	bool     resumed       = false;
};

// One blocking episode: the consumer waits until runnable, the producer becomes ready after
// ready_after_us. Returns how long after readiness the consumer resumed.
EpisodeResult RunEpisode(const BlockedPollConfig& config, uint32_t ready_after_us,
                         bool notify_on_ready, bool ready_before_wait = false) {
	FakeScheduler    fake;
	BlockedPollStats stats;
	BlockedPollState state;
	EpisodeResult    result;
	fake.notify_on_ready = notify_on_ready;

	std::atomic<bool> consumer_started {false};
	Clock::time_point resumed_at {};
	uint64_t          cpu_start = 0;
	uint64_t          cpu_end   = 0;

	// Completion-before-wait: the dependency is already satisfied before the consumer looks.
	// A wait that latches state incorrectly would hang here.
	if (ready_before_wait) {
		fake.mutex.Lock();
		fake.ready    = true;
		fake.ready_at = Clock::now();
		fake.mutex.Unlock();
	}

	std::thread consumer([&] {
		cpu_start = ThreadCpuCycles();
		fake.mutex.Lock();
		consumer_started.store(true, std::memory_order_release);
		while (!fake.Runnable() && !fake.Stopping()) {
			AwaitBlockedQueues(fake, config, state, stats);
			result.polls++;
		}
		resumed_at = Clock::now();
		fake.mutex.Unlock();
		cpu_end = ThreadCpuCycles();
	});

	if (!ready_before_wait) {
		while (!consumer_started.load(std::memory_order_acquire)) {
			std::this_thread::yield();
		}
		Common::Thread::SleepMicro(ready_after_us);
		fake.mutex.Lock();
		fake.ready    = true;
		fake.ready_at = Clock::now();
		if (notify_on_ready) {
			fake.cv.Signal();
		}
		fake.mutex.Unlock();
	}

	consumer.join();

	result.resumed = fake.ready;
	result.latency_ns =
	    std::chrono::duration_cast<std::chrono::nanoseconds>(resumed_at - fake.ready_at).count();
	result.consumer_cpu_cycles = cpu_end - cpu_start;
	result.timed_waits     = stats.timed_waits.load();
	result.short_pauses    = stats.short_pauses.load();
	result.timed_out       = stats.timed_out.load();
	result.woke_early      = stats.wake_before_timeout.load();

	// Guest comparison semantics: the predicate is re-tested once per wait, after the blocked
	// flags are cleared. Never fewer clears than waits, or a retry would test a stale flag.
	if (fake.clear_calls < result.polls) {
		std::printf("[FAIL]    clear/poll invariant                          clears=%llu polls=%llu\n",
		            static_cast<unsigned long long>(fake.clear_calls),
		            static_cast<unsigned long long>(result.polls));
		g_failures++;
	}
	return result;
}

// Rounds needed before a distribution means anything. Kept here so the header text and the loop
// cannot drift apart.
constexpr int kRounds = 31;

struct Spread {
	int64_t p10 = 0;
	int64_t med = 0;
	int64_t p90 = 0;
};

Spread Distribution(std::vector<int64_t> v) {
	if (v.empty()) {
		return {};
	}
	std::sort(v.begin(), v.end());
	const auto at = [&v](double q) {
		return v[static_cast<size_t>(q * static_cast<double>(v.size() - 1))];
	};
	return {at(0.10), at(0.50), at(0.90)};
}

// ---------------------------------------------------------------------------------------------

void TestDefaultIsUnchanged() {
	const char* name = "default config = one timed wait per poll";
	BlockedPollConfig config {}; // short_micros 0 => short phase disabled
	auto              r = RunEpisode(config, 3000, false);
	Check(name, r.resumed, "consumer never resumed");
	Check(name, r.short_pauses == 0,
	      "short pauses fired with the feature off: " + std::to_string(r.short_pauses));
	Check(name, r.timed_waits == r.polls,
	      "not every poll was a timed wait: waits=" + std::to_string(r.timed_waits) +
	          " polls=" + std::to_string(r.polls));
	if (r.resumed && r.short_pauses == 0 && r.timed_waits == r.polls) {
		Pass(name, std::to_string(r.polls) + " polls, all timed waits");
	}
}

void TestCompletionBeforeWait() {
	const char* name = "ready before the wait starts";
	for (uint32_t short_us: {0u, 100u}) {
		BlockedPollConfig config {.short_micros = short_us, .short_tries = 8};
		auto              r = RunEpisode(config, 0, false, /*ready_before_wait=*/true);
		Check(name, r.resumed, "consumer did not resume");
		Check(name, r.polls == 0,
		      "waited despite the dependency already being satisfied: polls=" +
		          std::to_string(r.polls) + " (short_us=" + std::to_string(short_us) + ")");
	}
	if (g_failures == 0) {
		Pass(name, "returns without waiting, both configs");
	}
}

void TestShutdownResponsiveness() {
	const char* name = "shutdown wakes a blocked consumer";
	FakeScheduler     fake;
	BlockedPollStats  stats;
	BlockedPollState  state;
	BlockedPollConfig config {.short_micros = 100, .short_tries = 8};
	std::atomic<bool> started {false};
	Clock::time_point stop_at {};
	Clock::time_point exit_at {};

	std::thread consumer([&] {
		fake.mutex.Lock();
		started.store(true, std::memory_order_release);
		while (!fake.Runnable() && !fake.Stopping()) {
			AwaitBlockedQueues(fake, config, state, stats);
		}
		exit_at = Clock::now();
		fake.mutex.Unlock();
	});
	while (!started.load(std::memory_order_acquire)) {
		std::this_thread::yield();
	}
	Common::Thread::SleepMicro(2000);
	fake.mutex.Lock();
	fake.stopping = true;
	stop_at       = Clock::now();
	fake.cv.SignalAll();
	fake.mutex.Unlock();
	consumer.join();

	const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(exit_at - stop_at).count();
	// One timed wait is the worst case, and CondVar rounds that to 1 ms. Anything past a few
	// milliseconds means shutdown is being delayed by the polling change.
	Check(name, ns < 5'000'000, "shutdown took " + std::to_string(ns / 1000) + " us");
	if (ns < 5'000'000) {
		Pass(name, "exited " + std::to_string(ns / 1000) + " us after stopping was set");
	}
}

void TestNoMissedReadiness() {
	const char* name = "no episode fails to resume";
	int         episodes = 0;
	for (uint32_t ready_us: {50u, 250u, 700u, 1500u}) {
		for (uint32_t short_us: {0u, 50u, 200u}) {
			for (bool notify: {false, true}) {
				BlockedPollConfig config {.short_micros = short_us, .short_tries = 8};
				auto              r = RunEpisode(config, ready_us, notify);
				episodes++;
				if (!r.resumed) {
					Check(name, false,
					      "no resume at ready_us=" + std::to_string(ready_us) +
					          " short_us=" + std::to_string(short_us));
				}
			}
		}
	}
	Pass(name, std::to_string(episodes) + " episodes across readiness/interval/notify combinations");
}

// Is the total short-pause spinning actually bounded across a blocking episode?
//
// ThreadRun resets the budget whenever a retried submission ADVANCES. But the poll only runs
// when every front is blocked; after it, all blocked flags are cleared and any queue may be
// selected. So a queue unrelated to the stuck dependency can advance, reset the budget, and let
// the short pauses start over - while the original dependency stays blocked. This models that
// driver loop and counts the total.
void TestBudgetReplenishment() {
	const char* name = "budget survives unrelated progress";

	FakeScheduler     fake;
	BlockedPollStats  stats;
	BlockedPollState  state;
	BlockedPollConfig config {.short_micros = 50, .short_tries = 8};

	// The stuck dependency never becomes ready during this test. "Unrelated progress" happens
	// every third poll, which is what ThreadRun would see when another queue's front advances.
	constexpr int kPolls = 60;
	fake.mutex.Lock();
	for (int i = 0; i < kPolls; i++) {
		AwaitBlockedQueues(fake, config, state, stats);
		const bool unrelated_queue_advanced = (i % 3) == 2;
		if (unrelated_queue_advanced) {
			state.Reset(); // exactly what ThreadRun does on NoteBlockedRetry(advanced)
		}
	}
	fake.mutex.Unlock();

	const auto pauses = stats.short_pauses.load();
	// If the budget were bounded per episode, the stuck dependency could never draw more than
	// short_tries short pauses no matter how long it stays blocked.
	const bool bounded = pauses <= config.short_tries;
	if (!bounded) {
		std::printf("[note]    %-46s %llu short pauses over %d polls with the budget capped at "
		            "%u - unrelated progress replenishes it\n",
		            name, static_cast<unsigned long long>(pauses), kPolls, config.short_tries);
	}
	// This is a characterisation, not a pass/fail: the behaviour is real and is why the total is
	// NOT described as bounded. Failing here would only mean the model stopped reproducing it.
	Check(name, !bounded,
	      "expected unrelated progress to replenish the budget, but it did not; the model no "
	      "longer reproduces the case this test exists to document");
	if (!bounded) {
		Pass(name, "REPRODUCED: total short pauses are NOT bounded per episode");
	}
}

// ---------------------------------------------------------------------------------------------
// Harness measurement, NOT a frame-time projection.

void MeasureLatencyAndCpu() {
	const auto g = MeasureHostGranularity();
	std::printf("  host granularity in THIS process: CondVar::WaitFor(100us) = %.0f us, "
	            "SleepMicro(100) = %.0f us, SleepMicro(50) = %.0f us\n",
	            g.condvar_us, g.sleep100_us, g.sleep50_us);
	// The emulator's own 618 us average is NOT a granularity reading: it mixes waits that hit
	// the deadline with waits that returned early, so it cannot be matched against a single
	// figure here. Matching resolution is established by pass 3 (real SDL init), not by this
	// average.
	std::printf("  Ranges are p10-p90 over %d rounds, not single values.\n\n", kRounds);
	std::printf("  %-9s %-22s %-22s %-24s %8s %8s\n", "ready in", "config",
	            "latency med (p10-p90)", "cpu cycles med (p10-p90)", "waits", "pauses");

	struct Arm {
		const char*       label;
		BlockedPollConfig config;
	};
	const Arm arms[] = {
	    {"timed only (current)", BlockedPollConfig {}},
	    {"short 100us x8", BlockedPollConfig {.short_micros = 100, .short_tries = 8}},
	    {"short 50us x16", BlockedPollConfig {.short_micros = 50, .short_tries = 16}},
	};

	for (uint32_t ready_us: {100u, 400u, 900u, 2500u}) {
		for (const auto& arm: arms) {
			std::vector<int64_t> latency;
			std::vector<int64_t> cpu;
			uint64_t             waits = 0;
			uint64_t             pauses = 0;
			for (int i = 0; i < kRounds; i++) {
				// Producer does NOT notify: this is the case the real waits are in, where the
				// unblocking write generates no signal.
				auto r = RunEpisode(arm.config, ready_us, /*notify_on_ready=*/false);
				latency.push_back(r.latency_ns);
				cpu.push_back(static_cast<int64_t>(r.consumer_cpu_cycles));
				waits += r.timed_waits;
				pauses += r.short_pauses;
			}
			const auto l = Distribution(latency);
			const auto c = Distribution(cpu);
			char       lat[64];
			char       cyc[64];
			std::snprintf(lat, sizeof(lat), "%.0f (%.0f-%.0f)", static_cast<double>(l.med) / 1000.0,
			              static_cast<double>(l.p10) / 1000.0, static_cast<double>(l.p90) / 1000.0);
			std::snprintf(cyc, sizeof(cyc), "%lld (%lld-%lld)", static_cast<long long>(c.med),
			              static_cast<long long>(c.p10), static_cast<long long>(c.p90));
			std::printf("  %-9s %-22s %-22s %-24s %8.1f %8.1f\n",
			            (std::to_string(ready_us) + "us").c_str(), arm.label, lat, cyc,
			            static_cast<double>(waits) / kRounds,
			            static_cast<double>(pauses) / kRounds);
		}
	}
	std::printf("\n  latency  = readiness -> consumer resumed, wall clock (steady_clock), us.\n"
	            "  cpu      = QueryThreadCycleTime delta for the CONSUMER THREAD ONLY over one\n"
	            "             episode. That API accounts cycles while this thread is actually\n"
	            "             scheduled on a core; it is NOT elapsed TSC and does NOT tick while\n"
	            "             the thread is blocked or descheduled, which is exactly why a\n"
	            "             sleeping arm and a spinning arm separate here. Cycles are also not\n"
	            "             convertible to time at a fixed rate under frequency scaling, so\n"
	            "             compare arms to each other, never to a wall-clock budget.\n");
}

} // namespace

int main() {
	Common::InitializeThreads();
	std::printf("Blocked-queue wait:\n\n");
	TestDefaultIsUnchanged();
	TestCompletionBeforeWait();
	TestShutdownResponsiveness();
	TestNoMissedReadiness();
	TestBudgetReplenishment();
	if (g_failures != 0) {
		std::printf("\n%d blocked-queue check(s) FAILED\n", g_failures);
		return 1;
	}
	std::printf("\nall blocked-queue checks passed\n");
	std::printf("\nHARNESS RESULTS - wait primitive in isolation, no PM4 work, no guest.\n");
	std::printf("Not frame times; do not project FPS from these.\n");

	// Three passes. The first version of this harness measured only pass 1, where every timed
	// wait landed on a 15.6 ms scheduler tick - which is not what the emulator sees. Pass 2
	// raises the resolution by hand. Pass 3 is the one that settles it: it performs the SAME SDL
	// initialisation the emulator performs, so whatever resolution the emulator actually ends up
	// with is what gets measured. Raising the resolution by hand proves nothing about the
	// emulator; running its own init does.
	std::printf("\n--- pass 1: default process timer resolution ---\n");
	MeasureLatencyAndCpu();
#ifdef _WIN32
	if (timeBeginPeriod(1) == TIMERR_NOERROR) {
		std::printf("\n--- pass 2: timer resolution raised to 1 ms by hand ---\n");
		MeasureLatencyAndCpu();
		timeEndPeriod(1);
	} else {
		std::printf("\n(could not raise timer resolution by hand)\n");
	}
#endif

	// src/graphics/presentation/window/window.cpp calls SDL_InitSubSystem(SDL_INIT_VIDEO |
	// SDL_INIT_GAMECONTROLLER). SDL_VideoInit calls SDL_TicksInit, which registers a callback on
	// SDL_HINT_TIMER_RESOLUTION; with the hint unset that callback runs immediately with
	// uPeriod = 1 and calls timeBeginPeriod(1). This pass executes that path for real instead of
	// trusting the reading.
	std::printf("\n--- pass 3: after the emulator's own SDL_InitSubSystem(SDL_INIT_VIDEO) ---\n");
	if (SDL_InitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) < 0) {
		std::printf("  SDL_InitSubSystem failed (%s); emulator resolution NOT verified here.\n",
		            SDL_GetError());
	} else {
		MeasureLatencyAndCpu();
		SDL_QuitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER);
	}
	return 0;
}
