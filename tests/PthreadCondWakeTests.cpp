// Multi-waiter tests for the guest pthread condition variable (src/kernel/pthread.cpp).
//
// Host threads call the real LibKernel entry points after PthreadInitSelfForMainThread gives each of
// them a PthreadPrivate. Wake counts come from the test-only KYTY_PTHREAD_COND_WAKE_COUNT counters:
// "notified_not_ready" counts condition-variable returns that were notified (not a poll timeout)
// but found no wake for that waiter, i.e. wakeups that did no work.

#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/singleton.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <filesystem>
#include <memory>

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace Libs::LibKernel;
using Clock = std::chrono::steady_clock;

namespace {

int g_failures = 0;

void Check(bool condition, const char* test, const char* message) {
	if (!condition) {
		std::printf("[FAIL] %s: %s\n", test, message);
		std::fflush(stdout);
		g_failures++;
	}
}

// Waits for a predicate with a hard deadline so a lost wake fails the test instead of hanging.
bool WaitUntil(const std::function<bool()>& predicate, std::chrono::milliseconds limit) {
	const auto deadline = Clock::now() + limit;
	while (!predicate()) {
		if (Clock::now() >= deadline) {
			return false;
		}
		std::this_thread::sleep_for(std::chrono::microseconds(200));
	}
	return true;
}

std::thread GuestThread(std::function<void()> body) {
	return std::thread([body = std::move(body)] {
		PthreadInitSelfForMainThread();
		body();
	});
}

struct Shared {
	PthreadMutex mutex = nullptr;
	PthreadCond  cond  = nullptr;

	Shared() {
		Check(PthreadMutexInit(&mutex, nullptr, "") == 0, "setup", "mutex init");
		Check(PthreadCondInit(&cond, nullptr, "") == 0, "setup", "cond init");
	}
};

// Signal with many waiters: every signal must complete exactly one consumer; count wakeups that did
// no work.
void TestSignalWakesOneWaiter(bool assert_wakeups) {
	constexpr const char* name      = "signal-herd";
	constexpr int         waiters   = 8;
	constexpr int         signals   = 400;
	Shared                s;
	int                   tokens    = 0;
	int                   waiting   = 0;
	std::atomic<int>      consumed  = 0;
	bool                  stop      = false;

	std::vector<std::thread> threads;
	for (int i = 0; i < waiters; i++) {
		threads.push_back(GuestThread([&] {
			PthreadMutexLock(&s.mutex);
			while (true) {
				while (tokens == 0 && !stop) {
					waiting++;
					PthreadCondWait(&s.cond, &s.mutex);
					waiting--;
				}
				if (tokens == 0 && stop) {
					break;
				}
				tokens--;
				consumed++;
			}
			PthreadMutexUnlock(&s.mutex);
		}));
	}

	const auto all_waiting = [&] {
		PthreadMutexLock(&s.mutex);
		const bool ok = (waiting == waiters && tokens == 0);
		PthreadMutexUnlock(&s.mutex);
		return ok;
	};
	Check(WaitUntil(all_waiting, std::chrono::seconds(10)), name, "waiters did not start");
	(void)PthreadCondTakeWakeCounts();

	const auto begin = Clock::now();
	bool       ok    = true;
	for (int i = 0; i < signals && ok; i++) {
		ok = WaitUntil(all_waiting, std::chrono::seconds(10));
		PthreadMutexLock(&s.mutex);
		tokens++;
		PthreadCondSignal(&s.cond);
		PthreadMutexUnlock(&s.mutex);
		ok = ok && WaitUntil([&] { return consumed.load() == i + 1; }, std::chrono::seconds(10));
	}
	const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
	const auto counts  = PthreadCondTakeWakeCounts();

	PthreadMutexLock(&s.mutex);
	stop = true;
	PthreadCondBroadcast(&s.cond);
	PthreadMutexUnlock(&s.mutex);
	for (auto& t: threads) {
		t.join();
	}

	std::printf("[info] %s waiters=%d signals=%d consumed=%d wait_returns=%" PRIu64
	            " notified_not_ready=%" PRIu64 " elapsed_ms=%.1f\n",
	            name, waiters, signals, consumed.load(), counts.wait_returns,
	            counts.notified_not_ready, elapsed);
	Check(ok && consumed.load() == signals, name, "a signal was lost or did not complete a waiter");
	if (assert_wakeups) {
		Check(counts.notified_not_ready <= static_cast<uint64_t>(signals / 4), name,
		      "signals woke waiters that were not selected");
	}
}

// Signal-to must complete only the targeted thread.
void TestSignalToTargetsOneThread() {
	constexpr const char* name    = "signal-to";
	constexpr int         waiters = 4;
	constexpr int         rounds  = 40;
	Shared                s;
	std::vector<int>      granted(waiters, 0);
	std::vector<int>      completed(waiters, 0);
	std::vector<Pthread>  handles(waiters, nullptr);
	std::vector<int>      waiting(waiters, 0);
	bool                  stop = false;

	std::vector<std::thread> threads;
	for (int i = 0; i < waiters; i++) {
		threads.push_back(GuestThread([&, i] {
			PthreadMutexLock(&s.mutex);
			handles[i] = PthreadSelf();
			while (true) {
				while (granted[i] == 0 && !stop) {
					waiting[i] = 1;
					PthreadCondWait(&s.cond, &s.mutex);
					waiting[i] = 0;
				}
				if (granted[i] == 0 && stop) {
					break;
				}
				granted[i]--;
				completed[i]++;
			}
			PthreadMutexUnlock(&s.mutex);
		}));
	}
	const auto all_waiting = [&] {
		PthreadMutexLock(&s.mutex);
		bool ok = true;
		for (int i = 0; i < waiters; i++) {
			ok = ok && waiting[i] == 1 && handles[i] != nullptr;
		}
		PthreadMutexUnlock(&s.mutex);
		return ok;
	};
	Check(WaitUntil(all_waiting, std::chrono::seconds(10)), name, "waiters did not start");

	bool ok = true;
	for (int r = 0; r < rounds && ok; r++) {
		const int target = (r * 3) % waiters;
		std::vector<int> before;
		PthreadMutexLock(&s.mutex);
		before = completed;
		granted[target]++;
		PthreadCondSignalto(&s.cond, handles[target]);
		PthreadMutexUnlock(&s.mutex);
		ok = WaitUntil(
		    [&] {
			    PthreadMutexLock(&s.mutex);
			    const bool done = completed[target] == before[target] + 1;
			    PthreadMutexUnlock(&s.mutex);
			    return done;
		    },
		    std::chrono::seconds(10));
		ok = ok && WaitUntil(all_waiting, std::chrono::seconds(10));
		PthreadMutexLock(&s.mutex);
		for (int i = 0; i < waiters; i++) {
			if (i != target && completed[i] != before[i]) {
				ok = false;
			}
		}
		PthreadMutexUnlock(&s.mutex);
	}
	PthreadMutexLock(&s.mutex);
	stop = true;
	PthreadCondBroadcast(&s.cond);
	PthreadMutexUnlock(&s.mutex);
	for (auto& t: threads) {
		t.join();
	}
	std::printf("[info] %s waiters=%d rounds=%d ok=%d\n", name, waiters, rounds, ok ? 1 : 0);
	Check(ok, name, "signal-to completed the wrong thread or none");
}

// Broadcast must complete every waiter each round.
void TestBroadcastWakesAll() {
	constexpr const char* name    = "broadcast";
	constexpr int         waiters = 6;
	constexpr int         rounds  = 50;
	Shared                s;
	int                   generation = 0;
	std::vector<int>      seen(waiters, 0);
	std::vector<int>      waiting(waiters, 0);
	bool                  stop = false;

	std::vector<std::thread> threads;
	for (int i = 0; i < waiters; i++) {
		threads.push_back(GuestThread([&, i] {
			PthreadMutexLock(&s.mutex);
			while (!stop) {
				while (seen[i] == generation && !stop) {
					waiting[i] = 1;
					PthreadCondWait(&s.cond, &s.mutex);
					waiting[i] = 0;
				}
				seen[i] = generation;
			}
			PthreadMutexUnlock(&s.mutex);
		}));
	}
	const auto all_current = [&] {
		PthreadMutexLock(&s.mutex);
		bool ok = true;
		for (int i = 0; i < waiters; i++) {
			ok = ok && waiting[i] == 1 && seen[i] == generation;
		}
		PthreadMutexUnlock(&s.mutex);
		return ok;
	};
	Check(WaitUntil(all_current, std::chrono::seconds(10)), name, "waiters did not start");
	(void)PthreadCondTakeWakeCounts();
	bool ok = true;
	for (int r = 0; r < rounds && ok; r++) {
		PthreadMutexLock(&s.mutex);
		generation++;
		PthreadCondBroadcast(&s.cond);
		PthreadMutexUnlock(&s.mutex);
		ok = WaitUntil(all_current, std::chrono::seconds(10));
	}
	const auto counts = PthreadCondTakeWakeCounts();
	PthreadMutexLock(&s.mutex);
	stop = true;
	PthreadCondBroadcast(&s.cond);
	PthreadMutexUnlock(&s.mutex);
	for (auto& t: threads) {
		t.join();
	}
	std::printf("[info] %s waiters=%d rounds=%d ok=%d wait_returns=%" PRIu64
	            " notified_not_ready=%" PRIu64 "\n",
	            name, waiters, rounds, ok ? 1 : 0, counts.wait_returns, counts.notified_not_ready);
	Check(ok, name, "broadcast did not complete every waiter");
}

// Timed wait: times out without a wake, and completes when signalled before the deadline.
void TestTimedWait() {
	constexpr const char* name = "timed-wait";
	Shared                s;
	int                   timeout_result  = 0;
	double                timeout_elapsed = 0.0;
	int                   signal_result   = -1;
	double                signal_elapsed  = 0.0;
	std::atomic<bool>     entered         = false;

	auto waiter = GuestThread([&] {
		PthreadMutexLock(&s.mutex);
		auto begin      = Clock::now();
		timeout_result  = PthreadCondTimedwait(&s.cond, &s.mutex, 20000);
		timeout_elapsed = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
		entered         = true;
		begin           = Clock::now();
		signal_result   = PthreadCondTimedwait(&s.cond, &s.mutex, 2000000);
		signal_elapsed  = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
		PthreadMutexUnlock(&s.mutex);
	});
	std::thread signaller([&] {
		PthreadInitSelfForMainThread();
		(void)WaitUntil([&] { return entered.load(); }, std::chrono::seconds(10));
		std::this_thread::sleep_for(std::chrono::milliseconds(30));
		PthreadMutexLock(&s.mutex);
		PthreadCondSignal(&s.cond);
		PthreadMutexUnlock(&s.mutex);
	});
	waiter.join();
	signaller.join();
	std::printf("[info] %s timeout_result=0x%08x timeout_ms=%.1f signal_result=0x%08x signal_ms=%.1f\n",
	            name, static_cast<unsigned>(timeout_result), timeout_elapsed,
	            static_cast<unsigned>(signal_result), signal_elapsed);
	Check(timeout_result != 0 && timeout_elapsed >= 19.0, name, "timed wait did not time out");
	Check(signal_result == 0 && signal_elapsed < 1000.0, name,
	      "signal before the deadline did not complete the timed wait");
}

// A signal-interruption wake must not complete the wait, and must not wake other waiters.
void TestSignalInterruption(bool assert_wakeups) {
	constexpr const char* name          = "signal-interruption";
	constexpr int         waiters       = 4;
	constexpr int         interruptions = 100;
	Shared                s;
	std::vector<int>      granted(waiters, 0);
	std::vector<int>      completed(waiters, 0);
	std::vector<int>      waiting(waiters, 0);
	std::vector<Pthread>  handles(waiters, nullptr);
	bool                  stop = false;

	std::vector<std::thread> threads;
	for (int i = 0; i < waiters; i++) {
		threads.push_back(GuestThread([&, i] {
			PthreadMutexLock(&s.mutex);
			handles[i] = PthreadSelf();
			while (true) {
				while (granted[i] == 0 && !stop) {
					waiting[i] = 1;
					PthreadCondWait(&s.cond, &s.mutex);
					waiting[i] = 0;
				}
				if (granted[i] == 0 && stop) {
					break;
				}
				granted[i]--;
				completed[i]++;
			}
			PthreadMutexUnlock(&s.mutex);
		}));
	}
	const auto all_waiting = [&] {
		PthreadMutexLock(&s.mutex);
		bool ok = true;
		for (int i = 0; i < waiters; i++) {
			ok = ok && waiting[i] == 1 && handles[i] != nullptr;
		}
		PthreadMutexUnlock(&s.mutex);
		return ok;
	};
	Check(WaitUntil(all_waiting, std::chrono::seconds(10)), name, "waiters did not start");
	(void)PthreadCondTakeWakeCounts();
	for (int i = 0; i < interruptions; i++) {
		PthreadWakeForSignal(handles[0]);
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	const auto counts = PthreadCondTakeWakeCounts();
	bool       none_completed = true;
	PthreadMutexLock(&s.mutex);
	for (int i = 0; i < waiters; i++) {
		none_completed = none_completed && completed[i] == 0;
	}
	granted[0]++;
	PthreadCondSignalto(&s.cond, handles[0]);
	PthreadMutexUnlock(&s.mutex);
	const bool target_completed = WaitUntil(
	    [&] {
		    PthreadMutexLock(&s.mutex);
		    const bool done = completed[0] == 1;
		    PthreadMutexUnlock(&s.mutex);
		    return done;
	    },
	    std::chrono::seconds(10));
	PthreadMutexLock(&s.mutex);
	stop = true;
	PthreadCondBroadcast(&s.cond);
	PthreadMutexUnlock(&s.mutex);
	for (auto& t: threads) {
		t.join();
	}
	std::printf("[info] %s waiters=%d interruptions=%d wait_returns=%" PRIu64
	            " notified_not_ready=%" PRIu64 " none_completed=%d target_completed_after=%d\n",
	            name, waiters, interruptions, counts.wait_returns, counts.notified_not_ready,
	            none_completed ? 1 : 0, target_completed ? 1 : 0);
	Check(none_completed, name, "a signal-interruption wake completed a condition wait");
	Check(target_completed, name, "waiter did not complete after its real signal");
	if (assert_wakeups) {
		Check(counts.notified_not_ready <= static_cast<uint64_t>(interruptions + interruptions / 4),
		      name, "signal interruption woke waiters other than the target");
	}
}

// Producers and consumers: every token is consumed exactly once within a bound.
void TestProducerConsumerStress() {
	constexpr const char* name      = "producer-consumer";
	constexpr int         producers = 4;
	constexpr int         consumers = 4;
	constexpr int         per_prod  = 5000;
	Shared                s;
	int                   tokens   = 0;
	int                   produced = 0;
	std::atomic<int>      consumed = 0;
	bool                  stop     = false;

	std::vector<std::thread> threads;
	for (int c = 0; c < consumers; c++) {
		threads.push_back(GuestThread([&] {
			PthreadMutexLock(&s.mutex);
			while (true) {
				while (tokens == 0 && !stop) {
					PthreadCondWait(&s.cond, &s.mutex);
				}
				if (tokens == 0 && stop) {
					break;
				}
				tokens--;
				consumed++;
			}
			PthreadMutexUnlock(&s.mutex);
		}));
	}
	std::vector<std::thread> prods;
	const auto               begin = Clock::now();
	for (int p = 0; p < producers; p++) {
		prods.push_back(GuestThread([&] {
			for (int i = 0; i < per_prod; i++) {
				PthreadMutexLock(&s.mutex);
				tokens++;
				produced++;
				PthreadCondSignal(&s.cond);
				PthreadMutexUnlock(&s.mutex);
			}
		}));
	}
	for (auto& t: prods) {
		t.join();
	}
	const bool drained =
	    WaitUntil([&] { return consumed.load() == producers * per_prod; }, std::chrono::seconds(20));
	const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
	PthreadMutexLock(&s.mutex);
	stop = true;
	PthreadCondBroadcast(&s.cond);
	PthreadMutexUnlock(&s.mutex);
	for (auto& t: threads) {
		t.join();
	}
	std::printf("[info] %s produced=%d consumed=%d drained=%d elapsed_ms=%.1f\n", name, produced,
	            consumed.load(), drained ? 1 : 0, elapsed);
	Check(drained && produced == producers * per_prod && consumed.load() == produced, name,
	      "tokens were lost or not drained within the bound");
}


// Kernel and guest-memory setup needed by PthreadCreate's guest stacks (same sequence as
// VirtualMemoryAllocationTests).
void InitSubsystems() {
	static Common::Subsystems subsystems;
	Common::VirtualMemory::Init();
	Common::InitializeThreads();
	subsystems.Initialize<Config::Lifecycle>();

	Config::ConfigOptions options;
	options.printf_direction = Config::OutputDirection::Silent;
	Config::Load(options);

	subsystems.Initialize<Log::Lifecycle>();

	const auto param_json = std::filesystem::temp_directory_path() / "kyty_pthread_cond_wake_tests.json";
	constexpr char json[] = R"({"kernel":{"flexibleMemorySize":3221225472}})";
	Common::File   param_file;
	Check(param_file.Create(param_json), "setup", "failed to create temporary param.json");
	uint32_t bytes_written = 0;
	param_file.Write(json, sizeof(json) - 1, &bytes_written);
	param_file.Close();
	Loader::SystemContentLoadParamSfo(param_json);
	(void)Common::File::DeleteFile(param_json);
	Libs::LibKernel::Memory::SetFlexibleMemorySize(Loader::SystemContentGetFlexibleMemorySize());

	subsystems.Initialize<Libs::LibKernel::Memory::Lifecycle>();
	// Thread exit (CleanupThread) reaches the RuntimeLinker singleton, whose constructor must run
	// on the main thread; the emulator creates it at startup.
	(void)Common::Singleton<Loader::RuntimeLinker>::Instance();
}

struct LifecycleArg {
	Shared*           shared     = nullptr;
	KernelUseconds    timeout_us = 0;
	std::atomic<int>* started    = nullptr;
	std::atomic<bool> done {false};
	int               result     = -1;
	Pthread           self       = nullptr;
};

KYTY_SYSV_ABI void* LifecycleEntry(void* arg) {
	auto* a = static_cast<LifecycleArg*>(arg);
	PthreadMutexLock(&a->shared->mutex);
	a->self = PthreadSelf();
	a->started->fetch_add(1);
	a->result = PthreadCondTimedwait(&a->shared->cond, &a->shared->mutex, a->timeout_us);
	PthreadMutexUnlock(&a->shared->mutex);
	a->done = true;
	return nullptr;
}

// Real guest threads (PthreadCreate/PthreadJoin, pool reuse) waiting with timeouts while signals,
// signal-to, broadcasts and signal-interruption wakes race with timeout removal, thread exit and
// reuse. A stale waiter entry left behind would absorb the final signal and fail the last check.
void TestRealLifecycleChurn() {
	constexpr const char*      name          = "real-lifecycle";
	constexpr int              iterations    = 200;
	constexpr int              per_iteration = 4;
	constexpr KernelUseconds   timeouts[per_iteration] = {200, 1000, 3000, 8000};
	Shared                     s;
	std::vector<Pthread>       seen;
	std::vector<Pthread>       stale;
	int                        reused          = 0;
	int                        create_failures = 0;
	int                        join_failures   = 0;
	int                        start_timeouts  = 0;
	int                        ok_results      = 0;
	int                        timeout_results = 0;
	int                        bad_results     = 0;
	const auto                 begin           = Clock::now();

	for (int it = 0; it < iterations; it++) {
		std::atomic<int>                           started {0};
		std::vector<std::unique_ptr<LifecycleArg>> args;
		std::vector<Pthread>                       handles(per_iteration, nullptr);
		int                                        created = 0;
		for (int i = 0; i < per_iteration; i++) {
			auto a        = std::make_unique<LifecycleArg>();
			a->shared     = &s;
			a->timeout_us = timeouts[(it + i) % per_iteration];
			a->started    = &started;
			if (PthreadCreate(&handles[i], nullptr, LifecycleEntry, a.get(), "cond-lifecycle") != 0) {
				create_failures++;
				handles[i] = nullptr;
			} else {
				created++;
				if (std::find(seen.begin(), seen.end(), handles[i]) != seen.end()) {
					reused++;
				} else {
					seen.push_back(handles[i]);
				}
			}
			args.push_back(std::move(a));
		}
		if (!WaitUntil([&] { return started.load() == created; }, std::chrono::seconds(10))) {
			start_timeouts++;
		}
		for (auto* h: stale) {
			PthreadWakeForSignal(h);
		}
		PthreadMutexLock(&s.mutex);
		switch (it % 4) {
			case 0: PthreadCondSignal(&s.cond); break;
			case 1:
				if (handles[it % per_iteration] != nullptr) {
					PthreadCondSignalto(&s.cond, handles[it % per_iteration]);
				}
				break;
			case 2: PthreadCondBroadcast(&s.cond); break;
			default:
				for (auto* h: handles) {
					if (h != nullptr) {
						PthreadWakeForSignal(h);
					}
				}
				break;
		}
		PthreadMutexUnlock(&s.mutex);
		stale.clear();
		for (int i = 0; i < per_iteration; i++) {
			if (handles[i] == nullptr) {
				continue;
			}
			if (PthreadJoin(handles[i], nullptr) != 0) {
				join_failures++;
				continue;
			}
			stale.push_back(handles[i]);
			const auto& a = *args[i];
			if (!a.done.load()) {
				bad_results++;
			} else if (a.result == 0) {
				ok_results++;
			} else if (a.result == Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT) {
				timeout_results++;
			} else {
				bad_results++;
			}
		}
	}
	const auto churn_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();

	std::atomic<int> started {0};
	LifecycleArg     last;
	last.shared                 = &s;
	last.timeout_us             = 2000000;
	last.started                = &started;
	Pthread    last_handle      = nullptr;
	const bool last_created     = PthreadCreate(&last_handle, nullptr, LifecycleEntry, &last, "cond-final") == 0;
	bool       completed_early  = false;
	bool       final_ok         = false;
	double     final_ms         = 0.0;
	if (last_created) {
		(void)WaitUntil([&] { return started.load() == 1; }, std::chrono::seconds(10));
		PthreadMutexLock(&s.mutex);
		PthreadMutexUnlock(&s.mutex);
		for (auto* h: seen) {
			PthreadWakeForSignal(h);
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(30));
		completed_early    = last.done.load();
		const auto signal_begin = Clock::now();
		PthreadMutexLock(&s.mutex);
		PthreadCondSignal(&s.cond);
		PthreadMutexUnlock(&s.mutex);
		(void)PthreadJoin(last_handle, nullptr);
		final_ms = std::chrono::duration<double, std::milli>(Clock::now() - signal_begin).count();
		final_ok = last.done.load() && last.result == 0 && final_ms < 1000.0;
	}

	std::printf("[info] %s iterations=%d threads=%d distinct_handles=%zu reused=%d create_fail=%d "
	            "join_fail=%d start_timeouts=%d ok=%d timedout=%d bad=%d churn_ms=%.1f "
	            "final_created=%d final_completed_early=%d final_result=0x%08x final_ms=%.1f\n",
	            name, iterations, iterations * per_iteration, seen.size(), reused, create_failures,
	            join_failures, start_timeouts, ok_results, timeout_results, bad_results, churn_ms,
	            last_created ? 1 : 0, completed_early ? 1 : 0, static_cast<unsigned>(last.result),
	            final_ms);
	Check(create_failures == 0 && join_failures == 0 && start_timeouts == 0, name,
	      "guest thread create/join/start failed");
	Check(reused > 0, name, "pool reuse was not exercised");
	Check(bad_results == 0 && ok_results + timeout_results == iterations * per_iteration, name,
	      "a waiter returned an unexpected result or did not finish");
	Check(last_created && !completed_early, name,
	      "signal-interruption wakes on stale or reused handles completed a condition wait");
	Check(final_ok, name, "a signal after lifecycle churn did not complete a fresh waiter");
}

} // namespace

int main(int argc, char** argv) {
	const bool correctness_only = argc == 2 && std::strcmp(argv[1], "--correctness-only") == 0;
	const bool lifecycle_only   = argc == 2 && std::strcmp(argv[1], "--lifecycle-only") == 0;
	InitSubsystems();
	Libs::LibKernel::Initialize();
	PthreadInitSelfForMainThread();

	if (!lifecycle_only) {
		TestSignalWakesOneWaiter(!correctness_only);
		TestSignalToTargetsOneThread();
		TestBroadcastWakesAll();
		TestTimedWait();
		TestSignalInterruption(!correctness_only);
		TestProducerConsumerStress();
	}
	TestRealLifecycleChurn();

	if (g_failures != 0) {
		std::printf("pthread cond wake tests failed: %d\n", g_failures);
		std::fflush(stdout);
		std::_Exit(1);
	}
	std::printf("pthread cond wake tests passed\n");
	std::fflush(stdout);
	std::_Exit(0);
}
