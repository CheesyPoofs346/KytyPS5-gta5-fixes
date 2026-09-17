#ifndef INCLUDE_KYTY_GRAPHICS_GUESTGPU_BLOCKEDQUEUEWAIT_H_
#define INCLUDE_KYTY_GRAPHICS_GUESTGPU_BLOCKEDQUEUEWAIT_H_

// What the GPU thread does when every non-empty queue's front submission is blocked.
//
// Kept in a header, templated on a backend, for one reason: the alternative below has to be
// testable without a GPU, a guest, or a game. The real backend binds to GuestGpu; the offline
// harness binds to a fake producer whose readiness time it controls.
//
// Scope note: this is SCHEDULER-LOCAL on purpose. CondVar::WaitFor is used by other subsystems
// and is deliberately left alone.

#include <atomic>
#include <cstdint>

namespace Libs::Graphics {

// How a wait on m_work_available ended.
//
// BeforeTimeout is NOT evidence that a producer notified us about the blocking condition:
//   - condition variables may wake spuriously;
//   - m_work_available is also signalled by SendCommand and SubmitToQueue, which wake this wait
//     whether or not the new work relates to what the blocked queue is waiting on.
// So this counter bounds how much of the wait is spent sitting at the timeout. It does not
// identify who woke us, and it does not establish that the awaited value became ready.
enum class BlockedWaitWake { BeforeTimeout, TimedOut };

struct BlockedPollConfig {
	// Length of one bounded short pause. 0 disables the short phase entirely, reproducing the
	// existing behaviour exactly: every poll is the timed wait.
	uint32_t short_micros = 0;
	// How many consecutive polls within ONE blocking episode may use the short pause before
	// falling back to the timed wait for the rest of that episode. This is what bounds the
	// alternative: a queue that stays blocked degrades to the timed wait instead of pausing
	// forever at a short interval.
	uint32_t short_tries = 0;
	// Passed to the timed fallback. Note it is rounded UP to 1 ms by the Windows
	// implementation of CondVar::WaitFor; that rounding is the thing under investigation.
	uint32_t fallback_micros = 100;
};

// Per-episode state. Reset when a retry actually advances, so the short budget applies to each
// blocking episode rather than to the process.
struct BlockedPollState {
	uint32_t consecutive_polls = 0;

	void Reset() { consecutive_polls = 0; }
};

struct BlockedPollStats {
	std::atomic<uint64_t> timed_waits {0};
	std::atomic<uint64_t> wake_before_timeout {0};
	std::atomic<uint64_t> timed_out {0};
	std::atomic<uint64_t> short_pauses {0};
	// Filled in by the caller once the retried submission has actually been processed: did
	// re-running the blocked packet let the queue move, or did it block again on the same
	// comparison? This is the only counter that separates a useful retry from repeated
	// unsuccessful polling. It still says nothing about WHEN the value became ready.
	std::atomic<uint64_t> retry_advanced {0};
	std::atomic<uint64_t> retry_still_blocked {0};

	void Reset() {
		timed_waits.store(0);
		wake_before_timeout.store(0);
		timed_out.store(0);
		short_pauses.store(0);
		retry_advanced.store(0);
		retry_still_blocked.store(0);
	}
};

inline void NoteBlockedRetry(BlockedPollStats& stats, bool advanced) {
	(advanced ? stats.retry_advanced : stats.retry_still_blocked)
	    .fetch_add(1, std::memory_order_relaxed);
}

// Backend contract. Every call happens with the queue mutex HELD except where noted.
//
//   bool Runnable()      - is there work the loop could select right now (a non-blocked queue
//                          front, or a pending command)?
//   bool Stopping()      - shutdown requested.
//   void ClearBlocked()  - clear the sticky blocked flags so fronts are retried. This does not
//                          change the guest comparison: the retry re-runs the same packet and
//                          re-tests the same value against the same reference.
//   BlockedWaitWake TimedWait(uint32_t micros)
//                        - wait on the work condition variable, RELEASING the mutex for the
//                          duration and re-acquiring before returning.
//   void PauseUnlocked(uint32_t micros)
//                        - release the mutex, sleep for micros, re-acquire. Releasing matters:
//                          producers take the same mutex to enqueue.
//
// Performs exactly ONE wait and returns whether there is work to select afterwards. The caller
// loops; the bound lives in state.consecutive_polls.
template <class Backend>
bool AwaitBlockedQueues(Backend& backend, const BlockedPollConfig& config,
                        BlockedPollState& state, BlockedPollStats& stats) {
	const bool use_short =
	    config.short_micros != 0 && state.consecutive_polls < config.short_tries;

	if (use_short) {
		backend.PauseUnlocked(config.short_micros);
		stats.short_pauses.fetch_add(1, std::memory_order_relaxed);
	} else {
		// Timed fallback, always reachable. The write that unblocks a queue comes from a
		// producer this thread has no notification from, so there is no event to wait on
		// instead; removing this would turn the loop into an unbounded spin.
		stats.timed_waits.fetch_add(1, std::memory_order_relaxed);
		const auto wake = backend.TimedWait(config.fallback_micros);
		(wake == BlockedWaitWake::TimedOut ? stats.timed_out : stats.wake_before_timeout)
		    .fetch_add(1, std::memory_order_relaxed);
	}

	// Checked after either path, so shutdown is never delayed by more than one interval.
	if (backend.Stopping()) {
		return true;
	}
	state.consecutive_polls++;
	backend.ClearBlocked();
	return backend.Runnable();
}

} // namespace Libs::Graphics

#endif // INCLUDE_KYTY_GRAPHICS_GUESTGPU_BLOCKEDQUEUEWAIT_H_
