#include "graphics/host_gpu/renderer/gpuTimestamps.h"

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <unordered_map>

namespace Libs::Graphics {

namespace {

struct IdentityHash {
	size_t operator()(const GpuTimestampIdentity& id) const noexcept {
		uint64_t h = id.ps_program * 0x9E3779B97F4A7C15ull;
		h ^= id.vs_program + 0x632BE59BD9B4E019ull + (h << 6u) + (h >> 2u);
		h ^= ((static_cast<uint64_t>(id.ps_chksum) << 32u) | id.vs_chksum) + (h << 6u) + (h >> 2u);
		return static_cast<size_t>(h);
	}
};

struct Entry {
	uint32_t             first = 0; // TOP query; BOTTOM is first + 1
	GpuTimestampIdentity identity;
	bool                 ended = false;
};

// Every pair written into one command buffer. Read once, after that buffer's tick completed.
struct Batch {
	uint64_t           tick = 0;
	std::vector<Entry> entries;
};

using StatsMap = std::unordered_map<GpuTimestampIdentity, GpuTimestampStats, IdentityHash>;

} // namespace

void GpuTimestampStats::Add(double ns) {
	if (count == 0 || ns < min_ns) {
		min_ns = ns;
	}
	if (count == 0 || ns > max_ns) {
		max_ns = ns;
	}
	count++;
	total_ns += ns;
	uint32_t bucket = 0;
	if (ns >= 2.0) {
		bucket = static_cast<uint32_t>(std::floor(std::log2(ns)));
	}
	histogram[std::min(bucket, Buckets - 1)]++;
}

double GpuTimestampStats::MedianBucketNs() const {
	if (count == 0) {
		return 0.0;
	}
	const uint64_t target = (count + 1) / 2;
	uint64_t       seen   = 0;
	for (uint32_t i = 0; i < Buckets; i++) {
		seen += histogram[i];
		if (seen >= target) {
			return i == 0 ? 0.0 : std::ldexp(1.0, static_cast<int>(i));
		}
	}
	return max_ns;
}

struct GpuTimestamps::State {
	vk::Device    device      = nullptr;
	vk::QueryPool pool        = nullptr;
	uint64_t      mask        = 0;
	double        ns_per_tick = 0.0;
	uint32_t      report_ms   = 0;

	mutable std::mutex                    mutex;
	std::vector<uint32_t>                 free;
	std::shared_ptr<Batch>                open_batch;
	StatsMap                              total;
	StatsMap                              window;
	GpuTimestampCounters                  counters;
	GpuTimestampCounters                  window_start_counters;
	std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();

	~State() {
		if (pool != nullptr) {
			device.destroyQueryPool(pool, nullptr);
		}
	}

	void Complete(Batch& batch);
	void ReportLocked(std::chrono::steady_clock::time_point now);
};

void GpuTimestamps::State::Complete(Batch& batch) {
	struct Result {
		const Entry* entry = nullptr;
		double       ns    = 0.0;
	};
	std::vector<Result> results;
	results.reserve(batch.entries.size());
	uint64_t unended = 0, unavailable = 0, read_errors = 0;

	for (const auto& entry: batch.entries) {
		if (!entry.ended) {
			unended++;
			continue;
		}
		// Two queries, each followed by its availability word. No WAIT flag: the owning buffer's
		// tick is already known complete, and if a value is still missing it is counted, not
		// waited for.
		std::array<uint64_t, 4> data {};
		const auto              r = device.getQueryPoolResults(
            pool, entry.first, 2, sizeof(data), data.data(), 2 * sizeof(uint64_t),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
		if (r != vk::Result::eSuccess && r != vk::Result::eNotReady) {
			read_errors++;
			continue;
		}
		if (data[1] == 0 || data[3] == 0) {
			unavailable++;
			continue;
		}
		// Masked subtraction handles counters narrower than 64 bits wrapping between the pair.
		const uint64_t ticks = (data[2] - data[0]) & mask;
		results.push_back({&entry, static_cast<double>(ticks) * ns_per_tick});
	}

	// Host reset is legal here: the only command buffer that used these queries has completed.
	for (const auto& entry: batch.entries) {
		device.resetQueryPool(pool, entry.first, 2);
	}

	const auto      now = std::chrono::steady_clock::now();
	std::lock_guard lock(mutex);
	for (const auto& result: results) {
		total[result.entry->identity].Add(result.ns);
		window[result.entry->identity].Add(result.ns);
	}
	counters.completed += results.size();
	counters.unended += unended;
	counters.unavailable += unavailable;
	counters.read_errors += read_errors;
	for (const auto& entry: batch.entries) {
		free.push_back(entry.first);
	}
	if (open_batch.get() == &batch) {
		open_batch.reset();
	}
	if (report_ms != 0 &&
	    now - last_report >= std::chrono::milliseconds(static_cast<int64_t>(report_ms))) {
		ReportLocked(now);
	}
}

void GpuTimestamps::State::ReportLocked(std::chrono::steady_clock::time_point now) {
	const double seconds = std::chrono::duration<double>(now - last_report).count();
	last_report          = now;

	std::vector<std::pair<GpuTimestampIdentity, GpuTimestampStats>> rows(window.begin(),
	                                                                     window.end());
	std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
		return a.second.total_ns > b.second.total_ns;
	});
	const auto delta = [&](uint64_t now_value, uint64_t start) {
		return static_cast<unsigned long long>(now_value - start);
	};
	const auto& c = counters;
	const auto& s = window_start_counters;
	std::printf(
	    "GpuTimestamps report: window=%.1fs recorded=%llu completed=%llu unavailable=%llu "
	    "unended=%llu exhausted=%llu skipped_secondary=%llu read_errors=%llu\n"
	    "  interval = TOP_OF_PIPELINE..BOTTOM_OF_PIPELINE around ONE primary draw command; GPU time, "
	    "not CPU submission; brackets may overlap, so sums are attributed, not additive costs\n",
	    seconds, delta(c.recorded, s.recorded), delta(c.completed, s.completed),
	    delta(c.unavailable, s.unavailable), delta(c.unended, s.unended),
	    delta(c.exhausted, s.exhausted), delta(c.skipped_secondary, s.skipped_secondary),
	    delta(c.read_errors, s.read_errors));
	const size_t limit = std::min<size_t>(rows.size(), 16);
	for (size_t i = 0; i < limit; i++) {
		const auto& id = rows[i].first;
		const auto& st = rows[i].second;
		std::printf("  ps=0x%08" PRIx32 " vs=0x%08" PRIx32 " ps_prog=0x%016" PRIx64
		            " vs_prog=0x%016" PRIx64 " draws=%" PRIu64
		            " sum=%.3fms median~%.3fms(log2 bucket) max=%.3fms\n",
		            id.ps_chksum, id.vs_chksum, id.ps_program, id.vs_program, st.count,
		            st.total_ns / 1e6, st.MedianBucketNs() / 1e6, st.max_ns / 1e6);
	}
	std::fflush(stdout);
	window.clear();
	window_start_counters = counters;
}

GpuTimestamps::~GpuTimestamps() {
	Shutdown();
}

void GpuTimestamps::Initialize(GraphicContext& graphics, CommandScheduler& scheduler, bool enabled,
                               uint32_t pair_capacity, uint32_t report_interval_ms) {
	Shutdown();
	m_scheduler = &scheduler;
	if (!enabled) {
		m_status = "off";
		return;
	}
	const auto& limits = graphics.physical_device_properties.limits;
	m_valid_bits       = graphics.timestamp_valid_bits;
	m_ns_per_tick      = static_cast<double>(limits.timestampPeriod);
	if (pair_capacity == 0 || pair_capacity > (UINT32_MAX / 2)) {
		m_status = "off: invalid pair capacity";
	} else if (m_valid_bits == 0) {
		m_status = "unsupported: queue family timestampValidBits is 0";
	} else if (!(m_ns_per_tick > 0.0)) {
		m_status = "unsupported: timestampPeriod is not positive";
	} else if (!graphics.host_query_reset_enabled) {
		m_status = "unsupported: hostQueryReset feature not enabled";
	} else {
		auto state         = std::make_shared<State>();
		state->device      = graphics.device;
		state->ns_per_tick = m_ns_per_tick;
		state->report_ms   = report_interval_ms;
		state->mask = m_valid_bits >= 64 ? ~0ull : ((1ull << m_valid_bits) - 1ull);

		vk::QueryPoolCreateInfo info {};
		info.sType      = vk::StructureType::eQueryPoolCreateInfo;
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = pair_capacity * 2;
		if (graphics.device.createQueryPool(&info, nullptr, &state->pool) != vk::Result::eSuccess) {
			state->pool = nullptr;
			m_status    = "off: timestamp query pool creation failed";
		} else {
			// New queries are in an undefined state and must be reset before their first use.
			graphics.device.resetQueryPool(state->pool, 0, info.queryCount);
			state->free.reserve(pair_capacity);
			for (uint32_t i = pair_capacity; i > 0; i--) {
				state->free.push_back((i - 1) * 2);
			}
			m_state  = std::move(state);
			m_active = true;
			char text[160];
			std::snprintf(text, sizeof(text),
			              "on: pairs=%" PRIu32 " timestampValidBits=%" PRIu32
			              " timestampPeriod=%.4fns timestampComputeAndGraphics=%d",
			              pair_capacity, m_valid_bits, m_ns_per_tick,
			              limits.timestampComputeAndGraphics == VK_TRUE ? 1 : 0);
			m_status = text;
		}
	}
	std::printf("GpuTimestamps: %s\n", m_status.c_str());
	std::fflush(stdout);
}

void GpuTimestamps::Shutdown() {
	m_active = false;
	// Deferred reads hold their own reference; the pool dies with the last of them.
	m_state.reset();
}

void GpuTimestamps::SetSelection(std::vector<uint64_t> ps_chksums) {
	m_selection.clear();
	for (const auto value: ps_chksums) {
		m_selection.push_back(static_cast<uint32_t>(value));
	}
}

bool GpuTimestamps::Selects(uint32_t ps_chksum) const {
	return m_selection.empty() ||
	       std::find(m_selection.begin(), m_selection.end(), ps_chksum) != m_selection.end();
}

GpuTimestamps::Token GpuTimestamps::Begin(vk::CommandBuffer command,
                                          const GpuTimestampIdentity& identity) {
	Token token;
	if (!m_active || command == nullptr) {
		return token;
	}
	auto&                  s    = *m_state;
	const uint64_t         tick = m_scheduler->CurrentTick();
	std::shared_ptr<Batch> schedule;
	{
		std::lock_guard lock(s.mutex);
		if (s.free.empty()) {
			// Every pair is still owned by an unread buffer. Resetting one now would destroy a
			// pending result, so this draw is not timed; rendering is untouched.
			s.counters.exhausted++;
			return token;
		}
		token.first = s.free.back();
		token.tick  = tick;
		s.free.pop_back();
		if (s.open_batch == nullptr || s.open_batch->tick != tick) {
			s.open_batch       = std::make_shared<Batch>();
			s.open_batch->tick = tick;
			schedule           = s.open_batch;
			s.counters.batches++;
		}
		s.open_batch->entries.push_back({token.first, identity, false});
		s.counters.recorded++;
	}
	command.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, s.pool, token.first);
	if (schedule != nullptr) {
		// DeferOperation tags the callback with CurrentTick(), which is this buffer's tick, so it
		// runs only after the scheduler has observed this buffer complete.
		auto state = m_state;
		m_scheduler->DeferOperation([state, schedule] { state->Complete(*schedule); });
	}
	return token;
}

void GpuTimestamps::End(vk::CommandBuffer command, const Token& token) {
	if (!m_active || !token.Valid() || command == nullptr) {
		return;
	}
	if (m_scheduler->CurrentTick() != token.tick) {
		// The buffer that holds the TOP query was submitted in between. Writing BOTTOM elsewhere
		// would pair timestamps from different buffers, so leave the entry unended.
		return;
	}
	auto& s = *m_state;
	command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, s.pool, token.first + 1);
	std::lock_guard lock(s.mutex);
	if (s.open_batch == nullptr) {
		return;
	}
	auto& entries = s.open_batch->entries;
	for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
		if (it->first == token.first) {
			it->ended = true;
			break;
		}
	}
}

void GpuTimestamps::NoteSkippedSecondary() {
	if (!m_active) {
		return;
	}
	std::lock_guard lock(m_state->mutex);
	m_state->counters.skipped_secondary++;
}

GpuTimestampCounters GpuTimestamps::GetCounters() const {
	if (m_state == nullptr) {
		return {};
	}
	std::lock_guard lock(m_state->mutex);
	return m_state->counters;
}

uint32_t GpuTimestamps::FreePairs() const {
	if (m_state == nullptr) {
		return 0;
	}
	std::lock_guard lock(m_state->mutex);
	return static_cast<uint32_t>(m_state->free.size());
}

std::vector<std::pair<GpuTimestampIdentity, GpuTimestampStats>>
GpuTimestamps::SnapshotStats() const {
	if (m_state == nullptr) {
		return {};
	}
	std::lock_guard lock(m_state->mutex);
	return {m_state->total.begin(), m_state->total.end()};
}

} // namespace Libs::Graphics
