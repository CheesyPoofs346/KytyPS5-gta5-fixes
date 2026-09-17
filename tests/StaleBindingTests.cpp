// Focused regression for the within-draw stale-binding case.
//
// Real flow being modelled (descriptors.cpp):
//   1. FindBuffers   - resolves a BufferId per buffer resource. May create/join.
//   2. RebindBuffers - for each resource, ObtainBuffer(address, size, cached_id) re-validates the
//                      id, re-resolves it if the buffer was deleted or is no longer in bounds,
//                      synchronizes dirty guest bytes into it, and captures a HANDLE + OFFSET
//                      (and a device address) into the descriptor.
//
// The gap: step 2 is a loop. A re-resolve for resource i can reach CreateBuffer and JOIN the
// buffer that resource j<i already captured a handle for. That handle stays VALID - retirement is
// deferred past GPU completion - but it stops being CURRENT: later synchronizes land in the
// replacement, and writes through the stale alias land in a buffer nothing else reads.
//
// Retaining old storage is not coherence. This models that distinction directly.

#include "graphics/host_gpu/renderer/cache/bufferGrowth.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
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

constexpr uint64_t kPage = 4096;

// ---------------------------------------------------------------------------------------------
// A model of the parts of BufferCache this case depends on: ranges, join-on-overlap, deferred
// retirement, per-address dirty tracking, and a device address per allocation.

struct ModelBuffer {
	uint64_t             begin   = 0;
	uint64_t             size    = 0;
	uint64_t             address = 0;   // stands in for the buffer device address
	bool                 deleted = false;
	std::vector<uint8_t> bytes;
};

class ModelCache {
public:
	// Guest memory and its dirty bits, keyed by page - exactly the property that makes the real
	// tracker independent of which host buffer covers a range.
	std::map<uint64_t, uint8_t> guest;        // page -> byte value written by the guest
	std::map<uint64_t, bool>    cpu_dirty;    // page -> needs upload

	std::map<uint64_t, uint32_t> live;        // begin -> id
	std::vector<ModelBuffer>     buffers;     // indexed by id; retired entries stay alive
	uint64_t                     generation = 0;

	void GuestWrite(uint64_t page, uint8_t value) {
		guest[page]     = value;
		cpu_dirty[page] = true;
	}

	uint32_t Create(uint64_t vaddr, uint64_t size) {
		BufferGrowthConfig off {};   // growth disabled: this case predates the prototype
		auto plan = ResolveOverlapsWithLeap(
		    live, [this](auto it) {
			    const auto& b = buffers[it->second];
			    return BufferGrowthRange {b.begin, b.begin + b.size, 0};
		    },
		    vaddr, vaddr + size, off);

		ModelBuffer replacement;
		replacement.begin   = plan.begin;
		replacement.size    = plan.end - plan.begin;
		replacement.address = 0x8000'0000 + buffers.size() * 0x1'0000;
		replacement.bytes.assign(static_cast<size_t>(replacement.size), 0);

		for (auto it = plan.first; it != plan.last;) {
			const auto current = it++;
			auto&      old     = buffers[current->second];
			// Copy forward, then retire. Retirement marks it deleted but the storage stays.
			for (uint64_t i = 0; i < old.size; i++) {
				replacement.bytes[static_cast<size_t>(old.begin - plan.begin + i)] =
				    old.bytes[static_cast<size_t>(i)];
			}
			old.deleted = true;
			live.erase(current);
		}
		const auto id = static_cast<uint32_t>(buffers.size());
		buffers.push_back(std::move(replacement));
		live[plan.begin] = id;
		generation++;
		return id;
	}

	uint32_t Find(uint64_t vaddr, uint64_t size) {
		auto it = live.upper_bound(vaddr);
		if (it != live.begin()) {
			--it;
			const auto& b = buffers[it->second];
			if (b.begin <= vaddr && vaddr + size <= b.begin + b.size) {
				return it->second;
			}
		}
		return Create(vaddr, size);
	}

	// Uploads every dirty page of [vaddr, vaddr+size) into the buffer, then clears the bits -
	// the real SynchronizeBuffer, reduced to what matters here.
	void Synchronize(uint32_t id, uint64_t vaddr, uint64_t size) {
		auto& b = buffers[id];
		for (uint64_t page = vaddr; page < vaddr + size; page += kPage) {
			if (!cpu_dirty[page]) {
				continue;
			}
			const auto offset = page - b.begin;
			for (uint64_t i = 0; i < kPage; i++) {
				b.bytes[static_cast<size_t>(offset + i)] = guest[page];
			}
			cpu_dirty[page] = false;
		}
	}

	// The real ObtainBuffer: re-validate the cached id, re-resolve on a miss, synchronize.
	uint32_t Obtain(uint64_t vaddr, uint64_t size, uint32_t cached_id) {
		auto id = cached_id;
		if (id >= buffers.size() || buffers[id].deleted ||
		    !(buffers[id].begin <= vaddr && vaddr + size <= buffers[id].begin + buffers[id].size)) {
			id = Find(vaddr, size);
		}
		Synchronize(id, vaddr, size);
		return id;
	}
};

// What a descriptor holds once materialised: a handle, an offset and a device address.
struct Binding {
	uint64_t vaddr      = 0;
	uint64_t size       = 0;
	uint32_t cached_id  = 0;
	uint32_t handle     = 0;   // captured buffer id == the VkBuffer handle in the real code
	uint64_t offset     = 0;
	uint64_t address    = 0;   // captured device address
	bool     writable   = false;
};

// Step 2 of the real flow, with the fix switchable so the test can watch it fail without it.
void MaterialiseBindings(ModelCache& cache, std::vector<Binding>& bindings, bool refresh_after) {
	const auto materialise = [&](Binding& b) {
		b.handle    = cache.Obtain(b.vaddr, b.size, b.cached_id);
		b.cached_id = b.handle;
		b.offset    = b.vaddr - cache.buffers[b.handle].begin;
		b.address   = cache.buffers[b.handle].address;
	};

	const auto before = cache.generation;
	for (auto& b: bindings) {
		materialise(b);
	}
	if (!refresh_after || cache.generation == before) {
		return;
	}
	// Bounded: only runs when a replacement happened during the loop above, and only re-does the
	// entries whose captured buffer was actually retired. Repeats only while that keeps happening,
	// which it cannot do indefinitely - each pass either retires something or stops.
	for (int pass = 0; pass < 4; pass++) {
		const auto pass_start = cache.generation;
		for (auto& b: bindings) {
			if (cache.buffers[b.handle].deleted) {
				materialise(b);
			}
		}
		if (cache.generation == pass_start) {
			return;
		}
	}
}

// "Recording" the draw: read through the captured handle and offset.
uint8_t ReadThroughBinding(const ModelCache& cache, const Binding& b) {
	return cache.buffers[b.handle].bytes[static_cast<size_t>(b.offset)];
}

// Builds the exact sequence: binding A resolved, binding B re-resolves and joins A's buffer, and
// the overlapping guest bytes are updated so the synchronize lands in the replacement.
std::vector<Binding> BuildCase(ModelCache& cache) {
	// A covers one page; B covers A's page plus the next, so resolving B joins A.
	cache.GuestWrite(0x10000, 0x11);
	cache.GuestWrite(0x11000, 0x22);

	std::vector<Binding> bindings;
	Binding a;
	a.vaddr    = 0x10000;
	a.size     = kPage;
	a.writable = true;
	Binding b;
	b.vaddr = 0x10000;
	b.size  = 2 * kPage;

	// Step 1: FindBuffers resolves ids. A resolves first and gets its own buffer.
	a.cached_id = cache.Find(a.vaddr, a.size);
	// B is left with a STALE id on purpose - the real case is a cached id that misses in step 2,
	// which is what forces the re-resolve inside the materialisation loop.
	b.cached_id = 0xFFFF'FFFF;

	bindings.push_back(a);
	bindings.push_back(b);
	return bindings;
}

// ---------------------------------------------------------------------------------------------

void TestCaseIsReachable() {
	const char* name = "stale binding reproduces without the fix";
	ModelCache  cache;
	auto        bindings = BuildCase(cache);

	MaterialiseBindings(cache, bindings, /*refresh_after=*/false);

	// The guest then updates the shared page and B's synchronize would have taken it - but the
	// update happens before recording, so re-read what each binding actually sees.
	const bool a_retired = cache.buffers[bindings[0].handle].deleted;
	Check(name, a_retired, "A's buffer was not retired; the case did not reproduce");
	// A's handle is still usable - retirement is deferred - which is exactly the trap.
	const bool a_valid = bindings[0].handle < cache.buffers.size();
	Check(name, a_valid, "A's handle became unusable; that is a different bug");
	// A and B now point at different allocations for the same guest address.
	const bool split = bindings[0].handle != bindings[1].handle;
	Check(name, split, "A and B ended up on the same buffer; no aliasing to test");
	if (a_retired && a_valid && split) {
		Pass(name, "A retained a valid handle to a RETIRED buffer while B moved to the replacement");
	}
}

void TestStaleBindingReadsStaleBytes() {
	const char* name = "stale alias misses a later update";
	ModelCache  cache;
	auto        bindings = BuildCase(cache);
	MaterialiseBindings(cache, bindings, /*refresh_after=*/false);

	// Guest updates the shared page after A was captured; a synchronize through B takes it.
	cache.GuestWrite(0x10000, 0x99);
	cache.Synchronize(bindings[1].handle, bindings[1].vaddr, bindings[1].size);

	const auto through_a = ReadThroughBinding(cache, bindings[0]);
	const auto through_b = ReadThroughBinding(cache, bindings[1]);
	Check(name, through_b == 0x99, "B did not see the update; the model is wrong");
	// This is the defect. It is asserted as PRESENT so the fix below has something to prove.
	Check(name, through_a != 0x99,
	      "A already saw the update without the fix; the case may no longer be reachable");
	if (through_b == 0x99 && through_a != 0x99) {
		Pass(name, "A read 0x" + std::to_string(through_a) + " where the current value is 0x99");
	}
}

void TestWritableAliasIsLost() {
	const char* name = "write through a stale alias is lost";
	ModelCache  cache;
	auto        bindings = BuildCase(cache);
	MaterialiseBindings(cache, bindings, /*refresh_after=*/false);

	// A is writable: the shader writes through the captured handle.
	cache.buffers[bindings[0].handle].bytes[static_cast<size_t>(bindings[0].offset)] = 0x77;
	const auto through_b = ReadThroughBinding(cache, bindings[1]);
	Check(name, through_b != 0x77,
	      "the write was visible to B; no aliasing problem to fix");
	if (through_b != 0x77) {
		Pass(name, "write landed in the retired buffer, invisible to the live one");
	}
}

void TestCapturedAddressGoesStale() {
	const char* name = "captured device address goes stale";
	ModelCache  cache;
	auto        bindings = BuildCase(cache);
	MaterialiseBindings(cache, bindings, /*refresh_after=*/false);
	const auto live_address = cache.buffers[bindings[1].handle].address;
	Check(name, bindings[0].address != live_address,
	      "A's captured address already matched the live buffer");
	if (bindings[0].address != live_address) {
		Pass(name, "A holds the retired allocation's address, not the replacement's");
	}
}

// ---------------------------------------------------------------------------------------------
// The fix: finalise resolution, then refresh anything whose buffer was retired during the loop.

void TestFixRestoresCoherence() {
	const char* name = "fix: refreshed binding sees current bytes";
	ModelCache  cache;
	auto        bindings = BuildCase(cache);
	MaterialiseBindings(cache, bindings, /*refresh_after=*/true);

	cache.GuestWrite(0x10000, 0x99);
	cache.Synchronize(bindings[1].handle, bindings[1].vaddr, bindings[1].size);

	const bool same_buffer = bindings[0].handle == bindings[1].handle;
	const bool a_live      = !cache.buffers[bindings[0].handle].deleted;
	const auto through_a   = ReadThroughBinding(cache, bindings[0]);
	Check(name, a_live, "A still points at a retired buffer after the refresh");
	Check(name, same_buffer, "A and B did not converge on the replacement");
	Check(name, through_a == 0x99,
	      "A read 0x" + std::to_string(through_a) + ", expected the current 0x99");
	Check(name, bindings[0].address == cache.buffers[bindings[1].handle].address,
	      "A's device address was not refreshed");
	if (a_live && same_buffer && through_a == 0x99) {
		Pass(name, "A refreshed onto the replacement: handle, offset and address all current");
	}
}

void TestFixKeepsWritesVisible() {
	const char* name = "fix: writable alias stays coherent";
	ModelCache  cache;
	auto        bindings = BuildCase(cache);
	MaterialiseBindings(cache, bindings, /*refresh_after=*/true);

	cache.buffers[bindings[0].handle].bytes[static_cast<size_t>(bindings[0].offset)] = 0x77;
	Check(name, ReadThroughBinding(cache, bindings[1]) == 0x77,
	      "write through A still invisible to B after the fix");
	if (ReadThroughBinding(cache, bindings[1]) == 0x77) {
		Pass(name, "both bindings now alias the same live allocation");
	}
}

// The refresh must not fire when nothing was replaced - it is a bounded repair, not a second
// resolution pass on every draw.
void TestFixIsInertWhenNothingReplaced() {
	const char* name = "fix does nothing when no replacement";
	ModelCache  cache;
	cache.GuestWrite(0x20000, 0x33);
	std::vector<Binding> bindings;
	Binding              only;
	only.vaddr     = 0x20000;
	only.size      = kPage;
	only.cached_id = cache.Find(only.vaddr, only.size);
	bindings.push_back(only);

	const auto before = cache.generation;
	MaterialiseBindings(cache, bindings, /*refresh_after=*/true);
	Check(name, cache.generation == before,
	      "a replacement happened where none was expected");
	Check(name, ReadThroughBinding(cache, bindings[0]) == 0x33, "contents wrong");
	if (cache.generation == before) {
		Pass(name, "no replacement, no extra work, contents correct");
	}
}

} // namespace

int main() {
	std::printf("Within-draw stale bindings:\n\n");
	TestCaseIsReachable();
	TestStaleBindingReadsStaleBytes();
	TestWritableAliasIsLost();
	TestCapturedAddressGoesStale();
	TestFixRestoresCoherence();
	TestFixKeepsWritesVisible();
	TestFixIsInertWhenNothingReplaced();
	if (g_failures != 0) {
		std::printf("\n%d stale-binding check(s) FAILED\n", g_failures);
		return 1;
	}
	std::printf("\nall stale-binding checks passed\n");
	return 0;
}
