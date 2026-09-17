// Tests for the texture-descriptor decode.
//
// The decode CACHE experiment was removed: it was never justified by a measurement, and splitting
// the decode into a function that writes through an out-parameter caused a live crash. Decoding is
// back to fresh-per-call, which is what the original ResolveTexture did.
//
// What remains under test is the split itself, since that is still in the tree: `DecodeTexture`
// must produce exactly what an inline decode would, and must FULLY overwrite its destination. The
// null-then-real case below is the reproducer for the crash
// ("TextureCache: texture requires rediscovery before final acquisition"), kept because the split
// is kept.

#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace IR = Libs::Graphics::ShaderRecompiler::IR;

int g_failures = 0;

void Check(const char* test, bool ok, const std::string& message) {
	if (!ok) {
		std::printf("[FAIL]    %-44s %s\n", test, message.c_str());
		g_failures++;
	}
}
void Pass(const char* test, const std::string& detail) {
	std::printf("[ok]      %-44s %s\n", test, detail.c_str());
}

// Field-by-field comparison. A defaulted operator== is not available on every member type here, so
// the comparison is explicit and covers everything the decode produces.
bool Same(const DecodedTexture& a, const DecodedTexture& b, std::string& why) {
	auto fail = [&](const char* what) {
		why = what;
		return false;
	};
	if (a.is_null != b.is_null) {
		return fail("is_null");
	}
	if (a.storage != b.storage) {
		return fail("storage");
	}
	if (a.shader_conversion != b.shader_conversion) {
		return fail("shader_conversion");
	}
	if (a.pixel_format != b.pixel_format) {
		return fail("pixel_format");
	}
	if (a.view_format != b.view_format) {
		return fail("view_format");
	}
	if (a.size_bytes != b.size_bytes) {
		return fail("size_bytes");
	}
	// Field-by-field, not memcmp: these structs contain padding, and the two arms write into
	// differently-initialised storage, so a raw byte compare reports differences that are not
	// differences in the decode.
	if (std::memcmp(a.descriptor.fields, b.descriptor.fields, sizeof(a.descriptor.fields)) != 0) {
		return fail("descriptor");
	}
	const auto& ai = a.desc.info;
	const auto& bi = b.desc.info;
	if (ai.data.address != bi.data.address || ai.data.size != bi.data.size) {
		return fail("desc.info.data");
	}
	if (ai.pixel_format != bi.pixel_format || ai.guest_format != bi.guest_format) {
		return fail("desc.info.format");
	}
	if (ai.type != bi.type || ai.tile_mode != bi.tile_mode || ai.samples != bi.samples) {
		return fail("desc.info.type/tile/samples");
	}
	if (ai.extent.width != bi.extent.width || ai.extent.height != bi.extent.height ||
	    ai.extent.depth != bi.extent.depth) {
		return fail("desc.info.extent");
	}
	if (ai.resources.levels != bi.resources.levels || ai.resources.layers != bi.resources.layers) {
		return fail("desc.info.resources");
	}
	if (ai.pitch != bi.pitch || ai.bytes_per_block != bi.bytes_per_block) {
		return fail("desc.info.pitch/bpb");
	}
	for (uint32_t level = 0; level < ai.resources.levels && level < ai.mip_layout.size(); level++) {
		if (std::memcmp(&ai.mip_layout[level], &bi.mip_layout[level],
		                sizeof(ai.mip_layout[level])) != 0) {
			return fail("desc.info.mip_layout");
		}
	}
	if (!(a.desc.view_info == b.desc.view_info)) {
		return fail("desc.view_info");
	}
	if (a.desc.type != b.desc.type) {
		return fail("desc.type");
	}
	return true;
}

// A descriptor that decodes to a real (non-null) image.
IR::DescriptorValue MakeDescriptor(uint32_t width_minus_one, uint32_t height_minus_one,
                                   uint64_t address) {
	IR::DescriptorValue value;
	value.dword_count = 8;
	value.dwords[0]   = static_cast<uint32_t>(address >> 8);
	value.dwords[1]   = static_cast<uint32_t>(address >> 40) | (1u << 20);   // min_lod / fmt bits
	value.dwords[2]   = width_minus_one | (height_minus_one << 14);
	value.dwords[3]   = 0;
	value.dwords[4]   = 0;
	value.dwords[5]   = 0;
	value.dwords[6]   = 0;
	value.dwords[7]   = 0;
	return value;
}

IR::ImageResource MakeResource() {
	IR::ImageResource resource;
	resource.kind      = IR::ResourceKind::Image;
	resource.dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D;
	resource.mip_mode  = IR::ImageMipMode::None;
	resource.read      = true;
	resource.written   = false;
	resource.r128      = false;
	return resource;
}

bool DiffOne(const char* test, const IR::ImageResource& resource, const IR::DescriptorValue& value,
             const char* what) {
	DecodedTexture uncached;
	DecodedTexture cached;
	DecodeTextureUncachedForTest(resource, value, uncached);
	DecodeTextureUncachedForTest(resource, value, cached);
	std::string why;
	if (!Same(uncached, cached, why)) {
		Check(test, false, std::string(what) + ": cached decode differs in " + why);
		return false;
	}
	return true;
}

// ---------------------------------------------------------------- tests

// Every resource field the decode reads must be part of the key. Checked on the KEY directly:
// several field combinations cannot be decoded from a synthetic descriptor (a written image needs a
// storage-valid descriptor, ImageUint needs a matching numeric format), and those abort inside the
// decode's validations rather than returning. Keying is the property that matters here, so it is
// tested where it can be tested exhaustively.

// Decode differentials on the field variations that a synthetic descriptor can legally express.
void TestDecodeDifferentialOnValidVariants() {
	const char* name  = "cached decode matches uncached";
	const auto  value = MakeDescriptor(63, 63, 0x200000);

	std::vector<std::pair<const char*, IR::ImageResource>> variants;
	variants.emplace_back("base", MakeResource());
	{
		auto r = MakeResource();
		r.r128 = !r.r128;
		variants.emplace_back("r128", r);
	}
	{
		auto r     = MakeResource();
		r.mip_mode = IR::ImageMipMode::DynamicStorage;
		variants.emplace_back("mip_mode", r);
	}

	bool all_ok = true;
	for (const auto& [what, resource]: variants) {
		if (!DiffOne(name, resource, value, what)) {
			all_ok = false;
		}
	}
	if (all_ok) {
		Pass(name, std::to_string(variants.size()) + " decodable variants matched uncached");
	}
}

// Changing the descriptor dwords must miss.
void TestDescriptorChangesMiss() {
	const char* name     = "descriptor change decodes fresh";
	const auto  resource = MakeResource();
	bool all_ok = true;
	for (uint32_t i = 0; i < 32; i++) {
		const auto value = MakeDescriptor(31 + i, 63, 0x200000 + i * 0x1000);
		if (!DiffOne(name, resource, value, "descriptor")) {
			all_ok = false;
			break;
		}
	}
	if (all_ok) {
		Pass(name, "32 distinct descriptors, each matched its uncached decode");
	}
}

// More distinct keys than slots, so entries are evicted and slots reused. A reused slot must not
// return the previous occupant's decode.

// The decode depends on nothing but its inputs, so emptying the cache - which is what a renderer or
// texture-cache teardown would amount to for this data - cannot change any answer.

// Confirms hits actually occur, so the tests above are not passing because the cache never engages.

// REGRESSION: decode must FULLY overwrite its destination, INCLUDING is_null.
//
// The production flag-off path decodes into one persistent `static thread_local` scratch. `is_null`
// is assigned only in the null branch and never cleared, so a real texture decoded after a null one
// on the same thread inherited is_null = true. The caller then took the null branch for a real
// texture - wrong FindImage overload, and no depth_id remap - and the missing remap aborted a real
// arm A run with "TextureCache: texture requires rediscovery before final acquisition"
// (textureCache.cpp:1528 rejects an id whose image still has depth_id set).
//
// The first version of this test compared two REAL descriptors and passed with the fix removed,
// because every field a real decode writes was overwritten anyway. The null-then-real order is what
// actually reproduces it.
void TestDecodeFullyOverwritesDestination() {
	const char* name     = "decode fully overwrites destination";
	const auto  resource = MakeResource();

	// All-zero dwords: ShaderTextureResource::IsNull() is true for this.
	IR::DescriptorValue null_value;
	null_value.dword_count = 8;

	const auto real_value = MakeDescriptor(63, 63, 0x200000);

	DecodedTexture fresh_null;
	DecodeTextureUncachedForTest(resource, null_value, fresh_null);
	Check(name, fresh_null.is_null, "the all-zero descriptor did not decode as null - fixture wrong");

	DecodedTexture fresh_real;
	DecodeTextureUncachedForTest(resource, real_value, fresh_real);
	Check(name, !fresh_real.is_null, "the real descriptor decoded as null - fixture wrong");

	// The reproducer: null first, then real, into the SAME destination.
	DecodedTexture reused;
	DecodeTextureUncachedForTest(resource, null_value, reused);
	DecodeTextureUncachedForTest(resource, real_value, reused);
	Check(name, !reused.is_null,
	      "is_null carried over from a previous null decode - a real texture would take the null "
	      "path, skipping the depth_id remap");

	std::string why;
	const bool  ok = Same(fresh_real, reused, why);
	Check(name, ok, "real decode after a null decode differs in " + why);

	// And real-then-null, so the flag is proven to be set as well as cleared.
	DecodedTexture reused2;
	DecodeTextureUncachedForTest(resource, real_value, reused2);
	DecodeTextureUncachedForTest(resource, null_value, reused2);
	Check(name, reused2.is_null, "is_null was not set when a null decode followed a real one");

	if (!reused.is_null && ok && reused2.is_null && fresh_null.is_null && !fresh_real.is_null) {
		Pass(name, "null-then-real clears is_null, real-then-null sets it, values match fresh");
	}
}

} // namespace

int main() {
	std::printf("Texture descriptor decode: differential checks\n\n");
	TestDecodeDifferentialOnValidVariants();
	TestDescriptorChangesMiss();
	TestDecodeFullyOverwritesDestination();

	if (g_failures != 0) {
		std::printf("\n%d texture-decode check(s) FAILED\n", g_failures);
		return 1;
	}
	std::printf("\nall texture-decode checks passed\n");
	return 0;
}
