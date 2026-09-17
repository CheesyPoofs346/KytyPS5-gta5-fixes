// Focused tests for the diagnostic --skip-ps-chksum selection.
//
// Covers default-off, exact selection against the 32-bit guest checksum register, and the value
// parser. Where the skip sits in the draw path is a source-structure property checked separately
// (check_ps_chksum_skip_placement.py); nothing here issues Vulkan commands.

#include "common/emulatorConfig.h"

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>

namespace {

int g_failures = 0;

void Check(const char* test, bool ok, const std::string& message) {
	if (!ok) {
		std::printf("[FAIL]    %-52s %s\n", test, message.c_str());
		g_failures++;
	}
}

std::string Hex(uint64_t value) {
	char text[32];
	std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
	return text;
}

void LoadSkipList(std::initializer_list<uint64_t> list) {
	Config::ConfigOptions cfg;
	cfg.skip_ps_chksum.assign(list.begin(), list.end());
	Config::Load(cfg);
}

void TestDefaultOff() {
	const char* test = "DefaultOff: empty list selects nothing";
	LoadSkipList({});
	for (uint64_t value: {uint64_t {0}, uint64_t {1}, uint64_t {0x1234abcd}, uint64_t {UINT32_MAX}}) {
		Check(test, !Config::ShouldSkipPixelShaderChksum(value), Hex(value));
	}
}

void TestExactSelection() {
	const char* test = "Exact: only the listed checksum is selected";
	LoadSkipList({0x1234abcd});
	Check(test, Config::ShouldSkipPixelShaderChksum(0x1234abcd), "listed value not selected");
	for (uint64_t value: {uint64_t {0x1234abcc}, uint64_t {0x1234abce}, uint64_t {0x11234abcd},
	                      uint64_t {0x1234abcd00}, uint64_t {0}}) {
		Check(test, !Config::ShouldSkipPixelShaderChksum(value), Hex(value));
	}
}

void TestMultipleEntries() {
	const char* test = "Multiple: each listed checksum, nothing else";
	LoadSkipList({0x11111111, 0x22222222, 0x22222222});
	Check(test, Config::ShouldSkipPixelShaderChksum(0x11111111), "first entry");
	Check(test, Config::ShouldSkipPixelShaderChksum(0x22222222), "duplicate entry");
	Check(test, !Config::ShouldSkipPixelShaderChksum(0x33333333), "unlisted value");
}

void TestParserAccepts() {
	const char* test = "Parser: accepts nonzero 32-bit values";
	const struct {
		const char* text;
		uint64_t    expected;
	} cases[] = {{"0x1234abcd", 0x1234abcd}, {"0X1234ABCD", 0x1234abcd}, {"305441741", 0x1234abcd},
	             {"0xffffffff", UINT32_MAX},  {"1", 1},                   {"0x1", 1}};
	for (const auto& c: cases) {
		uint64_t   value = 0;
		const bool ok    = Config::ParsePixelShaderChksum(c.text, value);
		Check(test, ok && value == c.expected, std::string(c.text) + " -> " + Hex(value));
	}
}

void TestParserRejects() {
	const char* test = "Parser: rejects values that cannot select exactly";
	const char* cases[] = {"",    "0",    "0x",          "0x0",  "-1",          "+1",        " 0x1",
	                       "0x1 ", "zz",  "0x12g",       "017",  "00",          "0x100000000",
	                       "4294967296", "18446744073709551616", "0x11234abcd"};
	for (const char* text: cases) {
		uint64_t value = 0xdeadbeef;
		Check(test, !Config::ParsePixelShaderChksum(text, value) && value == 0xdeadbeef,
		      std::string("\"") + text + "\" accepted or output modified");
	}
	uint64_t value = 0xdeadbeef;
	Check(test, !Config::ParsePixelShaderChksum(nullptr, value) && value == 0xdeadbeef, "nullptr");
}

void TestParseThenSelect() {
	const char* test = "End to end: parsed value selects the register value";
	uint64_t    value = 0;
	Check(test, Config::ParsePixelShaderChksum("0x9ABCDEF0", value), "parse failed");
	LoadSkipList({value});
	const uint32_t register_value = 0x9abcdef0u;   // written through the uint32_t chksum setter
	Check(test, Config::ShouldSkipPixelShaderChksum(register_value), "not selected");
	Check(test, !Config::ShouldSkipPixelShaderChksum(register_value ^ 1u), "neighbour selected");
}

} // namespace

int main() {
	Config::Initialize();
	TestDefaultOff();
	TestExactSelection();
	TestMultipleEntries();
	TestParserAccepts();
	TestParserRejects();
	TestParseThenSelect();
	LoadSkipList({});
	Config::Shutdown();
	if (g_failures != 0) {
		std::printf("%d pixel shader skip check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("pixel shader skip tests passed\n");
	return 0;
}
