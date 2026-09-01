#include "libs/controller.h"

#include "SDL.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "libs/padData.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace Libs::Controller {

LIB_NAME("Pad", "Pad");

constexpr int PAD_ERROR_INVALID_ARG    = -2137915391; /* 0x80920001 */
constexpr int PAD_ERROR_INVALID_HANDLE = -2137915389; /* 0x80920003 */

// SDL limits rumble commands to 0xffff ms; zero strengths stop immediately.
constexpr uint32_t RUMBLE_DURATION_MS = 0xffff;

struct PadControllerInformation {
	float    touch_pixel_density;
	uint16_t touch_resolution_x;
	uint16_t touch_resolution_y;
	uint8_t  stick_dead_zone_left;
	uint8_t  stick_dead_zone_right;
	uint8_t  connection_type;
	uint8_t  connected_count;
	bool     connected;
	int      device_class;
	uint8_t  reserve[8];
};

struct PadLightBarParam {
	uint8_t r        = 0;
	uint8_t g        = 0;
	uint8_t b        = 0;
	uint8_t reserve  = 0;
};

struct PadVibrationParam {
	uint8_t large_motor;
	uint8_t small_motor;
};

struct TouchPoint {
	uint16_t x    = 0;
	uint16_t y    = 0;
	bool     down = false;
};

struct ControllerState {
	uint64_t time                                  = 0;
	uint32_t buttons                               = 0;
	int      axes[static_cast<int>(Axis::AxisMax)] = {128, 128, 128, 128, 0, 0};
	TouchPoint touches[2]                            = {};
};

class GameController {
public:
	GameController()          = default;
	virtual ~GameController() = default;

	KYTY_CLASS_NO_COPY(GameController);

	void Connect(int id);
	void Disconnect(int id);
	void Button(int id, uint32_t button, bool down);
	void Axis(int id, Axis axis, int value);
	void Touch(int id, int finger, uint16_t x, uint16_t y, bool down);
	void SetLightBar(uint8_t r, uint8_t g, uint8_t b);
	void SetTriggerEffect(const uint8_t* left, const uint8_t* right);
	void ExpireTriggerEffect();
	void RightStick(int id, int x, int y);
	void ResetInputState();
	void GetConnectionInfo(bool* flag, int* count);
	void SetVibration(uint8_t large_motor, uint8_t small_motor);
	void ReadState(ControllerState* state, bool* flag, int* count);
	int  ReadStates(ControllerState* states, int states_num, bool* flag, int* count);

private:
	static constexpr uint32_t STATES_MAX = 64;

	struct StatePrivate {
		bool obtained = false;
	};

	void                          CheckActive();
	[[nodiscard]] ControllerState GetLastState() const;
	void                          AddState(const ControllerState& state);

	SDL_hid_device*  m_hid           = nullptr;
	bool             m_hid_tried     = false;
	bool             m_hid_bluetooth = false;
	uint8_t          m_last_trigger[22] = {};
	bool             m_trigger_sent     = false;
	uint64_t         m_trigger_time     = 0;
	Common::Mutex    m_mutex;
	std::vector<int> m_connected_ids;
	int              m_active_id       = -1;
	bool             m_connected       = false;
	int              m_connected_count = 0;
	ControllerState  m_states[STATES_MAX];
	StatePrivate     m_private[STATES_MAX];
	ControllerState  m_last_state;
	uint32_t         m_states_num  = 0;
	uint32_t         m_first_state = 0;
};

static GameController* g_controller = nullptr;

static uint8_t pad_connected_count_to_u8(int connected_count) {
	return static_cast<uint8_t>(connected_count > 255 ? 255 : connected_count);
}

static void pad_fill_data(PadData* data, const ControllerState& state, bool connected,
                          int connected_count) {
	EXIT_IF(data == nullptr);

	std::memset(data, 0, sizeof(*data));

	data->buttons                = state.buttons;
	data->left_stick_x           = state.axes[static_cast<int>(Axis::LeftX)];
	data->left_stick_y           = state.axes[static_cast<int>(Axis::LeftY)];
	data->right_stick_x          = state.axes[static_cast<int>(Axis::RightX)];
	data->right_stick_y          = state.axes[static_cast<int>(Axis::RightY)];
	data->analog_buttons_l2      = state.axes[static_cast<int>(Axis::TriggerLeft)];
	data->analog_buttons_r2      = state.axes[static_cast<int>(Axis::TriggerRight)];
	data->orientation_w          = 1.0f;
	// touch_data_touch_num was never set, so titles saw zero touch points and the touchpad
	// registered only as a button. Report the live points.
	data->touch_data_touch0_id   = 1;
	data->touch_data_touch1_id   = 2;
	data->touch_data_touch_num   = static_cast<uint8_t>((state.touches[0].down ? 1 : 0) +
	                                                    (state.touches[1].down ? 1 : 0));
	data->touch_data_touch0_x    = state.touches[0].x;
	data->touch_data_touch0_y    = state.touches[0].y;
	data->touch_data_touch1_x    = state.touches[1].x;
	data->touch_data_touch1_y    = state.touches[1].y;
	data->connected              = connected;
	data->timestamp              = state.time;
	data->connected_count        = pad_connected_count_to_u8(connected_count);
	data->device_unique_data_len = 0;
}

void Initialize() {
	EXIT_IF(g_controller != nullptr);

	g_controller = new GameController;
	g_controller->Connect(HOST_INPUT_CONTROLLER_ID);
}

void Shutdown() {
	// Release any held trigger effect: a DualSense keeps the last one until told otherwise,
	// so a stale effect would survive the emulator exiting.
	if (g_controller != nullptr) {
		const uint8_t off[11] = {};
		g_controller->SetTriggerEffect(off, off);
	}
	delete g_controller;
	g_controller = nullptr;
}

void GameController::Connect(int id) {
	Common::LockGuard lock(m_mutex);

	if (std::find(m_connected_ids.begin(), m_connected_ids.end(), id) != m_connected_ids.end()) {
		return;
	}

	m_connected_ids.push_back(id);

	CheckActive();
}

void GameController::Disconnect(int id) {
	Common::LockGuard lock(m_mutex);

	const auto it = std::find(m_connected_ids.begin(), m_connected_ids.end(), id);
	if (it == m_connected_ids.end()) {
		// Disconnecting an id that is not connected is a no-op, mirroring Connect() above, which
		// already returns early when the id is present. The guest can legitimately issue a
		// redundant disconnect (or one for a pad that was never connected); aborting there took
		// the whole emulator down mid-run.
		return;
	}

	m_connected_ids.erase(it);

	CheckActive();
}

void GameController::CheckActive() {
	bool reset         = false;
	int  new_active_id = -1;
	bool new_connected = false;

	if (!m_connected_ids.empty()) {
		new_active_id = m_connected_ids[0];
		for (const auto id: m_connected_ids) {
			if (id != HOST_INPUT_CONTROLLER_ID) {
				new_active_id = id;
				break;
			}
		}
		new_connected = true;
	}

	const bool was_connected = m_connected;

	if (m_connected != new_connected || m_active_id != new_active_id) {
		m_active_id = new_active_id;
		m_connected = new_connected;
		if (!was_connected && new_connected) {
			m_connected_count++;
		}
		reset = true;
	}

	if (reset) {
		m_states_num = 0;
		m_last_state = ControllerState();
	}
}

ControllerState GameController::GetLastState() const {
	if (m_states_num == 0) {
		return m_last_state;
	}

	auto last = (m_first_state + m_states_num - 1) % STATES_MAX;

	return m_states[last];
}

void GameController::AddState(const ControllerState& state) {
	if (m_states_num >= STATES_MAX) {
		m_states_num  = STATES_MAX - 1;
		m_first_state = (m_first_state + 1) % STATES_MAX;
	}

	auto index = (m_first_state + m_states_num) % STATES_MAX;

	m_states[index] = state;
	m_last_state    = state;

	m_private[index].obtained = false;

	m_states_num++;
}

void GameController::Button(int id, uint32_t button, bool down) {
	Common::LockGuard lock(m_mutex);

	// The keyboard shares the player-1 pad with the active gamepad.
	if (m_active_id == id || id == HOST_INPUT_CONTROLLER_ID) {
		auto state = GetLastState();

		state.time = LibKernel::KernelGetProcessTime();

		if (down) {
			state.buttons |= button;
		} else {
			state.buttons &= ~button;
		}

		AddState(state);
	}
}

void GameController::Axis(int id, Controller::Axis axis, int value) {
	Common::LockGuard lock(m_mutex);

	if (m_active_id == id || id == HOST_INPUT_CONTROLLER_ID) {
		auto state = GetLastState();

		state.time = LibKernel::KernelGetProcessTime();

		int axis_id = static_cast<int>(axis);

		EXIT_IF(axis_id < 0 || axis_id >= static_cast<int>(Controller::Axis::AxisMax));

		state.axes[axis_id] = value;

		if (axis == Controller::Axis::TriggerLeft) {
			if (value > 0) {
				state.buttons |= PAD_BUTTON_L2;
			} else {
				state.buttons &= ~PAD_BUTTON_L2;
			}
		}

		if (axis == Controller::Axis::TriggerRight) {
			if (value > 0) {
				state.buttons |= PAD_BUTTON_R2;
			} else {
				state.buttons &= ~PAD_BUTTON_R2;
			}
		}

		AddState(state);
	}
}

void GameController::RightStick(int id, int x, int y) {
	Common::LockGuard lock(m_mutex);

	if (m_active_id == id || id == HOST_INPUT_CONTROLLER_ID) {
		auto state                                 = GetLastState();
		state.time                                 = LibKernel::KernelGetProcessTime();
		state.axes[static_cast<int>(Axis::RightX)] = x;
		state.axes[static_cast<int>(Axis::RightY)] = y;
		AddState(state);
	}
}

void GameController::ResetInputState() {
	Common::LockGuard lock(m_mutex);
	ControllerState   state {};
	state.time    = LibKernel::KernelGetProcessTime();
	m_states_num  = 0;
	m_first_state = 0;
	AddState(state);
}

// A title re-sends its trigger effect every frame for as long as it wants it. When it stops - the
// player left the vehicle, holstered the weapon - it does not always send an explicit "off", and a
// DualSense holds the last effect forever. Release it once the requests stop.
void GameController::ExpireTriggerEffect() {
	uint64_t last = 0;
	{
		Common::LockGuard lock(m_mutex);
		if (!m_trigger_sent || m_trigger_time == 0) {
			return;
		}
		last = m_trigger_time;
	}
	constexpr uint64_t ExpiryUs = 400000; // 0.4s of silence
	if (LibKernel::KernelGetProcessTime() - last < ExpiryUs) {
		return;
	}
	const uint8_t off[11] = {};
	SetTriggerEffect(off, off);
	{
		Common::LockGuard lock(m_mutex);
		m_trigger_time = 0;
	}
}

void GameController::SetVibration(uint8_t large_motor, uint8_t small_motor) {
	Common::LockGuard lock(m_mutex);

	if (m_active_id == HOST_INPUT_CONTROLLER_ID) {
		return;
	}

	auto* pad = SDL_GameControllerFromInstanceID(static_cast<SDL_JoystickID>(m_active_id));
	if (pad == nullptr) {
		return;
	}

	const auto large = static_cast<uint16_t>(large_motor * 0x101U);
	const auto small = static_cast<uint16_t>(small_motor * 0x101U);
	if (SDL_GameControllerRumble(pad, large, small, RUMBLE_DURATION_MS) != 0) {
		LOGF("\t rumble failed: %s\n", SDL_GetError());
	}
}

void GameController::Touch(int id, int finger, uint16_t x, uint16_t y, bool down) {
	Common::LockGuard lock(m_mutex);

	if ((m_active_id != id && id != HOST_INPUT_CONTROLLER_ID) || finger < 0 || finger > 1) {
		return;
	}
	auto state = GetLastState();
	state.time = LibKernel::KernelGetProcessTime();
	state.touches[finger].x    = x;
	state.touches[finger].y    = y;
	state.touches[finger].down = down;
	AddState(state);
}

void GameController::SetLightBar(uint8_t r, uint8_t g, uint8_t b) {
	Common::LockGuard lock(m_mutex);
	if (m_active_id < 0) {
		return;
	}
	auto* pad = SDL_GameControllerFromInstanceID(static_cast<SDL_JoystickID>(m_active_id));
	if (pad == nullptr) {
		return;
	}
	// A pad without an LED just reports failure; there is nothing to recover from.
	(void)SDL_GameControllerSetLED(pad, r, g, b);
}

// SDL owns its own output report and refuses ours ("device disconnected"), so trigger effects go
// straight to the pad over HID - the same route the DualSense web testers use. The USB report is
// 0x02 with the effect payload at offset 1; Bluetooth is 0x31, offset 2, with a CRC32 over an 0xA2
// header byte plus the report. Right trigger sits at payload offset 10, left at 21.
void GameController::SetTriggerEffect(const uint8_t* left, const uint8_t* right) {
	Common::LockGuard lock(m_mutex);
	if (left == nullptr || right == nullptr) {
		return;
	}
	if (!m_hid_tried) {
		m_hid_tried = true;
		for (const auto product: {0x0ce6, 0x0df2}) {
			m_hid = SDL_hid_open(0x054c, static_cast<unsigned short>(product), nullptr);
			if (m_hid != nullptr) {
				// A Bluetooth DualSense reports 78-byte input; USB reports 64. Probe once.
				uint8_t probe[78] = {};
				const int read = SDL_hid_read_timeout(m_hid, probe, sizeof(probe), 50);
				m_hid_bluetooth = read > 64;
				::printf("DualSense HID opened (product=0x%04x, bluetooth=%d)\n", product,
				         m_hid_bluetooth ? 1 : 0);
				break;
			}
		}
		if (m_hid == nullptr) {
			::printf("DualSense HID open failed: %s\n", SDL_GetError());
		}
	}
	// The title re-sends the same effect every frame. A DualSense holds the last one, so writing it
	// again is pure Bluetooth traffic - only send when it actually changes.
	m_trigger_time = LibKernel::KernelGetProcessTime();
	uint8_t combined[22] = {};
	std::memcpy(combined, right, 11);
	std::memcpy(combined + 11, left, 11);
	if (m_trigger_sent && std::memcmp(combined, m_last_trigger, sizeof(combined)) == 0) {
		return;
	}
	std::memcpy(m_last_trigger, combined, sizeof(combined));
	m_trigger_sent = true;
	if (m_hid == nullptr) {
		return;
	}
	uint8_t report[79] = {};
	int     size       = 0;
	int     offset     = 0;
	if (m_hid_bluetooth) {
		report[0] = 0x31;
		report[1] = 0x02;
		size      = 78;
		offset    = 2;
	} else {
		report[0] = 0x02;
		size      = 48;
		offset    = 1;
	}
	// Only the trigger-effect bits, so rumble and LED state the driver owns are left alone.
	report[offset + 0] = 0x0c;
	std::memcpy(report + offset + 10, right, 11);
	std::memcpy(report + offset + 21, left, 11);
	if (m_hid_bluetooth) {
		const uint8_t header = 0xa2;
		uint32_t      crc    = SDL_crc32(0, &header, 1);
		crc                  = SDL_crc32(crc, report, static_cast<size_t>(size) - sizeof(crc));
		std::memcpy(report + size - sizeof(crc), &crc, sizeof(crc));
	}
	if (SDL_hid_write(m_hid, report, static_cast<size_t>(size)) < 0) {
		static std::atomic<uint32_t> fail_log {0};
		if (fail_log.fetch_add(1, std::memory_order_relaxed) < 5) {
			::printf("DualSense HID write failed: %s\n", SDL_GetError());
		}
	}
}

void GameController::GetConnectionInfo(bool* flag, int* count) {
	EXIT_IF(flag == nullptr);
	EXIT_IF(count == nullptr);

	Common::LockGuard lock(m_mutex);

	*flag  = m_connected;
	*count = m_connected_count;
}

void GameController::ReadState(ControllerState* state, bool* flag, int* count) {
	EXIT_IF(flag == nullptr);
	EXIT_IF(count == nullptr);
	EXIT_IF(state == nullptr);

	Common::LockGuard lock(m_mutex);

	*flag  = m_connected;
	*count = m_connected_count;
	*state = GetLastState();
}

int GameController::ReadStates(ControllerState* states, int states_num, bool* flag, int* count) {
	EXIT_IF(flag == nullptr);
	EXIT_IF(count == nullptr);
	EXIT_IF(states == nullptr);
	EXIT_IF(states_num < 1 || states_num > STATES_MAX);

	Common::LockGuard lock(m_mutex);

	*flag  = m_connected;
	*count = m_connected_count;

	int ret_num = 0;

	if (m_connected) {
		if (m_states_num != 0) {
			for (uint32_t i = 0; i < m_states_num; i++) {
				if (ret_num >= states_num) {
					break;
				}
				auto index = (m_first_state + i) % STATES_MAX;
				if (!m_private[index].obtained) {
					m_private[index].obtained = true;

					states[ret_num++] = m_states[index];
				}
			}
		}
	}

	return ret_num;
}

void ControllerConnect(int id) {
	EXIT_IF(g_controller == nullptr);

	g_controller->Connect(id);
}

void ControllerDisconnect(int id) {
	EXIT_IF(g_controller == nullptr);

	g_controller->Disconnect(id);
}

void ControllerButton(int id, uint32_t button, bool down) {
	EXIT_IF(g_controller == nullptr);

	g_controller->Button(id, button, down);
}

void ControllerAxis(int id, Axis axis, int value) {
	EXIT_IF(g_controller == nullptr);

	g_controller->Axis(id, axis, value);
}

void ControllerTouch(int id, int finger, uint16_t x, uint16_t y, bool down) {
	EXIT_IF(g_controller == nullptr);
	g_controller->Touch(id, finger, x, y, down);
}

void ControllerTriggerEffect(const uint8_t* left, const uint8_t* right) {
	EXIT_IF(g_controller == nullptr);
	g_controller->SetTriggerEffect(left, right);
}

void ControllerRightStick(int id, int x, int y) {
	EXIT_IF(g_controller == nullptr);

	g_controller->RightStick(id, x, y);
}

void ControllerResetInputState() {
	EXIT_IF(g_controller == nullptr);
	g_controller->ResetInputState();
}

int KYTY_SYSV_ABI PadInit() {
	PRINT_NAME();

	return OK;
}

static bool PadOpenArgsAreValid(int user_id, int type, int index) {
	constexpr int user_id_system     = 0xff;
	constexpr int port_type_standard = 0;
	constexpr int port_type_special  = 2;
	constexpr int port_type_remote   = 16;
	const bool    personal_port =
	    user_id == Config::GetUserId() && (type == port_type_standard || type == port_type_special);
	const bool system_remote_control = user_id == user_id_system && type == port_type_remote;
	return index == 0 && (personal_port || system_remote_control);
}

int KYTY_SYSV_ABI PadOpen(int user_id, int type, int index, const void* param) {
	PRINT_NAME();

	LOGF("\t user_id = %d\n"
	     "\t type    = %d\n"
	     "\t index   = %d\n"
	     "\t param   = 0x%016" PRIx64 "\n",
	     user_id, type, index, reinterpret_cast<uint64_t>(param));

	constexpr int pad_error_invalid_arg = -2137915391; /* 0x80920001 */

	if (!PadOpenArgsAreValid(user_id, type, index)) {
		return pad_error_invalid_arg;
	}

	int handle = 1;

	return handle;
}

int KYTY_SYSV_ABI PadGetHandle(int user_id, int type, int index) {
	PRINT_NAME();

	LOGF("\t user_id = %d\n"
	     "\t type    = %d\n"
	     "\t index   = %d\n",
	     user_id, type, index);

	constexpr int pad_error_device_no_handle = -2137915384; /* 0x80920008 */

	if (!PadOpenArgsAreValid(user_id, type, index)) {
		return pad_error_device_no_handle;
	}

	return 1;
}

int KYTY_SYSV_ABI PadSetMotionSensorState(int handle, bool enable) {
	PRINT_NAME();

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	LOGF("\t enable = %s\n", (enable ? "true" : "false"));

	return OK;
}

int KYTY_SYSV_ABI PadSetAngularVelocityDeadbandState(int handle, bool enable) {
	PRINT_NAME();

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	LOGF("\t enable = %s\n", (enable ? "true" : "false"));

	return OK;
}

int KYTY_SYSV_ABI PadResetOrientation(int handle) {
	PRINT_NAME();

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	return OK;
}

int KYTY_SYSV_ABI PadGetControllerInformation(int handle, PadControllerInformation* info) {
	PRINT_NAME();

	EXIT_IF(g_controller == nullptr);

	int  connected_count = 0;
	bool connected       = false;

	g_controller->GetConnectionInfo(&connected, &connected_count);

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (info == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	std::memset(info, 0, sizeof(*info));

	info->touch_pixel_density   = 44.86f;
	info->touch_resolution_x    = 1920;
	info->touch_resolution_y    = 943;
	info->stick_dead_zone_left  = controller_get_axis(-32768, 32767, 8000) - 128;
	info->stick_dead_zone_right = controller_get_axis(-32768, 32767, 8000) - 128;
	info->connection_type       = 0;
	info->connected_count       = pad_connected_count_to_u8(connected_count);
	info->connected             = connected;
	info->device_class          = 0;

	return OK;
}

int KYTY_SYSV_ABI PadReadState(int handle, PadData* data) {
	PRINT_NAME();

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (data == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	EXIT_IF(g_controller == nullptr);
	// Titles poll the pad every frame; use that to release a trigger effect nothing is asking for.
	g_controller->ExpireTriggerEffect();

	int             connected_count = 0;
	bool            connected       = false;
	ControllerState state;

	g_controller->ReadState(&state, &connected, &connected_count);

	pad_fill_data(data, state, connected, connected_count);

	return OK;
}

int KYTY_SYSV_ABI PadRead(int handle, PadData* data, int num) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(num < 1 || num > 64);
	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (data == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	std::memset(data, 0, sizeof(PadData) * static_cast<size_t>(num));

	EXIT_IF(g_controller == nullptr);

	int             connected_count = 0;
	bool            connected       = false;
	ControllerState states[64]      = {};

	int ret_num = g_controller->ReadStates(states, num, &connected, &connected_count);

	if (!connected || ret_num == 0) {
		if (connected) {
			g_controller->ReadState(&states[0], &connected, &connected_count);
		}
		ret_num = 1;
	}

	for (int i = 0; i < ret_num; i++) {
		pad_fill_data(&data[i], states[i], connected, connected_count);
	}

	return ret_num;
}

int KYTY_SYSV_ABI PadSetVibration(int handle, const PadVibrationParam* param) {
	PRINT_NAME();

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (param == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	LOGF("\t large_motor = %d\n"
	     "\t small_motor = %d\n",
	     static_cast<int>(param->large_motor), static_cast<int>(param->small_motor));

	EXIT_IF(g_controller == nullptr);
	g_controller->SetVibration(param->large_motor, param->small_motor);

	return OK;
}

int KYTY_SYSV_ABI PadResetLightBar(int handle) {
	PRINT_NAME();

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}

	return OK;
}

int KYTY_SYSV_ABI PadSetLightBar(int handle, const PadLightBarParam* param) {
	PRINT_NAME();

	if (handle != 1) {
		return PAD_ERROR_INVALID_HANDLE;
	}
	if (param == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}

	// The light bar was accepted and dropped. Drive the real pad so titles that signal state
	// through it (police lights, health, player colour) behave as intended.
	EXIT_IF(g_controller == nullptr);
	g_controller->SetLightBar(param->r, param->g, param->b);

	return OK;
}

} // namespace Libs::Controller
