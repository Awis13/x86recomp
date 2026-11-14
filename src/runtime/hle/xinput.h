#pragma once
// The XInput pads (xinput.cpp) as other APIs list them: DirectInput's game
// controllers (dinput8.cpp) and WMI's PnP devices (wbem.cpp), which games
// match up to find which DirectInput devices are XInput ones.

#include <cstdint>
#include <string>
#include <vector>

#include "runtime/hle/pad_effects.h"

namespace recomp::hle {

struct XInputPad {
  uint32_t index;     // XInput user index
  uint16_t vid, pid;  // USB vendor and product id
  std::string name;
  bool playstation = false;  // a PlayStation pad (its button symbols)
};

// The connected pads in user index order. Refreshes the pad list on the
// main thread (elsewhere: the pads already open).
std::vector<XInputPad> XInputPads();

// A pad's controls in SDL's (Xbox) layout: sticks -32768..32767 with Y
// down, triggers 0..32767, buttons indexed by SDL_GamepadButton.
struct PadState {
  int16_t axis[6];  // SDL_GamepadAxis order: LX, LY, RX, RY, LT, RT
  bool button[32];
};

// The pad at user index `index` (refreshed on the main thread); false if
// none is connected there.
bool ReadPad(uint32_t index, PadState* out);

// Rumble on the pad at `index` (motor speeds 0..65535) for `ms`; 0, 0
// stops it. No effect without a pad or motors.
void RumblePad(uint32_t index, uint16_t low, uint16_t high, uint32_t ms);

// The trigger effects and the light (pad_effects.h) of the pad at `index`,
// for a pad that has them (a DualSense): each is sent when it is not what
// the pad has taken, so a refused one goes again at the next call. False,
// and nothing sent, without such a pad there. The light is left alone
// until a colour is asked for, and gets the pad's own colour back when
// none is asked for any more.
bool SetPadEffects(uint32_t index, const PadEffects& effects);

// Frees the triggers of every pad that was sent effects and gives a light
// its own colour back, then waits a moment for that to reach the pad: for
// a port's before_exit (at exit only), as a pad keeps its last effects
// after the process has gone.
void ClearPadEffects();

// Test hook (<ID>_TEST_PAD=1, also in automated runs): a virtual gamepad
// SDL lists like a real one, driven by <ID>_INPUT_LIVE's `pad` and `stick`
// lines. Button and axis are SDL_GamepadButton / SDL_GamepadAxis.
void TestPadButton(int button, bool down);
void TestPadAxis(int axis, int16_t value);

}  // namespace recomp::hle
