#pragma once
// What some pads have beyond rumble: triggers with a motor that push back
// against the finger, and a light. Plain data, its encodings and the
// decision what a pad is sent next, with no device: xinput.cpp sends it
// (SetPadEffects in xinput.h).
//
// The one pad with such triggers so far is Sony's DualSense. Its trigger
// effects are 11-byte blocks in the output report (SDL's DS5EffectsState_t
// in SDL_hidapi_ps5.c gives the report's layout; the blocks' own layout is
// the one open DualSense libraries share): a mode byte, then for
//   off:        nothing;
//   feedback:   the active zones (10 bits), then 3 bits of strength per
//               zone (strength - 1);
//   weapon:     the start and end zones (a bit each), strength - 1;
//   vibration:  as feedback, with the amplitudes, and the frequency in Hz
//               in byte 9.
// The pull has ten zones, 0 (released) to 9 (fully in).

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace recomp::hle {

constexpr int kTriggerZones = 10;
constexpr uint8_t kTriggerMaxStrength = 8;
// A weapon effect's start and end zones: the pad takes no others.
constexpr uint8_t kWeaponStartMin = 2, kWeaponStartMax = 7, kWeaponEndMax = 8;

struct TriggerEffect {
  enum Mode : uint8_t {
    kOff,        // no resistance
    kFeedback,   // resists from `start` to the end of the pull
    kWeapon,     // resists from `start` and lets go at `end`, as a gun's trigger breaks
    kVibration,  // vibrates from `start` on, at `frequency`
  };
  Mode mode = kOff;
  uint8_t start = 0;      // zone 0..9 (kWeapon: 2..7)
  uint8_t end = 0;        // kWeapon only: a zone after `start`, up to 8
  uint8_t strength = 0;   // 1..8; 0 is off
  uint8_t frequency = 0;  // kVibration only, Hz; 0 is off

  bool operator==(const TriggerEffect&) const = default;
};

struct PadColour {
  uint8_t red = 0, green = 0, blue = 0;
  bool operator==(const PadColour&) const = default;
};

struct PadEffects {
  TriggerEffect left, right;
  // The light: `colour` when `light` is set, else the pad's own (the one
  // it had before any was set; a pad never given a colour is left alone).
  bool light = false;
  PadColour colour;

  bool operator==(const PadEffects&) const = default;
};

// The effect a pad can take: values out of range brought into it, and
// every effect that does nothing as kOff with nothing else set (so two
// effects that act the same compare equal).
inline TriggerEffect Normalized(TriggerEffect e) {
  const bool vibration = e.mode == TriggerEffect::kVibration;
  if (e.mode == TriggerEffect::kOff || e.mode > TriggerEffect::kVibration || e.strength == 0 ||
      (vibration && e.frequency == 0))
    return {};
  e.strength = std::min(e.strength, kTriggerMaxStrength);
  if (e.mode == TriggerEffect::kWeapon) {
    e.start = std::clamp(e.start, kWeaponStartMin, kWeaponStartMax);
    e.end = std::clamp(e.end, uint8_t(e.start + 1), kWeaponEndMax);
  } else {
    e.start = std::min<uint8_t>(e.start, kTriggerZones - 1);
    e.end = 0;
  }
  if (!vibration) e.frequency = 0;
  return e;
}

// A DualSense trigger effect block and its mode bytes.
constexpr int kDs5TriggerBlock = 11;
constexpr uint8_t kDs5ModeOff = 0x05, kDs5ModeFeedback = 0x21, kDs5ModeWeapon = 0x25,
                  kDs5ModeVibration = 0x26;
inline void EncodeDs5Trigger(TriggerEffect e, uint8_t out[kDs5TriggerBlock]) {
  e = Normalized(e);
  std::memset(out, 0, kDs5TriggerBlock);
  switch (e.mode) {
    case TriggerEffect::kFeedback:
    case TriggerEffect::kVibration: {
      uint32_t zones = 0, strengths = 0;
      for (int z = e.start; z < kTriggerZones; z++) {
        zones |= 1u << z;
        strengths |= uint32_t(e.strength - 1) << (3 * z);
      }
      out[0] = e.mode == TriggerEffect::kFeedback ? kDs5ModeFeedback : kDs5ModeVibration;
      out[1] = uint8_t(zones);
      out[2] = uint8_t(zones >> 8);
      out[3] = uint8_t(strengths);
      out[4] = uint8_t(strengths >> 8);
      out[5] = uint8_t(strengths >> 16);
      out[6] = uint8_t(strengths >> 24);
      if (e.mode == TriggerEffect::kVibration) out[9] = e.frequency;
      break;
    }
    case TriggerEffect::kWeapon: {
      const uint32_t zones = 1u << e.start | 1u << e.end;
      out[0] = kDs5ModeWeapon;
      out[1] = uint8_t(zones);
      out[2] = uint8_t(zones >> 8);
      out[3] = uint8_t(e.strength - 1);
      break;
    }
    default:
      out[0] = kDs5ModeOff;
      break;
  }
}

// The rumble motors as a DualSense has them now (0..255 each, the high
// bytes of the two motor speeds), and which of its two rumble modes its
// firmware takes.
struct PadMotors {
  uint8_t left = 0, right = 0;
  bool improved = false;
};

// Whether a DualSense takes the improved rumble mode, by SDL's rule
// (SDL_hidapi_ps5.c): Sony's own pads when they are a DualSense Edge, or
// their firmware is 2.24 or later, or not known (0).
constexpr uint16_t kSonyVendor = 0x054C, kDs5EdgeProduct = 0x0DF2, kDs5ImprovedRumbleFirmware = 0x0224;
constexpr bool Ds5ImprovedRumble(uint16_t vendor, uint16_t product, uint16_t firmware) {
  return vendor == kSonyVendor &&
         (product == kDs5EdgeProduct || firmware == 0 || firmware >= kDs5ImprovedRumbleFirmware);
}

// The DualSense's effects state (what SDL_SendGamepadEffect takes for it:
// the output report after its id): where its fields are and their enable
// bits.
constexpr int kDs5EffectsSize = 47;
constexpr int kDs5EnableBits1 = 0, kDs5RumbleRight = 2, kDs5RumbleLeft = 3, kDs5RightTrigger = 10,
              kDs5LeftTrigger = 21, kDs5EnableBits3 = 38;
constexpr uint8_t kDs5Bits1Rumble = 0x01,          // the rumble bytes are valid (older firmware's mode)
                  kDs5Bits1NoAudioHaptics = 0x02,  // rumble in place of audio haptics
                  kDs5Bits1RightTrigger = 0x04,    // the right trigger's block is valid
                  kDs5Bits1LeftTrigger = 0x08,     // the left trigger's
                  kDs5Bits3ImprovedRumble = 0x04;  // the rumble bytes are valid (improved mode)

// The effects state that sets both triggers. The pad reads every such
// state as the whole rumble state too: one without the rumble bits stops
// rumble that is running (it is what SDL sends to stop it). So the motors
// as they are now go into it, the way SDL's own states carry them
// (HIDAPI_DriverPS5_UpdateEffects): running, with the bits of the pad's
// rumble mode and halved in the older one; at rest, with no rumble bit.
// The light and the audio settings have their own enable bits, left off.
inline void EncodeDs5Triggers(const TriggerEffect& left, const TriggerEffect& right,
                              const PadMotors& motors, uint8_t out[kDs5EffectsSize]) {
  std::memset(out, 0, kDs5EffectsSize);
  out[kDs5EnableBits1] = kDs5Bits1RightTrigger | kDs5Bits1LeftTrigger;
  EncodeDs5Trigger(right, out + kDs5RightTrigger);
  EncodeDs5Trigger(left, out + kDs5LeftTrigger);
  if (motors.left || motors.right) {
    out[kDs5EnableBits1] |= kDs5Bits1NoAudioHaptics;
    if (motors.improved) {
      out[kDs5EnableBits3] |= kDs5Bits3ImprovedRumble;
      out[kDs5RumbleLeft] = motors.left;
      out[kDs5RumbleRight] = motors.right;
    } else {
      out[kDs5EnableBits1] |= kDs5Bits1Rumble;
      out[kDs5RumbleLeft] = motors.left >> 1;
      out[kDs5RumbleRight] = motors.right >> 1;
    }
  }
}

// The rumble a pad was last given, as SDL plays it: motor speeds 0..65535
// until changed, or for a time (at most SDL's 65535 ms), after which SDL
// stops it by itself.
struct PadRumble {
  uint16_t low = 0, high = 0;
  bool timed = false;
  uint64_t until = 0;  // ms

  void Set(uint16_t low_speed, uint16_t high_speed, uint32_t ms, uint64_t now) {
    low = low_speed;
    high = high_speed;
    timed = (low || high) && ms;
    until = now + std::min<uint32_t>(ms, 0xFFFF);
  }
  // The motors at `now`, in a DualSense's bytes.
  PadMotors At(uint64_t now, bool improved) const {
    if (timed && now >= until) return {0, 0, improved};
    return {uint8_t(low >> 8), uint8_t(high >> 8), improved};
  }
};

// The colour SDL gives a DualSense's light by its player index
// (SetLedsForPlayerIndex in SDL_hidapi_ps5.c). SDL has no call that goes
// back to it once a colour was set, so it is set from here.
constexpr PadColour Ds5PlayerColour(int player_index) {
  constexpr PadColour kColours[] = {{0x00, 0x00, 0x40}, {0x40, 0x00, 0x00}, {0x00, 0x40, 0x00},
                                    {0x20, 0x00, 0x20}, {0x20, 0x10, 0x00}, {0x00, 0x10, 0x10},
                                    {0x10, 0x10, 0x10}};
  return kColours[player_index >= 0 ? player_index % 7 : 0];
}

// The same effect in shares of the range, 0..1, as APIs that hide the
// zones take it (the frequency as a share of kUnitFrequencyHz).
constexpr float kUnitFrequencyHz = 255.0f;
struct UnitTrigger {
  float start, end, strength, frequency;
};
inline UnitTrigger ToUnit(TriggerEffect e) {
  e = Normalized(e);
  const float last_zone = kTriggerZones - 1;
  return {e.start / last_zone, e.end / last_zone, float(e.strength) / kTriggerMaxStrength,
          e.frequency / kUnitFrequencyHz};
}

// What a pad has taken so far: only what was sent and went through.
struct PadEffectsSent {
  bool tried = false;     // anything was ever sent, taken or not
  bool triggers = false;  // `left` and `right` are on the pad
  TriggerEffect left, right;
  enum Light : uint8_t { kUntouched, kOurs, kOwn } light = kUntouched;
  PadColour colour;  // kOurs, kOwn: the colour on the pad

  // Whether a trigger was left with an effect on.
  bool TriggersOn() const {
    return triggers && (left.mode != TriggerEffect::kOff || right.mode != TriggerEffect::kOff);
  }
};

// What to send a pad now.
struct PadSend {
  bool triggers = false;
  TriggerEffect left, right;
  bool light = false;
  bool own = false;  // the light: back to the pad's own colour
  PadColour colour;
};

// What a pad is sent for `want`: nothing unless it is a pad with such
// effects (`has_effects`); else the triggers when they are not the ones
// it has taken (a new pad: once, off too), and the light when a colour
// is wanted that it does not show, or its own colour (`own`) back after
// one of ours. A pad never given a colour gets none.
inline PadSend NextPadSend(bool has_effects, const PadEffectsSent& sent, const PadEffects& want,
                           PadColour own) {
  PadSend s;
  if (!has_effects) return s;
  s.left = Normalized(want.left);
  s.right = Normalized(want.right);
  s.triggers = !sent.triggers || !(s.left == sent.left) || !(s.right == sent.right);
  if (want.light) {
    s.colour = want.colour;
    s.light = sent.light != PadEffectsSent::kOurs || !(sent.colour == want.colour);
  } else if (sent.light == PadEffectsSent::kOurs) {
    s.colour = own;
    s.light = s.own = true;
  }
  return s;
}

// What ends them: free triggers for a pad that was ever sent any (taken
// or not: a refused "off" must not leave a trigger stiff), and its own
// colour back for a light that shows one of ours.
inline PadSend LastPadSend(const PadEffectsSent& sent, PadColour own) {
  PadSend s;
  s.triggers = sent.tried;
  if (sent.light == PadEffectsSent::kOurs) {
    s.colour = own;
    s.light = s.own = true;
  }
  return s;
}

// Notes a send's outcome: what went through is on the pad; what was
// refused is not, and is sent again the next time it is wanted.
inline void NotePadSend(PadEffectsSent& sent, const PadSend& s, bool triggers_ok, bool light_ok) {
  if (s.triggers || s.light) sent.tried = true;
  if (s.triggers && triggers_ok) {
    sent.triggers = true;
    sent.left = s.left;
    sent.right = s.right;
  }
  if (s.light && light_ok) {
    sent.light = s.own ? PadEffectsSent::kOwn : PadEffectsSent::kOurs;
    sent.colour = s.colour;
  }
}

}  // namespace recomp::hle
