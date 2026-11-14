// A pad's trigger effects (runtime/hle/pad_effects.h): the ranges they are
// brought into, the blocks a DualSense takes for them (written out here by
// hand from the layout), the effects state that sets the triggers with the
// rumble as it is, the pad's own light colour, and the decision what a pad
// is sent next: nothing for a pad without such effects, nothing twice, a
// refused send again.

#include <cmath>
#include <cstdio>
#include <cstring>

#include "runtime/hle/pad_effects.h"

namespace {

using namespace recomp::hle;

int g_failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    g_failures++;
  }
}

bool Near(float a, float b) { return std::fabs(a - b) < 1e-6f; }

TriggerEffect Effect(TriggerEffect::Mode mode, int start, int end, int strength, int frequency) {
  TriggerEffect e;
  e.mode = mode;
  e.start = uint8_t(start);
  e.end = uint8_t(end);
  e.strength = uint8_t(strength);
  e.frequency = uint8_t(frequency);
  return e;
}

void ExpectBlock(const TriggerEffect& e, const uint8_t want[kDs5TriggerBlock], const char* what) {
  uint8_t got[kDs5TriggerBlock];
  std::memset(got, 0xEE, sizeof got);
  EncodeDs5Trigger(e, got);
  if (std::memcmp(got, want, sizeof got) == 0) return;
  std::printf("FAIL: %s:", what);
  for (uint8_t b : got) std::printf(" %02X", b);
  std::printf("\n");
  g_failures++;
}

void TestNormalized() {
  const TriggerEffect off;
  Check(Normalized(Effect(TriggerEffect::kFeedback, 3, 0, 0, 0)) == off, "no strength is off");
  Check(Normalized(Effect(TriggerEffect::kVibration, 3, 0, 5, 0)) == off, "a vibration at 0 Hz is off");
  Check(Normalized(Effect(TriggerEffect::kOff, 3, 4, 5, 6)) == off, "off keeps nothing");
  Check(Normalized(Effect(TriggerEffect::Mode(9), 3, 4, 5, 6)) == off, "an unknown mode is off");
  Check(Normalized(Effect(TriggerEffect::kFeedback, 15, 7, 12, 40)) ==
            Effect(TriggerEffect::kFeedback, 9, 0, 8, 0),
        "feedback: start up to 9, strength up to 8, nothing else kept");
  Check(Normalized(Effect(TriggerEffect::kVibration, 4, 7, 3, 40)) ==
            Effect(TriggerEffect::kVibration, 4, 0, 3, 40),
        "vibration keeps its frequency");
  Check(Normalized(Effect(TriggerEffect::kWeapon, 0, 0, 4, 9)) ==
            Effect(TriggerEffect::kWeapon, 2, 3, 4, 0),
        "weapon: start from 2, the end after it");
  Check(Normalized(Effect(TriggerEffect::kWeapon, 9, 9, 4, 0)) ==
            Effect(TriggerEffect::kWeapon, 7, 8, 4, 0),
        "weapon: start up to 7, end up to 8");
  Check(Normalized(Effect(TriggerEffect::kWeapon, 3, 6, 5, 0)) ==
            Effect(TriggerEffect::kWeapon, 3, 6, 5, 0),
        "weapon in range stays");
}

void TestBlocks() {
  const uint8_t off[] = {0x05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  ExpectBlock({}, off, "off");
  ExpectBlock(Effect(TriggerEffect::kFeedback, 2, 0, 0, 0), off, "feedback without strength");
  // Zones 2..9 (0x3FC), strength 5 as 4 in each zone's three bits.
  const uint8_t feedback[] = {0x21, 0xFC, 0x03, 0x00, 0x49, 0x92, 0x24, 0, 0, 0, 0};
  ExpectBlock(Effect(TriggerEffect::kFeedback, 2, 0, 5, 0), feedback, "feedback from 2 at 5");
  const uint8_t full[] = {0x21, 0xFF, 0x03, 0xFF, 0xFF, 0xFF, 0x3F, 0, 0, 0, 0};
  ExpectBlock(Effect(TriggerEffect::kFeedback, 0, 0, 8, 0), full, "feedback from 0 at 8");
  // The start and end zones' bits, strength - 1.
  const uint8_t weapon[] = {0x25, 0x24, 0x00, 0x03, 0, 0, 0, 0, 0, 0, 0};
  ExpectBlock(Effect(TriggerEffect::kWeapon, 2, 5, 4, 0), weapon, "weapon 2 to 5 at 4");
  const uint8_t heavy[] = {0x25, 0x08, 0x01, 0x07, 0, 0, 0, 0, 0, 0, 0};
  ExpectBlock(Effect(TriggerEffect::kWeapon, 3, 8, 8, 0), heavy, "weapon 3 to 8 at 8");
  // Every zone at amplitude 6 (5 in three bits, ten times), 25 Hz.
  const uint8_t vibration[] = {0x26, 0xFF, 0x03, 0x6D, 0xDB, 0xB6, 0x2D, 0, 0, 25, 0};
  ExpectBlock(Effect(TriggerEffect::kVibration, 0, 0, 6, 25), vibration, "vibration from 0 at 6, 25 Hz");
  const uint8_t late[] = {0x26, 0x00, 0x03, 0x00, 0x00, 0x00, 0x09, 0, 0, 30, 0};
  ExpectBlock(Effect(TriggerEffect::kVibration, 8, 0, 2, 30), late, "vibration from 8 at 2, 30 Hz");
}

// The effects state. The offsets and bits here are literals on purpose
// (0, 2, 3, 10, 21, 38 from SDL's DS5EffectsState_t), an independent
// check of the header's constants.
void TestState() {
  const TriggerEffect left = Effect(TriggerEffect::kFeedback, 2, 0, 5, 0);
  const TriggerEffect right = Effect(TriggerEffect::kWeapon, 2, 5, 4, 0);
  uint8_t got[kDs5EffectsSize + 1];

  // Motors at rest: both triggers and no rumble bit.
  std::memset(got, 0xEE, sizeof got);
  EncodeDs5Triggers(left, right, PadMotors{0, 0, true}, got);
  uint8_t want[kDs5EffectsSize] = {};
  want[0] = 0x0C;
  EncodeDs5Trigger(right, want + 10);
  EncodeDs5Trigger(left, want + 21);
  Check(std::memcmp(got, want, kDs5EffectsSize) == 0, "state at rest: the two triggers and nothing else");
  Check(got[kDs5EffectsSize] == 0xEE, "state: 47 bytes and no more");
  Check(got[10] == 0x25 && got[21] == 0x21, "state: right trigger at 10, left at 21");
  Check(got[1] == 0, "state: no light bit");
  EncodeDs5Triggers(left, right, PadMotors{0, 0, false}, got);
  Check(std::memcmp(got, want, kDs5EffectsSize) == 0, "state at rest, older rumble mode: the same");

  // Motors running, improved mode: the motors as they are, its bit, and
  // rumble in place of audio haptics; the triggers as before.
  EncodeDs5Triggers(left, right, PadMotors{200, 90, true}, got);
  want[0] = 0x0C | 0x02;
  want[38] = 0x04;
  want[3] = 200;
  want[2] = 90;
  Check(std::memcmp(got, want, kDs5EffectsSize) == 0, "state with rumble running, improved mode");

  // The older mode: its bit in the first byte, the motors halved.
  EncodeDs5Triggers(left, right, PadMotors{200, 91, false}, got);
  want[0] = 0x0C | 0x02 | 0x01;
  want[38] = 0;
  want[3] = 100;
  want[2] = 45;
  Check(std::memcmp(got, want, kDs5EffectsSize) == 0, "state with rumble running, older mode");

  // One motor is rumble too.
  EncodeDs5Triggers({}, {}, PadMotors{0, 1, true}, got);
  Check(got[0] == 0x0E && got[38] == 0x04 && got[2] == 1 && got[3] == 0, "state with one motor running");
}

void TestRumbleMode() {
  Check(Ds5ImprovedRumble(0x054C, 0x0CE6, 0x0224), "firmware 2.24: improved");
  Check(Ds5ImprovedRumble(0x054C, 0x0CE6, 0x0300), "later firmware: improved");
  Check(!Ds5ImprovedRumble(0x054C, 0x0CE6, 0x0223), "firmware before 2.24: the older mode");
  Check(Ds5ImprovedRumble(0x054C, 0x0CE6, 0), "firmware not known: improved");
  Check(Ds5ImprovedRumble(0x054C, 0x0DF2, 0x0100), "the Edge: improved whatever the firmware");
  Check(!Ds5ImprovedRumble(0x0F0D, 0x0CE6, 0x0300), "another maker's pad: the older mode");
}

// The rumble as SDL plays it: until changed, or for a time.
void TestRumble() {
  PadRumble r;
  Check(r.At(5, true).left == 0 && r.At(5, true).right == 0, "no rumble yet");
  r.Set(0xC8FF, 0x5A00, 0xFFFFFFFF, 1000);
  PadMotors m = r.At(1001, true);
  Check(m.left == 0xC8 && m.right == 0x5A && m.improved, "motors: the speeds' high bytes");
  Check(!r.At(1001, false).improved, "motors: the mode asked for");
  Check(r.At(1000 + 0xFFFE, true).left == 0xC8, "a long rumble runs for SDL's longest time");
  Check(r.At(1000 + 0xFFFF, true).left == 0, "and is over after it");
  r.Set(0x8000, 0, 300, 2000);
  Check(r.At(2299, true).left == 0x80, "a timed rumble runs for its time");
  Check(r.At(2300, true).left == 0, "and is over at its end");
  r.Set(0x8000, 0x8000, 0, 3000);
  Check(r.At(999999, true).right == 0x80, "a rumble with no time runs until changed");
  r.Set(0, 0, 500, 4000);
  Check(r.At(4001, true).left == 0 && r.At(4001, true).right == 0, "stopped");
}

void TestOwnColour() {
  Check(Ds5PlayerColour(0) == PadColour{0x00, 0x00, 0x40}, "player 1: blue");
  Check(Ds5PlayerColour(1) == PadColour{0x40, 0x00, 0x00}, "player 2: red");
  Check(Ds5PlayerColour(6) == PadColour{0x10, 0x10, 0x10}, "player 7: white");
  Check(Ds5PlayerColour(7) == Ds5PlayerColour(0), "player 8: from the start again");
  Check(Ds5PlayerColour(-1) == Ds5PlayerColour(0), "no player index: the first colour");
}

void TestUnit() {
  const UnitTrigger w = ToUnit(Effect(TriggerEffect::kWeapon, 2, 5, 4, 0));
  Check(Near(w.start, 0.2222222f) && Near(w.end, 0.5555556f) && Near(w.strength, 0.5f),
        "weapon in shares: zones of 9, strength of 8");
  const UnitTrigger f = ToUnit(Effect(TriggerEffect::kFeedback, 9, 0, 8, 0));
  Check(Near(f.start, 1.0f) && Near(f.strength, 1.0f), "the last zone and full strength are 1");
  const UnitTrigger v = ToUnit(Effect(TriggerEffect::kVibration, 0, 0, 6, 51));
  Check(Near(v.start, 0.0f) && Near(v.strength, 0.75f) && Near(v.frequency, 0.2f),
        "vibration in shares: 51 Hz of 255");
}

const PadColour kOwn{0x00, 0x00, 0x40}, kRed{255, 0, 0}, kBlue{0, 0, 255};

// One call of the device layer with a pad that takes everything.
PadSend Step(PadEffectsSent& sent, const PadEffects& want, bool triggers_ok = true, bool light_ok = true,
             bool has_effects = true) {
  const PadSend s = NextPadSend(has_effects, sent, want, kOwn);
  NotePadSend(sent, s, triggers_ok, light_ok);
  return s;
}

void TestNotADualSense() {
  PadEffectsSent sent;
  PadEffects e;
  e.right = Effect(TriggerEffect::kWeapon, 2, 5, 4, 0);
  e.light = true;
  e.colour = kRed;
  const PadSend s = Step(sent, e, true, true, false);
  Check(!s.triggers && !s.light, "another pad: nothing is sent");
  Check(!sent.tried && !sent.triggers && sent.light == PadEffectsSent::kUntouched,
        "another pad: nothing is noted");
  const PadSend last = LastPadSend(sent, kOwn);
  Check(!last.triggers && !last.light, "another pad: nothing to end");
}

void TestTriggersSentOnce() {
  PadEffectsSent sent;
  PadEffects e;
  Check(!sent.TriggersOn(), "nothing on before anything is sent");
  PadSend s = Step(sent, e);
  Check(s.triggers && s.left == TriggerEffect{} && s.right == TriggerEffect{},
        "a new pad is sent its triggers once, off too");
  Check(!s.light, "a new pad's light is left alone");
  Check(!Step(sent, e).triggers, "the same triggers again are not sent");
  Check(!sent.TriggersOn(), "off triggers are not on");

  e.right = Effect(TriggerEffect::kWeapon, 2, 5, 4, 0);
  s = Step(sent, e);
  Check(s.triggers && s.right == e.right, "another right trigger is sent");
  Check(sent.TriggersOn() && sent.right == e.right, "and noted as on the pad");
  e.right = Effect(TriggerEffect::kWeapon, 2, 5, 4, 77);  // acts the same
  Check(!Step(sent, e).triggers, "an effect that acts the same is not sent");
  e.left = Effect(TriggerEffect::kFeedback, 1, 0, 2, 0);
  Check(Step(sent, e).triggers, "another left trigger is sent");
  e.left = e.right = {};
  Check(Step(sent, e).triggers, "off is sent after an effect");
  Check(!sent.TriggersOn(), "and nothing is on then");
}

// A send the pad refuses is not on the pad: it goes again.
void TestRefusedSendGoesAgain() {
  PadEffectsSent sent;
  PadEffects e;
  e.right = Effect(TriggerEffect::kFeedback, 1, 0, 4, 0);
  Step(sent, e);
  e.right = {};
  PadSend s = Step(sent, e, false);
  Check(s.triggers, "off is sent");
  Check(sent.TriggersOn(), "refused: the effect is still noted as on the pad");
  s = Step(sent, e, false);
  Check(s.triggers, "refused off is sent again at the next call");
  s = Step(sent, e, true);
  Check(s.triggers && !sent.TriggersOn(), "and noted once it goes through");
  Check(!Step(sent, e).triggers, "then not again");

  // The very first send refused: tried, nothing on the pad, and the end
  // still frees the triggers.
  PadEffectsSent fresh;
  e.right = Effect(TriggerEffect::kFeedback, 1, 0, 4, 0);
  Step(fresh, e, false);
  Check(fresh.tried && !fresh.triggers, "a refused first send is only noted as tried");
  Check(LastPadSend(fresh, kOwn).triggers, "the end frees the triggers of a pad that refused");

  // The light the same.
  e.light = true;
  e.colour = kRed;
  s = Step(sent, e, true, false);
  Check(s.light && sent.light == PadEffectsSent::kUntouched, "a refused colour is not noted");
  Check(Step(sent, e, true, false).light, "and is sent again");
  Step(sent, e);
  Check(sent.light == PadEffectsSent::kOurs && !Step(sent, e).light, "until it goes through");
}

void TestLight() {
  PadEffectsSent sent;
  PadEffects e;
  Check(!Step(sent, e).light, "no colour wanted, none ever set: the light is left alone");
  e.light = true;
  e.colour = kRed;
  PadSend s = Step(sent, e);
  Check(s.light && !s.own && s.colour == kRed, "a wanted colour is sent");
  Check(!s.triggers, "a colour alone sends no triggers");
  Check(!Step(sent, e).light, "the same colour is not sent again");
  e.colour = kBlue;
  Check(Step(sent, e).colour == kBlue, "another colour is sent");
  e.colour = PadColour{0, 1, 255};
  Check(Step(sent, e).light, "a colour that differs in green only is another colour");
  e.light = false;
  s = Step(sent, e);
  Check(s.light && s.own && s.colour == kOwn, "no colour wanted any more: the pad's own colour back");
  Check(sent.light == PadEffectsSent::kOwn, "noted as its own");
  Check(!Step(sent, e).light, "its own colour is sent once");
  e.light = true;
  e.colour = kOwn;
  Check(Step(sent, e).light, "a wanted colour equal to its own is still sent as ours");
}

void TestEnd() {
  PadEffectsSent sent;
  PadSend last = LastPadSend(sent, kOwn);
  Check(!last.triggers && !last.light, "a pad never sent anything: nothing to end");

  PadEffects e;
  e.right = Effect(TriggerEffect::kWeapon, 2, 5, 4, 0);
  e.light = true;
  e.colour = kRed;
  Step(sent, e);
  last = LastPadSend(sent, kOwn);
  Check(last.triggers && last.left == TriggerEffect{} && last.right == TriggerEffect{},
        "the end frees the triggers");
  Check(last.light && last.own && last.colour == kOwn, "the end gives the light its own colour back");
  NotePadSend(sent, last, true, true);
  Check(!sent.TriggersOn() && sent.light == PadEffectsSent::kOwn, "and that is noted");

  // The triggers already free are freed all the same (cheap, and sure).
  last = LastPadSend(sent, kOwn);
  Check(last.triggers && !last.light, "a second end: the triggers again, the light not");
}

}  // namespace

int main() {
  TestNormalized();
  TestBlocks();
  TestState();
  TestRumbleMode();
  TestRumble();
  TestOwnColour();
  TestUnit();
  TestNotADualSense();
  TestTriggersSentOnce();
  TestRefusedSendGoesAgain();
  TestLight();
  TestEnd();
  if (g_failures) {
    std::printf("%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("pad_effects_test: ok\n");
  return 0;
}
