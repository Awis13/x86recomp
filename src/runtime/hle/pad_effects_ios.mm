// A DualSense's adaptive triggers through Apple's GameController framework,
// for iOS and tvOS: the pad is that framework's there (SDL reads it through
// it and has no effect packets for it). Every connected DualSense gets the
// effects: the framework's controllers are not matched to SDL's.

#import <GameController/GameController.h>

#include "runtime/hle/pad_effects_ios.h"

namespace recomp::hle {
namespace {

void Apply(GCDualSenseAdaptiveTrigger* trigger, const TriggerEffect& effect) {
  const TriggerEffect e = Normalized(effect);
  const UnitTrigger u = ToUnit(e);
  switch (e.mode) {
    case TriggerEffect::kFeedback:
      [trigger setModeFeedbackWithStartPosition:u.start resistiveStrength:u.strength];
      break;
    case TriggerEffect::kWeapon:
      [trigger setModeWeaponWithStartPosition:u.start endPosition:u.end resistiveStrength:u.strength];
      break;
    case TriggerEffect::kVibration:
      [trigger setModeVibrationWithStartPosition:u.start amplitude:u.strength frequency:u.frequency];
      break;
    default:
      [trigger setModeOff];
      break;
  }
}

}  // namespace

bool SetAdaptiveTriggers(const TriggerEffect& left, const TriggerEffect& right) {
  bool any = false;
  @autoreleasepool {
    for (GCController* controller in [GCController controllers]) {
      GCExtendedGamepad* pad = controller.extendedGamepad;
      if (![pad isKindOfClass:[GCDualSenseGamepad class]]) continue;
      GCDualSenseGamepad* dualsense = (GCDualSenseGamepad*)pad;
      Apply(dualsense.leftTrigger, left);
      Apply(dualsense.rightTrigger, right);
      any = true;
    }
  }
  return any;
}

}  // namespace recomp::hle
