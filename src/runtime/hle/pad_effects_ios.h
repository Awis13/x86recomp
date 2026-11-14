#pragma once
// A DualSense's adaptive triggers on iOS and tvOS (pad_effects_ios.mm),
// where the pad belongs to Apple's GameController framework.

#include "runtime/hle/pad_effects.h"

namespace recomp::hle {

// Sets both triggers of every connected DualSense; false without one.
bool SetAdaptiveTriggers(const TriggerEffect& left, const TriggerEffect& right);

}  // namespace recomp::hle
