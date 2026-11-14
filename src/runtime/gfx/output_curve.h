#pragma once
// A curve the finished frame goes through on its way to the window
// (RECOMP_OUTPUT_CURVE), for games whose art was tuned on displays fed
// another transfer function than the PC's sRGB:
//   bt709  each channel decoded as sRGB and encoded with the BT.709 OETF,
//          as video outputs that encode with BT.709 left an sRGB frame:
//          darker shadows and middle, more contrast and saturation;
//   <n>    each channel to the power n (above 1 darker, below 1 lighter).
// Anything else, 1 or nothing: no curve (and no change to the frame).
// Input in 0..1 and perceptual (the 8-bit back buffer); the alpha is kept.

#include <cstdlib>
#include <cstring>

namespace recomp::gfx {

struct OutputCurve {
  enum Kind { kNone, kBt709, kPower } kind = kNone;
  float exponent = 1.0f;  // kPower's

  bool Active() const { return kind != kNone; }
  // output_curve's mode: 0 none, 1 bt709, 2 power.
  float ShaderMode() const { return float(kind); }
};

inline OutputCurve ParseOutputCurve(const char* s) {
  OutputCurve c;
  if (!s || !*s) return c;
  if (!std::strcmp(s, "bt709")) {
    c.kind = OutputCurve::kBt709;
    return c;
  }
  char* end = nullptr;
  const float n = std::strtof(s, &end);
  if (end != s && !*end && n > 0.0f && n != 1.0f) {
    c.kind = OutputCurve::kPower;
    c.exponent = n;
  }
  return c;
}

// output_curve(colour, mode, exponent): ShaderMode's mode, OutputCurve's
// exponent.
inline constexpr const char kOutputCurveMsl[] = R"(
#include <metal_stdlib>
using namespace metal;
inline float3 output_curve(float3 c, float mode, float exponent) {
  c = saturate(c);
  if (mode == 1.0) {
    const float3 lin = select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c < 0.04045);
    return select(1.099 * pow(lin, 0.45) - 0.099, lin * 4.5, lin < 0.018);
  }
  if (mode == 2.0) return pow(c, exponent);
  return c;
}
)";

}  // namespace recomp::gfx
