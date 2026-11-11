#pragma once
// Sharpening after an upscale (or after TAA): AMD's RCAS, "robust
// contrast adaptive sharpening", the pass FidelityFX FSR 1 runs after its
// upscale and FSR 2 and 3 after theirs. A cross of the four neighbours
// pulled away from the centre, by the largest amount that does not push
// any channel out of 0..1 (solved from the ring's minimum and maximum,
// one weight for all three channels, so no colour fringes), capped at
// 0.25 - 1/16, and less where the centre looks like noise (a high-pass
// normalised by the local contrast: grain and shimmer, not edges).
// Input in 0..1 and perceptual (after tonemapping, as 8-bit back buffers
// hold it); HDR or linear values would sharpen wrongly.
//
// Ported from FsrRcasF in ffx_fsr1.h (FSR 1, v1.20210629), with
// FSR_RCAS_DENOISE, in 32-bit float. Changes: the inputs are clamped to
// 0..1 (negative input gives NaN), and the three divisions by zero AMD's
// code meets are answered as it ends up answering them:
//   - a channel whose ring is all 0 (black): its limit toward 0 is none
//     (AMD's exact reciprocal gives 0 * inf = NaN, which the max drops);
//   - a channel whose ring is all 1 (white): its limit toward 1, the same;
//   - a flat neighbourhood (no contrast): no noise (AMD's approximate
//     reciprocal stays finite, times a difference of 0).
// Taken as limits of 0 instead, a channel at 0 or 1 all around (no blue
// in a dark street, a clipped sky) would stop all sharpening there.
//
//   Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
//   Permission is hereby granted, free of charge, to any person obtaining
//   a copy of this software and associated documentation files (the
//   "Software"), to deal in the Software without restriction, including
//   without limitation the rights to use, copy, modify, merge, publish,
//   distribute, sublicense, and/or sell copies of the Software, and to
//   permit persons to whom the Software is furnished to do so, subject to
//   the following conditions: The above copyright notice and this
//   permission notice shall be included in all copies or substantial
//   portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT
//   WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
//   THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
//   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
//   LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
//   OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
//   WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>
#include <cmath>

namespace recomp::gfx {

// Whether an amount sharpens at all: 0 is no pass (and no intermediate
// texture for it).
inline bool RcasActive(float amount) { return amount > 0.0f; }

// The amount (RECOMP_SHARPNESS, 0..1) as RCAS's constant: FSR 2's mapping,
// 0..1 to 2..0 stops of reduction from the strongest (each stop halves
// it), so 0.5 is one stop, 0.75 half a stop, 1 the strongest. 0 is no
// sharpening: the pass is not run at all.
inline float RcasConstant(float amount) {
  amount = std::clamp(amount, 0.0f, 1.0f);
  return std::exp2(-(2.0f - 2.0f * amount));
}

// rcas_cs: src (any 0..1 colour format) to out, the same size; buffer 0
// is RcasConstant's value. One thread a pixel.
inline constexpr const char kRcasMsl[] = R"(
#include <metal_stdlib>
using namespace metal;
kernel void rcas_cs(texture2d<half, access::read> src [[texture(0)]],
                    texture2d<half, access::write> out [[texture(1)]],
                    constant float& con [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  const uint2 size = uint2(src.get_width(), src.get_height());
  if (gid.x >= size.x || gid.y >= size.y) return;
  const int2 last = int2(size) - 1, g = int2(gid);
  //    b
  //  d e f
  //    h
  const float3 b = saturate(float3(src.read(uint2(clamp(g + int2(0, -1), 0, last))).rgb));
  const float3 d = saturate(float3(src.read(uint2(clamp(g + int2(-1, 0), 0, last))).rgb));
  const float3 e = saturate(float3(src.read(gid).rgb));
  const float3 f = saturate(float3(src.read(uint2(clamp(g + int2(1, 0), 0, last))).rgb));
  const float3 h = saturate(float3(src.read(uint2(clamp(g + int2(0, 1), 0, last))).rgb));
  // Luma times 2, and the noise detection.
  const float bL = b.b * 0.5 + (b.r * 0.5 + b.g), dL = d.b * 0.5 + (d.r * 0.5 + d.g);
  const float eL = e.b * 0.5 + (e.r * 0.5 + e.g), fL = f.b * 0.5 + (f.r * 0.5 + f.g);
  const float hL = h.b * 0.5 + (h.r * 0.5 + h.g);
  const float range = max(max(max(bL, dL), max(eL, fL)), hL) - min(min(min(bL, dL), min(eL, fL)), hL);
  float nz = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;
  nz = saturate(abs(nz) / max(range, 1.0e-5));
  nz = -0.5 * nz + 1.0;
  // The ring's minimum and maximum, and the weight that keeps each
  // channel inside 0..1.
  const float3 mn4 = min(min(b, d), min(f, h)), mx4 = max(max(b, d), max(f, h));
  // A ring all 0 (or all 1) sets no limit on its side: a value the max
  // below never picks.
  const float3 hit_min = select(min(mn4, e) / max(4.0 * mx4, 1.0e-30), float3(1.0), mx4 <= 0.0);
  const float3 hit_max = select((1.0 - max(mx4, e)) / min(4.0 * mn4 - 4.0, -1.0e-30), float3(-1.0), mn4 >= 1.0);
  const float3 lobe3 = max(-hit_min, hit_max);
  float lobe = max(-(0.25 - 1.0 / 16.0), min(max(max(lobe3.r, lobe3.g), lobe3.b), 0.0)) * con;
  lobe *= nz;
  const float3 o = (lobe * (b + d + h + f) + e) / (4.0 * lobe + 1.0);
  out.write(half4(half3(saturate(o)), 1.0h), gid);
}
)";

}  // namespace recomp::gfx
