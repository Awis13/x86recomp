#pragma once
#include <string>
#include "smaa_hlsl.h"
namespace recomp::gfx {
inline constexpr int kMetalSmaaSamplers[3] = {1, 3, 2};
inline std::string MetalSmaaGlsl(int pass, bool vertex) {
  std::string glsl = "#version 450\n#define SMAA_GLSL_4 1\n#define SMAA_PRESET_HIGH 1\n"
                         "#define SMAA_RT_METRICS smaa_metrics\n#define PASS " +
                         std::to_string(pass) + "\n";
      glsl += vertex ? "#define SMAA_INCLUDE_PS 0\n" : "#define SMAA_INCLUDE_VS 0\n";
      glsl += "layout(std140, set = 0, binding = 0) uniform SmaaMetrics { vec4 smaa_metrics; };\n";
      for (int i = 0; !vertex && i < kMetalSmaaSamplers[pass]; i++)
        glsl += "layout(set = 0, binding = " + std::to_string(i) + ") uniform sampler2D t" +
                std::to_string(i) + ";\n";
      glsl += kSmaaHlsl;
      glsl += vertex ? R"(
layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec2 v_pix;
layout(location = 2) out vec4 v_off[3];
void main() {
  // One triangle over the target; texture y runs down.
  vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
  v_uv = vec2(p.x, 1.0 - p.y);
  v_pix = vec2(0.0);
  v_off[0] = v_off[1] = v_off[2] = vec4(0.0);
#if PASS == 0
  SMAAEdgeDetectionVS(v_uv, v_off);
#elif PASS == 1
  SMAABlendingWeightCalculationVS(v_uv, v_pix, v_off);
#else
  SMAANeighborhoodBlendingVS(v_uv, v_off[0]);
#endif
}
)" : R"(
layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec2 v_pix;
layout(location = 2) in vec4 v_off[3];
layout(location = 0) out vec4 o_color;
void main() {
#if PASS == 0
  o_color = vec4(SMAALumaEdgeDetectionPS(v_uv, v_off, t0), 0.0, 0.0);
#elif PASS == 1
  o_color = SMAABlendingWeightCalculationPS(v_uv, v_pix, v_off, t0, t1, t2, vec4(0.0));
#else
  o_color = SMAANeighborhoodBlendingPS(v_uv, v_off[0], t0, t1);
#endif
}
)";
  return glsl;
}
}
