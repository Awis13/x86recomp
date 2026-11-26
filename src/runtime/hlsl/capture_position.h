#pragma once
// A vertex shader's own output positions, written to memory: for geometry
// whose vertices only the vertex shader places (skinning, displacement by a
// vertex texture, instances placed from per-instance data), so a backend can
// read where they ended up (a ray-traced scene's acceleration structures).
//
// The variant's GLSL adds to the hlsl.h contract:
//   layout(std430, R_CAPTURE_BINDING) writeonly buffer r_capture { vec4 r_cap[]; };
//   layout(std140, R_CAPTURE_RANGE_BINDING) uniform r_capture_range { ivec4 r_cap_range; };
// (bindings the backend defines) and, before each line writing
// gl_Position = r_position(X), writes X, the POSITION output as the shader
// computed it (before r_position's viewport, jitter, depth and clip plane
// adjustments), to r_cap[gl_VertexIndex - r_cap_range.x] when that index is
// below r_cap_range.y (the vertices captured). The rest of the shader is
// unchanged; a backend runs it with rasterization off, once per vertex.

#include <string>

#include "runtime/hlsl/hlsl.h"

namespace recomp::hlsl {

// The capture variant of `vs` (a vertex shader from TranslateBytecode), or
// an empty string if it has none: GLSL not in the translator's shape (one
// main, every r_position call on a line of its own writing gl_Position).
std::string VertexWithCapturedPosition(const Program& vs);

}  // namespace recomp::hlsl
