// VertexWithCapturedPosition (capture_position.h): a text transform of the
// GLSL TranslateBytecode writes, which ends with
//   <helpers, subroutines> ... void main() {
//   <inputs, definitions> <body> <outputs: gl_Position = r_position(X); ...>
//   }
// The capture's declarations go before main, and a line writing X before
// each line writing the position; nothing else changes (subroutines
// included: they run as before).

#include "runtime/hlsl/capture_position.h"

#include <string>
#include <string_view>

namespace recomp::hlsl {

std::string VertexWithCapturedPosition(const Program& vs) {
  if (vs.stage != Stage::Vertex) return {};
  const std::string& g = vs.glsl;
  static constexpr std::string_view kMain = "\nvoid main() {\n";
  const size_t m = g.find(kMain);
  if (m == std::string::npos || g.find(kMain, m + 1) != std::string::npos) return {};
  // Names the capture adds, not to be taken from the shader.
  if (g.find("r_cap") != std::string::npos) return {};

  static constexpr std::string_view kPos = "  gl_Position = r_position(";
  std::string out = g.substr(0, m + 1);
  out += "layout(std430, R_CAPTURE_BINDING) writeonly buffer r_capture { vec4 r_cap[]; };\n"
         "layout(std140, R_CAPTURE_RANGE_BINDING) uniform r_capture_range { ivec4 r_cap_range; };\n";
  size_t at = m + 1, lines = 0;
  for (size_t p = g.find(kPos, at); p != std::string::npos; p = g.find(kPos, at)) {
    // On a line of its own, in main.
    if (g[p - 1] != '\n') return {};
    const size_t nl = g.find('\n', p);
    if (nl == std::string::npos || g.compare(nl - 2, 2, ");") != 0) return {};
    const std::string x = g.substr(p + kPos.size(), nl - 2 - p - kPos.size());
    out.append(g, at, p - at);
    out += "  if (uint(gl_VertexIndex - r_cap_range.x) < uint(r_cap_range.y)) r_cap[gl_VertexIndex - r_cap_range.x] = " +
           x + ";\n";
    at = p;
    out.append(g, at, nl + 1 - at);
    at = nl + 1;
    lines++;
  }
  // Every r_position call one of those lines (none elsewhere, none
  // missing).
  size_t calls = 0;
  for (size_t p = g.find("r_position("); p != std::string::npos; p = g.find("r_position(", p + 1)) calls++;
  if (!lines || calls != lines) return {};
  out.append(g, at, std::string::npos);
  return out;
}

}  // namespace recomp::hlsl
