// hlsl::VertexWithCapturedPosition on hand-assembled vertex shaders (vs_3_0,
// vs_2_0 with relative addressing, vs_2_0 calling a subroutine): exactly
// one write of the position before each line writing gl_Position, the
// translation otherwise unchanged (removing what the variant adds gives it
// back), and the variant compiles with glslangValidator (argv[1]) against
// tests/hlsl/prelude.glsl plus the capture's bindings. Pixel shaders and
// GLSL not in the translator's shape get no variant.
//
//   hlsl_capture_position_test <glslangValidator> <prelude.glsl> <out dir>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

#include "runtime/hlsl/capture_position.h"
#include "runtime/hlsl/hlsl.h"

namespace {

using Tokens = std::vector<uint32_t>;

enum Reg : uint32_t { R = 0, V = 1, C = 2, A = 3, RAST = 4, O = 6, OC = 8, LBL = 18 };
enum Op : uint32_t { MOV = 1, ADD = 2, DP4 = 9, CALL = 25, RET = 28, LABEL = 30, DCL = 31, MOVA = 46 };

uint32_t TypeBits(uint32_t type) { return ((type & 7) << 28) | ((type & 0x18) << 8); }
uint32_t Dst(uint32_t type, uint32_t n, uint32_t mask = 0xF) { return 0x80000000u | n | TypeBits(type) | mask << 16; }
uint32_t Src(uint32_t type, uint32_t n, uint32_t swz = 0xE4, bool rel = false) {
  return 0x80000000u | n | TypeBits(type) | swz << 16 | (rel ? 0x2000u : 0);
}

struct Asm {
  Tokens t;
  Asm(bool pixel, int major, int minor) {
    t.push_back((pixel ? 0xFFFF0000u : 0xFFFE0000u) | uint32_t(major) << 8 | uint32_t(minor));
  }
  void I(uint32_t op, std::initializer_list<uint32_t> params) {
    t.push_back(op | uint32_t(params.size()) << 24);
    t.insert(t.end(), params);
  }
  void Dcl(uint32_t usage, uint32_t index, uint32_t dst) { I(DCL, {0x80000000u | usage | index << 16, dst}); }
  Tokens End() {
    t.push_back(0x0000FFFFu);
    return t;
  }
};

// vs_3_0: o0 = v0 * c0..c3 (rows), o1 = v1 (TEXCOORD0).
Tokens Vs3() {
  Asm a(false, 3, 0);
  a.Dcl(0, 0, Dst(V, 0));
  a.Dcl(5, 0, Dst(V, 1));
  a.Dcl(0, 0, Dst(O, 0));
  a.Dcl(5, 0, Dst(O, 1));
  for (uint32_t k = 0; k < 4; k++) a.I(DP4, {Dst(O, 0, 1u << k), Src(V, 0), Src(C, k)});
  a.I(MOV, {Dst(O, 1), Src(V, 1)});
  return a.End();
}

// vs_2_0 with a relative read: a0.x = r0.x; oPos = v0 + c[a0.x + 4].
Tokens Vs2Relative() {
  Asm a(false, 2, 0);
  a.Dcl(0, 0, Dst(V, 0));
  a.I(MOV, {Dst(R, 0), Src(V, 0)});
  a.I(MOVA, {Dst(A, 0, 1), Src(R, 0, 0x00)});
  a.I(ADD, {Dst(RAST, 0), Src(V, 0), Src(C, 4, 0xE4, true), 0x80000000u | TypeBits(A)});
  return a.End();
}

// vs_2_0 calling a subroutine (that reads c1).
Tokens Vs2Call() {
  Asm a(false, 2, 0);
  a.Dcl(0, 0, Dst(V, 0));
  a.I(CALL, {Src(LBL, 0)});
  a.I(ADD, {Dst(RAST, 0), Src(V, 0), Src(R, 0)});
  a.I(RET, {});
  a.I(LABEL, {Src(LBL, 0)});
  a.I(MOV, {Dst(R, 0), Src(C, 1)});
  a.I(RET, {});
  return a.End();
}

// ps_3_0: c0 as the colour.
Tokens Ps3() {
  Asm a(true, 3, 0);
  a.I(MOV, {Dst(OC, 0), Src(C, 0)});
  return a.End();
}

size_t Count(const std::string& s, const std::string& what) {
  size_t n = 0;
  for (size_t p = s.find(what); p != std::string::npos; p = s.find(what, p + 1)) n++;
  return n;
}

const char* const kDecls =
    "layout(std430, R_CAPTURE_BINDING) writeonly buffer r_capture { vec4 r_cap[]; };\n"
    "layout(std140, R_CAPTURE_RANGE_BINDING) uniform r_capture_range { ivec4 r_cap_range; };\n";
const char* const kWrite =
    "  if (uint(gl_VertexIndex - r_cap_range.x) < uint(r_cap_range.y)) r_cap[gl_VertexIndex - r_cap_range.x] = ";

// `v` without the declarations and the lines the capture adds, each of
// which must be right before a line writing gl_Position with the value it
// writes; false if one is not.
bool Strip(std::string v, std::string* out) {
  const size_t d = v.find(kDecls);
  if (d == std::string::npos) return false;
  v.erase(d, std::string(kDecls).size());
  const std::string write = kWrite;
  for (size_t p = v.find(write); p != std::string::npos; p = v.find(write)) {
    const size_t nl = v.find('\n', p);
    const std::string x = v.substr(p + write.size(), nl - 1 - p - write.size());  // (before ';')
    const std::string line = "  gl_Position = r_position(" + x + ");\n";
    if (v.compare(nl + 1, line.size(), line) != 0) return false;
    v.erase(p, nl + 1 - p);
  }
  *out = v;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: hlsl_capture_position_test <glslangValidator> <prelude.glsl> <out dir>\n");
    return 2;
  }
  std::stringstream prelude;
  prelude << std::ifstream(argv[2]).rdbuf();
  int failures = 0;
  auto check = [&](bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) failures++;
  };
  struct Case {
    const char* name;
    Tokens tokens;
  } cases[] = {{"capture_vs3", Vs3()}, {"capture_vs2_rel", Vs2Relative()}, {"capture_vs2_call", Vs2Call()}};
  for (Case& c : cases) {
    recomp::hlsl::Program p;
    std::string err;
    if (!recomp::hlsl::TranslateBytecode(c.tokens.data(), c.tokens.size(), &p, &err)) {
      check(false, std::string(c.name) + ": translation: " + err);
      continue;
    }
    const std::string v = recomp::hlsl::VertexWithCapturedPosition(p);
    if (v.empty()) {
      check(false, std::string(c.name) + ": no variant");
      continue;
    }
    std::string back;
    const size_t positions = Count(p.glsl, "  gl_Position = r_position(");
    check(positions >= 1 && Count(v, kWrite) == positions && Strip(v, &back) && back == p.glsl,
          std::string(c.name) + ": one write before each position line, the rest unchanged");
    const std::string path = std::string(argv[3]) + "/" + c.name + ".vert";
    std::ofstream(path) << "#version 450\n#define R_VERTEX_SHADER\n"
                        << "#define R_CAPTURE_BINDING set = 1, binding = 4\n"
                        << "#define R_CAPTURE_RANGE_BINDING set = 1, binding = 5\n"
                        << prelude.str() << v;
    const std::string cmd =
        std::string(argv[1]) + " -V --quiet -o " + path + ".spv " + path + " > " + path + ".log 2>&1";
    check(std::system(cmd.c_str()) == 0, std::string(c.name) + ": glslang takes it (" + path + ")");
  }

  // No variant: a pixel shader; two mains; no position; a position line
  // not ending as the translator ends it; r_position called elsewhere too;
  // a name the capture uses taken.
  recomp::hlsl::Program ps;
  std::string err;
  const Tokens pt = Ps3();
  check(recomp::hlsl::TranslateBytecode(pt.data(), pt.size(), &ps, &err) &&
            recomp::hlsl::VertexWithCapturedPosition(ps).empty(),
        "no variant of a pixel shader");
  recomp::hlsl::Program vs;
  const Tokens vt = Vs3();
  if (!recomp::hlsl::TranslateBytecode(vt.data(), vt.size(), &vs, &err)) return 1;
  const std::string orig = vs.glsl;
  const size_t at = orig.find("  gl_Position = r_position(");
  const size_t main_at = orig.find("\nvoid main() {\n");
  struct Bad {
    const char* what;
    std::string glsl;
  } bad[] = {
      {"two mains", orig + "\nvoid main() {\n}\n"},
      {"no position", orig.substr(0, at) + orig.substr(orig.find('\n', at) + 1)},
      {"a position line in another shape",
       orig.substr(0, at) + "  gl_Position = r_position(r_o[0]) ;\n" + orig.substr(orig.find('\n', at) + 1)},
      {"r_position called elsewhere",
       orig.substr(0, main_at) + "\nvec4 r_other() { return r_position(vec4(0.0)); }" + orig.substr(main_at)},
      {"r_cap taken", "vec4 r_cap0;\n" + orig},
  };
  for (const Bad& b : bad) {
    recomp::hlsl::Program x = vs;
    x.glsl = b.glsl;
    check(recomp::hlsl::VertexWithCapturedPosition(x).empty(), std::string("no variant: ") + b.what);
  }
  return failures ? 1 : 0;
}
