// The bytecode translator (hlsl::TranslateBytecode) on hand-assembled
// shaders covering shader models 1.1-3.0: each translation is completed
// with tests/hlsl/prelude.glsl and must compile with glslangValidator
// (argv[1]); interfaces are checked against what the tokens declare. Reads
// of a sampler made calls of r_rt_reflect (TranslateOptions::
// ray_reflection: texld, texldl and texldd; the sampler not declared, the
// origin's input, the read's coordinate and the registers, a def's among
// them, as its arguments; the registers declared and read; compiling) and
// not where the origin is not an input or the option is not given. Reads
// of a sampler made calls of r_rt_shadow likewise (TranslateOptions::
// ray_shadow: the read's coordinate and the light's register as its
// arguments, a def's register as such; the stub "lit"; compiling) and not
// without the option or with the register past c223.
//
//   hlsl_bytecode_test <glslangValidator> <prelude.glsl> <out dir>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "runtime/hlsl/hlsl.h"

namespace {

using recomp::hlsl::Program;
using Tokens = std::vector<uint32_t>;

enum Reg : uint32_t {
  R = 0, V = 1, C = 2, A = 3, T = 3, RAST = 4, OD = 5, OT = 6, O = 6, I = 7, OC = 8, ODEPTH = 9,
  S = 10, B = 14, AL = 15, MISC = 17, LBL = 18, P = 19,
};

uint32_t TypeBits(uint32_t type) { return ((type & 7) << 28) | ((type & 0x18) << 8); }
uint32_t Dst(uint32_t type, uint32_t n, uint32_t mask = 0xF, uint32_t mod = 0, int shift = 0) {
  return 0x80000000u | n | TypeBits(type) | mask << 16 | mod << 20 | (uint32_t(shift) & 0xF) << 24;
}
uint32_t Src(uint32_t type, uint32_t n, uint32_t swz = 0xE4, uint32_t mod = 0, bool rel = false) {
  return 0x80000000u | n | TypeBits(type) | swz << 16 | mod << 24 | (rel ? 0x2000u : 0);
}
constexpr uint32_t kX = 0x00, kW = 0xFF, kXXXX = 0x00, kYYYY = 0x55;
uint32_t F(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

// Instruction token: SM2+ carry the length; SM1 do not.
struct Asm {
  Tokens t;
  bool sm1;
  Asm(bool pixel, int major, int minor) : sm1(major == 1) {
    t.push_back((pixel ? 0xFFFF0000u : 0xFFFE0000u) | uint32_t(major) << 8 | uint32_t(minor));
  }
  void I(uint32_t op, std::initializer_list<uint32_t> params, uint32_t ctl = 0, uint32_t flags = 0) {
    t.push_back(op | ctl << 16 | (sm1 ? 0 : uint32_t(params.size()) << 24) | flags);
    t.insert(t.end(), params);
  }
  void Def(uint32_t reg, float x, float y, float z, float w) {
    I(81, {Dst(C, reg), F(x), F(y), F(z), F(w)});
  }
  void Dcl(uint32_t usage, uint32_t index, uint32_t dst) { I(31, {0x80000000u | usage | index << 16, dst}); }
  void DclSampler(uint32_t type, uint32_t n) { I(31, {0x80000000u | type << 27, Dst(S, n)}); }
  Tokens End() {
    t.push_back(0x0000FFFEu | 2u << 16);  // a comment, skipped
    t.push_back(0x42415443);
    t.push_back(0);
    t.push_back(0x0000FFFF);
    return t;
  }
};

enum Op : uint32_t {
  MOV = 1, ADD = 2, MAD = 4, MUL = 5, RCP = 6, RSQ = 7, DP3 = 8, DP4 = 9, MIN = 10, MAX = 11,
  SLT = 12, SGE = 13, EXP = 14, LOG = 15, LIT = 16, DST = 17, LRP = 18, FRC = 19, M4X4 = 20,
  M3X3 = 23, CALL = 25, CALLNZ = 26, LOOP = 27, RET = 28, ENDLOOP = 29, LABEL = 30, POW = 32,
  CRS = 33, SGN = 34, ABS = 35, NRM = 36, SINCOS = 37, REP = 38, ENDREP = 39, IF = 40, IFC = 41,
  ELSE = 42, ENDIF = 43, BREAK = 44, BREAKC = 45, MOVA = 46, DEFB = 47, DEFI = 48,
  TEXCOORD = 64, TEXKILL = 65, TEX = 66, EXPP = 78, CND = 80, CMP = 88, DP2ADD = 90, DSX = 91,
  SETP = 94, TEXLDL = 95, BREAKP = 96,
};

struct Case {
  const char* name;
  Tokens tokens;
  size_t inputs, varyings;
  uint32_t colors;
};

std::vector<Case> Cases() {
  std::vector<Case> cs;
  {  // vs_1_1: matrices, relative addressing, lighting helpers
    Asm a(false, 1, 1);
    a.Def(0, 0.5f, -0.5f, 0, 1);
    a.Dcl(0, 0, Dst(V, 0));  // POSITION
    a.Dcl(3, 0, Dst(V, 1));  // NORMAL
    a.Dcl(10, 0, Dst(V, 2));  // COLOR
    a.Dcl(5, 0, Dst(V, 3));  // TEXCOORD0
    a.I(M4X4, {Dst(RAST, 0), Src(V, 0), Src(C, 4)});
    a.I(MOV, {Dst(A, 0, 1), Src(V, 3, kXXXX)});
    a.I(M3X3, {Dst(R, 0, 7), Src(V, 1), Src(C, 8, 0xE4, 0, true)});
    a.I(DP3, {Dst(R, 1, 1), Src(R, 0), Src(C, 12)});
    a.I(LIT, {Dst(R, 2), Src(R, 1)});
    a.I(EXPP, {Dst(R, 3), Src(R, 1, kXXXX)});
    a.I(MAD, {Dst(OD, 0), Src(V, 2), Src(R, 2, kYYYY), Src(C, 0, kXXXX)});
    a.I(RCP, {Dst(R, 4, 8), Src(R, 3, kW)});
    a.I(MOV, {Dst(OT, 0, 3), Src(V, 3)});
    a.I(MUL, {Dst(RAST, 1, 1), Src(R, 4, kW), Src(C, 13)});
    cs.push_back({"vs11", a.End(), 4, 3, 0});
  }
  {  // vs_2_0: flow control, subroutines, integer and bool constants
    Asm a(false, 2, 0);
    a.I(DEFI, {Dst(I, 0), 4, 0, 1, 0});
    a.I(DEFB, {Dst(B, 1), 1});
    a.Dcl(0, 0, Dst(V, 0));
    a.I(MOV, {Dst(R, 0), Src(C, 0)});
    a.I(LOOP, {Src(AL, 0), Src(I, 0)});
    a.I(ADD, {Dst(R, 0), Src(R, 0), Src(C, 10, 0xE4, 0, true), 0x80000000u | TypeBits(AL)});
    a.I(ENDLOOP, {});
    a.I(REP, {Src(I, 1)});
    a.I(BREAKC, {Src(R, 0, kXXXX), Src(C, 1, kXXXX)}, 1);
    a.I(ENDREP, {});
    a.I(IF, {Src(B, 1)});
    a.I(CALL, {Src(LBL, 0)});
    a.I(ELSE, {});
    a.I(SINCOS, {Dst(R, 1, 3), Src(R, 0, kXXXX), Src(C, 2), Src(C, 3)});
    a.I(ENDIF, {});
    a.I(CALLNZ, {Src(LBL, 1), Src(B, 2)});
    a.I(MOVA, {Dst(A, 0, 1), Src(R, 0, kXXXX)});
    a.I(DP4, {Dst(RAST, 0), Src(V, 0), Src(C, 20, 0xE4, 0, true), 0x80000000u | TypeBits(A)});
    a.I(MOV, {Dst(OT, 1), Src(R, 1)});
    a.I(RET, {});
    a.I(LABEL, {Src(LBL, 0)});
    a.I(NRM, {Dst(R, 1), Src(R, 0)});
    a.I(RET, {});
    a.I(LABEL, {Src(LBL, 1)});
    a.I(CRS, {Dst(R, 1, 7), Src(R, 0), Src(C, 5)});
    a.I(RET, {});
    cs.push_back({"vs20", a.End(), 1, 1, 0});
  }
  {  // vs_3_0: declared outputs, vertex texture fetch
    Asm a(false, 3, 0);
    a.Dcl(0, 0, Dst(V, 0));
    a.Dcl(5, 0, Dst(V, 1));
    a.Dcl(0, 0, Dst(O, 0));
    a.Dcl(5, 0, Dst(O, 1));
    a.Dcl(10, 0, Dst(O, 2));
    a.DclSampler(2, 0);
    a.I(TEXLDL, {Dst(R, 0), Src(V, 1), Src(S, 0)});
    a.I(ADD, {Dst(O, 0), Src(V, 0), Src(R, 0)});
    a.I(MOV, {Dst(O, 1), Src(V, 1)});
    a.I(MOV, {Dst(O, 2, 0xF, 1), Src(R, 0)});
    cs.push_back({"vs30", a.End(), 2, 2, 0});
  }
  {  // ps_1_1: texture registers, co-issue, cnd, lrp, modifiers
    Asm a(true, 1, 1);
    a.Def(0, 0.5f, 0.5f, 0.5f, 0.5f);
    a.I(TEX, {Dst(T, 0)});
    a.I(TEXCOORD, {Dst(T, 1)});
    a.I(DP3, {Dst(R, 0, 7, 1), Src(T, 0, 0xE4, 4), Src(V, 0, 0xE4, 4)});
    a.I(MUL, {Dst(R, 0, 8), Src(T, 0, kW), Src(V, 1, kW)}, 0, 0x40000000u);
    a.I(LRP, {Dst(R, 1), Src(C, 0), Src(T, 1), Src(R, 0)});
    a.I(CND, {Dst(R, 0, 0xF, 0, 1), Src(R, 0, kW), Src(R, 1), Src(T, 0, 0xE4, 6)});
    a.I(TEXKILL, {Dst(T, 1)});
    cs.push_back({"ps11", a.End(), 0, 4, 1});
  }
  {  // ps_2_0: declared inputs and samplers, texld variants, multiple outputs
    Asm a(true, 2, 0);
    a.I(31, {0x80000000u, Dst(T, 0, 3)});
    a.I(31, {0x80000000u, Dst(V, 0)});
    a.DclSampler(2, 0);
    a.DclSampler(3, 1);
    a.I(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
    a.I(TEX, {Dst(R, 1), Src(R, 0), Src(S, 1)});
    a.I(TEX, {Dst(R, 2), Src(T, 0), Src(S, 0)}, 1);
    a.I(TEX, {Dst(R, 3), Src(T, 0), Src(S, 0)}, 2);
    a.I(TEXKILL, {Dst(R, 0)});
    a.I(CMP, {Dst(R, 0), Src(R, 0, 0xE4, 1), Src(R, 1), Src(R, 2)});
    a.I(DP2ADD, {Dst(R, 0, 1), Src(R, 3), Src(V, 0), Src(C, 0, kXXXX)});
    a.I(MOV, {Dst(OC, 0), Src(R, 0)});
    a.I(MOV, {Dst(OC, 1), Src(R, 3)});
    a.I(MOV, {Dst(ODEPTH, 0, 1), Src(R, 1, kXXXX)});
    cs.push_back({"ps20", a.End(), 0, 2, 3});
  }
  {  // ps_3_0: vPos, vFace, predication, setp, derivatives, if on a compare
    Asm a(true, 3, 0);
    a.Dcl(5, 0, Dst(V, 0, 3));
    a.Dcl(10, 0, Dst(V, 1));
    a.I(31, {0x80000000u, Dst(MISC, 0, 3)});
    a.I(31, {0x80000000u, Dst(MISC, 1, 1)});
    a.I(SETP, {Dst(P, 0), Src(V, 0), Src(C, 0)}, 4);
    a.I(MOV, {Dst(R, 0), Src(MISC, 0)});
    // Predicated: the predicate token follows the destination.
    a.I(ADD, {Dst(R, 0), Src(P, 0), Src(R, 0), Src(V, 1)}, 0, 0x10000000u);
    a.I(IFC, {Src(MISC, 1, kXXXX), Src(C, 0, kXXXX)}, 1);
    a.I(DSX, {Dst(R, 1), Src(V, 0)});
    a.I(ENDIF, {});
    a.I(DEFI, {Dst(I, 0), 3, 0, 0, 0});
    a.I(REP, {Src(I, 0)});
    a.I(BREAKP, {Src(P, 0, kXXXX)});
    a.I(ADD, {Dst(R, 0), Src(R, 0), Src(C, 2)});
    a.I(ENDREP, {});
    a.I(POW, {Dst(R, 1, 1), Src(R, 0, kXXXX), Src(C, 1, kXXXX)});
    a.I(MOV, {Dst(OC, 0), Src(R, 0)});
    cs.push_back({"ps30", a.End(), 0, 2, 1});
  }
  return cs;
}

// ReadBytecode finds END by stepping over instructions: def values that
// look like a comment (low half 0xFFFE) or END must not stop it.
int ReadChecks() {
  int failures = 0;
  for (int major : {1, 3}) {
    Asm a(true, major, major == 1 ? 1 : 0);
    a.I(81, {Dst(C, 0), 0u, 0xBF7FFFFEu, 0x0000FFFFu, F(1)});  // def c0, 0, -0.99999988, ...
    a.I(MOV, {Dst(major == 1 ? R : OC, 0), Src(C, 0)});
    Tokens t = a.End();
    t.push_back(0xDEADBEEF);  // past the end
    Tokens r = recomp::hlsl::ReadBytecode([&](size_t i) { return t[i]; }, t.size());
    if (r.size() != t.size() - 1) {
      std::printf("FAIL read sm%d: %zu of %zu tokens\n", major, r.size(), t.size() - 1);
      failures++;
    } else {
      std::printf("ok read sm%d\n", major);
    }
  }
  Asm a(true, 3, 0);
  a.I(MOV, {Dst(OC, 0), Src(C, 0)});
  Tokens t = a.End();
  t.pop_back();  // no END
  if (!recomp::hlsl::ReadBytecode([&](size_t i) { return t[i]; }, t.size()).empty()) {
    std::printf("FAIL read: a function without END\n");
    failures++;
  }
  return failures;
}

// GlslWithDefaultInputs: unwritten inputs become constants, the rest stay.
int DefaultInputChecks() {
  Asm a(true, 3, 0);
  a.Dcl(5, 0, Dst(V, 0));   // TEXCOORD0
  a.Dcl(10, 0, Dst(V, 1));  // COLOR0
  a.Dcl(5, 9, Dst(V, 2));   // TEXCOORD9
  a.I(ADD, {Dst(R, 0), Src(V, 0), Src(V, 1)});
  a.I(ADD, {Dst(OC, 0), Src(R, 0), Src(V, 2)});
  Tokens t = a.End();
  Program p;
  std::string err;
  if (!recomp::hlsl::TranslateBytecode(t.data(), t.size(), &p, &err)) {
    std::printf("FAIL default inputs: %s\n", err.c_str());
    return 1;
  }
  std::string g = recomp::hlsl::GlslWithDefaultInputs(p, recomp::hlsl::VaryingMask(p) & ~((1u << 16) | (1u << 9)));
  bool ok = g.find("layout(location = 0) in vec4") != std::string::npos &&
            g.find("const vec4 r_var16 = vec4(1.0);") != std::string::npos &&
            g.find("const vec4 r_var9 = vec4(0.0);") != std::string::npos &&
            g.find("layout(location = 9)") == std::string::npos &&
            g.find("layout(location = 0) out vec4") != std::string::npos;
  std::printf("%s default inputs\n", ok ? "ok" : "FAIL");
  if (!ok) std::printf("%s\n", g.c_str());
  return ok ? 0 : 1;
}

// glslangValidator's verdict on a pixel shader's translation.
bool Compiles(const char* glslang, const std::string& prelude, const std::string& dir, const char* name,
              const Program& p) {
  const std::string path = dir + "/bytecode_" + name + ".frag";
  std::ofstream(path) << "#version 450\n#define R_PIXEL_SHADER\n" << prelude << "\n" << p.glsl;
  const std::string cmd = std::string("\"") + glslang + "\" -V --target-env vulkan1.0 -S frag \"" + path +
                          "\" -o \"" + path + ".spv\" > \"" + path + ".log\" 2>&1";
  return std::system(cmd.c_str()) == 0;
}

// TranslateOptions::ray_reflection.
int RayReflectionChecks(const char* glslang, const std::string& prelude, const std::string& dir) {
  int failures = 0;
  auto check = [&](bool ok, const char* what, const Program& p) {
    std::printf("%s ray reflection: %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) {
      std::printf("%s\n", p.glsl.c_str());
      failures++;
    }
  };
  // texld (op 0) or texldl / texldd of s3 at r5 (the direction), the
  // origin TEXCOORD2 at v4.
  auto shader = [](uint32_t op) {
    Asm a(true, 3, 0);
    a.Def(0, 0.25f, 0.5f, 0.75f, 1.0f);
    a.Dcl(5, 0, Dst(V, 0));
    a.Dcl(5, 2, Dst(V, 4, 7));
    a.DclSampler(2, 0);
    a.DclSampler(2, 3);
    a.I(TEX, {Dst(R, 0), Src(V, 0), Src(S, 0)});
    a.I(NRM, {Dst(R, 5, 7), Src(V, 0)});
    if (op == 93)  // texldd
      a.I(93, {Dst(R, 2), Src(R, 5), Src(S, 3), Src(V, 0), Src(V, 0)});
    else
      a.I(op, {Dst(R, 2), Src(R, 5), Src(S, 3)});
    a.I(LRP, {Dst(R, 0, 7), Src(R, 2, kW), Src(R, 2), Src(R, 0)});
    a.I(MOV, {Dst(OC, 0), Src(R, 0)});
    return a.End();
  };
  recomp::hlsl::TranslateOptions o;
  o.ray_reflection.sampler = 3;
  o.ray_reflection.origin = 4;
  o.ray_reflection.registers[0] = 17;
  o.ray_reflection.registers[1] = 18;
  o.ray_reflection.registers[2] = 37;
  o.ray_reflection.registers[3] = 0;  // the def's c0
  const std::string call = "r_rt_reflect(gl_FragCoord, r_v[4], r_t[5], c[17], c[18], c[37], r_dc0)";
  const struct {
    uint32_t op;
    const char* name;
  } kReads[] = {{TEX, "texld"}, {TEXLDL, "texldl"}, {93, "texldd"}};
  for (const auto& r : kReads) {
    const Tokens t = shader(r.op);
    Program p;
    std::string err;
    if (!recomp::hlsl::TranslateBytecode(t.data(), t.size(), &p, &err, o)) {
      std::printf("FAIL ray reflection %s: %s\n", r.name, err.c_str());
      failures++;
      continue;
    }
    bool only_s0 = p.samplers.size() == 1 && p.samplers[0].reg == 0;
    check(p.ray_reflection_sampler == 3 && p.glsl.find(call) != std::string::npos &&
              p.glsl.find("r_s3") == std::string::npos && only_s0 && p.float_regs >= 38 &&
              p.float_reads.test(17) && p.float_reads.test(18) && p.float_reads.test(37) && !p.float_reads.test(0) &&
              p.glsl.find("vec4 r_rt_reflect(vec4 pixel") != std::string::npos,
          (std::string(r.name) + " of s3: the call, s3 not declared, the registers read").c_str(), p);
    check(Compiles(glslang, prelude, dir, (std::string("rt_reflect_") + r.name).c_str(), p) &&
              p.ray_reflection_location == recomp::hlsl::VaryingLocation({recomp::hlsl::kTexcoord, 2}) &&
              p.ray_reflection_stub_calls == 0,
          (std::string(r.name) + ": glslang takes it; the origin's location, no call in a subroutine").c_str(), p);
    if (r.op != TEX) continue;
    // A variant for unwritten inputs keeps the calls.
    const std::string g = recomp::hlsl::GlslWithDefaultInputs(p, recomp::hlsl::VaryingMask(p) & ~1u);
    check(g.find(call) != std::string::npos, "a variant without TEXCOORD0 keeps the call", p);
    // Not without the option, nor with an origin that is not an input.
    Program q;
    recomp::hlsl::TranslateBytecode(t.data(), t.size(), &q, &err);
    recomp::hlsl::TranslateOptions bad = o;
    bad.ray_reflection.origin = 6;
    Program b;
    recomp::hlsl::TranslateBytecode(t.data(), t.size(), &b, &err, bad);
    recomp::hlsl::TranslateOptions far = o;
    far.ray_reflection.registers[1] = 224;
    Program fr;
    recomp::hlsl::TranslateBytecode(t.data(), t.size(), &fr, &err, far);
    check(q.ray_reflection_sampler == -1 && q.glsl.find("r_rt_reflect") == std::string::npos &&
              q.glsl.find("uniform sampler2D r_s3") != std::string::npos && b.ray_reflection_sampler == -1 &&
              b.glsl.find("uniform sampler2D r_s3") != std::string::npos && fr.ray_reflection_sampler == -1,
          "s3 read as usual without the option, with v6 (not declared) as the origin, a register past c223", q);
  }
  // texldp: the coordinate over its w; a read in a subroutine counted.
  {
    Asm a(true, 3, 0);
    a.Dcl(5, 2, Dst(V, 4, 7));
    a.Dcl(5, 1, Dst(V, 1));
    a.DclSampler(2, 3);
    a.I(TEX, {Dst(R, 0), Src(V, 1), Src(S, 3)}, 1);  // texldp r0, v1, s3
    a.I(CALL, {Src(LBL, 0)});
    a.I(MOV, {Dst(OC, 0), Src(R, 0)});
    a.I(RET, {});
    a.I(LABEL, {Src(LBL, 0)});
    a.I(TEX, {Dst(R, 1), Src(V, 1), Src(S, 3)});
    a.I(ADD, {Dst(R, 0), Src(R, 0), Src(R, 1)});
    a.I(RET, {});
    const Tokens t = a.End();
    Program p;
    std::string err;
    const bool ok = recomp::hlsl::TranslateBytecode(t.data(), t.size(), &p, &err, o);
    check(ok && p.glsl.find("r_rt_reflect(gl_FragCoord, r_v[4], (r_v[1] / (r_v[1]).w), c[17]") != std::string::npos &&
              p.ray_reflection_stub_calls == 1 && Compiles(glslang, prelude, dir, "rt_reflect_texldp", p),
          "texldp's coordinate over its w; the read in a subroutine counted; glslang takes it", p);
  }
  return failures;
}

// TranslateOptions::ray_shadow.
int RayShadowChecks(const char* glslang, const std::string& prelude, const std::string& dir) {
  int failures = 0;
  auto check = [&](bool ok, const char* what, const Program& p) {
    std::printf("%s ray shadow: %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) {
      std::printf("%s\n", p.glsl.c_str());
      failures++;
    }
  };
  // texld (op 0) or texldl / texldd of s15 at r5 (the point), the light
  // in c17; the result's x scales the colour.
  auto shader = [](uint32_t op) {
    Asm a(true, 3, 0);
    a.Def(0, 0.25f, 0.5f, 0.75f, 1.0f);
    a.Dcl(5, 0, Dst(V, 0));
    a.Dcl(5, 6, Dst(V, 4, 7));
    a.DclSampler(2, 0);
    a.DclSampler(2, 15);
    a.I(TEX, {Dst(R, 0), Src(V, 0), Src(S, 0)});
    a.I(MOV, {Dst(R, 5, 7), Src(V, 4)});
    if (op == 93)  // texldd
      a.I(93, {Dst(R, 2), Src(R, 5), Src(S, 15), Src(V, 0), Src(V, 0)});
    else
      a.I(op, {Dst(R, 2), Src(R, 5), Src(S, 15)});
    a.I(MUL, {Dst(R, 0, 7), Src(R, 0), Src(R, 2, kX)});
    a.I(MOV, {Dst(OC, 0), Src(R, 0)});
    return a.End();
  };
  recomp::hlsl::TranslateOptions o;
  o.ray_shadow.sampler = 15;
  o.ray_shadow.light_register = 17;
  const std::string call = "r_rt_shadow(gl_FragCoord, r_t[5], c[17])";
  const struct {
    uint32_t op;
    const char* name;
  } kReads[] = {{TEX, "texld"}, {TEXLDL, "texldl"}, {93, "texldd"}};
  for (const auto& r : kReads) {
    const Tokens t = shader(r.op);
    Program p;
    std::string err;
    if (!recomp::hlsl::TranslateBytecode(t.data(), t.size(), &p, &err, o)) {
      std::printf("FAIL ray shadow %s: %s\n", r.name, err.c_str());
      failures++;
      continue;
    }
    const bool only_s0 = p.samplers.size() == 1 && p.samplers[0].reg == 0;
    if (r.op == 93) {
      // A texldd is the shadow toward a point: the point, the vector to
      // the light and its weight (the gradients), no light register read.
      const size_t at = p.glsl.find("r_rt_shadow_to(gl_FragCoord, r_t[5], ");
      check(p.ray_shadow_sampler == 15 && p.ray_reflection_sampler == -1 && at != std::string::npos &&
                p.glsl.find("r_rt_shadow(") == std::string::npos && p.glsl.find("r_s15") == std::string::npos &&
                only_s0 && !p.float_reads.test(17) &&
                p.glsl.find("vec4 r_rt_shadow_to(vec4 pixel, vec4 world, vec4 to, vec4 weight) {\n  return vec4(1.0);") !=
                    std::string::npos &&
                p.ray_shadow_stub_calls == 0,
            "texldd of s15: the call toward a point, s15 not declared, c17 not read, the stub lit", p);
      check(Compiles(glslang, prelude, dir, "rt_shadow_texldd", p), "texldd: glslang takes it", p);
      continue;
    }
    check(p.ray_shadow_sampler == 15 && p.ray_reflection_sampler == -1 && p.glsl.find(call) != std::string::npos &&
              p.glsl.find("r_s15") == std::string::npos && only_s0 && p.float_reads.test(17) &&
              p.glsl.find("vec4 r_rt_shadow(vec4 pixel, vec4 world, vec4 light) {\n  return vec4(1.0);") !=
                  std::string::npos &&
              p.ray_shadow_stub_calls == 0,
          (std::string(r.name) + " of s15: the call, s15 not declared, c17 read, the stub lit").c_str(), p);
    check(Compiles(glslang, prelude, dir, (std::string("rt_shadow_") + r.name).c_str(), p),
          (std::string(r.name) + ": glslang takes it").c_str(), p);
    if (r.op != TEX) continue;
    Program q, fr, d;
    recomp::hlsl::TranslateBytecode(t.data(), t.size(), &q, &err);
    recomp::hlsl::TranslateOptions far = o;
    far.ray_shadow.light_register = 224;
    recomp::hlsl::TranslateBytecode(t.data(), t.size(), &fr, &err, far);
    recomp::hlsl::TranslateOptions def = o;
    def.ray_shadow.light_register = 0;  // the def's c0
    recomp::hlsl::TranslateBytecode(t.data(), t.size(), &d, &err, def);
    check(q.ray_shadow_sampler == -1 && q.glsl.find("r_rt_shadow") == std::string::npos &&
              q.glsl.find("uniform sampler2D r_s15") != std::string::npos && fr.ray_shadow_sampler == -1 &&
              fr.glsl.find("uniform sampler2D r_s15") != std::string::npos &&
              d.glsl.find("r_rt_shadow(gl_FragCoord, r_t[5], r_dc0)") != std::string::npos && !d.float_reads.test(0),
          "s15 read as usual without the option or with a register past c223; a def's register as the def", q);
  }
  return failures;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: hlsl_bytecode_test <glslangValidator> <prelude.glsl> <out dir>\n");
    return 2;
  }
  std::ifstream pf(argv[2]);
  std::stringstream pre;
  pre << pf.rdbuf();
  int failures = 0;
  for (const Case& c : Cases()) {
    Program p;
    std::string err;
    if (!recomp::hlsl::TranslateBytecode(c.tokens.data(), c.tokens.size(), &p, &err)) {
      std::printf("FAIL %s: %s\n", c.name, err.c_str());
      failures++;
      continue;
    }
    // The translation the same with the coverage ray tracing asks for.
    {
      Program q;
      recomp::hlsl::TranslateOptions o;
      o.coverage = true;
      if (!recomp::hlsl::TranslateBytecode(c.tokens.data(), c.tokens.size(), &q, &err, o) || q.glsl != p.glsl) {
        std::printf("FAIL %s: another translation with coverage\n", c.name);
        failures++;
      }
    }
    bool vs = p.stage == recomp::hlsl::Stage::Vertex;
    if (p.inputs.size() != c.inputs || p.varyings.size() != c.varyings || p.color_outputs != c.colors) {
      std::printf("FAIL %s: %zu inputs, %zu varyings, colours %X\n", c.name, p.inputs.size(),
                  p.varyings.size(), p.color_outputs);
      failures++;
    }
    std::string path = std::string(argv[3]) + "/bytecode_" + c.name + (vs ? ".vert" : ".frag");
    std::ofstream(path) << "#version 450\n#define " << (vs ? "R_VERTEX_SHADER" : "R_PIXEL_SHADER")
                        << "\n" << pre.str() << "\n" << p.glsl;
    std::string cmd = std::string("\"") + argv[1] + "\" -V --target-env vulkan1.0 -S " +
                      (vs ? "vert" : "frag") + " \"" + path + "\" -o \"" + path + ".spv\" > \"" +
                      path + ".log\" 2>&1";
    if (std::system(cmd.c_str()) != 0) {
      std::printf("FAIL %s: glslang rejected %s (see %s.log)\n", c.name, path.c_str(), path.c_str());
      failures++;
      continue;
    }
    std::printf("ok %s\n", c.name);
  }
  failures += ReadChecks() + DefaultInputChecks() + RayReflectionChecks(argv[1], pre.str(), argv[3]) +
              RayShadowChecks(argv[1], pre.str(), argv[3]);
  return failures ? 1 : 0;
}
