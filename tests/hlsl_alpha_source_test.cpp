// What compiled shaders say about a pixel's coverage (hlsl::Program's
// pixel_alpha and moved_lanes, TranslateOptions::coverage), on
// hand-assembled ps_1_1, ps_2_0, ps_3_0, vs_2_0 and vs_3_0 shaders:
// COLOR0's alpha from a texel straight, times a colour varying, through
// temporaries and swizzles, times a constant register; a texkill of the
// alpha less a number or a constant register; source modifiers, result
// shifts, saturation, the other texture fetches, several COLOR0 writes;
// the cases that must stay unknown (texldp, a swizzled sampler, if blocks,
// predicated writes, loops, other instructions, two texkills, cube
// samplers, unlike components); what a texkill may read; coverage only
// on request and bounded in length; and the vertex shader's varyings
// moved from its inputs.
//
//   hlsl_alpha_source_test

#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

#include "runtime/hlsl/hlsl.h"

namespace {

using recomp::hlsl::AlphaValue;
using recomp::hlsl::MovedLane;
using recomp::hlsl::Program;
using Tokens = std::vector<uint32_t>;

enum Reg : uint32_t { R = 0, V = 1, C = 2, T = 3, OD = 5, OT = 6, O = 6, OC = 8, S = 10, I = 7, AL = 15, MISC = 17, P = 19 };
enum Op : uint32_t {
  MOV = 1, ADD = 2, MAD = 4, MUL = 5, RCP = 6, DP4 = 9, M3X3 = 23, CALL = 25, CALLNZ = 26, RET = 28, LABEL = 30, REP = 38,
  ENDREP = 39, IFC = 41, ELSE = 42, ENDIF = 43, BREAK = 44, BREAKC = 45, DEFI = 48, TEXCOORD = 64,
  TEXKILL = 65, TEX = 66, CMP = 88,
  TEXLDD = 93, SETP = 94, TEXLDL = 95,
};
constexpr uint32_t B = 14, LBL = 18;
constexpr uint32_t kXXXX = 0x00, kYYYY = 0x55, kWWWW = 0xFF, kZWZW = 0xEE;

uint32_t TypeBits(uint32_t type) { return ((type & 7) << 28) | ((type & 0x18) << 8); }
uint32_t Dst(uint32_t type, uint32_t n, uint32_t mask = 0xF, uint32_t mod = 0, int shift = 0) {
  return 0x80000000u | n | TypeBits(type) | mask << 16 | mod << 20 | (uint32_t(shift) & 0xF) << 24;
}
uint32_t Src(uint32_t type, uint32_t n, uint32_t swz = 0xE4, uint32_t mod = 0) {
  return 0x80000000u | n | TypeBits(type) | swz << 16 | mod << 24;
}
uint32_t F(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

struct Asm {
  Tokens t;
  bool sm1;
  Asm(bool pixel, int major, int minor) : sm1(major == 1) {
    t.push_back((pixel ? 0xFFFF0000u : 0xFFFE0000u) | uint32_t(major) << 8 | uint32_t(minor));
  }
  void Ins(uint32_t op, std::initializer_list<uint32_t> params, uint32_t ctl = 0, uint32_t flags = 0) {
    t.push_back(op | ctl << 16 | (sm1 ? 0 : uint32_t(params.size()) << 24) | flags);
    t.insert(t.end(), params);
  }
  void Def(uint32_t reg, float x, float y, float z, float w) { Ins(81, {Dst(C, reg), F(x), F(y), F(z), F(w)}); }
  void Dcl(uint32_t usage, uint32_t index, uint32_t dst) { Ins(31, {0x80000000u | usage | index << 16, dst}); }
  void DclSampler(uint32_t type, uint32_t n) { Ins(31, {0x80000000u | type << 27, Dst(S, n)}); }
  Tokens End() {
    t.push_back(0x0000FFFF);
    return t;
  }
};

int failures = 0;

void Check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) failures++;
}

// Translated with coverage (TranslateOptions), as with ray tracing on.
Program Translate(const Tokens& t, bool coverage = true) {
  Program p;
  std::string err;
  recomp::hlsl::TranslateOptions o;
  o.coverage = coverage;
  if (!recomp::hlsl::TranslateBytecode(t.data(), t.size(), &p, &err, o)) {
    std::printf("FAIL translate: %s\n", err.c_str());
    failures++;
  }
  return p;
}

// A texel of sampler `s` (channel `ch`) at TEXCOORDn lanes u, v, and the
// rest of a value as given.
bool Texel(const AlphaValue& a, int s, int ch, uint32_t n, int u, int v) {
  return a.known && a.sampler == s && a.channel == ch && a.texcoord == n && a.u == u && a.v == v;
}

// ps_2_0 with t0 and v0 declared and 2D s0 (cube s1): the body, then END.
template <typename F>
Program Ps20(F body, bool coverage = true) {
  Asm a(true, 2, 0);
  a.Ins(31, {0x80000000u, Dst(T, 0)});
  a.Ins(31, {0x80000000u, Dst(V, 0)});
  a.DclSampler(2, 0);
  a.DclSampler(3, 1);
  body(a);
  return Translate(a.End(), coverage);
}

// The texel's alpha at t0 into r0, the body, then oC0 = r0.
template <typename F>
AlphaValue Color0(F body) {
  return Ps20([&](Asm& a) {
           a.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
           body(a);
           a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
         })
      .pixel_alpha.color0;
}

// Modifiers, shifts, saturation, predication, the other texture fetches,
// several COLOR0 writes, a ps_3_0 colour varying, a swizzled sampler; the
// coverage only on request and bounded; what texkills may read.
void MoreChecks() {
  // Source modifiers (ps_1_x's too, read the same way): r0.a op'd into r0.a.
  struct Mod {
    uint32_t mod;
    float scale, offset;
    const char* name;
  };
  const Mod mods[] = {{2, 1, -0.5f, "_bias"}, {4, 2, -1, "_bx2"}, {6, -1, 1, "1-x"}, {7, 2, 0, "_x2"},
                      {1, -1, 0, "negation"}};
  bool ok = true;
  for (const Mod& m : mods) {
    const AlphaValue a = Color0([&](Asm& as) { as.Ins(MOV, {Dst(R, 0, 8), Src(R, 0, kWWWW, m.mod)}); });
    const bool one = Texel(a, 0, 3, 0, 0, 1) && a.scale == m.scale && a.offset == m.offset;
    if (!one) std::printf("  (%s: scale %g offset %g)\n", m.name, a.scale, a.offset);
    ok = ok && one;
  }
  const AlphaValue abs = Color0([](Asm& as) { as.Ins(MOV, {Dst(R, 0, 8), Src(R, 0, kWWWW, 11)}); });
  Check(ok && !abs.known, "source modifiers: _bias, _bx2, 1-x, _x2 and negation as scale and offset; abs not known");
  // Result shifts _x2 and _d2; saturation refused above 1 or with a
  // register's factor, kept on COLOR0 itself.
  const AlphaValue x2 = Color0([](Asm& as) { as.Ins(MOV, {Dst(R, 0, 8, 0, 1), Src(R, 0, kWWWW)}); });
  const AlphaValue d2 = Color0([](Asm& as) { as.Ins(MOV, {Dst(R, 0, 8, 0, -1), Src(R, 0, kWWWW)}); });
  Check(Texel(x2, 0, 3, 0, 0, 1) && x2.scale == 2.0f && Texel(d2, 0, 3, 0, 0, 1) && d2.scale == 0.5f,
        "result shifts: _x2 and _d2");
  const AlphaValue sat_big = Color0([](Asm& as) { as.Ins(MOV, {Dst(R, 0, 8, 1, 1), Src(R, 0, kWWWW)}); });
  const AlphaValue sat_const = Color0([](Asm& as) {
    as.Ins(MUL, {Dst(R, 0, 8, 1), Src(R, 0, kWWWW), Src(C, 4, kXXXX)});
  });
  const AlphaValue sat_half = Color0([](Asm& as) { as.Ins(MOV, {Dst(R, 0, 8, 1, -1), Src(R, 0, kWWWW)}); });
  const Program sat_out = Ps20([](Asm& as) {
    as.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
    as.Ins(MUL, {Dst(OC, 0, 0xF, 1), Src(R, 0), Src(C, 4, kXXXX)});
  });
  Check(!sat_big.known && !sat_const.known && Texel(sat_half, 0, 3, 0, 0, 1) &&
            sat_half.scale == 0.5f && Texel(sat_out.pixel_alpha.color0, 0, 3, 0, 0, 1) &&
            sat_out.pixel_alpha.color0.constant == 4,
        "saturation: refused at scale 2 and with a register's factor, kept at 0.5 and on COLOR0 (the test clamps)");
  // A predicated write: not known after it.
  const AlphaValue pred = Color0([](Asm& as) {
    as.Ins(SETP, {Dst(P, 0), Src(V, 0), Src(C, 0)}, 4);
    as.Ins(MOV, {Dst(R, 0, 8), Src(P, 0), Src(C, 1)}, 0, 0x10000000u);
  });
  Check(!pred.known, "a predicated write: not known");
  // texldb, texldl, texldd as texld; texldp not (checked above).
  {
    bool fetch = true;
    for (int kind = 0; kind < 3; kind++) {
      const Program p = Ps20([&](Asm& as) {
        if (kind == 0) as.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)}, 2);
        if (kind == 1) as.Ins(TEXLDL, {Dst(R, 0), Src(T, 0), Src(S, 0)});
        if (kind == 2) as.Ins(TEXLDD, {Dst(R, 0), Src(T, 0), Src(S, 0), Src(C, 0), Src(C, 1)});
        as.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
      });
      fetch = fetch && Texel(p.pixel_alpha.color0, 0, 3, 0, 0, 1);
    }
    Check(fetch, "texldb, texldl and texldd: the texel at the coordinate");
  }
  // Several writes to oC0.a: the last one counts.
  {
    const Program p = Ps20([](Asm& as) {
      as.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
      as.Ins(MOV, {Dst(OC, 0), Src(C, 3)});
      as.Ins(MOV, {Dst(OC, 0, 8), Src(R, 0, kWWWW)});
      as.Ins(MOV, {Dst(OC, 0, 7), Src(C, 3)});
    });
    const Program q = Ps20([](Asm& as) {
      as.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
      as.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
      as.Ins(MOV, {Dst(OC, 0, 8), Src(C, 3, kXXXX)});
    });
    Check(Texel(p.pixel_alpha.color0, 0, 3, 0, 0, 1) && q.pixel_alpha.color0.known &&
              q.pixel_alpha.color0.sampler < 0 && q.pixel_alpha.color0.constant == 3,
          "several writes to oC0: the last one to its alpha counts");
  }
  // ps_3_0: a COLOR1 varying's alpha alone; a swizzled sampler not known.
  {
    Asm a(true, 3, 0);
    a.Dcl(10, 1, Dst(V, 2));
    a.Ins(MOV, {Dst(OC, 0), Src(V, 2)});
    const AlphaValue c = Translate(a.End()).pixel_alpha.color0;
    Asm b(true, 3, 0);
    b.Dcl(5, 0, Dst(V, 0));
    b.DclSampler(2, 0);
    b.Ins(TEX, {Dst(R, 0), Src(V, 0), Src(S, 0, 0x1B)});  // s0.wzyx
    b.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    const AlphaValue sw = Translate(b.End()).pixel_alpha.color0;
    Check(c.known && c.sampler < 0 && c.color == 1 && c.color_lane == 3 && c.scale == 1.0f && !sw.known,
          "ps_3_0: COLOR1's alpha alone known; a swizzled sampler not");
  }
  // No coverage asked for: nothing found (the translation the same).
  {
    auto body = [](Asm& as) {
      as.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
      as.Ins(TEXKILL, {Dst(R, 0)});
      as.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    };
    const Program off = Ps20(body, false), on = Ps20(body, true);
    Check(!off.pixel_alpha.color0.known && off.pixel_alpha.kills == 0 && off.glsl == on.glsl &&
              on.pixel_alpha.color0.known,
          "coverage only on request; the translation the same either way");
  }
  // Bounded: a shader past kCoverageInstructions is not followed; one at
  // the limit (long straight code of the shape that reaches it: temporaries
  // passed along, co-issue-free) still is, in one pass.
  {
    auto chain = [](size_t n) {
      return Ps20([n](Asm& as) {
        as.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
        for (size_t i = 0; i < n; i++) as.Ins(MOV, {Dst(R, uint32_t(1 + i % 30)), Src(R, uint32_t(i % 30))});
        as.Ins(MOV, {Dst(OC, 0), Src(R, uint32_t(n % 30))});
      });
    };
    const size_t fits = recomp::hlsl::kCoverageInstructions - 8;
    const Program in = chain(fits), past = chain(recomp::hlsl::kCoverageInstructions);
    Check(!past.pixel_alpha.color0.known && in.pixel_alpha.color0.known,
          "bounded: past kCoverageInstructions nothing is followed");
    const Program big = Ps20([](Asm& as) {
      as.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
      for (size_t i = 0; i < recomp::hlsl::kCoverageInstructions; i++) as.Ins(MOV, {Dst(R, 1), Src(R, 0)});
      as.Ins(TEXKILL, {Dst(R, 1)});
      as.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    });
    Check(big.pixel_alpha.kills == 1 && big.pixel_alpha.kill_reads_texel && !big.pixel_alpha.kill.known,
          "past kCoverageInstructions: texkills and what they read still found");
  }
  // What a texkill may read: a texel through a loop and an if block (the
  // usual guess then applies); a texture coordinate alone (a clip plane:
  // not).
  {
    Asm a(true, 3, 0);
    a.Dcl(5, 0, Dst(V, 0));
    a.DclSampler(2, 0);
    a.Ins(DEFI, {Dst(I, 0), 2, 0, 0, 0});
    a.Ins(MOV, {Dst(R, 1), Src(C, 0)});
    a.Ins(REP, {Src(I, 0)});
    a.Ins(ADD, {Dst(R, 2), Src(R, 1), Src(C, 1)});
    a.Ins(TEX, {Dst(R, 1), Src(V, 0), Src(S, 0)});
    a.Ins(ENDREP, {});
    a.Ins(TEXKILL, {Dst(R, 2)});
    a.Ins(MOV, {Dst(OC, 0), Src(R, 1)});
    const Program texel = Translate(a.End());
    const Program plane = Ps20([](Asm& as) {
      as.Ins(ADD, {Dst(R, 1), Src(T, 0), Src(C, 2)});
      as.Ins(TEXKILL, {Dst(R, 1)});
      as.Ins(MOV, {Dst(OC, 0), Src(C, 0)});
    });
    Check(texel.pixel_alpha.kill_reads_texel && !texel.pixel_alpha.kill.known && !plane.pixel_alpha.kill_reads_texel &&
              !plane.pixel_alpha.kill.known && plane.pixel_alpha.kills == 1,
          "texkill: a texel reached through a loop (a later iteration) read; a texture coordinate alone not");
  }
  // Registers reused in program order (fxc does): a texkill before a texld
  // into its register, a register written clean after a texld, a
  // component left from it; a texel deciding by predicate or if block
  // whether a texkill runs.
  {
    auto ps3 = [](auto body) {
      Asm a(true, 3, 0);
      a.Dcl(5, 0, Dst(V, 0));
      a.Ins(31, {0x80000000u, Dst(MISC, 0, 3)});  // vPos
      a.DclSampler(2, 0);
      body(a);
      return Translate(a.End()).pixel_alpha;
    };
    const auto before = ps3([](Asm& a) {
      a.Ins(ADD, {Dst(R, 0), Src(MISC, 0, kYYYY), Src(C, 0, kXXXX, 1)});
      a.Ins(TEXKILL, {Dst(R, 0)});
      a.Ins(TEX, {Dst(R, 0), Src(V, 0), Src(S, 0)});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    });
    const auto after = ps3([](Asm& a) {
      a.Ins(TEX, {Dst(R, 0), Src(V, 0), Src(S, 0)});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
      a.Ins(ADD, {Dst(R, 0), Src(MISC, 0, kYYYY), Src(C, 0, kXXXX, 1)});
      a.Ins(TEXKILL, {Dst(R, 0)});
    });
    const auto partial = ps3([](Asm& a) {
      a.Ins(TEX, {Dst(R, 0), Src(V, 0), Src(S, 0)});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
      a.Ins(MOV, {Dst(R, 0, 7), Src(C, 0)});
      a.Ins(TEXKILL, {Dst(R, 0)});
    });
    const auto in_if = ps3([](Asm& a) {
      a.Ins(TEX, {Dst(R, 0), Src(V, 0), Src(S, 0)});
      a.Ins(MOV, {Dst(R, 1), Src(C, 0)});
      a.Ins(IFC, {Src(R, 0, kWWWW), Src(C, 1, kXXXX)}, 4);
      a.Ins(MOV, {Dst(R, 1), Src(C, 2)});
      a.Ins(ENDIF, {});
      a.Ins(TEXKILL, {Dst(R, 1)});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    });
    const auto cond_kill = ps3([](Asm& a) {
      a.Ins(TEX, {Dst(R, 0), Src(V, 0), Src(S, 0)});
      a.Ins(IFC, {Src(R, 0, kWWWW), Src(C, 1, kXXXX)}, 4);
      a.Ins(TEXKILL, {Dst(R, 5)});
      a.Ins(ENDIF, {});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    });
    const auto pred_kill = ps3([](Asm& a) {
      a.Ins(TEX, {Dst(R, 0), Src(V, 0), Src(S, 0)});
      a.Ins(SETP, {Dst(P, 0), Src(R, 0, kWWWW), Src(C, 1, kXXXX)}, 4);
      a.Ins(TEXKILL, {Dst(R, 5), Src(P, 0)}, 0, 0x10000000u);
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    });
    Check(before.kills == 1 && !before.kill_reads_texel && !after.kill_reads_texel && partial.kill_reads_texel,
          "texkill in program order: before a texld into its register, after its register is written clean: no "
          "texel read; a component the clean write leaves: read");
    Check(in_if.kill_reads_texel && cond_kill.kill_reads_texel && pred_kill.kill_reads_texel,
          "texkill decided by a texel: a register written in an if block on it, a texkill inside one, a predicated one");
    // A screen-door fade: a dither texture read at the screen position,
    // compared, killed by: no texel of the surface; the same read at an
    // interpolated coordinate: one.
    auto dither = [&](bool at_texcoord) {
      return ps3([&](Asm& a) {
        a.DclSampler(2, 10);
        if (at_texcoord) a.Ins(MOV, {Dst(R, 0, 3), Src(V, 0)});
        else a.Ins(MUL, {Dst(R, 0, 3), Src(C, 3, 0xAA), Src(MISC, 0)});
        a.Ins(MOV, {Dst(R, 0, 0xC), Src(C, 6)});
        a.Ins(TEXLDL, {Dst(R, 0), Src(R, 0), Src(S, 10)});
        a.Ins(CMP, {Dst(R, 0), Src(R, 0, kYYYY, 1), Src(C, 4), Src(C, 5)});
        a.Ins(TEXKILL, {Dst(R, 0)});
        a.Ins(MOV, {Dst(OC, 0), Src(C, 0)});
      });
    };
    const auto by_position = dither(false), by_texcoord = dither(true);
    Check(by_position.kills == 1 && !by_position.kill_reads_texel && by_texcoord.kill_reads_texel,
          "a dither texture read at vPos: no texel read (uncut); read at TEXCOORD0: read");
    // Branches each on their own state: a clean write in one clears there.
    const auto branch = ps3([](Asm& a) {
      a.Ins(TEX, {Dst(R, 0), Src(V, 0), Src(S, 0)});
      a.Ins(IFC, {Src(V, 0, kXXXX), Src(C, 0, kXXXX)}, 4);
      a.Ins(MOV, {Dst(R, 0), Src(C, 1)});
      a.Ins(TEXKILL, {Dst(R, 0)});
      a.Ins(ELSE, {});
      a.Ins(MOV, {Dst(R, 1), Src(C, 2)});
      a.Ins(ENDIF, {});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    });
    // Only the components a per-component instruction reads.
    const auto lanes = ps3([](Asm& a) {
      a.Ins(MOV, {Dst(R, 1), Src(C, 0)});
      a.Ins(TEX, {Dst(R, 2), Src(V, 0), Src(S, 0)});
      a.Ins(MOV, {Dst(R, 1, 2), Src(R, 2)});
      a.Ins(MOV, {Dst(R, 3, 1), Src(R, 1)});
      a.Ins(TEXKILL, {Dst(R, 3)});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 2)});
    });
    Check(!branch.kill_reads_texel && !lanes.kill_reads_texel,
          "a branch's clean write clears in it; a mov's components as written (r1.y a texel's, r3.x = r1.x not)");
    // A break's state carried past the loop (the clean write after it
    // only where it did not break).
    const auto broke = ps3([](Asm& a) {
      a.Ins(DEFI, {Dst(I, 0), 2, 0, 0, 0});
      a.Ins(REP, {Src(I, 0)});
      a.Ins(TEX, {Dst(R, 1), Src(V, 0), Src(S, 0)});
      a.Ins(BREAKC, {Src(V, 0, kXXXX), Src(C, 0, kXXXX)}, 1);
      a.Ins(MOV, {Dst(R, 1), Src(C, 1)});
      a.Ins(ENDREP, {});
      a.Ins(TEXKILL, {Dst(R, 1)});
      a.Ins(MOV, {Dst(OC, 0), Src(C, 0)});
    });
    // A call's clean write clears; a conditional call's (callnz) does not;
    // a predicated write does not.
    auto call = [&](bool nz) {
      return ps3([&](Asm& a) {
        a.Ins(TEX, {Dst(R, 0), Src(V, 0), Src(S, 0)});
        if (nz) a.Ins(CALLNZ, {Src(LBL, 0), Src(B, 0)});
        else a.Ins(CALL, {Src(LBL, 0)});
        a.Ins(TEXKILL, {Dst(R, 0)});
        a.Ins(MOV, {Dst(OC, 0), Src(C, 0)});
        a.Ins(RET, {});
        a.Ins(LABEL, {Src(LBL, 0)});
        a.Ins(MOV, {Dst(R, 0), Src(C, 1)});
        a.Ins(RET, {});
      });
    };
    const auto called = call(false), maybe = call(true);
    // A callnz decided by a texel: what follows the call does not depend
    // on it (control comes back).
    const auto texel_call = ps3([](Asm& a) {
      a.Ins(TEX, {Dst(R, 0), Src(V, 0), Src(S, 0)});
      a.Ins(SETP, {Dst(P, 0), Src(R, 0, kWWWW), Src(C, 0, kXXXX)}, 1);
      a.Ins(CALLNZ, {Src(LBL, 0), Src(P, 0, kXXXX)});
      a.Ins(MOV, {Dst(R, 1), Src(C, 1)});
      a.Ins(TEXKILL, {Dst(R, 1)});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
      a.Ins(RET, {});
      a.Ins(LABEL, {Src(LBL, 0)});
      a.Ins(MOV, {Dst(R, 2), Src(C, 2)});
      a.Ins(RET, {});
    });
    Check(!texel_call.kill_reads_texel, "a callnz decided by a texel: what follows the call does not depend on it");
    const auto predicated = ps3([](Asm& a) {
      a.Ins(TEX, {Dst(R, 0), Src(V, 0), Src(S, 0)});
      a.Ins(SETP, {Dst(P, 0), Src(V, 0, kXXXX), Src(C, 0, kXXXX)}, 1);
      a.Ins(MOV, {Dst(R, 0), Src(P, 0), Src(C, 1)}, 0, 0x10000000u);
      a.Ins(TEXKILL, {Dst(R, 0)});
      a.Ins(MOV, {Dst(OC, 0), Src(C, 0)});
    });
    Check(broke.kill_reads_texel && !called.kill_reads_texel && maybe.kill_reads_texel &&
              predicated.kill_reads_texel,
          "a break's state past the loop; a call's clean write clears, a callnz's and a predicated one's do not");
    // A break decided by a texel inside a clean if block: what follows in
    // the loop (and its later iterations) depends on that texel.
    const auto nested_break = ps3([](Asm& a) {
      a.Ins(DEFI, {Dst(I, 0), 2, 0, 0, 0});
      a.Ins(REP, {Src(I, 0)});
      a.Ins(IFC, {Src(V, 0, kXXXX), Src(C, 0, kXXXX)}, 1);
      a.Ins(TEX, {Dst(R, 1), Src(V, 0), Src(S, 0)});
      a.Ins(BREAKC, {Src(R, 1, kWWWW), Src(C, 1, kXXXX)}, 1);
      a.Ins(ENDIF, {});
      a.Ins(MOV, {Dst(R, 2), Src(C, 2)});
      a.Ins(ENDREP, {});
      a.Ins(TEXKILL, {Dst(R, 2)});
      a.Ins(MOV, {Dst(OC, 0), Src(C, 0)});
    });
    // A break of a loop nested in an if block on a texel: no control
    // dependence for what follows the if in the outer loop.
    const auto inner_break = ps3([](Asm& a) {
      a.Ins(DEFI, {Dst(I, 0), 2, 0, 0, 0});
      a.Ins(TEX, {Dst(R, 0), Src(V, 0), Src(S, 0)});
      a.Ins(REP, {Src(I, 0)});
      a.Ins(IFC, {Src(R, 0, kWWWW), Src(C, 0, kXXXX)}, 1);
      a.Ins(REP, {Src(I, 0)});
      a.Ins(BREAK, {});
      a.Ins(ENDREP, {});
      a.Ins(ENDIF, {});
      a.Ins(MOV, {Dst(R, 2), Src(C, 2)});
      a.Ins(ENDREP, {});
      a.Ins(TEXKILL, {Dst(R, 2)});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    });
    Check(nested_break.kill_reads_texel && !inner_break.kill_reads_texel,
          "a texel's break inside a clean if block: what follows depends on it; an inner loop's break: not");
    // Scalar instructions read the swizzle's first component; predication
    // reads the predicate per component written; matrix rows after the
    // second source; a coordinate of vPos and an interpolated input.
    auto scalar = [&](uint32_t swz) {
      return ps3([&](Asm& a) {
        a.Ins(MOV, {Dst(R, 1), Src(C, 0)});
        a.Ins(TEX, {Dst(R, 2), Src(V, 0), Src(S, 0)});
        a.Ins(MOV, {Dst(R, 1, 2), Src(R, 2)});  // r1.y a texel's
        a.Ins(RCP, {Dst(R, 3), Src(R, 1, swz)});
        a.Ins(TEXKILL, {Dst(R, 3)});
        a.Ins(MOV, {Dst(OC, 0), Src(R, 2)});
      });
    };
    const auto pred_lane = ps3([](Asm& a) {
      a.Ins(TEX, {Dst(R, 0), Src(V, 0), Src(S, 0)});
      a.Ins(SETP, {Dst(P, 0, 2), Src(R, 0, kWWWW), Src(C, 0, kXXXX)}, 1);  // p0.y a texel's
      a.Ins(RCP, {Dst(R, 1, 2), Src(P, 0), Src(C, 1, kXXXX)}, 0, 0x10000000u);
      a.Ins(TEXKILL, {Dst(R, 1)});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    });
    const auto rows = ps3([](Asm& a) {
      a.Ins(MOV, {Dst(R, 4), Src(C, 0)});
      a.Ins(TEX, {Dst(R, 5), Src(V, 0), Src(S, 0)});
      a.Ins(M3X3, {Dst(R, 1, 7), Src(C, 1), Src(R, 4)});
      a.Ins(TEXKILL, {Dst(R, 1)});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 5)});
    });
    const auto mixed = ps3([](Asm& a) {
      a.Ins(MAD, {Dst(R, 0, 3), Src(MISC, 0), Src(C, 3), Src(V, 0)});
      a.Ins(MOV, {Dst(R, 0, 0xC), Src(C, 6)});
      a.Ins(TEXLDL, {Dst(R, 0), Src(R, 0), Src(S, 0)});
      a.Ins(TEXKILL, {Dst(R, 0)});
      a.Ins(MOV, {Dst(OC, 0), Src(C, 0)});
    });
    Check(scalar(kYYYY).kill_reads_texel && !scalar(kXXXX).kill_reads_texel && pred_lane.kill_reads_texel &&
              rows.kill_reads_texel && mixed.kill_reads_texel,
          "rcp of a texel's component read, of a clean one not; a predicate per component; m3x3's rows; a "
          "coordinate of vPos plus TEXCOORD0");
    // ps_1_1 texcoord t0; texkill t0 (a clip plane): no texel read.
    {
      Asm a(true, 1, 1);
      a.Ins(TEXCOORD, {Dst(T, 0)});
      a.Ins(TEXKILL, {Dst(T, 0)});
      a.Ins(MOV, {Dst(R, 0), Src(C, 0)});
      const auto plane = Translate(a.End()).pixel_alpha;
      Check(plane.kills == 1 && !plane.kill.known && !plane.kill_reads_texel,
            "ps_1_1 texcoord t0; texkill t0: no texel read (uncut)");
    }
    // ps_1_1 co-issue: the second reads r0 before the first clears it.
    {
      Asm a(true, 1, 1);
      a.Ins(TEX, {Dst(T, 0)});
      a.Ins(MOV, {Dst(R, 0), Src(T, 0)});
      a.Ins(MOV, {Dst(R, 0, 7), Src(C, 0)});
      a.Ins(MOV, {Dst(R, 1, 7), Src(R, 0)}, 0, 0x40000000u);
      a.Ins(TEXKILL, {Dst(R, 1)});
      const auto co = Translate(a.End()).pixel_alpha;
      Check(co.kill_reads_texel, "ps_1_1 co-issue: the pair reads its sources before either writes");
    }
  }
}

void PixelChecks() {
  // texld s0 at t0, its alpha straight to oC0.a.
  {
    const Program p = Ps20([](Asm& a) {
      a.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    });
    const AlphaValue& c = p.pixel_alpha.color0;
    Check(Texel(c, 0, 3, 0, 0, 1) && c.color < 0 && c.constant < 0 && c.scale == 1.0f && c.offset == 0.0f &&
              c.offset_constant < 0 && p.pixel_alpha.kills == 0 && !p.pixel_alpha.kill.known,
          "ps_2_0: texld s0 at t0, its alpha to oC0.a");
  }
  // Times v0.a (COLOR0), saturated into oC0.
  {
    const Program p = Ps20([](Asm& a) {
      a.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
      a.Ins(MUL, {Dst(R, 0, 8), Src(R, 0, kWWWW), Src(V, 0, kWWWW)});
      a.Ins(MOV, {Dst(OC, 0, 0xF, 1), Src(R, 0)});
    });
    const AlphaValue& c = p.pixel_alpha.color0;
    Check(Texel(c, 0, 3, 0, 0, 1) && c.color == 0 && c.color_lane == 3 && c.scale == 1.0f,
          "ps_2_0: the texel's alpha times v0.a (COLOR0), saturated into oC0");
  }
  // ps_3_0: TEXCOORD1's zw through a temporary, s1's red replicated, times
  // c5.y and a number the shader defines, COLOR0 declared but not used.
  {
    Asm a(true, 3, 0);
    a.Def(7, 0.5f, 2.0f, 0, 0);
    a.Dcl(5, 1, Dst(V, 0));
    a.Dcl(10, 0, Dst(V, 1));
    a.DclSampler(2, 1);
    a.Ins(MOV, {Dst(R, 1, 3), Src(V, 0, kZWZW)});
    a.Ins(TEX, {Dst(R, 2), Src(R, 1), Src(S, 1)});
    a.Ins(MOV, {Dst(R, 3), Src(R, 2, kXXXX)});
    a.Ins(MUL, {Dst(R, 4, 8), Src(R, 3, kXXXX), Src(C, 5, kYYYY)});
    a.Ins(MUL, {Dst(R, 4, 8), Src(R, 4, kWWWW), Src(C, 7, kYYYY)});
    a.Ins(MOV, {Dst(OC, 0, 8), Src(R, 4, kWWWW)});
    a.Ins(MOV, {Dst(OC, 0, 7), Src(V, 1)});
    const Program p = Translate(a.End());
    const AlphaValue& c = p.pixel_alpha.color0;
    Check(Texel(c, 1, 0, 1, 2, 3) && c.color < 0 && c.constant == 5 && c.constant_lane == 1 && c.scale == 2.0f,
          "ps_3_0: through temporaries, TEXCOORD1.zw, s1's red, times c5.y and a defined 2");
  }
  // texkill of alpha - 0.5 (a defined constant), alpha test value as well.
  {
    const Program p = Ps20([](Asm& a) {
      a.Def(0, 0.5f, 0, 0, 0);
      a.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
      a.Ins(ADD, {Dst(R, 1), Src(R, 0, kWWWW), Src(C, 0, kXXXX, 1)});
      a.Ins(TEXKILL, {Dst(R, 1)});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    });
    const AlphaValue& k = p.pixel_alpha.kill;
    Check(p.pixel_alpha.kills == 1 && Texel(k, 0, 3, 0, 0, 1) && k.scale == 1.0f && k.offset == -0.5f &&
              k.offset_constant < 0 && Texel(p.pixel_alpha.color0, 0, 3, 0, 0, 1),
          "ps_2_0: texkill of the alpha less a defined 0.5");
  }
  // texkill of alpha * v0.a - c3.z (a register the application sets).
  {
    const Program p = Ps20([](Asm& a) {
      a.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
      a.Ins(MAD, {Dst(R, 1), Src(R, 0, kWWWW), Src(V, 0, kWWWW), Src(C, 3, 0xAA, 1)});
      a.Ins(TEXKILL, {Dst(R, 1)});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    });
    const AlphaValue& k = p.pixel_alpha.kill;
    Check(Texel(k, 0, 3, 0, 0, 1) && k.color == 0 && k.offset == 0.0f && k.offset_constant == 3 &&
              k.offset_lane == 2 && k.offset_k == -1.0f,
          "ps_2_0: texkill of the alpha times v0.a less c3.z");
  }
  // ps_1_1: tex t0, r0.a = t0.a * v0.a co-issued with the colour.
  {
    Asm a(true, 1, 1);
    a.Ins(TEX, {Dst(T, 0)});
    a.Ins(MUL, {Dst(R, 0, 7), Src(T, 0), Src(V, 0)});
    a.Ins(MUL, {Dst(R, 0, 8), Src(T, 0, kWWWW), Src(V, 0, kWWWW)}, 0, 0x40000000u);
    const Program p = Translate(a.End());
    const AlphaValue& c = p.pixel_alpha.color0;
    Check(Texel(c, 0, 3, 0, 0, 1) && c.color == 0 && c.color_lane == 3, "ps_1_1: tex t0, r0.a = t0.a * v0.a (co-issued)");
  }
  // ps_1_1: texkill t0 tests the texel's r, g and b: unlike components.
  {
    Asm a(true, 1, 1);
    a.Ins(TEX, {Dst(T, 0)});
    a.Ins(TEXKILL, {Dst(T, 0)});
    a.Ins(MOV, {Dst(R, 0), Src(T, 0)});
    const Program p = Translate(a.End());
    Check(p.pixel_alpha.kills == 1 && !p.pixel_alpha.kill.known && Texel(p.pixel_alpha.color0, 0, 3, 0, 0, 1),
          "ps_1_1: texkill of a texel's colour (unlike components): not known");
  }

  // Cases that must stay unknown.
  struct Unknown {
    const char* what;
    Program p;
  };
  const Unknown cases[] = {
      {"texldp", Ps20([](Asm& a) {
         a.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)}, 1);
         a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
       })},
      {"a cube sampler", Ps20([](Asm& a) {
         a.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 1)});
         a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
       })},
      {"dp4 into the alpha", Ps20([](Asm& a) {
         a.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
         a.Ins(DP4, {Dst(R, 0, 8), Src(R, 0), Src(V, 0)});
         a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
       })},
      {"a texel plus a texel", Ps20([](Asm& a) {
         a.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
         a.Ins(MUL, {Dst(R, 0), Src(R, 0), Src(R, 0)});
         a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
       })},
      {"the alpha written in an if block", Ps20([](Asm& a) {
         a.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
         a.Ins(IFC, {Src(V, 0, kXXXX), Src(C, 0, kXXXX)}, 1);
         a.Ins(MOV, {Dst(R, 0, 8), Src(C, 1)});
         a.Ins(ENDIF, {});
         a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
       })},
      {"a loop", Ps20([](Asm& a) {
         a.Ins(DEFI, {Dst(I, 0), 2, 0, 0, 0});
         a.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
         a.Ins(REP, {Src(I, 0)});
         a.Ins(MUL, {Dst(R, 0), Src(R, 0), Src(V, 0)});
         a.Ins(ENDREP, {});
         a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
       })},
      {"the coordinate scaled", Ps20([](Asm& a) {
         a.Ins(MUL, {Dst(R, 1), Src(T, 0), Src(C, 0)});
         a.Ins(TEX, {Dst(R, 0), Src(R, 1), Src(S, 0)});
         a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
       })},
  };
  bool ok = true;
  for (const Unknown& u : cases) {
    if (u.p.pixel_alpha.color0.known) std::printf("  (%s: known)\n", u.what);
    ok = ok && !u.p.pixel_alpha.color0.known;
  }
  Check(ok, "not known: texldp, a cube sampler, dp4, a texel times a texel, an if block, a loop, scaled coordinates");
  // Two texkills; a texkill in an if block; one of components unlike.
  {
    const Program two = Ps20([](Asm& a) {
      a.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
      a.Ins(TEXKILL, {Dst(R, 0)});
      a.Ins(TEXKILL, {Dst(R, 0)});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    });
    const Program in_if = Ps20([](Asm& a) {
      a.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
      a.Ins(MOV, {Dst(R, 1), Src(R, 0, kWWWW)});
      a.Ins(IFC, {Src(V, 0, kXXXX), Src(C, 0, kXXXX)}, 1);
      a.Ins(TEXKILL, {Dst(R, 1)});
      a.Ins(ENDIF, {});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    });
    const Program unlike = Ps20([](Asm& a) {
      a.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
      a.Ins(TEXKILL, {Dst(R, 0)});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    });
    const Program numbers = Ps20([](Asm& a) {
      a.Def(0, 0.25f, 0, 0, 0);
      a.Ins(TEX, {Dst(R, 0), Src(T, 0), Src(S, 0)});
      a.Ins(MOV, {Dst(R, 1), Src(C, 0, kXXXX)});
      a.Ins(MOV, {Dst(R, 1, 8), Src(R, 0, kWWWW)});
      a.Ins(TEXKILL, {Dst(R, 1)});
      a.Ins(MOV, {Dst(OC, 0), Src(R, 0)});
    });
    Check(two.pixel_alpha.kills == 2 && !two.pixel_alpha.kill.known && in_if.pixel_alpha.kills == 1 &&
              !in_if.pixel_alpha.kill.known && unlike.pixel_alpha.kills == 1 && !unlike.pixel_alpha.kill.known,
          "texkill not known: two of them, one in an if block, one of a texel's unlike channels");
    Check(Texel(numbers.pixel_alpha.kill, 0, 3, 0, 0, 1) && numbers.pixel_alpha.kill.offset == 0.0f,
          "texkill of (0.25, 0.25, 0.25, alpha): the numbers >= 0 never discard, the alpha is the value");
  }
}

bool Has(const std::vector<MovedLane>& m, uint32_t usage, uint32_t index, int lane, int input, int input_lane) {
  for (const MovedLane& x : m)
    if (x.varying.usage == usage && x.varying.index == index && x.lane == lane)
      return x.input == input && x.input_lane == input_lane;
  return input < 0;
}

void VertexChecks() {
  // vs_2_0: v0 POSITION, v1 TEXCOORD0, v2 COLOR0 (slots 0, 1, 2).
  {
    Asm a(false, 2, 0);
    a.Dcl(0, 0, Dst(V, 0));
    a.Dcl(5, 0, Dst(V, 1));
    a.Dcl(10, 0, Dst(V, 2));
    a.Ins(MOV, {0x80000000u | (4u << 28) | 0xF0000u, Src(V, 0)});  // oPos
    a.Ins(MOV, {Dst(OT, 0), Src(V, 1)});
    a.Ins(MOV, {Dst(R, 0), Src(V, 2)});
    a.Ins(MOV, {Dst(OD, 0), Src(R, 0)});
    a.Ins(MOV, {Dst(OT, 1, 3), Src(V, 1, kZWZW)});
    a.Ins(MUL, {Dst(OT, 2), Src(V, 1), Src(C, 0)});
    a.Ins(MOV, {Dst(OT, 3, 3), Src(V, 1, 0xE4, 1)});
    const Program p = Translate(a.End());
    const auto& m = p.moved_lanes;
    Check(Has(m, 5, 0, 0, 1, 0) && Has(m, 5, 0, 1, 1, 1) && Has(m, 5, 0, 3, 1, 3) && Has(m, 10, 0, 3, 2, 3) &&
              Has(m, 5, 1, 0, 1, 2) && Has(m, 5, 1, 1, 1, 3) && Has(m, 5, 1, 2, -1, 0) && Has(m, 5, 2, 0, -1, 0) &&
              Has(m, 5, 3, 0, -1, 0),
          "vs_2_0: oT0 and oD0 (through r0) moved from inputs, oT1.xy from TEXCOORD0.zw; not oT2 (a product), "
          "not oT3 (negated)");
  }
  // vs_3_0: declared outputs; a write in an if block, a loop.
  {
    Asm a(false, 3, 0);
    a.Dcl(0, 0, Dst(V, 0));
    a.Dcl(5, 2, Dst(V, 1));
    a.Dcl(0, 0, Dst(O, 0));
    a.Dcl(5, 0, Dst(O, 1));
    a.Dcl(5, 1, Dst(O, 2));
    a.Ins(MOV, {Dst(O, 0), Src(V, 0)});
    a.Ins(MOV, {Dst(O, 1), Src(V, 1)});
    a.Ins(MOV, {Dst(O, 2), Src(V, 1)});
    a.Ins(IFC, {Src(V, 0, kXXXX), Src(C, 0, kXXXX)}, 1);
    a.Ins(MOV, {Dst(O, 2, 1), Src(V, 0)});
    a.Ins(ENDIF, {});
    const Program p = Translate(a.End());
    const auto& m = p.moved_lanes;
    Check(Has(m, 5, 0, 0, 1, 0) && Has(m, 5, 0, 1, 1, 1) && Has(m, 5, 1, 0, -1, 0) && Has(m, 5, 1, 1, 1, 1),
          "vs_3_0: TEXCOORD0 moved from TEXCOORD2; TEXCOORD1.x written in an if block: not known");
    Asm b(false, 3, 0);
    b.Dcl(0, 0, Dst(V, 0));
    b.Dcl(5, 0, Dst(V, 1));
    b.Dcl(0, 0, Dst(O, 0));
    b.Dcl(5, 0, Dst(O, 1));
    b.Ins(DEFI, {Dst(I, 0), 2, 0, 0, 0});
    b.Ins(MOV, {Dst(O, 1), Src(V, 1)});
    b.Ins(REP, {Src(I, 0)});
    b.Ins(MOV, {Dst(O, 0), Src(V, 0)});
    b.Ins(ENDREP, {});
    Check(Translate(b.End()).moved_lanes.empty(), "vs_3_0 with a loop: nothing known");
  }
}

}  // namespace

int main() {
  PixelChecks();
  MoreChecks();
  VertexChecks();
  return failures ? 1 : 0;
}
