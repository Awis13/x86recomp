// Compiled Direct3D 9 shaders (shader model 1.1-3.0 tokens) translated to
// the GLSL contract of hlsl.h, for games that ship their shaders compiled.
//
// Registers become GLSL variables: temps r_t[], inputs r_v[] (filled from
// r_fetch or the varyings), address a0 (r_a0), loop counter aL (r_aL),
// predicate p0 (r_p0), outputs per kind; constants read the register file
// c[] unless the shader defines them (def/defi/defb), and relative reads
// go through r_crel, which honours the definitions too. Every instruction
// computes a vec4 from its swizzled, modified sources, then applies the
// shift and saturate result modifiers, the write mask and predication.
// Co-issued pairs (ps_1_x) read their sources before either writes.
// Subroutines (label/call) become GLSL functions; flow control (rep, loop,
// if, break) maps onto GLSL's.
//
// The scalar instructions (rcp, rsq, exp, log, pow, ...) take the first
// component of the swizzled source, which compilers always replicate.
// As in the HLSL path, rsq, log and pow take |x| as the instructions do.
// ps_1_x: t# registers hold tex / texcoord results from TEXCOORD#, samplers
// are 2D (the stage's texture), r0 is the colour. The ps_1_x texture-matrix
// and bump instructions and ps_1_4 phases are not supported (named).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "runtime/hlsl/hlsl.h"

namespace recomp::hlsl {
namespace {

enum RegType : uint32_t {
  TEMP = 0, INPUT = 1, CONST = 2, ADDR = 3, TEXTURE = 3, RASTOUT = 4, ATTROUT = 5,
  TEXCRDOUT = 6, OUTPUT = 6, CONSTINT = 7, COLOROUT = 8, DEPTHOUT = 9, SAMPLER = 10,
  CONST2 = 11, CONST3 = 12, CONST4 = 13, CONSTBOOL = 14, LOOP = 15, TEMPFLOAT16 = 16,
  MISCTYPE = 17, LABEL = 18, PREDICATE = 19,
};

enum Op : uint32_t {
  NOP = 0, MOV, ADD, SUB, MAD, MUL, RCP, RSQ, DP3, DP4, MIN, MAX, SLT, SGE, EXP, LOG, LIT, DST,
  LRP, FRC, M4X4, M4X3, M3X4, M3X3, M3X2, CALL, CALLNZ, LOOP_, RET, ENDLOOP, LABEL_, DCL, POW,
  CRS, SGN, ABS, NRM, SINCOS, REP, ENDREP, IF, IFC, ELSE, ENDIF, BREAK, BREAKC, MOVA, DEFB, DEFI,
  TEXCOORD = 64, TEXKILL, TEX, TEXBEM, TEXBEML, TEXREG2AR, TEXREG2GB, TEXM3X2PAD, TEXM3X2TEX,
  TEXM3X3PAD, TEXM3X3TEX, TEXM3X3SPEC = 76, TEXM3X3VSPEC, EXPP, LOGP, CND, DEF, TEXREG2RGB,
  TEXDP3TEX, TEXM3X2DEPTH, TEXDP3, TEXM3X3, TEXDEPTH, CMP, BEM, DP2ADD, DSX, DSY, TEXLDD, SETP,
  TEXLDL, BREAKP, PHASE = 0xFFFD, COMMENT = 0xFFFE, END = 0xFFFF,
};

struct Error {
  std::string msg;
};
[[noreturn]] void Fail(std::string msg) { throw Error{std::move(msg)}; }

uint32_t RegTypeOf(uint32_t tok) { return ((tok >> 28) & 7) | ((tok >> 8) & 0x18); }

struct Param {
  uint32_t type = 0, num = 0;
  uint32_t mask = 0xF;     // destination
  uint32_t swz = 0xE4;     // source: 2 bits per component (xyzw = 0xE4)
  uint32_t mod = 0;        // source modifier / destination result modifier
  int shift = 0;           // destination shift scale
  bool rel = false;
  uint32_t rel_type = ADDR, rel_num = 0, rel_comp = 0;  // relative address
};

struct Ins {
  uint32_t op = 0, ctl = 0;
  bool coissue = false, predicated = false, has_dst = false;
  Param dst, pred;
  std::vector<Param> src;
  uint32_t raw[4] = {};  // def/defi/defb values, dcl usage token
};

bool HasDst(uint32_t op) {
  switch (op) {
    case NOP: case CALL: case CALLNZ: case LOOP_: case RET: case ENDLOOP: case LABEL_: case REP:
    case ENDREP: case IF: case IFC: case ELSE: case ENDIF: case BREAK: case BREAKC: case BREAKP:
      return false;
    default:
      return true;
  }
}

const char* kComp = "xyzw";

std::string FloatBitsExpr(uint32_t bits) {
  char buf[48];
  std::snprintf(buf, sizeof buf, "uintBitsToFloat(0x%08Xu)", bits);
  return buf;
}

class Translator {
 public:
  Translator(const uint32_t* t, size_t n, Program* out, const TranslateOptions& o)
      : t_(t), n_(n), out_(out), coverage_(o.coverage), rt_call_(o.ray_reflection), rt_shadow_(o.ray_shadow) {}

  void Run() {
    if (n_ < 2) Fail("shader too short");
    uint32_t v = t_[0];
    pixel_ = (v >> 16) == 0xFFFF;
    if (!pixel_ && (v >> 16) != 0xFFFE) Fail("not a shader version token");
    major_ = (v >> 8) & 0xFF;
    minor_ = v & 0xFF;
    if (major_ < 1 || major_ > 3) Fail("unsupported shader model");
    if (pixel_ && major_ == 1 && minor_ >= 4) Fail("ps_1_4 is not supported");
    out_->stage = pixel_ ? Stage::Pixel : Stage::Vertex;
    out_->major = major_;
    out_->minor = minor_;
    Decode();
    Declare();
    if (!pixel_) out_->position_matrix = PositionMatrix(&out_->position_note);
    if (coverage_ && ins_.size() <= kCoverageInstructions) {
      if (pixel_) out_->pixel_alpha = PixelAlphaOf();
      else out_->moved_lanes = MovedLanes();
    } else if (coverage_ && pixel_) {
      // Past the limit only what discards may read (bounded as well).
      for (const Ins& in : ins_) out_->pixel_alpha.kills += in.op == TEXKILL ? 1 : 0;
      if (out_->pixel_alpha.kills) out_->pixel_alpha.kill_reads_texel = KillReadsTexel();
    }
    out_->coverage_skipped = !coverage_;
    Emit();
  }

 private:
  // ---- decoding

  Param Dst(size_t& i) {
    uint32_t tok = t_[i++];
    Param p;
    p.type = RegTypeOf(tok);
    p.num = tok & 0x7FF;
    p.mask = (tok >> 16) & 0xF;
    p.mod = (tok >> 20) & 0xF;
    int s = int((tok >> 24) & 0xF);
    p.shift = s >= 8 ? s - 16 : s;
    p.rel = tok & 0x2000;
    if (p.rel && major_ >= 3) RelToken(i, &p);
    return p;
  }

  Param Src(size_t& i) {
    uint32_t tok = t_[i++];
    Param p;
    p.type = RegTypeOf(tok);
    p.num = tok & 0x7FF;
    p.swz = (tok >> 16) & 0xFF;
    p.mod = (tok >> 24) & 0xF;
    p.rel = tok & 0x2000;
    if (p.rel) {
      if (major_ >= 2) {
        RelToken(i, &p);
      } else {
        p.rel_type = ADDR;  // vs_1_1: a0.x
        p.rel_num = 0;
        p.rel_comp = 0;
      }
    }
    return p;
  }

  void RelToken(size_t& i, Param* p) {
    if (i >= n_) Fail("truncated relative address");
    uint32_t tok = t_[i++];
    p->rel_type = RegTypeOf(tok);
    p->rel_num = tok & 0x7FF;
    p->rel_comp = (tok >> 16) & 3;
  }

  void Decode() {
    size_t i = 1;
    while (i < n_) {
      uint32_t tok = t_[i];
      uint32_t op = tok & 0xFFFF;
      if (op == END) return;
      if (op == COMMENT) {
        i += 1 + ((tok >> 16) & 0x7FFF);
        continue;
      }
      Ins in;
      in.op = op;
      in.ctl = (tok >> 16) & 0xFF;
      in.coissue = tok & 0x40000000;
      in.predicated = tok & 0x10000000;
      size_t start = ++i, end;
      if (major_ >= 2) {
        end = start + ((tok >> 24) & 0xF);
      } else if (op == DEF || op == DEFI) {
        end = start + 5;
      } else if (op == DEFB) {
        end = start + 2;
      } else {
        end = start;
        while (end < n_ && (t_[end] & 0x80000000)) end++;
      }
      if (end > n_) Fail("truncated instruction");
      if (op == DEF || op == DEFI || op == DEFB) {
        in.has_dst = true;
        in.dst = Dst(i);
        for (int k = 0; k < (op == DEFB ? 1 : 4); k++) in.raw[k] = t_[i++];
      } else if (op == DCL) {
        in.raw[0] = t_[i++];
        in.has_dst = true;
        in.dst = Dst(i);
      } else {
        if (HasDst(op)) {
          in.has_dst = true;
          in.dst = Dst(i);
        }
        if (in.predicated) in.pred = Src(i);
        while (i < end) in.src.push_back(Src(i));
      }
      if (i != end) Fail("bad instruction length for opcode " + std::to_string(op));
      ins_.push_back(std::move(in));
    }
    Fail("shader without END");
  }

  // ---- the position's matrix (Program::position_matrix)

  // What a temporary's z and w hold: the sum over `rows` of a scalar times
  // c[k].z (lane z) and c[k].w (lane w), the same scalar in both, plus
  // c[trans].zw once (-1: none). `known` false: anything else, `why` says
  // what (Program::position_note).
  struct ZwSum {
    bool known = false;
    std::set<uint32_t> rows;
    int trans = -1;
    const char* why = "";
  };
  static ZwSum Unknown(const char* why) {
    ZwSum r;
    r.why = why;
    return r;
  }

  // The b of Program::position_matrix, or -1 and why not. Followed
  // through mov, mul, mad and add in straight-line code; a write in an if
  // block may or may not happen (unknown after it); loops and calls are
  // not followed.
  int PositionMatrix(std::string* why) const {
    std::map<uint32_t, ZwSum> temps;
    ZwSum pos;
    int writes = 0, depth = 0;
    auto is_pos = [&](const Param& p) {
      if (major_ < 3) return p.type == RASTOUT && p.num == 0;
      auto it = vs3_outputs_.find(p.num);
      return p.type == OUTPUT && it != vs3_outputs_.end() && it->second.usage == kPosition &&
             it->second.index == 0;
    };
    // Lanes z and w read .z and .w; read one component (a scalar).
    auto zw = [](const Param& p) { return ((p.swz >> 4) & 3) == 2 && ((p.swz >> 6) & 3) == 3; };
    auto scalar = [](const Param& p) { return ((p.swz >> 4) & 3) == ((p.swz >> 6) & 3) && p.mod <= 1; };
    // A register the application sets (a def in the shader overrides it).
    auto app = [&](const Param& p) { return p.type == CONST && !p.rel && !defs_.count(p.num); };
    // Why a row or a translation read as `p` is none.
    auto bad = [&](const Param& p) -> const char* {
      if (p.rel) return "a register read through a0 or aL";
      if (p.mod > 1) return "a source modifier other than negation";
      if (!zw(p)) return "z and w not read from .z and .w";
      if (p.type == CONST && defs_.count(p.num)) return "a constant the shader defines";
      return nullptr;
    };
    // A source added as it is: the translation only unnegated.
    auto term = [&](const Param& p) -> ZwSum {
      if (const char* w = bad(p)) return Unknown(w);
      if (app(p)) {
        if (p.mod) return Unknown("the translation negated");
        ZwSum r;
        r.known = true;
        r.trans = int(p.num);
        return r;
      }
      auto it = p.type == TEMP ? temps.find(p.num) : temps.end();
      if (it == temps.end()) return Unknown("a term not made from the float registers (an input, a temporary set otherwise)");
      if (p.mod && it->second.trans >= 0) return Unknown("the translation negated");
      return it->second;
    };
    // A scalar times a row, or times a sum without the translation.
    auto prod = [&](const Param& a, const Param& b) -> ZwSum {
      for (int k = 0; k < 2; k++) {
        const Param& f = k ? a : b;
        if (!scalar(k ? b : a) || bad(f)) continue;
        if (app(f)) {
          ZwSum r;
          r.known = true;
          r.rows.insert(f.num);
          return r;
        }
        auto it = f.type == TEMP ? temps.find(f.num) : temps.end();
        if (it != temps.end() && it->second.known && it->second.trans < 0) return it->second;
      }
      return Unknown("a product that is not a scalar times a row");
    };
    auto sum = [](ZwSum a, const ZwSum& b) -> ZwSum {
      if (!a.known) return a;
      if (!b.known) return b;
      if (a.trans >= 0 && b.trans >= 0) return Unknown("the translation added twice");
      a.rows.insert(b.rows.begin(), b.rows.end());
      if (b.trans >= 0) a.trans = b.trans;
      return a;
    };
    for (const Ins& in : ins_) {
      if (in.op == IF || in.op == IFC) {
        depth++;
        continue;
      }
      if (in.op == ENDIF) {
        depth--;
        continue;
      }
      if (in.op == RET && depth == 0) break;  // the end of main (subroutines follow)
      if (in.op == CALL || in.op == CALLNZ || in.op == LABEL_) {
        *why = "a subroutine (not followed)";
        return -1;
      }
      if (in.op == LOOP_ || in.op == REP || in.op == BREAK || in.op == BREAKC || in.op == BREAKP) {
        *why = "a loop (not followed)";
        return -1;
      }
      if (!in.has_dst || in.op == DCL || in.op == DEF || in.op == DEFI || in.op == DEFB) continue;
      ZwSum v;
      const std::vector<Param>& a = in.src;
      if (depth > 0) v = Unknown("written in an if block");
      else if (in.predicated) v = Unknown("a predicated write");
      else if ((in.dst.mod & 1) || in.dst.shift != 0) v = Unknown("a saturated or shifted write");
      else if (in.op == MOV && a.size() == 1) v = term(a[0]);
      else if (in.op == ADD && a.size() == 2) v = sum(term(a[0]), term(a[1]));
      else if (in.op == MUL && a.size() == 2) v = prod(a[0], a[1]);
      else if (in.op == MAD && a.size() == 3) v = sum(prod(a[0], a[1]), term(a[2]));
      else v = Unknown("an instruction other than mov, add, mul and mad (dp4, m4x4: a column-major matrix)");
      const uint32_t m = in.dst.mask & 0xC;
      if (!m) continue;
      if (m != 0xC) v = Unknown("z and w written apart");
      if (is_pos(in.dst)) {
        writes++;
        pos = v;
      } else if (in.dst.type == TEMP) {
        temps[in.dst.num] = v;
      }
    }
    if (writes != 1) {
      *why = writes ? "z and w written more than once" : "z and w not written";
      return -1;
    }
    if (!pos.known) {
      *why = pos.why;
      return -1;
    }
    if (pos.trans < 3) {
      *why = "no translation added";
      return -1;
    }
    const uint32_t b = uint32_t(pos.trans) - 3;
    for (uint32_t r : pos.rows)
      if (r < b || r > b + 2) {
        *why = "a row c" + std::to_string(r) + " not with the translation c" + std::to_string(pos.trans);
        return -1;
      }
    return int(b);
  }

  // ---- the pixel's alpha (Program::pixel_alpha)

  // What a component of a register holds, as far as PixelAlphaOf follows
  // it: not known, a varying's lane as interpolated, or an AlphaValue.
  struct Sym {
    enum Kind : uint8_t { kUnknown, kVarying, kValue } kind = kUnknown;
    Semantic var;
    uint8_t lane = 0;
    AlphaValue v;
  };
  using Lanes = std::array<Sym, 4>;

  static Sym Literal(float x) {
    Sym s;
    s.kind = Sym::kValue;
    s.v.known = true;
    s.v.offset = x;
    return s;
  }
  // A number alone: no factor, no register in the offset.
  static bool IsLiteral(const AlphaValue& v) {
    return v.sampler < 0 && v.color < 0 && v.constant < 0 && v.offset_constant < 0;
  }
  // A COLOR varying's lane as a factor; other varyings are none.
  static Sym AsValue(const Sym& s) {
    if (s.kind != Sym::kVarying) return s;
    if (s.var.usage != kColor) return Sym{};
    Sym r;
    r.kind = Sym::kValue;
    r.v.known = true;
    r.v.color = int(s.var.index);
    r.v.color_lane = s.lane;
    r.v.scale = 1.0f;
    return r;
  }
  // m * s + c.
  static Sym Affine(Sym s, float m, float c) {
    s = AsValue(s);
    if (s.kind != Sym::kValue) return Sym{};
    s.v.scale *= m;
    s.v.offset = s.v.offset * m + c;
    s.v.offset_k = s.v.offset_constant >= 0 ? s.v.offset_k * m : 0.0f;
    return s;
  }
  // a + b: one of them a number, or a constant register's lane (times a
  // number, plus one) that goes into the other's offset.
  static Sym Sum(Sym a, Sym b) {
    a = AsValue(a);
    b = AsValue(b);
    if (a.kind != Sym::kValue || b.kind != Sym::kValue) return Sym{};
    for (int k = 0; k < 2; k++) {
      Sym x = k ? b : a;
      const AlphaValue& y = (k ? a : b).v;
      if (IsLiteral(y)) {
        x.v.offset += y.offset;
        return x;
      }
      if (y.sampler < 0 && y.color < 0 && y.constant >= 0 && y.offset_constant < 0 && x.v.offset_constant < 0) {
        x.v.offset_constant = y.constant;
        x.v.offset_lane = y.constant_lane;
        x.v.offset_k = y.scale;
        x.v.offset += y.offset;
        return x;
      }
    }
    return Sym{};
  }
  // a * b: one of them a number, or both products without offsets, each
  // kind of factor from one side at most.
  static Sym Product(Sym a, Sym b) {
    a = AsValue(a);
    b = AsValue(b);
    if (a.kind != Sym::kValue || b.kind != Sym::kValue) return Sym{};
    if (IsLiteral(a.v)) return Affine(b, a.v.offset, 0.0f);
    if (IsLiteral(b.v)) return Affine(a, b.v.offset, 0.0f);
    const AlphaValue &x = a.v, &y = b.v;
    if (x.offset != 0.0f || y.offset != 0.0f || x.offset_constant >= 0 || y.offset_constant >= 0) return Sym{};
    if ((x.sampler >= 0 && y.sampler >= 0) || (x.color >= 0 && y.color >= 0) || (x.constant >= 0 && y.constant >= 0))
      return Sym{};
    Sym r = a;
    if (y.sampler >= 0) {
      r.v.sampler = y.sampler;
      r.v.channel = y.channel;
      r.v.texcoord = y.texcoord;
      r.v.u = y.u;
      r.v.v = y.v;
    }
    if (y.color >= 0) {
      r.v.color = y.color;
      r.v.color_lane = y.color_lane;
    }
    if (y.constant >= 0) {
      r.v.constant = y.constant;
      r.v.constant_lane = y.constant_lane;
    }
    r.v.scale = x.scale * y.scale;
    return r;
  }

  // Component `lane` of source `p`, its swizzle and modifier applied:
  // temporaries from `temps`, ps_1_x's texture registers from `tx`.
  Sym SrcSym(const Param& p, int lane, const std::map<uint32_t, Lanes>& temps,
             const std::map<uint32_t, Lanes>& tx) const {
    if (p.rel) return Sym{};
    const uint8_t c = uint8_t((p.swz >> (2 * lane)) & 3);
    Sym s;
    switch (p.type) {
      case TEMP: {
        auto it = temps.find(p.num);
        s = it != temps.end() ? it->second[c] : Literal(0.0f);  // (temporaries start at 0)
        break;
      }
      case INPUT:
        if (major_ >= 3) {
          auto it = ps3_inputs_.find(p.num);
          if (it == ps3_inputs_.end() || it->second.first.usage == kPosition) return Sym{};
          s.var = it->second.first;
        } else {
          s.var = {kColor, p.num};
        }
        s.kind = Sym::kVarying;
        s.lane = c;
        break;
      case TEXTURE:
        if (major_ == 1) {
          auto it = tx.find(p.num);
          if (it == tx.end()) return Sym{};
          s = it->second[c];
        } else {
          s.kind = Sym::kVarying;
          s.var = {kTexcoord, p.num};
          s.lane = c;
        }
        break;
      case CONST: {
        auto it = defs_.find(p.num);
        if (it != defs_.end()) {
          float f;
          std::memcpy(&f, &it->second[c], 4);
          s = Literal(f);
        } else if (p.num < 256) {
          s.kind = Sym::kValue;
          s.v.known = true;
          s.v.constant = int(p.num);
          s.v.constant_lane = c;
          s.v.scale = 1.0f;
        } else {
          return Sym{};
        }
        break;
      }
      default:
        return Sym{};
    }
    switch (p.mod) {
      case 0: return s;
      case 1: return Affine(s, -1.0f, 0.0f);
      case 2: return Affine(s, 1.0f, -0.5f);
      case 3: return Affine(s, -1.0f, 0.5f);
      case 4: return Affine(s, 2.0f, -1.0f);
      case 5: return Affine(s, -2.0f, 1.0f);
      case 6: return Affine(s, -1.0f, 1.0f);
      case 7: return Affine(s, 2.0f, 0.0f);
      case 8: return Affine(s, -2.0f, 0.0f);
      case 11:
        if (s.kind == Sym::kValue && IsLiteral(s.v)) return Literal(std::fabs(s.v.offset));
        return Sym{};
      default: return Sym{};
    }
  }

  // A texture sample's components: `coord`'s first two components must be
  // lanes of one TEXCOORD varying, the sampler 2D.
  Lanes Texel(uint32_t sampler, uint32_t sampler_swz, const Sym& cu, const Sym& cv) const {
    Lanes r{};
    if (sampler_swz != 0xE4) return r;  // (the translation does not swizzle it)
    auto st = sampler_type_.find(sampler);
    if (st != sampler_type_.end() && st->second != ParamType::Sampler2D) return r;
    if (cu.kind != Sym::kVarying || cv.kind != Sym::kVarying || cu.var.usage != kTexcoord || !(cu.var == cv.var))
      return r;
    for (int l = 0; l < 4; l++) {
      Sym& s = r[l];
      s.kind = Sym::kValue;
      s.v.known = true;
      s.v.sampler = int(sampler);
      s.v.channel = uint8_t((sampler_swz >> (2 * l)) & 3);
      s.v.texcoord = cu.var.index;
      s.v.u = cu.lane;
      s.v.v = cv.lane;
      s.v.scale = 1.0f;
    }
    return r;
  }

  // ---- what a texkill may read (PixelAlpha::kill_reads_texel)

  // Per register (type << 16 | number), the components that may hold
  // something computed from a texel (`tex`) and from an interpolated input
  // (`var`): a sample taints only where its coordinate has either, so a
  // texture read at the screen position (a dither) is not a texel's.
  struct Lanes2 {
    std::map<uint32_t, uint8_t> lanes;
    bool operator==(const Lanes2&) const = default;
    void Join(const Lanes2& o) {
      for (const auto& [k, m] : o.lanes) lanes[k] |= m;
    }
    void Set(uint32_t key, uint8_t mask, bool on) {
      if (on) {
        lanes[key] |= mask;
        return;
      }
      auto it = lanes.find(key);
      if (it == lanes.end()) return;
      it->second &= uint8_t(~mask);
      if (!it->second) lanes.erase(it);
    }
  };
  struct Taint {
    Lanes2 tex, var;
    bool operator==(const Taint&) const = default;
    void Join(const Taint& o) {
      tex.Join(o.tex);
      var.Join(o.var);
    }
  };
  static uint32_t TaintKey(const Param& p) { return p.type << 16 | p.num; }
  // The components of source `p` an instruction `op` writing `dst_mask`
  // reads: per component, the swizzle's for the components written; the
  // scalar instructions, the first; others, all four the swizzle names.
  static uint8_t ReadMask(uint32_t op, const Param& p, uint32_t dst_mask) {
    uint32_t lanes = 0xF;
    switch (op) {
      case MOV: case ADD: case SUB: case MAD: case MUL: case MIN: case MAX: case SLT: case SGE: case FRC:
      case ABS: case CMP: case CND: case LRP: case SGN: case MOVA: case SETP: case DSX: case DSY:
        lanes = dst_mask & 0xF;
        break;
      case RCP: case RSQ: case EXP: case LOG: case EXPP: case LOGP: case POW: case IFC: case BREAKC:
        lanes = 1;
        break;
      default: break;
    }
    uint8_t m = 0;
    for (int l = 0; l < 4; l++)
      if (lanes & (1u << l)) m |= uint8_t(1u << ((p.swz >> (2 * l)) & 3));
    return m;
  }
  // Whether the components `m` of source `p` (or the register addressing
  // it) are in `map`.
  static bool In(const Lanes2& map, const Param& p, uint8_t m) {
    if (p.rel) {
      Param r;
      r.type = p.rel_type;
      r.num = p.rel_num;
      if (In(map, r, uint8_t(1u << (p.rel_comp & 3)))) return true;
    }
    auto it = map.lanes.find(TaintKey(p));
    return it != map.lanes.end() && (it->second & m) != 0;
  }
  // Whether source `p` is an interpolated input itself (v#, t# after
  // ps_1_x; not the position, vPos, vFace or constants).
  bool Interpolated(const Param& p) const {
    if (p.type == INPUT) {
      if (major_ < 3) return true;
      auto it = ps3_inputs_.find(p.num);
      return it != ps3_inputs_.end() && it->second.first.usage != kPosition;
    }
    return p.type == TEXTURE && major_ >= 2;
  }
  bool Tex(const Taint& t, uint32_t op, const Param& p, uint32_t dst_mask) const {
    return In(t.tex, p, ReadMask(op, p, dst_mask));
  }
  bool Var(const Taint& t, uint32_t op, const Param& p, uint32_t dst_mask) const {
    return Interpolated(p) || In(t.var, p, ReadMask(op, p, dst_mask));
  }
  static bool Opens(uint32_t op, bool loop) { return loop ? op == REP || op == LOOP_ : op == IF || op == IFC; }
  static bool Closes(uint32_t op, bool loop) { return loop ? op == ENDREP || op == ENDLOOP : op == ENDIF; }
  // The index of the instruction closing the if block (or loop) opened at
  // `at`; ins_.size() if none.
  size_t BlockEnd(size_t at, bool loop) const {
    int depth = 0;
    for (size_t i = at + 1; i < ins_.size(); i++) {
      if (Opens(ins_[i].op, loop)) depth++;
      else if (Closes(ins_[i].op, loop) && depth-- == 0) return i;
    }
    return ins_.size();
  }
  // Whether [from, to) breaks out of the loop it is in (breaks of loops
  // nested in it not counted).
  bool Breaks(size_t from, size_t to) const {
    for (size_t i = from; i < to; i++) {
      if (Opens(ins_[i].op, true)) {
        i = BlockEnd(i, true);
        continue;
      }
      if (ins_[i].op == BREAK || ins_[i].op == BREAKC || ins_[i].op == BREAKP) return true;
    }
    return false;
  }
  // The instructions [from, to) run on taint `t`, in program order: a write
  // whose sources are clean clears what it writes when it surely happens
  // (not `weak`, not predicated), else adds nothing; a write under a
  // condition that may depend on a texel (`ctrl`, a predicate) is a
  // texel's. If blocks run each branch on its own copy, joined after;
  // loops repeat to a fixed point (the maps only grow: it ends); a break
  // leaves its state in `exits`, what follows it runs where it did not
  // break; calls run their subroutine in place (conditional ones weak).
  // Texkills reading a texel's set *kill; a break decided by a texel sets
  // *texel_break (the loop's). Returns `ctrl` as the instructions leave it
  // (a break decided by a texel in them sets it).
  bool RunTaint(size_t from, size_t to, Taint& t, bool weak, bool ctrl, Taint* exits, bool* texel_break, int calls,
                bool* kill) const {
    for (size_t i = from; i < to; i++) {
      const Ins& in = ins_[i];
      switch (in.op) {
        case IF: case IFC: {
          bool cond = ctrl;
          for (const Param& p : in.src) cond = cond || Tex(t, in.op, p, 1);
          const size_t end = BlockEnd(i, false);
          size_t mid = end;
          for (size_t k = i + 1, depth = 0; k < end; k++) {
            if (ins_[k].op == IF || ins_[k].op == IFC) depth++;
            else if (ins_[k].op == ENDIF) depth--;
            else if (ins_[k].op == ELSE && depth == 0) mid = k;
          }
          Taint a = t, b = t;
          const bool ca = RunTaint(i + 1, mid, a, weak, cond, exits, texel_break, calls, kill);
          const bool cb = mid < end && RunTaint(mid + 1, end, b, weak, cond, exits, texel_break, calls, kill);
          a.Join(b);
          t = a;
          // A break in it: what follows runs only where it did not.
          if (Breaks(i + 1, end)) ctrl = ctrl || cond || ca || cb;
          i = end;
          continue;
        }
        case REP: case LOOP_: {
          const size_t end = BlockEnd(i, true);
          Taint head = t, out;
          bool texel = false;  // a break in it decided by a texel
          for (bool all = ctrl;;) {
            for (;;) {
              Taint b = head;
              RunTaint(i + 1, end, b, weak, all, &out, &texel, calls, kill);
              Taint next = head;
              next.Join(b);
              if (next == head) break;
              head = next;
            }
            // Then every later iteration depends on that texel: again, all
            // under it.
            if (!texel || all) break;
            all = true;
          }
          head.Join(out);
          t = head;
          i = end;
          continue;
        }
        case BREAK: case BREAKC: case BREAKP: {
          bool cond = false;
          for (const Param& p : in.src) cond = cond || Tex(t, in.op, p, 1);
          if (exits) exits->Join(t);
          // What follows in the loop runs only where it did not break.
          if (in.op != BREAK) ctrl = ctrl || cond;
          if ((cond || ctrl) && texel_break) *texel_break = true;
          continue;
        }
        case CALL: case CALLNZ: {
          if (in.src.empty() || calls >= 8) continue;
          bool cond = ctrl;
          for (size_t k = 1; k < in.src.size(); k++) cond = cond || Tex(t, in.op, in.src[k], 1);
          for (size_t k = 0; k < ins_.size(); k++)
            if (ins_[k].op == LABEL_ && !ins_[k].src.empty() && ins_[k].src[0].num == in.src[0].num) {
              size_t ret = k + 1;
              while (ret < ins_.size() && ins_[ret].op != RET && ins_[ret].op != LABEL_) ret++;
              // (Its ctrl not kept: control comes back after the call, a
              // break cannot leave a subroutine, which the translation makes
              // a GLSL function.)
              RunTaint(k + 1, ret, t, weak || in.op == CALLNZ, cond, exits, texel_break, calls + 1, kill);
              break;
            }
          continue;
        }
        case RET: case LABEL_:
          return ctrl;  // (main's end, or a subroutine's)
        default: break;
      }
      if (in.op == TEXKILL) {
        // The components the translation tests: x, y, z in ps_1_x, all four after.
        const uint8_t m = major_ == 1 ? 0x7 : 0xF;
        *kill = *kill || ctrl || (in.predicated && Tex(t, MOV, in.pred, m)) || In(t.tex, in.dst, m);
        continue;
      }
      if (!in.has_dst || in.op == DCL || in.op == DEF || in.op == DEFI || in.op == DEFB) continue;
      // A ps_1_x co-issued pair reads its sources before either writes.
      const bool pair = i + 1 < to && ins_[i + 1].coissue && ins_[i + 1].has_dst;
      bool tv[2] = {}, vv[2] = {};
      for (int k = 0; k < (pair ? 2 : 1); k++) {
        const Ins& x = ins_[i + size_t(k)];
        const uint32_t dm = x.dst.mask & 0xF;
        // (Predication applies per component written, whatever the instruction.)
        bool tx = ctrl || (x.predicated && Tex(t, MOV, x.pred, dm)), vr = false;
        if (major_ == 1 && x.op == TEX) {
          tx = true;  // (at TEXCOORDn)
        } else if (x.op == TEX || x.op == TEXLDL || x.op == TEXLDD) {
          // A texel where its coordinate (or gradients) is a texel's or
          // interpolated; the sampler itself is no source.
          for (size_t s = 0; s < x.src.size(); s++)
            if (s != 1) {
              tx = tx || Tex(t, x.op, x.src[s], 0xF) || Var(t, x.op, x.src[s], 0xF);
              vr = vr || Var(t, x.op, x.src[s], 0xF);
            }
        } else {
          for (const Param& p : x.src) {
            tx = tx || Tex(t, x.op, p, dm);
            vr = vr || Var(t, x.op, p, dm);
          }
          // The matrix instructions read the rows after their second source.
          const int rows = x.op == M4X4 || x.op == M3X4 ? 4 : x.op == M4X3 || x.op == M3X3 ? 3 : x.op == M3X2 ? 2 : 1;
          for (int r = 1; r < rows && x.src.size() > 1; r++) {
            Param row = x.src[1];
            row.num += uint32_t(r);
            tx = tx || Tex(t, x.op, row, dm);
            vr = vr || Var(t, x.op, row, dm);
          }
          if (major_ == 1 && x.op == TEXCOORD) vr = true;
        }
        tv[k] = tx;
        vv[k] = vr;
      }
      for (int k = 0; k < (pair ? 2 : 1); k++) {
        const Ins& x = ins_[i + size_t(k)];
        const uint8_t mask = uint8_t(x.dst.mask & 0xF);
        const bool sure = !weak && !x.predicated;
        if (tv[k] || sure) t.tex.Set(TaintKey(x.dst), mask, tv[k]);
        if (vv[k] || sure) t.var.Set(TaintKey(x.dst), mask, vv[k]);
      }
      if (pair) i++;
    }
    return ctrl;
  }

  bool KillReadsTexel() const {
    Taint t;
    bool kill = false;
    RunTaint(0, ins_.size(), t, false, false, nullptr, nullptr, 0, &kill);
    return kill;
  }

  PixelAlpha PixelAlphaOf() const {
    PixelAlpha r;
    for (const Ins& in : ins_) r.kills += in.op == TEXKILL ? 1 : 0;
    if (r.kills) r.kill_reads_texel = KillReadsTexel();
    std::map<uint32_t, Lanes> temps, tx;
    Lanes oc0{};
    int depth = 0;
    bool kill_ok = false;
    AlphaValue kill;
    // The value instruction `in` gives each component (before its result
    // modifiers).
    auto value = [&](const Ins& in) -> Lanes {
      Lanes v{};
      const std::vector<Param>& s = in.src;
      auto src = [&](size_t i, int l) { return SrcSym(s[i], l, temps, tx); };
      if (major_ == 1 && in.op == TEX) {
        Sym cu, cv;
        cu.kind = cv.kind = Sym::kVarying;
        cu.var = cv.var = {kTexcoord, in.dst.num};
        cv.lane = 1;
        return Texel(in.dst.num, 0xE4, cu, cv);
      }
      for (int l = 0; l < 4; l++) {
        if (!(in.dst.mask & (1u << l))) continue;
        if (in.op == MOV && s.size() == 1) v[l] = src(0, l);
        else if (in.op == ADD && s.size() == 2) v[l] = Sum(src(0, l), src(1, l));
        else if (in.op == MUL && s.size() == 2) v[l] = Product(src(0, l), src(1, l));
        else if (in.op == MAD && s.size() == 3) v[l] = Sum(Product(src(0, l), src(1, l)), src(2, l));
      }
      if ((in.op == TEX && in.ctl != 1 && s.size() == 2) || ((in.op == TEXLDL || in.op == TEXLDD) && s.size() >= 2)) {
        const Lanes t = Texel(s[1].num, s[1].swz, src(0, 0), src(0, 1));
        for (int l = 0; l < 4; l++)
          if (in.dst.mask & (1u << l)) v[l] = t[l];
      }
      return v;
    };
    auto write = [&](const Ins& in, Lanes v) {
      const Param& d = in.dst;
      const Lanes zero = {Literal(0.0f), Literal(0.0f), Literal(0.0f), Literal(0.0f)};
      Lanes* to = d.type == TEMP                    ? &temps.try_emplace(d.num, zero).first->second
                  : d.type == COLOROUT && d.num == 0 ? &oc0
                  : major_ == 1 && d.type == TEXTURE ? &tx[d.num]
                                                     : nullptr;
      if (!to) return;
      for (int l = 0; l < 4; l++) {
        if (!(d.mask & (1u << l))) continue;
        Sym x = v[l];
        if (depth > 0 || in.predicated) x = Sym{};
        if (d.shift) x = Affine(x, std::ldexp(1.0f, d.shift), 0.0f);
        if ((d.mod & 1) && x.kind != Sym::kUnknown) {
          // Saturated: a number clamped; COLOR0 as it is (the alpha test
          // clamps it); a product of values in [0, 1] (texels, colours,
          // a scale up to 1) as it is; anything else not known.
          x = AsValue(x);
          const AlphaValue& a = x.v;
          if (x.kind == Sym::kValue && IsLiteral(a)) x = Literal(std::clamp(a.offset, 0.0f, 1.0f));
          else if (!(d.type == COLOROUT && d.num == 0) &&
                   !(x.kind == Sym::kValue && a.constant < 0 && a.offset == 0.0f && a.offset_constant < 0 &&
                     a.scale >= 0.0f && a.scale <= 1.0f))
            x = Sym{};
        }
        (*to)[l] = x;
      }
    };
    for (size_t i = 0; i < ins_.size(); i++) {
      const Ins& in = ins_[i];
      if (in.op == IF || in.op == IFC) {
        depth++;
        continue;
      }
      if (in.op == ENDIF) {
        depth--;
        continue;
      }
      if (in.op == ELSE) continue;
      if (in.op == RET && depth == 0) break;
      if (in.op == CALL || in.op == CALLNZ || in.op == LABEL_ || in.op == LOOP_ || in.op == REP ||
          in.op == BREAK || in.op == BREAKC || in.op == BREAKP) {
        PixelAlpha none;
        none.kills = r.kills;
        none.kill_reads_texel = r.kill_reads_texel;
        return none;
      }
      if (in.op == TEXKILL) {
        // Every component tested (as the translation tests them: x, y and
        // z in ps_1_x, all four after) holds one value, or a number >= 0.
        Param p = in.dst;
        p.swz = 0xE4;
        p.mod = 0;
        bool ok = r.kills == 1 && depth == 0 && !in.predicated, have = false;
        for (int l = 0; l < (major_ == 1 ? 3 : 4) && ok; l++) {
          const Sym s = AsValue(SrcSym(p, l, temps, tx));
          if (s.kind != Sym::kValue) ok = false;
          else if (IsLiteral(s.v)) ok = s.v.offset >= 0.0f;
          else if (!have) kill = s.v, have = true;
          else ok = kill == s.v;
        }
        kill_ok = ok && have;
        continue;
      }
      if (!in.has_dst || in.op == DCL || in.op == DEF || in.op == DEFI || in.op == DEFB) continue;
      // ps_1_x co-issued pairs read their sources before either writes.
      if (i + 1 < ins_.size() && ins_[i + 1].coissue && ins_[i + 1].has_dst) {
        const Lanes a = value(in), b = value(ins_[i + 1]);
        write(in, a);
        write(ins_[i + 1], b);
        i++;
        continue;
      }
      write(in, value(in));
    }
    const Sym a = AsValue(major_ == 1 ? (temps.count(0) ? temps[0][3] : Literal(0.0f)) : oc0[3]);
    if (a.kind == Sym::kValue) r.color0 = a.v;
    if (kill_ok) r.kill = kill;
    return r;
  }

  // ---- vertex shaders' moved varying lanes (Program::moved_lanes)

  std::vector<MovedLane> MovedLanes() const {
    // Per lane of temporaries and outputs: input register * 4 + its lane
    // moved there, or -1.
    using Src = std::array<int, 4>;
    std::map<uint32_t, Src> temps, outs;  // outputs by type << 16 | number
    int depth = 0;
    for (const Ins& in : ins_) {
      if (in.op == IF || in.op == IFC) {
        depth++;
        continue;
      }
      if (in.op == ENDIF) {
        depth--;
        continue;
      }
      if (in.op == RET && depth == 0) break;
      if (in.op == CALL || in.op == CALLNZ || in.op == LABEL_ || in.op == LOOP_ || in.op == REP ||
          in.op == BREAK || in.op == BREAKC || in.op == BREAKP)
        return {};
      if (!in.has_dst || in.op == DCL || in.op == DEF || in.op == DEFI || in.op == DEFB) continue;
      Src v = {-1, -1, -1, -1};
      if (in.op == MOV && in.src.size() == 1 && depth == 0 && !in.predicated && in.dst.mod == 0 && in.dst.shift == 0) {
        const Param& s = in.src[0];
        for (int l = 0; l < 4 && s.mod == 0 && !s.rel; l++) {
          const int c = int((s.swz >> (2 * l)) & 3);
          if (s.type == INPUT) {
            v[l] = int(s.num) * 4 + c;
          } else if (s.type == TEMP) {
            auto it = temps.find(s.num);
            v[l] = it != temps.end() ? it->second[size_t(c)] : -1;
          }
        }
      }
      const Param& d = in.dst;
      const Src none = {-1, -1, -1, -1};
      Src* to = d.type == TEMP ? &temps.try_emplace(d.num, none).first->second
                : d.type == TEXCRDOUT || (major_ < 3 && d.type == ATTROUT)
                    ? &outs.try_emplace(d.type << 16 | d.num, none).first->second
                    : nullptr;
      if (!to) continue;
      for (int l = 0; l < 4; l++)
        if (d.mask & (1u << l)) (*to)[size_t(l)] = v[l];
    }
    std::vector<MovedLane> r;
    for (const auto& [key, src] : outs) {
      const uint32_t type = key >> 16, num = key & 0xFFFF;
      Semantic s;
      if (major_ >= 3) {
        auto it = vs3_outputs_.find(num);
        if (it == vs3_outputs_.end()) continue;
        s = it->second;
      } else {
        s = type == ATTROUT ? Semantic{kColor, num} : Semantic{kTexcoord, num};
      }
      if (s.usage != kTexcoord && s.usage != kColor) continue;
      for (int l = 0; l < 4; l++) {
        if (src[size_t(l)] < 0) continue;
        const uint32_t reg = uint32_t(src[size_t(l)]) / 4;
        auto it = vs_inputs_.find(reg);
        if (it == vs_inputs_.end()) continue;
        r.push_back({s, uint8_t(l), uint8_t(std::distance(vs_inputs_.begin(), it)), uint8_t(src[size_t(l)] % 4)});
      }
    }
    return r;
  }

  // ---- declarations: registers used, inputs, outputs, samplers

  void Declare() {
    for (const Ins& in : ins_) {
      if (in.op == DEF) {
        // NaNs read as 0, as in the registers the application sets
        // (SetVertexShaderConstantF / SetPixelShaderConstantF).
        auto f = [](uint32_t b) { return (b & 0x7FFFFFFFu) > 0x7F800000u ? 0u : b; };
        defs_[ConstIndex(in.dst)] = {f(in.raw[0]), f(in.raw[1]), f(in.raw[2]), f(in.raw[3])};
      } else if (in.op == DEFI) {
        defi_[in.dst.num] = {in.raw[0], in.raw[1], in.raw[2], in.raw[3]};
      } else if (in.op == DEFB) {
        defb_[in.dst.num] = in.raw[0] != 0;
      } else if (in.op == DCL) {
        Dcl(in);
      }
      auto note = [&](const Param& p, bool write) {
        if (p.type == TEMP) temps_ = std::max(temps_, p.num + 1);
        if (p.type == CONST || p.type == CONST2 || p.type == CONST3 || p.type == CONST4) {
          if (p.rel) {
            rel_const_ = true;
            const int base = int(ConstIndex(p));
            if (!write && base < 256 && (out_->rel_const_base < 0 || base < out_->rel_const_base))
              out_->rel_const_base = base;
          } else if (!defs_.count(ConstIndex(p))) {
            max_c_ = std::max(max_c_, ConstIndex(p) + 1);
            if (!write && ConstIndex(p) < 256) out_->float_reads.set(ConstIndex(p));
          }
        }
        if (p.type == CONSTINT || p.type == CONSTBOOL) uses_int_ = true;
        if (write && !pixel_ && (p.type == RASTOUT || p.type == ATTROUT || p.type == TEXCRDOUT))
          written_.insert({p.type, p.num});
        if (write && pixel_ && (p.type == COLOROUT || p.type == DEPTHOUT))
          written_.insert({p.type, p.num});
        if (!write && pixel_ && major_ < 3 && p.type == INPUT) ps_colors_.insert(p.num);
        if (!write && pixel_ && major_ == 2 && p.type == TEXTURE) ps_texcoords_.insert(p.num);
      };
      if (in.has_dst && in.op != DCL && in.op != DEF && in.op != DEFI && in.op != DEFB) note(in.dst, true);
      for (const Param& p : in.src) note(p, false);
      if (in.op == M4X4 || in.op == M4X3 || in.op == M3X4 || in.op == M3X3 || in.op == M3X2) {
        Param p = in.src[1];
        for (int k = 1; k < 4; k++) {
          p.num++;
          note(p, false);
        }
      }
      if (pixel_ && major_ == 1 && (in.op == TEX || in.op == TEXCOORD)) {
        ps_texcoords_.insert(in.dst.num);
        if (in.op == TEX) sampler_type_[in.dst.num] = ParamType::Sampler2D;
      }
    }
    if (rel_const_) max_c_ = pixel_ ? 224 : 256;
  }

  uint32_t ConstIndex(const Param& p) const {
    switch (p.type) {
      case CONST2: return p.num + 2048;
      case CONST3: return p.num + 4096;
      case CONST4: return p.num + 6144;
      default: return p.num;
    }
  }

  void Dcl(const Ins& in) {
    uint32_t u = in.raw[0];
    const Param& d = in.dst;
    if (d.type == SAMPLER) {
      uint32_t tt = (u >> 27) & 0xF;
      sampler_type_[d.num] = tt == 3 ? ParamType::SamplerCube
                             : tt == 4 ? ParamType::Sampler3D
                                       : ParamType::Sampler2D;
      return;
    }
    Semantic s{u & 0x1F, (u >> 16) & 0xF};
    if (!pixel_ && d.type == INPUT) {
      vs_inputs_[d.num] = s;
    } else if (!pixel_ && d.type == OUTPUT && major_ >= 3) {
      vs3_outputs_[d.num] = s;
    } else if (pixel_ && d.type == INPUT && major_ >= 3) {
      ps3_inputs_[d.num] = {s, d.mod};
    } else if (pixel_ && d.type == INPUT) {
      ps_colors_.insert(d.num);
      if (d.mod & 4) centroid_.insert({kColor, d.num});
    } else if (pixel_ && d.type == TEXTURE) {
      ps_texcoords_.insert(d.num);
      if (d.mod & 4) centroid_.insert({kTexcoord, d.num});
    } else if (pixel_ && d.type == MISCTYPE) {
      // vPos / vFace: read through the prelude.
    } else {
      Fail("unsupported dcl of register type " + std::to_string(d.type));
    }
  }

  // ---- expressions

  std::string IntReg(uint32_t n) {
    if (defi_.count(n)) {
      const auto& d = defi_[n];
      return "ivec4(" + std::to_string(int32_t(d[0])) + ", " + std::to_string(int32_t(d[1])) + ", " +
             std::to_string(int32_t(d[2])) + ", " + std::to_string(int32_t(d[3])) + ")";
    }
    uses_int_ = true;
    return "r_i[" + std::to_string(n) + "]";
  }

  std::string BoolReg(uint32_t n) {
    if (defb_.count(n)) return defb_[n] ? "true" : "false";
    uses_int_ = true;
    return "(r_b[" + std::to_string(n >> 2) + "][" + std::to_string(n & 3) + "] != 0)";
  }

  std::string RelIndex(const Param& p) {
    if (p.rel_type == LOOP) return "r_aL";
    if (p.rel_type == ADDR) return std::string("r_a0.") + kComp[p.rel_comp];
    Fail("unsupported relative address register");
  }

  // A register as a vec4 (no swizzle or modifier).
  std::string Reg(const Param& p) {
    switch (p.type) {
      case TEMP: return "r_t[" + std::to_string(p.num) + "]";
      case INPUT: return "r_v[" + std::to_string(p.num) + "]";
      case CONST: case CONST2: case CONST3: case CONST4: {
        uint32_t k = ConstIndex(p);
        if (p.rel) return "r_crel(" + RelIndex(p) + " + " + std::to_string(k) + ")";
        if (defs_.count(k)) return "r_dc" + std::to_string(k);
        return "c[" + std::to_string(k) + "]";
      }
      case TEXTURE:  // ps (ADDR in vertex shaders)
        if (!pixel_) return "vec4(r_a0)";
        return major_ == 1 ? "r_tx[" + std::to_string(p.num) + "]"
                           : "r_tc[" + std::to_string(p.num) + "]";
      case CONSTINT: return "vec4(" + IntReg(p.num) + ")";
      case CONSTBOOL: return "vec4(" + BoolReg(p.num) + ")";
      case LOOP: return "vec4(float(r_aL))";
      case MISCTYPE:
        if (p.num == 0) return "vec4(r_vpos(), 0.0, 0.0)";
        return "vec4(r_vface())";
      case PREDICATE: return "vec4(r_p0)";
      default: Fail("unsupported source register type " + std::to_string(p.type));
    }
  }

  static std::string Swizzle(uint32_t swz) {
    if (swz == 0xE4) return "";
    std::string s = ".";
    for (int i = 0; i < 4; i++) s += kComp[(swz >> (2 * i)) & 3];
    return s;
  }

  std::string S(const Param& p) {
    std::string x = Reg(p) + Swizzle(p.swz);
    switch (p.mod) {
      case 0: return x;
      case 1: return "(-" + x + ")";
      case 2: return "(" + x + " - 0.5)";
      case 3: return "(0.5 - " + x + ")";
      case 4: return "(2.0 * " + x + " - 1.0)";
      case 5: return "(1.0 - 2.0 * " + x + ")";
      case 6: return "(1.0 - " + x + ")";
      case 7: return "(2.0 * " + x + ")";
      case 8: return "(-2.0 * " + x + ")";
      case 9: return "(" + x + " / " + x + ".z)";
      case 10: return "(" + x + " / " + x + ".w)";
      case 11: return "abs(" + x + ")";
      case 12: return "(-abs(" + x + "))";
      default: Fail("unsupported source modifier " + std::to_string(p.mod));
    }
  }

  // First component of a source (scalar instructions).
  std::string S1(const Param& p) { return S(p) + ".x"; }

  // Comparison for ifc / breakc / setp (control bits).
  static const char* Cmp(uint32_t ctl) {
    switch (ctl & 7) {
      case 1: return ">";
      case 2: return "==";
      case 3: return ">=";
      case 4: return "<";
      case 5: return "!=";
      case 6: return "<=";
      default: Fail("bad comparison");
    }
  }

  // ---- emission

  std::string Ind() const { return std::string(size_t(indent_) * 2, ' '); }

  std::string& Out() { return func_ < 0 ? main_ : funcs_[func_]; }

  void Line(const std::string& s) { Out() += Ind() + s + "\n"; }

  std::string SamplerName(uint32_t n) {
    if (!sampler_type_.count(n)) sampler_type_[n] = ParamType::Sampler2D;
    used_samplers_.insert(n);
    return "r_s" + std::to_string(n);
  }

  // Whether a read of sampler `smp` becomes a call of r_rt_reflect
  // (TranslateOptions::ray_reflection): ps_3_0, the sampler asked for, the
  // origin an input declared with a semantic other than POSITION, the
  // registers in range.
  bool RayReflectionRead(const Param& smp) const {
    if (!pixel_ || major_ < 3 || rt_call_.sampler < 0 || smp.type != SAMPLER || smp.num != uint32_t(rt_call_.sampler))
      return false;
    auto it = ps3_inputs_.find(rt_call_.origin);
    if (it == ps3_inputs_.end() || it->second.first.usage == kPosition) return false;
    for (uint32_t r : rt_call_.registers)
      if (r >= 224) return false;
    return true;
  }

  // The call that reads ray-traced reflections in its place, the read's
  // coordinate `dir` the ray's direction (`projected`: texldp's, over its
  // w).
  std::string RtReflectCall(const Param& dir, bool projected = false) {
    helpers_.insert("rt_reflect");
    out_->ray_reflection_sampler = rt_call_.sampler;
    out_->ray_reflection_location = VaryingLocation(ps3_inputs_.at(rt_call_.origin).first);
    if (func_ >= 0) out_->ray_reflection_stub_calls++;
    const std::string d = S(dir);
    std::string call = "r_rt_reflect(gl_FragCoord, r_v[" + std::to_string(rt_call_.origin) + "], " +
                       (projected ? "(" + d + " / (" + d + ").w)" : d);
    for (uint32_t r : rt_call_.registers) {
      Param c;
      c.type = CONST;
      c.num = r;
      if (!defs_.count(r)) {
        max_c_ = std::max(max_c_, r + 1);
        out_->float_reads.set(r);
      }
      call += ", " + Reg(c);
    }
    return call + ")";
  }

  // A texldd of the shadow's sampler: the shadow toward a point (hlsl.h,
  // RayShadowCall): the read's coordinate the point in world space, its
  // first gradient the vector from it to the light, its second the light's
  // weight there (x).
  std::string RtShadowToCall(const Param& point, const Param& to, const Param& weight) {
    helpers_.insert("rt_shadow_to");
    out_->ray_shadow_sampler = rt_shadow_.sampler;
    if (func_ >= 0) out_->ray_shadow_stub_calls++;
    return "r_rt_shadow_to(gl_FragCoord, " + S(point) + ", " + S(to) + ", " + S(weight) + ")";
  }

  // Whether a read of sampler `smp` becomes a call of r_rt_shadow
  // (TranslateOptions::ray_shadow): ps_3_0, the sampler asked for, the
  // register in range.
  bool RayShadowRead(const Param& smp) const {
    return pixel_ && major_ >= 3 && rt_shadow_.sampler >= 0 && smp.type == SAMPLER &&
           smp.num == uint32_t(rt_shadow_.sampler) && rt_shadow_.light_register < 224;
  }

  // The call that reads the traced shadow in its place, the read's
  // coordinate `point` the point in world space (`projected`: texldp's,
  // over its w).
  std::string RtShadowCall(const Param& point, bool projected = false) {
    helpers_.insert("rt_shadow");
    out_->ray_shadow_sampler = rt_shadow_.sampler;
    if (func_ >= 0) out_->ray_shadow_stub_calls++;
    const std::string p = S(point);
    Param c;
    c.type = CONST;
    c.num = rt_shadow_.light_register;
    if (!defs_.count(c.num)) {
      max_c_ = std::max(max_c_, c.num + 1);
      out_->float_reads.set(c.num);
    }
    return "r_rt_shadow(gl_FragCoord, " + (projected ? "(" + p + " / (" + p + ").w)" : p) + ", " + Reg(c) + ")";
  }

  // The value of an instruction as a vec4 (before result modifiers).
  std::string Value(const Ins& in) {
    const auto& s = in.src;
    auto need = [&](size_t n) {
      if (s.size() < n) Fail("opcode " + std::to_string(in.op) + " with too few sources");
    };
    switch (in.op) {
      case MOV: need(1); return S(s[0]);
      case MOVA: need(1); return S(s[0]);
      case ADD: need(2); return "(" + S(s[0]) + " + " + S(s[1]) + ")";
      case SUB: need(2); return "(" + S(s[0]) + " - " + S(s[1]) + ")";
      case MUL:
        need(2);
        if (LegacyMul()) return Mulz(S(s[0]), S(s[1]));
        return "(" + S(s[0]) + " * " + S(s[1]) + ")";
      case MAD:
        need(3);
        if (LegacyMul()) return "(" + Mulz(S(s[0]), S(s[1])) + " + " + S(s[2]) + ")";
        return "(" + S(s[0]) + " * " + S(s[1]) + " + " + S(s[2]) + ")";
      case MIN: need(2); return "min(" + S(s[0]) + ", " + S(s[1]) + ")";
      case MAX: need(2); return "max(" + S(s[0]) + ", " + S(s[1]) + ")";
      case SLT: need(2); return "vec4(lessThan(" + S(s[0]) + ", " + S(s[1]) + "))";
      case SGE: need(2); return "vec4(greaterThanEqual(" + S(s[0]) + ", " + S(s[1]) + "))";
      // RCP, RSQ, LOG, LOGP, POW and NRM of 0 and EXP, EXPP of 128 and up
      // give infinities in D3D9 too, but D3D9 multiplies them by 0 to 0,
      // IEEE to NaN (black lighting): their inputs are kept off 0 and
      // below 128 (inputs: Metal's fast math may drop a clamp of a result)
      // so the results stay finite (huge), and 0 times them is 0 again.
      case RCP: need(1); return "vec4(1.0 / r_off0(" + S1(s[0]) + "))";
      case RSQ: need(1); return "vec4(inversesqrt(max(abs(" + S1(s[0]) + "), 1e-37)))";
      case EXP: need(1); return "vec4(exp2(min(" + S1(s[0]) + ", 127.99)))";
      case LOG: need(1); return "vec4(log2(max(abs(" + S1(s[0]) + "), 1e-37)))";
      case EXPP:
        need(1);
        if (!pixel_ && major_ == 1) {
          helpers_.insert("expp");
          return "r_expp1(" + S1(s[0]) + ")";
        }
        return "vec4(exp2(min(" + S1(s[0]) + ", 127.99)))";
      case LOGP:
        need(1);
        if (!pixel_ && major_ == 1) {
          helpers_.insert("logp");
          return "r_logp1(" + S1(s[0]) + ")";
        }
        return "vec4(log2(max(abs(" + S1(s[0]) + "), 1e-37)))";
      case POW: need(2); return "vec4(pow(max(abs(" + S1(s[0]) + "), 1e-37), " + S1(s[1]) + "))";
      case DP3:
        need(2);
        if (LegacyMul()) return "vec4(dot(" + Mulz(S(s[0]), S(s[1])) + ".xyz, vec3(1.0)))";
        return "vec4(dot((" + S(s[0]) + ").xyz, (" + S(s[1]) + ").xyz))";
      case DP4:
        need(2);
        if (LegacyMul()) return "vec4(dot(" + Mulz(S(s[0]), S(s[1])) + ", vec4(1.0)))";
        return "vec4(dot(" + S(s[0]) + ", " + S(s[1]) + "))";
      case DP2ADD:
        need(3);
        return "vec4(dot((" + S(s[0]) + ").xy, (" + S(s[1]) + ").xy) + " + S1(s[2]) + ")";
      case LIT: need(1); helpers_.insert("lit"); return "r_lit(" + S(s[0]) + ")";
      case DST:
        need(2);
        return "vec4(1.0, (" + S(s[0]) + ").y * (" + S(s[1]) + ").y, (" + S(s[0]) + ").z, (" +
               S(s[1]) + ").w)";
      case LRP:
        need(3);
        if (LegacyMul())
          return "(" + Mulz(S(s[0]), S(s[1])) + " + " + Mulz("(vec4(1.0) - " + S(s[0]) + ")", S(s[2])) + ")";
        return "mix(" + S(s[2]) + ", " + S(s[1]) + ", " + S(s[0]) + ")";
      case FRC: need(1); return "fract(" + S(s[0]) + ")";
      case ABS: need(1); return "abs(" + S(s[0]) + ")";
      case SGN: need(1); return "sign(" + S(s[0]) + ")";
      case CRS: need(2); return "vec4(cross((" + S(s[0]) + ").xyz, (" + S(s[1]) + ").xyz), 0.0)";
      case NRM:
        need(1);
        return "(" + S(s[0]) + " * inversesqrt(max(dot((" + S(s[0]) + ").xyz, (" + S(s[0]) + ").xyz), 1e-37)))";
      case SINCOS: need(1); return "vec4(cos(" + S1(s[0]) + "), sin(" + S1(s[0]) + "), 0.0, 0.0)";
      case CMP:
        need(3);
        return "mix(" + S(s[2]) + ", " + S(s[1]) + ", greaterThanEqual(" + S(s[0]) + ", vec4(0.0)))";
      case CND:
        need(3);
        return "mix(" + S(s[2]) + ", " + S(s[1]) + ", greaterThan(" + S(s[0]) + ", vec4(0.5)))";
      case DSX: need(1); return "dFdx(" + S(s[0]) + ")";
      case DSY: need(1); return "dFdy(" + S(s[0]) + ")";
      case M4X4: case M4X3: case M3X4: case M3X3: case M3X2: {
        need(2);
        int rows = in.op == M4X4 || in.op == M3X4 ? 4 : in.op == M3X2 ? 2 : 3;
        bool three = in.op == M3X4 || in.op == M3X3 || in.op == M3X2;
        std::string a = S(s[0]);
        std::string r = "vec4(";
        for (int k = 0; k < 4; k++) {
          if (k) r += ", ";
          if (k >= rows) {
            r += "0.0";
            continue;
          }
          Param row = s[1];
          row.num += uint32_t(k);
          std::string b = S(row);
          r += three ? "dot((" + a + ").xyz, (" + b + ").xyz)" : "dot(" + a + ", " + b + ")";
        }
        return r + ")";
      }
      case TEX: {  // texld (ps_2+); ps_1_x handled in Instruction
        need(2);
        if (RayReflectionRead(s[1])) return RtReflectCall(s[0], in.ctl == 1);
        if (RayShadowRead(s[1])) return RtShadowCall(s[0], in.ctl == 1);
        ParamType ty = sampler_type_.count(s[1].num) ? sampler_type_[s[1].num] : ParamType::Sampler2D;
        std::string smp = SamplerName(s[1].num), c = S(s[0]);
        std::string co = ty == ParamType::Sampler2D ? "(" + c + ").xy" : "(" + c + ").xyz";
        // The sampler's LOD bias (R_BIAS, hlsl.h); texld is pixel-only.
        std::string b = "R_BIAS(" + std::to_string(s[1].num) + ")";
        if (in.ctl == 1) {  // texldp
          if (ty == ParamType::Sampler2D)
            return DepthTex(s[1].num, "textureProj(" + smp + ", " + c + ", " + b + ")");
          return "texture(" + smp + ", (" + c + ").xyz / (" + c + ").w, " + b + ")";
        }
        if (in.ctl == 2)  // texldb
          return DepthTex(s[1].num, ty, "texture(" + smp + ", " + co + ", (" + c + ").w + " + b + ")");
        return DepthTex(s[1].num, ty, "texture(" + smp + ", " + co + ", " + b + ")");
      }
      case TEXLDL: {
        need(2);
        if (RayReflectionRead(s[1])) return RtReflectCall(s[0]);
        if (RayShadowRead(s[1])) return RtShadowCall(s[0]);
        ParamType ty = sampler_type_.count(s[1].num) ? sampler_type_[s[1].num] : ParamType::Sampler2D;
        std::string smp = SamplerName(s[1].num), c = S(s[0]);
        std::string co = ty == ParamType::Sampler2D ? "(" + c + ").xy" : "(" + c + ").xyz";
        return DepthTex(s[1].num, ty, "textureLod(" + smp + ", " + co + ", (" + c + ").w)");
      }
      case TEXLDD: {
        need(4);
        if (RayReflectionRead(s[1])) return RtReflectCall(s[0]);
        if (RayShadowRead(s[1])) return RtShadowToCall(s[0], s[2], s[3]);
        ParamType ty = sampler_type_.count(s[1].num) ? sampler_type_[s[1].num] : ParamType::Sampler2D;
        std::string smp = SamplerName(s[1].num);
        const char* sw = ty == ParamType::Sampler2D ? ".xy" : ".xyz";
        return DepthTex(s[1].num, ty,
                        "textureGrad(" + smp + ", (" + S(s[0]) + ")" + sw + ", (" + S(s[2]) + ")" + sw +
                            ", (" + S(s[3]) + ")" + sw + ")");
      }
      default:
        Fail("unsupported opcode " + std::to_string(in.op) + " in " +
             std::string(pixel_ ? "ps_" : "vs_") + std::to_string(major_) + "_" +
             std::to_string(minor_));
    }
  }

  // The destination as an assignable name.
  std::string DstName(const Param& p) {
    switch (p.type) {
      case TEMP: return "r_t[" + std::to_string(p.num) + "]";
      case TEXTURE:
        if (!pixel_) return "r_a0";
        if (major_ == 1) return "r_tx[" + std::to_string(p.num) + "]";
        break;
      case RASTOUT:
        if (p.num == 0) return "r_opos";
        if (p.num == 1) return "r_ofog";
        return "r_opts";
      case ATTROUT: return "r_od[" + std::to_string(p.num) + "]";
      case TEXCRDOUT:  // OUTPUT in vs_3_0
        if (major_ >= 3) {
          if (p.rel) Fail("relative output registers are not supported");
          return "r_o[" + std::to_string(p.num) + "]";
        }
        return "r_ot[" + std::to_string(p.num) + "]";
      case COLOROUT: return "r_oc[" + std::to_string(p.num) + "]";
      case DEPTHOUT: return "r_odepth";
      case PREDICATE: return "r_p0";
      default: break;
    }
    Fail("unsupported destination register type " + std::to_string(p.type));
  }

  static std::string Mask(uint32_t m) {
    std::string s;
    for (int i = 0; i < 4; i++)
      if (m & (1u << i)) s += kComp[i];
    return s;
  }

  // Applies result modifiers to the value `v` (a variable name) and
  // writes it to the destination, honouring predication.
  void Write(const Ins& in, const std::string& v) {
    const Param& d = in.dst;
    std::string x = v;
    if (d.shift > 0) x = "(" + x + " * " + std::to_string(1 << d.shift) + ".0)";
    if (d.shift < 0) x = "(" + x + " / " + std::to_string(1 << -d.shift) + ".0)";
    if (d.mod & 1) x = "clamp(" + x + ", 0.0, 1.0)";
    std::string name = DstName(d);
    uint32_t mask = d.mask ? d.mask : 0xF;
    std::string m = Mask(mask);
    bool full = mask == 0xF;
    if (d.type == TEXTURE && !pixel_) {
      // a0: vs_1_1 mov truncates toward -inf; mova rounds to nearest.
      std::string r = in.op == MOVA ? "floor(" + x + " + 0.5)" : "floor(" + x + ")";
      x = "ivec4(" + r + ")";
    }
    if (in.predicated) {
      std::string pr = "r_p0" + Swizzle(in.pred.swz);
      if (in.pred.mod == 13) pr = "not(" + pr + ")";
      for (int i = 0; i < 4; i++)
        if (mask & (1u << i))
          Line("if ((" + pr + ")." + kComp[i] + ") " + name + "." + kComp[i] + " = (" + x + ")." + kComp[i] + ";");
      return;
    }
    if (full)
      Line(name + " = " + x + ";");
    else
      Line(name + "." + m + " = (" + x + ")." + m + ";");
  }

  void Instruction(size_t idx) {
    const Ins& in = ins_[idx];
    switch (in.op) {
      case NOP: case DCL: case DEF: case DEFI: case DEFB: return;
      case PHASE: Fail("ps_1_4 phases are not supported");
      case REP: {
        int k = loops_++;
        Line("for (int r_rep" + std::to_string(k) + " = 0; r_rep" + std::to_string(k) + " < " +
             IntReg(in.src[0].num) + ".x; r_rep" + std::to_string(k) + "++) {");
        loop_kinds_.push_back(false);
        indent_++;
        return;
      }
      case LOOP_: {
        // loop aL, i#: aL starts at i.y and steps by i.z, i.x times; the
        // enclosing loop's aL is restored after.
        int k = loops_++;
        std::string ks = std::to_string(k), iv = IntReg(in.src[1].num);
        Line("{");
        indent_++;
        Line("int r_saved_aL" + ks + " = r_aL;");
        Line("ivec4 r_li" + ks + " = " + iv + ";");
        Line("r_aL = r_li" + ks + ".y;");
        Line("for (int r_n" + ks + " = 0; r_n" + ks + " < r_li" + ks + ".x; r_n" + ks + "++, r_aL += r_li" + ks + ".z) {");
        indent_++;
        loop_kinds_.push_back(true);
        loop_ids_.push_back(k);
        return;
      }
      case ENDREP: case ENDLOOP: {
        if (loop_kinds_.empty()) Fail("unbalanced end of loop");
        bool aloop = loop_kinds_.back();
        loop_kinds_.pop_back();
        indent_--;
        Line("}");
        if (aloop) {
          int k = loop_ids_.back();
          loop_ids_.pop_back();
          Line("r_aL = r_saved_aL" + std::to_string(k) + ";");
          indent_--;
          Line("}");
        }
        return;
      }
      case IF:
        if (in.src[0].type == PREDICATE) {
          std::string p = "r_p0" + Swizzle(in.src[0].swz) + ".x";
          Line(std::string("if (") + (in.src[0].mod == 13 ? "!" : "") + p + ") {");
        } else {
          Line("if (" + BoolReg(in.src[0].num) + ") {");
        }
        indent_++;
        return;
      case IFC:
        Line("if (" + S1(in.src[0]) + " " + Cmp(in.ctl) + " " + S1(in.src[1]) + ") {");
        indent_++;
        return;
      case ELSE:
        indent_--;
        Line("} else {");
        indent_++;
        return;
      case ENDIF:
        indent_--;
        Line("}");
        return;
      case BREAK: Line("break;"); return;
      case BREAKC:
        Line("if (" + S1(in.src[0]) + " " + Cmp(in.ctl) + " " + S1(in.src[1]) + ") break;");
        return;
      case BREAKP: {
        std::string p = "r_p0" + Swizzle(in.src[0].swz) + ".x";
        Line(std::string("if (") + (in.src[0].mod == 13 ? "!" : "") + p + ") break;");
        return;
      }
      case CALL: Line("r_l" + std::to_string(in.src[0].num) + "();"); return;
      case CALLNZ: {
        const Param& b = in.src[1];
        std::string cond = b.type == PREDICATE
                               ? std::string(b.mod == 13 ? "!" : "") + "r_p0" + Swizzle(b.swz) + ".x"
                               : BoolReg(b.num);
        Line("if (" + cond + ") r_l" + std::to_string(in.src[0].num) + "();");
        return;
      }
      case RET:
        if (func_ < 0) {
          main_done_ = true;
        } else {
          indent_--;
          Line("}");
          func_ = -1;
          indent_ = 1;
        }
        return;
      case LABEL_: {
        if (func_ >= 0) Fail("label inside a subroutine");
        func_ = int(in.src[0].num);
        labels_.insert(in.src[0].num);
        indent_ = 0;
        Line("void r_l" + std::to_string(func_) + "() {");
        indent_ = 1;
        return;
      }
      case TEXKILL: {
        std::string v = pixel_ && major_ == 1 ? Reg(in.dst) : DstName(in.dst);
        const char* sw = major_ == 1 ? ".xyz" : "";
        Line(std::string("if (any(lessThan(") + v + sw + ", " + (major_ == 1 ? "vec3" : "vec4") + "(0.0)))) discard;");
        return;
      }
      case SETP: {
        std::string m = Mask(in.dst.mask ? in.dst.mask : 0xF);
        static const char* const kFn[] = {"", "greaterThan", "equal", "greaterThanEqual", "lessThan",
                                          "notEqual", "lessThanEqual"};
        uint32_t ctl = in.ctl & 7;
        if (ctl < 1 || ctl > 6) Fail("bad setp comparison");
        Line("r_p0." + m + " = " + kFn[ctl] + "(" + S(in.src[0]) + ", " + S(in.src[1]) + ")." + m + ";");
        return;
      }
      default: break;
    }
    if (pixel_ && major_ == 1 && in.op == TEX) {
      std::string n = std::to_string(in.dst.num);
      const std::string v = "texture(" + SamplerName(in.dst.num) + ", r_tc[" + n + "].xy, R_BIAS(" + n + "))";
      Line("r_tx[" + n + "] = " + DepthTex(in.dst.num, v) + ";");
      return;
    }
    if (pixel_ && major_ == 1 && in.op == TEXCOORD) {
      std::string n = std::to_string(in.dst.num);
      Line("r_tx[" + n + "] = vec4(clamp(r_tc[" + n + "].xyz, 0.0, 1.0), 1.0);");
      return;
    }
    switch (in.op) {
      case TEXBEM: case TEXBEML: case TEXREG2AR: case TEXREG2GB: case TEXM3X2PAD: case TEXM3X2TEX:
      case TEXM3X3PAD: case TEXM3X3TEX: case TEXM3X3SPEC: case TEXM3X3VSPEC: case TEXREG2RGB:
      case TEXDP3TEX: case TEXM3X2DEPTH: case TEXDP3: case TEXM3X3: case TEXDEPTH: case BEM:
        Fail("ps_1_x texture instruction " + std::to_string(in.op) + " is not supported");
      default:
        break;
    }
    // Co-issued pairs read their sources before either writes.
    if (idx + 1 < ins_.size() && ins_[idx + 1].coissue) return;  // emitted with its partner
    if (in.coissue && idx > 0) {
      const Ins& first = ins_[idx - 1];
      Line("{");
      indent_++;
      Line("vec4 r_va = " + Value(first) + ";");
      Line("vec4 r_vb = " + Value(in) + ";");
      Write(first, "r_va");
      Write(in, "r_vb");
      indent_--;
      Line("}");
      return;
    }
    Line("{");
    indent_++;
    Line("vec4 r_va = " + Value(in) + ";");
    Write(in, "r_va");
    indent_--;
    Line("}");
  }

  // ---- interface

  std::string Varying(Semantic s, bool out, uint32_t mods = 0) {
    int l = VaryingLocation(s);
    if (l < 0) Fail("semantic usage " + std::to_string(s.usage) + " cannot be a varying");
    std::string name = "r_var" + std::to_string(l);
    if (declared_.insert(name).second) {
      std::string q = (mods & 4) ? "centroid " : "";
      io_ += "layout(location = " + std::to_string(l) + ") " + q + (out ? "out" : "in") +
             " vec4 " + name + ";\n";
      out_->varyings.push_back(s);
    }
    return name;
  }

  void Emit() {
    indent_ = 1;
    for (size_t i = 0; i < ins_.size(); i++) {
      if (main_done_ && ins_[i].op != LABEL_ && func_ < 0) continue;
      Instruction(i);
    }
    if (func_ >= 0) {
      indent_ = 0;
      Line("}");
      func_ = -1;
    }
    if (!loop_kinds_.empty()) Fail("unterminated loop");

    std::string h = std::string("// ") + (pixel_ ? "ps_" : "vs_") + std::to_string(major_) + "_" +
                    std::to_string(minor_) + ", translated from shader bytecode\n";
    if (max_c_) {
      h += "layout(std140, R_FLOAT_BINDING) uniform r_float_regs { vec4 c[" + std::to_string(max_c_) +
           "]; };\n";
      out_->float_regs = max_c_;
    }
    if (uses_int_)
      h += "layout(std140, R_INT_BINDING) uniform r_int_regs { ivec4 r_i[16]; ivec4 r_b[4]; };\n";
    for (uint32_t n : used_samplers_) {
      ParamType t = sampler_type_[n];
      const char* ty = t == ParamType::SamplerCube ? "samplerCube"
                       : t == ParamType::Sampler3D ? "sampler3D"
                                                   : "sampler2D";
      h += "layout(R_SAMPLER_BINDING(" + std::to_string(n) + ")) uniform " + ty + " r_s" +
           std::to_string(n) + ";\n";
      out_->samplers.push_back(Sampler{n, t});
    }

    // main: inputs, definitions, the body, outputs.
    std::string pre, post;
    for (const auto& [k, d] : defs_)
      pre += "  r_dc" + std::to_string(k) + " = vec4(" + FloatBitsExpr(d[0]) + ", " + FloatBitsExpr(d[1]) +
             ", " + FloatBitsExpr(d[2]) + ", " + FloatBitsExpr(d[3]) + ");\n";
    std::string g;  // globals
    g += "vec4 r_t[" + std::to_string(std::max(temps_, 1u)) + "];\n";
    g += "ivec4 r_a0;\nint r_aL;\nbvec4 r_p0;\n";
    for (const auto& [k, d] : defs_) g += "vec4 r_dc" + std::to_string(k) + ";\n";
    if (!pixel_) {
      g += "vec4 r_v[16];\n";
      for (const auto& [reg, s] : vs_inputs_) {
        size_t slot = out_->inputs.size();
        out_->inputs.push_back(s);
        pre += "  r_v[" + std::to_string(reg) + "] = r_fetch(" + std::to_string(slot) + ");\n";
      }
      if (major_ >= 3) {
        g += "vec4 r_o[12];\n";
        for (const auto& [reg, s] : vs3_outputs_) {
          std::string v = "r_o[" + std::to_string(reg) + "]";
          // Components its writes leave are UnwrittenVarying's.
          pre += "  " + v + " = " + UnwrittenVarying(s) + ";\n";
          if (s.usage == kPosition && s.index == 0) {
            post += "  gl_Position = r_position(" + v + ");\n";
          } else if (s.usage == kPSize) {
            post += "  gl_PointSize = " + v + ".x;\n";
            out_->writes_point_size = true;
          } else if (s.usage == kFog) {
            post += "  " + Varying(s, true) + " = vec4(" + v + ".x, 0.0, 0.0, 1.0);\n";
          } else {
            post += "  " + Varying(s, true) + " = " + v + ";\n";
          }
        }
      } else {
        g += "vec4 r_opos, r_ofog, r_opts;\nvec4 r_od[2];\nvec4 r_ot[8];\n";
        // Components an output's writes leave are UnwrittenVarying's.
        pre += "  r_opos = vec4(0.0); r_ofog = vec4(1.0); r_opts = vec4(1.0);\n";
        pre += "  r_od[0] = vec4(1.0); r_od[1] = vec4(0.0, 0.0, 0.0, 1.0);\n";
        pre += "  for (int i = 0; i < 8; i++) r_ot[i] = vec4(0.0);\n";
        post += "  gl_Position = r_position(r_opos);\n";
        for (const auto& [type, num] : written_) {
          if (type == RASTOUT && num == 1) {
            post += "  " + Varying({kFog, 0}, true) + " = vec4(r_ofog.x, 0.0, 0.0, 1.0);\n";
          } else if (type == RASTOUT && num == 2) {
            post += "  gl_PointSize = r_opts.x;\n";
            out_->writes_point_size = true;
          } else if (type == ATTROUT) {
            // Colour outputs are clamped to [0, 1] before shader model 3.
            post += "  " + Varying({kColor, num}, true) + " = clamp(r_od[" + std::to_string(num) +
                    "], 0.0, 1.0);\n";
          } else if (type == TEXCRDOUT) {
            post += "  " + Varying({kTexcoord, num}, true) + " = r_ot[" + std::to_string(num) + "];\n";
          }
        }
      }
    } else {
      g += "vec4 r_v[16];\nvec4 r_oc[4];\nvec4 r_odepth;\n";
      if (major_ >= 3) {
        for (const auto& [reg, sm] : ps3_inputs_) {
          const Semantic& s = sm.first;
          std::string v = "r_v[" + std::to_string(reg) + "]";
          if (s.usage == kPosition)
            pre += "  " + v + " = vec4(0.0);\n";
          else
            pre += "  " + v + " = " + Varying(s, false, sm.second) + ";\n";
        }
      } else {
        for (uint32_t n : ps_colors_)
          pre += "  r_v[" + std::to_string(n) + "] = " +
                 Varying({kColor, n}, false, centroid_.count({kColor, n}) ? 4 : 0) + ";\n";
        g += major_ == 1 ? "vec4 r_tx[8];\n" : "";
        g += "vec4 r_tc[8];\n";
        for (uint32_t n : ps_texcoords_)
          pre += "  r_tc[" + std::to_string(n) + "] = " +
                 Varying({kTexcoord, n}, false, centroid_.count({kTexcoord, n}) ? 4 : 0) + ";\n";
      }
      // Outputs: r0 in ps_1_x, else what was written.
      std::vector<uint32_t> colors;
      if (major_ == 1) {
        colors.push_back(0);
        post += "  r_oc[0] = r_t[0];\n";
      }
      for (const auto& [type, num] : written_)
        if (type == COLOROUT) colors.push_back(num);
      for (uint32_t n : colors) {
        if (n > 3) Fail("bad colour output");
        if (out_->color_outputs & (1u << n)) continue;
        out_->color_outputs |= 1u << n;
        io_ += "layout(location = " + std::to_string(n) + ") out vec4 r_c" + std::to_string(n) + ";\n";
        post += "  r_c" + std::to_string(n) + " = r_oc[" + std::to_string(n) + "];\n";
      }
      for (const auto& [type, num] : written_)
        if (type == DEPTHOUT) {
          post += "  gl_FragDepth = R_DEPTH_OUT(r_odepth.x);\n";
          out_->writes_depth = true;
        }
      if (out_->color_outputs & 1) post += "  r_color0(r_c0);\n";
    }
    pre = "  for (int i = 0; i < " + std::to_string(std::max(temps_, 1u)) + "; i++) r_t[i] = vec4(0.0);\n" + pre;

    std::string help;
    if (rel_const_) {
      help += "vec4 r_crel(int i) {\n";
      for (const auto& [k, d] : defs_)
        help += "  if (i == " + std::to_string(k) + ") return r_dc" + std::to_string(k) + ";\n";
      help += "  return i >= 0 && i < " + std::to_string(max_c_) + " ? c[i] : vec4(0.0);\n}\n";
    }
    help += "float r_off0(float x) { return abs(x) < 1e-37 ? 1e-37 : x; }\n";
    if (out_->writes_depth) help += kDepthOutDefault;
    if (depth_tex_) help += kDepthTexDefault;
    if (helpers_.count("mulz"))
      help += "// D3D9's multiply: 0 times anything (infinity, NaN) is 0.\n"
              "vec4 r_mulz(vec4 a, vec4 b) {\n"
              "  bvec4 z = bvec4(vec4(equal(a, vec4(0.0))) + vec4(equal(b, vec4(0.0))));\n"
              "  return mix(a * b, vec4(0.0), z);\n"
              "}\n";
    if (helpers_.count("lit"))
      help += "vec4 r_lit(vec4 s) {\n"
              "  float p = clamp(s.w, -128.0, 128.0);\n"
              "  return vec4(1.0, max(s.x, 0.0), s.x > 0.0 ? pow(max(s.y, 0.0), p) : 0.0, 1.0);\n"
              "}\n";
    if (helpers_.count("expp"))
      help += "vec4 r_expp1(float s) {\n"
              "  return vec4(exp2(min(floor(s), 127.0)), fract(s), exp2(min(s, 127.99)), 1.0);\n"
              "}\n";
    if (helpers_.count("logp"))
      help += "vec4 r_logp1(float s) {\n"
              "  float a = max(abs(s), 1e-37), e = floor(log2(a));\n"
              "  return vec4(e, a / exp2(e), log2(a), 1.0);\n"
              "}\n";
    if (helpers_.count("rt_reflect"))
      help += "// Ray-traced reflections (hlsl::RayReflectionCall): what a ray from\n"
              "// origin.xyz along dir.xyz hits (rgb, w 1 where it hit). Nothing\n"
              "// here: a backend that traces rays makes the calls ray queries.\n"
              "vec4 r_rt_reflect(vec4 pixel, vec4 origin, vec4 dir, vec4 light, vec4 colour, vec4 ambient0,\n"
              "                  vec4 ambient1) {\n"
              "  return vec4(0.0);\n"
              "}\n";
    if (helpers_.count("rt_shadow"))
      help += "// Ray-traced shadows (hlsl::RayShadowCall): whether world.xyz (world space)\n"
              "// sees the light along light.xyz (x: 1 lit, 0 in shadow). Lit here: a\n"
              "// backend that traces rays makes the calls ray queries.\n"
              "vec4 r_rt_shadow(vec4 pixel, vec4 world, vec4 light) {\n"
              "  return vec4(1.0);\n"
              "}\n";
    if (helpers_.count("rt_shadow_to"))
      help += "// Ray-traced shadows toward a point (hlsl::RayShadowCall, a texldd):\n"
              "// whether world.xyz sees the light at world.xyz + to.xyz (x: 1 lit, 0 in\n"
              "// shadow), asked only where weight.x is above 0. Lit here: a backend that\n"
              "// traces rays makes the calls ray queries.\n"
              "vec4 r_rt_shadow_to(vec4 pixel, vec4 world, vec4 to, vec4 weight) {\n"
              "  return vec4(1.0);\n"
              "}\n";
    std::string protos;
    for (uint32_t l : labels_) protos += "void r_l" + std::to_string(l) + "();\n";
    std::string funcs;
    for (const auto& [l, body] : funcs_) funcs += body;

    out_->glsl = h + io_ + "\n" + g + help + protos + funcs + "\nvoid main() {\n" + pre + main_ +
                 post + "}\n";
  }

  const uint32_t* t_;
  size_t n_;
  Program* out_;
  bool coverage_ = false;  // TranslateOptions::coverage
  RayReflectionCall rt_call_;  // TranslateOptions::ray_reflection
  RayShadowCall rt_shadow_;    // TranslateOptions::ray_shadow
  bool pixel_ = false;
  uint32_t major_ = 0, minor_ = 0;
  std::vector<Ins> ins_;

  std::map<uint32_t, std::array<uint32_t, 4>> defs_, defi_;
  std::map<uint32_t, bool> defb_;
  uint32_t temps_ = 0, max_c_ = 0;
  bool rel_const_ = false, uses_int_ = false;
  std::map<uint32_t, Semantic> vs_inputs_, vs3_outputs_;
  std::map<uint32_t, std::pair<Semantic, uint32_t>> ps3_inputs_;  // semantic, dcl modifiers
  std::set<uint32_t> ps_colors_, ps_texcoords_;
  std::set<std::pair<uint32_t, uint32_t>> written_, centroid_;
  std::map<uint32_t, ParamType> sampler_type_;
  std::set<uint32_t> used_samplers_;
  std::set<std::string> helpers_;
  bool depth_tex_ = false;  // R_DEPTH_TEX used

  // What 2D sampler register n read, as the shader expects it (R_DEPTH_TEX,
  // hlsl.h: a depth texture may be stored otherwise).
  std::string DepthTex(uint32_t n, const std::string& v) {
    depth_tex_ = true;
    return "R_DEPTH_TEX(" + std::to_string(n) + ", " + v + ")";
  }
  std::string DepthTex(uint32_t n, ParamType ty, const std::string& v) {
    return ty == ParamType::Sampler2D ? DepthTex(n, v) : v;
  }

  // RECOMP_D3D9_LEGACY_MUL=1: every multiply (MUL, MAD, DP3, DP4, LRP)
  // as D3D9 hardware does it, 0 times anything giving 0. Costly where
  // shading is heavy (GPU time up to three times in places); the finite
  // RCP/RSQ/LOG/POW/NRM above cover what games need.
  static bool LegacyMul() {
    static const bool on = [] {
      const char* e = std::getenv("RECOMP_D3D9_LEGACY_MUL");
      return e && *e == '1';
    }();
    return on;
  }
  std::string Mulz(const std::string& a, const std::string& b) {
    helpers_.insert("mulz");
    return "r_mulz(vec4(" + a + "), vec4(" + b + "))";
  }

  std::string main_, io_;
  std::map<int, std::string> funcs_;
  std::set<uint32_t> labels_;
  std::set<std::string> declared_;
  int func_ = -1, indent_ = 1, loops_ = 0;
  bool main_done_ = false;
  std::vector<bool> loop_kinds_;  // true: loop (aL), false: rep
  std::vector<int> loop_ids_;
};

}  // namespace

bool TranslateBytecode(const uint32_t* tokens, size_t count, Program* out, std::string* errors) {
  return TranslateBytecode(tokens, count, out, errors, TranslateOptions{});
}

bool TranslateBytecode(const uint32_t* tokens, size_t count, Program* out, std::string* errors,
                       const TranslateOptions& options) {
  *out = Program{};
  try {
    Translator(tokens, count, out, options).Run();
  } catch (const Error& e) {
    *errors = "bytecode: " + e.msg;
    return false;
  }
  return true;
}

}  // namespace recomp::hlsl
