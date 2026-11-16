// Where an alpha-cut draw's alpha comes from (rt_alpha.h).

#import "runtime/gfx/rt_alpha.h"

#include <algorithm>

namespace recomp::gfx {

namespace {

// Where the alpha comes from before the draw's inputs are read.
struct Source {
  int sampler = -1;  // -1: no texel
  uint8_t channel = 3;
  int uv = -1, color = -1;  // input slots (-1: none)
  uint8_t u = 0, v = 1, color_lane = 3;
  float scale = 1.0f, offset = 0.0f;  // the alpha test's value
  bool kill = false;
  float kill_scale = 0.0f, kill_offset = 0.0f;
  uint32_t func = 0, ref = 0;  // D3DCMPFUNC (0: no alpha test), ALPHAREF
};

// The vertex input slot and component a varying's lane is moved from.
bool MovedFrom(const RtAlphaShaders& sh, hlsl::Semantic varying, uint8_t lane, int* slot, uint8_t* input_lane) {
  if (!sh.moved) return false;
  for (const hlsl::MovedLane& m : *sh.moved)
    if (m.varying == varying && m.lane == lane) {
      *slot = m.input;
      *input_lane = m.input_lane;
      return true;
    }
  return false;
}

// The bytecode's value `a` into s (its UVs and colour factor followed back
// to vertex inputs, its constants from the draw's registers), the scale
// and offset into *scale, *offset: false if not followed back.
bool Value(const ShaderDraw& d, const RtAlphaShaders& sh, const hlsl::AlphaValue& a, float* scale, float* offset,
           Source* s) {
  float k = 1.0f, o = 0.0f;
  if (a.constant >= 0 || a.offset_constant >= 0) {
    if (!d.ps_float || a.constant >= 224 || a.offset_constant >= 224) return false;
    if (a.constant >= 0) k = d.ps_float[a.constant][a.constant_lane];
    if (a.offset_constant >= 0) o = a.offset_k * d.ps_float[a.offset_constant][a.offset_lane];
  }
  *scale = a.scale * k;
  *offset = a.offset + o;
  if (a.sampler >= 0) {
    int su = -1, sv = -1;
    uint8_t lu = 0, lv = 0;
    const hlsl::Semantic tc{hlsl::kTexcoord, a.texcoord};
    if (!MovedFrom(sh, tc, a.u, &su, &lu) || !MovedFrom(sh, tc, a.v, &sv, &lv) || su != sv) return false;
    s->sampler = a.sampler;
    s->channel = a.channel;
    s->uv = su;
    s->u = lu;
    s->v = lv;
  }
  if (a.color >= 0 && !MovedFrom(sh, {hlsl::kColor, uint32_t(a.color)}, a.color_lane, &s->color, &s->color_lane))
    return false;
  return true;
}

// The same product (the texel, its UVs, the colour factor, the constant
// factor's register).
bool Same(const hlsl::AlphaValue& a, const hlsl::AlphaValue& b) {
  return a.sampler == b.sampler && a.channel == b.channel && a.texcoord == b.texcoord && a.u == b.u && a.v == b.v &&
         a.color == b.color && a.color_lane == b.color_lane && a.constant == b.constant &&
         a.constant_lane == b.constant_lane;
}

// Input `slot` of the draw as an attribute (and its input); false if none.
bool Input(const ShaderDraw& d, int slot, RtAttribute* a, VertexInput* vin) {
  if (slot < 0 || uint32_t(slot) >= d.input_count || !d.inputs) return false;
  const VertexInput in = d.inputs[slot];
  if (in.type >= 17 || in.stream >= 16 || !d.streams[in.stream].data) return false;
  const VertexStream& st = d.streams[in.stream];
  a->data = st.data + in.offset;
  a->size = st.size > in.offset ? st.size - in.offset : 0;
  a->stride = st.stride;
  a->type = in.type;
  *vin = in;
  return true;
}

// Whether every product in [0, 1] passes (the tests are monotonic in it,
// but EQUAL and NOTEQUAL).
bool AlwaysPasses(const RtMaterial& m, float lo = 0.0f) {
  const uint32_t func = (m.flags >> 8) & 15u;
  if (func == 3 || func == 6) return false;
  return RtAlphaPasses(m, lo) && RtAlphaPasses(m, 1.0f);
}

}  // namespace

RtAlphaResult RtResolveAlpha(const ShaderDraw& d, const RtAlphaShaders& sh, const RtAlphaTexture textures[16],
                             bool test, uint32_t func, uint32_t ref) {
  RtAlphaResult r;
  const bool tested = test && func != 8;
  if (!tested && !sh.discards) {
    r.kind = RtAlphaKind::Opaque;
    return r;
  }
  if (tested && func == 1) return r;  // NEVER
  // D3D9's usual arrangement, from sampler `sampler` and input TEXCOORD<tc>.
  auto usual = [&](uint32_t sampler, uint32_t tc, Source* s) {
    *s = Source{};
    if (tc >= 8 || sh.tex_slot[tc] < 0) return false;
    s->sampler = int(sampler);
    s->uv = sh.tex_slot[tc];
    s->color = sh.color0_slot;
    s->func = tested ? func : 5;  // (GREATER 0: a discard not followed)
    s->ref = tested ? ref & 0xFF : 0;
    return true;
  };
  Source src;
  if (sh.hook) {
    if (!usual(sh.hook_sampler, sh.hook_texcoord, &src)) return r;
  } else {
    const hlsl::PixelAlpha* pa = sh.pixel;
    if (tested && pa && pa->color0.known) {
      if (!Value(d, sh, pa->color0, &src.scale, &src.offset, &src)) return r;  // known, not followed back
      src.func = func;
      src.ref = ref & 0xFF;
      if (sh.discards && pa->kill.known && Same(pa->kill, pa->color0))
        src.kill = Value(d, sh, pa->kill, &src.kill_scale, &src.kill_offset, &src);
    } else if (!tested && pa && pa->kill.known) {
      if (!Value(d, sh, pa->kill, &src.kill_scale, &src.kill_offset, &src)) return r;
      src.kill = true;
    } else if (!tested && !(pa && pa->kill_reads_texel)) {
      r.kind = RtAlphaKind::Uncut;
      return r;
    } else {
      if (!usual(0, 0, &src)) return r;
      r.generic = true;
    }
  }
  // The texture: 2D, as the pixel shader declares the sampler; one whose
  // alpha reads 1 (X8R8G8B8) read as none where the alpha is the channel.
  const uint32_t* ss = nullptr;
  bool tex = false;
  if (src.sampler >= 0) {
    if (src.sampler >= 16 || sh.sampler_type[size_t(src.sampler)] != hlsl::ParamType::Sampler2D) return r;
    const RtAlphaTexture& t = textures[src.sampler];
    if (!t.present || !t.flat) return r;
    tex = !(t.opaque && src.channel == 3);
    ss = d.sampler_states ? d.sampler_states[src.sampler] : nullptr;
  }
  if (tex && !Input(d, src.uv, &r.uv, &r.uv_in)) return r;
  const bool valpha = src.color >= 0 && Input(d, src.color, &r.alpha, &r.alpha_in);
  if (tex && valpha && r.uv_in.stream != r.alpha_in.stream) return r;
  if (tex) {
    r.uv.lane[0] = src.u;
    r.uv.lane[1] = src.v;
    r.stream = int(r.uv_in.stream);
  } else {
    r.uv = RtAttribute{};
  }
  if (valpha) {
    r.alpha.lane[0] = src.color_lane;
    if (r.stream < 0) r.stream = int(r.alpha_in.stream);
  } else {
    r.alpha = RtAttribute{};
  }
  RtMaterial& m = r.material;
  m.alpha_scale = src.scale;
  m.alpha_offset = src.offset;
  m.kill_scale = src.kill_scale;
  m.kill_offset = src.kill_offset;
  static constexpr uint32_t kDefaults[14] = {0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0};  // D3D9's
  if (!ss) ss = kDefaults;
  m.max_level = float(ss[9]);  // MAXMIPLEVEL
  m.flags = RtMaterialFlags((valpha ? kRtMatVertexAlpha : 0) | (src.kill ? kRtMatKill : 0) |
                                (ss[5] >= 2 ? kRtMatLinear : 0) |                 // MAGFILTER
                                ((ss[4] >> 24) >= 0x80 ? kRtMatBorderAlpha : 0),  // BORDERCOLOR's alpha, as Metal's
                            src.func, src.ref, src.channel, ss[1], ss[2]);       // ADDRESSU, ADDRESSV
  // Nothing that varies over the draw: its alpha is the same everywhere.
  if (!tex && !valpha) {
    r.kind = RtAlphaPasses(m, 1.0f) ? RtAlphaKind::Opaque : RtAlphaKind::Skip;
    return r;
  }
  // (Only where the product stays in 0..1: no float texture, a vertex
  // alpha of a normalized unsigned type: D3DCOLOR, UBYTE4N, USHORT2N/4N;
  // a float colour reaches a ps_3_0 shader unclamped.)
  const bool unit_alpha = !valpha || r.alpha.type == 4 || r.alpha.type == 8 || r.alpha.type == 11 ||
                          r.alpha.type == 12;
  // The product's least value: a texture's alpha alone (no vertex alpha)
  // goes no lower than its least texel.
  const float lo = tex && !valpha && src.channel == 3 ? std::clamp(textures[src.sampler].alpha_min, 0.0f, 1.0f) : 0.0f;
  if ((!tex || textures[src.sampler].unit) && unit_alpha && AlwaysPasses(m, lo)) {
    r.kind = RtAlphaKind::Opaque;
    return r;
  }
  r.sampler = tex ? src.sampler : -1;
  r.kind = RtAlphaKind::Cut;
  return r;
}

RtAlbedoResult RtResolveAlbedo(const ShaderDraw& d, const RtAlphaShaders& sh, const RtAlphaTexture textures[16],
                               const RtAlphaResult* cut) {
  RtAlbedoResult r;
  const bool cut_uv = cut && cut->kind == RtAlphaKind::Cut && cut->uv.data;
  int sampler = -1, slot = -1;
  uint8_t lu = 0, lv = 1;
  if (sh.albedo_hook) {
    if (sh.albedo_texcoord >= 8) return r;
    sampler = int(sh.albedo_sampler);
    slot = sh.tex_slot[sh.albedo_texcoord];
  } else if (cut_uv && cut->sampler >= 0) {
    sampler = cut->sampler;
    r.uv = cut->uv;
    r.uv_in = cut->uv_in;
    r.stream = int(cut->uv_in.stream);
  } else {
    sampler = 0;
    slot = sh.tex_slot[0];
  }
  if (sampler < 0 || sampler >= 16 || sh.sampler_type[size_t(sampler)] != hlsl::ParamType::Sampler2D) return r;
  const RtAlphaTexture& t = textures[sampler];
  if (!t.present || !t.flat) return r;
  if (r.stream < 0) {
    if (!Input(d, slot, &r.uv, &r.uv_in)) return r;
    r.uv.lane[0] = lu;
    r.uv.lane[1] = lv;
    r.stream = int(r.uv_in.stream);
    // An alpha-cut draw's triangles keep its alpha's UVs: the same ones only.
    if (cut_uv && (r.uv.data != cut->uv.data || r.uv.type != cut->uv.type || r.uv.stride != cut->uv.stride ||
                   cut->uv.lane[0] != lu || cut->uv.lane[1] != lv))
      return r;
  }
  static constexpr uint32_t kDefaults[14] = {0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0};  // D3D9's
  const uint32_t* ss = d.sampler_states ? d.sampler_states[sampler] : kDefaults;
  r.albedo.max_level = float(ss[9]);  // MAXMIPLEVEL
  r.albedo.address_u = ss[1];         // ADDRESSU
  r.albedo.address_v = ss[2];         // ADDRESSV
  r.albedo.border = ss[4];            // BORDERCOLOR
  r.sampler = sampler;
  r.on = true;
  return r;
}

}  // namespace recomp::gfx
