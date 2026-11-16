#pragma once
// Where the alpha of a draw that discards by it comes from, for the
// ray-traced scene (gfx/rt_scene.h, RtCut), and its albedo (RtAlbedo): from
// the draw (gfx::ShaderDraw) and what its shaders say (hlsl::PixelAlpha,
// MovedLane). On the CPU, apart from the Metal backend that calls it, so it
// is tested alone.

#if !__has_feature(objc_arc)
#error "rt_alpha.h needs ARC (-fobjc-arc)"
#endif

#include <array>
#include <cstdint>
#include <vector>

#include "runtime/gfx/device.h"
#include "runtime/gfx/rt_scene.h"
#include "runtime/hlsl/hlsl.h"

namespace recomp::gfx {

// What RtResolveAlpha reads of a draw's shaders.
struct RtAlphaShaders {
  // The pixel shader: its bytecode's alpha (null: none, as HLSL source
  // has), whether it may discard (texkill, clip, discard), its samplers'
  // types by register.
  const hlsl::PixelAlpha* pixel = nullptr;
  bool discards = false;
  std::array<hlsl::ParamType, 16> sampler_type{};
  // The vertex shader: its moved varying lanes (null: none), the input
  // slots of TEXCOORD0-7 and COLOR0 (-1: none).
  const std::vector<hlsl::MovedLane>* moved = nullptr;
  std::array<int8_t, 8> tex_slot{-1, -1, -1, -1, -1, -1, -1, -1};
  int color0_slot = -1;
  // The game's answer for this pixel shader (RayTracing::alpha_source):
  // the sampler register and the input TEXCOORD<n>.
  bool hook = false;
  uint32_t hook_sampler = 0, hook_texcoord = 0;
  // The game's answer for the albedo (RayTracing::albedo_source), likewise.
  bool albedo_hook = false;
  uint32_t albedo_sampler = 0, albedo_texcoord = 0;
};

// A pixel sampler's texture, as RtResolveAlpha sees it.
struct RtAlphaTexture {
  bool present = false;
  bool flat = false;    // 2D (not a cube, a volume or a depth texture)
  bool opaque = false;  // alpha read as 1 (X8R8G8B8)
  bool unit = true;     // its values in 0..1 (not a float format)
  // The least alpha its texels read as (0..1; 0 where not known): a draw
  // whose alpha is this texture's alone is opaque where the test passes it.
  float alpha_min = 0.0f;
};

enum class RtAlphaKind : uint8_t {
  Opaque,  // nothing discarded, or the alpha passes everywhere: added opaque
  Cut,     // alpha-cut (RtAlphaResult)
  Skip,    // left out (RtSkip::AlphaSource): never passes, or not resolved
  Uncut,   // discards by what is not known to be a texture's alpha: added opaque, counted
};

struct RtAlphaResult {
  RtAlphaKind kind = RtAlphaKind::Skip;
  bool generic = false;  // from D3D9's usual arrangement (no hook, the bytecode not followed)
  int sampler = -1;      // the texture read (-1: none, texel 1)
  // The vertex inputs read (RtGeometry::uv, alpha; data null: none), and
  // their stream (the UVs', or without UVs the alpha's; -1: none).
  RtAttribute uv, alpha;
  VertexInput uv_in, alpha_in;
  int stream = -1;
  RtMaterial material{};  // alpha_texture left for the scene
};

// The alpha source of draw `d` with the alpha test `test` (ALPHATESTENABLE)
// by D3DCMPFUNC `func` against `ref` (ALPHAREF), `textures` its pixel
// samplers' textures. In order:
// - the game's hook (sh.hook);
// - the pixel shader's bytecode (hlsl::PixelAlpha): with the alpha test,
//   COLOR0's alpha; without it, the texkill's operand; the texel's UVs and
//   the colour factor followed back to vertex inputs through the vertex
//   shader's moved lanes, constants from the draw's registers. Known there
//   but not followed back (UVs the vertex shader computes, say): Skip, not
//   a guess against it. A texkill with the alpha test counts where it
//   tests the same product, else is left to the test;
// - with the alpha test and nothing known: D3D9's usual arrangement, as
//   dxvk-remix takes it: sampler 0's alpha at the vertex input TEXCOORD0's
//   x and y, times COLOR0's alpha where the vertex has one;
// - a discard without the alpha test and not known: if it may depend on a
//   texel (PixelAlpha::kill_reads_texel), the usual arrangement cut where
//   it is 0 (GREATER 0), an approximation; else Uncut (a clip plane, a
//   screen-door fade by position, a dither texture read at it, HLSL
//   source).
// Skip also for ALPHAFUNC NEVER, a texture missing or not 2D at the
// sampler, an input missing, the UVs and the colour factor in different
// streams, and an alpha the same everywhere that never passes. Opaque
// where the test passes for every alpha the product can take, from 0 to 1
// (GREATEREQUAL 0, LESSEQUAL 255; not EQUAL or NOTEQUAL; not with a
// texture or vertex alpha whose values may lie outside: a float format's);
// where the alpha is a texture's alone, from its least alpha (alpha_min)
// to 1: a DXT texture with no texel the test discards is opaque.
RtAlphaResult RtResolveAlpha(const ShaderDraw& d, const RtAlphaShaders& sh, const RtAlphaTexture textures[16],
                             bool test, uint32_t func, uint32_t ref);

// Where a draw's albedo comes from (scenes with hit data: RtAlbedo), and
// the UVs it is read at.
struct RtAlbedoResult {
  bool on = false;
  int sampler = -1;
  RtAttribute uv;  // RtGeometry::uv (lanes set)
  VertexInput uv_in;
  int stream = -1;  // the UVs' stream
  RtAlbedo albedo;  // texture left for the backend (the sampler's)
};

// The albedo of draw `d`, `cut` what RtResolveAlpha made of it (null: not
// asked), textures as there. In order:
// - the game's hook (sh.albedo_hook: its sampler and input TEXCOORD<n>);
// - an alpha-cut draw's alpha texture (cut->kind Cut, a sampler): for the
//   usual shaders the alpha that discards is the diffuse texture's;
// - D3D9's usual arrangement: sampler 0 at the input TEXCOORD0's x and y.
// Off where the sampler is not a 2D texture the shader declares, the
// input is missing, or the draw is alpha-cut with UVs from other inputs
// or lanes (a triangle keeps one set of UVs). The vertex colour is not
// read (RtMaterial: it does not multiply the albedo).
RtAlbedoResult RtResolveAlbedo(const ShaderDraw& d, const RtAlphaShaders& sh, const RtAlphaTexture textures[16],
                               const RtAlphaResult* cut);

}  // namespace recomp::gfx
