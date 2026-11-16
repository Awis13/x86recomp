#pragma once
// HLSL as D3DX 9's compiler accepts it (shader models 1.1-3.0: the
// D3DXCompileShader dialect, effect syntax skipped) translated to GLSL
// 4.50 for Vulkan-style SPIR-V (and MSL through SPIRV-Cross). Generated
// from source, so the translation keeps Direct3D 9's interface instead of
// a compiled shader's: the constant registers the application sets, the
// constant table it queries, and vertex input / varying semantics.
//
// The GLSL is a body without #version: the graphics backend prepends
// `#version 450` and a prelude that defines
//   R_FLOAT_BINDING, R_INT_BINDING  layout qualifiers (set, binding) of the
//                                   uniform blocks below;
//   R_SAMPLER_BINDING(n)            layout qualifiers of sampler register n;
//   vec4 r_fetch(int slot)          vertex shaders: input `slot` (see
//                                   Program::inputs) as the input register
//                                   holds it, defaults for missing ones;
//   vec4 r_position(vec4 p)         vertex shaders: the POSITION output to
//                                   gl_Position (clip-space conventions);
//   vec2 r_vpos(), float r_vface()  pixel shaders: VPOS and VFACE;
//   float R_BIAS(n)                 pixel shaders: an LOD bias added where
//                                   sampler register n is sampled with an
//                                   implicit level (0.0 if the backend's
//                                   samplers apply D3D9's MIPMAPLODBIAS);
//   void r_color0(inout vec4 c)     pixel shaders that write COLOR0: called
//                                   last with it (the alpha test);
// and may define
//   float R_DEPTH_OUT(d)            pixel shaders: the depth a shader writes
//                                   (DEPTH, oDepth) as the depth buffer
//                                   stores it ((d) if not defined);
//   vec4 R_DEPTH_TEX(n, v)          what a 2D sampler register n read: v
//                                   as the shader expects it (a depth
//                                   texture stored otherwise than D3D9
//                                   does; (v) if not defined).
// The body declares, if it reads them:
//   layout(std140, R_FLOAT_BINDING) uniform r_float_regs { vec4 c[n]; };
//   layout(std140, R_INT_BINDING) uniform r_int_regs { ivec4 r_i[16];
//                                                       ivec4 r_b[4]; };
//   (bool register k is r_b[k / 4][k % 4])
//   layout(R_SAMPLER_BINDING(k)) uniform samplerXX r_s<k>;
// and varyings at fixed locations per semantic (VaryingLocation), so any
// vertex shader pairs with any pixel shader without linking.

#include <bitset>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace recomp::hlsl {

// The body's defaults for the optional R_DEPTH_OUT and R_DEPTH_TEX.
inline constexpr const char* kDepthOutDefault =
    "#ifndef R_DEPTH_OUT\n#define R_DEPTH_OUT(d) (d)\n#endif\n";
inline constexpr const char* kDepthTexDefault =
    "#ifndef R_DEPTH_TEX\n#define R_DEPTH_TEX(n, v) (v)\n#endif\n";

struct Define {
  std::string name, value;
};

enum class Stage : uint8_t { Vertex, Pixel };

// D3DXREGISTER_SET, D3DXPARAMETER_CLASS and D3DXPARAMETER_TYPE values.
enum class RegisterSet : uint32_t { Bool = 0, Int4 = 1, Float4 = 2, Sampler = 3 };
enum class ParamClass : uint32_t {
  Scalar = 0, Vector = 1, MatrixRows = 2, MatrixColumns = 3, Object = 4, Struct = 5,
};
enum class ParamType : uint32_t {
  Void = 0, Bool = 1, Int = 2, Float = 3, String = 4, Texture = 5,
  Texture1D = 6, Texture2D = 7, Texture3D = 8, TextureCube = 9,
  Sampler = 10, Sampler1D = 11, Sampler2D = 12, Sampler3D = 13, SamplerCube = 14,
};

// One entry of the constant table (D3DXCONSTANT_DESC).
struct Constant {
  std::string name;
  RegisterSet set = RegisterSet::Float4;
  uint32_t index = 0, count = 0;  // registers
  ParamClass cls = ParamClass::Scalar;
  ParamType type = ParamType::Float;
  uint32_t rows = 1, columns = 1, elements = 1, struct_members = 0, bytes = 0;
};

// D3DDECLUSAGE values, plus the pixel shader's own input semantics.
enum Usage : uint32_t {
  kPosition = 0, kBlendWeight = 1, kBlendIndices = 2, kNormal = 3, kPSize = 4,
  kTexcoord = 5, kTangent = 6, kBinormal = 7, kTessFactor = 8, kPositionT = 9,
  kColor = 10, kFog = 11, kDepth = 12, kSample = 13,
  kVPos = 0x100, kVFace = 0x101,
};

struct Semantic {
  uint32_t usage = 0, index = 0;
  bool operator==(const Semantic&) const = default;
};

// Location of a varying (vertex shader output, pixel shader input), the
// same in every shader; -1 if the semantic cannot be a varying.
int VaryingLocation(Semantic s);

struct Sampler {
  uint32_t reg = 0;
  ParamType type = ParamType::Sampler2D;  // Sampler2D, Sampler3D or SamplerCube
};

// A value a pixel shader computes (PixelAlpha), as found in its bytecode:
//   scale * P + offset + offset_k * c[offset_constant].(offset_lane)
// P being the product of what is named (each optional, 1 if none): one
// channel of a 2D texture's texel sampled at a TEXCOORD varying's two
// lanes, one lane of a COLOR varying and one lane of a float register the
// application sets. With none of them P is 0 (the value is the offset).
struct AlphaValue {
  bool known = false;
  int sampler = -1;       // the texel's sampler register (-1: no texel)
  uint8_t channel = 3;    // its channel (0-3: r, g, b, a)
  uint32_t texcoord = 0;  // sampled at varying TEXCOORDn's lanes u and v
  uint8_t u = 0, v = 1;
  int color = -1;  // varying COLORn's lane color_lane (-1: none)
  uint8_t color_lane = 3;
  int constant = -1;  // c[constant]'s lane constant_lane (-1: none)
  uint8_t constant_lane = 0;
  float scale = 0.0f, offset = 0.0f;
  int offset_constant = -1;  // -1: none
  uint8_t offset_lane = 0;
  float offset_k = 0.0f;
  bool operator==(const AlphaValue&) const = default;
};

// Pixel shaders from bytecode (TranslateOptions::coverage): where the
// coverage of a pixel comes from, for effects that need it without
// running the shader (rays meeting alpha-tested geometry). Followed
// through mov, add, mul, mad, texld (texldb, texldl, texldd; not texldp,
// not a swizzled sampler) and ps_1_x's tex, in main's straight-line code;
// a write in an if block or a predicated one may or may not happen
// (unknown after it); loops and subroutines are not followed (nothing
// known). One pass over the instructions, none for shaders of more than
// kCoverageInstructions.
struct PixelAlpha {
  AlphaValue color0;  // COLOR0's alpha, what the alpha test compares
  uint32_t kills = 0;  // texkill instructions (HLSL clip)
  // Known if there is one texkill, outside if blocks, and every component
  // it tests holds this value or a number >= 0: the pixel is discarded
  // where it is below 0.
  AlphaValue kill;
  // Whether a component a texkill tests may depend on a texel: data from
  // a texture sample whose coordinate depends on an interpolated input
  // (v#, t#; not the screen position, vFace or constants: a dither
  // pattern read at vPos is not a texel's) or on such a texel, reaching
  // it in program order (a write from clean sources that surely happens
  // clears what it writes; one predicated, in a branch on another path,
  // or in a conditional call does not; loops to a fixed point,
  // subroutines where called, a break's state carried past the loop), or
  // a condition that may depend on one (a predicate, an if, a break before
  // it in the loop, or in an earlier iteration) deciding whether it runs.
  // Found past kCoverageInstructions as well.
  bool kill_reads_texel = false;
};
inline constexpr size_t kCoverageInstructions = 4096;

// Vertex shaders from bytecode (TranslateOptions::coverage): a lane of a
// TEXCOORDn or COLORn varying that is a lane of an input moved there
// unchanged (mov, through temporaries; main's straight-line code, outside
// if blocks, not predicated).
struct MovedLane {
  Semantic varying;
  uint8_t lane = 0;
  uint8_t input = 0;       // the input's slot (Program::inputs)
  uint8_t input_lane = 0;
};

struct Program {
  Stage stage = Stage::Vertex;
  uint32_t major = 0, minor = 0;  // shader model
  std::string glsl;
  std::vector<Constant> constants;
  std::vector<Semantic> inputs;    // vertex shaders: r_fetch slot -> semantic
  std::vector<Semantic> varyings;  // written (vertex) or read (pixel)
  uint32_t color_outputs = 0;      // pixel shaders: mask of COLORn written
  bool writes_depth = false, writes_point_size = false;
  std::vector<Sampler> samplers;
  uint32_t float_regs = 0;  // c registers the GLSL declares
  // Vertex shaders from bytecode: b if the POSITION output's z and w are
  // the vertex times the matrix in float registers c[b]..c[b + 3] (rows,
  // D3D's row vectors: c[b + 3] the translation, added once), z and w
  // alike: z = sum of s_k c[b + k].z and w = the same sum of the .w, for
  // the same scalars s_k (s_3 = 1), nothing else added to either. A
  // backend may then compute the depth from w alone (gfx/reversed_z.h).
  // -1: not known to be so, and position_note says why.
  int position_matrix = -1;
  std::string position_note;
  // Bytecode: the float registers c0-c255 read by number (not through a0
  // or aL, not ones the shader defines).
  std::bitset<256> float_reads;
  // Bytecode: the lowest register a read through a0 or aL is based at (a
  // skinned vertex shader's bone palette, c[n + a0.x]); -1: none.
  int rel_const_base = -1;
  // Bytecode with TranslateOptions::coverage: pixel shaders' alpha
  // (PixelAlpha), vertex shaders' moved varying lanes (MovedLane). Not
  // serialized: HLSL source has neither.
  PixelAlpha pixel_alpha;
  std::vector<MovedLane> moved_lanes;
  // Set by TranslateBytecode without TranslateOptions::coverage: the two
  // above were not looked for. False for HLSL source (Compile,
  // Deserialize), which has neither to look for.
  bool coverage_skipped = false;
  // Bytecode with TranslateOptions::ray_reflection: the sampler whose reads
  // became calls of r_rt_reflect (RayReflectionCall); -1 if none did. Kept
  // by the copies a backend makes its variants from (it makes those calls
  // ray queries in each). The varying location of the origin's input
  // (VaryingLocation: a variant without it has no ray to trace), and how
  // many of the calls are in subroutines (a backend may give ray queries to
  // main's only). Not serialized.
  int ray_reflection_sampler = -1;
  int ray_reflection_location = -1;
  uint32_t ray_reflection_stub_calls = 0;
  // Likewise with TranslateOptions::ray_shadow: the sampler whose reads
  // became calls of r_rt_shadow (RayShadowCall), -1 if none did, and how
  // many of them are in subroutines. Not serialized.
  int ray_shadow_sampler = -1;
  uint32_t ray_shadow_stub_calls = 0;
};

// D3DXSHADER flags that change the translation.
enum : uint32_t {
  kPackMatrixRowMajor = 0x8,
  kPackMatrixColumnMajor = 0x10,
};

// Translates `source` (entry function `entry`, profile such as "vs_3_0").
// On failure returns false with fxc-style messages in *errors.
bool Compile(std::string_view source, const std::vector<Define>& defines,
             const std::string& entry, const std::string& profile, uint32_t flags,
             Program* out, std::string* errors);

// Translates a compiled D3D9 shader (shader model 1.1-3.0 tokens, as
// CreateVertexShader / CreatePixelShader take them, END included) to the
// same GLSL contract. No constant table: applications set the registers
// the bytecode reads. On failure returns false with a message in *errors.
bool TranslateBytecode(const uint32_t* tokens, size_t count, Program* out, std::string* errors);
// Pixel shaders (ps_3_0) whose reads of one sampler a backend that traces
// rays makes ray queries (TranslateOptions::ray_reflection): every texld,
// texldl and texldd of `sampler` becomes
//   r_rt_reflect(gl_FragCoord, v<origin>, coordinate, c[registers[0]],
//                c[registers[1]], c[registers[2]], c[registers[3]])
// (vec4s: the pixel's position, the input register holding the reflection
// ray's start in world space (xyz), the read's coordinate as the shader
// computed it (the ray's direction, xyz; texldp's divided by its w, as the
// read divides it), four float registers the hit is
// lit with), and the sampler is not declared. The GLSL defines the
// function as "nothing hit" (vec4(0.0)); the Metal backend replaces the
// calls (gfx/rt_forward_reflection.h). Not taken (the sampler read as
// usual) unless the shader declares input `origin` with a semantic other
// than POSITION and the registers are below 224.
struct RayReflectionCall {
  int sampler = -1;  // 0-15; -1: none
  uint32_t origin = 0;
  uint32_t registers[4] = {};
};
// Pixel shaders (ps_3_0) whose reads of one sampler a backend that traces
// rays makes shadow rays toward a far light (TranslateOptions::ray_shadow):
// every texld, texldl and texldd of `sampler` becomes
//   r_rt_shadow(gl_FragCoord, coordinate, c[light_register])
// (vec4s: the pixel's position, the read's coordinate as the shader
// computed it (the point in world space, xyz; texldp's divided by its w),
// the float register holding the light's direction), and the sampler is
// not declared. The call's x is 1 where the point sees the light, 0 where
// it is in shadow. The GLSL defines the function as "lit" (vec4(1.0)); the
// Metal backend replaces the calls (gfx/rt_forward_reflection.h). Not taken
// (the sampler read as usual) unless the register is below 224.
// A texldd of the sampler is the shadow toward a point instead, a local
// light's:
//   r_rt_shadow_to(gl_FragCoord, coordinate, ddx, ddy)
// the coordinate the point in world space (xyz), ddx the vector from it to
// the light (xyz), ddy.x the light's weight at the point: nothing is asked
// (lit) where it is not above 0. A shader may make several such reads, one
// a light.
struct RayShadowCall {
  int sampler = -1;  // 0-15; -1: none
  uint32_t light_register = 0;
};
struct TranslateOptions {
  // Program::pixel_alpha and moved_lanes found (for ray tracing; the
  // translation is the same either way).
  bool coverage = false;
  RayReflectionCall ray_reflection;
  RayShadowCall ray_shadow;
};
bool TranslateBytecode(const uint32_t* tokens, size_t count, Program* out, std::string* errors,
                       const TranslateOptions& options);

// A compiled shader function's tokens, version to END included, token i
// being read(i): D3D9 is given no length, only where the function starts.
// Instructions are stepped over by their length (shader model 2 and up:
// the count in the instruction token; 1.x: def's fixed five tokens, else
// the parameter tokens, which have bit 31 set), so operands such as def's
// floats cannot pass for END or a comment. Empty if no END within
// `limit` tokens.
template <class Read>
std::vector<uint32_t> ReadBytecode(Read read, size_t limit) {
  std::vector<uint32_t> t{read(0)};
  const bool sm1 = ((t[0] >> 8) & 0xFF) < 2;
  for (size_t i = 1; i < limit;) {
    uint32_t tok = read(i), op = tok & 0xFFFF;
    size_t n;  // tokens after the instruction token
    if (op == 0xFFFF) {  // END
      t.push_back(tok);
      return t;
    } else if (op == 0xFFFE) {  // comment
      n = (tok >> 16) & 0x7FFF;
    } else if (!sm1) {
      n = (tok >> 24) & 0xF;
    } else if (op == 81) {  // def
      n = 5;
    } else {
      n = 0;
      while (i + 1 + n < limit && (read(i + 1 + n) & 0x80000000u)) n++;
    }
    if (i + 1 + n > limit) break;
    for (size_t k = 0; k <= n; k++) t.push_back(read(i + k));
    i += 1 + n;
  }
  return {};
}

// The value of an output no shader instruction writes, as the D3D9
// runtimes on other APIs (DXVK, wined3d) set it: COLOR0 (1, 1, 1, 1),
// other colours (0, 0, 0, 1), FOG 1, the rest 0 (a GLSL vec4).
const char* UnwrittenVarying(Semantic s);

// The varying locations a shader writes (vertex) or reads (pixel), one
// bit per VaryingLocation.
uint32_t VaryingMask(const Program& p);

// A pixel shader's GLSL for a vertex shader that writes only the varying
// locations in `written`: inputs it reads from other locations become
// constants (UnwrittenVarying), as Metal rejects a pipeline whose
// fragment inputs the vertex function lacks while D3D9 allows it.
std::string GlslWithDefaultInputs(const Program& ps, uint32_t written);

// A Program as bytes and back (shader containers).
std::vector<uint8_t> Serialize(const Program& p);
bool Deserialize(const uint8_t* data, size_t size, Program* p);

}  // namespace recomp::hlsl
