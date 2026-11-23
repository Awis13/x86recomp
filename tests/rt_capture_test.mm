// Captured draws in the ray-traced scene (gfx/rt_scene.h, AddCaptured;
// hlsl/capture_position.h). Hand-assembled vertex shaders, translated and
// compiled as the Metal backend compiles them (the capture variant with
// rasterization off), drawn in a render pass with colour and depth
// attachments as a game's, each draw followed by its capture in the same
// encoder (run with Metal's API validation, MTL_DEBUG_LAYER=1, from ctest):
// a vs_3_0 skinning with a bone palette at c64 and up (relative reads
// through a0) and displacing by a vertex texture (texldl), its vertices
// read in place (base vertex in the vertex numbers) and copied (not), with
// a camera 4 km from the origin. Checked: the world positions against the
// CPU (1 cm), rays hitting the captured instances where expected with
// D3D's front side (as the screen's winding says, mirrored view-projection
// included), a draw split per instance (a stride-0 instance stream) as an
// instance each, a snapshot's carried capture, no camera, the budget, the
// BLAS memory pooled across frames, and an alpha-cut capture (its UVs from
// the vertex stream). Rays as effects trace them (kRtMsl's RtTraceClosest).
//
//   rt_capture_test        (skipped, exit 77, without a GPU that traces rays)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "runtime/gfx/metal_prelude.h"
#include "runtime/gfx/rt_scene.h"
#include "runtime/gfx/shader_compile.h"
#include "runtime/hlsl/capture_position.h"
#include "runtime/hlsl/hlsl.h"

namespace {

using namespace recomp;
using Tokens = std::vector<uint32_t>;

int Check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what);
  return ok ? 0 : 1;
}

// ---- hand-assembled shaders (as tests/metal_depth_test.mm)

enum Reg : uint32_t { R = 0, V = 1, C = 2, A = 3, O = 6, OC = 8, S = 10 };
enum Op : uint32_t { MOV = 1, ADD = 2, MAD = 4, MUL = 5, MOVA = 46, TEXLDL = 95 };
constexpr uint32_t kX = 0x00, kY = 0x55, kZ = 0xAA;

uint32_t TypeBits(uint32_t type) { return ((type & 7) << 28) | ((type & 0x18) << 8); }
uint32_t Dst(uint32_t type, uint32_t n, uint32_t mask = 0xF) { return 0x80000000u | n | TypeBits(type) | mask << 16; }
uint32_t Src(uint32_t type, uint32_t n, uint32_t swz = 0xE4) { return 0x80000000u | n | TypeBits(type) | swz << 16; }
// c[a0.x + n] (vs_3_0: the address register token follows).
uint32_t Rel(uint32_t n) { return Src(C, n) | 0x2000u; }
constexpr uint32_t kA0x = 0x80000000u | (3u << 28);  // a0.x

struct Asm {
  Tokens t;
  explicit Asm(bool pixel) { t.push_back((pixel ? 0xFFFF0000u : 0xFFFE0000u) | 3u << 8); }
  void I(uint32_t op, std::initializer_list<uint32_t> params) {
    t.push_back(op | uint32_t(params.size()) << 24);
    t.insert(t.end(), params);
  }
  void Dcl(uint32_t usage, uint32_t index, uint32_t dst) { I(31, {0x80000000u | usage | index << 16, dst}); }
  void DclSampler(uint32_t type, uint32_t n) { I(31, {0x80000000u | type << 27, Dst(S, n)}); }
  Tokens End() {
    t.push_back(0x0000FFFF);
    return t;
  }
  // o0 = r(n) * c8..c11 (the view-projection's rows), w taken as 1.
  void ToClip(uint32_t n) {
    I(MUL, {Dst(R, 2), Src(R, n, kY), Src(C, 9)});
    I(MAD, {Dst(R, 2), Src(R, n, kX), Src(C, 8), Src(R, 2)});
    I(MAD, {Dst(R, 2), Src(R, n, kZ), Src(C, 10), Src(R, 2)});
    I(ADD, {Dst(O, 0), Src(R, 2), Src(C, 11)});
  }
};

// vs_3_0 skinning: bone v1.x / 4 of the palette at c64 (4 rows each, read
// as c[a0.x + 64..67]), then displaced by s0's red at TEXCOORD0 (texldl)
// along c5; to clip space by c8-c11.
Tokens SkinVs() {
  Asm a(false);
  a.Dcl(0, 0, Dst(V, 0));  // POSITION
  a.Dcl(2, 0, Dst(V, 1));  // BLENDINDICES
  a.Dcl(5, 0, Dst(V, 2));  // TEXCOORD0
  a.Dcl(0, 0, Dst(O, 0));
  a.DclSampler(2, 0);
  a.I(MOVA, {Dst(A, 0, 1), Src(V, 1, kX)});
  a.I(MUL, {Dst(R, 0), Src(V, 0, kY), Rel(65), kA0x});
  a.I(MAD, {Dst(R, 0), Src(V, 0, kX), Rel(64), kA0x, Src(R, 0)});
  a.I(MAD, {Dst(R, 0), Src(V, 0, kZ), Rel(66), kA0x, Src(R, 0)});
  a.I(ADD, {Dst(R, 0), Src(R, 0), Rel(67), kA0x});
  a.I(TEXLDL, {Dst(R, 1), Src(V, 2), Src(S, 0)});
  a.I(MAD, {Dst(R, 0, 0x7), Src(R, 1, kX), Src(C, 5), Src(R, 0)});
  a.ToClip(0);
  return a.End();
}

// vs_3_0 instancing: POSITION plus the instance's offset (TEXCOORD1).
Tokens InstanceVs() {
  Asm a(false);
  a.Dcl(0, 0, Dst(V, 0));
  a.Dcl(5, 1, Dst(V, 1));
  a.Dcl(0, 0, Dst(O, 0));
  a.I(ADD, {Dst(R, 0), Src(V, 0), Src(V, 1)});
  a.ToClip(0);
  return a.End();
}

// ps_3_0: c0 as the colour.
Tokens ConstPs() {
  Asm a(true);
  a.I(MOV, {Dst(OC, 0), Src(C, 0)});
  return a.End();
}

// ---- matrices (D3D's row vectors), in double

struct M4 {
  double m[4][4] = {};
};
M4 Identity() {
  M4 r;
  for (int i = 0; i < 4; i++) r.m[i][i] = 1;
  return r;
}
M4 Mul(const M4& a, const M4& b) {
  M4 r;
  for (int i = 0; i < 4; i++)
    for (int k = 0; k < 4; k++)
      for (int j = 0; j < 4; j++) r.m[i][k] += a.m[i][j] * b.m[j][k];
  return r;
}
M4 Translation(double x, double y, double z) {
  M4 r = Identity();
  r.m[3][0] = x, r.m[3][1] = y, r.m[3][2] = z;
  return r;
}
M4 RotY(double a) {  // +z to (sin a, 0, cos a)
  M4 r = Identity();
  r.m[0][0] = std::cos(a), r.m[0][2] = -std::sin(a);
  r.m[2][0] = std::sin(a), r.m[2][2] = std::cos(a);
  return r;
}
M4 Perspective(double fov_y, double aspect, double n, double f) {
  M4 r;
  const double ys = 1.0 / std::tan(fov_y * 0.5);
  r.m[0][0] = ys / aspect;
  r.m[1][1] = ys;
  r.m[2][2] = f / (f - n);
  r.m[2][3] = 1;
  r.m[3][2] = -n * f / (f - n);
  return r;
}
// The matrix as the GPU gets it (float), back in double.
M4 AsFloat(const M4& a) {
  M4 r;
  for (int i = 0; i < 4; i++)
    for (int k = 0; k < 4; k++) r.m[i][k] = double(float(a.m[i][k]));
  return r;
}
std::array<double, 4> Apply(const M4& m, const std::array<double, 3>& p) {
  std::array<double, 4> o{};
  for (int k = 0; k < 4; k++) o[k] = p[0] * m.m[0][k] + p[1] * m.m[1][k] + p[2] * m.m[2][k] + m.m[3][k];
  return o;
}
void Rows(const M4& m, float out[4][4]) {
  for (int i = 0; i < 4; i++)
    for (int k = 0; k < 4; k++) out[i][k] = float(m.m[i][k]);
}

// The camera: 4 km out, yawed, looking along (sin 0.4, 0, cos 0.4).
constexpr double kYaw = 0.4;
const std::array<double, 3> kEye = {4000.25, 31.5, 4100.75};
std::array<double, 3> Ahead(double d, double right = 0, double up = 0) {
  const double s = std::sin(kYaw), c = std::cos(kYaw);
  return {kEye[0] + s * d + c * right, kEye[1] + up, kEye[2] + c * d - s * right};
}
M4 ViewProj(bool mirrored) {
  M4 vp = Mul(Mul(Translation(-kEye[0], -kEye[1], -kEye[2]), RotY(-kYaw)), Perspective(1.0, 1.0, 0.5, 2000.0));
  if (mirrored)
    for (auto& row : vp.m) row[0] = -row[0];
  return AsFloat(vp);
}

// ---- the GPU

// (After gfx::kRtMsl: alpha-cut candidates tested.)
constexpr const char* kTraceMsl = R"(
struct Query { float4 o; float4 d; };  // o.w: the ray's length; d.w: 1 to cull back faces
struct Hit { float t; uint id; uint front; uint prim; };
kernel void trace_cs(instance_acceleration_structure scene [[buffer(0)]],
                     const device Query* q [[buffer(1)]], device Hit* out [[buffer(2)]],
                     const device RtMaterial* materials [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  ray r(q[i].o.xyz, q[i].d.xyz, 0.0, q[i].o.w);
  intersection_params ip;
  if (q[i].d.w > 0.5) ip.set_triangle_cull_mode(triangle_cull_mode::back);
  const RtHit h = RtTraceClosest(scene, r, 0xFF, materials, ip);
  Hit o = {};
  o.t = h.hit ? h.distance : -1.0;
  o.id = h.instance;
  o.front = h.front ? 1 : 0;
  o.prim = h.primitive;
  out[i] = o;
}
// kRtMsl's rt_shadowed both ways (o.w: tmax, d.w: the gap): t the one
// traversal's answer (with the info), id the two phases', front whether
// the one traversal's first hit was a captured instance within the gap.
kernel void shadow_cs(instance_acceleration_structure scene [[buffer(0)]],
                      const device Query* q [[buffer(1)]], device Hit* out [[buffer(2)]],
                      const device RtMaterial* materials [[buffer(3)]], const device RtInfo* info [[buffer(4)]],
                      uint i [[thread_position_in_grid]]) {
  const float3 o = q[i].o.xyz, l = normalize(q[i].d.xyz);
  const float tmax = q[i].o.w, gap = q[i].d.w;
  Hit h = {};
  h.t = rt_shadowed(scene, o, l, tmax, gap, materials, info) ? 1.0 : 0.0;
  h.id = rt_shadowed(scene, o, l, tmax, gap, materials) ? 1u : 0u;
  const RtHit a = RtTraceClosest(scene, ray(o, l, 0.0, tmax), 0xFF, materials, intersection_params());
  h.front = a.hit && a.distance < gap && (info[a.instance].flags & (1u << 17)) != 0u ? 1u : 0u;
  out[i] = h;
}
)";

struct Query {
  float o[4], d[4];
};
struct Hit {
  float t;
  uint32_t id, front, prim;
};

// The prelude's vertex state block (std140), as metal_device.mm fills it.
struct VsState {
  int32_t in[16][4];
  float pixel[4];
  float clip[6][4];
  int32_t clip_mask[4];
  float depth[4];
};
struct PsState {
  int32_t alpha[4];
  float bias[16];
  float vscale[4];
};

// A vertex of the test's buffers: POSITION, BLENDINDICES, TEXCOORD0,
// TEXCOORD1, FLOAT4 each.
struct Vertex {
  float pos[4], blend[4], uv[4], inst[4];
};
constexpr uint32_t kStride = sizeof(Vertex);

struct Shader {
  std::vector<hlsl::Semantic> inputs;
  id<MTLFunction> fn = nil, capture = nil;
};

struct Arena {
  id<MTLBuffer> buf = nil;
  size_t used = 0;
};
gfx::RtUpload ArenaAlloc(void* ctx, size_t bytes, size_t align) {
  auto* a = static_cast<Arena*>(ctx);
  const size_t off = (a->used + align - 1) & ~(align - 1);
  if (off + bytes > a->buf.length) return {nil, 0, nullptr};
  a->used = off + bytes;
  return {a->buf, off, static_cast<uint8_t*>(a->buf.contents) + off};
}

struct Gpu {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [dev newCommandQueue];
  id<MTLComputePipelineState> trace = nil, shadow = nil;
  id<MTLRenderPipelineState> skin_draw = nil, skin_capture = nil, inst_draw = nil, inst_capture = nil;
  id<MTLDepthStencilState> ds = nil;
  id<MTLTexture> color = nil, depth = nil, vtex = nil;
  id<MTLSamplerState> smp = nil;
  Shader skin, inst;
  Arena arena;

  id<MTLFunction> Msl(const std::string& glsl, bool vertex, bool capture) {
    std::vector<uint32_t> spirv;
    std::string msl, log;
    gfx::SpirvMslOptions o;
    o.disable_rasterization = capture;
    if (!gfx::GlslToSpirv(glsl, vertex ? gfx::ShaderStage::Vertex : gfx::ShaderStage::Fragment, &spirv, &log) ||
        !gfx::SpirvToMsl(spirv, &msl, &log, o)) {
      std::printf("FAIL compile: %s\n%s\n", log.c_str(), glsl.c_str());
      return nil;
    }
    MTLCompileOptions* co = nil;
    if (gfx::MetalLeanMacros(&msl, vertex)) {
      co = [MTLCompileOptions new];
      co.preprocessorMacros = @{@"R_ALPHA_TEST" : @"true"};
    }
    NSError* e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(msl.c_str()) options:co error:&e];
    if (!lib) {
      std::printf("FAIL MSL: %s\n%s\n", e.localizedDescription.UTF8String, msl.c_str());
      return nil;
    }
    return [lib newFunctionWithName:@"main0"];
  }

  // The shader as the backend compiles it, and a vertex shader's capture
  // variant as it does.
  bool Compile(const Tokens& tokens, Shader* out) {
    hlsl::Program p;
    std::string err;
    if (!hlsl::TranslateBytecode(tokens.data(), tokens.size(), &p, &err)) {
      std::printf("FAIL translate: %s\n", err.c_str());
      return false;
    }
    const bool vertex = p.stage == hlsl::Stage::Vertex;
    out->inputs = p.inputs;
    out->fn = Msl("#version 450\n" + gfx::MetalShaderPrelude(p.stage) + p.glsl, vertex, false);
    if (!vertex) return out->fn != nil;
    const std::string v = hlsl::VertexWithCapturedPosition(p);
    if (v.empty()) {
      std::printf("FAIL no capture variant\n");
      return false;
    }
    out->capture = Msl("#version 450\n" + gfx::MetalShaderPrelude(p.stage) + gfx::MetalCaptureDefines() + v, true, true);
    return out->fn && out->capture;
  }

  id<MTLRenderPipelineState> Pipe(id<MTLFunction> vs, id<MTLFunction> ps) {
    MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = vs;
    if (ps) {
      pd.fragmentFunction = ps;
      pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
      pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
      pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    } else {
      // As metal_device.mm's kCapture pipelines: no attachments.
      pd.rasterizationEnabled = NO;
    }
    NSError* e = nil;
    id<MTLRenderPipelineState> p = [dev newRenderPipelineStateWithDescriptor:pd error:&e];
    if (!p) std::printf("FAIL pipeline: %s\n", e.localizedDescription.UTF8String);
    return p;
  }

  bool Init() {
    Shader ps;
    if (!Compile(SkinVs(), &skin) || !Compile(InstanceVs(), &inst) || !Compile(ConstPs(), &ps)) return false;
    skin_draw = Pipe(skin.fn, ps.fn);
    skin_capture = Pipe(skin.capture, nil);
    inst_draw = Pipe(inst.fn, ps.fn);
    inst_capture = Pipe(inst.capture, nil);
    NSError* e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@((std::string(gfx::kRtMsl) + kTraceMsl).c_str())
                                           options:nil
                                             error:&e];
    if (!lib) {
      std::printf("FAIL trace kernel: %s\n", e.localizedDescription.UTF8String);
      return false;
    }
    trace = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"trace_cs"] error:&e];
    shadow = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"shadow_cs"] error:&e];
    MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
    dd.depthCompareFunction = MTLCompareFunctionLessEqual;
    dd.depthWriteEnabled = YES;
    ds = [dev newDepthStencilStateWithDescriptor:dd];
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                  width:8
                                                                                 height:8
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget;
    color = [dev newTextureWithDescriptor:td];
    td.pixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    depth = [dev newTextureWithDescriptor:td];
    // The vertex texture: 4 x 1, red 0, 0.25, 0.5, 0.75 (nearest).
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:4 height:1 mipmapped:NO];
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead;
    vtex = [dev newTextureWithDescriptor:td];
    float texels[16] = {};
    for (int i = 0; i < 4; i++) texels[i * 4] = 0.25f * float(i);
    [vtex replaceRegion:MTLRegionMake2D(0, 0, 4, 1) mipmapLevel:0 withBytes:texels bytesPerRow:64];
    smp = [dev newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
    arena.buf = [dev newBufferWithLength:16u << 20 options:MTLResourceStorageModeShared];
    return skin_draw && skin_capture && inst_draw && inst_capture && trace && arena.buf;
  }

  std::unique_ptr<gfx::RtScene> Scene() {
    std::unique_ptr<gfx::RtScene> s = gfx::MakeRtScene(dev, {});
    if (s) s->SetAllocator(ArenaAlloc, &arena);
    return s;
  }

  bool Commit(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
      std::printf("FAIL command buffer: %s\n", cb.error.localizedDescription.UTF8String);
      return false;
    }
    return true;
  }

  std::vector<Hit> Shadow(const gfx::RtFrame* f, const std::vector<Query>& q) { return Trace(f, q, true); }

  std::vector<Hit> Trace(const gfx::RtFrame* f, const std::vector<Query>& q, bool shadows = false) {
    std::vector<Hit> out(q.size());
    for (Hit& h : out) h.t = -2;
    if (!f || !f->tlas || q.empty()) return out;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBuffer> qb = [dev newBufferWithBytes:q.data() length:q.size() * sizeof(Query) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hb = [dev newBufferWithLength:q.size() * sizeof(Hit) options:MTLResourceStorageModeShared];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:shadows ? shadow : trace];
    [e setAccelerationStructure:f->tlas atBufferIndex:0];
    [e setBuffer:f->info offset:f->info_offset atIndex:4];
    gfx::RtUseResources(e, f);
    [e setBuffer:qb offset:0 atIndex:1];
    [e setBuffer:hb offset:0 atIndex:2];
    [e setBuffer:f->materials offset:f->materials_offset atIndex:3];
    [e dispatchThreads:MTLSizeMake(q.size(), 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [e endEncoding];
    if (Commit(cb)) std::memcpy(out.data(), hb.contents, out.size() * sizeof(Hit));
    return out;
  }
};

// A draw as a game's, then its capture, in the same encoder: buffers 0-3
// and the vertex texture bound once for both, as the backend leaves them.
struct Draw {
  const Shader* sh;
  id<MTLRenderPipelineState> draw, capture;
  id<MTLBuffer> vb;            // vertices (r_vb)
  int32_t vertex0 = 0;         // byte offset of vertex 0 in vb (negative for a copy)
  std::vector<uint16_t> ib;    // raw indices
  int32_t base_vertex = 0;     // given to Metal (in place) or 0 (copied)
  int32_t instance_at = -1;    // byte offset of the instance's element (a stride-0 stream), or -1
  std::vector<float> c;        // c0-c255
  gfx::RtCaptureSlot slot;
  uint32_t start = 0, count = 0;  // the capture's vertices, as the draw numbers them
};

void Encode(Gpu& g, id<MTLRenderCommandEncoder> e, const Draw& d) {
  VsState st{};
  for (auto& in : st.in) in[2] = -1;
  for (size_t i = 0; i < d.sh->inputs.size(); i++) {
    const hlsl::Semantic s = d.sh->inputs[i];
    const int32_t field = s.usage == hlsl::kPosition ? 0 : s.usage == hlsl::kBlendIndices ? 16 : s.index == 0 ? 32 : 48;
    const bool instance = s.usage == hlsl::kTexcoord && s.index == 1 && d.instance_at >= 0;
    st.in[i][0] = instance ? d.instance_at : d.vertex0 + field;
    st.in[i][1] = instance ? 0 : int32_t(kStride);
    st.in[i][2] = 3;  // FLOAT4
  }
  const int32_t ints[80] = {};
  [e setRenderPipelineState:d.draw];
  [e setDepthStencilState:g.ds];
  [e setVertexBytes:d.c.data() length:d.c.size() * 4 atIndex:0];
  [e setVertexBytes:ints length:sizeof ints atIndex:1];
  [e setVertexBytes:&st length:sizeof st atIndex:2];
  [e setVertexBuffer:d.vb offset:0 atIndex:3];
  [e setVertexTexture:g.vtex atIndex:0];
  [e setVertexSamplerState:g.smp atIndex:0];
  const float pc[2][4] = {{1, 1, 1, 1}, {}};
  PsState pst{};
  pst.alpha[0] = 8;
  pst.vscale[0] = pst.vscale[1] = 1;
  [e setFragmentBytes:pc length:sizeof pc atIndex:0];
  [e setFragmentBytes:&pst length:sizeof pst atIndex:2];
  id<MTLBuffer> ib = [g.dev newBufferWithBytes:d.ib.data() length:d.ib.size() * 2 options:MTLResourceStorageModeShared];
  [e drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:d.ib.size() indexType:MTLIndexTypeUInt16 indexBuffer:ib
         indexBufferOffset:0 instanceCount:1 baseVertex:d.base_vertex baseInstance:0];
  if (!d.slot.buf) return;
  // The capture (metal_device.mm's RtCaptureDraw).
  const int32_t range[4] = {int32_t(d.start), int32_t(d.count), 0, 0};
  [e setRenderPipelineState:d.capture];
  [e setVertexBuffer:d.slot.buf offset:d.slot.capture_off atIndex:gfx::kMetalCaptureBuffer];
  [e setVertexBytes:range length:sizeof range atIndex:gfx::kMetalCaptureRangeBuffer];
  [e drawPrimitives:MTLPrimitiveTypePoint vertexStart:d.start vertexCount:d.count];
}

void EncodePass(Gpu& g, id<MTLCommandBuffer> cb, const std::vector<Draw>& draws) {
  MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
  rp.colorAttachments[0].texture = g.color;
  rp.colorAttachments[0].loadAction = MTLLoadActionClear;
  rp.colorAttachments[0].storeAction = MTLStoreActionStore;
  rp.depthAttachment.texture = g.depth;
  rp.depthAttachment.loadAction = MTLLoadActionClear;
  rp.depthAttachment.storeAction = MTLStoreActionStore;
  rp.stencilAttachment.texture = g.depth;
  rp.stencilAttachment.loadAction = MTLLoadActionClear;
  rp.stencilAttachment.storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
  [e setFrontFacingWinding:MTLWindingClockwise];
  [e setCullMode:MTLCullModeNone];
  for (const Draw& d : draws) Encode(g, e, d);
  [e endEncoding];
}

// ---- the skinned mesh

// Two bones 15 m ahead of the camera (the second turned and 2.5 m to the
// right); quads of 4 vertices on each, in bone space z = 0.
struct Skinned {
  M4 bone[2];
  std::array<double, 3> dir = {0.0, 0.0, -1.0};  // c5: the displacement's direction
  std::vector<Vertex> verts;                     // kJunk vertices, then the mesh's (index 1 on)
  std::vector<uint16_t> ib;                      // raw indices 1..8
  static constexpr int kJunk = 3;                // vertices before vertex 0 of the draw (its base vertex)
  static constexpr uint32_t kMin = 1;  // MinVertexIndex (8 vertices from it)

  Skinned() {
    const std::array<double, 3> a = Ahead(15, -1.0), b = Ahead(15, 2.5, 0.5);
    bone[0] = AsFloat(Mul(RotY(kYaw), Translation(a[0], a[1], a[2])));
    bone[1] = AsFloat(Mul(RotY(kYaw + 0.3), Translation(b[0], b[1], b[2])));
    verts.resize(kJunk + 1 + 8);
    for (Vertex& v : verts) v = {{1e9f, 1e9f, 1e9f, 1}, {0, 0, 0, 0}, {0.125f, 0.5f, 0, 0}, {}};
    const float corner[4][2] = {{-0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, -0.5f}, {0.5f, 0.5f}};
    for (int i = 0; i < 8; i++) {
      Vertex& v = verts[kJunk + 1 + i];
      v.pos[0] = corner[i % 4][0];
      v.pos[1] = corner[i % 4][1];
      v.pos[2] = 0;
      v.pos[3] = 1;
      v.blend[0] = float(4 * (i / 4));          // a0.x: the bone's first row
      v.uv[0] = (float(i % 4) + 0.5f) / 4.0f;  // texel i % 4: displacement 0.25 * (i % 4)
      v.uv[1] = 0.5f;
    }
    // Two triangles a quad; the second quad's wound the other way.
    ib = {1, 2, 3, 3, 2, 4, 5, 7, 6, 7, 8, 6};
  }

  // Vertex i (0-7) in world space, from what the GPU gets (floats).
  std::array<double, 3> World(int i) const {
    const Vertex& v = verts[kJunk + 1 + i];
    const std::array<double, 4> p = Apply(bone[i / 4], {v.pos[0], v.pos[1], v.pos[2]});
    const double d = 0.25 * (i % 4);
    return {p[0] + d * double(float(dir[0])), p[1] + d * double(float(dir[1])), p[2] + d * double(float(dir[2]))};
  }

  std::vector<float> Constants(const M4& vp) const {
    std::vector<float> c(256 * 4, 0.0f);
    for (int k = 0; k < 3; k++) c[5 * 4 + k] = float(dir[k]);
    float rows[4][4];
    Rows(vp, rows);
    std::memcpy(&c[8 * 4], rows, sizeof rows);
    for (int b = 0; b < 2; b++) {
      Rows(bone[b], rows);
      std::memcpy(&c[(64 + 4 * b) * 4], rows, sizeof rows);
    }
    return c;
  }

  // Its triangles as the scene takes them.
  gfx::RtGeometry Geometry() const {
    gfx::RtGeometry g;
    g.prim = 4;
    g.primitives = uint32_t(ib.size() / 3);
    g.base_vertex = kJunk;
    g.indices = ib.data();
    g.index_size = 2;
    return g;
  }
};

// D3D's front side of triangle (a, b, c) faces the camera of `vp` if it is
// clockwise as the screen shows it: x right, y up in clip space.
bool FrontToCamera(const M4& vp, const std::array<double, 3>& a, const std::array<double, 3>& b,
                   const std::array<double, 3>& c) {
  const std::array<double, 4> p = Apply(vp, a), q = Apply(vp, b), r = Apply(vp, c);
  const double ax = p[0] / p[3], ay = p[1] / p[3], bx = q[0] / q[3], by = q[1] / q[3], cx = r[0] / r[3], cy = r[1] / r[3];
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) < 0;
}

// Skinned triangle t's corners in world space.
std::array<std::array<double, 3>, 3> Corners(const Skinned& m, size_t t) {
  std::array<std::array<double, 3>, 3> v;
  for (int k = 0; k < 3; k++) v[k] = m.World(int(m.ib[t * 3 + k]) - 1);
  return v;
}

// A ray from the camera through point p, and the distance to it.
Query RayTo(const std::array<double, 3>& p, double* dist) {
  const double d[3] = {p[0] - kEye[0], p[1] - kEye[1], p[2] - kEye[2]};
  *dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  return {{float(kEye[0]), float(kEye[1]), float(kEye[2]), 1e4f},
          {float(d[0] / *dist), float(d[1] / *dist), float(d[2] / *dist), 0}};
}

// A static square for the camera vote: unit, 40 m ahead, upright.
struct StaticSquare {
  std::vector<float> pos = {0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0};
  std::vector<uint16_t> ib = {0, 1, 2, 3};
  gfx::RtGeometry g;
  gfx::RtMeshKey key;
  float world[4][4], wvp[4][4];
  explicit StaticSquare(const M4& vp) {
    g.positions = reinterpret_cast<const uint8_t*>(pos.data());
    g.size = pos.size() * 4;
    g.stride = 12;
    g.type = 2;
    g.prim = 5;
    g.primitives = 2;
    g.indices = ib.data();
    key.vertex_version = 1;
    key.index_version = 2;
    key.vertex_buffer = 1;
    key.index_buffer = 2;
    key.prim = 5;
    key.primitives = 2;
    key.stride = 12;
    key.type = 2;
    key.index_size = 2;
    const std::array<double, 3> at = Ahead(40, 30);
    const M4 w = AsFloat(Mul(RotY(kYaw), Translation(at[0], at[1], at[2])));
    Rows(w, world);
    Rows(AsFloat(Mul(w, vp)), wvp);
  }
  bool Add(gfx::RtScene& s) { return s.Add(key, g, world, wvp, 3, 4); }
};

// The skinned draw (read in place, or copied as metal_device.mm copies
// vertices) and its capture; D3DCULL `cull`.
Draw SkinnedDraw(Gpu& g, gfx::RtScene& s, const Skinned& m, const M4& vp, bool in_place, uint32_t cull = 3,
                 const gfx::RtCut* cut = nullptr, const gfx::RtMeshKey* key = nullptr, uint64_t actor = 0) {
  Draw d;
  d.sh = &g.skin;
  d.draw = g.skin_draw;
  d.capture = g.skin_capture;
  d.ib = m.ib;
  d.c = m.Constants(vp);
  uint32_t first = 0, count = 0;
  gfx::RtGeometry geo = m.Geometry();
  if (cut) {
    // The UVs from the vertex stream (TEXCOORD0, FLOAT4), as the backend
    // gives them: vertex v's at v * stride.
    const auto* vb = reinterpret_cast<const uint8_t*>(m.verts.data());
    geo.uv = {vb + offsetof(Vertex, uv), m.verts.size() * kStride - offsetof(Vertex, uv), kStride, 3, {0, 1}};
  }
  gfx::RtIndexRange(geo, &first, &count);
  d.count = count;
  if (in_place) {
    // Vertex v at v * stride; Metal's base vertex: gl_VertexIndex includes it.
    d.vb = [g.dev newBufferWithBytes:m.verts.data() length:m.verts.size() * kStride options:MTLResourceStorageModeShared];
    d.vertex0 = 0;
    d.base_vertex = Skinned::kJunk;
  } else {
    // The draw's vertices from base vertex + MinVertexIndex copied, the
    // offsets biased by -MinVertexIndex: gl_VertexIndex the raw index.
    const size_t at = size_t(Skinned::kJunk + Skinned::kMin);
    d.vb = [g.dev newBufferWithBytes:m.verts.data() + at length:(m.verts.size() - at) * kStride
                             options:MTLResourceStorageModeShared];
    d.vertex0 = -int32_t(Skinned::kMin * kStride);
    d.base_vertex = 0;
  }
  d.start = gfx::RtCaptureStart(first, Skinned::kJunk, in_place);
  d.slot = s.AddCaptured(geo, first, count, cull, cut, nullptr, key, actor);
  return d;
}

// Copies the world positions a capture's conversion wrote (once cb ran).
id<MTLBuffer> EncodeReadPositions(Gpu& g, id<MTLCommandBuffer> cb, const Draw& d) {
  id<MTLBuffer> out = [g.dev newBufferWithLength:d.count * 12 options:MTLResourceStorageModeShared];
  if (!d.slot.buf) return out;
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  [bl copyFromBuffer:d.slot.buf sourceOffset:d.slot.vertices_off toBuffer:out destinationOffset:0 size:d.count * 12];
  [bl endEncoding];
  return out;
}

double MaxError(const Skinned& m, id<MTLBuffer> got) {
  const float* p = static_cast<const float*>(got.contents);
  double worst = 0;
  for (int i = 0; i < 8; i++) {
    const std::array<double, 3> w = m.World(i);
    for (int k = 0; k < 3; k++) worst = std::max(worst, std::fabs(double(p[i * 3 + k]) - w[k]));
  }
  return worst;
}

// Rays from the camera to each skinned triangle's centre: hit at its
// distance (1 cm), as instance `id`, front-facing on the side the game
// draws (D3DCULL `cull`: D3D's front, clockwise as `vp` shows it, or the
// other side for D3DCULL_CW); and the front sides of the two quads differ.
// Rays back from 1 m behind each, culling back faces: they hit where the
// side they see is the drawn one, or where nothing is culled
// (D3DCULL_NONE).
bool SkinnedHits(Gpu& g, const gfx::RtFrame* f, const Skinned& m, const M4& vp, uint32_t id, uint32_t cull = 3) {
  std::vector<Query> q;
  std::vector<double> dist;
  std::vector<bool> front;
  const size_t tris = m.ib.size() / 3;
  for (size_t t = 0; t < tris; t++) {
    const auto v = Corners(m, t);
    const std::array<double, 3> c = {(v[0][0] + v[1][0] + v[2][0]) / 3, (v[0][1] + v[1][1] + v[2][1]) / 3,
                                     (v[0][2] + v[1][2] + v[2][2]) / 3};
    double dd;
    q.push_back(RayTo(c, &dd));
    dist.push_back(dd);
    front.push_back(FrontToCamera(vp, v[0], v[1], v[2]) != (cull == 2));
  }
  for (size_t t = 0; t < tris; t++) {
    Query b = q[t];
    for (int k = 0; k < 3; k++) {
      b.o[k] += b.d[k] * float(dist[t] + 1.0);
      b.d[k] = -b.d[k];
    }
    b.d[3] = 1;  // back faces culled
    q.push_back(b);
  }
  const std::vector<Hit> h = g.Trace(f, q);
  bool ok = front[0] != front[2];
  for (size_t i = 0; i < tris; i++) {
    const bool hit_ok = std::fabs(double(h[i].t) - dist[i]) < 0.01 && h[i].id == id && (h[i].front != 0) == front[i];
    const bool behind = cull == 1 || !front[i];
    const bool back_ok = behind ? std::fabs(h[tris + i].t - 1.0f) < 0.01f && h[tris + i].front == (front[i] ? 0u : 1u)
                                : h[tris + i].t == -1;
    if (!hit_ok || !back_ok)
      std::printf("  (triangle %zu: t %.4f want %.4f, id %u want %u, front %u want %d; from behind t %.4f, want %s)\n",
                  i, h[i].t, dist[i], h[i].id, id, h[i].front, int(front[i]), h[tris + i].t, behind ? "a hit" : "none");
    ok = ok && hit_ok && back_ok;
  }
  return ok;
}

// One presented frame (serial `serial`, the GPU done with `completed`): the
// static square for the camera, the draws `make` gives, all encoded and
// run; the frame's TLAS.
template <typename F>
const gfx::RtFrame* CapturedFrame(Gpu& g, gfx::RtScene& s, uint64_t serial, uint64_t completed, const M4& vp, F make) {
  g.arena.used = 0;
  s.BeginFrame(serial, completed);
  StaticSquare sq(vp);
  sq.Add(s);
  const std::vector<Draw> draws = make();
  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  EncodePass(g, cb, draws);
  const gfx::RtFrame* f = s.EndFrame(true);
  s.EncodeFrame(cb, f);
  return g.Commit(cb) ? f : nullptr;
}

int MoreCaptureChecks(Gpu& g, const Skinned& sk);

int CaptureChecks(Gpu& g) {
  int failures = 0;
  const Skinned m;
  // No camera yet (no static draw ever): the capture is left out.
  {
    auto s = g.Scene();
    if (!s) return Check(false, "scene made");
    g.arena.used = 0;
    s->BeginFrame(1, 0);
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    const M4 vp = ViewProj(false);
    std::vector<Draw> draws = {SkinnedDraw(g, *s, m, vp, true)};
    EncodePass(g, cb, draws);
    const gfx::RtFrame* f = s->EndFrame(true);
    s->EncodeFrame(cb, f);
    const bool ok = g.Commit(cb);
    const gfx::RtStats st = s->Stats();
    failures += Check(ok && draws[0].slot.buf && (!f || !f->tlas) &&
                          st.capture_skipped[size_t(gfx::RtSkip::BadCamera)] == 1 && st.captured == 0,
                      "no camera known: the capture left out (counted)");
  }

  auto s = g.Scene();
  if (!s) return Check(false, "scene made");
  const Skinned& sk = m;
  // Frame 1: a static draw (the camera), the skinned draw read in place,
  // and a draw split per instance, three instances.
  const M4 vp = ViewProj(false);
  {
    g.arena.used = 0;
    s->BeginFrame(1, 0);
    StaticSquare sq(vp);
    const bool added = sq.Add(*s);
    std::vector<Draw> draws = {SkinnedDraw(g, *s, sk, vp, true)};
    // One triangle, upright, 20 m ahead; instances 3 m apart to the right,
    // each from the instance stream's element (stride 0).
    std::vector<Vertex> iv(3 + 3);
    const float tri[3][3] = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}};
    const std::array<double, 3> base = Ahead(20, -4, 3);
    for (int i = 0; i < 3; i++) {
      for (int k = 0; k < 3; k++) iv[i].pos[k] = float(base[k]) + tri[i][k];
      iv[i].pos[3] = 1;
      iv[3 + i].inst[0] = 3.0f * float(i);  // element i: (3 i, 0, 0, 0)
    }
    id<MTLBuffer> ivb = [g.dev newBufferWithBytes:iv.data() length:iv.size() * kStride options:MTLResourceStorageModeShared];
    gfx::RtGeometry ig;
    ig.prim = 4;
    ig.primitives = 1;
    const std::vector<uint16_t> iib = {0, 1, 2};
    ig.indices = iib.data();
    for (int i = 0; i < 3; i++) {
      Draw d;
      d.sh = &g.inst;
      d.draw = g.inst_draw;
      d.capture = g.inst_capture;
      d.vb = ivb;
      d.ib = iib;
      d.instance_at = int32_t((3 + i) * kStride + offsetof(Vertex, inst));
      d.c = sk.Constants(vp);
      d.start = 0;
      d.count = 3;
      d.slot = s->AddCaptured(ig, 0, 3, 1);
      draws.push_back(d);
    }
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    EncodePass(g, cb, draws);
    const gfx::RtFrame* f = s->EndFrame(true);
    s->EncodeFrame(cb, f);
    id<MTLBuffer> pos = EncodeReadPositions(g, cb, draws[0]);
    const bool ran = g.Commit(cb);
    const double err = MaxError(sk, pos);
    std::printf("  (in place: world positions within %.2f mm)\n", err * 1000);
    failures += Check(added && ran && f && f->instances == 5 && err < 0.01,
                      "skinned (bones at c64+ through a0) and displaced (texldl), read in place with a base vertex, 4 "
                      "km out: world positions within 1 cm");
    failures += Check(SkinnedHits(g, f, sk, vp, 1),
                      "rays hit the captured triangles at their distances, D3D's front side as the screen's winding");
    std::vector<Query> q;
    std::vector<double> dist;
    for (int i = 0; i < 3; i++) {
      double dd;
      q.push_back(RayTo({base[0] + 3.0 * i + 0.25, base[1] + 0.25, base[2]}, &dd));
      dist.push_back(dd);
    }
    const std::vector<Hit> h = g.Trace(f, q);
    bool ok = true;
    for (int i = 0; i < 3; i++) ok = ok && std::fabs(double(h[i].t) - dist[i]) < 0.01 && h[i].id == uint32_t(2 + i);
    failures += Check(ok, "a draw split per instance (a stride-0 instance stream): an instance each, at its place");
    const gfx::RtStats st = s->Stats();
    failures += Check(st.captured == 4 && st.capture_builds == 4 && st.capture_triangles == 4 + 3,
                      "statistics: captured instances, builds, triangles");
  }
  // Frame 2: the view-projection mirrored (x), the vertices copied: the
  // same world positions, D3D's front side the other one.
  {
    const M4 mvp = ViewProj(true);
    g.arena.used = 0;
    s->BeginFrame(2, 1);
    StaticSquare sq(mvp);
    sq.Add(*s);
    std::vector<Draw> draws = {SkinnedDraw(g, *s, sk, mvp, false)};
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    EncodePass(g, cb, draws);
    const gfx::RtFrame* f = s->EndFrame(true);
    s->EncodeFrame(cb, f);
    id<MTLBuffer> pos = EncodeReadPositions(g, cb, draws[0]);
    const bool ran = g.Commit(cb);
    const double err = MaxError(sk, pos);
    std::printf("  (copied: world positions within %.2f mm)\n", err * 1000);
    failures += Check(ran && f && f->instances == 2 && err < 0.01,
                      "copied vertices (no base vertex in the numbering): world positions within 1 cm");
    const auto v = Corners(sk, 0);
    failures += Check(SkinnedHits(g, f, sk, mvp, 1) &&
                          FrontToCamera(mvp, v[0], v[1], v[2]) != FrontToCamera(vp, v[0], v[1], v[2]),
                      "a mirrored view-projection: D3D's front side flips");
  }
  // Frames 3 and 4: a capture after frame 3's snapshot goes into frame 4's
  // snapshot (carried, at frame 3's place).
  {
    g.arena.used = 0;
    s->BeginFrame(3, 2);
    StaticSquare sq(vp);
    sq.Add(*s);
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    const gfx::RtFrame* snap = s->Snapshot();
    s->EncodeFrame(cb, snap);
    std::vector<Draw> draws = {SkinnedDraw(g, *s, sk, vp, true)};
    EncodePass(g, cb, draws);
    s->EncodeFrame(cb, s->EndFrame(true));
    bool ok = g.Commit(cb);
    g.arena.used = 0;
    s->BeginFrame(4, 3);
    sq.Add(*s);
    cb = [g.queue commandBuffer];
    snap = s->Snapshot();
    s->EncodeFrame(cb, snap);
    ok = g.Commit(cb) && ok;
    ok = ok && snap && snap->instances == 2 && SkinnedHits(g, snap, sk, vp, 1);
    s->EncodeFrame(cb = [g.queue commandBuffer], s->EndFrame(true));
    ok = g.Commit(cb) && ok;
    failures += Check(ok, "a capture after a snapshot: in the next frame's snapshots (its BLAS kept)");
  }
  // The budget: more vertices than a frame captures.
  {
    g.arena.used = 0;
    s->BeginFrame(5, 4);
    gfx::RtGeometry big;
    big.prim = 4;
    big.primitives = gfx::RtScene::kCaptureVerticesPerFrame / 3 + 1;
    const gfx::RtCaptureSlot slot = s->AddCaptured(big, 0, big.primitives * 3, 3);
    s->EndFrame(true);
    failures += Check(!slot.buf && s->Stats().capture_skipped[size_t(gfx::RtSkip::CaptureBudget)] == 1,
                      "past the frame's capture budget: left out (counted)");
  }
  // Frames 6-17, the skinned draw in each (the GPU done with each before
  // the next): the BLAS memory pooled, not growing.
  {
    uint64_t at10 = 0, buffers10 = 0;
    bool ok = true;
    for (uint64_t serial = 6; serial <= 17; serial++) {
      g.arena.used = 0;
      s->BeginFrame(serial, serial - 1);
      StaticSquare sq(vp);
      sq.Add(*s);
      std::vector<Draw> draws = {SkinnedDraw(g, *s, sk, vp, serial % 2 == 0)};
      id<MTLCommandBuffer> cb = [g.queue commandBuffer];
      EncodePass(g, cb, draws);
      const gfx::RtFrame* f = s->EndFrame(true);
      s->EncodeFrame(cb, f);
      ok = g.Commit(cb) && ok && f && f->instances == 2;
      const gfx::RtStats st = s->Stats();
      if (serial == 10) at10 = st.capture_blas_bytes, buffers10 = st.capture_buffer_bytes;
      if (serial > 10) ok = ok && st.capture_blas_bytes == at10 && st.capture_buffer_bytes == buffers10;
    }
    failures += Check(ok && at10 > 0, "captured BLASes pooled: no new memory frame after frame");
    const std::string line = s->TakeStats();
    failures += Check(line.find("captured per frame") != std::string::npos && line.find("capture budget") != std::string::npos,
                      "the perf line names captured draws and their skips");
  }
  failures += MoreCaptureChecks(g, m);
  return failures;
}

// The capture's vertex numbering; culling; a frame a readback ends; the
// GPU's progress lagging.
int MoreCaptureChecks(Gpu& g, const Skinned& sk) {
  int failures = 0;
  failures += Check(gfx::RtCaptureStart(7, 3, true) == 7 && gfx::RtCaptureStart(7, 3, false) == 4 &&
                        gfx::RtCaptureStart(2, -5, false) == 7 && gfx::RtCaptureStart(0, 0, false) == 0,
                    "capture start: base vertex kept in place, taken off where copied");
  const M4 vp = ViewProj(false);
  // D3DCULL_CW (the game draws the counterclockwise side) and D3DCULL_NONE.
  {
    auto s = g.Scene();
    if (!s) return Check(false, "scene made");
    bool ok = true;
    uint64_t serial = 0;
    for (uint32_t cull : {2u, 1u, 3u}) {
      serial++;
      const gfx::RtFrame* f = CapturedFrame(g, *s, serial, serial - 1, vp, [&] {
        return std::vector<Draw>{SkinnedDraw(g, *s, sk, vp, true, cull)};
      });
      const bool cull_ok = f && f->instances == 2 && SkinnedHits(g, f, sk, vp, 1, cull);
      if (!cull_ok) std::printf("  (D3DCULL %u)\n", cull);
      ok = ok && cull_ok;
    }
    failures += Check(ok, "captures with D3DCULL_CW, _NONE and _CCW: the side drawn front-facing, nothing culled "
                          "with _NONE");
  }
  // A readback ends the command buffer after the capture: its conversion
  // and build in that one; the instance in the presented frame's TLAS.
  {
    auto s = g.Scene();
    if (!s) return Check(false, "scene made");
    g.arena.used = 0;
    s->BeginFrame(1, 0);
    StaticSquare sq(vp);
    sq.Add(*s);
    std::vector<Draw> draws = {SkinnedDraw(g, *s, sk, vp, false)};
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    EncodePass(g, cb, draws);
    const gfx::RtFrame* part = s->EndFrame(false);
    s->EncodeFrame(cb, part);
    id<MTLBuffer> pos = EncodeReadPositions(g, cb, draws[0]);
    bool ok = g.Commit(cb) && part && !part->tlas;
    const double err = MaxError(sk, pos);
    s->BeginFrame(2, 1);
    sq.Add(*s);
    const gfx::RtFrame* f = s->EndFrame(true);
    s->EncodeFrame(cb = [g.queue commandBuffer], f);
    ok = g.Commit(cb) && ok && f && f->instances == 3 && err < 0.01 && SkinnedHits(g, f, sk, vp, 1);
    failures += Check(ok, "a frame a readback ends: its captures converted and built in it, traced in the presented "
                          "frame's TLAS");
  }
  // The GPU's progress (BeginFrame's `completed`) stuck at 0: no captured
  // BLAS memory reused, a new one every frame.
  {
    auto s = g.Scene();
    if (!s) return Check(false, "scene made");
    bool ok = true;
    uint64_t last = 0;
    for (uint64_t serial = 1; serial <= 6; serial++) {
      const gfx::RtFrame* f = CapturedFrame(g, *s, serial, 0, vp, [&] {
        return std::vector<Draw>{SkinnedDraw(g, *s, sk, vp, true)};
      });
      const uint64_t bytes = s->Stats().capture_blas_bytes;
      ok = ok && f && f->instances == 2 && bytes > last;
      last = bytes;
    }
    // Then the GPU caught up: the pooled ones reused, no new memory.
    for (uint64_t serial = 7; serial <= 9; serial++) {
      const gfx::RtFrame* f = CapturedFrame(g, *s, serial, serial - 1, vp, [&] {
        return std::vector<Draw>{SkinnedDraw(g, *s, sk, vp, true)};
      });
      ok = ok && f && s->Stats().capture_blas_bytes == last;
    }
    failures += Check(ok, "the GPU's progress lagging: no captured BLAS memory reused until it catches up");
  }
  // A capture with a key (its triangles named, as the backend gives every
  // capture from buffers that stay), moving every frame: the BLAS an
  // earlier frame built for it refit to the new positions, built anew
  // after kCaptureRefits refits; each frame's rays hit it where it is now.
  // Its BLASes resident through the scene's residency set on the queue
  // (the rays' encoders declare none).
  {
    auto s = g.Scene();
    if (!s) return Check(false, "scene made");
    const bool resident = s->UseResidency(g.queue);
    bool flagged = resident;
    gfx::RtMeshKey key;
    key.index_buffer = 7;
    key.index_version = 1;
    key.index_size = 2;
    key.base_vertex = Skinned::kJunk;
    key.prim = 4;
    key.primitives = uint32_t(sk.ib.size() / 3);
    bool ok = true;
    for (uint64_t serial = 1; serial <= 60; serial++) {
      Skinned moved = sk;
      moved.bone[0] = AsFloat(Mul(sk.bone[0], Translation(0.02 * double(serial % 7), 0.03 * double(serial % 5), 0)));
      moved.bone[1] = AsFloat(Mul(sk.bone[1], Translation(0, -0.02 * double(serial % 3), 0.01 * double(serial % 4))));
      const gfx::RtFrame* f = CapturedFrame(g, *s, serial, serial - 1, vp, [&] {
        return std::vector<Draw>{SkinnedDraw(g, *s, moved, vp, serial % 2 == 0, 3, nullptr, &key)};
      });
      const bool hit = f && f->instances == 2 && SkinnedHits(g, f, moved, vp, 1);
      flagged = flagged && f && f->blas_resident;
      if (!hit) std::printf("  (frame %llu)\n", (unsigned long long)serial);
      ok = ok && hit;
    }
    const gfx::RtStats st = s->Stats();
    const uint64_t rebuilt = st.capture_builds - st.capture_refits;
    std::printf("  (60 frames: %llu refits, %llu builds)\n", (unsigned long long)st.capture_refits,
                (unsigned long long)rebuilt);
    // (The first frames' BLASes, before any is let go, are built; each then
    // refit kCaptureRefits times between builds.)
    failures += Check(ok && st.capture_builds == 60 && st.capture_refits >= 40 &&
                          rebuilt * gfx::RtScene::kCaptureRefits >= st.capture_refits,
                      "a keyed capture moving every frame: refit (rebuilt every kCaptureRefits), hit where it is");
    failures += Check(s->TakeStats().find("of them refits") != std::string::npos, "the perf line counts the refits");
    if (@available(macOS 15.0, *))
      failures += Check(resident && flagged, "residency set: on the queue, every frame's BLASes resident through it");
  }
  // A part the game hides with zero bone matrices: its vertices at the
  // world's origin come out NaN (their triangles left out of the BLAS,
  // which would else reach from the mesh to the origin); the rest where
  // they are, hit.
  {
    auto s = g.Scene();
    if (!s) return Check(false, "scene made");
    Skinned z = sk;
    z.dir = {0.0, 0.0, 0.0};
    z.bone[1] = M4{};
    g.arena.used = 0;
    s->BeginFrame(1, 0);
    StaticSquare sq(vp);
    sq.Add(*s);
    std::vector<Draw> draws = {SkinnedDraw(g, *s, z, vp, true)};
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    EncodePass(g, cb, draws);
    const gfx::RtFrame* f = s->EndFrame(true);
    s->EncodeFrame(cb, f);
    id<MTLBuffer> pos = EncodeReadPositions(g, cb, draws[0]);
    bool ok = g.Commit(cb) && f && f->instances == 2;
    const float* p = static_cast<const float*>(pos.contents);
    for (int i = 0; i < 8 && ok; i++) {
      const std::array<double, 3> w = z.World(i);
      for (int k = 0; k < 3; k++)
        ok = ok && (i < 4 ? std::fabs(double(p[i * 3 + k]) - w[k]) < 0.01 : std::isnan(p[i * 3 + k]));
    }
    std::vector<Query> q;
    std::vector<double> dist;
    for (size_t t = 0; t < 2; t++) {
      const auto v = Corners(z, t);
      double dd;
      q.push_back(RayTo({(v[0][0] + v[1][0] + v[2][0]) / 3, (v[0][1] + v[1][1] + v[2][1]) / 3,
                         (v[0][2] + v[1][2] + v[2][2]) / 3},
                        &dd));
      dist.push_back(dd);
    }
    const std::vector<Hit> h = g.Trace(f, q);
    for (size_t t = 0; t < 2; t++) ok = ok && std::fabs(double(h[t].t) - dist[t]) < 0.01 && h[t].id == 1;
    failures += Check(ok, "a hidden part's vertices (zero bones, at the origin) left out as NaN; the rest hit");
  }
  // kRtMsl's rt_shadowed in one traversal with the instances' info: the
  // same answer as its two phases (static instances from the start,
  // captured ones beyond the gap) for rays from the eye and from the
  // skinned quads' surface, toward the quads, the static square and the
  // sky, gaps from none to past everything; blocked and not, and the
  // fallback (a captured hit within the gap) taken.
  {
    auto s = g.Scene();
    if (!s) return Check(false, "scene made");
    const gfx::RtFrame* f = CapturedFrame(g, *s, 1, 0, vp, [&] {
      return std::vector<Draw>{SkinnedDraw(g, *s, sk, vp, true)};
    });
    const std::array<double, 3> at = Ahead(40, 30);
    const std::array<double, 4> sq = Apply(AsFloat(Mul(RotY(kYaw), Translation(at[0], at[1], at[2]))), {0.5, 0.5, 0});
    std::vector<std::array<double, 3>> from = {kEye}, to = {{sq[0], sq[1], sq[2]}, Ahead(100, 0, 60)};
    for (size_t t = 0; t < sk.ib.size() / 3; t++) {
      const auto v = Corners(sk, t);
      const std::array<double, 3> c = {(v[0][0] + v[1][0] + v[2][0]) / 3, (v[0][1] + v[1][1] + v[2][1]) / 3,
                                       (v[0][2] + v[1][2] + v[2][2]) / 3};
      from.push_back(c);
      to.push_back(c);
    }
    std::vector<Query> q;
    for (const auto& o : from)
      for (const auto& t : to) {
        double d[3] = {t[0] - o[0], t[1] - o[1], t[2] - o[2]};
        const double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if (len < 1e-3) continue;
        for (double gap : {0.0, 0.3, 5.0, 20.0, 1e4})
          q.push_back({{float(o[0]), float(o[1]), float(o[2]), 1e4f},
                       {float(d[0] / len), float(d[1] / len), float(d[2] / len), float(gap)}});
      }
    const std::vector<Hit> h = g.Shadow(f, q);
    int same = 0, blocked = 0, fallback = 0;
    for (size_t i = 0; i < h.size(); i++) {
      same += (h[i].t > 0.5f) == (h[i].id != 0) ? 1 : 0;
      blocked += h[i].id ? 1 : 0;
      fallback += h[i].front ? 1 : 0;
    }
    std::printf("  (rt_shadowed: %zu rays, %d alike, %d blocked, %d through the fallback)\n", h.size(), same, blocked,
                fallback);
    failures += Check(f && same == int(h.size()) && blocked > 0 && blocked < int(h.size()) && fallback > 0,
                      "rt_shadowed in one traversal: the two phases' answer, blocked and not, the gap's fallback");
  }
  // An actor: two skinned parts drawn in a row with the same actor (and a
  // third, another actor's): one BLAS of two geometries, one instance,
  // its parts' slots consecutive from its user ID (a hit's instance: user
  // ID + geometry ID); refit as one frame after frame, each part hit where
  // it is now.
  {
    auto s = g.Scene();
    if (!s) return Check(false, "scene made");
    gfx::RtMeshKey ka, kb;
    ka.index_buffer = 11;
    ka.index_version = kb.index_version = 1;
    kb.index_buffer = 12;
    bool ok = true;
    for (uint64_t serial = 1; serial <= 24; serial++) {
      Skinned a = sk, b = sk, c = sk;
      a.bone[0] = AsFloat(Mul(sk.bone[0], Translation(0.01 * double(serial % 5), 0, 0)));
      b.bone[0] = AsFloat(Mul(sk.bone[0], Translation(0, 2.5, 0)));
      b.bone[1] = AsFloat(Mul(sk.bone[1], Translation(0, 2.5 + 0.01 * double(serial % 3), 0)));
      c.bone[0] = AsFloat(Mul(sk.bone[0], Translation(0, -2.5, 0)));
      c.bone[1] = AsFloat(Mul(sk.bone[1], Translation(0, -2.5, 0)));
      const gfx::RtFrame* f = CapturedFrame(g, *s, serial, serial - 1, vp, [&] {
        return std::vector<Draw>{SkinnedDraw(g, *s, a, vp, true, 3, nullptr, &ka, 77),
                                 SkinnedDraw(g, *s, b, vp, true, 3, nullptr, &kb, 77),
                                 SkinnedDraw(g, *s, c, vp, true, 3, nullptr, &ka, 78)};
      });
      // The square (user ID 0), actor 77 (slots 1-2), actor 78 (slot 3).
      const bool hit = f && f->instances == 3 && SkinnedHits(g, f, a, vp, 1) && SkinnedHits(g, f, b, vp, 2) &&
                       SkinnedHits(g, f, c, vp, 3);
      if (!hit) std::printf("  (actor frame %llu: %u instances)\n", (unsigned long long)serial, f ? f->instances : 0);
      ok = ok && hit;
    }
    const gfx::RtStats st = s->Stats();
    std::printf("  (24 frames: %llu BLASes of %llu draws, %llu refits)\n", (unsigned long long)st.capture_builds,
                (unsigned long long)st.capture_draws, (unsigned long long)st.capture_refits);
    failures += Check(ok && st.capture_builds == 48 && st.capture_draws == 72 && st.capture_refits >= 24,
                      "an actor's parts: one BLAS and instance, a slot a part, refit as one; another actor apart");
  }
  // Alpha-cut: a 4 x 1 texture whose texels 0 and 1 have alpha 0 (nearest,
  // clamped), at the skinned vertices' TEXCOORD0 (u = (vertex % 4 + 0.5) /
  // 4): each quad's first triangle (UVs about u 0.375 at its centre) cut,
  // its second (0.625) kept.
  {
    auto s = g.Scene();
    if (!s) return Check(false, "scene made");
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                  width:4
                                                                                 height:1
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> tex = [g.dev newTextureWithDescriptor:td];
    const uint32_t texels[4] = {0x00FFFFFFu, 0x00FFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    [tex replaceRegion:MTLRegionMake2D(0, 0, 4, 1) mipmapLevel:0 withBytes:texels bytesPerRow:16];
    gfx::RtCut cut;
    cut.texture = tex;
    cut.material.alpha_scale = 1.0f;
    cut.material.flags = gfx::RtMaterialFlags(0, 7, 128, 3, 3, 3);  // GREATEREQUAL 128, clamped
    bool ok = true;
    for (bool in_place : {true, false}) {
      const gfx::RtFrame* f = CapturedFrame(g, *s, in_place ? 1 : 2, in_place ? 0 : 1, vp, [&] {
        return std::vector<Draw>{SkinnedDraw(g, *s, sk, vp, in_place, 3, &cut)};
      });
      std::vector<Query> q;
      std::vector<double> dist;
      for (size_t t = 0; t < sk.ib.size() / 3; t++) {
        const auto v = Corners(sk, t);
        double dd;
        q.push_back(RayTo({(v[0][0] + v[1][0] + v[2][0]) / 3, (v[0][1] + v[1][1] + v[2][1]) / 3,
                           (v[0][2] + v[1][2] + v[2][2]) / 3},
                          &dd));
        dist.push_back(dd);
      }
      const std::vector<Hit> h = g.Trace(f, q);
      for (size_t t = 0; t < q.size(); t++) {
        const bool at = std::fabs(double(h[t].t) - dist[t]) < 0.01;
        const bool want = t % 2 == 1;
        if (at != want || (want && h[t].id != 1))
          std::printf("  (%s, triangle %zu: t %.4f, distance %.4f)\n", in_place ? "in place" : "copied", t, h[t].t,
                      dist[t]);
        ok = ok && at == want && (!want || h[t].id == 1);
      }
      ok = ok && f && f->instances == 2 && f->texture_count == 1;
    }
    failures += Check(ok && s->Stats().cut == 2,
                      "a captured alpha-cut draw (skinned, read in place and copied): its UVs from the vertex stream, "
                      "the triangles over alpha 0 cut");
  }
  return failures;
}

}  // namespace

int main() {
  @autoreleasepool {
    Gpu g;
    if (!g.dev || !gfx::RtSupported(g.dev)) {
      std::printf("skipped: %s\n", g.dev ? "the GPU does not trace rays in hardware (Apple9)" : "no Metal device");
      return 77;
    }
    if (!g.Init()) return 1;
    return CaptureChecks(g) ? 1 : 0;
  }
}
