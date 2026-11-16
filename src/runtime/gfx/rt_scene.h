#pragma once
// Ray tracing of the scene (Objective-C++), on GPUs with hardware ray
// tracing: the geometry a frame draws (opaque or alpha-cut), as Metal
// acceleration structures. One bottom-level structure (BLAS) per mesh,
// cached across frames by what identifies the mesh's data (RtMeshKey);
// top-level structures (TLAS) of the frame's draws, one instance each,
// placed by the draw's world matrix: one at the end of each presented
// frame, and snapshots in the middle of one for effects traced before it
// ends.
//
// Two sides, as the Metal backend records and encodes: the recording side
// (one thread, the game's drawing thread) adds draws and prepares work;
// the encoding side encodes that work into the frame's command buffer, in
// order after the recording calls that made it. Frames here are the
// backend's command buffers (serials; a readback in the middle of a
// presented frame ends one), grouped into presented frames, which is what
// the scene and the statistics count. What one side hands the other
// (RtBuildBatch, RtFrame) stays valid until its frame slot is reused,
// kSlots serials later.
//
// A BLAS is built from a copy of the draw's data made on the recording
// side (the vertices its triangles use, decoded to float, and 32-bit
// triangle lists): Metal reads vertex data only at offsets that are
// multiples of the stride, which a mesh inside a game's vertex buffer
// seldom is.
//
// Captured meshes: draws whose vertices only their vertex shader places
// (skinning, displacement by a vertex texture, instances placed from
// per-instance data) are added with AddCaptured. The backend runs the draw's
// vertex shader a second time in the same render pass, as a capture variant
// (hlsl/capture_position.h) with rasterization off, writing each vertex's
// clip-space position (before the viewport adjustments) to memory the scene
// gives; the scene turns those into world positions with the frame's camera
// (the view-projection its static draws share) and builds a BLAS of them
// every frame (fast build, no refit), placed by the identity. Their budgets,
// memory and BLASes are apart from the static ones', which they never take
// from.
//
// Alpha-cut geometry: draws that discard pixels by their alpha (the alpha
// test, a pixel shader's texkill) are added with an RtCut, which says
// where the alpha comes from (a texture's channel at the vertices' UVs,
// times the vertices' alpha, scaled and offset) and how it is compared.
// Their BLASes keep each triangle's UVs and vertex alphas (RtCutPrim) as
// primitive data, their instances are non-opaque, and the TLAS's material
// table (RtMaterial per instance) names the texture: rays test each
// candidate hit on such an instance (kRtMsl's RtAlphaPass), the others
// never return candidates. The scene holds each texture a table names
// until no frame that may trace that table is in flight.
//
// Hit data (RtSceneOptions::hit_data, for effects that shade what rays
// hit: reflections): every BLAS of a mesh drawn with UVs keeps RtHitPrim per
// triangle (the UVs and alphas, the geometric normal, a texture LOD
// constant), and an instance drawn with a texture names it as its albedo
// (RtAlbedo, RtMaterial's albedo_texture). Meshes without UVs keep what they
// would without hit data; off, opaque meshes carry no primitive data.
//
// Retained instances: a static instance two presented frames in a row
// listed in their TLASes at the same world matrix (its mesh, world matrix
// and material; what moves is never retained) stays in the scene after
// that, listed in every later TLAS the frames that do not draw it make, so
// that what the camera turned away from (behind or above it) still
// occludes and reflects. One the frame draws again is that draw's
// instance. It is let go (RtRetainDrop) when the main camera's view has
// held all of its bounds for some frames in a row with the frame not
// drawing it (removed, hidden; not at once: what the game's occlusion
// culling leaves out comes back), when the camera is much closer to it than
// at any draw of it or within its bounds (a far level of detail), when the frame draws its geometry at
// another world matrix, or other geometry at its world matrix, with much
// the same bounds (moved off; another level of detail), when its BLAS goes
// (evicted, or new contents where its data is), after RtSceneOptions::
// retain_frames presented frames without a draw, and past max_retained
// (the farthest from the camera first). A BLAS a retained
// instance names is not evicted for age (kEvictAfter), only for memory and
// then after every BLAS none names, and the textures its material names are
// held as long as it is. Captured draws are never retained: they move.

#if !__has_feature(objc_arc)
#error "rt_scene.h needs ARC (-fobjc-arc)"
#endif

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "runtime/gfx/rt_support.h"

namespace recomp::gfx {

// ---- meshes (CPU only)

// IEEE half to float (subnormals, infinities and NaN included).
float HalfToFloat(uint16_t h);

// Whether positions of this D3DDECLTYPE can be gathered: FLOAT3 (2),
// FLOAT4 (3, its w left out) or FLOAT16_4 (16, likewise).
bool RtPositionType(uint32_t decl_type);

// A vertex input read on the CPU (RtGeometry::uv, alpha): vertex v's at
// data + v * stride, as D3D9 fills the input register from D3DDECLTYPE
// `type` (components the type lacks 0, w 1).
struct RtAttribute {
  const uint8_t* data = nullptr;  // vertex 0's (the stream's data + the input's offset); null: none
  size_t size = 0;                // bytes readable from `data`
  uint32_t stride = 0;            // 0: one value every vertex shares
  uint32_t type = 17;             // D3DDECLTYPE (0-16; 17: none)
  uint8_t lane[2] = {0, 1};       // the components read: u and v (uv), the alpha (alpha: lane[0])
};
// Bytes an input of D3DDECLTYPE `type` takes (0: not one of 0-16).
uint32_t RtAttributeBytes(uint32_t type);
// The input register of D3DDECLTYPE `type` read from p (as D3D9 fills it).
void RtFetch(const uint8_t* p, uint32_t type, float out[4]);
// Component `lane` of `a` at vertex v; false if outside `size` (or no type).
bool RtReadLane(const RtAttribute& a, uint32_t v, uint32_t lane, float* out);

// A draw's triangles, as gfx::ShaderDraw describes them: vertex v's
// position is at positions + v * stride, v being index + base_vertex
// (indexed) or base_vertex + i (not indexed).
struct RtGeometry {
  const uint8_t* positions = nullptr;  // vertex 0's position (the stream's data + the input's offset)
  size_t size = 0;                     // bytes readable from `positions`
  uint32_t stride = 0;
  uint32_t type = 2;                   // D3DDECLTYPE (RtPositionType)
  uint32_t prim = 4;                   // D3DPRIMITIVETYPE: 4 list, 5 strip, 6 fan
  uint32_t primitives = 0;
  int32_t base_vertex = 0;
  const void* indices = nullptr;       // the draw's first index; null: not indexed
  uint32_t index_size = 2;             // bytes
  // Alpha-cut draws (RtCut): the UVs and the alpha factor of each vertex,
  // read for the BLAS's primitive data (RtCutPrim); none, the UVs (0, 0)
  // and the alpha 1.
  RtAttribute uv, alpha;
};

// The range of vertices the triangles use: first (base_vertex included)
// and count. False if the primitive type is not a triangle one, there are
// too many primitives for 32-bit index counts, or a vertex lies outside
// `size` or before vertex 0.
bool RtVertexRange(const RtGeometry& g, uint32_t* first, uint32_t* count);
// The same range whatever the positions (only the indices read): false if
// the primitive type is not a triangle one, the counts overflow, or a
// vertex lies before vertex 0.
bool RtIndexRange(const RtGeometry& g, uint32_t* first, uint32_t* count);
// RtIndexRange kept by where a draw's indices are: the scan reads every
// index, and a mesh drawn every frame from the same indices (a captured
// one: a skinned model, a moving part) gives the same range every time.
// The indices are named by their index buffer (`buffer`, any identifier of
// it), the version of its contents (`version`: a buffer that stays; 0 is
// unknown) and the byte offset of the draw's first index in it (`offset`);
// with g's primitive type and count, base vertex and index size, that is
// the key. A direct-mapped table: a key whose slot holds another (a miss)
// scans and takes the slot. Not kept (scanned, counted neither way): draws
// not indexed (g.indices null) and version 0. One thread.
class RtIndexRangeMemo {
 public:
  explicit RtIndexRangeMemo(size_t slots = 4096);  // rounded up to a power of two
  // As RtIndexRange(g, first, count).
  bool Range(const RtGeometry& g, uint64_t buffer, uint64_t version, uint32_t offset, uint32_t* first, uint32_t* count);
  uint64_t hits() const { return hits_; }
  uint64_t misses() const { return misses_; }

 private:
  struct Slot {
    uint64_t buffer = 0, version = 0;
    uint32_t offset = 0, primitives = 0, prim = 0, index_size = 0;
    int32_t base_vertex = 0;
    uint32_t first = 0, count = 0;
    bool ok = false, set = false;
  };
  std::vector<Slot> slots_;
  uint64_t hits_ = 0, misses_ = 0;
};
// A captured draw's first vertex (`first`, base vertex included, as
// RtIndexRange gives it) as its vertex shader numbers it (gl_VertexIndex):
// with the base vertex where its vertices are read in place (the backend
// gives it to Metal), without where they are copied (the copy starts at
// it).
uint32_t RtCaptureStart(uint32_t first, int32_t base_vertex, bool in_place);
// The triangles as a list, 3 indices each relative to `first`: strips with
// every other triangle's order swapped back to the strip's winding, fans
// around their first vertex.
void RtTriangles(const RtGeometry& g, uint32_t first, uint32_t* out);

// A mesh gathered for its BLAS: only the vertices its triangles use, each
// once (at most 3 per triangle).
struct RtGathered {
  std::vector<float> positions;   // 4 floats a vertex (x, y, z, 0): Metal's offsets stay multiples of 16
  std::vector<uint32_t> indices;  // 3 a triangle, into `positions`
  // Vertex positions[i] is the draw's vertex first + used[i].
  uint32_t first = 0;
  std::vector<uint32_t> remap, used;  // working memory
};
// False as RtVertexRange.
bool RtGather(const RtGeometry& g, RtGathered* out);

// An alpha-cut triangle's primitive data in its BLAS (MSL: RtCutPrim):
// its corners' UVs and alpha factors (unorm8 in bytes 0-2, corner order).
// The debug view's normal (12 bytes) follows where it is kept.
struct RtCutPrim {
  float uv[3][2];
  uint32_t alpha;
};
static_assert(sizeof(RtCutPrim) == 28, "kRtMsl's RtCutPrim");
// Triangle t's primitive data from `g`'s uv and alpha (RtGeometry), its
// corners the draw's vertices first + tri[0..2]. False if one lies outside
// an attribute's data.
bool RtCutTriangle(const RtGeometry& g, uint32_t first, const uint32_t tri[3], RtCutPrim* out);

// A triangle's primitive data where the scene keeps hit data (RtScene
// Options::hit_data; MSL: RtHitPrim), for meshes drawn with UVs, in 28 bytes
// (Metal keeps primitive data in steps: 4 bytes cost a BLAS 52 bytes a
// triangle more, up to 32 bytes 80, up to 64 bytes 112): corner 0's UV, the
// other two's less it as halves (clamped to +-65504; an error of about
// |difference| 2^-12, so it grows with the repeats of a texture across one
// triangle, not with where the triangle's UVs lie), the corners' alpha
// factors as RtCutPrim's, the geometric normal in the BLAS's
// space octahedrally encoded (RtOctNormal), and the texture LOD constant of
// ray cones, 0.5 log2(uv area / area) (the area in the BLAS's space; 0
// without UVs).
struct RtHitPrim {
  float uv0[2];
  uint32_t duv[2];  // half2 each (x in the low 16 bits): uv1 - uv0, uv2 - uv0
  uint32_t alpha;
  uint32_t normal;  // RtOctNormal
  float lod;
};
static_assert(sizeof(RtHitPrim) == 28, "kRtMsl's RtHitPrim");
// The triangle with corners a, b, c and the UVs and alphas in `cut` as
// RtHitPrim.
void RtHitTriangle(const RtCutPrim& cut, const float a[3], const float b[3], const float c[3], RtHitPrim* out);
// Float to IEEE half, rounded to nearest even (overflow to infinity).
uint16_t FloatToHalf(float f);
// A direction (any length; 0 gives +z) octahedrally encoded as two snorm16
// (x in the low 16 bits, as MSL's pack_float_to_snorm2x16), and decoded to a
// unit vector.
uint32_t RtOctNormal(const float n[3]);
void RtOctDecode(uint32_t v, float out[3]);

// What a mesh's BLAS is built from. The versions change whenever their
// buffer's contents may have (gfx::VertexStream::version); the buffers'
// identities, the offsets in them and the counts tell one mesh of a buffer
// from another. Without the versions (Place) it names where a mesh's data
// is, whatever its contents. An alpha-cut mesh's (cut 1) also names its
// UVs and alpha factor (RtGeometry::uv, alpha): their stream (version,
// buffer, offset, stride), their places in a vertex, types and components;
// the alpha's input is in the UVs' stream (or, without UVs, is that
// stream). Zero for other meshes.
struct RtMeshKey {
  uint64_t vertex_version = 0, index_version = 0;
  uint64_t vertex_buffer = 0, index_buffer = 0;  // gfx::VertexStream::buffer (0: not indexed)
  uint64_t attr_version = 0, attr_buffer = 0;    // the UVs' (or alpha's) stream
  uint32_t vertex_offset = 0;  // the stream's vertex 0 in its buffer, bytes
  uint32_t index_offset = 0;   // the draw's first index in its buffer, bytes
  int32_t base_vertex = 0;
  uint32_t min_index = 0, vertex_count = 0;
  uint32_t prim = 0, primitives = 0;
  uint32_t offset = 0, stride = 0;  // the position in a vertex, the stream's stride
  uint16_t type = 0, index_size = 0;
  uint32_t attr_stream_offset = 0, attr_stride = 0;
  uint32_t uv_offset = 0, alpha_offset = 0;  // in a vertex
  uint8_t uv_type = 0, alpha_type = 0;       // D3DDECLTYPE (17: none)
  uint8_t uv_lanes = 0, alpha_lane = 0;      // u | v << 2
  uint32_t cut = 0;
  // Every field's equality (no padding: the static_assert below), as
  // words: compared on every lookup of a draw.
  bool operator==(const RtMeshKey& o) const {
    uint64_t a[14], b[14];
    std::memcpy(a, this, sizeof a);
    std::memcpy(b, &o, sizeof b);
    uint64_t d = 0;
    for (int i = 0; i < 14; i++) d |= a[i] ^ b[i];
    return d == 0;
  }
  RtMeshKey Place() const {
    RtMeshKey k = *this;
    k.vertex_version = k.index_version = k.attr_version = 0;
    return k;
  }
};
static_assert(sizeof(RtMeshKey) == 112, "no padding: hashed as words");
struct RtMeshKeyHash {
  size_t operator()(const RtMeshKey& k) const;
};

// ---- which draws go into the scene (CPU only)

// The view a draw belongs to. Main: the main camera's (its target the back
// buffer's size, depth tested and written, a colour channel written): its
// draws place the scene and vote for the frame's camera. Other: another
// pass's that tests and writes depth and writes no colour, or writes it only
// to single-channel float targets (a shadow map's casters, the map's depth
// kept in colour as well, as GTA IV's cascades keep their light-space depth
// in R32F; with `other_color` any such pass: reflections, environment maps
// and models drawn into textures are then taken too, which a game may leave
// out by its filter): static meshes only, placed by their world matrix
// alone, so that what the main camera does not see (behind or beside it)
// still occludes rays. None: not in the scene.
enum class RtView : uint8_t { Main, Other, None };

// What decides a draw's view: the pass it is recorded in and its state.
struct RtDrawPass {
  bool depth = false;      // a depth buffer is bound
  bool test = false;       // depth test on (ZENABLE)
  bool write = false;      // depth write on (ZWRITEENABLE)
  bool color = false;      // a colour channel is written (COLORWRITEENABLE)
  bool main_size = false;  // the target is the back buffer's size
  bool depth_color = false;  // every colour target single-channel float (R16F, R32F): a depth map's
};
// The view of a draw in pass `p`; `captured`: its positions only its
// vertex shader makes (RtScene::AddCaptured: converted with the main
// camera, so main view only); `other_passes`: Other is taken at all,
// `other_color`: also from passes that write colour (gfx::Device::
// RayTracing::other_passes, other_color_passes).
RtView RtViewOf(const RtDrawPass& p, bool captured, bool other_passes, bool other_color = false);
// RtDrawPass::main_size of a target the game made w x h (its own size, not
// the texture's: a target an upscaler makes at the internal resolution keeps
// the game's size) with a back buffer of bb_w x bb_h. The scene's targets at
// a render scale below 1 are then the main view's, and the scene's smaller
// buffers never are (at scale 0.5 a half-size one's texture measures as the
// back buffer's internal size).
inline bool RtMainSize(int w, int h, int bb_w, int bb_h) { return bb_w > 0 && bb_h > 0 && w == bb_w && h == bb_h; }

// ---- the scene

enum class RtDebug : uint8_t {
  Off,
  Normals,    // each pixel's first hit, its geometric normal
  Instances,  // each pixel's first hit, a colour per instance
};

struct RtSceneOptions {
  // A debug view drawn over the presented frame (EncodeDebug). Normals keep
  // each triangle's normal in its BLAS (12 bytes a triangle, after an
  // alpha-cut triangle's RtCutPrim; RtHitPrim's with hit_data).
  RtDebug debug = RtDebug::Off;
  // Every BLAS of a mesh drawn with UVs keeps RtHitPrim per triangle (28
  // bytes) and instances their albedo (RtAlbedo): for effects that shade
  // what rays hit. A draw of another view (RtView::Other) of a mesh the main
  // view drew with UVs takes that BLAS (one BLAS for both). Off: opaque
  // meshes have no primitive data, alpha-cut ones RtCutPrim (28 bytes).
  bool hit_data = false;
  // BLAS memory at most this: at it no BLAS is built, and the least
  // recently used go at the end of a presented frame (those retained
  // instances name last). 0: by the device (RtDefaultBlasBytes).
  uint64_t max_blas_bytes = 0;
  // Triangles a presented frame's builds take at most; its first build
  // always goes (up to max_mesh_triangles). The gathering runs on the
  // game's drawing thread (some milliseconds per 256k triangles): this
  // bounds what a frame spends on it, as kBuildsPerFrame bounds the count.
  uint32_t build_triangles_per_frame = 256u << 10;
  // The time a presented frame's builds take on the drawing thread at most
  // (gathering a mesh, its pieces and its primitive data: some 0.3 ms a
  // mesh of a city), milliseconds; past it the frame's other new meshes
  // wait for the next frames, the first build of a frame always goes. 0:
  // not bounded (the two budgets above only). A city streaming in gave
  // frames of 30 to 60 builds, 10 to 20 ms of them on the drawing thread:
  // one frame in twenty took 37 to 51 ms for it (2026-11-15).
  float build_ms_per_frame = 0.0f;
  // Larger meshes are left out (their gathering alone would take a frame).
  uint32_t max_mesh_triangles = 1u << 20;
  // A mesh whose data (RtMeshKey::Place) is rebuilt with new contents this
  // many times within rebuild_window presented frames is taken as
  // rewritten and left out, until its contents stay the same for
  // calm_frames.
  uint32_t rebuild_limit = 3, rebuild_window = 60, calm_frames = 120;
  // Retained instances (see the top of this file): let go after this many
  // presented frames without a draw (900: some 15 s at 60 fps, a camera
  // turning round and back); 0: none retained. At most max_retained of them
  // (each costs its TLAS a slot: some 150 bytes of upload a TLAS, and a
  // few hundred bytes of CPU memory).
  uint32_t retain_frames = 900;
  uint32_t max_retained = 8192;
  // EndFrame of a presented frame makes a TLAS of all its instances: for
  // the debug view (EncodeDebug), the frame's traced effects having used
  // its snapshots. Off: that frame's work is its builds alone (no TLAS, no
  // instance list uploaded), and the instance counts (RtStats::instances,
  // captured, cut, albedo, other, retained) are its last snapshot's.
  bool end_tlas = true;
};

// Why a draw is not in the scene (counted; RtStats::skipped).
enum class RtSkip : uint8_t {
  NotTriangles,    // points or lines
  NotScene,        // not drawn into the main view's depth-tested scene
  Blended,         // alpha blending
  // Discards by alpha (alpha test, texkill) and never passes (ALPHAFUNC
  // NEVER), or its alpha's source is not resolved (no RtCut).
  AlphaSource,
  NoMatrices,      // world or world-view-projection registers not known for its shader
  NoCapture,       // to be captured, but its vertex shader has no capture variant
  NoPosition,      // no position input, or none bound
  ConstantStream,  // positions from a stream of stride 0
  PositionType,    // positions in a format not gathered (RtPositionType)
  UserMemory,      // vertices or indices not from a buffer (no version)
  Dynamic,         // a buffer rewritten (D3DUSAGE_DYNAMIC, or locked frame after frame)
  Filtered,        // the game's filter (gfx::Device::RayTracing::filter) left it out
  NotAffine,       // a world matrix that is not affine, or singular
  OtherCamera,     // drawn with another view-projection than most of the frame
  OtherMirrored,   // another view's (RtView::Other) with a mirroring world matrix (a reflection's)
  BadCamera,       // the frame's view-projection cannot be inverted
  BadIndices,      // indices outside the stream (RtVertexRange)
  TooLarge,        // more triangles than RtSceneOptions::max_mesh_triangles
  Rebuilt,         // its data rebuilt too often (RtSceneOptions::rebuild_limit)
  BuildBudget,     // its BLAS waits for a later frame (kBuildsPerFrame, build_triangles_per_frame)
  CaptureBudget,   // captured, past the frame's capture budgets (kCapturesPerFrame, ...)
  NoMemory,        // memory for its build, or BLAS memory past max_blas_bytes
  Count,
};
const char* RtSkipName(RtSkip s);

// Why a retained instance is let go (counted; RtStats::retained_dropped).
enum class RtRetainDrop : uint8_t {
  InView,  // the main camera's view held its bounds some frames in a row, and they did not draw it
  Replaced,  // its geometry drawn at another world matrix, or other geometry at its world matrix, in its place
  Gone,    // its BLAS evicted, or new contents where its data is
  Idle,    // not drawn for RtSceneOptions::retain_frames presented frames
  Cap,     // past RtSceneOptions::max_retained (the farthest from the camera first)
  Closer,  // not drawn with the camera within its bounds, or much closer to it than at any draw of it (kRetainCloser)
  Count,
};

// Memory for a frame's GPU inputs (build inputs, instances), read by that
// frame's command buffer only. cpu null: none (the draw is left out).
struct RtUpload {
  __unsafe_unretained id<MTLBuffer> buf;
  size_t off;
  uint8_t* cpu;
};
using RtAlloc = RtUpload (*)(void* ctx, size_t bytes, size_t align);

// An instance's material in a TLAS's table (RtFrame::materials, by the
// instance's user ID plus the geometry's ID: an actor's BLAS, AddCaptured,
// takes a slot a geometry), the layout kRtMsl's RtMaterial has. Version
// kRtMaterialVersion; `reserved` is room for what later effects add (a
// normal and specular texture, ...), each such change bumping the version
// here and in kRtMsl. Instances with neither (opaque, no albedo) have a
// zero one, but for kRtMatHitPrim.
//
// Alpha-cut instances (flags kRtMatCut): with P the texel's channel at the
// hit's UV (1 without kRtMatTexture) times the hit's interpolated vertex
// alpha (1 without kRtMatVertexAlpha), a candidate hit is kept if
//   kill_scale * P + kill_offset >= 0  (with kRtMatKill: a texkill), and
//   clamp(alpha_scale * P + alpha_offset, 0, 1), rounded to 8 bits,
//   passes D3DCMPFUNC `func` against `ref` (func 0: no alpha test),
// as the translated pixel shader's r_color0 compares it. The texture is
// read at one level (max_level, D3DSAMP_MAXMIPLEVEL, clamped to its
// levels: rays have no derivatives), nearest or bilinear (kRtMatLinear,
// the sampler's MAGFILTER), its addressing (D3DTADDRESS U and V) applied by
// hand.
//
// Version 2: the albedo (kRtMatAlbedo; scenes with hit data): a 2D
// texture's rgb at the hit's UV (RtHitPrim), trilinear at a level from the
// ray's cone (kRtMsl's RtHitAlbedo), no finer than albedo_max_level
// (D3DSAMP_MAXMIPLEVEL), addressed by albedo_flags (D3DTADDRESS U in bits
// 0-2, V in 3-5), albedo_border outside a BORDER texture (D3DSAMP_
// BORDERCOLOR, a D3DCOLOR). The vertex colour does not multiply it.
// kRtMatHitPrim: the instance's primitive data is RtHitPrim (its mesh was
// drawn with UVs in a scene with hit data); alpha-cut ones then read their
// UVs and alphas from it.
inline constexpr uint32_t kRtMaterialVersion = 2;
enum : uint32_t {
  kRtMatCut = 1u << 0,          // alpha-cut: its BLAS has RtCutPrim primitive data
  kRtMatTexture = 1u << 1,      // alpha_texture read
  kRtMatVertexAlpha = 1u << 2,  // the vertex alphas read
  kRtMatKill = 1u << 3,         // kill_scale, kill_offset tested
  kRtMatLinear = 1u << 4,       // bilinear (else nearest)
  kRtMatBorderAlpha = 1u << 5,  // D3DTADDRESS_BORDER's alpha 1 (else 0)
  kRtMatHitPrim = 1u << 6,      // its primitive data is RtHitPrim
  kRtMatAlbedo = 1u << 7,       // albedo_texture read
};
// The rest of RtMaterial::flags: func (bits 8-11), ref (16-23), the
// texel's channel (24-25), address U (26-28) and V (29-31).
constexpr uint32_t RtMaterialFlags(uint32_t bits, uint32_t func, uint32_t ref, uint32_t channel, uint32_t u,
                                   uint32_t v) {
  return bits | (func & 15u) << 8 | (ref & 255u) << 16 | (channel & 3u) << 24 | (u & 7u) << 26 | (v & 7u) << 29;
}
struct RtMaterial {
  MTLResourceID alpha_texture;  // a 2D texture's gpuResourceID (RtCut::texture)
  float alpha_scale, alpha_offset, kill_scale, kill_offset;
  float max_level;
  uint32_t flags;
  MTLResourceID albedo_texture;  // a 2D texture's gpuResourceID (RtAlbedo::texture)
  float albedo_max_level;
  uint32_t albedo_flags;   // D3DTADDRESS U (bits 0-2), V (3-5)
  uint32_t albedo_border;  // D3DCOLOR
  uint32_t reserved[3];
};
static_assert(sizeof(RtMaterial) == 64, "kRtMsl's RtMaterial");
// kRtMsl's RtAlphaPass on the CPU, for P = p: whether a hit is kept (the
// tests of an alpha-cut material; no kRtMatCut needed).
bool RtAlphaPasses(const RtMaterial& m, float p);

// What makes a draw alpha-cut (RtScene::Add, AddCaptured): its material,
// alpha_texture left for the scene to set from `texture`.
struct RtCut {
  RtMaterial material{};
  // 2D (null: no kRtMatTexture). Held by the scene while a TLAS that may
  // still be traced names it, so it may be released after the call.
  __unsafe_unretained id<MTLTexture> texture = nil;
  // The least value the material's texel takes (0..1): its texture's least
  // value in the channel read, 1 with no texture; 0 where not known. With
  // the least vertex alpha of the mesh (its BLAS's primitive data), what
  // the product can take no less than: where the test passes it and 1, the
  // draw is traced opaque (RtCutAlwaysPasses): no hit of it returns to a
  // shader's alpha test only to be kept.
  float texel_min = 0.0f;
};
// Whether material `m` keeps every hit whose alpha product lies in
// [lo, 1] (its tests are monotonic in it, but EQUAL and NOTEQUAL).
bool RtCutAlwaysPasses(const RtMaterial& m, float lo);

// What an instance's albedo is (RtScene::Add, AddCaptured, with hit data):
// a 2D texture (held by the scene as RtCut::texture is) read at the UVs the
// draw's RtGeometry::uv gives, D3DSAMP_MAXMIPLEVEL `max_level`, D3DTADDRESS
// `address_u` and `address_v`, D3DSAMP_BORDERCOLOR `border`. Not taken where
// the mesh's UVs cannot be read (the instance is added without one).
struct RtAlbedo {
  __unsafe_unretained id<MTLTexture> texture = nil;
  float max_level = 0;
  uint32_t address_u = 1, address_v = 1;
  uint32_t border = 0;
};

// Instance masks: static meshes' opaque and alpha-cut instances, captured
// draws' opaque and alpha-cut ones, for rays that want only some kinds
// (0xFF: all).
inline constexpr uint32_t kRtMaskOpaque = 1, kRtMaskCut = 2, kRtMaskCapturedOpaque = 4, kRtMaskCapturedCut = 8;

// MSL for kernels tracing rays against an RtFrame (put before their own
// source; it includes metal_stdlib and metal_raytracing): RtInfo and
// RtMaterial (RtInstanceInfo, RtMaterial), RtCutPrim, RtHitPrim, and
//   bool RtAlphaPass(const device RtMaterial& m, const device void* prim, float2 bary)
//     whether a candidate hit on an instance of material m is kept (prim:
//     its primitive data, bary: its barycentric coordinates);
//   RtHit RtTraceClosest(instance_acceleration_structure, ray, uint mask,
//                        const device RtMaterial* table, intersection_params)
//     the closest hit RtAlphaPass keeps (reflections, debug views);
//   bool RtTraceAny(...same...)
//     whether any hit is kept (shadows: the traversal ends at the first);
// both through an intersection_query loop, committing only the
// candidates that pass;
//   bool rt_shadowed(instance_acceleration_structure, float3 o, float3 l,
//                    float tmax, float gap, const device RtMaterial* table)
//     whether the light along unit l is blocked from o: static instances
//     from o on, captured ones beyond `gap` (two traversals);
//   bool rt_shadowed(..., const device RtMaterial* table, const device RtInfo* info)
//     the same in one traversal of both, the two only where it met a
//     captured instance within the gap. RtHit: hit, distance, instance (user ID
// plus geometry ID: its slot in the tables),
// primitive, bary, front (D3D's front face), world_to_object, prim.
// With hit data (kRtMatHitPrim):
//   float3 RtHitNormal(RtHit h)  the hit triangle's unit normal, world space
//     (the triangle's own: flat, not the vertices' normals)
//   float RtHitLod(RtHit h, float w, float h, float cone)  the albedo's
//     level for a ray cone `cone` wide at the hit (ray cones, Akenine-Moller
//     et al.: the triangle's constant, the texture's size, the width over
//     the cosine of the incidence) on a w x h texture
//   float3 RtHitAlbedo(const device RtMaterial& m, RtHit h, float cone)
//     the albedo at the hit (white without kRtMatAlbedo)
extern const char* const kRtMsl;

// Per instance of a TLAS (RtFrame::info, by the instance's user ID plus the
// geometry's ID, as RtFrame::materials).
struct RtInstanceInfo {
  uint32_t draw;       // the draw's number in its presented frame (0 = first added)
  uint32_t mesh;       // the BLAS's number (stable while it is cached); 0: a captured draw's
  uint32_t triangles;
  uint32_t flags;      // MTLAccelerationStructureInstanceOptions given (bits 0-15), kRtInfo*
};
// RtInstanceInfo::flags beyond the options: the instance is another view's
// (RtView::Other: placed by a pass whose culling and projection need not
// say which side the main camera sees: its front faces are not known), a
// captured draw's, retained from an earlier presented frame (its `draw` that
// frame's).
inline constexpr uint32_t kRtInfoOtherView = 1u << 16, kRtInfoCaptured = 1u << 17, kRtInfoRetained = 1u << 18;

// A TLAS (EndFrame, Snapshot) and what tracing against it needs.
struct RtFrame {
  id<MTLAccelerationStructure> tlas = nil;  // nil: no instance
  uint32_t instances = 0;
  // Every BLAS the instances use: resident (useResources) wherever rays
  // are traced against `tlas`; unless `blas_resident`: they are in the
  // scene's residency set, which the queue tracing it has
  // (RtScene::UseResidency), and need no declaring (RtUseResources then
  // declares the textures alone).
  const __unsafe_unretained id<MTLResource>* blas = nullptr;
  uint32_t blas_count = 0;
  bool blas_resident = false;
  __unsafe_unretained id<MTLBuffer> info = nil;  // RtInstanceInfo per instance
  size_t info_offset = 0;
  __unsafe_unretained id<MTLBuffer> materials = nil;  // RtMaterial per instance
  size_t materials_offset = 0;
  // Every texture the materials name: resident (useResources) wherever
  // rays are traced against `tlas`, as `blas`.
  const __unsafe_unretained id<MTLResource>* textures = nullptr;
  uint32_t texture_count = 0;
  // The camera most instances were drawn with: clip space to world (D3D's
  // row vectors: p_world = p_clip * inv_view_proj, rows), and whether its
  // draws keep the greatest depth (D3DCMP_GREATER(EQUAL): depth 1 nearest).
  float inv_view_proj[16] = {};
  // The same in double, as the scene inverted it: positions kilometres
  // from the origin made from it keep their precision (the float matrix's
  // translation rows lose centimetres there once divided by a far point's
  // small w). Zeros: not known (inv_view_proj alone).
  double inv_view_proj_exact[16] = {};
  bool depth_reversed = false;
  // That camera's number (RtScene::CameraId); 0: none voted in this
  // presented frame (the last presented frame's camera is used).
  uint64_t camera_id = 0;
};

// What a compute encoder tracing rays against `f` declares: its BLASes
// (none where RtFrame::blas_resident) and the materials' textures, read.
void RtUseResources(id<MTLComputeCommandEncoder> e, const RtFrame* f);
// Likewise a render encoder whose `stages` (MTLRenderStageFragment, say)
// trace rays against it.
void RtUseResources(id<MTLRenderCommandEncoder> e, const RtFrame* f, MTLRenderStages stages);

struct RtBuildBatch;  // BLAS builds to encode (TakeBuilds)

// What a TLAS instance is (RtScene::DescribeNextTlas), for diagnostics:
// its view (0 the main view's camera, 1 listed with any camera: another
// view's, 2 captured), whether it is the last presented frame's (carried
// into a snapshot), whether it is retained from an earlier one (view 1), its
// draw's number in its frame, its mesh (RtInstanceInfo) and, for static
// meshes, its world-space bounds.
struct RtInstanceSummary {
  uint8_t view = 0;
  bool carried = false, retained = false, box = false;
  uint32_t draw = 0, mesh = 0, triangles = 0;
  float lo[3] = {}, hi[3] = {};
};

// Where a captured draw's vertex shader writes (AddCaptured), in the frame
// slot's memory: buf nil if the draw is not added.
struct RtCaptureSlot {
  __unsafe_unretained id<MTLBuffer> buf = nil;
  // The capture variant's r_cap (hlsl/capture_position.h): a float4 a
  // vertex, the clip-space position, from the draw's vertex `first`.
  size_t capture_off = 0;
  // Its world positions (packed float3 a vertex, the same order) once the
  // frame's work that converts them has run (EncodeFrame of the snapshot
  // or end of frame after it).
  size_t vertices_off = 0;
};

struct RtStats {
  // Since the last TakeStats: presented frames, their instances (end of
  // frame), snapshots, builds.
  uint64_t frames = 0, instances = 0, snapshots = 0, builds = 0, build_triangles = 0;
  // Of the builds, the meshes built in pieces and their pieces (RtBlas::pieces).
  uint64_t pieced = 0, pieces = 0;
  uint64_t skipped[size_t(RtSkip::Count)] = {};
  // Captured draws (AddCaptured) since the last TakeStats: instances in
  // presented frames' TLASes, BLAS builds and their triangles, and draws
  // left out by reason.
  uint64_t captured = 0, capture_builds = 0, capture_triangles = 0;
  // Of those builds, the ones that refit a BLAS an earlier frame made of
  // the same triangles (AddCaptured's key) instead of building it anew.
  uint64_t capture_refits = 0;
  // Captured draws in those BLASes (more than one a BLAS where an actor's
  // parts are one: AddCaptured's `actor`).
  uint64_t capture_draws = 0;
  uint64_t capture_skipped[size_t(RtSkip::Count)] = {};
  // Captured draws whose primitive data was kept from an earlier frame
  // (AddCaptured's key), since the last TakeStats; now: the memory kept.
  uint64_t capture_prims_reused = 0, capture_prim_bytes = 0;
  // Alpha-cut instances (RtCut) in presented frames' TLASes, static and
  // captured, since the last TakeStats.
  uint64_t cut = 0;
  // Instances with an albedo (RtAlbedo) in presented frames' TLASes, since
  // the last TakeStats.
  uint64_t albedo = 0;
  // Draws of other views given the main view's BLAS of their mesh (hit
  // data), since the last TakeStats.
  uint64_t shared_blas = 0;
  // Since the last TakeStats: instances listed with any camera in
  // presented frames' TLASes (other views' draws, and the main view's that
  // another view drew too), and draws not added as their geometry and world
  // matrix the frame already had from the other kind of view.
  uint64_t other = 0, deduped = 0;
  // Mesh keys hashed (RtMeshKeyHash: a mix of all of a key's words) for
  // static draws and their instances, since the last TakeStats: a draw
  // hashes its key once, and its geometry's (the key without what
  // alpha-cuts it) where that differs and its BLAS is not cached; every
  // later lookup of the draw or its instance in the frame takes those.
  // Building or letting go a BLAS, retaining or letting go an instance
  // hash keys again.
  uint64_t key_hashes = 0;
  // Draws that discard by what is not known to be a texture's alpha (a
  // clip plane, a fade by screen position), added opaque (NoteUncut).
  uint64_t uncut = 0;
  // Alpha-cut draws added opaque all the same: their mesh's least vertex
  // alpha times their texture's least texel passes the test everywhere
  // (RtCut::texel_min, RtCutAlwaysPasses).
  uint64_t cut_opaque = 0;
  // Retained instances listed in presented frames' TLASes (not drawn that
  // frame) and those let go, by reason, since the last TakeStats; now: those
  // kept.
  uint64_t retained = 0, retained_dropped[size_t(RtRetainDrop::Count)] = {}, retained_kept = 0;
  uint64_t retained_cap_near = 0;  // of the Cap drops, those within 100 m of the camera
  // Now: textures the scene holds for material tables (alpha and albedo),
  // their memory.
  uint64_t cut_textures = 0, cut_texture_bytes = 0;
  // Now.
  uint64_t cached = 0, blas_bytes = 0, retired = 0, retired_bytes = 0, total_builds = 0;
  // Captured draws' memory: their BLASes (in use and pooled) and the frame
  // slots' capture buffers.
  uint64_t capture_blas_bytes = 0, capture_buffer_bytes = 0;
  // Now: the places (a geometry at a world matrix) of static draws the
  // scene keeps from presented frame to frame to dedupe two views' draws,
  // the frame's and earlier ones' (at most kPlacedSlack times as many as an
  // epoch took, at least kPlacedFloor, plus this epoch's); since the last
  // TakeStats: those let go (at most kPlacedSlack times as many as the
  // epoch before took, or kPlacedFloor, at once).
  uint64_t places = 0, places_dropped = 0;
  // Since the last TakeStats, sampled in one presented frame of
  // RtScene::kStaticSampleEvery: the frames sampled, the static instances
  // (not captured: drawn, carried and retained) their first TLAS listed (a
  // snapshot lists the retained ones not drawn yet, so much the same set
  // as a later one), and the distinct pairs of world matrix and instance
  // options among those (by a 64-bit hash of the pair).
  // Instances that share a pair are parts drawn under one world matrix
  // (a model's materials, a draw each): what one BLAS of several
  // geometries could hold as one instance.
  uint64_t static_samples = 0, static_listed = 0, static_distinct = 0;
};

class RtScene {
 public:
  // Frames recorded but not finished on the GPU (the backend's frame slots,
  // which may be fewer).
  static constexpr int kSlots = 4;
  // BLAS builds a frame encodes at most; meshes beyond them wait for later
  // frames (not in the scene until then). Keeps a level's first frames from
  // stalling on hundreds of builds.
  static constexpr uint32_t kBuildsPerFrame = 64;
  // A BLAS no draw has used for this many presented frames is let go
  // (unless a retained instance names it).
  static constexpr uint64_t kEvictAfter = 300;
  // Presented frames in a row the camera sees all of a retained instance
  // and none draws it before it is let go (half a second at 60 fps): the
  // game's occlusion culling leaves out what is hidden behind something
  // nearer, which must not come and go with it.
  // (Off by default since 2026-11-11: a building the game leaves out as
  // hidden behind a nearer one still casts its shadow into the view, and
  // was let go with it: shadows near the camera went out while it stood
  // still. RECOMP_RT_IN_VIEW_FRAMES=n: let go after n frames, as before.)
  static constexpr uint32_t kInViewFrames = 30;
  // Retained instances within this many metres of the camera are kept
  // however long they go undrawn (Idle applies beyond it): the game leaves
  // out what it hides or culls for as long as it does, and its shadow is
  // still there. max_retained still bounds them (the farthest go).
  static constexpr double kRetainNear = 200.0;
  // A retained instance not drawn is let go once the camera is closer to
  // its bounds' centre than this share of the least distance it was ever
  // drawn from: a game draws a far level of detail only from afar and its
  // near models in its place from close by, at other world matrices and
  // with other bounds (several near models to one far one), so Replaced
  // does not see it; kept, the far model's coarse faces cast shadows
  // through the near ones (a road's far model above the road itself). What
  // a game hides behind something nearer was drawn from about where the
  // camera is, and stays.
  static constexpr double kRetainCloser = 0.5;
  // The places of static draws kept from frame to frame (RtStats::places):
  // past kPlacedSlack times as many as the epoch that just ended took (a
  // presented frame's, at least kPlacedFloor), those of earlier epochs go.
  static constexpr uint64_t kPlacedSlack = 2, kPlacedFloor = 256;
  // Captured draws a presented frame takes at most, with their triangles
  // and vertices: their BLASes are refit (or built) every frame on the GPU,
  // and their indices gathered on the drawing thread where not kept from an
  // earlier frame (about a millisecond per 512k triangles). A street's
  // people and vehicles took some 100-300k triangles; Star Junction at
  // midday asks for 360-440 draws a frame.
  static constexpr uint32_t kCapturesPerFrame = 512;
  static constexpr uint32_t kCaptureTrianglesPerFrame = 1u << 20;
  static constexpr uint32_t kCaptureVerticesPerFrame = 1u << 20;
  // Refits of a captured draw's BLAS (AddCaptured's key) before it is built
  // anew: a refit keeps the tree the build made, which fits the triangles
  // less well the further they move from where they were then.
  static constexpr uint32_t kCaptureRefits = 8;
  // Parts an actor's BLAS takes at most (AddCaptured's `actor`).
  static constexpr uint32_t kActorParts = 64;
  // One presented frame in this many counts the distinct world matrices of
  // its static instances (RtStats::static_distinct): a hash an instance and
  // a sort of them, a few tenths of a millisecond at 8000 instances
  // (estimated, not measured), so not every frame.
  static constexpr uint64_t kStaticSampleEvery = 120;

  virtual ~RtScene() = default;

  // ---- recording side
  virtual void SetAllocator(RtAlloc alloc, void* ctx) = 0;
  // Frame (command buffer) `serial` begins; serials rise by one. Its slot's
  // last frame (kSlots before) is done on the GPU; `completed` says up to
  // which frame the caller knows the GPU has finished (it may lag: a
  // backend pacing frames by the display frees a slot when its frame is
  // shown).
  virtual void BeginFrame(uint64_t serial, uint64_t completed) = 0;
  // A draw: mesh `key` with `g`'s triangles, its object-to-world `world`
  // and object-to-clip `wvp` matrices (4 rows each, D3D's row vectors),
  // D3DCULL `cull` (1 none, 2 clockwise, 3 counterclockwise) and
  // D3DCMPFUNC `zfunc` (its depth test). A mesh not cached is gathered now
  // and its BLAS built in this frame's work. `cut`: alpha-cut (the key's
  // cut 1, g's uv and alpha read for its primitive data), null: opaque.
  // `view` (RtView): Main draws vote for the frame's camera and are listed
  // with it; Other draws do not vote and are listed in every TLAS of the
  // presented frame (not carried into the next one's snapshots: drawn again
  // each frame); one whose world matrix mirrors is left out. The same mesh
  // (its geometry: the key without what alpha-cuts it) at the same world
  // matrix from both kinds is one instance (counted, RtStats::deduped):
  // the main view's, with its material, listed with any camera; an Other
  // draw after any adds none, a Main one after an Other one takes that
  // instance's place (and votes). False if it is not
  // added (counted under the reason, or a duplicate), and after EndFrame
  // until the next BeginFrame.
  // `albedo` (with hit data; null: none): its texture named in the
  // instance's material, read at g's uv (as an alpha-cut draw's UVs).
  virtual bool Add(const RtMeshKey& key, const RtGeometry& g, const float (*world)[4],
                   const float (*wvp)[4], uint32_t cull, uint32_t zfunc, const RtCut* cut = nullptr,
                   RtView view = RtView::Main, const RtAlbedo* albedo = nullptr) = 0;
  virtual void Skip(RtSkip why) = 0;
  // A draw whose positions its vertex shader makes on the GPU: `g` its
  // triangles (positions unused), `first` and `count` the vertices it
  // captures (RtIndexRange), D3DCULL `cull`. Its vertex shader's capture
  // variant writes them (RtCaptureSlot) in this frame's work, before the
  // next Snapshot or EndFrame; those convert them with the camera most
  // static draws so far share (that of the last presented frame if none):
  // drawn with another (a reflection, say), it lands in the wrong place. Not
  // added (buf nil, counted under the reason) past the budgets or without
  // memory, and after EndFrame until the next BeginFrame. `cut`: alpha-cut,
  // g's uv and alpha read on the CPU now (from the vertex streams, not the
  // vertex shader's outputs); `albedo` as Add's. `key` (null: none): what
  // the UVs, alphas and triangles are made from (the indices', the UVs' and
  // alphas' stream, versions included; the positions' fields unused, `first`
  // and `count` as min_index and vertex_count): with versions, the primitive
  // data made on the CPU is kept for the next frames' draws of the same
  // (RtStats::capture_prims_reused), as only the positions change. And with
  // a key, the BLAS an earlier frame built for a draw of the same key (the
  // same triangles) is refit to the new positions, not built anew, while it
  // has been refit fewer than kCaptureRefits times in a row
  // (RtStats::capture_refits): Apple's way for meshes that deform
  // (WWDC22 10105, "Maximize your Metal ray tracing performance").
  // `actor` (0: none): the draw is one part of a whole the caller can tell
  // apart (a person or vehicle: the vertex shader state that places its
  // vertices). With a key, consecutive captures of the same actor and
  // D3DCULL in one batch (up to kActorParts) become one BLAS, a geometry
  // each, and one TLAS instance (Apple: fewer instances trace faster,
  // WWDC23 10128, "Your guide to Metal ray tracing"); refit as one. Its
  // materials and instance info take a slot each, consecutive from the
  // instance's user ID: a hit's is user ID + geometry ID (kRtMsl's RtHit::
  // instance).
  virtual RtCaptureSlot AddCaptured(const RtGeometry& g, uint32_t first, uint32_t count, uint32_t cull,
                                    const RtCut* cut = nullptr, const RtAlbedo* albedo = nullptr,
                                    const RtMeshKey* key = nullptr, uint64_t actor = 0) = 0;
  // A draw to be captured, left out before AddCaptured (counted apart).
  virtual void SkipCapture(RtSkip why) = 0;
  // A draw that discards pixels by what is not known to be a texture's
  // alpha, added (or to be added) opaque: counted (RtStats::uncut).
  virtual void NoteUncut() = 0;
  // The next TLAS made (Snapshot, EndFrame) also summarises its instances
  // into `out`, by user ID (cleared first; null: none).
  virtual void DescribeNextTlas(std::vector<RtInstanceSummary>* out) = 0;
  // The retained instances `f` (a snapshot just made) lists: each one's
  // first slot (RtInstanceInfo, RtMaterial; a ray's hit names it) and its
  // number, for DropRetained (cleared first).
  virtual void RetainedSlots(const RtFrame* f, std::vector<std::pair<uint32_t, uint64_t>>* out) = 0;
  // Lets go the retained instances with these numbers (those still kept and
  // not drawn this frame): what rays found lying on the surfaces the camera
  // sees, a model the game no longer draws there (RtRetainDrop::Replaced).
  virtual void DropRetained(const uint64_t* ids, size_t n) = 0;
  // The camera of the last Add's Main draw (it voted for it), then 0 until
  // the next: a number for this presented frame's camera, as CameraId and
  // RtFrame::camera_id give them; 0 if the last Add was not a Main draw or
  // not taken before its camera was known.
  virtual uint64_t TakeVote() = 0;
  // The camera the frame's TLASes use now (most votes so far this
  // presented frame); 0 if no Main draw voted yet.
  virtual uint64_t CameraId() const = 0;
  // The camera of this presented frame a draw with these matrices (the
  // object-to-world rows `world`, object to clip `wvp`) was drawn with:
  // world^-1 * wvp one of the frame's main-view cameras (as Add matches
  // them; no vote); 0 if none is.
  virtual uint64_t CameraOf(const float (*world)[4], const float (*wvp)[4]) const = 0;
  // Whether BLAS builds are waiting for the caller to encode between two
  // render passes (TakeBuilds, EncodeBuilds). Never by default: the scene
  // keeps them for the next Snapshot's or EndFrame's work, which encodes
  // them with the captured draws' builds and refits in one encoder (Metal
  // runs the builds of one encoder side by side; an encoder of their own
  // at a pass change cost about 0.45 ms in the frames that had one).
  // RECOMP_RT_BLAS_POOL=0: as before, whenever any waits.
  virtual bool PendingBuilds() const = 0;
  // The builds added since the last call, to encode next (null: none).
  virtual const RtBuildBatch* TakeBuilds() = 0;
  // A TLAS now, in the middle of a presented frame, for effects traced
  // before it ends (EncodeFrame builds it, after the builds not yet taken):
  // the instances added so far in this presented frame, and those the last
  // presented frame added after its last snapshot (drawn later in a frame,
  // characters and vehicles after the lighting, say), at that frame's
  // places, so one frame behind; and the retained ones not drawn so far.
  // Null if there is nothing to encode.
  virtual const RtFrame* Snapshot() = 0;
  // The end of a frame. Presented: its TLAS of every instance the
  // presented frame added and the retained ones it did not draw (the
  // retained set updated first: this frame's static instances in, those let
  // go out; EncodeFrame builds it, after the builds not yet taken; none
  // without RtSceneOptions::end_tlas). Not presented (a readback ended the command buffer): only the
  // builds, the instances go on into the next frame. Null if there is
  // nothing to encode.
  virtual const RtFrame* EndFrame(bool presented) = 0;

  // ---- encoding side
  virtual void EncodeBuilds(id<MTLCommandBuffer> cmd, const RtBuildBatch* b) const = 0;
  virtual void EncodeFrame(id<MTLCommandBuffer> cmd, const RtFrame* f) const = 0;
  // The debug view (RtSceneOptions::debug) over `target` (BGRA8, read by
  // shaders): each pixel's ray from the frame's camera, its hit shown,
  // pixels without one darkened.
  virtual void EncodeDebug(id<MTLCommandBuffer> cmd, const RtFrame* f, id<MTLTexture> target) = 0;

  // ---- statistics (recording side)
  virtual RtStats Stats() const = 0;
  // A line for the perf report, per presented frame; resets the counts.
  virtual std::string TakeStats() = 0;
  // BLASes (retired ones the GPU may still use included), TLASes, scratch,
  // captured draws' BLASes and buffers, textures held for material tables,
  // retained instances' records, the places of static draws kept
  // (RtStats::places), the last and this frame's static instances.
  virtual uint64_t MemoryBytes() const = 0;

  // ---- residency
  // From now on every BLAS the scene holds is kept in a residency set
  // (MTLResidencySet, committed before the snapshots and ends of frames
  // that may name new ones) that `queue` has: its frames' BLASes need no
  // declaring in every encoder that traces them (RtFrame::blas_resident),
  // as Apple recommends for resources a GPU reaches indirectly, through an
  // acceleration structure or argument buffer, instead of useResource
  // calls in each encoder (MTLResidencySet,
  // https://developer.apple.com/documentation/metal/mtlresidencyset).
  // Only command buffers of `queue` may then trace the scene. False
  // (declared as before) where the OS has none (before macOS 15, iOS 18)
  // or it cannot be made.
  virtual bool UseResidency(id<MTLCommandQueue> queue) = 0;
};

// (Whether `dev` traces rays in hardware with what RtScene uses:
// RtSupported, gfx/rt_support.h.)

// The BLAS memory cap for `dev` (RtSceneOptions::max_blas_bytes 0).
uint64_t RtDefaultBlasBytes(id<MTLDevice> dev);
// Null if !RtSupported(dev).
std::unique_ptr<RtScene> MakeRtScene(id<MTLDevice> dev, const RtSceneOptions& o);

}  // namespace recomp::gfx
