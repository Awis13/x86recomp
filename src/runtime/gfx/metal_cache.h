// What the Metal backend keeps between runs, and how a pipeline is made
// from it: shared by the device (metal_device.mm) and PrewarmCaches
// (prewarm.mm), which must make exactly the same functions and pipelines
// for the pipeline archive (metal_archive.h) to hold what the game looks
// for. Objective-C++ only.
//
// In the user's caches, folder x86recomp (MetalCacheDir):
// - metal-msl-v<kMslCacheVersion>.bin: translated shaders' MSL, records of
//   Shader::hash (8 bytes), length (4) and text, the last of a hash wins;
// - metal-pipelines-v1.bin: pipelines compiled in earlier runs, as
//   PipelineRecipe records;
// - metal-archives-v1/: the pipeline archive.

#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include "runtime/gfx/metal_archive.h"
#include "runtime/gfx/metal_prelude.h"

namespace recomp::gfx {

// A render pipeline: shaders, the pass's attachment formats, blending and
// write masks (Metal enums). Compared and hashed as bytes: no padding.
struct PipeKey {
  uint32_t vs, ps;
  uint16_t color[4];  // MTLPixelFormat per colour target (0: none)
  uint16_t depth, stencil;
  uint8_t blend, src, dst, op, src_a, dst_a, op_a;
  uint8_t mask[4];
  uint8_t lean;  // kLean*: the shaders' lean variants
  bool operator==(const PipeKey& o) const { return std::memcmp(this, &o, sizeof *this) == 0; }
};
static_assert(sizeof(PipeKey) == 32, "no padding");
// PipeKey::lean: a shader made without what the draw does not use.
constexpr uint8_t kLeanPs = 1;  // the pixel shader without the alpha test

// A pipeline compiled in a run, for later runs: its shaders' translations
// (Shader::hash) and the rest of its key (vs, ps 0). The pipeline
// archive's key for it is the record's hash (ArchiveKey).
struct PipelineRecipe {
  uint64_t vs, ps;
  PipeKey key;
};
static_assert(sizeof(PipelineRecipe) == 48, "no padding");

inline uint64_t Fnv64(const void* p, size_t n) {
  uint64_t h = 0xCBF29CE484222325ull;
  for (size_t i = 0; i < n; i++) h = (h ^ static_cast<const uint8_t*>(p)[i]) * 0x100000001B3ull;
  return h;
}

inline uint64_t ArchiveKey(const PipelineRecipe& r) { return Fnv64(&r, sizeof r); }

// A pipeline's descriptor from its key and functions.
inline MTLRenderPipelineDescriptor* DescribePipeline(const PipeKey& k, id<MTLFunction> vs, id<MTLFunction> ps) {
  MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
  pd.vertexFunction = vs;
  pd.fragmentFunction = ps;
  for (int i = 0; i < 4; i++) {
    if (!k.color[i]) continue;
    MTLRenderPipelineColorAttachmentDescriptor* a = pd.colorAttachments[NSUInteger(i)];
    a.pixelFormat = MTLPixelFormat(k.color[i]);
    a.blendingEnabled = k.blend != 0;
    a.sourceRGBBlendFactor = MTLBlendFactor(k.src);
    a.destinationRGBBlendFactor = MTLBlendFactor(k.dst);
    a.rgbBlendOperation = MTLBlendOperation(k.op);
    a.sourceAlphaBlendFactor = MTLBlendFactor(k.src_a);
    a.destinationAlphaBlendFactor = MTLBlendFactor(k.dst_a);
    a.alphaBlendOperation = MTLBlendOperation(k.op_a);
    a.writeMask = MTLColorWriteMask(k.mask[i]);
  }
  pd.depthAttachmentPixelFormat = MTLPixelFormat(k.depth);
  pd.stencilAttachmentPixelFormat = MTLPixelFormat(k.stencil);
  return pd;
}

// Shaders' lean variants (MetalLeanMacros): pixel shaders without the
// alpha test, from the same MSL with the macro R_ALPHA_TEST false; `full`:
// with it true, as every shader that has the macro is compiled.
inline MTLCompileOptions* LeanMslOptions(bool full) {
  static MTLCompileOptions* options[2];
  static std::once_flag once;
  std::call_once(once, [] {
    for (int i = 0; i < 2; i++) {
      options[i] = [MTLCompileOptions new];
      options[i].preprocessorMacros = @{@"R_ALPHA_TEST" : i ? @"true" : @"false"};
    }
  });
  return options[full ? 1 : 0];
}

// The MSL cache's version: it changes with the translation's settings
// (shader_compile.cpp).
constexpr uint32_t kMslCacheVersion = 1;

// The runtime's cache folder, made if missing.
inline std::string MetalCacheDir() {
  NSString* dir = [NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES).firstObject
      stringByAppendingPathComponent:@"x86recomp"];
  [NSFileManager.defaultManager createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
  return dir.fileSystemRepresentation;
}

inline std::string RecipeFile(const std::string& dir) { return dir + "/metal-pipelines-v1.bin"; }
inline std::string MslCacheFile(const std::string& dir) {
  return dir + "/metal-msl-v" + std::to_string(kMslCacheVersion) + ".bin";
}
inline std::string ArchiveFolder(const std::string& dir) { return dir + "/metal-archives-v1"; }

// The pipeline archive's scope: PipelineArchive::DefaultScope with the
// shader translation's version (the MSL cache's and the preludes'): after
// a change of it, older shards would only be looked in for nothing.
inline std::string ArchiveScope(id<MTLDevice> dev) {
  const std::string prelude = MetalShaderPrelude(hlsl::Stage::Vertex) + MetalShaderPrelude(hlsl::Stage::Pixel);
  char msl[64];
  std::snprintf(msl, sizeof msl, "msl=%u-%016llx\n", kMslCacheVersion,
                (unsigned long long)Fnv64(prelude.data(), prelude.size()));
  return PipelineArchive::DefaultScope(dev) + msl;
}

}  // namespace recomp::gfx
