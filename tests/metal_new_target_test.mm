// A new render target on the Metal backend (gfx/metal_new_target.h):
// textures first filled with not-a-number halves (float targets) or a
// colour (8-bit ones) in every level, as a texture fresh from Metal may
// be, must read back as zeros in every level after ClearNewTarget.
//
//   metal_new_target_test   (skipped, exit 77, without a Metal device)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "runtime/gfx/metal_new_target.h"

namespace {

int g_failed = 0;

// The bytes of `level` of `tex` (bytes_per_pixel each).
std::vector<uint8_t> ReadLevel(id<MTLDevice> dev, id<MTLCommandQueue> queue, id<MTLTexture> tex,
                               NSUInteger level, NSUInteger bytes_per_pixel) {
  const NSUInteger w = std::max<NSUInteger>(1, tex.width >> level);
  const NSUInteger h = std::max<NSUInteger>(1, tex.height >> level);
  id<MTLBuffer> out = [dev newBufferWithLength:w * h * bytes_per_pixel options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cmd blitCommandEncoder];
  [bl copyFromTexture:tex
                   sourceSlice:0
                   sourceLevel:level
                  sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:MTLSizeMake(w, h, 1)
                      toBuffer:out
             destinationOffset:0
        destinationBytesPerRow:w * bytes_per_pixel
      destinationBytesPerImage:w * h * bytes_per_pixel];
  [bl endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  const auto* p = static_cast<const uint8_t*>(out.contents);
  return std::vector<uint8_t>(p, p + w * h * bytes_per_pixel);
}

// Every level of `tex` set to `fill` (one pixel's bytes) from a buffer.
void FillLevels(id<MTLDevice> dev, id<MTLCommandQueue> queue, id<MTLTexture> tex,
                const std::vector<uint8_t>& fill) {
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cmd blitCommandEncoder];
  for (NSUInteger level = 0; level < tex.mipmapLevelCount; level++) {
    const NSUInteger w = std::max<NSUInteger>(1, tex.width >> level);
    const NSUInteger h = std::max<NSUInteger>(1, tex.height >> level);
    id<MTLBuffer> src = [dev newBufferWithLength:w * h * fill.size() options:MTLResourceStorageModeShared];
    auto* p = static_cast<uint8_t*>(src.contents);
    for (NSUInteger i = 0; i < w * h; i++) std::memcpy(p + i * fill.size(), fill.data(), fill.size());
    [bl copyFromBuffer:src
               sourceOffset:0
          sourceBytesPerRow:w * fill.size()
        sourceBytesPerImage:w * h * fill.size()
                 sourceSize:MTLSizeMake(w, h, 1)
                  toTexture:tex
           destinationSlice:0
           destinationLevel:level
          destinationOrigin:MTLOriginMake(0, 0, 0)];
  }
  [bl endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
}

void Case(id<MTLDevice> dev, id<MTLCommandQueue> queue, const char* name, MTLPixelFormat pf,
          NSUInteger w, NSUInteger h, NSUInteger levels, const std::vector<uint8_t>& fill) {
  MTLTextureDescriptor* td = [MTLTextureDescriptor new];
  td.textureType = MTLTextureType2D;
  td.pixelFormat = pf;
  td.width = w;
  td.height = h;
  td.mipmapLevelCount = levels;
  td.storageMode = MTLStorageModePrivate;
  td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
  id<MTLTexture> tex = [dev newTextureWithDescriptor:td];
  FillLevels(dev, queue, tex, fill);
  // The fill is there: a level that reads zeros below was cleared.
  for (NSUInteger level = 0; level < levels; level++) {
    const std::vector<uint8_t> before = ReadLevel(dev, queue, tex, level, fill.size());
    bool filled = !before.empty();
    for (size_t i = 0; i < before.size(); i += fill.size())
      filled = filled && std::memcmp(before.data() + i, fill.data(), fill.size()) == 0;
    if (!filled) {
      std::printf("FAIL %s: level %lu was not filled\n", name, (unsigned long)level);
      g_failed++;
    }
  }
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  recomp::gfx::ClearNewTarget(cmd, tex);
  [cmd commit];
  [cmd waitUntilCompleted];
  for (NSUInteger level = 0; level < levels; level++) {
    const std::vector<uint8_t> after = ReadLevel(dev, queue, tex, level, fill.size());
    size_t bad = 0;
    for (uint8_t b : after) bad += b != 0;
    if (bad) {
      std::printf("FAIL %s: level %lu has %zu bytes that are not zero\n", name, (unsigned long)level, bad);
      g_failed++;
    }
  }
}

}  // namespace

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      std::printf("no Metal device: skipped\n");
      return 77;
    }
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    // Halves 0xFFFF: not a number.
    Case(dev, queue, "float target, one level", MTLPixelFormatRGBA16Float, 64, 64, 1,
         {0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF});
    Case(dev, queue, "float target, levels", MTLPixelFormatRGBA16Float, 64, 32, 7,
         {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF});
    Case(dev, queue, "8-bit target, levels", MTLPixelFormatBGRA8Unorm, 32, 32, 6, {0xFF, 0x00, 0xFF, 0xFF});
    Case(dev, queue, "one-channel float target", MTLPixelFormatR32Float, 16, 16, 1, {0xFF, 0xFF, 0xFF, 0xFF});
  }
  if (g_failed) return 1;
  std::printf("ok\n");
  return 0;
}
