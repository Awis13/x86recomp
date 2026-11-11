// Sharpening on the Metal backend (gfx/rcas.h): rcas_cs, run on the GPU,
// must give what AMD's FsrRcasF gives (a C++ copy of it below, with
// FSR_RCAS_DENOISE, dividing by zero as AMD's code does: IEEE results, the
// NaN dropped by max and min as HLSL's are) on random images at several
// amounts, at a small size and a screen's, and on images with a channel
// at 0 or 1 all over (no blue in a dark street, a clipped sky), which
// must still be sharpened; leave a flat image as it is; make a
// soft edge steeper without leaving 0..1; leave an edge from black to
// white as it is (no room to sharpen without clipping); and sharpen a
// lone noisy pixel much less than it would without the noise detection.
//
//   metal_rcas_test        (skipped, exit 77, without a Metal device)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "runtime/gfx/rcas.h"

namespace {

using recomp::gfx::RcasConstant;

int failures = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    failures++;
  }
}

struct Rgb {
  float r, g, b;
};
using Image = std::vector<Rgb>;  // w * h, rows from the top

// FsrRcasF (ffx_fsr1.h, 32-bit path, FSR_RCAS_DENOISE), the reference
// (AMD's code, MIT: its notice is in runtime/gfx/rcas.h); denoise false is
// the same without FSR_RCAS_DENOISE. Its reciprocals exact (ARcpF1), so
// 0/0 is NaN here as there, and std::fmax/fmin drop it as HLSL's max/min
// do; the noise term's reciprocal is AMD's approximate one, finite at 0
// (times a difference of 0 there: 0).
Image Reference(const Image& in, int w, int h, float con, bool denoise = true) {
  auto at = [&](int x, int y) {
    const Rgb p = in[size_t(std::clamp(y, 0, h - 1) * w + std::clamp(x, 0, w - 1))];
    return Rgb{std::clamp(p.r, 0.f, 1.f), std::clamp(p.g, 0.f, 1.f), std::clamp(p.b, 0.f, 1.f)};
  };
  Image out(in.size());
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      const Rgb b = at(x, y - 1), d = at(x - 1, y), e = at(x, y), f = at(x + 1, y), hh = at(x, y + 1);
      auto luma = [](Rgb c) { return c.b * 0.5f + (c.r * 0.5f + c.g); };
      const float bL = luma(b), dL = luma(d), eL = luma(e), fL = luma(f), hL = luma(hh);
      const float range = std::max({bL, dL, eL, fL, hL}) - std::min({bL, dL, eL, fL, hL});
      float nz = 0.25f * bL + 0.25f * dL + 0.25f * fL + 0.25f * hL - eL;
      nz = range > 0.0f ? std::clamp(std::fabs(nz) / range, 0.f, 1.f) : 0.0f;
      nz = -0.5f * nz + 1.0f;
      auto lobe_of = [](float b, float d, float e, float f, float h) {
        const float mn4 = std::min({b, d, f, h}), mx4 = std::max({b, d, f, h});
        const float hit_min = std::min(mn4, e) * (1.0f / (4.0f * mx4));
        const float hit_max = (1.0f - std::max(mx4, e)) * (1.0f / (4.0f * mn4 - 4.0f));
        return std::fmax(-hit_min, hit_max);
      };
      const float lr = lobe_of(b.r, d.r, e.r, f.r, hh.r), lg = lobe_of(b.g, d.g, e.g, f.g, hh.g),
                  lb = lobe_of(b.b, d.b, e.b, f.b, hh.b);
      float lobe = std::fmax(-(0.25f - 1.0f / 16.0f), std::fmin(std::fmax(std::fmax(lr, lg), lb), 0.0f)) * con;
      if (denoise) lobe *= nz;
      const float rcp = 1.0f / (4.0f * lobe + 1.0f);
      auto res = [&](float b, float d, float e, float f, float h) {
        return std::clamp((lobe * (b + d + h + f) + e) * rcp, 0.f, 1.f);
      };
      out[size_t(y * w + x)] = {res(b.r, d.r, e.r, f.r, hh.r), res(b.g, d.g, e.g, f.g, hh.g),
                                res(b.b, d.b, e.b, f.b, hh.b)};
    }
  return out;
}

struct Gpu {
  id<MTLDevice> dev;
  id<MTLCommandQueue> q;
  id<MTLComputePipelineState> pipe;

  // rcas_cs on an RGBA32F copy of the image, each value first rounded
  // through half (as the back buffer and TAA's history hold it), into
  // RGBA32F (no 8-bit rounding in the comparison).
  Image Run(const Image& in, int w, int h, float con) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                  width:NSUInteger(w)
                                                                                 height:NSUInteger(h)
                                                                              mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> src = [dev newTextureWithDescriptor:td], dst = [dev newTextureWithDescriptor:td];
    std::vector<float> px(size_t(w * h * 4));
    for (size_t i = 0; i < in.size(); i++) {
      // Through half, as the back buffer and TAA's history hold it.
      px[i * 4 + 0] = float(__fp16(in[i].r));
      px[i * 4 + 1] = float(__fp16(in[i].g));
      px[i * 4 + 2] = float(__fp16(in[i].b));
      px[i * 4 + 3] = 1.0f;
    }
    [src replaceRegion:MTLRegionMake2D(0, 0, NSUInteger(w), NSUInteger(h))
           mipmapLevel:0
             withBytes:px.data()
           bytesPerRow:NSUInteger(w) * 16];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pipe];
    [e setTexture:src atIndex:0];
    [e setTexture:dst atIndex:1];
    [e setBytes:&con length:sizeof con atIndex:0];
    [e dispatchThreads:MTLSizeMake(NSUInteger(w), NSUInteger(h), 1) threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    [dst getBytes:px.data() bytesPerRow:NSUInteger(w) * 16 fromRegion:MTLRegionMake2D(0, 0, NSUInteger(w), NSUInteger(h))
        mipmapLevel:0];
    Image out(in.size());
    for (size_t i = 0; i < out.size(); i++) out[i] = {px[i * 4], px[i * 4 + 1], px[i * 4 + 2]};
    return out;
  }
};

float MaxDiff(const Image& a, const Image& b) {
  float m = 0;
  for (size_t i = 0; i < a.size(); i++)
    m = std::max({m, std::fabs(a[i].r - b[i].r), std::fabs(a[i].g - b[i].g), std::fabs(a[i].b - b[i].b)});
  return m;
}

// The input through half, as the GPU reads it, for the reference.
Image AsHalf(Image in) {
  for (Rgb& p : in) p = {float(__fp16(p.r)), float(__fp16(p.g)), float(__fp16(p.b))};
  return in;
}

}  // namespace

int main() {
  @autoreleasepool {
    // The amount's mapping: 0..1 to 2..0 stops.
    Check(std::fabs(RcasConstant(1.0f) - 1.0f) < 1e-6f, "amount 1 is the strongest");
    Check(std::fabs(RcasConstant(0.75f) - std::exp2(-0.5f)) < 1e-6f, "amount 0.75 is half a stop");
    Check(std::fabs(RcasConstant(0.5f) - 0.5f) < 1e-6f, "amount 0.5 is one stop");
    Check(std::fabs(RcasConstant(0.0f) - 0.25f) < 1e-6f, "amount 0 maps to two stops");
    Check(RcasConstant(2.0f) == RcasConstant(1.0f) && RcasConstant(-1.0f) == RcasConstant(0.0f),
          "the amount is clamped");
    // 0 is no pass at all (the device skips the dispatch and its texture).
    Check(!recomp::gfx::RcasActive(0.0f) && !recomp::gfx::RcasActive(-0.5f), "amount 0 is no sharpening");
    Check(recomp::gfx::RcasActive(0.01f) && recomp::gfx::RcasActive(1.0f), "any amount above 0 sharpens");

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      std::printf("metal_rcas_test: no Metal device, skipped\n");
      return 77;
    }
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(recomp::gfx::kRcasMsl) options:nil error:&err];
    if (!lib) {
      std::fprintf(stderr, "FAIL: rcas_cs does not compile: %s\n", err.localizedDescription.UTF8String);
      return 1;
    }
    Gpu gpu{dev, [dev newCommandQueue],
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rcas_cs"] error:&err]};
    if (!gpu.pipe) {
      std::fprintf(stderr, "FAIL: rcas_cs pipeline: %s\n", err.localizedDescription.UTF8String);
      return 1;
    }

    // Random images (and values a little outside 0..1, clamped by both),
    // against the reference.
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-0.05f, 1.05f);
    const int w = 61, h = 37;  // not multiples of the threadgroup
    for (float amount : {0.5f, 0.75f, 0.9f, 1.0f}) {
      Image in(size_t(w * h));
      for (Rgb& p : in) p = {u(rng), u(rng), u(rng)};
      const float con = RcasConstant(amount);
      const float d = MaxDiff(gpu.Run(in, w, h, con), Reference(AsHalf(in), w, h, con));
      char what[96];
      std::snprintf(what, sizeof what, "GPU matches FsrRcasF at amount %.2f (max diff %.5f)", amount, d);
      Check(d < 2e-3f, what);
    }

    // At a screen's size (the back buffer after an upscale).
    {
      const int bw = 1920, bh = 1080;
      Image in(size_t(bw * bh));
      for (Rgb& p : in) p = {u(rng), u(rng), u(rng)};
      const float con = RcasConstant(0.75f);
      const float d = MaxDiff(gpu.Run(in, bw, bh, con), Reference(AsHalf(in), bw, bh, con));
      Check(d < 2e-3f, "GPU matches FsrRcasF at 1920x1080");
    }

    // A channel at 0 or at 1 all over, the others varying: the same as
    // AMD's, and still sharpened (the others' values move).
    for (float fixed : {0.0f, 1.0f}) {
      std::uniform_real_distribution<float> mid(0.2f, 0.8f);
      Image in(size_t(w * h));
      for (Rgb& p : in) p = {mid(rng), mid(rng), fixed};
      const Image out = gpu.Run(in, w, h, 1.0f);
      const float d = MaxDiff(out, Reference(AsHalf(in), w, h, 1.0f));
      char what[96];
      std::snprintf(what, sizeof what, "GPU matches FsrRcasF with blue at %.0f all over (max diff %.5f)", fixed, d);
      Check(d < 2e-3f, what);
      std::snprintf(what, sizeof what, "an image with blue at %.0f all over is sharpened", fixed);
      Check(MaxDiff(out, AsHalf(in)) > 0.05f, what);
    }

    // A flat image stays flat.
    {
      Image in(size_t(w * h), Rgb{0.4f, 0.55f, 0.3f});
      Check(MaxDiff(gpu.Run(in, w, h, 1.0f), AsHalf(in)) < 1e-3f, "a flat image is unchanged");
    }

    // A soft vertical edge (0.3, 0.5, 0.7 across three columns): the dark
    // side darker, the bright side brighter, inside 0..1.
    {
      Image in(size_t(w * h));
      for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
          const float v = x < 30 ? 0.3f : x == 30 ? 0.5f : 0.7f;
          in[size_t(y * w + x)] = {v, v, v};
        }
      const Image out = gpu.Run(in, w, h, 1.0f);
      const Rgb dark = out[size_t(10 * w + 29)], bright = out[size_t(10 * w + 31)];
      Check(dark.g < 0.3f - 0.01f, "the edge's dark side is darker");
      Check(bright.g > 0.7f + 0.01f, "the edge's bright side is brighter");
      bool inside = true;
      for (const Rgb& p : out) inside = inside && p.r >= 0.f && p.r <= 1.f && p.g >= 0.f && p.g <= 1.f;
      Check(inside, "the edge stays inside 0..1");
    }

    // Black to white: no room to sharpen, unchanged.
    {
      Image in(size_t(w * h));
      for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) in[size_t(y * w + x)] = x < 30 ? Rgb{0, 0, 0} : Rgb{1, 1, 1};
      Check(MaxDiff(gpu.Run(in, w, h, 1.0f), in) < 1e-3f, "a black-to-white edge is unchanged");
    }

    // A lone bright pixel (noise): its rise over the input under half of
    // what it would be without the noise detection.
    {
      Image noise(size_t(w * h), Rgb{0.4f, 0.4f, 0.4f});
      noise[size_t(18 * w + 30)] = {0.6f, 0.6f, 0.6f};
      const float rise = gpu.Run(noise, w, h, 1.0f)[size_t(18 * w + 30)].g - 0.6f;
      const float plain = Reference(AsHalf(noise), w, h, 1.0f, false)[size_t(18 * w + 30)].g - 0.6f;
      Check(rise > 0.0f && rise < 0.5f * plain, "a lone noisy pixel is sharpened less than without the noise detection");
    }
  }
  if (failures) return 1;
  std::printf("metal_rcas_test: ok\n");
  return 0;
}
