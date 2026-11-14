// The output curve on the Metal backend (gfx/output_curve.h): the frame's
// colours put through a curve on their way to the window. RECOMP_OUTPUT_CURVE
// as the device reads it (none, "bt709", or a power), and output_curve, run
// on the GPU, against the curves written out here: the sRGB decode followed
// by the BT.709 encode, and a power of each channel. Both keep 0 and 1 and
// darken the middle (more contrast); out of 0..1 input is clamped.
//
//   metal_output_curve_test        (the GPU part skipped, exit 77, without a Metal device)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "runtime/gfx/output_curve.h"

namespace {

using recomp::gfx::OutputCurve;
using recomp::gfx::ParseOutputCurve;

int failures = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    failures++;
  }
}

float SrgbDecode(float x) { return x < 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f); }
float Bt709Encode(float x) { return x < 0.018f ? x * 4.5f : 1.099f * std::pow(x, 0.45f) - 0.099f; }

float Reference(float x, const OutputCurve& c) {
  x = std::clamp(x, 0.0f, 1.0f);
  switch (c.kind) {
    case OutputCurve::kBt709: return Bt709Encode(SrgbDecode(x));
    case OutputCurve::kPower: return std::pow(x, c.exponent);
    default: return x;
  }
}

// output_curve on every value of `in` (each in all three channels), on the GPU.
std::vector<float> RunGpu(id<MTLDevice> dev, id<MTLComputePipelineState> pipe, id<MTLCommandQueue> q,
                          const std::vector<float>& in, const OutputCurve& c) {
  const NSUInteger w = in.size();
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                width:w
                                                                               height:1
                                                                            mipmapped:NO];
  td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  td.storageMode = MTLStorageModeShared;
  id<MTLTexture> src = [dev newTextureWithDescriptor:td], dst = [dev newTextureWithDescriptor:td];
  std::vector<float> px(w * 4);
  for (NSUInteger i = 0; i < w; i++) px[i * 4] = px[i * 4 + 1] = px[i * 4 + 2] = in[i], px[i * 4 + 3] = 1.0f;
  [src replaceRegion:MTLRegionMake2D(0, 0, w, 1) mipmapLevel:0 withBytes:px.data() bytesPerRow:w * 16];
  const float params[2] = {c.ShaderMode(), c.exponent};
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pipe];
  [e setTexture:src atIndex:0];
  [e setTexture:dst atIndex:1];
  [e setBytes:params length:sizeof params atIndex:0];
  [e dispatchThreads:MTLSizeMake(w, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [e endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  [dst getBytes:px.data() bytesPerRow:w * 16 fromRegion:MTLRegionMake2D(0, 0, w, 1) mipmapLevel:0];
  std::vector<float> out(w);
  for (NSUInteger i = 0; i < w; i++) {
    out[i] = px[i * 4];
    Check(px[i * 4 + 1] == px[i * 4] && px[i * 4 + 2] == px[i * 4], "each channel the same curve");
    Check(px[i * 4 + 3] == 1.0f, "alpha kept");
  }
  return out;
}

constexpr const char kTestKernel[] = R"(
kernel void output_curve_test_cs(texture2d<float, access::read> src [[texture(0)]],
                                 texture2d<float, access::write> out [[texture(1)]],
                                 constant float2& p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= src.get_width()) return;
  const float4 c = src.read(gid);
  out.write(float4(output_curve(c.rgb, p.x, p.y), c.a), gid);
}
)";

}  // namespace

int main() {
  @autoreleasepool {
    // The setting.
    for (const char* none : {(const char*)nullptr, "", "0", "off", "none", "srgb", "1", "1.0", "-1.2", "abc"})
      Check(ParseOutputCurve(none).kind == OutputCurve::kNone && !ParseOutputCurve(none).Active(),
            "no curve unless one is named");
    Check(ParseOutputCurve("bt709").kind == OutputCurve::kBt709 && ParseOutputCurve("bt709").Active(),
          "bt709 is the sRGB decode and BT.709 encode");
    const OutputCurve p = ParseOutputCurve("1.2");
    Check(p.kind == OutputCurve::kPower && std::fabs(p.exponent - 1.2f) < 1e-6f && p.Active(), "a number is a power");
    Check(ParseOutputCurve("0.8").kind == OutputCurve::kPower, "a power below 1 too");
    Check(OutputCurve{}.ShaderMode() == 0.0f && ParseOutputCurve("bt709").ShaderMode() == 1.0f &&
              p.ShaderMode() == 2.0f,
          "the shader's mode numbers");

    // The curves themselves (the reference): ends kept, the middle darker.
    for (const OutputCurve& c : {ParseOutputCurve("bt709"), p}) {
      Check(Reference(0.0f, c) == 0.0f, "black stays black");
      Check(std::fabs(Reference(1.0f, c) - 1.0f) < 1e-6f, "white stays white");
      Check(Reference(0.5f, c) < 0.5f, "the middle is darker");
      float last = -1.0f;
      for (int i = 0; i <= 255; i++) {
        const float v = Reference(i / 255.0f, c);
        Check(v >= last, "rises with its input");
        last = v;
      }
    }
    Check(std::fabs(Reference(0.5f, ParseOutputCurve("bt709")) - 0.4502f) < 1e-3f, "bt709 at 0.5");
    Check(std::fabs(Reference(0.5f, p) - 0.4353f) < 1e-3f, "power 1.2 at 0.5");

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      std::printf("metal_output_curve_test: no Metal device, GPU part skipped\n");
      return failures ? 1 : 77;
    }
    NSError* err = nil;
    const std::string src = std::string(recomp::gfx::kOutputCurveMsl) + kTestKernel;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(src.c_str()) options:nil error:&err];
    if (!lib) {
      std::fprintf(stderr, "FAIL: output_curve does not compile: %s\n", err.localizedDescription.UTF8String);
      return 1;
    }
    id<MTLComputePipelineState> pipe =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"output_curve_test_cs"] error:&err];
    if (!pipe) {
      std::fprintf(stderr, "FAIL: test pipeline: %s\n", err.localizedDescription.UTF8String);
      return 1;
    }
    id<MTLCommandQueue> q = [dev newCommandQueue];

    // Every 8-bit level, and a little outside 0..1 (clamped).
    std::vector<float> in;
    for (int i = 0; i <= 255; i++) in.push_back(i / 255.0f);
    in.push_back(-0.25f);
    in.push_back(1.25f);
    for (const OutputCurve& c : {OutputCurve{}, ParseOutputCurve("bt709"), p, ParseOutputCurve("0.8")}) {
      const std::vector<float> got = RunGpu(dev, pipe, q, in, c);
      float worst = 0.0f;
      for (size_t i = 0; i < in.size(); i++) worst = std::max(worst, std::fabs(got[i] - Reference(in[i], c)));
      // Well under one 8-bit step (the window's format).
      Check(worst < 0.25f / 255.0f, "the GPU's curve matches the reference");
      if (worst >= 0.25f / 255.0f) std::fprintf(stderr, "  mode %.0f: off by %g\n", c.ShaderMode(), worst);
    }
  }
  if (failures) return 1;
  std::puts("metal_output_curve_test: ok");
  return 0;
}
