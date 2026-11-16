// What a forward pixel shader's ray queries cost to compile (gfx/
// rt_forward_reflection.h): a shader with as many calls of r_rt_shadow_to as
// a game's has at most (two light sets of four lanes), with and without a
// reflection's ray, patched, its library
// and its render pipeline made, each timed. A pipeline made in a frame stalls
// that frame, so the cost is bounded here.
//
//   rt_forward_cost_test [calls]   (exit 77 without ray tracing from render
//                                   pipelines)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "runtime/gfx/rt_forward_reflection.h"
#include "runtime/gfx/rt_support.h"

namespace {

using namespace recomp::gfx;

double Ms(std::chrono::steady_clock::time_point a) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
}

// A shader of the translator's shape with `calls` rays, each toward its own
// point with its own weight; `nonce` makes the text one no cache has seen.
std::string Shape(int calls, unsigned nonce, bool reflect) {
  std::string s = R"(#include <metal_stdlib>
using namespace metal;
struct main0_out { float4 r_c0 [[color(0)]]; };
struct main0_in { float4 r_var1 [[user(locn0)]]; float4 r_var2 [[user(locn1)]]; };
struct r_float_regs { float4 c[224]; };
static inline __attribute__((always_inline))
float4 r_rt_shadow_to(thread const float4& pixel, thread const float4& world, thread const float4& to, thread const float4& weight)
{
    return float4(1.0);
}
static inline __attribute__((always_inline))
float4 r_rt_reflect(thread const float4& pixel, thread const float4& origin, thread const float4& dir, thread const float4& light, thread const float4& colour, thread const float4& ambient0, thread const float4& ambient1)
{
    return float4(0.0);
}

fragment main0_out main0(main0_in in [[stage_in]], constant r_float_regs& _29 [[buffer(0)]], float4 gl_FragCoord [[position]])
{
    main0_out out = {};
    float4 param = gl_FragCoord;
    float4 sum = float4()" +
                  std::to_string(nonce % 1000) + R"(.0 / 4096.0);
)";
  for (int i = 0; i < calls; i++) {
    const std::string n = std::to_string(i);
    s += "    { float4 to = _29.c[" + n + "] - in.r_var1; float4 w = _29.c[" + std::to_string(16 + i) +
         "] * in.r_var2; sum += r_rt_shadow_to(param, in.r_var1, to, w) * w; }\n";
  }
  if (reflect)
    s += "    sum += r_rt_reflect(param, in.r_var1, in.r_var2, _29.c[40], _29.c[41], _29.c[42], _29.c[43]);\n";
  s += "    out.r_c0 = sum;\n    return out;\n}\n";
  return s;
}

const char* const kVs = R"(#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float4 a [[user(locn0)]]; float4 b [[user(locn1)]]; };
vertex VOut quad_vs(uint vid [[vertex_id]]) {
  VOut o;
  o.pos = float4((vid & 1u) != 0u ? 1.0 : -1.0, (vid & 2u) != 0u ? -1.0 : 1.0, 0.5, 1.0);
  o.a = o.pos;
  o.b = float4(1.0);
  return o;
}
)";

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev || !RtSupportedFromRender(dev)) {
      std::printf("skipped: no ray tracing from render pipelines\n");
      return 77;
    }
    const int calls = argc > 1 ? std::atoi(argv[1]) : 8;
    const unsigned nonce = static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count());
    NSError* err = nil;
    id<MTLLibrary> vl = [dev newLibraryWithSource:@(kVs) options:nil error:&err];
    id<MTLFunction> vs = [vl newFunctionWithName:@"quad_vs"];
    double worst = 0;
    struct Case {
      int rays;
      bool reflect;
    };
    unsigned k = 0;
    for (const Case c : {Case{0, false}, Case{1, false}, Case{calls, false}, Case{0, true}, Case{calls, true}}) {
      const int n = c.rays;
      std::string msl = Shape(n, nonce + k++, c.reflect);
      const bool rays = n > 0 || c.reflect;
      if (rays && !RtForwardReflectionPatch(&msl)) {
        std::printf("FAIL the patch of %d calls\n", n);
        return 1;
      }
      MTLCompileOptions* co = [MTLCompileOptions new];
      co.preprocessorMacros = @{@"R_ALPHA_TEST" : @"false"};
      auto t = std::chrono::steady_clock::now();
      id<MTLLibrary> fl = [dev newLibraryWithSource:@(msl.c_str()) options:co error:&err];
      const double lib = Ms(t);
      if (!fl) {
        std::printf("FAIL the library of %d calls: %s\n", n, err.localizedDescription.UTF8String);
        return 1;
      }
      MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
      pd.vertexFunction = vs;
      pd.fragmentFunction = [fl newFunctionWithName:@"main0"];
      pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
      t = std::chrono::steady_clock::now();
      id<MTLRenderPipelineState> pipe = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
      const double pl = Ms(t);
      if (!pipe) {
        std::printf("FAIL the pipeline of %d calls: %s\n", n, err.localizedDescription.UTF8String);
        return 1;
      }
      std::printf("%d light rays%s: library %.0f ms, pipeline %.0f ms\n", n, c.reflect ? " and a reflection" : "", lib, pl);
      if (rays && pl > worst) worst = pl;
    }
    const double limit = std::getenv("RT_FORWARD_COST_LIMIT_MS") ? std::atof(std::getenv("RT_FORWARD_COST_LIMIT_MS")) : 0;
    if (limit > 0 && worst > limit) {
      std::printf("FAIL a pipeline took %.0f ms (limit %.0f)\n", worst, limit);
      return 1;
    }
  }
  std::puts("rt_forward_cost: ok");
  return 0;
}
