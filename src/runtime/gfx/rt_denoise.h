#pragma once
// MetalFX's denoiser (MTLFXTemporalDenoisedScaler, Objective-C++) over a
// deferred renderer's lit scene, at the scene's own size (1:1), before the
// game's post-processing, while rays are traced (gfx::Device::RayTracing::
// denoise). Where: the colour target the lighting pass (gfx::RayDenoise,
// not post_input) draws into is the lit scene; once a frame, before the
// first draw of a post-process shader the game names that reads that
// texture (or a whole copy of it), the backend denoises it. Its output
// takes the texture's place (the two trade places every frame: no copy;
// MetalFX keeps its history itself).
//
// What MetalFX is given:
// - colour: the lit scene as it is then (lighting, and what was drawn over
//   it since);
// - depth and motion: the scene's depth (a Depth32Float copy) and the
//   camera's motion (previous minus current, in texture coordinates, the
//   jitter not in it: the camera's matrices are taken without it), as the
//   backend's MetalFX temporal scaler has them;
// - the guides, made by this file's kernel from the G-buffer when the
//   lighting draws (the G-buffer may be reused later in the frame):
//   diffuse albedo; the world-space normal decoded to a signed format; the
//   roughness of the Blinn-Phong exponent, sqrt(2 / (e + 2)) (as
//   rt_reflection.h makes it: Apple asks for "linear roughness" and does
//   not say whether that is this alpha or its square root); the specular
//   albedo with the game's Fresnel term (RayDenoise); the reflections' hit
//   distance where they were traced (gfx/rt_reflection.h: where none was,
//   or a ray missed, kRtReflectionSkyDistance: the game's own environment
//   map, far away);
// - a reactive mask: 1 where the depth changed between the lighting and the
//   denoise (people, vehicles, anything drawn forward that writes depth:
//   their pixels have the G-buffer of what is behind them and no motion of
//   their own, so MetalFX is told to keep no history there), else 0;
// - the camera split into world to view and view to clip (RtSplitCamera),
//   the jitter (the scene's shift, pixels right and down), the depth's
//   direction (reversed where stored so; the projection given reversed
//   alike), auto exposure, and a history reset on cuts, resizes, the first
//   frame and after a frame not denoised.
// Not used: the denoise strength mask. Its header says 1 marks a pixel the
// denoiser ignores; Apple's WWDC26 talk says 0 is no denoising and 1 the
// most. Until one is checked on a GPU, it stays off.
//
// Limits: macOS 26 / iOS 26 on GPUs MetalFX's denoiser supports (Apple9 and
// later: RtDenoiseSupported); pixels drawn after the lighting are kept out
// of the history by the reactive mask, not denoised; blended draws that do
// not write depth (particles, glass) are not masked; motion is the camera's
// only (no per-object motion); wet surfaces' darker albedo in the lighting
// is not in the guide's.

#if !__has_feature(objc_arc)
#error "rt_denoise.h needs ARC (-fobjc-arc)"
#endif

#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace recomp::gfx {

// A perspective world-to-clip matrix `vp` (D3D's row vectors: clip = (x, 1)
// vp, depth 0..1, w the view depth) split into a rigid world-to-view one
// (its axes the camera's: right, up and forward, unit and orthogonal; the
// camera's position) and the view-to-clip one, view * proj == vp. The
// view's handedness is whatever vp's is (a left-handed view of a
// right-handed world has a determinant of -1). False if vp is not a
// perspective (no w from the view depth) or is degenerate.
bool RtSplitCamera(const double vp[16], double view[16], double proj[16]);

// The guides' per-pixel conversions (the kernel's, on the CPU for tests).
inline float RtDenoiseRoughness(float gloss, float exponent_scale) {
  const float e = exponent_scale * gloss * gloss;
  return std::sqrt(2.0f / (e + 2.0f));
}
inline float RtDenoiseSpecular(float spec, float cos_view, float spec_scale, float fresnel_min, float fresnel_power) {
  const float c = std::fmin(std::fabs(cos_view), 1.0f);
  const float f = fresnel_min + (1.0f - fresnel_min) * std::pow(1.0f - c, fresnel_power);
  return std::fmin(std::fmax(spec_scale * spec * f, 0.0f), 1.0f);
}

// What the guides' kernel is given (MSL: RtDenoiseGuideParams).
struct RtDenoiseGuideParams {
  float inv_view_proj[16];  // clip to world (row vectors), the camera without the jitter
  uint32_t size[2];         // the lit scene's
  float jitter[2];          // the scene's shift, pixels right and down
  float exponent_scale, spec_scale, fresnel_min, fresnel_power;
  uint32_t spec_channel, gloss_channel, pad[2];
};
static_assert(sizeof(RtDenoiseGuideParams) == 112, "kRtDenoiseMsl's RtDenoiseGuideParams");

// What a denoise is given besides its textures.
struct RtDenoiseFrame {
  float jitter[2] = {};        // the scene's shift, pixels right and down (MetalFX's jitter offset)
  float motion_scale[2] = {};  // the motion texture's values times these: pixels
  // World to view and view to clip, row vectors (RtSplitCamera; the
  // projection's depth reversed where the depth is stored so).
  double view[16] = {}, proj[16] = {};
  bool depth_reversed = false;
  bool reset = false;  // the history starts over
};

// Whether MetalFX's denoiser runs on `dev` at 1:1: macOS 26 / iOS 26, a GPU
// it supports, and 1 within the input scales it takes (both reported
// where asked). False with why (where asked) otherwise.
bool RtDenoiseSupported(id<MTLDevice> dev, std::string* why = nullptr, float* min_scale = nullptr,
                        float* max_scale = nullptr);

class RtDenoiser {
 public:
  // What the work for one size uses: the scaler and its textures (private),
  // unretained (the denoiser holds them; Ensure hands back those it lets
  // go, for work recorded with them that has not run yet). Taken on the
  // recording side, so that work encoded later on another thread uses the
  // set it was recorded with.
  struct Set {
    __unsafe_unretained id scaler = nil;  // id<MTLFXTemporalDenoisedScaler> (macOS 26 / iOS 26)
    // MetalFX's motion (RG16Float) and depth (Depth32Float): render
    // targets the backend draws.
    __unsafe_unretained id<MTLTexture> motion = nil, depth = nil;
    // The guides and masks (the kernels'), and the depth at the guides.
    __unsafe_unretained id<MTLTexture> albedo = nil, normal = nil, roughness = nil, specular = nil, hit = nil,
                                       reactive = nil, depth_lit = nil;
    uint32_t w = 0, h = 0;
    bool hit_on = false;
  };

  // Null (why said in *why) where RtDenoiseSupported is false or the
  // kernels do not compile.
  static std::unique_ptr<RtDenoiser> Make(id<MTLDevice> dev, std::string* why);
  // The scaler and its textures for a w x h colour of format `color`, with
  // the hit distance input or not (remade, history and all, when any of
  // these changes: the objects let go added to `retired` where given).
  // False if MetalFX cannot make it (said on stderr).
  bool Ensure(uint32_t w, uint32_t h, MTLPixelFormat color, bool hit_distance, std::vector<id>* retired = nullptr);
  const Set& Current() const { return set_; }
  MTLPixelFormat ColorFormat() const { return color_; }
  // The usage MetalFX asks of the colour it reads (after Ensure).
  MTLTextureUsage ColorUsage() const { return color_usage_; }
  // An output texture (private, the colour's format and size) with usage
  // `extra` besides what MetalFX asks: one to trade places with the lit
  // scene's (which has `extra`). Nil before Ensure.
  id<MTLTexture> NewOutput(MTLTextureUsage extra) const;
  // The usage NewOutput gives besides `extra`.
  MTLTextureUsage OutputUsage() const { return output_usage_ | MTLTextureUsageShaderRead; }

  // The guides (one compute encoder) from the G-buffer's `albedo`,
  // `normal` and `surface` (2D, ShaderRead, any size: read at each pixel's
  // place scaled to theirs), and the scene's depth `depth` (a depth texture
  // of the set's size, ShaderRead) kept for the reactive mask.
  void EncodeGuides(id<MTLCommandBuffer> cmd, const Set& s, id<MTLTexture> albedo, id<MTLTexture> normal,
                    id<MTLTexture> surface, id<MTLTexture> depth, const RtDenoiseGuideParams& p) const;
  // At the denoise (one compute encoder): the reactive mask, 1 where
  // `depth` (as EncodeGuides took it) differs from the depth it kept, and
  // the hit distance from `hit` (R16Float, 0 where none was traced; nil:
  // none traced this frame), the reactive pixels added to the 32-bit count
  // at `counts` + `offset`.
  void EncodeMasks(id<MTLCommandBuffer> cmd, const Set& s, id<MTLTexture> depth, id<MTLTexture> hit,
                   id<MTLBuffer> counts, size_t offset) const;
  // The denoise of `color` (Ensure's format and size, ColorUsage) into
  // `out` (NewOutput's), with the set's motion, depth, guides and masks.
  static void EncodeDenoise(id<MTLCommandBuffer> cmd, const Set& s, id<MTLTexture> color, id<MTLTexture> out,
                            const RtDenoiseFrame& f);

 private:
  id<MTLDevice> dev_ = nil;
  id<MTLComputePipelineState> guide_pipe_ = nil, mask_pipe_ = nil;
  // The set's owners.
  id scaler_ = nil;
  id<MTLTexture> motion_ = nil, depth_ = nil, albedo_ = nil, normal_ = nil, roughness_ = nil, specular_ = nil,
                 hit_ = nil, reactive_ = nil, depth_lit_ = nil;
  Set set_;
  MTLPixelFormat color_ = MTLPixelFormatInvalid;
  MTLTextureUsage color_usage_ = 0, output_usage_ = 0;
};

}  // namespace recomp::gfx
