// A new render target on the Metal backend starts as zeros.
//
// Metal leaves a new texture's contents undefined. Direct3D 9 does not
// promise zeros either, but its drivers give them, other Direct3D 9
// implementations clear every new image, and games rely on it: a target
// drawn into with blending before its colour is ever cleared (a pass that
// clears only depth) keeps whatever it started with under every blended
// pixel. Not-a-number halves in a float target stay not-a-number for good.
#pragma once

#import <Metal/Metal.h>

namespace recomp::gfx {

// Every level of `tex` (a 2D colour render target) cleared to zeros, in
// passes of their own on `cmd`. No encoder of `cmd` may be open.
inline void ClearNewTarget(id<MTLCommandBuffer> cmd, id<MTLTexture> tex) {
  for (NSUInteger level = 0; level < tex.mipmapLevelCount; level++) {
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = tex;
    rp.colorAttachments[0].level = level;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    [[cmd renderCommandEncoderWithDescriptor:rp] endEncoding];
  }
}

}  // namespace recomp::gfx
