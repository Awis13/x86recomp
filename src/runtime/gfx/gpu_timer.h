#pragma once
// GPU time of the compute and acceleration-structure work a backend's
// helpers encode (ray-traced masks, reflections, BLAS and TLAS builds), in
// the same report as its render passes (RECOMP_GPU_PASSES): the backend
// sets the encoding thread's timer, and the helpers make their encoders
// through TimedCompute and TimedAccelerationStructure, which give each one
// a sample at its start and end under a name. Header-only (Objective-C++,
// Metal).

#import <Metal/Metal.h>

namespace recomp::gfx {

struct GpuTimer {
  void* ctx = nullptr;
  // The next pair of samples for a pass named `name`: its buffer and the
  // start and end indices; false where none (not timing, or full).
  bool (*next)(void* ctx, const char* name, __autoreleasing id<MTLCounterSampleBuffer>* buf, NSUInteger* start,
               NSUInteger* end) = nullptr;
};

// The encoding thread's timer (none: helpers make plain encoders).
inline GpuTimer& CurrentGpuTimer() {
  static thread_local GpuTimer timer;
  return timer;
}

// `type`: MTLDispatchTypeConcurrent lets its dispatches overlap (the
// caller puts in the barriers between those that depend on each other).
inline id<MTLComputeCommandEncoder> TimedCompute(id<MTLCommandBuffer> cmd, const char* name,
                                                 MTLDispatchType type = MTLDispatchTypeSerial) {
  GpuTimer& t = CurrentGpuTimer();
  id<MTLCounterSampleBuffer> buf = nil;
  NSUInteger start = 0, end = 0;
  if (t.next && t.next(t.ctx, name, &buf, &start, &end)) {
    MTLComputePassDescriptor* pd = [MTLComputePassDescriptor computePassDescriptor];
    pd.dispatchType = type;
    pd.sampleBufferAttachments[0].sampleBuffer = buf;
    pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = start;
    pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = end;
    return [cmd computeCommandEncoderWithDescriptor:pd];
  }
  return [cmd computeCommandEncoderWithDispatchType:type];
}

// A render pass sampled under `name` (its vertex start and fragment end);
// a descriptor not timed has its samples cleared, as one may be reused.
inline id<MTLRenderCommandEncoder> TimedRender(id<MTLCommandBuffer> cmd, MTLRenderPassDescriptor* rp,
                                               const char* name) {
  GpuTimer& t = CurrentGpuTimer();
  id<MTLCounterSampleBuffer> buf = nil;
  NSUInteger start = 0, end = 0;
  MTLRenderPassSampleBufferAttachmentDescriptor* a = rp.sampleBufferAttachments[0];
  if (t.next && t.next(t.ctx, name, &buf, &start, &end)) {
    a.sampleBuffer = buf;
    a.startOfVertexSampleIndex = start;
    a.endOfVertexSampleIndex = MTLCounterDontSample;
    a.startOfFragmentSampleIndex = MTLCounterDontSample;
    a.endOfFragmentSampleIndex = end;
  } else {
    a.sampleBuffer = nil;
  }
  return [cmd renderCommandEncoderWithDescriptor:rp];
}

inline id<MTLBlitCommandEncoder> TimedBlit(id<MTLCommandBuffer> cmd, const char* name) {
  GpuTimer& t = CurrentGpuTimer();
  id<MTLCounterSampleBuffer> buf = nil;
  NSUInteger start = 0, end = 0;
  if (t.next && t.next(t.ctx, name, &buf, &start, &end)) {
    MTLBlitPassDescriptor* pd = [MTLBlitPassDescriptor blitPassDescriptor];
    pd.sampleBufferAttachments[0].sampleBuffer = buf;
    pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = start;
    pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = end;
    return [cmd blitCommandEncoderWithDescriptor:pd];
  }
  return [cmd blitCommandEncoder];
}

// Work that takes no sample of its own (MetalFX's encodeToCommandBuffer):
// a timed one-texel copy out of its output right after it, which waits for
// it, so the work is charged to `name` (each pass's time runs from the end
// of the one before). `scratch` takes the texel (256 bytes or more). Only
// when timing; colour formats of up to 16 bytes a texel.
inline void TimedMarkerAfter(id<MTLCommandBuffer> cmd, const char* name, id<MTLTexture> written,
                             id<MTLBuffer> scratch) {
  if (!CurrentGpuTimer().next || !written || !scratch) return;
  id<MTLBlitCommandEncoder> bl = TimedBlit(cmd, name);
  [bl copyFromTexture:written
                   sourceSlice:0
                   sourceLevel:0
                  sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:MTLSizeMake(1, 1, 1)
                      toBuffer:scratch
             destinationOffset:0
        destinationBytesPerRow:256
      destinationBytesPerImage:256];
  [bl endEncoding];
}

inline id<MTLAccelerationStructureCommandEncoder> TimedAccelerationStructure(id<MTLCommandBuffer> cmd,
                                                                              const char* name) {
  GpuTimer& t = CurrentGpuTimer();
  id<MTLCounterSampleBuffer> buf = nil;
  NSUInteger start = 0, end = 0;
  if (t.next && t.next(t.ctx, name, &buf, &start, &end)) {
    MTLAccelerationStructurePassDescriptor* pd = [MTLAccelerationStructurePassDescriptor accelerationStructurePassDescriptor];
    pd.sampleBufferAttachments[0].sampleBuffer = buf;
    pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = start;
    pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = end;
    return [cmd accelerationStructureCommandEncoderWithDescriptor:pd];
  }
  return [cmd accelerationStructureCommandEncoder];
}

}  // namespace recomp::gfx
