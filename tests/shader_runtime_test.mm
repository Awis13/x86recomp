#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <barrier>
#include <thread>
#include <pthread.h>
#include <sys/resource.h>
#include "runtime/gfx/metal_cache.h"
#include "metal_test_profile.h"
#include "runtime/gfx/metal_internal.h"

using namespace recomp::gfx;
static void Require(bool ok, const char* message) {
  if (!ok) { std::fprintf(stderr, "shader runtime test: %s\n", message); std::exit(1); }
}
__attribute__((noinline)) static void ReuseStackAndDrain() {
  volatile unsigned char stack[16384];
  for (size_t i = 0; i < sizeof stack; i++) stack[i] = 0xA5;
  MetalDrainPipelineWrites();
  Require(stack[0] == 0xA5, "stack lifetime stress setup failed");
}
int main(int argc, char** argv) {
  Require(argc == 4, "expected game, cache, and expected source");
  const bool descriptor_budget = std::strcmp(argv[3], "descriptor-budget") == 0;
  if (descriptor_budget) {
    rlimit limit{};
    Require(getrlimit(RLIMIT_NOFILE, &limit) == 0 && limit.rlim_max >= 64,
            "cannot inspect descriptor limit");
    limit.rlim_cur = 64;
    Require(setrlimit(RLIMIT_NOFILE, &limit) == 0, "cannot set descriptor budget");
  }
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    Require(dev != nil, "Metal GPU unavailable");
    ConfigureMetalPreparation(argv[1], argv[2], MetalTestProfile(argv[1]));
    Require(!MetalPreparationDirectory().empty(), "cache configuration failed");
    if (std::strcmp(argv[3], "cancel-preload") == 0) {
      std::jthread background(MetalPreloadTask(dev));
      background.request_stop();
      background.join();
      return 0;
    }
    const bool snapshot = std::strcmp(argv[3], "catalog-snapshot") == 0;
    if (snapshot) {
      std::function<void(std::stop_token)> preload;
      @autoreleasepool { preload = MetalPreloadTask(dev); }
      // Loading-screen draws can publish observations after selection. They
      // must not shrink a first-play task to an observed-only catalog.
      NSString* directory = @(MetalPreparationDirectory().c_str());
      NSData* bytes = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"prepared-descriptors.json"]];
      NSDictionary* records = [NSJSONSerialization JSONObjectWithData:bytes options:0 error:nil];
      NSString* observations = [directory stringByAppendingPathComponent:@"descriptors"];
      [NSFileManager.defaultManager createDirectoryAtPath:observations withIntermediateDirectories:YES attributes:nil error:nil];
      bool recorded = false;
      for (NSString* key in records) if (![records[key][@"core"] boolValue]) {
        NSData* record = [NSJSONSerialization dataWithJSONObject:records[key] options:0 error:nil];
        Require([record writeToFile:[observations stringByAppendingPathComponent:[key stringByAppendingString:@".json"]] atomically:YES],
                "could not record loading-screen observation");
        recorded = true; break;
      }
      Require(recorded, "snapshot fixture needs a non-core descriptor");
      std::jthread background(std::move(preload));
      background.join();
    } else MetalPreloadPipelines(dev);
    if (std::strcmp(argv[3], "preload-only") == 0) return 0;
    if (std::strcmp(argv[3], "depth-copy") == 0) {
      auto lib = MetalCachedLibrary(dev, @(kInternalMsl), nil, nullptr);
      for (auto format : {MTLPixelFormatDepth16Unorm, MTLPixelFormatDepth32Float,
                          MTLPixelFormatDepth32Float_Stencil8}) {
        auto p = MetalDepthCopyDescriptor(format, MetalCachedFunction(lib, @"tri_vs"),
                                         MetalCachedFunction(lib, @"depth_copy_fs"));
        Require(p.depthAttachmentPixelFormat == format && p.stencilAttachmentPixelFormat ==
                    (format == MTLPixelFormatDepth32Float_Stencil8 ? format : MTLPixelFormatInvalid),
                "depth-copy descriptor must match the destination attachments");
        auto state = MetalCachedPipeline(dev, p, nullptr, true);
        Require(state && MetalPipelineOrigin(state) == MetalPipelineSource::Preloaded,
                "runtime depth-copy descriptor was not prepared");
      }
      return 0;
    }
    if (descriptor_budget) {
      // A retained URL-backed library can consume a descriptor for its lifetime.
      // First-play preload must leave room for the game's next resource open.
      const std::string path = std::string(argv[1]) + "/shaders.txt";
      FILE* resource = std::fopen(path.c_str(), "rb");
      Require(resource != nullptr, "prepared preload exhausted game resource descriptors");
      char magic[2]{};
      const size_t read = std::fread(magic, 1, sizeof magic, resource);
      std::fclose(resource);
      Require(read == sizeof magic && std::memcmp(magic, "64", sizeof magic) == 0,
              "game shader resource could not be read after preload");
      return 0;
    }
    if (std::strcmp(argv[3], "catalog") == 0 || snapshot) {
      // Consume every expected state after the startup barrier. Archive-backed
      // lazy creation here would fail the origin check even if it renders.
      NSString* directory = @(MetalPreparationDirectory().c_str());
      NSData* bytes = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"prepared-descriptors.json"]];
      NSDictionary* records = [NSJSONSerialization JSONObjectWithData:bytes options:0 error:nil];
      Require(records.count > 100, "catalog stress fixture is too small");
      NSMutableDictionary* libraries = [NSMutableDictionary new];
      auto function = [&](NSString* identity) {
        NSArray* parts = [identity componentsSeparatedByString:@"/"];
        id<MTLLibrary> library = libraries[parts[0]];
        if (!library) {
          NSString* path = [directory stringByAppendingPathComponent:[NSString stringWithFormat:@"libraries/%@.metallib", parts[0]]];
          library = [dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:nil];
          Require(library != nil, "catalog library missing");
          library.label = parts[0]; libraries[parts[0]] = library;
        }
        return MetalCachedFunction(library, parts[1]);
      };
      for (NSString* key in records) { @autoreleasepool {
        NSDictionary* record = records[key];
        auto p = [MTLRenderPipelineDescriptor new];
        p.vertexFunction = function(record[@"vertex"]); p.fragmentFunction = function(record[@"fragment"]);
        NSArray* attachments = record[@"attachments"];
        for (int i = 0; i < 4; i++) {
          NSArray* values = attachments[i]; auto a = p.colorAttachments[i];
          a.pixelFormat = MTLPixelFormat([values[0] unsignedIntegerValue]); a.blendingEnabled = [values[1] boolValue];
          a.sourceRGBBlendFactor = MTLBlendFactor([values[2] unsignedIntegerValue]);
          a.destinationRGBBlendFactor = MTLBlendFactor([values[3] unsignedIntegerValue]);
          a.rgbBlendOperation = MTLBlendOperation([values[4] unsignedIntegerValue]);
          a.sourceAlphaBlendFactor = MTLBlendFactor([values[5] unsignedIntegerValue]);
          a.destinationAlphaBlendFactor = MTLBlendFactor([values[6] unsignedIntegerValue]);
          a.alphaBlendOperation = MTLBlendOperation([values[7] unsignedIntegerValue]);
          a.writeMask = MTLColorWriteMask([values[8] unsignedIntegerValue]);
        }
        p.depthAttachmentPixelFormat = MTLPixelFormat([record[@"depth"] unsignedIntegerValue]);
        p.stencilAttachmentPixelFormat = MTLPixelFormat([record[@"stencil"] unsignedIntegerValue]);
        p.rasterSampleCount = [record[@"samples"] unsignedIntegerValue];
        auto state = MetalCachedPipeline(dev, p, nullptr, true, "validation");
        Require(state && MetalPipelineOrigin(state) == MetalPipelineSource::Preloaded,
                "catalog state was deferred past preload completion");
        Require(MetalCachedPipeline(dev, p, nullptr, true, "validation") == state,
                "catalog state identity changed after preload");
      } }
      return 0;
    }
    if (std::strncmp(argv[3], "firstplay", 9) == 0) {
      // The worker knows renderer pipelines without any gameplay recordings.
      NSError* error = nil;
      auto lib = MetalCachedLibrary(dev, @(kInternalMsl), nil, &error);
      auto p = [MTLRenderPipelineDescriptor new];
      p.vertexFunction = MetalCachedFunction(lib, @"quad_vs");
      p.fragmentFunction = MetalCachedFunction(lib, @"quad_fs");
      p.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
      auto state = MetalCachedPipeline(dev, p, &error, true);
      auto expected = std::strcmp(argv[3], "firstplay-archive") == 0 ? MetalPipelineSource::Archive : MetalPipelineSource::Preloaded;
      Require(state && MetalPipelineOrigin(state) == expected,
              "first-play pipeline was not ready before drawing");
      if (std::strcmp(argv[3], "firstplay-record") == 0) {
        p.colorAttachments[0].writeMask = MTLColorWriteMaskRed;
        Require(MetalCachedPipeline(dev, p, &error) != nil, "new first-play state failed");
        MetalDrainPipelineWrites();
      }
      return 0;
    }
    NSString* source = @R"MSL(
#include <metal_stdlib>
using namespace metal;
vertex float4 test_vs(uint i [[vertex_id]]) {
  const float2 positions[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
  return float4(positions[i],0,1);
}
fragment float4 test_fs() { return float4(TEST_RED,0,0,1); }
kernel void test_cs(device uint* out [[buffer(0)]]) { out[0] = TEST_VALUE; }
)MSL";
    NSError* error = nil;
    auto options = [MTLCompileOptions new];
    options.preprocessorMacros = @{@"TEST_RED": @"1", @"TEST_VALUE": @"42"};
    id<MTLLibrary> lib = MetalCachedLibrary(dev, source, options, &error);
    Require(lib != nil, error.localizedDescription.UTF8String ?: "library failed");
    if (std::strcmp(argv[3], "cancel-pending") == 0 || std::strcmp(argv[3], "queue-order") == 0 ||
        std::strcmp(argv[3], "stress-exit") == 0 || std::strcmp(argv[3], "metadata-order") == 0) {
      const int variants = std::strcmp(argv[3], "cancel-pending") == 0 ? 64 : 8;
      for (int i = 0; i < variants; i++)
        Require(MetalCachedLibrary(dev, [source stringByAppendingFormat:@"\n// variant %d", i], options, &error) != nil,
                "cancellation setup failed");
    }
    auto fragmentLibrary = lib;
    if (std::strncmp(argv[3], "split-", 6) == 0 || std::strcmp(argv[3], "metadata-order") == 0)
      fragmentLibrary = MetalCachedLibrary(dev, [source stringByAppendingString:@"\n// fragment library"], options, &error);
    Require(fragmentLibrary != nil, "fragment library failed");
    auto descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = MetalCachedFunction(lib, @"test_vs");
    descriptor.fragmentFunction = MetalCachedFunction(fragmentLibrary, @"test_fs");
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    if (std::strcmp(argv[3], "unavailable") == 0) {
      Require(!MetalCachedPipeline(dev, descriptor, &error, true), "archive-only lookup compiled a missing pipeline");
      MetalDrainPipelineWrites();
      return 0;
    }
    if (std::strcmp(argv[3], "requested") == 0) {
      MetalRequestPipeline(dev, descriptor);
      MetalRequestPipeline(dev, descriptor);
      std::array<id<MTLRenderPipelineState>, 4> states{};
      std::array<std::thread, 4> callers;
      std::barrier start(4);
      for (size_t i = 0; i < callers.size(); i++) {
        MTLRenderPipelineDescriptor* copy = [descriptor copy];
        callers[i] = std::thread([dev, copy, &states, &start, i] {
          @autoreleasepool {
            start.arrive_and_wait();
            states[i] = MetalCachedPipeline(dev, copy, nullptr);
          }
        });
      }
      for (auto& caller : callers) caller.join();
      for (auto state : states) Require(state && state == states[0], "request waiters did not share a pipeline");
    }
    id<MTLRenderPipelineState> state = MetalCachedPipeline(dev, descriptor, &error);
    Require(state != nil, error.localizedDescription.UTF8String ?: "pipeline failed");
    auto expected = (std::strcmp(argv[3], "preloaded") == 0 || std::strcmp(argv[3], "split-preloaded") == 0) ? MetalPipelineSource::Preloaded
                  : (std::strcmp(argv[3], "archive") == 0 || std::strcmp(argv[3], "split-archive") == 0)
                    ? MetalPipelineSource::Archive : MetalPipelineSource::Runtime;
    Require(std::strcmp(argv[3], "resume") == 0 || MetalPipelineOrigin(state) == expected, "incorrect pipeline source");
    Require(MetalCachedPipeline(dev, descriptor, &error) == state, "ready pipeline was recreated");
    Require(MetalCachedLibrary(dev, source, options, &error) == lib, "loaded library was recreated");
    const bool process_exit = std::strcmp(argv[3], "stress-exit") == 0;
    if (std::strcmp(argv[3], "stress") == 0 || process_exit) {
      // Reuse caller stack and heap while queued writes finish. Async writes
      // must own their temporary cache keys.
      for (unsigned variant = 0; variant < 128; variant++) {
        MTLRenderPipelineDescriptor* varied = [descriptor copy];
        auto color = varied.colorAttachments[0];
        color.writeMask = MTLColorWriteMask(variant & 15);
        color.blendingEnabled = (variant & 16) != 0;
        color.destinationRGBBlendFactor = (variant & 32) ? MTLBlendFactorOne : MTLBlendFactorZero;
        color.sourceAlphaBlendFactor = (variant & 64) ? MTLBlendFactorZero : MTLBlendFactorOne;
        if (process_exit) MetalRequestPipeline(dev, varied);
        else Require(MetalCachedPipeline(dev, varied, &error) != nil, "queued pipeline failed");
        if (!process_exit) ReuseStackAndDrain();
      }
      std::vector<std::string> churn(4096, std::string(71, 'z'));
      if (!process_exit) MetalDrainPipelineWrites();
      Require(churn.back().size() == 71, "key lifetime stress setup failed");
    }
    auto textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
        width:2 height:2 mipmapped:NO];
    textureDescriptor.storageMode = MTLStorageModeShared;
    textureDescriptor.usage = MTLTextureUsageRenderTarget;
    auto texture = [dev newTextureWithDescriptor:textureDescriptor];
    auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    auto queue = [dev newCommandQueue];
    auto command = [queue commandBuffer];
    auto encoder = [command renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:state];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    auto compute = MetalCachedComputePipeline(dev, MetalCachedFunction(fragmentLibrary, @"test_cs"), &error);
    Require(compute != nil, "compute pipeline failed");
    auto buffer = [dev newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    auto computeEncoder = [command computeCommandEncoder];
    [computeEncoder setComputePipelineState:compute];
    [computeEncoder setBuffer:buffer offset:0 atIndex:0];
    [computeEncoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [computeEncoder endEncoding];
    [command commit]; [command waitUntilCompleted];
    Require(command.status == MTLCommandBufferStatusCompleted, "GPU command failed");
    uint8_t pixels[16]{};
    [texture getBytes:pixels bytesPerRow:8 fromRegion:MTLRegionMake2D(0,0,2,2) mipmapLevel:0];
    for (size_t i = 0; i < sizeof pixels; i += 4)
      Require(pixels[i] == 0 && pixels[i+1] == 0 && pixels[i+2] == 255 && pixels[i+3] == 255, "wrong rendered pixels");
    Require(*static_cast<uint32_t*>(buffer.contents) == 42, "wrong compute output");
    if (!process_exit) {
      if (std::strcmp(argv[3], "cancel-pending") == 0) {
        // A guest exit need not arrive on a high-priority thread. The drain
        // must give its required batch and later continuations a minimum QoS.
        std::thread shutdown([] {
          @autoreleasepool {
            Require(pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0) == 0,
                    "cannot set shutdown caller priority");
            MetalDrainPipelineWrites(true);
          }
        });
        shutdown.join();
      } else MetalDrainPipelineWrites();
    }
    std::puts("shader runtime test: rendered and computed successfully");
  }
}
