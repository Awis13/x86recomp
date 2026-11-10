#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "runtime/gfx/metal_recipe.h"
#include <stop_token>
#include <functional>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace recomp::gfx {
MTLRenderPipelineDescriptor* MetalDescribe(const MetalPipeKey&, id<MTLFunction>, id<MTLFunction>);
// Load a prepared library, otherwise compile through Metal's normal runtime path.
// Compile policy is fixed to Metal 2.1, fast math and default options, with the
// caller's preprocessor macros. Additional options require a key/policy revision.
id<MTLLibrary> MetalCachedLibrary(id<MTLDevice>, NSString*, MTLCompileOptions*, NSError**);
id<MTLRenderPipelineState> MetalCachedPipeline(id<MTLDevice>, MTLRenderPipelineDescriptor*, NSError**, bool archive_only = false, const char* phase = "runtime");
// Request only a descriptor used by a recorded draw. Two compilation lanes
// overlap first-use work; normal lookup joins its own request if still pending.
// The request owns the descriptor/functions and uses the same cache/persistence.
void MetalRequestPipeline(id<MTLDevice>, MTLRenderPipelineDescriptor*);
id<MTLFunction> MetalCachedFunction(id<MTLLibrary>, NSString*);
std::string MetalPreparedMsl(const std::string& glsl);
id<MTLComputePipelineState> MetalCachedComputePipeline(id<MTLDevice>, id<MTLFunction>, NSError**);
std::string MetalPreparationDirectory();
std::vector<MetalRecipe> MetalPreparedRecipes();
// Freeze baseline/observed selection now; run existing-binary restoration later.
// The task accepts cancellation. No missing entry is compiled.
std::function<void(std::stop_token)> MetalPreloadTask(id<MTLDevice>);
// Blocking variant for callers requiring all states ready before proceeding.
void MetalPreloadPipelines(id<MTLDevice>);
// Runtime library/archive writes are best effort, serialized away from drawing.
// Shutdown cancels offline compiler work; already queued archive writes still drain.
void MetalDrainPipelineWrites(bool stop_libraries = false);
enum class MetalPipelineSource { Runtime, Archive, Preloaded };
MetalPipelineSource MetalPipelineOrigin(id<MTLRenderPipelineState>);
}  // namespace recomp::gfx
