#pragma once
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "runtime/gfx/metal_recipe.h"
#include "runtime/hlsl/hlsl.h"

namespace recomp::gfx {
struct MetalPreparationShader {
  std::string name;
  hlsl::Program program;
};
struct MetalPreparationInputs {
  std::vector<MetalPreparationShader> shaders;
  std::vector<MetalRecipe> recipes;
  std::vector<std::pair<std::string, std::string>> errors;
};
// The caller owns content discovery and render-state policy. Callbacks run
// synchronously before compilation; they may throw to report an invalid input.
// identity must change with discovery or recipe policy, even across source builds.
struct MetalPreparationProfile {
  std::string cache_namespace = "x86recomp";
  std::string identity;
  std::function<std::string(const std::string&)> content_identity;
  std::function<MetalPreparationInputs(const std::string&)> discover;
};
// Use the renderer's exact shader identity when constructing pipeline recipes.
uint64_t MetalShaderIdentity(const hlsl::Program&);
// macOS only. Returns -1 unless --shader-preparation is the first argument.
// Commands prepare/status/clean emit one JSON object per line on stdout.
int MetalPreparationCommand(int argc, char** argv, const MetalPreparationProfile&);
// Call before creating the renderer. Failures leave runtime compilation available.
void ConfigureMetalPreparation(const std::string& content_dir, const std::string& cache_root,
                               const MetalPreparationProfile&);
}  // namespace recomp::gfx
