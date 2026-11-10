#import <Metal/Metal.h>
#include <bit>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include "metal_test_profile.h"

using namespace recomp;
using namespace recomp::gfx;
namespace {
std::string Read(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("Missing test input: " + path.string());
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
hlsl::Program Translate(const std::vector<uint32_t>& tokens) {
  hlsl::Program program; std::string error;
  if (!hlsl::TranslateBytecode(tokens.data(), tokens.size(), &program, &error))
    throw std::runtime_error(error);
  return program;
}
}  // namespace
MetalPreparationProfile MetalTestProfile(const std::string& directory) {
  MetalPreparationProfile profile;
  profile.identity = "synthetic-programs-v1";
  // A caller's policy can change without changing any shader input bytes.
  // This fixture reads it separately to exercise profile cache isolation.

  if (std::filesystem::exists(std::filesystem::path(directory) / "policy.txt"))
    profile.identity += Read(std::filesystem::path(directory) / "policy.txt");
  profile.content_identity = [](const std::string& directory) { return Read(std::filesystem::path(directory) / "shaders.txt"); };
  profile.discover = [](const std::string& directory) {
    const int count = std::stoi(Read(std::filesystem::path(directory) / "shaders.txt"));
    if (count < 1 || count > 64) throw std::runtime_error("Invalid test shader count");
    MetalPreparationInputs inputs;
    auto vertex = Translate({0xFFFE0300, 0x0200001F, 0x80000000, 0x900F0000,
                            0x0200001F, 0x80000000, 0xE00F0000,
                            0x02000001, 0xE00F0000, 0x90E40000, 0xFFFF});
    const auto vs = MetalShaderIdentity(vertex);
    inputs.shaders.push_back({"test vertex", std::move(vertex)});
    for (int i = 0; i < count; i++) {
      const auto value = std::bit_cast<uint32_t>(float(i + 1) / 32);
      auto pixel = Translate({0xFFFF0300, 0x05000051, 0xA00F0000, value, 0, 0, 0x3F800000,
                             0x02000001, 0x800F0800, 0xA0E40000, 0xFFFF});
      const auto ps = MetalShaderIdentity(pixel);
      inputs.shaders.push_back({"test pixel " + std::to_string(i), std::move(pixel)});
      for (auto color : {MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGBA16Float})
        for (auto depth : {MTLPixelFormatInvalid, MTLPixelFormatDepth32Float_Stencil8})
          for (int lean = 0; lean < 2; lean++) {
            MetalRecipe recipe{}; recipe.vs = vs; recipe.ps = ps;
            recipe.key.color[0] = color; recipe.key.mask[0] = MTLColorWriteMaskAll;
            recipe.key.depth = recipe.key.stencil = depth;
            recipe.key.src = recipe.key.src_a = MTLBlendFactorOne;
            recipe.key.dst = recipe.key.dst_a = MTLBlendFactorZero;
            recipe.key.lean = lean;
            inputs.recipes.push_back(recipe);
          }
    }
    return inputs;
  };
  return profile;
}
