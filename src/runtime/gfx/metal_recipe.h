#pragma once
#include <cstdint>
#include <cstring>

namespace recomp::gfx {
// Shared by preparation and rendering. Bump the cache schema on layout changes.
struct MetalPipeKey {
  uint32_t vs, ps;
  uint16_t color[4];
  uint16_t depth, stencil;
  uint8_t blend, src, dst, op, src_a, dst_a, op_a;
  uint8_t mask[4];
  uint8_t lean;
  bool operator==(const MetalPipeKey& o) const { return std::memcmp(this, &o, sizeof *this) == 0; }
};
static_assert(sizeof(MetalPipeKey) == 32);
struct MetalRecipe { uint64_t vs, ps; MetalPipeKey key; };
static_assert(sizeof(MetalRecipe) == 48);

}  // namespace recomp::gfx
