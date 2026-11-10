#include <cstring>
#include "metal_test_profile.h"
int main(int argc, char** argv) {
  std::string directory;
  for (int i = 3; i + 1 < argc; i++)
    if (!std::strcmp(argv[i], "--game-dir")) directory = argv[i + 1];
  int result = recomp::gfx::MetalPreparationCommand(argc, argv, MetalTestProfile(directory));
  return result < 0 ? 2 : result;
}
