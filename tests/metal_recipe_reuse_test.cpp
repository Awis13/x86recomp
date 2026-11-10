#include <SDL3/SDL.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "runtime/gfx/device.h"
#include "metal_test_profile.h"

using namespace recomp::gfx;

static void Require(bool ok, const char* message) {
  if (!ok) { std::fprintf(stderr, "recipe reuse test: %s\n", message); std::exit(1); }
}

int main(int argc, char** argv) {
  Require(argc == 4, "expected game, cache and mode");
  ConfigureMetalPreparation(argv[1], argv[2], MetalTestProfile(argv[1]));
  Require(SDL_Init(SDL_INIT_VIDEO), SDL_GetError());
  auto* window = SDL_CreateWindow("Recipe reuse test", 16, 16, SDL_WINDOW_METAL | SDL_WINDOW_HIDDEN);
  Require(window != nullptr, SDL_GetError());
  {
    auto device = CreateMetalDevice(window);
    Require(device != nullptr, "Metal device unavailable");
    device->ResetSwapChain(16, 16, false);
    auto target = device->CreateTexture(4, 4, 1, Format::A8R8G8B8, true);
    ColorTarget targets[4] = {{target, 0}};
    device->SetRenderTargets(targets, 0);
    device->SetViewport({0, 0, 4, 4, 0, 1});
    device->SetRenderState(7, 0);   // ZENABLE
    device->SetRenderState(22, 1);  // CULLMODE: none
    // Keep the same full fragment variant in seed and reuse. With alpha test
    // disabled, a seed draw can race the optional lean variant's compilation.
    device->SetRenderState(15, 1);  // ALPHATESTENABLE
    device->SetRenderState(24, 0);  // ALPHAREF
    device->SetRenderState(25, 5);  // ALPHAFUNC: greater (alpha 1 passes)
    recomp::hlsl::Program vertex, fragment;
    vertex.stage = recomp::hlsl::Stage::Vertex;
    vertex.glsl = "void main() { vec2 p[3] = vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3)); gl_Position=vec4(p[gl_VertexIndex],0,1); }";
    fragment.stage = recomp::hlsl::Stage::Pixel;
    fragment.color_outputs = 1;
    fragment.glsl = "layout(location=0) out vec4 color; void main() { color=vec4(1,0,0,1); }";
    if (std::string(argv[3]) == "unused") {
      // Game startup registers many shaders that this scene never draws.
      // Registration/destruction must not compile or publish their libraries.
      std::fprintf(stderr, "unused begin\n");
      auto unused = fragment;
      unused.glsl += "\n// unused startup shader\n";
      auto sid = device->CreateShader(unused);
      device->DestroyShader(sid);
      std::fprintf(stderr, "unused end\n");
    }
    ShaderDraw draw;
    draw.vs = device->CreateShader(vertex);
    draw.ps = device->CreateShader(fragment);
    draw.prim = 4; draw.primitives = 1; draw.vertex_count = 3;
    if (std::string(argv[3]) == "prefetch") {
      std::vector<ShaderId> pending;
      for (int i = 0; i < 64; i++) {
        auto spare = fragment;
        spare.glsl += "\n// prefetched spare " + std::to_string(i) + "\n";
        pending.push_back(device->CreateShader(spare));
      }
      // Presentation starts bounded prefetch. The draw below races its own
      // jobs, while destruction cancels queued jobs and shutdown joins lanes.
      device->Present();
      for (auto sid : pending) device->DestroyShader(sid);
    }
    if (std::string(argv[3]) == "reuse") {
      // The parent waits for an archive-only background hit before releasing
      // the draw. No timer or readiness barrier is added to the renderer.
      std::puts("shaders created"); std::fflush(stdout);
      Require(std::getchar() == '\n', "parent did not release draw");
    }
    if (std::string(argv[3]) == "first-use") {
      // Fewer than a batch's worth of operations: the encoder has not seen
      // these draws. Both real descriptors must finish before readback flushes
      // the recording, and repeated binds must share the same request.
      device->DrawShader(draw);
      device->SetRenderState(168, 7);  // RGB only
      device->DrawShader(draw);
      device->SetRenderState(168, 15);
      device->DrawShader(draw);
      std::puts("draws recorded"); std::fflush(stdout);
      Require(std::getchar() == '\n', "parent did not release readback");
    }
    device->DrawShader(draw);
    uint32_t pixels[16]{};
    device->ReadBack({0, 0, 4, 4}, pixels, 16, Format::A8R8G8B8);
    for (auto pixel : pixels) Require(pixel == 0xffff0000, "wrong rendered pixels");
    device->DestroyShader(draw.ps);
    device->DestroyShader(draw.vs);
  }
  SDL_DestroyWindow(window);
  SDL_Quit();
}
