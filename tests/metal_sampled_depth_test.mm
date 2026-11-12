// End-to-end sampled-depth specialization through the actual Metal backend.
// The first draw must use the generic shader, then a background-built pipeline
// must become active and remain active after its handoff to the encoder.
// Changing texture types must preserve pixels; a fifth mask falls back safely.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <SDL3/SDL.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include "runtime/gfx/device.h"

extern char** environ;
using namespace recomp;
namespace {
int failures = 0;
void Check(bool ok, const char* what) {
  if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); failures++; }
}
hlsl::Program Vertex() {
  hlsl::Program p;
  p.inputs = {{hlsl::kPosition, 0}};
  p.varyings = {{hlsl::kTexcoord, 0}};
  p.glsl = R"(layout(location = 0) out vec4 uv;
void main() { gl_Position = r_position(r_fetch(0)); uv = vec4(0.5); }
)";
  return p;
}
hlsl::Program Pixel(int nonce) {
  hlsl::Program p;
  p.stage = hlsl::Stage::Pixel;
  p.varyings = {{hlsl::kTexcoord, 0}};
  p.color_outputs = 1;
  p.glsl = "// handoff test " + std::to_string(nonce) + "\n";
  p.glsl += "layout(location = 0) in vec4 uv; layout(location = 0) out vec4 color;\n";
  for (unsigned i = 0; i < 8; i++) {
    p.samplers.push_back({i, hlsl::ParamType::Sampler2D});
    p.glsl += "layout(R_SAMPLER_BINDING(" + std::to_string(i) + ")) uniform sampler2D s" + std::to_string(i) + ";\n";
  }
  p.glsl += "void main() { vec4 v = vec4(0.0);\n";
  for (unsigned i = 0; i < 8; i++)
    p.glsl += "v += R_DEPTH_TEX(" + std::to_string(i) + ", texture(s" + std::to_string(i) + ", uv.xy));\n";
  p.glsl += "color = v * 0.125; r_color0(color); }\n";
  return p;
}
bool Specialized(const std::string& stats) {
  const size_t at = stats.find(" sampled-depth specialized draws per frame");
  if (at == std::string::npos) return false;
  const size_t start = stats.rfind(' ', at - 1);
  return start != std::string::npos && std::strtod(stats.c_str() + start + 1, nullptr) > 0;
}
struct Fixture {
  std::unique_ptr<gfx::Device> dev;
  gfx::TextureId output = 0, color = 0, depth = 0;
  gfx::ShaderId vs = 0;
  gfx::ShaderDraw draw{};
  const float vertices[4][4] = {{-1,-1,0,1}, {1,-1,0,1}, {-1,1,0,1}, {1,1,0,1}};
  gfx::VertexInput input{0, 0, 3};  // FLOAT4
  explicit Fixture(SDL_Window* window) : dev(gfx::CreateMetalDevice(window)) {
    if (!dev) return;
    dev->ResetSwapChain(512, 512, false);
    dev->EnablePerfStats();
    dev->SetRenderState(168, 15);
    dev->SetRenderState(22, 1);  // no culling
    output = dev->CreateTexture(512,512,1,gfx::Format::A8R8G8B8,true);
    color = dev->CreateTexture(1,1,1,gfx::Format::A32B32G32R32F,false);
    const float pixel[4] = {0.125f,0.375f,0.75f,0.875f};
    dev->UpdateTexture(color,0,{0,0,1,1},pixel,sizeof pixel);
    gfx::DepthId z = 0;
    depth = dev->CreateDepthTexture(1,1,&z);
    auto scratch = dev->CreateTexture(1,1,1,gfx::Format::A8R8G8B8,true);
    dev->SetRenderTarget(scratch,z);
    dev->SetViewport({0,0,1,1,0,1});
    dev->Clear(0,nullptr,3,0,0.25f,0);
    uint32_t ignored;
    dev->ReadBack({0,0,1,1},&ignored,sizeof ignored,gfx::Format::A8R8G8B8);
    dev->DestroyTexture(scratch);
    vs = dev->CreateShader(Vertex());
    draw.vs = vs; draw.prim = 5; draw.primitives = 2; draw.vertex_count = 4;
    draw.inputs = &input; draw.input_count = 1;
    draw.streams[0] = {reinterpret_cast<const uint8_t*>(vertices),sizeof vertices,sizeof vertices[0]};
  }
  std::pair<uint32_t,bool> Frame(gfx::ShaderId ps, unsigned mask) {
    draw.ps = ps;
    for (unsigned i = 0; i < 8; i++) draw.textures[i] = mask & (1u << i) ? depth : color;
    dev->SetRenderTarget(output,0);
    dev->SetViewport({0,0,512,512,0,1});
    dev->DrawShader(draw);
    uint32_t pixel = 0;
    dev->ReadBack({256,256,257,257},&pixel,sizeof pixel,gfx::Format::A8R8G8B8);
    dev->Present();
    return {pixel, Specialized(dev->TakePerfStats())};
  }
  void Equal(uint32_t a,uint32_t b) {
    Check(a == b,"generic/optimized packed RGB pixels match");
  }
  void Activate(gfx::ShaderId ps,unsigned mask) {
    const auto [baseline,active] = Frame(ps,mask);
    Check((baseline & 0x00FFFFFFu) != 0 && (baseline & 0x00FFFFFFu) != 0x00FFFFFFu,
          "test draw covers sample with initialized texture data");
    Check(!active,"new mask starts with generic fallback");
    bool ready = false;
    const auto deadline = std::chrono::steady_clock::now()+std::chrono::seconds(8);
    while (std::chrono::steady_clock::now()<deadline) {
      auto [pixel,on] = Frame(ps,mask); Equal(baseline,pixel);
      if (on) { ready = true; break; }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    Check(ready,"background pipeline becomes active under its original key");
    if (!ready) return;
    for (int i = 0; i < 3; i++) {
      auto [pixel,on] = Frame(ps,mask); Equal(baseline,pixel);
      Check(on,"pipeline remains active after encoder consumes prebuilt entry");
    }
  }
};
int Run() {
  @autoreleasepool {
    if (!MTLCreateSystemDefaultDevice()) return 77;
    if (!SDL_Init(SDL_INIT_VIDEO)) return 77;
    SDL_Window* window = SDL_CreateWindow("Metal sampled-depth test",512,512,SDL_WINDOW_METAL|SDL_WINDOW_HIDDEN);
    if (!window) { SDL_Quit(); return 77; }
    setenv("RECOMP_SHADER_LANES","1",1); setenv("RECOMP_AHEAD_BOOST","0",1);
    setenv("RECOMP_PIPELINE_ARCHIVE","0",1); setenv("RECOMP_SAMPLED_DEPTH","1",1);
    setenv("RECOMP_REVERSED_Z","1",1); setenv("RECOMP_RENDER_SCALE","1",1);
    unsetenv("RECOMP_UPSCALER");
    {
      Fixture f(window);
      if (!f.dev) return 1;
      auto ps = f.dev->CreateShader(Pixel(1));
      for (unsigned mask : {0u,1u,2u,3u}) { f.Activate(ps,mask); if (failures) return 1; }
      auto [reference,active] = f.Frame(ps,4);
      Check(!active,"fifth mask uses generic fallback");
      for (int i = 0; i < 5; i++) {
        auto [pixel,on] = f.Frame(ps,4); f.Equal(reference,pixel); Check(!on,"variant limit remains bounded");
      }
      f.dev->DestroyShader(ps);
      auto pending = f.dev->CreateShader(Pixel(2));
      Check(!f.Frame(pending,0).second,"pending shader still uses generic fallback");
      f.dev->DestroyShader(pending);
      auto next = f.dev->CreateShader(Pixel(3)); f.Activate(next,0);
      f.dev->DestroyShader(next);
    }
    setenv("RECOMP_SAMPLED_DEPTH","0",1);
    {
      Fixture f(window);
      if (!f.dev) return 1;
      auto ps = f.dev->CreateShader(Pixel(4));
      for (int i = 0; i < 5; i++) Check(!f.Frame(ps,0).second,"opt-out keeps original renderer path");
      f.dev->DestroyShader(ps);
    }
    SDL_DestroyWindow(window); SDL_Quit();
    std::printf("%s sampled-depth pipeline handoff, texture rebinding, variant bound, destruction, opt-out\n",failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
  }
}
}
int main(int argc,char** argv) {
  if (argc>1) return Run();
  // Start a fresh process so CoreFoundation sees the isolated cache home
  // before initialization; never add test shaders to the user's cache.
  char path[] = "/tmp/metal_sampled_depth_test.XXXXXX";
  if (!mkdtemp(path)) return 1;
  setenv("CFFIXED_USER_HOME",path,1);
  char* child[] = {argv[0],const_cast<char*>("--child"),nullptr};
  pid_t pid; int status = 0;
  if (posix_spawn(&pid,argv[0],nullptr,nullptr,child,environ)) {
    std::filesystem::remove_all(path);
    return 1;
  }
  pid_t waited;
  do { waited = waitpid(pid,&status,0); } while (waited<0 && errno==EINTR);
  std::filesystem::remove_all(path);
  return waited==pid && WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
