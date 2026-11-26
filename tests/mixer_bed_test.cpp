// The mixer's 5.1 mix (runtime/audio/mixer.h), rendered offline: a 5.1
// source reaches the mix channel for channel, a 7.1 one with its sides in
// the back pair, a stereo voice the front pair only; DirectSound's pan
// takes one side's pair down and centre and LFE to the mean; a 5.1 source
// at half the mix's rate moves at half the pace and interpolates, one that
// ends raises its end; every channel is clamped; the ring is sized for the
// device and its fill kept within it; a stereo mixer's frames stay two
// floats and its ring is as before.

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include <SDL3/SDL.h>

#include <chrono>
#include <thread>

#include "runtime/audio/mixer.h"
#include "runtime/audio/spatial_output.h"

namespace {

using namespace recomp::audio;

int g_failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    g_failures++;
  }
}

bool Near(float a, float b) { return std::fabs(a - b) < 1e-5f; }

// Channel c of frame k holds values[c] + step * k; `frames` of them (-1:
// endless).
class RampSource : public Source {
 public:
  RampSource(std::vector<float> values, int rate, float step = 0, int64_t frames = -1)
      : v_(std::move(values)), step_(step), frames_(frames) {
    channels_ = int(v_.size());
    rate_ = rate;
  }
  int Read(float* out, int n) override {
    int got = 0;
    for (; got < n && (frames_ < 0 || next_ < frames_); got++, next_++)
      for (float x : v_) *out++ = x + step_ * float(next_);
    return got;
  }
  bool Seek(int64_t f) override {
    next_ = f;
    return true;
  }
  int64_t Length() const override { return frames_; }

 private:
  std::vector<float> v_;
  float step_;
  int64_t frames_, next_ = 0;
};

std::unique_ptr<Voice> MakeVoice(std::vector<float> values, int rate = 44100, float step = 0,
                                 int64_t frames = -1) {
  auto v = std::make_unique<Voice>();
  v->src = std::make_unique<RampSource>(std::move(values), rate, step, frames);
  v->dsound = true;  // linear volume, DSBPAN
  v->volume = 1.0f;
  v->pan = 0.0f;
  v->state = Voice::State::Playing;
  return v;
}

// The last frame of a mix of n frames (past the start's interpolation).
std::vector<float> MixLast(Mixer& m, int n) {
  std::vector<float> out(size_t(m.channels()) * size_t(n) + 8, 7.0f);  // 8 sentinels after
  m.Mix(out.data(), n);
  return out;
}

void FiveOne() {
  Mixer m;
  m.OpenOffline(44100, 6);
  Check(m.channels() == 6, "a 5.1 mixer mixes six channels");
  auto bed = MakeVoice({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f});
  auto st = MakeVoice({0.1f, 0.2f});
  {
    std::lock_guard<std::mutex> l(m.mu());
    m.Add(bed.get());
    m.Add(st.get());
  }
  const int n = 256;
  std::vector<float> out = MixLast(m, n);
  const float* f = &out[6 * size_t(n - 1)];
  Check(Near(f[kFL], 0.11f) && Near(f[kFR], 0.22f), "front pair: the 5.1 source's and the stereo voice's");
  Check(Near(f[kFC], 0.03f), "centre: the 5.1 source's only");
  Check(Near(f[kLFE], 0.04f), "LFE: the 5.1 source's only");
  Check(Near(f[kBL], 0.05f) && Near(f[kBR], 0.06f), "back pair: the 5.1 source's only");
  Check(out[6 * size_t(n)] == 7.0f, "the mix writes n frames of six and no more");

  {
    std::lock_guard<std::mutex> l(m.mu());
    m.Remove(st.get());
    bed->pan = -10000.0f;  // DSBPAN_LEFT: the right side off
  }
  out = MixLast(m, n);
  f = &out[6 * size_t(n - 1)];
  Check(Near(f[kFL], 0.01f) && Near(f[kBL], 0.05f), "panned left: the left pair at full");
  Check(f[kFR] < 1e-6f && f[kBR] < 1e-6f, "panned left: the right pair silent");
  // The right side at 10^(-10000/2000) = 1e-5: centre and LFE at the mean
  // of the two sides' gains.
  const float mid = 0.5f * (1.0f + 1e-5f);
  Check(Near(f[kFC], 0.03f * mid) && Near(f[kLFE], 0.04f * mid), "panned left: centre and LFE at the mean gain");
}

// A 5.1 source at 22050 Hz in a 44100 Hz mix: two mix frames a source
// frame, the second interpolated half way.
void Rate() {
  Mixer m;
  m.OpenOffline(44100, 6);
  auto v = MakeVoice({0, 0, 0, 0, 0, 0}, 22050, 0.001f);
  {
    std::lock_guard<std::mutex> l(m.mu());
    m.Add(v.get());
  }
  const int n = 200;
  std::vector<float> out = MixLast(m, n);
  Check(v->Position() == n / 2, "half the mix's frames of a source at half its rate");
  bool ramp = true;
  for (int i = 0; i < n; i++) ramp &= Near(out[6 * size_t(i) + kBR], 0.001f * 0.5f * float(i));
  Check(ramp, "every other frame interpolated half way");
}

// A 5.1 source of 100 frames ends within a mix of 256.
void End() {
  Mixer m;
  m.OpenOffline(44100, 6);
  auto v = MakeVoice({0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f}, 44100, 0, 100);
  {
    std::lock_guard<std::mutex> l(m.mu());
    m.Add(v.get());
  }
  std::vector<float> out = MixLast(m, 256);
  Check(v->state == Voice::State::Done && v->eos, "a 5.1 source that ends is done and says so");
  Check(Near(out[6 * 50 + kBR], 0.1f) && out[6 * 200 + kBR] == 0.0f, "heard until its end, silent after");
}

// Over full scale in every channel: each clamped.
void Clamp() {
  Mixer m;
  m.OpenOffline(44100, 6);
  auto v = MakeVoice({1.5f, -1.5f, 1.5f, -1.5f, 1.5f, -1.5f});
  {
    std::lock_guard<std::mutex> l(m.mu());
    m.Add(v.get());
  }
  std::vector<float> out = MixLast(m, 64);
  bool clamped = true;
  for (int i = 0; i < 6 * 64; i++) clamped &= std::fabs(out[size_t(i)]) <= 1.0f;
  Check(clamped, "every channel of every frame clamped to full scale");
}

// The ring for the spatial output's device: two requests and a period,
// room for the largest request a device may make, the fill kept within
// the ring when a device asks for more than it holds.
void Ring() {
  Mixer m;
  m.OpenOffline(44100, 6);
  m.SizeRing(471, 11290);  // a 512-frame device at 48 kHz; up to 12288 frames there
  Check(m.fill_target() == 2 * 471 + 441, "the fill: two requests and a period");
  Check(m.ring_capacity() >= 2 * 11290 + 2 * 441, "room for the largest request's fill and a period");
  m.SetRequest(4096);  // a device of 4096-frame requests at 44.1 kHz (the offline mixer's rate)
  Check(m.fill_target() == 2 * 4096 + 441, "a larger device: its fill");
  m.SetRequest(1 << 20);
  Check(m.fill_target() == m.ring_capacity() - 441, "a device asking for more than the ring: the fill kept in it");
  // The stereo device's ring as before: 941-frame requests, a 4096-frame
  // ring filled to 2323.
  Mixer st;
  st.OpenOffline(44100);
  st.SizeRing(941, 0);
  Check(st.ring_capacity() == 4096 && st.fill_target() == 2323, "the stereo ring as before");
}

void SevenOne() {
  Mixer m;
  m.OpenOffline(44100, 6);
  auto v = MakeVoice({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f});
  {
    std::lock_guard<std::mutex> l(m.mu());
    m.Add(v.get());
  }
  const int n = 128;
  std::vector<float> out = MixLast(m, n);
  const float* f = &out[6 * size_t(n - 1)];
  Check(Near(f[kBL], 0.05f + 0.07f) && Near(f[kBR], 0.06f + 0.08f), "7.1: sides into the back pair");
  Check(Near(f[kFL], 0.01f) && Near(f[kFC], 0.03f), "7.1: fronts and centre in place");
}

// A stereo mixer as before: frames of two floats.
void Stereo() {
  Mixer m;
  m.OpenOffline(44100);
  Check(m.channels() == 2, "a mixer opened without channels is stereo");
  auto st = MakeVoice({0.1f, 0.2f});
  {
    std::lock_guard<std::mutex> l(m.mu());
    m.Add(st.get());
  }
  const int n = 128;
  std::vector<float> out = MixLast(m, n);
  Check(Near(out[2 * size_t(n - 1)], 0.1f) && Near(out[2 * size_t(n - 1) + 1], 0.2f), "stereo frames");
  Check(out[2 * size_t(n)] == 7.0f, "a stereo mix writes two floats a frame and no more");
}

}  // namespace

// A spatial output of our own, no device: what the mixer gives it.
struct FakeOutput : SpatialOutput {
  bool start_ok = true;
  Pull pull = nullptr;
  Changed changed = nullptr;
  void* ctx = nullptr;
  bool* stopped = nullptr;
  int rate() const override { return 44100; }
  int device_frames() const override { return 512; }
  int max_frames() const override { return 8192; }
  bool Start(Pull p, Changed c, void* x) override {
    pull = p, changed = c, ctx = x;
    return start_ok;
  }
  void Stop() override {
    if (stopped) *stopped = true;
  }
};

// The mixer with a spatial output: the ring sized for its device and for
// the largest request it may make, the mix reaching it through its pull
// from the mixer thread, a change of device moving the fill, Close
// stopping it; an output that will not start leaves the mixer closed and
// stereo.
void Spatial() {
  SDL_SetHint(SDL_HINT_AUDIO_DRIVER, "dummy");
  SDL_InitSubSystem(SDL_INIT_AUDIO);
  {
    Mixer m;
    auto refused = std::make_unique<FakeOutput>();
    refused->start_ok = false;
    Check(!m.OpenSpatial(44100, std::move(refused)) && m.channels() == 2, "an output that will not start: stereo");
  }
  Mixer m;
  bool stopped = false;
  auto owned = std::make_unique<FakeOutput>();
  FakeOutput* out = owned.get();
  out->stopped = &stopped;
  if (!m.OpenSpatial(44100, std::move(owned))) {
    Check(false, "the mixer opened with a spatial output");
    return;
  }
  Check(m.channels() == 6, "with a spatial output the mix is 5.1");
  Check(m.fill_target() == 2 * 512 + 441, "the ring filled for the device's requests");
  Check(m.ring_capacity() >= 2 * 8192 + 2 * 441, "room for the largest request the output may ask for");
  auto v = MakeVoice({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f});
  {
    std::lock_guard<std::mutex> l(m.mu());
    m.Add(v.get());
  }
  std::vector<float> buf(6 * 512);
  bool heard = false;
  for (int k = 0; k < 200 && !heard; k++) {  // up to 2 s
    out->pull(out->ctx, buf.data(), 512);
    heard = Near(buf[6 * 511 + kBR], 0.06f) && Near(buf[6 * 511 + kFL], 0.01f);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  Check(heard, "the mix reaches the output through its pull, channel for channel");
  out->changed(out->ctx, 4096);
  Check(m.fill_target() == 2 * 4096 + 441, "a new device's requests move the fill");
  {
    std::lock_guard<std::mutex> l(m.mu());
    m.Remove(v.get());
  }
  m.Close();
  Check(stopped && m.channels() == 2, "Close stops the output");
}

int main() {
  FiveOne();
  SevenOne();
  Rate();
  End();
  Clamp();
  Ring();
  Spatial();
  Stereo();
  if (g_failures) return 1;
  std::printf("ok\n");
  return 0;
}
