#pragma once
// What the spatial output (spatial_output_mac.cpp) runs between the mix and
// the device, without the device (macOS): for headphones, the built-in
// speakers or a stereo pair of speakers, Apple's spatial mixer audio unit
// set up for the 5.1 mix (one input, the mix as an ambience bed,
// kAudioChannelLayoutTag_MPEG_5_1_A: L R C LFE Ls Rs, surround.h's order;
// planar float; stereo out; the output type for the kind; head tracking
// asked for on headphones and the personal profile's Auto mode), the back
// pair given the back's spectral cue first on headphones (surround.h's
// BackCue), the LFE folded first where the renderer drops it and a peak
// limiter after it;
// for a 5.1 device, the mix as it is; failing those, the mix folded to
// stereo here. The output unit's render callback calls Render; a test
// renders it offline.

#include <AudioToolbox/AudioToolbox.h>

#include <atomic>
#include <cstdint>
#include <vector>

#include "runtime/audio/surround.h"

namespace recomp::audio {

// The mixer's kAudioUnitProperty_SpatialMixerOutputType for a rendered kind.
UInt32 SpatialMixerOutputType(OutputKind k);
// Planar float of `channels` at `rate`.
AudioStreamBasicDescription PlanarFloat(double rate, int channels);
// The channel labels of a layout (a tag or a bitmap expanded).
std::vector<uint32_t> ChannelLabels(const AudioChannelLayout* layout);

class SpatialChain {
 public:
  // n frames of the 5.1 mix, interleaved, at the chain's rate.
  using Pull = void (*)(void* ctx, float* frames, int n);

  SpatialChain() = default;
  SpatialChain(const SpatialChain&) = delete;
  SpatialChain& operator=(const SpatialChain&) = delete;
  ~SpatialChain();

  // Sets the chain up for `kind` at `rate`, at most `max_frames` a render,
  // the mix from `pull`. False (and failed(), error()) if the host
  // refuses; only while Render is not called.
  bool Configure(OutputKind kind, double rate, UInt32 max_frames, Pull pull, void* ctx);
  // Undoes Configure; only while Render is not called.
  void Teardown();
  // From one rendered kind to another while Render runs: the mixer takes a
  // new output type (and head tracking) as it renders.
  bool SetRenderedKind(OutputKind kind);

  OutputKind kind() const { return kind_.load(std::memory_order_relaxed); }
  // What the output unit's input takes: 6 channels for a 5.1 device, else 2.
  int output_channels() const { return kind() == OutputKind::kStraight ? kBedChannels : 2; }
  AudioChannelLayoutTag output_layout() const {
    return kind() == OutputKind::kStraight ? kAudioChannelLayoutTag_MPEG_5_1_A : kAudioChannelLayoutTag_Stereo;
  }
  // The real-time render: n frames into io (output_channels() planar
  // buffers).
  OSStatus Render(AudioUnitRenderActionFlags* flags, const AudioTimeStamp* ts, UInt32 n, AudioBufferList* io);

  // What the mixer took when set up (false for kinds without it).
  bool head_tracking() const { return head_tracking_; }
  bool personal() const { return personal_; }
  double latency_ms() const { return latency_ms_; }
  AudioUnit mixer() const { return mixer_; }
  const char* failed() const { return failed_; }
  OSStatus error() const { return error_; }
  // Frames the limiter took down since the start, and the lowest gain.
  uint64_t limited_frames() const { return limited_.load(std::memory_order_relaxed); }
  float lowest_gain() const { return lowest_gain_.load(std::memory_order_relaxed); }

 private:
  bool Fail(const char* what, OSStatus e);
  bool SetHeadTracking(bool on);
  void Fill(AudioBufferList* io, UInt32 n);
  void Limit(AudioBufferList* io, UInt32 n);
  static OSStatus MixerInput(void* self, AudioUnitRenderActionFlags*, const AudioTimeStamp*, UInt32, UInt32 n,
                             AudioBufferList* io);

  AudioUnit mixer_ = nullptr;
  bool mixer_ready_ = false;
  std::atomic<OutputKind> kind_{OutputKind::kHeadphones};
  Pull pull_ = nullptr;
  void* ctx_ = nullptr;
  UInt32 max_frames_ = 0;
  std::vector<float> inter_;  // a pull's interleaved frames
  PeakLimiter limiter_;
  BackCue back_cue_;
  std::atomic<uint64_t> limited_{0};
  std::atomic<float> lowest_gain_{1.0f};
  bool head_tracking_ = false, personal_ = false;
  double latency_ms_ = 0;
  const char* failed_ = nullptr;
  OSStatus error_ = noErr;
};

}  // namespace recomp::audio
