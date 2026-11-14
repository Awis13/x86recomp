#pragma once
// The 5.1 mix and what carries it to the listener: how sources of each
// channel count land in the mix (and fold to stereo), the LFE fold for
// outputs that drop it, the speaker configuration DirectSound reports for
// the mix, which way a host output device is rendered for, the fallback
// when a device refuses that, the back's spectral cue before a headphone
// renderer and the peak limiter after it.
// Pure, no host audio API; the mixer (mixer.h), DirectSound (dsound.cpp)
// and the Apple output (spatial_output.h, spatial_chain_mac.h) use them.
//
// The mix is WAVEFORMATEXTENSIBLE's KSAUDIO_SPEAKER_5POINT1 order, mask
// 0x3F: front left, front right, centre, LFE, back left, back right. The
// same order as Core Audio's kAudioChannelLayoutTag_MPEG_5_1_A (L R C LFE
// Ls Rs), so it goes to a host 5.1 bed as it is.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>

namespace recomp::audio {

constexpr int kBedChannels = 6;
enum BedSlot : int { kFL = 0, kFR = 1, kFC = 2, kLFE = 3, kBL = 4, kBR = 5 };

// The slot of a source's channel c in the mix. Sources without a channel
// mask are in the WAVEFORMATEXTENSIBLE order FL FR FC LFE BL BR SL SR (as
// DirectSound reads them); a 7.1 source's side pair joins the back pair.
inline int BedSlotOf(int c) {
  if (c < kBedChannels) return c;
  return c == 6 ? kBL : kBR;
}

// One frame of a source of `channels` channels (more than two) added into
// a 5.1 frame.
inline void AddToBed(const float* src, int channels, float* bed) {
  for (int c = 0; c < channels && c < 8; c++) bed[BedSlotOf(c)] += src[c];
}

// Stereo frames (interleaved) added to the front pair of 5.1 frames.
inline void AddStereoToBed(const float* stereo, float* bed, int frames) {
  for (int i = 0; i < frames; i++) {
    bed[kBedChannels * i + kFL] += stereo[2 * i];
    bed[kBedChannels * i + kFR] += stereo[2 * i + 1];
  }
}

// The LFE moved into the front pair at -6 dB each, for an output that
// drops it (planar buffers of n frames). The LFE buffer is left silent.
constexpr float kLfeFoldGain = 0.5f;
inline void FoldLfe(float* fl, float* fr, float* lfe, int n) {
  for (int i = 0; i < n; i++) {
    fl[i] += kLfeFoldGain * lfe[i];
    fr[i] += kLfeFoldGain * lfe[i];
    lfe[i] = 0.0f;
  }
}

// A DirectSound buffer of `buffer_channels` as a source of a mix of
// `mix_channels`: a 5.1 mix takes more than two as they are, a stereo mix
// gets them folded to two (FoldToStereo), and mono or stereo stay so.
inline int SourceChannels(int buffer_channels, int mix_channels) {
  if (buffer_channels > 2 && mix_channels >= kBedChannels) return buffer_channels;
  return buffer_channels < 2 ? buffer_channels : 2;
}

// A frame of more than two channels (FL FR FC LFE BL BR SL SR) folded to
// stereo: centre and the surround pairs at -3 dB into both sides, the LFE
// left out. Not clamped.
constexpr float kFoldGain = 0.7071f;
inline void FoldToStereo(const float* f, int channels, float* l, float* r) {
  *l = f[0];
  *r = channels > 1 ? f[1] : f[0];
  if (channels >= 3) *l += kFoldGain * f[2], *r += kFoldGain * f[2];
  for (int c = 4; c + 1 < channels; c += 2) *l += kFoldGain * f[c], *r += kFoldGain * f[c + 1];
}

// IDirectSound8::GetSpeakerConfig for a mix of `channels`:
// DSSPEAKER_5POINT1 (6) for the 5.1 mix, the configuration whose buffers
// are KSAUDIO_SPEAKER_5POINT1 (back speakers, mask 0x3F, the mix's
// order); DSSPEAKER_5POINT1_SURROUND (9) would ask for side speakers
// (0x60F). Otherwise DSSPEAKER_STEREO (4). The geometry byte is 0.
constexpr uint32_t kSpeakerStereo = 4, kSpeaker5Point1 = 6;
inline uint32_t SpeakerConfigFor(int channels) {
  return channels >= kBedChannels ? kSpeaker5Point1 : kSpeakerStereo;
}

// Whether a device's speaker layout (Core Audio channel labels, as Audio
// MIDI Setup's speaker configuration sets them) has surround positions: a
// left and a right one behind or beside the listener. Unlabelled channels
// (kAudioChannelLabel_Unknown, _Unused, _Discrete_N) are not positions.
constexpr uint32_t kLabelLeftSurround = 5, kLabelRightSurround = 6, kLabelLeftSurroundDirect = 10,
                   kLabelRightSurroundDirect = 11, kLabelRearSurroundLeft = 33, kLabelRearSurroundRight = 34;
inline bool LayoutHasSurround(const uint32_t* labels, int n) {
  bool left = false, right = false;
  for (int i = 0; i < n; i++) {
    const uint32_t l = labels[i];
    left |= l == kLabelLeftSurround || l == kLabelLeftSurroundDirect || l == kLabelRearSurroundLeft;
    right |= l == kLabelRightSurround || l == kLabelRightSurroundDirect || l == kLabelRearSurroundRight;
  }
  return left && right;
}

// A host output device as far as the choice below needs it.
struct OutputDevice {
  enum class Transport { kBuiltIn, kBluetooth, kUsb, kHdmi, kDisplayPort, kAirPlay, kOther };
  // What the system says plays it (its output data source), where it says.
  enum class Source { kUnknown, kHeadphones, kInternalSpeakers, kExternalSpeakers };
  Transport transport = Transport::kOther;
  int channels = 2;              // output channels of all its streams
  bool surround_layout = false;  // its speaker layout has surround positions
  Source source = Source::kUnknown;
};

// Core Audio's output data source (kAudioDevicePropertyDataSource, a
// four-character code: 'hdpn' headphones, 'ispk' internal speakers,
// 'espk' external speakers) as OutputDevice::Source.
inline OutputDevice::Source SourceOf(uint32_t data_source) {
  using S = OutputDevice::Source;
  switch (data_source) {
    case 0x6864706E: return S::kHeadphones;        // 'hdpn'
    case 0x6973706B: return S::kInternalSpeakers;  // 'ispk'
    case 0x6573706B: return S::kExternalSpeakers;  // 'espk'
    default: return S::kUnknown;
  }
}

// How the 5.1 mix is rendered for a device:
//  - kStraight: a device whose speaker layout has surround positions takes
//    the 5.1 as it is;
//  - kHeadphones: binaural, for headphones;
//  - kBuiltInSpeakers: the Mac's own speakers (their own virtualizer);
//  - kExternalSpeakers: a stereo pair of speakers, panned without binaural;
//  - kFoldedStereo: no renderer at all, the mix folded to stereo here
//    (FoldToStereo): only where setting up the others failed.
enum class OutputKind { kStraight, kHeadphones, kBuiltInSpeakers, kExternalSpeakers, kFoldedStereo };

struct OutputChoice {
  OutputKind kind = OutputKind::kHeadphones;
  bool guessed = false;  // the system did not say what plays it
};

// The kind for a device: 5.1 straight only for a layout with surround
// positions (a device of many channels without one is taken by its front
// pair); then what the system says plays it; then by transport: the
// built-in output is its speakers, Bluetooth and USB devices are taken as
// headphones (guessed: the system does not tell a Bluetooth speaker from
// headphones), anything else as a pair of speakers.
inline OutputChoice ChooseOutput(const OutputDevice& d) {
  using T = OutputDevice::Transport;
  using S = OutputDevice::Source;
  if (d.channels >= kBedChannels && d.surround_layout) return {OutputKind::kStraight, false};
  switch (d.source) {
    case S::kHeadphones: return {OutputKind::kHeadphones, false};
    case S::kInternalSpeakers: return {OutputKind::kBuiltInSpeakers, false};
    case S::kExternalSpeakers: return {OutputKind::kExternalSpeakers, false};
    case S::kUnknown: break;
  }
  switch (d.transport) {
    case T::kBuiltIn: return {OutputKind::kBuiltInSpeakers, false};
    case T::kBluetooth:
    case T::kUsb: return {OutputKind::kHeadphones, true};
    default: return {OutputKind::kExternalSpeakers, false};
  }
}

// Whether the host's spatial renderer drops the LFE channel for a kind of
// output (measured on macOS 26 rendering an LFE-only tone offline: the
// built-in speakers' renderer gives silence; the headphone renderer plays
// it, low-passed, about 10 dB up as cinema's LFE is; external speakers'
// about 4 dB up). A 5.1 device keeps its own LFE; the folded stereo
// leaves it out.
inline bool OutputDropsLfe(OutputKind k) { return k == OutputKind::kBuiltInSpeakers; }

// Head tracking is asked for on headphones only.
inline bool OutputTracksHead(OutputKind k) { return k == OutputKind::kHeadphones; }

// The back pair is given the back's spectral cue on headphones only.
inline bool OutputCuesBack(OutputKind k) { return k == OutputKind::kHeadphones; }

// Whether a kind goes through the host's spatial renderer.
inline bool OutputIsRendered(OutputKind k) {
  return k == OutputKind::kHeadphones || k == OutputKind::kBuiltInSpeakers || k == OutputKind::kExternalSpeakers;
}

inline const char* OutputKindName(OutputKind k) {
  switch (k) {
    case OutputKind::kStraight: return "5.1 straight to the device";
    case OutputKind::kHeadphones: return "headphones";
    case OutputKind::kBuiltInSpeakers: return "built-in speakers";
    case OutputKind::kExternalSpeakers: return "external speakers";
    case OutputKind::kFoldedStereo: return "the mix folded to stereo";
  }
  return "?";
}

// Sets an output up for `want` with `configure(kind)` (true when it
// worked): once more if it fails, then as kFoldedStereo, so that the mix
// keeps being taken. The kind in effect, and in *ok whether any worked.
template <typename Configure>
OutputKind ConfigureWithFallback(OutputKind want, Configure configure, bool* ok) {
  *ok = true;
  if (configure(want) || configure(want)) return want;
  if (want != OutputKind::kFoldedStereo && configure(OutputKind::kFoldedStereo)) return OutputKind::kFoldedStereo;
  *ok = false;
  return want;
}

// A peak limiter for the renderer's output: the renderer (float throughout)
// sums the bed's channels into two ears and its peaks can pass full scale
// (measured: 1.14 on headphones, 1.24 on external speakers for a bed that
// peaked at 0.74), which the device would clip. Gain is taken down at once
// to keep every sample at or under the threshold (no look-ahead: no
// latency) and comes back with a 150 ms release, so the level of normal
// play is left alone and only its rare peaks are touched. All channels
// share one gain (the image does not move).
class PeakLimiter {
 public:
  static constexpr float kThreshold = 0.891f;  // -1 dBFS
  static constexpr double kReleaseMs = 150.0;

  void SetRate(double rate) {
    release_ = rate > 0 ? float(std::exp(-1.0 / (kReleaseMs * 0.001 * rate))) : 0.0f;
  }
  // n frames of `channels` planar buffers, in place.
  void Process(float* const* ch, int channels, int n) {
    for (int i = 0; i < n; i++) {
      float peak = 0;
      for (int c = 0; c < channels; c++) peak = std::max(peak, std::fabs(ch[c][i]));
      env_ = std::max(peak, env_ * release_);
      if (env_ > kThreshold) {
        const float g = kThreshold / env_;
        for (int c = 0; c < channels; c++) ch[c][i] *= g;
        limited_++;
        min_gain_ = std::min(min_gain_, g);
      }
    }
  }
  // Frames whose gain was taken down, and the lowest gain, since the last
  // TakeStats.
  void TakeStats(uint64_t* limited, float* min_gain) {
    *limited = limited_;
    *min_gain = min_gain_;
    limited_ = 0;
    min_gain_ = 1.0f;
  }

 private:
  float release_ = 0.0f, env_ = 0.0f, min_gain_ = 1.0f;
  uint64_t limited_ = 0;
};

// The back's spectral cue for the back pair, before a headphone renderer.
// Apple's spatial mixer renders the bed's back pair at +-110 degrees, and
// the image of a sound panned between them (a sound behind) comes out
// with nearly the spectrum of one panned between the front pair (a sound
// ahead): measured offline (macOS 26, headphones, the generic HRTF, the
// same noise in FL+FR and in BL+BR), the ears' 2-12 kHz spectra differ by
// 4.4 dB RMS and the back image is 0.7 dB brighter above 4 kHz, not
// duller. Both sit in the middle, so ahead and behind sound the same.
//
// Blauert's directional bands (Spatial Hearing, 1997, 2.3.2): a band near
// 1 kHz is heard as behind, one near 4-5 kHz as ahead, as the pinna
// shadows the treble of a sound from behind. So the back pair is cut
// 10 dB at 4.5 kHz (Q 0.7) and lifted 4 dB at 1.1 kHz (Q 1), RBJ peaking
// filters: through the renderer the spectra then differ by 8.2 dB RMS and
// the back image is 6.4 dB duller above 4 kHz; a single back channel's
// interaural level difference stays near what it was (14.3 dB, from 16.7)
// and the front pair is not touched. Of the shapes measured (high shelves
// at 3 and 4 kHz of -6 and -9 dB, a 5 kHz dip, these two bands at -8/+3
// dB) this gave the largest difference with the side cues kept.
class BackCue {
 public:
  static constexpr double kDipHz = 4500, kDipDb = -10, kDipQ = 0.7;
  static constexpr double kBumpHz = 1100, kBumpDb = 4, kBumpQ = 1.0;

  // Coefficients for `rate`; clears the filters' state.
  void SetRate(double rate) {
    dip_ = Peaking(rate, kDipHz, kDipDb, kDipQ);
    bump_ = Peaking(rate, kBumpHz, kBumpDb, kBumpQ);
    std::fill(std::begin(state_), std::end(state_), 0.0);
  }
  // n frames of the back left and back right channels, in place.
  void Process(float* bl, float* br, int n) {
    float* ch[2] = {bl, br};
    for (int c = 0; c < 2; c++)
      for (int i = 0; i < n; i++) ch[c][i] = float(Run(bump_, &state_[c * 4 + 2], Run(dip_, &state_[c * 4], ch[c][i])));
  }

 private:
  struct Coefs {
    double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
  };
  static Coefs Peaking(double rate, double hz, double db, double q) {
    const double a = std::pow(10.0, db / 40), w = 2 * 3.14159265358979323846 * hz / rate, cw = std::cos(w);
    const double alpha = std::sin(w) / (2 * q), a0 = 1 + alpha / a;
    return {(1 + alpha * a) / a0, -2 * cw / a0, (1 - alpha * a) / a0, -2 * cw / a0, (1 - alpha / a) / a0};
  }
  // Transposed direct form II; z holds the two state values.
  static double Run(const Coefs& k, double* z, double x) {
    const double y = k.b0 * x + z[0];
    z[0] = k.b1 * x - k.a1 * y + z[1];
    z[1] = k.b2 * x - k.a2 * y;
    return y;
  }
  Coefs dip_, bump_;
  double state_[8] = {};
};

}  // namespace recomp::audio
