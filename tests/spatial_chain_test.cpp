// What the spatial output runs between the mix and the device
// (runtime/audio/spatial_chain_mac.h), rendered offline without a device,
// through the same SpatialChain::Render the output unit's callback calls:
//  - headphones: each front channel heard on its own side, the back ones
//    too and unlike the front ones, the LFE heard, head tracking asked for,
//    the back pair duller than the front (the back's spectral cue), a
//    room after an impulse;
//  - external speakers: no such cue, no room;
//  - the built-in speakers: the renderer itself drops the LFE (rendered
//    without the chain), and through the chain it is heard (folded); no
//    head tracking;
//  - external speakers: a front channel on its own side only (panned, not
//    binaural); no head tracking;
//  - a full-scale 5.1 moment: no sample over the limiter's threshold;
//  - 5.1 straight: the mix as it is, channel for channel;
//  - the folded stereo: each channel where the fold puts it;
//  - headphones to the built-in speakers as it renders;
//  - the channel labels of a 5.1 tag and of stereo.
// macOS only; skipped (exit 77) where the host has no spatial mixer.

#include <AudioToolbox/AudioToolbox.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "runtime/audio/spatial_chain_mac.h"

namespace {

using namespace recomp::audio;

int g_failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    g_failures++;
  }
}

constexpr double kRate = 48000;
constexpr UInt32 kSlice = 512;

// The mix the chain pulls: a tone in one bed channel (or all), amplitude
// `amp`; or a step pattern (channel c holds c + 1 tenths) for the
// straight and folded checks.
struct Feed {
  int channel = 0;  // -1: every channel
  double freq = 1000;
  float amp = 0.5f;
  bool pattern = false;
  bool impulse = false;  // one sample of `amp` in `channel`, 0.5 s in
  uint64_t t = 0;
};

void Pull(void* ctx, float* frames, int n) {
  auto* f = static_cast<Feed*>(ctx);
  for (int i = 0; i < n; i++, f->t++)
    for (int c = 0; c < kBedChannels; c++) {
      float v = 0;
      if (f->impulse)
        v = c == f->channel && f->t == uint64_t(kRate / 2) ? f->amp : 0.0f;
      else if (f->pattern)
        v = 0.1f * float(c + 1);
      else if (f->channel < 0 || f->channel == c)
        v = f->amp * float(std::sin(2 * M_PI * f->freq * double(f->t) / kRate));
      frames[size_t(i) * kBedChannels + size_t(c)] = v;
    }
}

struct Out {
  std::vector<std::vector<float>> ch;
  double Rms(int c) const {
    double s = 0;
    for (float x : ch[size_t(c)]) s += double(x) * x;
    return std::sqrt(s / double(ch[size_t(c)].size() + 1));
  }
  float Peak() const {
    float p = 0;
    for (const auto& c : ch)
      for (float x : c) p = std::max(p, std::fabs(x));
    return p;
  }
};

double Db(double x) { return 20 * std::log10(x + 1e-12); }

// About 1 s through the chain, the first 0.2 s left out.
bool Render(SpatialChain& chain, Out* out) {
  const int nch = chain.output_channels();
  std::vector<std::vector<float>> buf(static_cast<size_t>(nch), std::vector<float>(size_t{kSlice}));
  out->ch.assign(size_t(nch), {});
  AudioTimeStamp ts{};
  ts.mFlags = kAudioTimeStampSampleTimeValid;
  for (int k = 0; k < 100; k++) {
    std::vector<uint8_t> mem(sizeof(AudioBufferList) + sizeof(AudioBuffer) * size_t(nch));
    auto* abl = reinterpret_cast<AudioBufferList*>(mem.data());
    abl->mNumberBuffers = UInt32(nch);
    for (int c = 0; c < nch; c++) abl->mBuffers[c] = {1, UInt32(kSlice * sizeof(float)), buf[size_t(c)].data()};
    AudioUnitRenderActionFlags flags = 0;
    if (chain.Render(&flags, &ts, kSlice, abl) != noErr) return false;
    ts.mSampleTime += kSlice;
    if (k < 20) continue;
    for (int c = 0; c < nch; c++) out->ch[size_t(c)].insert(out->ch[size_t(c)].end(), buf[size_t(c)].begin(), buf[size_t(c)].end());
  }
  return true;
}

bool Through(OutputKind kind, Feed feed, Out* out, SpatialChain* keep = nullptr) {
  SpatialChain local;
  SpatialChain& chain = keep ? *keep : local;
  static Feed f;
  f = feed;
  if (!chain.Configure(kind, kRate, 4096, &Pull, &f)) {
    std::printf("set-up for %s failed at %s (%d)\n", OutputKindName(kind), chain.failed(), int(chain.error()));
    return false;
  }
  return Render(chain, out);
}

// The renderer alone, without the chain's fold: the LFE on the built-in
// speakers.
OSStatus RawInput(void*, AudioUnitRenderActionFlags*, const AudioTimeStamp*, UInt32, UInt32 n, AudioBufferList* io) {
  static uint64_t t = 0;
  for (UInt32 b = 0; b < io->mNumberBuffers; b++) {
    auto* d = static_cast<float*>(io->mBuffers[b].mData);
    for (UInt32 i = 0; i < n; i++) d[i] = b == UInt32(kLFE) ? 0.5f * float(std::sin(2 * M_PI * 60 * double(t + i) / kRate)) : 0.0f;
  }
  t += n;
  return noErr;
}

double RawSpeakersLfe() {
  SpatialChain chain;
  Feed dummy;
  if (!chain.Configure(OutputKind::kBuiltInSpeakers, kRate, 4096, &Pull, &dummy)) return 0;
  // The chain's mixer pulled straight, past the chain's fold.
  AURenderCallbackStruct cb = {&RawInput, nullptr};
  AudioUnit m = chain.mixer();
  AudioUnitUninitialize(m);
  AudioUnitSetProperty(m, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &cb, sizeof cb);
  AudioUnitInitialize(m);
  std::vector<float> l(kSlice), r(kSlice);
  double s = 0;
  AudioTimeStamp ts{};
  ts.mFlags = kAudioTimeStampSampleTimeValid;
  for (int k = 0; k < 60; k++) {
    std::vector<uint8_t> mem(sizeof(AudioBufferList) + sizeof(AudioBuffer));
    auto* abl = reinterpret_cast<AudioBufferList*>(mem.data());
    abl->mNumberBuffers = 2;
    abl->mBuffers[0] = {1, UInt32(kSlice * sizeof(float)), l.data()};
    abl->mBuffers[1] = {1, UInt32(kSlice * sizeof(float)), r.data()};
    AudioUnitRenderActionFlags flags = 0;
    AudioUnitRender(m, &flags, &ts, 0, kSlice, abl);
    ts.mSampleTime += kSlice;
    if (k >= 20)
      for (UInt32 i = 0; i < kSlice; i++) s += double(l[i]) * l[i] + double(r[i]) * r[i];
  }
  return Db(std::sqrt(s / (80.0 * kSlice)));
}

}  // namespace

int main() {
  AudioComponentDescription d = {kAudioUnitType_Mixer, kAudioUnitSubType_SpatialMixer, kAudioUnitManufacturer_Apple,
                                 0, 0};
  if (!AudioComponentFindNext(nullptr, &d)) {
    std::printf("no spatial mixer on this host: skipped\n");
    return 77;
  }
  Out fl, fr, bl, br, lfe, spk_lfe, ext_fl, loud, straight, folded;
  SpatialChain hp;
  bool ok = Through(OutputKind::kHeadphones, {kFL}, &fl, &hp);
  Check(hp.head_tracking(), "headphones: head tracking asked for and taken");
  ok = ok && Through(OutputKind::kHeadphones, {kFR}, &fr) && Through(OutputKind::kHeadphones, {kBL}, &bl) &&
       Through(OutputKind::kHeadphones, {kBR}, &br) && Through(OutputKind::kHeadphones, {kLFE, 60}, &lfe);
  SpatialChain spk;
  ok = ok && Through(OutputKind::kBuiltInSpeakers, {kLFE, 60}, &spk_lfe, &spk);
  Check(!spk.head_tracking(), "built-in speakers: no head tracking");
  SpatialChain ext;
  ok = ok && Through(OutputKind::kExternalSpeakers, {kFL}, &ext_fl, &ext);
  Check(!ext.head_tracking(), "external speakers: no head tracking");
  ok = ok && Through(OutputKind::kHeadphones, {-1, 1000, 1.0f}, &loud);
  Feed pattern;
  pattern.pattern = true;
  ok = ok && Through(OutputKind::kStraight, pattern, &straight) && Through(OutputKind::kFoldedStereo, pattern, &folded);
  if (!ok) {
    std::printf("FAIL: the chain did not render\n");
    return 1;
  }
  const double raw_lfe = RawSpeakersLfe();
  std::printf("headphones: FL L %.1f R %.1f, FR L %.1f R %.1f, BL L %.1f R %.1f, BR L %.1f R %.1f, LFE %.1f dB; "
              "full scale peak %.3f; built-in speakers LFE raw %.1f, through the chain %.1f dB; external FL L %.1f "
              "R %.1f dB\n",
              Db(fl.Rms(0)), Db(fl.Rms(1)), Db(fr.Rms(0)), Db(fr.Rms(1)), Db(bl.Rms(0)), Db(bl.Rms(1)), Db(br.Rms(0)),
              Db(br.Rms(1)), Db(lfe.Rms(0)), loud.Peak(), raw_lfe, Db(spk_lfe.Rms(0)), Db(ext_fl.Rms(0)),
              Db(ext_fl.Rms(1)));
  Check(Db(fl.Rms(0)) > Db(fl.Rms(1)) + 3, "headphones: front left louder in the left ear");
  Check(Db(fr.Rms(1)) > Db(fr.Rms(0)) + 3, "headphones: front right louder in the right ear");
  Check(Db(bl.Rms(0)) > Db(bl.Rms(1)) + 3, "headphones: back left louder in the left ear");
  Check(Db(br.Rms(1)) > Db(br.Rms(0)) + 3, "headphones: back right louder in the right ear");
  double diff = 0, energy = 0;
  for (size_t i = 0; i < bl.ch[0].size(); i++)
    for (int c = 0; c < 2; c++) {
      const double e = bl.ch[size_t(c)][i] - fl.ch[size_t(c)][i];
      diff += e * e;
      energy += double(fl.ch[size_t(c)][i]) * fl.ch[size_t(c)][i];
    }
  Check(std::sqrt(diff / energy) > 0.3, "headphones: back left rendered unlike front left");
  Check(Db(lfe.Rms(0)) > -40, "headphones: the LFE heard without a fold");

  // The back's spectral cue: on headphones a back channel's 5 kHz against
  // its 500 Hz (both ears' energy) is well under a front channel's; on
  // external speakers (no cue) the two are alike.
  auto tilt = [](OutputKind kind, int channel, double* out) {
    Out hi, lo;
    if (!Through(kind, {channel, 5000}, &hi) || !Through(kind, {channel, 500}, &lo)) return false;
    *out = 10 * std::log10((hi.Rms(0) * hi.Rms(0) + hi.Rms(1) * hi.Rms(1)) /
                           (lo.Rms(0) * lo.Rms(0) + lo.Rms(1) * lo.Rms(1)));
    return true;
  };
  double hp_f = 0, hp_b = 0, hp_br = 0, ext_f = 0, ext_b = 0;
  if (!tilt(OutputKind::kHeadphones, kFL, &hp_f) || !tilt(OutputKind::kHeadphones, kBL, &hp_b) ||
      !tilt(OutputKind::kHeadphones, kBR, &hp_br) || !tilt(OutputKind::kExternalSpeakers, kFL, &ext_f) ||
      !tilt(OutputKind::kExternalSpeakers, kBL, &ext_b)) {
    std::printf("FAIL: the tilt tones did not render\n");
    return 1;
  }
  std::printf("5 kHz against 500 Hz: headphones FL %+.1f BL %+.1f BR %+.1f dB; external speakers FL %+.1f BL %+.1f dB\n",
              hp_f, hp_b, hp_br, ext_f, ext_b);
  Check(hp_b < hp_f - 6 && hp_br < hp_f - 6, "headphones: the back pair duller than the front by the back cue");
  Check(std::fabs(ext_b - ext_f) < 1, "external speakers: the back pair's spectrum as the front's (no cue)");

  // The room: an impulse in front left, the direct sound (2.5 ms from its
  // peak) against everything after it, both ears. Without a room the
  // renderer gives about 42 dB; with it about 11.5 dB (surround.h's Room).
  auto drr = [](OutputKind kind, double* out) {
    Feed f;
    f.channel = kFL;
    f.impulse = true;
    Out o;
    if (!Through(kind, f, &o)) return false;
    size_t peak = 0;
    double top = 0;
    for (size_t i = 0; i < o.ch[0].size(); i++) {
      const double e = double(o.ch[0][i]) * o.ch[0][i] + double(o.ch[1][i]) * o.ch[1][i];
      if (e > top) top = e, peak = i;
    }
    double direct = 0, rest = 0;
    for (size_t i = peak > 120 ? peak - 120 : 0; i < o.ch[0].size(); i++) {
      const double e = double(o.ch[0][i]) * o.ch[0][i] + double(o.ch[1][i]) * o.ch[1][i];
      (i < peak + 120 ? direct : rest) += e;
    }
    *out = 10 * std::log10(direct / (rest + 1e-30));
    return true;
  };
  double drr_hp = 0, drr_ext = 0;
  if (!drr(OutputKind::kHeadphones, &drr_hp) || !drr(OutputKind::kExternalSpeakers, &drr_ext)) {
    std::printf("FAIL: the impulse did not render\n");
    return 1;
  }
  std::printf("an impulse in front left, direct sound over the rest: headphones %.1f dB, external speakers %.1f dB\n",
              drr_hp, drr_ext);
  Check(drr_hp > 6 && drr_hp < 20, "headphones: the room heard after the direct sound");
  Check(drr_ext > 30, "external speakers: no room added");
  Check(raw_lfe < -100, "built-in speakers: the renderer itself drops the LFE");
  Check(Db(spk_lfe.Rms(0)) > -40 && Db(spk_lfe.Rms(1)) > -40, "built-in speakers: the LFE heard through the chain");
  Check(Db(ext_fl.Rms(0)) > -20 && Db(ext_fl.Rms(1)) < -60, "external speakers: front left on the left only");
  Check(loud.Peak() <= PeakLimiter::kThreshold + 1e-6f, "a full-scale 5.1 moment: no sample over the threshold");
  Check(hp.limited_frames() == 0, "a tone at half scale in one channel: the limiter leaves it alone");
  bool same = straight.ch.size() == 6;
  for (size_t c = 0; same && c < 6; c++)
    for (float x : straight.ch[c]) same &= x == 0.1f * float(c + 1);
  Check(same, "5.1 straight: the mix as it is, channel for channel");
  const float l = 0.1f + kFoldGain * (0.3f + 0.5f), r = 0.2f + kFoldGain * (0.3f + 0.6f);
  bool fold = folded.ch.size() == 2;
  for (size_t i = 0; fold && i < folded.ch[0].size(); i++)
    fold &= std::fabs(folded.ch[0][i] - l) < 1e-6f && std::fabs(folded.ch[1][i] - r) < 1e-6f;
  Check(fold, "folded stereo: centre and back pair at -3 dB into their sides, no LFE");

  // Headphones to the built-in speakers as it renders: no head tracking,
  // the LFE folded from then on.
  static Feed live;
  live = {kLFE, 60};
  SpatialChain change;
  Out before, after;
  if (!change.Configure(OutputKind::kHeadphones, kRate, 4096, &Pull, &live) || !Render(change, &before) ||
      !change.SetRenderedKind(OutputKind::kBuiltInSpeakers) || !Render(change, &after)) {
    std::printf("FAIL: the change of kind did not render\n");
    return 1;
  }
  Check(change.kind() == OutputKind::kBuiltInSpeakers && !change.head_tracking(),
        "changed to the built-in speakers: no head tracking");
  Check(Db(after.Rms(0)) > -40, "changed to the built-in speakers: the LFE folded and heard");

  AudioChannelLayout five{}, stereo{};
  five.mChannelLayoutTag = kAudioChannelLayoutTag_MPEG_5_1_A;
  stereo.mChannelLayoutTag = kAudioChannelLayoutTag_Stereo;
  const std::vector<uint32_t> fl5 = ChannelLabels(&five), fl2 = ChannelLabels(&stereo);
  Check(LayoutHasSurround(fl5.data(), int(fl5.size())), "a 5.1 tag spelt out has surround positions");
  Check(!LayoutHasSurround(fl2.data(), int(fl2.size())) && fl2.size() == 2, "stereo spelt out has none");
  if (g_failures) return 1;
  std::printf("ok\n");
  return 0;
}
