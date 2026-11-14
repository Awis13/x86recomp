// The 5.1 mix's pure parts (runtime/audio/surround.h): where each channel
// of a source lands in the mix, stereo into the front pair, the LFE fold,
// the speaker configuration DirectSound reports, and the way a device is
// rendered for; portable.

#include <cmath>
#include <cstdio>
#include <vector>

#include "runtime/audio/surround.h"

namespace {

using namespace recomp::audio;

int g_failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    g_failures++;
  }
}

bool Near(float a, float b) { return std::fabs(a - b) < 1e-6f; }

// A 5.1 source's channels each land in their own slot (a distinct value a
// channel, so a swap or a sum shows); a 7.1 source's side pair joins the
// back pair.
void Routing() {
  const float six[6] = {1, 2, 4, 8, 16, 32};
  float bed[6] = {};
  AddToBed(six, 6, bed);
  bool same = true;
  for (int c = 0; c < 6; c++) same &= bed[c] == six[c];
  Check(same, "a 5.1 source goes to the mix channel for channel");

  const float eight[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  float b8[6] = {};
  AddToBed(eight, 8, b8);
  Check(b8[kFL] == 1 && b8[kFR] == 2 && b8[kFC] == 4 && b8[kLFE] == 8, "7.1: the front, centre and LFE in place");
  Check(b8[kBL] == 16 + 64 && b8[kBR] == 32 + 128, "7.1: side left into back left, side right into back right");

  // Adds, does not overwrite: two sources in one frame.
  float twice[6] = {};
  AddToBed(six, 6, twice);
  AddToBed(six, 6, twice);
  Check(twice[kBR] == 64, "two sources add up in the mix");
}

// Stereo frames to the front pair only, frame by frame.
void Stereo() {
  const float st[4] = {0.25f, -0.5f, 0.75f, 1.0f};  // two frames
  std::vector<float> bed(12, 0.0f);
  bed[kBL] = 0.125f;  // something there already
  AddStereoToBed(st, bed.data(), 2);
  Check(bed[kFL] == 0.25f && bed[kFR] == -0.5f, "frame 0: left to FL, right to FR");
  Check(bed[6 + kFL] == 0.75f && bed[6 + kFR] == 1.0f, "frame 1: left to FL, right to FR");
  bool rest = bed[kFC] == 0 && bed[kLFE] == 0 && bed[kBL] == 0.125f && bed[kBR] == 0;
  for (int c : {kFC, kLFE, kBL, kBR}) rest &= bed[6 + c] == 0;
  Check(rest, "stereo leaves centre, LFE and the back pair as they were");
}

// The fold: LFE at -6 dB into both fronts, the LFE left silent.
void Fold() {
  float fl[3] = {0.1f, 0.0f, -0.2f}, fr[3] = {0.0f, 0.3f, 0.0f}, lfe[3] = {0.4f, -0.8f, 0.0f};
  FoldLfe(fl, fr, lfe, 3);
  Check(Near(fl[0], 0.3f) && Near(fl[1], -0.4f) && Near(fl[2], -0.2f), "LFE at half into the front left");
  Check(Near(fr[0], 0.2f) && Near(fr[1], -0.1f) && Near(fr[2], 0.0f), "LFE at half into the front right");
  Check(lfe[0] == 0 && lfe[1] == 0 && lfe[2] == 0, "the LFE is silent after the fold");
}

// DSSPEAKER_5POINT1 (6) for the 5.1 mix, DSSPEAKER_STEREO (4) otherwise.
void Speakers() {
  Check(SpeakerConfigFor(6) == 6, "a 5.1 mix reports DSSPEAKER_5POINT1");
  Check(SpeakerConfigFor(2) == 4, "a stereo mix reports DSSPEAKER_STEREO");
  Check(SpeakerConfigFor(1) == 4, "fewer channels report stereo");
}

void Devices() {
  using T = OutputDevice::Transport;
  using S = OutputDevice::Source;
  auto pick = [](T t, int ch, bool surround, S src) { return ChooseOutput({t, ch, surround, src}); };
  auto kind = [&](T t, int ch, bool surround, S src) { return pick(t, ch, surround, src).kind; };
  // What the system says plays it, first.
  Check(kind(T::kBuiltIn, 2, false, S::kHeadphones) == OutputKind::kHeadphones, "the headphone jack: headphones");
  Check(kind(T::kBuiltIn, 2, false, S::kInternalSpeakers) == OutputKind::kBuiltInSpeakers, "the Mac's own speakers");
  Check(kind(T::kBluetooth, 2, false, S::kExternalSpeakers) == OutputKind::kExternalSpeakers,
        "a Bluetooth device that says it is a speaker: speakers");
  Check(kind(T::kBluetooth, 2, false, S::kHeadphones) == OutputKind::kHeadphones,
        "a Bluetooth device that says it is headphones: headphones");
  Check(!pick(T::kBluetooth, 2, false, S::kHeadphones).guessed, "said by the system: not a guess");
  // Then by transport.
  Check(kind(T::kBuiltIn, 2, false, S::kUnknown) == OutputKind::kBuiltInSpeakers, "built-in without a source: speakers");
  Check(kind(T::kBluetooth, 2, false, S::kUnknown) == OutputKind::kHeadphones,
        "Bluetooth that does not say (AirPods): headphones");
  Check(pick(T::kBluetooth, 2, false, S::kUnknown).guessed, "Bluetooth that does not say: a guess");
  Check(kind(T::kBluetooth, 1, false, S::kUnknown) == OutputKind::kHeadphones, "a Bluetooth headset: headphones");
  Check(kind(T::kUsb, 2, false, S::kUnknown) == OutputKind::kHeadphones && pick(T::kUsb, 2, false, S::kUnknown).guessed,
        "USB that does not say: headphones, a guess");
  Check(kind(T::kHdmi, 2, false, S::kUnknown) == OutputKind::kExternalSpeakers, "a stereo TV on HDMI: speakers");
  Check(kind(T::kDisplayPort, 2, false, S::kUnknown) == OutputKind::kExternalSpeakers, "a display's speakers");
  Check(kind(T::kAirPlay, 2, false, S::kUnknown) == OutputKind::kExternalSpeakers, "AirPlay: speakers");
  Check(!pick(T::kHdmi, 2, false, S::kUnknown).guessed, "speakers by transport: not a guess");
  // 5.1 straight only for a speaker layout with surround positions.
  Check(kind(T::kHdmi, 6, true, S::kUnknown) == OutputKind::kStraight, "a 5.1 receiver takes the 5.1 as it is");
  Check(kind(T::kUsb, 8, true, S::kHeadphones) == OutputKind::kStraight, "a surround layout wins over the source");
  Check(kind(T::kHdmi, 8, false, S::kUnknown) == OutputKind::kExternalSpeakers,
        "eight channels without surround positions: its front pair as speakers");
  Check(kind(T::kUsb, 8, false, S::kUnknown) == OutputKind::kHeadphones,
        "an 8-channel interface without surround positions: by its front pair");
  Check(kind(T::kHdmi, 2, true, S::kUnknown) == OutputKind::kExternalSpeakers, "two channels are never 5.1");

  Check(SourceOf(0x6864706E) == S::kHeadphones, "'hdpn' is headphones");
  Check(SourceOf(0x6973706B) == S::kInternalSpeakers, "'ispk' is the internal speakers");
  Check(SourceOf(0x6573706B) == S::kExternalSpeakers, "'espk' is external speakers");
  Check(SourceOf(0x6C696E65) == S::kUnknown, "any other source says nothing");

  Check(OutputDropsLfe(OutputKind::kBuiltInSpeakers), "the built-in speakers' renderer gets the LFE folded");
  for (OutputKind k : {OutputKind::kHeadphones, OutputKind::kExternalSpeakers, OutputKind::kStraight,
                       OutputKind::kFoldedStereo})
    Check(!OutputDropsLfe(k), "only the built-in speakers get the LFE folded");
  Check(OutputTracksHead(OutputKind::kHeadphones), "head tracking on headphones");
  for (OutputKind k : {OutputKind::kBuiltInSpeakers, OutputKind::kExternalSpeakers, OutputKind::kStraight,
                       OutputKind::kFoldedStereo})
    Check(!OutputTracksHead(k), "no head tracking but on headphones");
  Check(!OutputIsRendered(OutputKind::kStraight) && !OutputIsRendered(OutputKind::kFoldedStereo),
        "5.1 straight and the folded stereo do without the renderer");
}

// Speaker layouts, by Core Audio's channel labels.
void Layouts() {
  const uint32_t five_one[] = {1, 2, 3, 4, 5, 6};          // L R C LFE Ls Rs
  const uint32_t stereo[] = {1, 2};
  const uint32_t rear[] = {1, 2, 33, 34};                  // L R Rls Rrs
  const uint32_t direct[] = {1, 2, 10, 11};                // L R Ls Rs (direct)
  const uint32_t one_side[] = {1, 2, 3, 5};                // a left surround alone
  const uint32_t discrete[] = {0x10000, 0x10001, 0x10002, 0x10003, 0x10004, 0x10005};
  const uint32_t unknown[] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
  Check(LayoutHasSurround(five_one, 6), "a 5.1 layout has surround positions");
  Check(LayoutHasSurround(rear, 4), "rear surrounds are surround positions");
  Check(LayoutHasSurround(direct, 4), "direct surrounds are surround positions");
  Check(!LayoutHasSurround(stereo, 2), "stereo has none");
  Check(!LayoutHasSurround(one_side, 4), "a left surround alone is not a pair");
  Check(!LayoutHasSurround(discrete, 6), "discrete channels are no positions");
  Check(!LayoutHasSurround(unknown, 6), "unknown channels are no positions");
}

// DirectSound's buffers as sources: kept for a 5.1 mix, folded for stereo.
void Sources() {
  Check(SourceChannels(6, 6) == 6, "a 5.1 buffer into a 5.1 mix as it is");
  Check(SourceChannels(8, 6) == 8, "a 7.1 buffer into a 5.1 mix as it is");
  Check(SourceChannels(6, 2) == 2, "a 5.1 buffer into a stereo mix folded to two");
  Check(SourceChannels(2, 6) == 2 && SourceChannels(1, 6) == 1, "stereo and mono stay so");
  const float f[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  float l = 0, r = 0;
  FoldToStereo(f, 6, &l, &r);
  Check(Near(l, 1 + kFoldGain * (4 + 16)) && Near(r, 2 + kFoldGain * (4 + 32)),
        "5.1 folded: centre and back pair at -3 dB, the LFE left out");
  FoldToStereo(f, 8, &l, &r);
  Check(Near(l, 1 + kFoldGain * (4 + 16 + 64)) && Near(r, 2 + kFoldGain * (4 + 32 + 128)),
        "7.1 folded: the sides too");
}

void Fallback() {
  std::vector<OutputKind> tried;
  bool ok = false;
  auto rec = [&](std::vector<bool> answers) {
    tried.clear();
    size_t i = 0;
    return [&tried, answers, i](OutputKind k) mutable {
      tried.push_back(k);
      return i < answers.size() && answers[i++];
    };
  };
  Check(ConfigureWithFallback(OutputKind::kHeadphones, rec({true}), &ok) == OutputKind::kHeadphones && ok &&
            tried.size() == 1,
        "set up at once");
  Check(ConfigureWithFallback(OutputKind::kHeadphones, rec({false, true}), &ok) == OutputKind::kHeadphones && ok &&
            tried.size() == 2,
        "refused once: set up the second time");
  Check(ConfigureWithFallback(OutputKind::kStraight, rec({false, false, true}), &ok) == OutputKind::kFoldedStereo &&
            ok && tried.size() == 3 && tried[2] == OutputKind::kFoldedStereo,
        "refused twice: the mix folded to stereo");
  ConfigureWithFallback(OutputKind::kStraight, rec({false, false, false}), &ok);
  Check(!ok, "nothing set up: said so");
}

void Limiter() {
  constexpr int n = 4800;
  PeakLimiter lim;
  lim.SetRate(48000);
  std::vector<float> l(n), r(n);
  for (int i = 0; i < n; i++) l[i] = r[i] = 0.5f * float(std::sin(i * 0.05));
  const std::vector<float> l0 = l;
  float* ch[2] = {l.data(), r.data()};
  lim.Process(ch, 2, n);
  Check(l == l0, "under the threshold nothing is touched");
  // A full-scale burst on one side, then quiet.
  for (int i = 0; i < n; i++) {
    l[i] = (i < 480 ? 1.5f : 0.3f) * float(std::sin(i * 0.05));
    r[i] = 0.3f * float(std::sin(i * 0.05));
  }
  std::vector<float> l1 = l, r1 = r;
  lim.Process(ch, 2, n);
  float peak = 0;
  for (int i = 0; i < n; i++) peak = std::max(peak, std::max(std::fabs(l[i]), std::fabs(r[i])));
  Check(peak <= PeakLimiter::kThreshold + 1e-6f, "every sample at or under the threshold");
  bool linked = true;
  for (int i = 0; i < n; i++)
    if (std::fabs(l1[i]) > 1e-3f && std::fabs(r1[i]) > 1e-3f) linked &= Near(l[i] / l1[i], r[i] / r1[i]);
  Check(linked, "both channels take the same gain");
  // 150 ms after the burst the gain is back.
  const int late = n - 1;  // 100 ms after a 10 ms burst: most of the way
  Check(std::fabs(r[late] / r1[late]) > 0.9f, "the gain comes back after the burst");
  Check(std::fabs(r[600] / r1[600]) < 1.0f, "right after the burst it is still down (no pumping per sample)");
}

}  // namespace

int main() {
  Routing();
  Stereo();
  Fold();
  Speakers();
  Devices();
  Layouts();
  Sources();
  Fallback();
  Limiter();
  if (g_failures) return 1;
  std::printf("ok\n");
  return 0;
}
