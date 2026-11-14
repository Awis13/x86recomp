#include "runtime/audio/spatial_chain_mac.h"

#include <algorithm>
#include <cstring>

namespace recomp::audio {

AudioStreamBasicDescription PlanarFloat(double rate, int channels) {
  AudioStreamBasicDescription d{};
  d.mSampleRate = rate;
  d.mFormatID = kAudioFormatLinearPCM;
  d.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved;
  d.mBytesPerPacket = d.mBytesPerFrame = sizeof(float);
  d.mFramesPerPacket = 1;
  d.mChannelsPerFrame = UInt32(channels);
  d.mBitsPerChannel = 32;
  return d;
}

UInt32 SpatialMixerOutputType(OutputKind k) {
  switch (k) {
    case OutputKind::kBuiltInSpeakers: return kSpatialMixerOutputType_BuiltInSpeakers;
    case OutputKind::kExternalSpeakers: return kSpatialMixerOutputType_ExternalSpeakers;
    default: return kSpatialMixerOutputType_Headphones;
  }
}

std::vector<uint32_t> ChannelLabels(const AudioChannelLayout* layout) {
  std::vector<uint32_t> out;
  if (!layout) return out;
  if (layout->mChannelLayoutTag == kAudioChannelLayoutTag_UseChannelDescriptions) {
    for (UInt32 i = 0; i < layout->mNumberChannelDescriptions; i++)
      out.push_back(layout->mChannelDescriptions[i].mChannelLabel);
    return out;
  }
  // A tag or a bitmap: Core Audio spells it out.
  const bool bitmap = layout->mChannelLayoutTag == kAudioChannelLayoutTag_UseChannelBitmap;
  const AudioFormatPropertyID prop =
      bitmap ? kAudioFormatProperty_ChannelLayoutForBitmap : kAudioFormatProperty_ChannelLayoutForTag;
  const UInt32 spec = bitmap ? UInt32(layout->mChannelBitmap) : layout->mChannelLayoutTag;
  UInt32 size = 0;
  if (AudioFormatGetPropertyInfo(prop, sizeof spec, &spec, &size) != noErr || size < sizeof(AudioChannelLayout))
    return out;
  std::vector<uint8_t> buf(size);
  auto* full = reinterpret_cast<AudioChannelLayout*>(buf.data());
  if (AudioFormatGetProperty(prop, sizeof spec, &spec, &size, full) != noErr) return out;
  for (UInt32 i = 0; i < full->mNumberChannelDescriptions; i++) out.push_back(full->mChannelDescriptions[i].mChannelLabel);
  return out;
}

SpatialChain::~SpatialChain() {
  Teardown();
  if (mixer_) AudioComponentInstanceDispose(mixer_);
}

bool SpatialChain::Fail(const char* what, OSStatus e) {
  failed_ = what;
  error_ = e;
  return false;
}

bool SpatialChain::SetHeadTracking(bool on) {
  if (__builtin_available(macOS 12.3, *)) {
    const UInt32 v = on ? 1 : 0;
    return AudioUnitSetProperty(mixer_, kAudioUnitProperty_SpatialMixerEnableHeadTracking, kAudioUnitScope_Global, 0,
                                &v, sizeof v) == noErr &&
           on;
  }
  return false;
}

bool SpatialChain::Configure(OutputKind kind, double rate, UInt32 max_frames, Pull pull, void* ctx) {
  Teardown();
  failed_ = nullptr;
  error_ = noErr;
  kind_.store(kind, std::memory_order_relaxed);
  pull_ = pull;
  ctx_ = ctx;
  max_frames_ = max_frames;
  inter_.assign(size_t(max_frames) * kBedChannels, 0.0f);
  limiter_.SetRate(rate);
  back_cue_.SetRate(rate);
  head_tracking_ = personal_ = false;
  latency_ms_ = 0;
  if (!OutputIsRendered(kind)) return true;  // no renderer: Fill alone
  if (!mixer_) {
    AudioComponentDescription d = {kAudioUnitType_Mixer, kAudioUnitSubType_SpatialMixer, kAudioUnitManufacturer_Apple,
                                   0, 0};
    AudioComponent comp = AudioComponentFindNext(nullptr, &d);
    if (!comp) return Fail("no spatial mixer", -1);
    if (const OSStatus e = AudioComponentInstanceNew(comp, &mixer_); e != noErr) {
      mixer_ = nullptr;
      return Fail("spatial mixer", e);
    }
  }
  auto set = [&](AudioUnitPropertyID id, AudioUnitScope scope, const void* v, UInt32 size, const char* what) {
    if (failed_) return;
    if (const OSStatus e = AudioUnitSetProperty(mixer_, id, scope, 0, v, size); e != noErr) Fail(what, e);
  };
  const AudioStreamBasicDescription bed = PlanarFloat(rate, kBedChannels), stereo = PlanarFloat(rate, 2);
  AudioChannelLayout bed_layout{}, stereo_layout{};
  bed_layout.mChannelLayoutTag = kAudioChannelLayoutTag_MPEG_5_1_A;  // L R C LFE Ls Rs
  stereo_layout.mChannelLayoutTag = kAudioChannelLayoutTag_Stereo;
  const UInt32 one = 1, algo = kSpatializationAlgorithm_UseOutputType, mode = kSpatialMixerSourceMode_AmbienceBed,
               type = SpatialMixerOutputType(kind);
  const AURenderCallbackStruct cb = {&SpatialChain::MixerInput, this};
  set(kAudioUnitProperty_ElementCount, kAudioUnitScope_Input, &one, sizeof one, "mixer inputs");
  set(kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, &max_frames, sizeof max_frames, "mixer slice");
  set(kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, &bed, sizeof bed, "mixer input format");
  set(kAudioUnitProperty_AudioChannelLayout, kAudioUnitScope_Input, &bed_layout, sizeof bed_layout,
      "mixer input layout");
  set(kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, &stereo, sizeof stereo, "mixer output format");
  set(kAudioUnitProperty_SpatializationAlgorithm, kAudioUnitScope_Input, &algo, sizeof algo, "algorithm");
  set(kAudioUnitProperty_SpatialMixerSourceMode, kAudioUnitScope_Input, &mode, sizeof mode, "ambience bed");
  set(kAudioUnitProperty_SpatialMixerOutputType, kAudioUnitScope_Global, &type, sizeof type, "output type");
  set(kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, &cb, sizeof cb, "mixer callback");
  if (failed_) return false;
  AudioUnitSetProperty(mixer_, kAudioUnitProperty_AudioChannelLayout, kAudioUnitScope_Output, 0, &stereo_layout,
                       sizeof stereo_layout);
  head_tracking_ = SetHeadTracking(OutputTracksHead(kind));
  if (__builtin_available(macOS 13.0, *)) {
    const UInt32 hrtf = kSpatialMixerPersonalizedHRTFMode_Auto;
    personal_ = AudioUnitSetProperty(mixer_, kAudioUnitProperty_SpatialMixerPersonalizedHRTFMode,
                                     kAudioUnitScope_Global, 0, &hrtf, sizeof hrtf) == noErr;
  }
  if (const OSStatus e = AudioUnitInitialize(mixer_); e != noErr) return Fail("mixer start", e);
  mixer_ready_ = true;
  Float64 lat = 0;
  UInt32 size = sizeof lat;
  AudioUnitGetProperty(mixer_, kAudioUnitProperty_Latency, kAudioUnitScope_Global, 0, &lat, &size);
  latency_ms_ = lat * 1000.0;
  return true;
}

void SpatialChain::Teardown() {
  if (mixer_ready_) AudioUnitUninitialize(mixer_);
  mixer_ready_ = false;
}

bool SpatialChain::SetRenderedKind(OutputKind kind) {
  if (!mixer_ready_ || !OutputIsRendered(kind) || !OutputIsRendered(this->kind())) return false;
  const UInt32 type = SpatialMixerOutputType(kind);
  if (AudioUnitSetProperty(mixer_, kAudioUnitProperty_SpatialMixerOutputType, kAudioUnitScope_Global, 0, &type,
                           sizeof type) != noErr)
    return false;
  head_tracking_ = SetHeadTracking(OutputTracksHead(kind));
  kind_.store(kind, std::memory_order_relaxed);
  return true;
}

// n frames of the mix into io: the 5.1 bed (6 planar buffers) with the
// back pair's spectral cue on headphones and the LFE folded where the
// renderer drops it, or for kFoldedStereo two buffers of
// the mix folded to stereo.
void SpatialChain::Fill(AudioBufferList* io, UInt32 n) {
  n = std::min(n, max_frames_);
  pull_(ctx_, inter_.data(), int(n));
  const OutputKind k = kind();
  if (k == OutputKind::kFoldedStereo) {
    if (io->mNumberBuffers < 2) return;
    auto* l = static_cast<float*>(io->mBuffers[0].mData);
    auto* r = static_cast<float*>(io->mBuffers[1].mData);
    for (UInt32 i = 0; i < n; i++) FoldToStereo(&inter_[size_t(i) * kBedChannels], kBedChannels, &l[i], &r[i]);
    return;
  }
  for (UInt32 b = 0; b < io->mNumberBuffers && b < UInt32(kBedChannels); b++) {
    auto* dst = static_cast<float*>(io->mBuffers[b].mData);
    for (UInt32 i = 0; i < n; i++) dst[i] = inter_[size_t(i) * kBedChannels + b];
  }
  if (io->mNumberBuffers >= UInt32(kBedChannels) && OutputCuesBack(k))
    back_cue_.Process(static_cast<float*>(io->mBuffers[kBL].mData), static_cast<float*>(io->mBuffers[kBR].mData),
                      int(n));
  if (io->mNumberBuffers >= UInt32(kBedChannels) && OutputDropsLfe(k))
    FoldLfe(static_cast<float*>(io->mBuffers[kFL].mData), static_cast<float*>(io->mBuffers[kFR].mData),
            static_cast<float*>(io->mBuffers[kLFE].mData), int(n));
}

void SpatialChain::Limit(AudioBufferList* io, UInt32 n) {
  if (io->mNumberBuffers != 2) return;
  float* ch[2] = {static_cast<float*>(io->mBuffers[0].mData), static_cast<float*>(io->mBuffers[1].mData)};
  limiter_.Process(ch, 2, int(std::min(n, max_frames_)));
  uint64_t limited = 0;
  float gain = 1.0f;
  limiter_.TakeStats(&limited, &gain);
  if (limited) {
    limited_.fetch_add(limited, std::memory_order_relaxed);
    if (gain < lowest_gain_.load(std::memory_order_relaxed)) lowest_gain_.store(gain, std::memory_order_relaxed);
  }
}

OSStatus SpatialChain::MixerInput(void* self, AudioUnitRenderActionFlags*, const AudioTimeStamp*, UInt32, UInt32 n,
                                  AudioBufferList* io) {
  static_cast<SpatialChain*>(self)->Fill(io, n);
  return noErr;
}

OSStatus SpatialChain::Render(AudioUnitRenderActionFlags* flags, const AudioTimeStamp* ts, UInt32 n,
                              AudioBufferList* io) {
  const OutputKind k = kind();
  if (k == OutputKind::kStraight) {
    Fill(io, n);
    return noErr;
  }
  if (k == OutputKind::kFoldedStereo) {
    Fill(io, n);
  } else {
    const OSStatus e = AudioUnitRender(mixer_, flags, ts, 0, n, io);
    if (e != noErr) {
      for (UInt32 b = 0; b < io->mNumberBuffers; b++) std::memset(io->mBuffers[b].mData, 0, io->mBuffers[b].mDataByteSize);
      return e;
    }
  }
  Limit(io, n);
  return noErr;
}

}  // namespace recomp::audio
