// SpatialOutput on macOS: the default output unit, fed by a SpatialChain
// (spatial_chain_mac.h: Apple's spatial mixer for headphones and stereo
// speakers, or the 5.1 straight for a device whose speaker layout has
// surround positions). spatial_output.h has the contract.
//
// The output unit's render callback is the one the device's real-time
// thread runs: it renders the chain (which pulls the mix), then dumps and
// mutes what the device is given; it takes no lock. Changes of the device
// (the default one, its data source, rate, channels or speaker layout)
// arrive on a serial queue of our own and rebuild the chain under mu_;
// Stop waits for any still on that queue, so none runs after it.

#include "runtime/audio/spatial_output.h"

#include <AudioToolbox/AudioToolbox.h>
#include <CoreAudio/CoreAudio.h>
#include <Block.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "runtime/audio/spatial_chain_mac.h"
#include "runtime/audio/surround.h"
#include "runtime/log.h"

namespace recomp::audio {
namespace {

// The largest render the output unit may ask of the chain: a device's
// largest buffer (4096 frames) at the lowest rate an output runs at
// (16 kHz, a Bluetooth headset's), converted to the chain's rate.
constexpr double kLargestDeviceBuffer = 4096, kLowestDeviceRate = 16000;
UInt32 MaxSlice(int rate) { return UInt32(std::ceil(kLargestDeviceBuffer * rate / kLowestDeviceRate)); }

template <typename T>
bool DeviceProperty(AudioObjectID obj, AudioObjectPropertySelector sel, AudioObjectPropertyScope scope, T* out) {
  AudioObjectPropertyAddress a = {sel, scope, kAudioObjectPropertyElementMain};
  UInt32 size = sizeof(T);
  return AudioObjectGetPropertyData(obj, &a, 0, nullptr, &size, out) == noErr;
}

AudioDeviceID DefaultDevice() {
  AudioDeviceID dev = kAudioObjectUnknown;
  DeviceProperty(kAudioObjectSystemObject, kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal,
                 &dev);
  return dev;
}

// What the choice of output needs, and what the log shows.
struct DeviceInfo {
  OutputDevice desc;
  std::string name;
  double rate = 0;
  int buffer_frames = 0;
  int latency_frames = 0;  // the device's own, its safety offset and its stream's
};

int OutputChannels(AudioDeviceID dev) {
  AudioObjectPropertyAddress a = {kAudioDevicePropertyStreamConfiguration, kAudioObjectPropertyScopeOutput,
                                  kAudioObjectPropertyElementMain};
  UInt32 size = 0;
  if (AudioObjectGetPropertyDataSize(dev, &a, 0, nullptr, &size) != noErr || !size) return 0;
  std::vector<uint8_t> buf(size);
  auto* list = reinterpret_cast<AudioBufferList*>(buf.data());
  if (AudioObjectGetPropertyData(dev, &a, 0, nullptr, &size, list) != noErr) return 0;
  int n = 0;
  for (UInt32 i = 0; i < list->mNumberBuffers; i++) n += int(list->mBuffers[i].mNumberChannels);
  return n;
}

// The device's speaker layout (Audio MIDI Setup's speaker configuration).
bool SurroundLayout(AudioDeviceID dev) {
  AudioObjectPropertyAddress a = {kAudioDevicePropertyPreferredChannelLayout, kAudioObjectPropertyScopeOutput,
                                  kAudioObjectPropertyElementMain};
  UInt32 size = 0;
  if (AudioObjectGetPropertyDataSize(dev, &a, 0, nullptr, &size) != noErr || size < sizeof(AudioChannelLayout))
    return false;
  std::vector<uint8_t> buf(size);
  auto* layout = reinterpret_cast<AudioChannelLayout*>(buf.data());
  if (AudioObjectGetPropertyData(dev, &a, 0, nullptr, &size, layout) != noErr) return false;
  const std::vector<uint32_t> labels = ChannelLabels(layout);
  return LayoutHasSurround(labels.data(), int(labels.size()));
}

DeviceInfo Describe(AudioDeviceID dev) {
  using T = OutputDevice::Transport;
  DeviceInfo info;
  UInt32 transport = 0;
  DeviceProperty(dev, kAudioDevicePropertyTransportType, kAudioObjectPropertyScopeGlobal, &transport);
  switch (transport) {
    case kAudioDeviceTransportTypeBuiltIn: info.desc.transport = T::kBuiltIn; break;
    case kAudioDeviceTransportTypeBluetooth:
    case kAudioDeviceTransportTypeBluetoothLE: info.desc.transport = T::kBluetooth; break;
    case kAudioDeviceTransportTypeUSB: info.desc.transport = T::kUsb; break;
    case kAudioDeviceTransportTypeHDMI: info.desc.transport = T::kHdmi; break;
    case kAudioDeviceTransportTypeDisplayPort: info.desc.transport = T::kDisplayPort; break;
    case kAudioDeviceTransportTypeAirPlay: info.desc.transport = T::kAirPlay; break;
    default: info.desc.transport = T::kOther; break;
  }
  info.desc.channels = OutputChannels(dev);
  info.desc.surround_layout = SurroundLayout(dev);
  UInt32 source = 0;
  if (DeviceProperty(dev, kAudioDevicePropertyDataSource, kAudioObjectPropertyScopeOutput, &source))
    info.desc.source = SourceOf(source);
  CFStringRef name = nullptr;
  if (DeviceProperty(dev, kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal, &name) && name) {
    char buf[256] = {};
    CFStringGetCString(name, buf, sizeof buf, kCFStringEncodingUTF8);
    info.name = buf;
    CFRelease(name);
  }
  Float64 rate = 0;
  if (DeviceProperty(dev, kAudioDevicePropertyNominalSampleRate, kAudioObjectPropertyScopeGlobal, &rate))
    info.rate = rate;
  UInt32 frames = 0;
  if (DeviceProperty(dev, kAudioDevicePropertyBufferFrameSize, kAudioObjectPropertyScopeOutput, &frames))
    info.buffer_frames = int(frames);
  UInt32 lat = 0, safety = 0;
  DeviceProperty(dev, kAudioDevicePropertyLatency, kAudioObjectPropertyScopeOutput, &lat);
  DeviceProperty(dev, kAudioDevicePropertySafetyOffset, kAudioObjectPropertyScopeOutput, &safety);
  info.latency_frames = int(lat + safety);
  AudioObjectPropertyAddress sa = {kAudioDevicePropertyStreams, kAudioObjectPropertyScopeOutput,
                                   kAudioObjectPropertyElementMain};
  UInt32 size = 0;
  if (AudioObjectGetPropertyDataSize(dev, &sa, 0, nullptr, &size) == noErr && size >= sizeof(AudioStreamID)) {
    std::vector<AudioStreamID> streams(size / sizeof(AudioStreamID));
    if (AudioObjectGetPropertyData(dev, &sa, 0, nullptr, &size, streams.data()) == noErr) {
      UInt32 sl = 0;
      if (DeviceProperty(streams[0], kAudioStreamPropertyLatency, kAudioObjectPropertyScopeGlobal, &sl))
        info.latency_frames += int(sl);
    }
  }
  return info;
}

FILE* DumpFile(const char* name) {
  const char* dir = std::getenv("RECOMP_AUDIO_DUMP");
  if (!dir || !*dir) return nullptr;
  return std::fopen((std::string(dir) + "/" + name).c_str(), "wb");
}

// The device's properties whose change re-chooses the output.
constexpr AudioObjectPropertySelector kDeviceWatched[] = {
    kAudioDevicePropertyDataSource, kAudioDevicePropertyNominalSampleRate, kAudioDevicePropertyStreamConfiguration,
    kAudioDevicePropertyPreferredChannelLayout};

class MacSpatialOutput final : public SpatialOutput {
 public:
  ~MacSpatialOutput() override {
    Stop();
    if (out_) AudioComponentInstanceDispose(out_);
  }

  bool Init() {
    AudioComponentDescription od = {kAudioUnitType_Output, kAudioUnitSubType_DefaultOutput,
                                    kAudioUnitManufacturer_Apple, 0, 0};
    AudioComponent oc = AudioComponentFindNext(nullptr, &od);
    if (!oc || AudioComponentInstanceNew(oc, &out_) != noErr) {
      out_ = nullptr;
      WarnF("audio: no output unit on this host\n");
      return false;
    }
    dev_ = DefaultDevice();
    info_ = Describe(dev_);
    rate_ = info_.rate > 0 ? int(info_.rate + 0.5) : 48000;
    max_slice_ = MaxSlice(rate_);
    return true;
  }

  int rate() const override { return rate_; }
  int device_frames() const override { return info_.buffer_frames > 0 ? info_.buffer_frames : 512; }
  int max_frames() const override { return int(max_slice_); }

  bool Start(Pull pull, Changed changed, void* ctx) override {
    std::lock_guard<std::mutex> l(mu_);
    pull_ = pull;
    changed_ = changed;
    ctx_ = ctx;
    mute_ = [] {
      const char* e = std::getenv("RECOMP_AUDIO_MUTE");
      return e && *e == '1';
    }();
    dump_ = DumpFile("spatial.f32");
    const OutputChoice choice = ChooseOutput(info_.desc);
    // Refused twice, the mixer opens the plain stereo device instead.
    if (!Configure(choice.kind) && !Configure(choice.kind)) return false;
    guessed_ = choice.guessed;
    queue_ = dispatch_queue_create("recomp-spatial-output", DISPATCH_QUEUE_SERIAL);
    alive_ = std::make_shared<std::atomic<bool>>(true);
    std::shared_ptr<std::atomic<bool>> alive = alive_;
    listener_ = Block_copy(^(UInt32, const AudioObjectPropertyAddress*) {
      if (alive->load()) OnChange();
    });
    AudioObjectPropertyAddress a = {kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal,
                                    kAudioObjectPropertyElementMain};
    AudioObjectAddPropertyListenerBlock(kAudioObjectSystemObject, &a, queue_, listener_);
    Watch(dev_, true);
    Log("opened");
    // Whether the personal profile is in use is known once it renders.
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 1500 * NSEC_PER_MSEC), queue_, ^{
      if (alive->load()) LogRunning();
    });
    return true;
  }

  void Stop() override {
    if (queue_) {
      AudioObjectPropertyAddress a = {kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal,
                                      kAudioObjectPropertyElementMain};
      AudioObjectRemovePropertyListenerBlock(kAudioObjectSystemObject, &a, queue_, listener_);
      AudioDeviceID watched;
      {
        std::lock_guard<std::mutex> l(mu_);
        watched = dev_;
      }
      Watch(watched, false);
      // A notification already on the queue runs to its end before this
      // returns; none after it does anything.
      std::shared_ptr<std::atomic<bool>> alive = alive_;
      dispatch_sync(queue_, ^{
        alive->store(false);
      });
      dispatch_release(queue_);
      queue_ = nullptr;
      Block_release(listener_);
      listener_ = nullptr;
    }
    std::lock_guard<std::mutex> l(mu_);
    Teardown();
    if (dump_) std::fclose(dump_);
    dump_ = nullptr;
  }

 private:
  // The chain and the output unit for `kind`, started. With mu_ held,
  // nothing running.
  bool Configure(OutputKind kind) {
    Teardown();
    auto fail = [&](const char* what, OSStatus e) {
      WarnF("audio: spatial output (%s): %s failed (%d)\n", OutputKindName(kind), what, int(e));
      Teardown();
      return false;
    };
    if (!chain_.Configure(kind, rate_, max_slice_, pull_, ctx_))
      return fail(chain_.failed(), chain_.error());
    UInt32 slice = max_slice_;
    AudioUnitSetProperty(out_, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0, &slice,
                         sizeof slice);
    const AudioStreamBasicDescription f = PlanarFloat(rate_, chain_.output_channels());
    AudioChannelLayout layout{};
    layout.mChannelLayoutTag = chain_.output_layout();
    if (OSStatus e = AudioUnitSetProperty(out_, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &f,
                                          sizeof f);
        e != noErr)
      return fail("output format", e);
    AudioUnitSetProperty(out_, kAudioUnitProperty_AudioChannelLayout, kAudioUnitScope_Input, 0, &layout,
                         sizeof layout);
    const AURenderCallbackStruct cb = {&MacSpatialOutput::OutputRender, this};
    if (OSStatus e = AudioUnitSetProperty(out_, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &cb,
                                          sizeof cb);
        e != noErr)
      return fail("output callback", e);
    if (OSStatus e = AudioUnitInitialize(out_); e != noErr) return fail("output init", e);
    out_ready_ = true;
    if (OSStatus e = AudioOutputUnitStart(out_); e != noErr) return fail("output start", e);
    running_ = true;
    return true;
  }

  void Teardown() {
    if (running_) AudioOutputUnitStop(out_);  // waits for a render in progress
    running_ = false;
    if (out_ready_) AudioUnitUninitialize(out_);
    out_ready_ = false;
    chain_.Teardown();
  }

  void Watch(AudioDeviceID dev, bool on) {
    if (dev == kAudioObjectUnknown || !queue_) return;
    for (AudioObjectPropertySelector sel : kDeviceWatched) {
      const AudioObjectPropertyScope scope = sel == kAudioDevicePropertyNominalSampleRate
                                                 ? kAudioObjectPropertyScopeGlobal
                                                 : kAudioObjectPropertyScopeOutput;
      AudioObjectPropertyAddress a = {sel, scope, kAudioObjectPropertyElementMain};
      if (on)
        AudioObjectAddPropertyListenerBlock(dev, &a, queue_, listener_);
      else
        AudioObjectRemovePropertyListenerBlock(dev, &a, queue_, listener_);
    }
  }

  void Log(const char* what) {
    const OutputKind k = chain_.kind();
    const double ms = 1000.0 / rate_;
    char rendering[256];
    if (OutputIsRendered(k))
      std::snprintf(rendering, sizeof rendering, "the spatial mixer for %s%s%s, personal profile %s, LFE %s; "
                    "mixer latency %.2f ms",
                    OutputKindName(k), guessed_ ? " (the system does not say what plays it: taken as headphones)" : "",
                    OutputTracksHead(k) ? (chain_.head_tracking() ? ", head tracking on" : ", head tracking refused")
                                        : "",
                    chain_.personal() ? "asked for" : "refused", OutputDropsLfe(k) ? "into the front pair" : "kept",
                    chain_.latency_ms());
    else
      std::snprintf(rendering, sizeof rendering, "%s", OutputKindName(k));
    WarnF("audio: spatial audio %s: '%s' (%d channels%s, %d Hz): %s; device buffer %d frames (%.1f ms), device "
          "latency %.1f ms\n",
          what, info_.name.c_str(), info_.desc.channels, info_.desc.surround_layout ? ", surround layout" : "",
          int(info_.rate), rendering, info_.buffer_frames, info_.buffer_frames * ms, info_.latency_frames * ms);
  }

  void LogRunning() {
    std::lock_guard<std::mutex> l(mu_);
    const OutputKind k = chain_.kind();
    if (!OutputIsRendered(k)) return;
    UInt32 used = 0, size = sizeof used;
    if (__builtin_available(macOS 14.0, *))
      AudioUnitGetProperty(chain_.mixer(), kAudioUnitProperty_SpatialMixerAnyInputIsUsingPersonalizedHRTF,
                           kAudioUnitScope_Global, 0, &used, &size);
    WarnF("audio: spatial audio running: %spersonal profile %s, %llu frames rendered\n",
          OutputTracksHead(k) ? (chain_.head_tracking() ? "head tracking on, " : "head tracking off, ") : "",
          used ? "in use" : "not in use", (unsigned long long)rendered_.load(std::memory_order_relaxed));
  }

  // On the queue: the default device changed, or one of its watched
  // properties did.
  void OnChange() {
    std::lock_guard<std::mutex> l(mu_);
    const AudioDeviceID dev = DefaultDevice();
    if (dev != dev_) {
      Watch(dev_, false);
      Watch(dev, true);
      dev_ = dev;
    }
    info_ = Describe(dev);
    const OutputChoice choice = ChooseOutput(info_.desc);
    const OutputKind was = chain_.kind();
    guessed_ = choice.guessed;
    if (choice.kind != was && !chain_.SetRenderedKind(choice.kind)) {
      // To or from a 5.1 device, or back from the folded stereo: rebuilt,
      // once more if it fails, then the mix folded to stereo here, so
      // that it keeps being taken (and the game's sound keeps its pace).
      bool ok = false;
      const OutputKind now = ConfigureWithFallback(choice.kind, [&](OutputKind k) { return Configure(k); }, &ok);
      if (!ok)
        WarnF("audio: spatial audio: no output for '%s'\n", info_.name.c_str());
      else if (now != choice.kind && !fallback_logged_) {
        fallback_logged_ = true;
        WarnF("audio: spatial audio: '%s' refused %s: the mix folded to stereo\n", info_.name.c_str(),
              OutputKindName(choice.kind));
      }
    }
    if (changed_) {
      const int frames = info_.buffer_frames > 0 && info_.rate > 0
                             ? int(double(info_.buffer_frames) * rate_ / info_.rate + 0.5)
                             : 512;
      changed_(ctx_, frames);
    }
    Log("now on");
  }

  // The device's real-time thread.
  static OSStatus OutputRender(void* self, AudioUnitRenderActionFlags* flags, const AudioTimeStamp* ts, UInt32,
                               UInt32 n, AudioBufferList* io) {
    auto* o = static_cast<MacSpatialOutput*>(self);
    o->chain_.Render(flags, ts, n, io);
    if (o->dump_ && io->mNumberBuffers == 2) {
      const auto* l = static_cast<const float*>(io->mBuffers[0].mData);
      const auto* r = static_cast<const float*>(io->mBuffers[1].mData);
      float buf[2 * 512];
      for (UInt32 done = 0; done < n;) {
        const UInt32 k = std::min<UInt32>(n - done, 512);
        for (UInt32 i = 0; i < k; i++) buf[2 * i] = l[done + i], buf[2 * i + 1] = r[done + i];
        std::fwrite(buf, sizeof(float), size_t(2) * k, o->dump_);
        done += k;
      }
    }
    o->rendered_.fetch_add(n, std::memory_order_relaxed);
    if (o->mute_)
      for (UInt32 b = 0; b < io->mNumberBuffers; b++) std::memset(io->mBuffers[b].mData, 0, io->mBuffers[b].mDataByteSize);
    return noErr;
  }

  AudioUnit out_ = nullptr;
  SpatialChain chain_;
  std::mutex mu_;
  AudioDeviceID dev_ = kAudioObjectUnknown;
  DeviceInfo info_;
  int rate_ = 48000;
  UInt32 max_slice_ = 4096;
  Pull pull_ = nullptr;
  Changed changed_ = nullptr;
  void* ctx_ = nullptr;
  bool mute_ = false, guessed_ = false, fallback_logged_ = false;
  bool running_ = false, out_ready_ = false;
  dispatch_queue_t queue_ = nullptr;
  AudioObjectPropertyListenerBlock listener_ = nullptr;
  std::shared_ptr<std::atomic<bool>> alive_;
  FILE* dump_ = nullptr;
  std::atomic<uint64_t> rendered_{0};
};

}  // namespace

std::unique_ptr<SpatialOutput> OpenSpatialOutput() {
  auto o = std::make_unique<MacSpatialOutput>();
  if (!o->Init()) return nullptr;
  return o;
}

}  // namespace recomp::audio
