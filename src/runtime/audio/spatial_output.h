#pragma once
// The 5.1 mix to the host's default output device through the host's own
// spatial renderer, where it has one (macOS: Apple's spatial mixer audio
// unit in front of the default output unit; spatial_chain_mac.h).
// Headphones hear it binaural, with head tracking where they have it and
// the listener's personal profile where one is set up; the Mac's speakers
// through their own virtualizer; a stereo pair of speakers panned; a
// device whose speaker layout has surround positions takes the 5.1 as it
// is (surround.h, ChooseOutput). A peak limiter keeps the renderer's
// output from clipping. The choice follows the default device as it
// changes; a device that refuses its kind twice gets the mix folded to
// stereo instead, so that the mix keeps being taken.
//
// The mixer (mixer.h) owns it: it pulls the mix on the device's real-time
// thread, which takes no lock and reads no guest state.
//
// RECOMP_AUDIO_MUTE=1 silences what the device is given (after the
// renderer, so its dump below is still the real output);
// RECOMP_AUDIO_DUMP=dir writes dir/spatial.f32, the stereo the device is
// given before the mute (raw float, interleaved, at rate()).

#include <memory>

namespace recomp::audio {

class SpatialOutput {
 public:
  // n frames of the 5.1 mix (interleaved, surround.h's order) at rate().
  using Pull = void (*)(void* ctx, float* frames, int n);
  // The default device changed: the frames it now asks for at a time, at
  // rate().
  using Changed = void (*)(void* ctx, int device_frames);

  virtual ~SpatialOutput() = default;
  // The rate the chain runs at: the default device's when it was opened
  // (a later device at another rate is converted to by the output unit).
  virtual int rate() const = 0;
  // The frames the device asks for at a time, at rate().
  virtual int device_frames() const = 0;
  // The most it may ask for at a time, at rate(), from any device the
  // default may become.
  virtual int max_frames() const = 0;
  // Sets the chain up for the default device and starts it. False (and
  // nothing running) if the host refuses.
  virtual bool Start(Pull pull, Changed changed, void* ctx) = 0;
  // Stops the device and the listening to device changes; no Pull after.
  virtual void Stop() = 0;
};

// The default output device's spatial output, opened (not started); null
// where the host has none or it cannot be set up.
std::unique_ptr<SpatialOutput> OpenSpatialOutput();

}  // namespace recomp::audio
