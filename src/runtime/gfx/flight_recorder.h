// A flight recorder of what the player sees: the shown frames (and other
// textures a backend adds as further streams, a traced shadow mask, say)
// encoded as they are drawn, by the system's hardware video encoder, into
// a ring of short segments on disk, with a line of numbers per frame (the
// camera, timings); on a mark the last seconds of all of it are kept in a
// folder of their own, to be looked at frame by frame afterwards.
//
// RECOMP_RECORD_DIR=<dir> turns it on (RECOMP_RECORD=0 off all the same):
// <dir>/live holds the ring (kSegmentFiles files per stream, reused in
// turn), <dir>/marks/<time>-<label>/ each mark's copy:
// stream<N>_<k>.mov (oldest first) and frames.jsonl (one JSON object per
// presented frame: "f" its number, "t" seconds, then what the backend
// noted). A mark comes from FlightMark (a game's control command), the
// key RECOMP_RECORD_KEY names (default F12; the key is not passed on) or a
// gamepad's two sticks pressed together.

#pragma once

#include <string>

namespace recomp::gfx {

// Keeps the last seconds of every stream and the frames' notes in a new
// folder under <dir>/marks named after the time and `label` (letters,
// digits, '-' and '_' kept). Its path, or "" (no recorder running, or the
// folder could not be made).
std::string FlightMark(const std::string& label);

// Whether a recorder is running.
bool FlightRecording();

}  // namespace recomp::gfx

#if defined(__OBJC__)
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>

#include <memory>

namespace recomp::gfx {

// The backend's side (Objective-C++).
class FlightRecorder {
 public:
  static constexpr double kSegmentSeconds = 5.0;
  static constexpr int kSegmentFiles = 7;  // the ring: about 30-35 s kept
  static constexpr int kMaxStreams = 4;
  // A recorder for `streams` streams of w x h frames into `dir` (made if
  // missing); null if it cannot start. One at a time (FlightMark's).
  static std::unique_ptr<FlightRecorder> Make(id<MTLDevice> dev, const std::string& dir, int streams, int w, int h);
  ~FlightRecorder();
  int Width() const { return w_; }
  int Height() const { return h_; }
  // A frame buffer to draw a stream's next frame into (BGRA8, Width() x
  // Height()): its texture, the buffer and the texture's wrapper retained
  // for Append. False if none.
  bool NewFrame(CVPixelBufferRef* pb, CVMetalTextureRef* ref, id<MTLTexture>* tex);
  // The frame drawn into `pb` (NewFrame's; both released here) for
  // `stream`, at `t` seconds (CACurrentMediaTime). Any thread, once the GPU
  // wrote it.
  void Append(int stream, CVPixelBufferRef pb, CVMetalTextureRef ref, double t);
  // A frame's line of numbers (JSON members without braces, after "f" and
  // "t"), kept in a ring of the last ~40 s of frames.
  void Note(uint64_t frame, double t, const std::string& members);

 private:
  struct Impl;
  explicit FlightRecorder(std::unique_ptr<Impl> impl, int w, int h);
  std::unique_ptr<Impl> impl_;
  int w_, h_;
  friend std::string FlightMark(const std::string& label);
};

}  // namespace recomp::gfx
#endif
