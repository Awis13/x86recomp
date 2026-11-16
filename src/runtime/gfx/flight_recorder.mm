// The flight recorder (flight_recorder.h): AVAssetWriter segments of HEVC
// from IOSurface-backed pixel buffers the backend draws into.

#include "runtime/gfx/frame_log.h"
#include "runtime/gfx/flight_recorder.h"

#if !__has_feature(objc_arc)
#error "flight_recorder.mm needs ARC (-fobjc-arc)"
#endif

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <QuartzCore/QuartzCore.h>

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <deque>
#include <mutex>
#include <vector>

namespace recomp::gfx {

struct FlightRecorder::Impl {
  struct Segment {
    AVAssetWriter* writer = nil;
    AVAssetWriterInput* input = nil;
    AVAssetWriterInputPixelBufferAdaptor* adaptor = nil;
    double t0 = 0, last = -1;
    int file = 0;
    bool open = false;
  };
  struct Done {
    std::string path;
    double t0 = 0, t1 = 0;
    int file = 0;
  };
  std::string live, marks;
  int streams = 1, w = 0, h = 0;
  CVPixelBufferPoolRef pool = nullptr;
  CVMetalTextureCacheRef cache = nullptr;
  dispatch_queue_t queue = nil;  // every writer's work, in order
  Segment seg[kMaxStreams];
  int next_file[kMaxStreams] = {};
  std::mutex done_mu;
  std::deque<Done> done[kMaxStreams];
  std::mutex notes_mu;
  std::deque<std::string> notes;

  std::string PathOf(int stream, int file) const {
    return live + "/stream" + std::to_string(stream) + "_" + std::to_string(file) + ".mov";
  }

  // On `queue`.
  void Open(int stream, double t) {
    Segment& s = seg[stream];
    s = Segment{};
    s.file = next_file[stream];
    next_file[stream] = (next_file[stream] + 1) % kSegmentFiles;
    const std::string path = PathOf(stream, s.file);
    {
      std::lock_guard<std::mutex> l(done_mu);
      auto& d = done[stream];
      d.erase(std::remove_if(d.begin(), d.end(), [&](const Done& x) { return x.file == s.file; }), d.end());
    }
    // (The ring's own file of a turn ago.)
    [[NSFileManager defaultManager] removeItemAtPath:@(path.c_str()) error:nil];
    NSError* err = nil;
    AVAssetWriter* writer = [AVAssetWriter assetWriterWithURL:[NSURL fileURLWithPath:@(path.c_str())]
                                                     fileType:AVFileTypeQuickTimeMovie
                                                        error:&err];
    if (!writer) return;
    NSDictionary* settings = @{
      AVVideoCodecKey : AVVideoCodecTypeHEVC,
      AVVideoWidthKey : @(w),
      AVVideoHeightKey : @(h),
      AVVideoCompressionPropertiesKey : @{
        AVVideoAverageBitRateKey : @(16000000),
        AVVideoExpectedSourceFrameRateKey : @(60),
        AVVideoMaxKeyFrameIntervalDurationKey : @(1.0),
      },
    };
    AVAssetWriterInput* input = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo
                                                                   outputSettings:settings];
    input.expectsMediaDataInRealTime = YES;
    AVAssetWriterInputPixelBufferAdaptor* adaptor =
        [AVAssetWriterInputPixelBufferAdaptor assetWriterInputPixelBufferAdaptorWithAssetWriterInput:input
                                                                           sourcePixelBufferAttributes:nil];
    if (![writer canAddInput:input]) return;
    [writer addInput:input];
    if (![writer startWriting]) return;
    [writer startSessionAtSourceTime:kCMTimeZero];
    s.writer = writer;
    s.input = input;
    s.adaptor = adaptor;
    s.t0 = t;
    s.open = true;
  }

  // On `queue`; `wait`: until the file is complete.
  void Close(int stream, bool wait) {
    Segment& s = seg[stream];
    if (!s.open) return;
    s.open = false;
    Done d{PathOf(stream, s.file), s.t0, s.t0 + std::max(s.last, 0.0), s.file};
    AVAssetWriter* writer = s.writer;
    [s.input markAsFinished];
    dispatch_semaphore_t finished = dispatch_semaphore_create(0);
    Impl* self = this;
    [writer finishWritingWithCompletionHandler:^{
      if (writer.status == AVAssetWriterStatusCompleted) {
        std::lock_guard<std::mutex> l(self->done_mu);
        self->done[stream].push_back(d);
      }
      dispatch_semaphore_signal(finished);
    }];
    s = Segment{};
    if (wait) dispatch_semaphore_wait(finished, dispatch_time(DISPATCH_TIME_NOW, int64_t(5 * NSEC_PER_SEC)));
  }
};

namespace {
std::mutex g_mu;
FlightRecorder* g_active = nullptr;

bool MakeDirs(const std::string& path) {
  return [[NSFileManager defaultManager] createDirectoryAtPath:@(path.c_str())
                                   withIntermediateDirectories:YES
                                                    attributes:nil
                                                         error:nil];
}
}  // namespace

FlightRecorder::FlightRecorder(std::unique_ptr<Impl> impl, int w, int h) : impl_(std::move(impl)), w_(w), h_(h) {}

std::unique_ptr<FlightRecorder> FlightRecorder::Make(id<MTLDevice> dev, const std::string& dir, int streams, int w,
                                                     int h) {
  if (!dev || dir.empty() || streams < 1 || streams > kMaxStreams || w < 16 || h < 16) return nullptr;
  auto impl = std::make_unique<Impl>();
  impl->live = dir + "/live";
  impl->marks = dir + "/marks";
  impl->streams = streams;
  impl->w = w & ~1;
  impl->h = h & ~1;
  if (!MakeDirs(impl->live) || !MakeDirs(impl->marks)) return nullptr;
  NSDictionary* attrs = @{
    (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA),
    (id)kCVPixelBufferWidthKey : @(impl->w),
    (id)kCVPixelBufferHeightKey : @(impl->h),
    (id)kCVPixelBufferIOSurfacePropertiesKey : @{},
    (id)kCVPixelBufferMetalCompatibilityKey : @YES,
  };
  if (CVPixelBufferPoolCreate(nullptr, nullptr, (__bridge CFDictionaryRef)attrs, &impl->pool) != kCVReturnSuccess)
    return nullptr;
  const NSDictionary* cache_attrs = @{(id)kCVMetalTextureCacheMaximumTextureAgeKey : @0};
  if (CVMetalTextureCacheCreate(nullptr, (__bridge CFDictionaryRef)cache_attrs, dev, nullptr, &impl->cache) !=
      kCVReturnSuccess) {
    CVPixelBufferPoolRelease(impl->pool);
    return nullptr;
  }
  impl->queue = dispatch_queue_create("recomp.flight_recorder", DISPATCH_QUEUE_SERIAL);
  const int iw = impl->w, ih = impl->h;
  std::unique_ptr<FlightRecorder> r(new FlightRecorder(std::move(impl), iw, ih));
  std::lock_guard<std::mutex> l(g_mu);
  g_active = r.get();
  std::fprintf(stderr, "gfx: flight recorder on: %d stream(s) of %dx%d into %s (mark: FlightMark, the key, both sticks)\n",
               streams, iw, ih, dir.c_str());
  return r;
}

FlightRecorder::~FlightRecorder() {
  {
    std::lock_guard<std::mutex> l(g_mu);
    if (g_active == this) g_active = nullptr;
  }
  Impl* m = impl_.get();
  dispatch_sync(m->queue, ^{
    for (int s = 0; s < m->streams; s++) m->Close(s, true);
  });
  if (m->cache) CFRelease(m->cache);
  if (m->pool) CVPixelBufferPoolRelease(m->pool);
}

bool FlightRecorder::NewFrame(CVPixelBufferRef* pb, CVMetalTextureRef* ref, id<MTLTexture>* tex) {
  Impl* m = impl_.get();
  *pb = nullptr;
  *ref = nullptr;
  *tex = nil;
  if (CVPixelBufferPoolCreatePixelBuffer(nullptr, m->pool, pb) != kCVReturnSuccess || !*pb) return false;
  if (CVMetalTextureCacheCreateTextureFromImage(nullptr, m->cache, *pb, nullptr, MTLPixelFormatBGRA8Unorm, size_t(m->w),
                                                size_t(m->h), 0, ref) != kCVReturnSuccess ||
      !*ref) {
    CVPixelBufferRelease(*pb);
    *pb = nullptr;
    return false;
  }
  *tex = CVMetalTextureGetTexture(*ref);
  if (!*tex) {
    CFRelease(*ref);
    CVPixelBufferRelease(*pb);
    *pb = nullptr;
    *ref = nullptr;
    return false;
  }
  return true;
}

void FlightRecorder::Append(int stream, CVPixelBufferRef pb, CVMetalTextureRef ref, double t) {
  Impl* m = impl_.get();
  // (The cache keeps every texture it wrapped, each counted in the
  // device's allocated memory, until it is flushed: a frame's at once, or
  // some 160 MB a second piled up and let go together.)
  if (ref) {
    CFRelease(ref);
    CVMetalTextureCacheFlush(m->cache, 0);
  }
  if (!pb) return;
  if (stream < 0 || stream >= m->streams) {
    CVPixelBufferRelease(pb);
    return;
  }
  dispatch_async(m->queue, ^{
    Impl::Segment& s = m->seg[stream];
    if (!s.open || t - s.t0 >= kSegmentSeconds) {
      m->Close(stream, false);
      m->Open(stream, t);
    }
    const double pts = t - s.t0;
    if (s.open && pts > s.last && s.input.readyForMoreMediaData &&
        [s.adaptor appendPixelBuffer:pb withPresentationTime:CMTimeMakeWithSeconds(pts, 60000)])
      s.last = pts;
    CVPixelBufferRelease(pb);
  });
}

void FlightRecorder::Note(uint64_t frame, double t, const std::string& members) {
  Impl* m = impl_.get();
  char head[64];
  std::snprintf(head, sizeof head, "{\"f\":%llu,\"t\":%.4f", (unsigned long long)frame, t);
  std::string line = head;
  if (!members.empty()) line += "," + members;
  line += "}";
  std::lock_guard<std::mutex> l(m->notes_mu);
  m->notes.push_back(std::move(line));
  while (m->notes.size() > 3000) m->notes.pop_front();
}

std::string FlightMark(const std::string& label) {
  FrameLogLine("e %.4f mark %s", FrameLogSeconds(), label.c_str());  // (the frame log's event)
  std::lock_guard<std::mutex> l(g_mu);
  if (!g_active) return "";
  FlightRecorder::Impl* m = g_active->impl_.get();
  char stamp[32];
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
  localtime_r(&now, &tm);
  std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &tm);
  std::string clean;
  for (char c : label)
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_')
      clean += c;
  const std::string dir = m->marks + "/" + stamp + (clean.empty() ? "" : "-" + clean);
  if (!MakeDirs(dir)) return "";
  const double t_now = CACurrentMediaTime();
  // Every stream's open segment finished, so that the last seconds are in
  // complete files (the next frame opens a new one).
  dispatch_sync(m->queue, ^{
    for (int s = 0; s < m->streams; s++) m->Close(s, true);
  });
  NSFileManager* fm = [NSFileManager defaultManager];
  const double keep = FlightRecorder::kSegmentSeconds * (FlightRecorder::kSegmentFiles - 1);
  for (int s = 0; s < m->streams; s++) {
    std::vector<FlightRecorder::Impl::Done> files;
    {
      std::lock_guard<std::mutex> dl(m->done_mu);
      for (const auto& d : m->done[s])
        if (d.t1 >= t_now - keep) files.push_back(d);
    }
    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) { return a.t0 < b.t0; });
    for (size_t k = 0; k < files.size(); k++) {
      const std::string to = dir + "/stream" + std::to_string(s) + "_" + std::to_string(k) + ".mov";
      [fm copyItemAtPath:@(files[k].path.c_str()) toPath:@(to.c_str()) error:nil];
    }
  }
  std::vector<std::string> lines;
  {
    std::lock_guard<std::mutex> nl(m->notes_mu);
    lines.assign(m->notes.begin(), m->notes.end());
  }
  if (FILE* f = std::fopen((dir + "/frames.jsonl").c_str(), "w")) {
    for (const std::string& line : lines) std::fprintf(f, "%s\n", line.c_str());
    std::fclose(f);
  }
  std::fprintf(stderr, "gfx: flight recorder: marked %s\n", dir.c_str());
  return dir;
}

bool FlightRecording() {
  std::lock_guard<std::mutex> l(g_mu);
  return g_active != nullptr;
}

}  // namespace recomp::gfx
