#import <CommonCrypto/CommonDigest.h>
#include <TargetConditionals.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <dispatch/dispatch.h>
#include <sys/file.h>
#include <sys/stat.h>
#include "runtime/gfx/metal_cache.h"
#include "runtime/gfx/metal_preparation.h"
#include "runtime/gfx/metal_prelude.h"
#include "runtime/gfx/metal_internal.h"
#include "runtime/gfx/metal_smaa.h"
#include "runtime/gfx/shader_compile.h"
#include "runtime/hlsl/hlsl.h"

#ifndef RECOMP_METAL_BUILD_ID
#define RECOMP_METAL_BUILD_ID "development"
#endif

namespace recomp::gfx {
namespace {
namespace fs = std::filesystem;
std::string directory;
fs::path artifact_directory;
NSDictionary* artifact_provenance;
std::mutex cache_mutex;
volatile sig_atomic_t cancelled = 0;
std::atomic<bool> stopping_libraries{false};
std::map<std::string, dispatch_group_t> requested_pipelines;
size_t next_pipeline_lane = 0;  // protected by cache_mutex

dispatch_queue_t PipelineLane(size_t lane) {
  // Actual recorded draws need these states. Bound concurrency independently
  // of the host's core count and the background archive/library writers.
  static dispatch_queue_t queues[2] = {
      dispatch_queue_create("recomp.gfx.pipeline.0", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0)),
      dispatch_queue_create("recomp.gfx.pipeline.1", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0))};
  return queues[lane];
}

std::string PipelineStateKey(MTLRenderPipelineDescriptor* p);

void Cancel(int) { cancelled = 1; }
NSString* Text(const std::string& s) { return @(s.c_str()); }
std::string Bytes(NSData* d) { return d ? std::string(static_cast<const char*>(d.bytes), d.length) : std::string(); }
std::string Hash(const std::string& s) {
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(s.data(), CC_LONG(s.size()), digest);
  std::string h;
  for (unsigned char b : digest) { char x[3]; std::snprintf(x, sizeof x, "%02x", b); h += x; }
  return h;
}
void RequireSpace(const fs::path& location) {
  fs::path existing = location;
  while (!existing.empty() && !fs::exists(existing)) existing = existing.parent_path();
  // Leave a reserve for the OS, Metal's own compiler cache, and four in-flight
  // entries. This is checked before each compilation and each published write.
  constexpr uintmax_t reserve = uintmax_t(1) << 30;
  if (existing.empty() || fs::space(existing).available < reserve)
    throw std::runtime_error("Less than 1 GiB is free on the cache volume. Free disk space or choose a cache folder on another volume, then rerun preparation. Completed entries are kept.");
}
std::string Read(const fs::path& p) {
  // libc++ file streams can split even istream::read into small buffered reads
  // and read ahead when seeking. Use the opened file's size and direct reads.
  int fd = open(p.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error("Cannot read " + p.string() + ". Check the game installation and permissions.");
  struct Close { int fd; ~Close() { close(fd); } } closer{fd};
  struct stat info;
  if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0)
    throw std::runtime_error("Cannot size " + p.string());
  std::string bytes(size_t(info.st_size), '\0');
  size_t offset = 0;
  while (offset < bytes.size()) {
    ssize_t n = read(fd, bytes.data() + offset, bytes.size() - offset);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) throw std::runtime_error("Cannot completely read " + p.string());
    offset += size_t(n);
  }
  return bytes;
}
// URL-backed libraries retain open files on cold Metal loads. Keep the binary
// in owned dispatch data so a prepared catalog cannot exhaust the game's file
// descriptors. The default destructor copies the bytes before NSData is released.
id<MTLLibrary> LoadLibrary(id<MTLDevice> dev, const fs::path& path, NSError** error = nullptr) {
  NSData* bytes = [NSData dataWithContentsOfFile:Text(path.string()) options:0 error:error];
  if (!bytes) return nil;
  dispatch_data_t data = dispatch_data_create(bytes.bytes, bytes.length, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  return [dev newLibraryWithData:data error:error];
}
void Atomic(const fs::path& p, const std::string& data) {
  RequireSpace(p.parent_path());
  NSError* error = nil;
  NSData* d = [NSData dataWithBytes:data.data() length:data.size()];
  if (![d writeToURL:[NSURL fileURLWithPath:Text(p.string())] options:NSDataWritingAtomic error:&error])
    throw std::runtime_error("Cannot write " + p.string() + ": " + error.localizedDescription.UTF8String);
}
void JsonFile(const fs::path& p, NSDictionary* object, bool state = false) {
  NSData* d = [NSJSONSerialization dataWithJSONObject:object options:NSJSONWritingSortedKeys error:nil];
  if (state) {
    NSError* error = nil;
    if (![d writeToURL:[NSURL fileURLWithPath:Text(p.string())] options:NSDataWritingAtomic error:&error])
      throw std::runtime_error("Cannot save shader preparation status: " + std::string(error.localizedDescription.UTF8String));
  } else Atomic(p, Bytes(d));
}
NSDictionary* JsonRead(const fs::path& p) {
  NSData* d = [NSData dataWithContentsOfFile:Text(p.string())];
  id v = d ? [NSJSONSerialization JSONObjectWithData:d options:0 error:nil] : nil;
  return [v isKindOfClass:NSDictionary.class] ? v : @{};
}
void Emit(NSDictionary* object) {
  NSData* d = [NSJSONSerialization dataWithJSONObject:object options:NSJSONWritingSortedKeys error:nil];
  std::printf("%.*s\n", int(d.length), static_cast<const char*>(d.bytes));
  std::fflush(stdout);
}
fs::path Owned(const std::string& root, const MetalPreparationProfile& profile) {
  NSString* base = root.empty() ? [NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES).firstObject stringByAppendingPathComponent:Text(profile.cache_namespace)] : Text(root);
  return fs::path(base.fileSystemRepresentation) / "metal";
}
const std::string& SelectedToolchain();
void ConfigureCache(const std::string& content_dir, const std::string& root, id<MTLDevice> dev,
                    const MetalPreparationProfile& profile) {
  if (!dev) throw std::runtime_error("No compatible Metal device is available.");
  if (!profile.content_identity) throw std::runtime_error("Missing shader content identity provider");
  std::string content = profile.content_identity(content_dir);
  std::string settings;
  NSDictionary* env = NSProcessInfo.processInfo.environment;
  for (NSString* key in [[env allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
    // Instrumentation does not affect the generated code or pipeline descriptors.
    if ([key hasPrefix:@"RECOMP_"] && ![key hasPrefix:@"RECOMP_METAL_"] &&
        ![key containsString:@"PERF"] && ![key containsString:@"DUMP"] && ![key containsString:@"LABEL"] &&
        ![key containsString:@"REPORT"])
      settings += std::string(key.UTF8String) + "=" + [env[key] UTF8String] + "\n";
  }
  std::string compatibility = std::string(RECOMP_METAL_BUILD_ID) + "\nprofile:" + profile.identity;
  compatibility += NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String;
  compatibility += std::to_string(dev.registryID) + std::string(dev.name.UTF8String) + settings;
  const auto& toolchain = SelectedToolchain();
  // Local binaries must be isolated too: checking only the shared store would
  // reuse a previous compiler's offline libraries before consulting provenance.
  compatibility += "\nrecipient-toolchain-v1:" + (toolchain.empty() ? "unavailable" : toolchain);
  directory = (Owned(root, profile) / "v1" / Hash(content + compatibility)).string();
  // Keep the conservative game/OS/device/settings restrictions. Only source
  // build identity is omitted; translation and preparation keep that identity.
  // Bump this policy when keys, serialized descriptors, or compiler flags change.
  artifact_provenance = @{@"policy": @"compiled-artifacts-v2;metal2.1;fastMath=1",
    @"content": Text(Hash(content)), @"os": NSProcessInfo.processInfo.operatingSystemVersionString,
    @"device": Text(std::to_string(dev.registryID) + ":" + dev.name.UTF8String),
    @"settings": Text(settings), @"toolchain": Text(toolchain.empty() ? "unavailable" : toolchain)};
  NSData* provenance = [NSJSONSerialization dataWithJSONObject:artifact_provenance options:NSJSONWritingSortedKeys error:nil];
  // The unavailable namespace can share source-IR archives, but never receives
  // offline libraries. It remains separate when tools become available again.
  artifact_directory = Owned(root, profile) / "artifacts" / "v1" / Hash(Bytes(provenance));
}
void Report(const char* type, const char* result, const std::chrono::steady_clock::time_point& start,
            const char* phase = "runtime") {
  const char* enabled = std::getenv("RECOMP_SHADER_CACHE_REPORT");
  if (enabled && *enabled == '1') {
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    uint64_t thread = 0; pthread_threadid_np(nullptr, &thread);
    std::fprintf(stderr, "gfx: shader-cache %s %s %.3f ms phase=%s thread=%llu\n",
                 type, result, ms, phase, (unsigned long long)thread);
  }
}
struct ReadyPipeline {
  id<MTLRenderPipelineState> state;
  MetalPipelineSource source;
};
std::map<std::string, ReadyPipeline> ready_pipelines;
std::map<id<MTLRenderPipelineState>, MetalPipelineSource> pipeline_origins;
std::set<std::string> pending_writes, recorded_descriptors, attempted_libraries;
struct LibraryWrite { std::string key; std::function<bool(std::string*)> build; };
std::deque<LibraryWrite> library_writes;
bool library_write_active = false;
dispatch_group_t library_group = dispatch_group_create();
std::map<std::string, MTLRenderPipelineDescriptor*> source_pipelines;
std::map<std::string, id<MTLFunction>> source_compute;
// Metal may produce different IR for source-built and offline-loaded libraries.
// Keep their device archives separate; source text identity alone is insufficient.
std::set<id<MTLLibrary>> runtime_libraries;
std::set<id<MTLFunction>> runtime_functions;
bool RuntimeFunction(id<MTLFunction> f) {
  std::lock_guard<std::mutex> lock(cache_mutex);
  return runtime_functions.contains(f);
}
dispatch_queue_t WriteQueue() {
  static dispatch_queue_t queue = dispatch_queue_create("recomp.gfx.archive-writes",
      dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_BACKGROUND, 0));
  // Construct both demand lanes before registering the exit callback too.
  // Function-local ARC queue owners must outlive that callback.
  PipelineLane(0); PipelineLane(1);
  // Also cover normal process exit paths that bypass the renderer destructor.
  // Register after queue construction so the queue and cache maps are still live.
  static const bool drain_at_exit = [] {
    std::atexit([] { MetalDrainPipelineWrites(true); });
    return true;
  }();
  return queue;
}
std::string LibraryKey(NSString* source, MTLCompileOptions* options) {
  std::string s = source.UTF8String;
  s += "\nmetal2.1;fastMath=1;";
  NSDictionary* macros = options.preprocessorMacros ?: @{};
  for (NSString* key in [[macros allKeys] sortedArrayUsingSelector:@selector(compare:)])
    s += std::string(key.UTF8String) + "=" + [[macros[key] description] UTF8String] + ";";
  return Hash(s);
}
fs::path LibraryPath(const std::string& key) { return fs::path(directory) / "libraries" / (key + ".metallib"); }
bool ValidBinary(const fs::path& path) {
  try { return Read(path.string() + ".sha256") == Hash(Read(path)); }
  catch (...) { return false; }
}
void PublishBinary(const fs::path& temporary, const fs::path& path) {
  std::string digest = Hash(Read(temporary));
  fs::rename(temporary, path);
  Atomic(path.string() + ".sha256", digest);
}
std::array<fs::path, 2> ArtifactCandidates(const fs::path& local) {
  return {local, artifact_directory.empty() ? fs::path()
    : artifact_directory / local.parent_path().filename() / local.filename()};
}
bool HasArtifact(const fs::path& local) {
  for (const auto& candidate : ArtifactCandidates(local)) {
    std::error_code ec;
    if (!candidate.empty() && fs::is_regular_file(candidate, ec)) return true;
  }
  return false;
}
// Called only by the worker or serialized write queue, under metal.lock.
// Published files are immutable: replacements rename a new inode, so hard links
// in older scopes keep their bytes and scope-relative manifest entries intact.
void MaterializeArtifact(const fs::path& source, const fs::path& target) {
  if (source == target) return;
  if (ValidBinary(target) && Read(target.string() + ".sha256") == Read(source.string() + ".sha256")) return;
  fs::create_directories(target.parent_path());
  for (const char* suffix : {"", ".sha256"}) {
    const fs::path from = source.string() + suffix, to = target.string() + suffix;
    const fs::path temporary = to.string() + "." + NSUUID.UUID.UUIDString.UTF8String;
    try {
      std::error_code ec;
      fs::create_hard_link(from, temporary, ec);
      if (ec) { RequireSpace(to.parent_path()); fs::copy_file(from, temporary); }
      fs::rename(temporary, to);
    } catch (...) { std::error_code ec; fs::remove(temporary, ec); throw; }
  }
}
void ShareArtifact(const fs::path& local) {
  if (artifact_directory.empty()) return;
  try {
    fs::create_directories(artifact_directory);
    if (!fs::exists(artifact_directory / "compatibility.json"))
      JsonFile(artifact_directory / "compatibility.json", artifact_provenance);
    MaterializeArtifact(local, ArtifactCandidates(local)[1]);
  } catch (const std::exception& e) {
    // The local verified artifact remains usable even if sharing fails.
    std::fprintf(stderr, "gfx: sharing compiled shader artifact: %s\n", e.what());
  }
}
// A checksum is only an integrity check. The caller must load the library or
// require a real Metal archive hit before a candidate can be used or shared.
template<class Load>
auto LoadArtifact(const fs::path& local, Load load, bool materialize = false) -> decltype(load(local)) {
  for (const auto& candidate : ArtifactCandidates(local)) {
    if (candidate.empty() || !ValidBinary(candidate)) continue;
    if (auto result = load(candidate)) {
      if (materialize) { MaterializeArtifact(candidate, local); ShareArtifact(local); }
      return result;
    }
  }
  return nil;
}
void PublishArtifact(const fs::path& temporary, const fs::path& local) {
  PublishBinary(temporary, local);
  ShareArtifact(local);
}
std::string FunctionIdentity(id<MTLFunction> f) {
  return f ? std::string(f.label.UTF8String ?: "") : std::string();
}
std::string PipelineKey(MTLRenderPipelineDescriptor* p) {
  std::string v = FunctionIdentity(p.vertexFunction), f = FunctionIdentity(p.fragmentFunction);
  if (v.empty() || f.empty()) return {};
  std::string key = v + ":" + f;
  for (int i = 0; i < 4; i++) {
    auto a = p.colorAttachments[i];
    for (NSUInteger n : {NSUInteger(a.pixelFormat), NSUInteger(a.blendingEnabled), NSUInteger(a.sourceRGBBlendFactor),
                        NSUInteger(a.destinationRGBBlendFactor), NSUInteger(a.rgbBlendOperation), NSUInteger(a.sourceAlphaBlendFactor),
                        NSUInteger(a.destinationAlphaBlendFactor), NSUInteger(a.alphaBlendOperation), NSUInteger(a.writeMask)})
      key += ":" + std::to_string(n);
  }
  key += ":" + std::to_string(p.depthAttachmentPixelFormat) + ":" + std::to_string(p.stencilAttachmentPixelFormat);
  key += ":" + std::to_string(p.rasterSampleCount);
  return Hash(key);
}
unsigned RenderSourceMask(MTLRenderPipelineDescriptor* p) {
  return unsigned(RuntimeFunction(p.vertexFunction)) | (unsigned(RuntimeFunction(p.fragmentFunction)) << 1);
}
std::string RenderArchiveKey(const std::string& key, unsigned source_mask) {
  // A restart can have only one of the two libraries published. Source/source,
  // source/offline, offline/source and offline/offline can all have different IR.
  return source_mask ? Hash(key + ":render-source=" + std::to_string(source_mask)) : key;
}
NSDictionary* PipelineRecord(MTLRenderPipelineDescriptor* p) {
  NSMutableArray* attachments = [NSMutableArray new];
  for (int i = 0; i < 4; i++) {
    auto a = p.colorAttachments[i];
    [attachments addObject:@[@(a.pixelFormat), @(a.blendingEnabled), @(a.sourceRGBBlendFactor),
      @(a.destinationRGBBlendFactor), @(a.rgbBlendOperation), @(a.sourceAlphaBlendFactor),
      @(a.destinationAlphaBlendFactor), @(a.alphaBlendOperation), @(a.writeMask)]];
  }
  return @{@"vertex": Text(FunctionIdentity(p.vertexFunction)), @"fragment": Text(FunctionIdentity(p.fragmentFunction)),
    @"attachments": attachments, @"depth": @(p.depthAttachmentPixelFormat), @"stencil": @(p.stencilAttachmentPixelFormat),
    @"samples": @(p.rasterSampleCount)};
}
MTLRenderPipelineDescriptor* RecordedDescriptor(NSDictionary* r, const std::map<std::string, id<MTLLibrary>>& libraries,
                                               std::string* error = nullptr) {
  auto invalid = [&](const std::string& message) -> MTLRenderPipelineDescriptor* {
    if (error) *error = message;
    return nil;
  };
  auto function = [&](NSString* stage) -> id<MTLFunction> {
    NSString* identity = r[stage];
    if (![identity isKindOfClass:NSString.class]) {
      if (error) *error = std::string(stage.UTF8String) + " shader identity must be a library/function string";
      return nil;
    }
    NSArray* parts = [identity componentsSeparatedByString:@"/"];
    if (parts.count != 2 || ![parts[0] length] || ![parts[1] length]) {
      if (error) *error = std::string(stage.UTF8String) + " shader identity is invalid: " + identity.UTF8String;
      return nil;
    }
    auto lib = libraries.find([parts[0] UTF8String]);
    if (lib == libraries.end()) {
      if (error) *error = std::string(stage.UTF8String) + " shader " + identity.UTF8String +
        " requires an unavailable library; expected a valid libraries/" + [parts[0] UTF8String] +
        ".metallib or sources/" + [parts[0] UTF8String] + ".json in this cache scope";
      return nil;
    }
    auto result = MetalCachedFunction(lib->second, parts[1]);
    if (!result && error) *error = std::string(stage.UTF8String) + " function " + [parts[1] UTF8String] +
      " does not exist in compiled library " + [parts[0] UTF8String];
    return result;
  };
  auto p = [MTLRenderPipelineDescriptor new];
  p.vertexFunction = function(@"vertex");
  if (!p.vertexFunction) return nil;
  p.fragmentFunction = function(@"fragment");
  if (!p.fragmentFunction) return nil;
  NSArray* attachments = r[@"attachments"];
  if (![attachments isKindOfClass:NSArray.class] || attachments.count != 4)
    return invalid("attachments must contain exactly four color attachment records");
  for (int i = 0; i < 4; i++) {
    NSArray* values = attachments[i];
    if (![values isKindOfClass:NSArray.class] || values.count != 9)
      return invalid("attachments[" + std::to_string(i) + "] must contain nine numeric values");
    for (id value in values) if (![value isKindOfClass:NSNumber.class])
      return invalid("attachments[" + std::to_string(i) + "] contains a nonnumeric value");
    auto a = p.colorAttachments[i];
    a.pixelFormat = MTLPixelFormat([values[0] unsignedIntegerValue]); a.blendingEnabled = [values[1] boolValue];
    a.sourceRGBBlendFactor = MTLBlendFactor([values[2] unsignedIntegerValue]);
    a.destinationRGBBlendFactor = MTLBlendFactor([values[3] unsignedIntegerValue]);
    a.rgbBlendOperation = MTLBlendOperation([values[4] unsignedIntegerValue]);
    a.sourceAlphaBlendFactor = MTLBlendFactor([values[5] unsignedIntegerValue]);
    a.destinationAlphaBlendFactor = MTLBlendFactor([values[6] unsignedIntegerValue]);
    a.alphaBlendOperation = MTLBlendOperation([values[7] unsignedIntegerValue]);
    a.writeMask = MTLColorWriteMask([values[8] unsignedIntegerValue]);
  }
  for (NSString* field in @[@"depth", @"stencil", @"samples"])
    if (![r[field] isKindOfClass:NSNumber.class]) return invalid(std::string(field.UTF8String) + " must be numeric");
  if ([r[@"samples"] integerValue] <= 0) return invalid("samples must be greater than zero");
  p.depthAttachmentPixelFormat = MTLPixelFormat([r[@"depth"] unsignedIntegerValue]);
  p.stencilAttachmentPixelFormat = MTLPixelFormat([r[@"stencil"] unsignedIntegerValue]);
  p.rasterSampleCount = [r[@"samples"] unsignedIntegerValue];
  return p;
}
void QueueArchiveWrite(std::string key, const char* kind, std::function<bool(std::string*)> build, std::function<void()> next = {});
void RememberDescriptor(MTLRenderPipelineDescriptor* p) {
  const std::string key = PipelineKey(p);
  if (directory.empty() || key.empty()) return;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (!recorded_descriptors.insert(key).second) return;
  }
  // Own metadata and paths across the queue boundary, including ready states
  // that are actually requested by drawing. No disk writes on the encoder.
  const fs::path path = fs::path(directory) / "descriptors" / (key + ".json");
  NSDictionary* record = PipelineRecord(p);
  QueueArchiveWrite("descriptor:" + key, "descriptor", [path, record](std::string*) {
    fs::create_directories(path.parent_path());
    if (!fs::exists(path)) JsonFile(path, record);
    return true;
  });
}
std::map<std::string, id<MTLLibrary>> loaded_libraries, offline_libraries;
// Runtime publication uses this same compiler on the background write queue.
// Bound each tool invocation and cancel runtime tools during renderer shutdown.
bool Tool(NSArray<NSString*>* args, std::string* log, bool runtime = false, bool stdout_only = false) {
  NSTask* t = [NSTask new];
  t.executableURL = [NSURL fileURLWithPath:@"/usr/bin/xcrun"];
  t.arguments = args;
  if (runtime) t.qualityOfService = NSQualityOfServiceBackground;
  // A file avoids pipe deadlock on compiler diagnostics.
  NSString* tmp = [NSTemporaryDirectory() stringByAppendingPathComponent:NSUUID.UUID.UUIDString];
  [NSFileManager.defaultManager createFileAtPath:tmp contents:nil attributes:nil];
  NSFileHandle* output = [NSFileHandle fileHandleForWritingAtPath:tmp];
  t.standardOutput = output;
  t.standardError = stdout_only ? NSFileHandle.fileHandleWithNullDevice : output;
  // Wake as soon as the compiler exits. Polling added up to 25 ms to each
  // of the two short-lived tools per library, even when compilation was done.
  dispatch_semaphore_t exited = dispatch_semaphore_create(0);
  t.terminationHandler = ^(NSTask*) { dispatch_semaphore_signal(exited); };
  NSError* e = nil;
  bool ok = [t launchAndReturnError:&e];
  if (ok) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(runtime ? 15 : 120);
    while (dispatch_semaphore_wait(exited, dispatch_time(DISPATCH_TIME_NOW, 25 * NSEC_PER_MSEC))) {
      if (cancelled || (runtime && stopping_libraries) || std::chrono::steady_clock::now() >= deadline) {
        [t terminate];
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (t.running) kill(t.processIdentifier, SIGKILL);
        *log = "Metal compiler cancelled or timed out";
        [t waitUntilExit]; [output closeFile];
        [NSFileManager.defaultManager removeItemAtPath:tmp error:nil];
        return false;
      }
    }
    ok = t.terminationStatus == 0 && !cancelled;
  }
  [output closeFile];
  *log = e ? e.localizedDescription.UTF8String : Bytes([NSData dataWithContentsOfFile:tmp]);
  [NSFileManager.defaultManager removeItemAtPath:tmp error:nil];
  return ok;
}
const std::string& SelectedToolchain() {
  // Probe once during cache configuration, before drawing or publication starts.
  // Builder recipients need their selected tools, independent of the kit maker.
  static const std::string selected = [] {
    std::string identity, output;
    for (NSArray* probe in @[@[@"-sdk", @"macosx", @"metal", @"--version"],
        @[@"-sdk", @"macosx", @"--show-sdk-path"], @[@"-sdk", @"macosx", @"--show-sdk-build-version"],
        @[@"-sdk", @"macosx", @"--find", @"metallib"]]) {
      if (!Tool(probe, &output, true, true) || output.empty()) return std::string();
      identity += output;
    }
    return Hash(identity);
  }();
  return selected;
}
bool OfflineToolchainAvailable(std::string* log) {
  if (!SelectedToolchain().empty()) return true;
  *log = "Metal compiler or SDK is unavailable. Select a working Xcode installation and install the Metal Toolchain in Xcode Settings > Components, or run xcodebuild -downloadComponent MetalToolchain. Relaunch to retry. Runtime rendering remains enabled.";
  return false;
}
id<MTLLibrary> CompileLibrary(id<MTLDevice> dev, NSString* source, MTLCompileOptions* options, std::string* log, bool runtime = false) {
  std::string key = LibraryKey(source, options);
  fs::path path = LibraryPath(key);
  NSError* error = nil;
  if (auto library = LoadArtifact(path, [dev](const fs::path& candidate) {
      return LoadLibrary(dev, candidate);
    }, true)) { library.label = Text(key); return library; }
  if (!OfflineToolchainAvailable(log)) return nil;
  RequireSpace(path.parent_path());
  fs::create_directories(path.parent_path());
  fs::path temp = path.parent_path() / (key + "." + NSUUID.UUID.UUIDString.UTF8String);
  struct Temporary {
    fs::path path;
    ~Temporary() { for (const char* suffix : {".metal", ".air", ".metallib"}) {
      std::error_code ec; fs::remove(path.string() + suffix, ec);
    } }
  } cleanup{temp};
  Atomic(temp.string() + ".metal", source.UTF8String);
  NSMutableArray* args = [NSMutableArray arrayWithArray:@[@"-sdk", @"macosx", @"metal", @"-std=macos-metal2.1", @"-ffast-math", @"-c",
      Text(temp.string() + ".metal"), @"-o", Text(temp.string() + ".air")]];
  for (NSString* macro in options.preprocessorMacros)
    [args addObject:[NSString stringWithFormat:@"-D%@=%@", macro, options.preprocessorMacros[macro]]];
  bool ok = Tool(args, log, runtime);
  if (ok) ok = Tool(@[@"-sdk", @"macosx", @"metallib", Text(temp.string() + ".air"), @"-o", Text(temp.string() + ".metallib")], log, runtime);
  id<MTLLibrary> library = ok ? LoadLibrary(dev, temp.string() + ".metallib", &error) : nil;
  if (library) {
    PublishArtifact(temp.string() + ".metallib", path);
    library.label = Text(key);
  } else if (error) *log = error.localizedDescription.UTF8String;
  return library;
}
MTLCompileOptions* Options(bool lean, bool full) {
  MTLCompileOptions* o = [MTLCompileOptions new];
  o.languageVersion = MTLLanguageVersion2_1;
  o.fastMathEnabled = YES;
  if (lean) o.preprocessorMacros = @{@"R_ALPHA_TEST": full ? @"true" : @"false"};
  return o;
}
// Function identity survives changes in process-local shader ids.
id<MTLFunction> Function(id<MTLLibrary> lib, NSString* name) {
  id<MTLFunction> f = [lib newFunctionWithName:name];
  if (f) {
    f.label = [NSString stringWithFormat:@"%@/%@", lib.label, name];
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (runtime_libraries.contains(lib)) runtime_functions.insert(f);
  }
  return f;
}
fs::path ArchivePath(const std::string& key, bool runtime_ir = false) {
  return fs::path(directory) / (runtime_ir ? "runtime-pipelines" : "pipelines") / (key + ".metallib");
}
id<MTLBinaryArchive> Archive(id<MTLDevice> dev, const fs::path& path) {
  std::error_code ec;
  if (!fs::is_regular_file(path, ec)) return nil;
  MTLBinaryArchiveDescriptor* d = [MTLBinaryArchiveDescriptor new];
  d.url = [NSURL fileURLWithPath:Text(path.string())];
  return [dev newBinaryArchiveWithDescriptor:d error:nil];
}
id<MTLRenderPipelineState> LoadRenderArtifact(id<MTLDevice> dev, MTLRenderPipelineDescriptor* p,
                                             const fs::path& local, bool materialize = false, const char* phase = nullptr,
                                             std::string* validated_digest = nullptr) {
  auto state = LoadArtifact(local, [dev, p, phase, validated_digest](const fs::path& candidate) -> id<MTLRenderPipelineState> {
    std::string digest;
    if (validated_digest) {
      try { digest = Read(candidate.string() + ".sha256"); }
      catch (...) { return nil; }
    }
    id<MTLBinaryArchive> a = Archive(dev, candidate);
    if (!a) return nil;
    p.binaryArchives = @[a];
    const auto started = std::chrono::steady_clock::now();
    auto state = [dev newRenderPipelineStateWithDescriptor:p options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:nil];
    if (phase) Report("driver", state ? "hit" : "rejected", started, phase);
    if (state && validated_digest) *validated_digest = std::move(digest);
    return state;
  }, materialize);
  if (!state) p.binaryArchives = @[];
  return state;
}
id<MTLComputePipelineState> LoadComputeArtifact(id<MTLDevice> dev, MTLComputePipelineDescriptor* p,
                                               const fs::path& local, bool materialize = false) {
  auto state = LoadArtifact(local, [dev, p](const fs::path& candidate) -> id<MTLComputePipelineState> {
    id<MTLBinaryArchive> a = Archive(dev, candidate);
    if (!a) return nil;
    p.binaryArchives = @[a];
    return [dev newComputePipelineStateWithDescriptor:p options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:nil];
  }, materialize);
  if (!state) p.binaryArchives = @[];
  return state;
}
bool BuildPipeline(id<MTLDevice> dev, MTLRenderPipelineDescriptor* p, std::string* log, bool runtime_ir = false) {
  const std::string key = PipelineKey(p);
  if (key.empty()) { *log = "Missing shader identity"; return false; }
  fs::path path = ArchivePath(RenderArchiveKey(key, runtime_ir ? RenderSourceMask(p) : 0), runtime_ir);
  NSError* e = nil;
  if (LoadRenderArtifact(dev, p, path, true, "persist")) return true;
  RequireSpace(path.parent_path());
  id<MTLBinaryArchive> a = [dev newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&e];
  if (!a || ![a addRenderPipelineFunctionsWithDescriptor:p error:&e]) { *log = e.localizedDescription.UTF8String; return false; }
  fs::create_directories(path.parent_path());
  fs::path tmp = path.string() + "." + NSUUID.UUID.UUIDString.UTF8String;
  if (![a serializeToURL:[NSURL fileURLWithPath:Text(tmp.string())] error:&e]) { *log = e.localizedDescription.UTF8String; return false; }
  // Verify the binary before publishing it. Existing completed work is untouched.
  id<MTLBinaryArchive> saved = Archive(dev, tmp);
  p.binaryArchives = saved ? @[saved] : @[];
  bool valid = saved && [dev newRenderPipelineStateWithDescriptor:p options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&e];
  if (valid) PublishArtifact(tmp, path);
  else { fs::remove(tmp); *log = e.localizedDescription.UTF8String ?: "Archive verification failed"; }
  return valid;
}
std::string ComputeKey(id<MTLFunction> f) { return f && !FunctionIdentity(f).empty() ? Hash("compute:" + FunctionIdentity(f)) : std::string(); }
bool BuildCompute(id<MTLDevice> dev, id<MTLFunction> f, std::string* log, bool runtime_ir = false) {
  auto p = [MTLComputePipelineDescriptor new]; p.computeFunction = f;
  fs::path path = ArchivePath(ComputeKey(f), runtime_ir);
  NSError* error = nil;
  if (LoadComputeArtifact(dev, p, path, true)) return true;
  RequireSpace(path.parent_path());
  id<MTLBinaryArchive> a = [dev newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&error];
  if (!a || ![a addComputePipelineFunctionsWithDescriptor:p error:&error]) { *log = error.localizedDescription.UTF8String; return false; }
  fs::create_directories(path.parent_path());
  fs::path temp = path.string() + "." + NSUUID.UUID.UUIDString.UTF8String;
  if (![a serializeToURL:[NSURL fileURLWithPath:Text(temp.string())] error:&error]) { *log = error.localizedDescription.UTF8String; return false; }
  id<MTLBinaryArchive> saved = Archive(dev, temp); p.binaryArchives = saved ? @[saved] : @[];
  if (!saved || ![dev newComputePipelineStateWithDescriptor:p options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&error]) {
    fs::remove(temp); *log = error.localizedDescription.UTF8String ?: "Compute archive verification failed"; return false;
  }
  PublishArtifact(temp, path); return true;
}

// A reference parameter would remain a reference inside an Objective-C block.
// Own the key so a queued write cannot outlive its caller's temporary string.
void QueueArchiveWrite(std::string key, const char* kind, std::function<bool(std::string*)> build, std::function<void()> next) {
  if (directory.empty() || key.empty()) return;
  const bool shutdown_write = stopping_libraries.load();
  const bool archive_write = std::strcmp(kind, "pipeline") == 0 || std::strcmp(kind, "compute") == 0;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (!pending_writes.insert(key).second) return;
  }
  // Workers and Clean own the same lock. Never delay rendering for that lock.
  const auto owned = fs::path(directory).parent_path().parent_path();
  dispatch_block_t write = ^{
    @autoreleasepool {
      int fd = -1;
      const char* result = "failed";
      std::string message;
      try {
        if (fs::is_symlink(owned)) throw std::runtime_error("Owned shader cache is a symbolic link");
        fs::create_directories(owned.parent_path());
        fd = open((owned.string() + ".lock").c_str(), O_CREAT | O_RDWR | O_NOFOLLOW, 0600);
        if (fd < 0 || flock(fd, LOCK_EX | LOCK_NB) != 0) result = "busy";
        else if (build(&message)) result = "saved";
      } catch (const std::exception& e) { message = e.what(); }
      if (std::strcmp(kind, "library") == 0 && stopping_libraries && std::strcmp(result, "busy") != 0) result = "deferred";
      if (fd >= 0) { flock(fd, LOCK_UN); close(fd); }
      if (std::strcmp(result, "failed") == 0)
        std::fprintf(stderr, "gfx: saving runtime shader cache: %s\n", message.c_str());
      if (const char* report = std::getenv("RECOMP_SHADER_CACHE_REPORT"); report && *report == '1' && std::strcmp(kind, "descriptor") != 0)
        std::fprintf(stderr, "gfx: shader-persist %s %s shutdown=%d requested_qos=%u\n",
                     kind, result, shutdown_write, static_cast<unsigned>(qos_class_self()));
      { std::lock_guard<std::mutex> lock(cache_mutex); pending_writes.erase(key); }
      if (next) next();
    }
  };
  // Required reusable archives must make progress while drawing has priority.
  // Utility stays below demanded pipeline creation; offline library jobs keep
  // their background scheduling. After quit starts, continuations submitted
  // beyond the first queue join need explicit priority during the group wait.
  if (shutdown_write || archive_write)
    write = dispatch_block_create_with_qos_class(DISPATCH_BLOCK_ENFORCE_QOS_CLASS,
        shutdown_write ? QOS_CLASS_USER_INITIATED : QOS_CLASS_UTILITY, 0, write);
  dispatch_async(WriteQueue(), write);
}
void NextLibraryWrite() {
  LibraryWrite work;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (library_writes.empty()) { library_write_active = false; return; }
    work = std::move(library_writes.front()); library_writes.pop_front();
  }
  QueueArchiveWrite(work.key, "library", work.build, [] {
    NextLibraryWrite();
    dispatch_group_leave(library_group);
  });
}
void QueueLibraryWrite(std::string key, std::function<bool(std::string*)> build) {
  bool start;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    dispatch_group_enter(library_group);
    library_writes.push_back({std::move(key), std::move(build)});
    start = !library_write_active; library_write_active = true;
  }
  // Only one library job is submitted at a time. Archive/descriptor writes
  // already queued by drawing run before the following offline compiler job.
  if (start) NextLibraryWrite();
}
id<MTLFunction> OfflineFunction(id<MTLDevice> dev, id<MTLFunction> function) {
  NSArray* parts = [Text(FunctionIdentity(function)) componentsSeparatedByString:@"/"];
  if (parts.count != 2) return nil;
  const std::string key = [parts[0] UTF8String];
  // Used only by the serialized runtime write queue.
  auto& lib = offline_libraries[key];
  if (!lib) {
    lib = LoadArtifact(LibraryPath(key), [dev](const fs::path& candidate) {
      return LoadLibrary(dev, candidate);
    });
    lib.label = Text(key);
  }
  return lib ? Function(lib, parts[1]) : nil;
}
// A successful drawing lookup already validated these exact archive bytes with
// the driver. Under metal.lock, retain integrity checks and scope materialization
// without asking the driver to create the same pipeline a second time.
bool MaterializeValidatedArtifact(const fs::path& local, const std::string& digest) {
  if (digest.empty()) return false;
  for (const auto& candidate : ArtifactCandidates(local)) {
    if (candidate.empty() || !ValidBinary(candidate)) continue;
    if (Read(candidate.string() + ".sha256") != digest) continue;
    MaterializeArtifact(candidate, local);
    ShareArtifact(local);
    return true;
  }
  return false;
}
void PersistPipeline(id<MTLDevice> dev, MTLRenderPipelineDescriptor* p, const std::string& key, bool runtime_ir,
                     std::string validated_digest = {}) {
  MTLRenderPipelineDescriptor* copy = [p copy];
  if (runtime_ir) { std::lock_guard<std::mutex> lock(cache_mutex); source_pipelines[key] = copy; }
  QueueArchiveWrite(key, "pipeline", [dev, copy, runtime_ir, validated_digest](std::string* log) {
    const auto path = ArchivePath(RenderArchiveKey(PipelineKey(copy), RenderSourceMask(copy)), runtime_ir);
    if (!MaterializeValidatedArtifact(path, validated_digest) && !BuildPipeline(dev, copy, log, runtime_ir)) return false;
    if (runtime_ir) {
      // A normal exit can leave only one stage's offline library published.
      // Preserve each available transition, not just the all-offline endpoint:
      // the next process may select either mixed source/offline IR origin.
      const unsigned source_mask = RenderSourceMask(copy);
      auto vertex = (source_mask & 1) ? OfflineFunction(dev, copy.vertexFunction) : copy.vertexFunction;
      auto fragment = (source_mask & 2) ? OfflineFunction(dev, copy.fragmentFunction) : copy.fragmentFunction;
      for (unsigned replace = 1; replace <= 3; replace++) {
        if ((replace & source_mask) != replace || ((replace & 1) && !vertex) || ((replace & 2) && !fragment)) continue;
        MTLRenderPipelineDescriptor* mixed = [copy copy]; mixed.binaryArchives = @[];
        if (replace & 1) mixed.vertexFunction = vertex;
        if (replace & 2) mixed.fragmentFunction = fragment;
        if (!BuildPipeline(dev, mixed, log, RenderSourceMask(mixed) != 0)) return false;
      }
    }
    return true;
  });
}
void PersistCompute(id<MTLDevice> dev, id<MTLFunction> f, bool runtime_ir) {
  if (runtime_ir) { std::lock_guard<std::mutex> lock(cache_mutex); source_compute[ComputeKey(f)] = f; }
  QueueArchiveWrite(ComputeKey(f) + (runtime_ir ? ":source" : ":offline"), "compute", [dev, f, runtime_ir](std::string* log) {
    if (!BuildCompute(dev, f, log, runtime_ir)) return false;
    if (runtime_ir) if (auto offline = OfflineFunction(dev, f)) return BuildCompute(dev, offline, log);
    return true;
  });
}
void MigrateLibraryPipelines(id<MTLDevice> dev, const std::string& library_key) {
  std::map<std::string, MTLRenderPipelineDescriptor*> render;
  std::vector<id<MTLFunction>> compute;
  const std::string prefix = library_key + "/";
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    for (const auto& [key, p] : source_pipelines)
      if (FunctionIdentity(p.vertexFunction).starts_with(prefix) || FunctionIdentity(p.fragmentFunction).starts_with(prefix)) render[key] = p;
    for (const auto& [key, f] : source_compute) if (FunctionIdentity(f).starts_with(prefix)) compute.push_back(f);
  }
  for (const auto& [key, p] : render) PersistPipeline(dev, p, key, true);
  for (auto f : compute) PersistCompute(dev, f, true);
}
id<MTLRenderPipelineState> KeepPipeline(const std::string& key, id<MTLRenderPipelineState> state,
                                      MetalPipelineSource source) {
  if (!state) return nil;
  std::lock_guard<std::mutex> lock(cache_mutex);
  if (!key.empty()) {
    auto [it, added] = ready_pipelines.emplace(key, ReadyPipeline{state, source});
    state = it->second.state; source = it->second.source;
  }
  pipeline_origins[state] = source;
  return state;
}

}

MTLRenderPipelineDescriptor* MetalDescribe(const MetalPipeKey& k, id<MTLFunction> vs, id<MTLFunction> ps) {
  MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
  pd.vertexFunction = vs; pd.fragmentFunction = ps;
  for (int i = 0; i < 4; i++) {
    if (!k.color[i]) continue;
    auto a = pd.colorAttachments[i];
    a.pixelFormat = MTLPixelFormat(k.color[i]); a.blendingEnabled = k.blend != 0;
    a.sourceRGBBlendFactor = MTLBlendFactor(k.src); a.destinationRGBBlendFactor = MTLBlendFactor(k.dst);
    a.rgbBlendOperation = MTLBlendOperation(k.op); a.sourceAlphaBlendFactor = MTLBlendFactor(k.src_a);
    a.destinationAlphaBlendFactor = MTLBlendFactor(k.dst_a); a.alphaBlendOperation = MTLBlendOperation(k.op_a);
    a.writeMask = MTLColorWriteMask(k.mask[i]);
  }
  pd.depthAttachmentPixelFormat = MTLPixelFormat(k.depth);
  pd.stencilAttachmentPixelFormat = MTLPixelFormat(k.stencil);
  return pd;
}
std::string MetalPreparedMsl(const std::string& glsl) {
  if (directory.empty()) return {};
  try {
    fs::path path = fs::path(directory) / "translations" / (Hash(glsl) + ".msl");
    return ValidBinary(path) ? Read(path) : std::string();
  }
  catch (...) { return {}; }
}
std::string MetalPreparationDirectory() { return directory; }
std::vector<MetalRecipe> MetalPreparedRecipes() {
  if (directory.empty()) return {};
  try {
    fs::path path = fs::path(directory) / "prepared-recipes-v1.bin";
    if (!ValidBinary(path)) return {};
    std::string bytes = Read(path);
    if (bytes.size() % sizeof(MetalRecipe) || bytes.size() > (16u << 20)) return {};
    std::vector<MetalRecipe> recipes(bytes.size() / sizeof(MetalRecipe));
    std::memcpy(recipes.data(), bytes.data(), bytes.size());
    return recipes;
  } catch (...) { return {}; }
}
void ConfigureMetalPreparation(const std::string& content_dir, const std::string& root,
                               const MetalPreparationProfile& profile) {
  try { ConfigureCache(content_dir, root, MTLCreateSystemDefaultDevice(), profile); }
  catch (const std::exception& e) {
    directory.clear(); artifact_directory.clear();
    std::fprintf(stderr, "gfx: shader cache unavailable: %s; runtime compilation enabled\n", e.what());
  }
}
id<MTLLibrary> MetalCachedLibrary(id<MTLDevice> dev, NSString* source, MTLCompileOptions* options, NSError** error) {
  // Match the offline compiler settings even when the caller supplies defaults.
  NSDictionary* macros = options.preprocessorMacros;
  options = Options(false, true);
  options.preprocessorMacros = macros;
  // Label all libraries, including fallback libraries, for exact pipeline lookup.
  const auto started = std::chrono::steady_clock::now();
  const std::string key = LibraryKey(source, options);
  id<MTLLibrary> lib = nil;
  bool shared_hit = false;
  if (!directory.empty()) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = loaded_libraries.find(key);
    if (it != loaded_libraries.end()) { Report("library", "memory", started); return it->second; }
    const auto local = LibraryPath(key);
    lib = LoadArtifact(local, [dev, &shared_hit, &local](const fs::path& candidate) {
      auto library = LoadLibrary(dev, candidate);
      if (library) shared_hit = candidate != local;
      return library;
    });
  }
  const bool hit = lib != nil;
  if (!lib) lib = [dev newLibraryWithSource:source options:options error:error];
  Report("library", hit ? "hit" : "miss", started);
  if (lib) {
    lib.label = Text(key);
    if (!directory.empty()) {
      bool publish = false;
      {
        std::lock_guard<std::mutex> lock(cache_mutex);
        loaded_libraries[key] = lib;
        if (!hit) { runtime_libraries.insert(lib); publish = attempted_libraries.insert(key).second; }
      }
      // Own source/macros across the queue seam. Only encountered libraries are
      // published; this never runs discovery or marks full preparation complete.
      const fs::path path = fs::path(directory) / "sources" / (key + ".json");
      source = [source copy]; options.preprocessorMacros = [options.preprocessorMacros copy];
      // Publish the small source record before descriptors can reference it.
      // Library jobs are chained behind archive writes; storing metadata only
      // in those jobs leaves orphan descriptors after an interrupted session.
      if (shared_hit || publish) QueueArchiveWrite("source:" + key, "descriptor", [path, source, options](std::string*) {
        fs::create_directories(path.parent_path());
        if (!fs::exists(path)) JsonFile(path, @{@"source": source, @"macros": options.preprocessorMacros ?: @{}});
        return true;
      });
      if (publish) QueueLibraryWrite("library:" + key, [dev, source, options, key](std::string* log) {
        if (stopping_libraries) return true;
        if (!CompileLibrary(dev, source, options, log, true)) return false;
        MigrateLibraryPipelines(dev, key);
        return true;
      });
    }
  }
  return lib;
}
static id<MTLRenderPipelineState> CreateCachedPipeline(id<MTLDevice> dev, MTLRenderPipelineDescriptor* p, NSError** error,
                                               bool archive_only, const char* phase) {
  const auto started = std::chrono::steady_clock::now();
  const std::string key = PipelineKey(p);
  const unsigned source_mask = RenderSourceMask(p);
  const bool runtime_ir = source_mask != 0;
  const std::string archive_key = RenderArchiveKey(key, source_mask);
  const std::string state_key = key.empty() ? std::string() : key + ":" + std::to_string(source_mask);
  id<MTLRenderPipelineState> ready = nil;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = ready_pipelines.find(state_key);
    if (it != ready_pipelines.end()) ready = it->second.state;
  }
  if (ready) {
    Report("pipeline", "ready", started, phase);
    if (!archive_only) RememberDescriptor(p);
    return ready;
  }
  if (!directory.empty() && !key.empty()) {
    std::string validated_digest;
    if (auto state = LoadRenderArtifact(dev, p, ArchivePath(archive_key, runtime_ir), false, phase,
                                       !archive_only && runtime_ir ? &validated_digest : nullptr)) {
      if (!archive_only) {
        RememberDescriptor(p);
        if (runtime_ir) PersistPipeline(dev, p, state_key, true, std::move(validated_digest));
      }
      state = KeepPipeline(state_key, state, std::strcmp(phase, "preload") == 0 ? MetalPipelineSource::Preloaded : MetalPipelineSource::Archive);
      // Report completion only after the restored state is ready for drawing.
      Report("pipeline", "hit", started, phase);
      return state;
    }
  }
  if (archive_only) { Report("pipeline", "unavailable", started, phase); return nil; }
  const auto driver_started = std::chrono::steady_clock::now();
  id<MTLRenderPipelineState> state = [dev newRenderPipelineStateWithDescriptor:p error:error];
  Report("driver", "miss", driver_started, phase);
  Report("pipeline", "miss", started, phase);
  if (state) {
    RememberDescriptor(p);
    PersistPipeline(dev, p, state_key, runtime_ir);
  }
  return KeepPipeline(state_key, state, MetalPipelineSource::Runtime);
}
namespace {
std::string PipelineStateKey(MTLRenderPipelineDescriptor* p) {
  const std::string key = PipelineKey(p);
  return key.empty() ? std::string() : key + ":" + std::to_string(RenderSourceMask(p));
}
}

void MetalRequestPipeline(id<MTLDevice> dev, MTLRenderPipelineDescriptor* p) {
  const std::string key = PipelineStateKey(p);
  if (key.empty()) return;
  WriteQueue();  // registers the normal-exit drain, including cache-disabled use
  MTLRenderPipelineDescriptor* descriptor = [p copy];
  dispatch_group_t completion = nil;
  size_t lane = 0;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (ready_pipelines.contains(key) || requested_pipelines.contains(key)) return;
    completion = dispatch_group_create();
    dispatch_group_enter(completion);
    requested_pipelines.emplace(key, completion);
    lane = next_pipeline_lane++ % 2;
    // Submit before exposing the request to a drawing caller or shutdown.
    dispatch_async(PipelineLane(lane), ^{
      @autoreleasepool {
        CreateCachedPipeline(dev, descriptor, nullptr, false, "first-use");
        { std::lock_guard<std::mutex> lock(cache_mutex); requested_pipelines.erase(key); }
        dispatch_group_leave(completion);
      }
    });
  }
}

id<MTLRenderPipelineState> MetalCachedPipeline(id<MTLDevice> dev, MTLRenderPipelineDescriptor* p, NSError** error,
                                             bool archive_only, const char* phase) {
  if (!archive_only) {
    dispatch_group_t completion = nil;
    const std::string key = PipelineStateKey(p);
    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      auto it = requested_pipelines.find(key);
      if (it != requested_pipelines.end() && !ready_pipelines.contains(key)) completion = it->second;
    }
    // Archive-only restoration never waits or compiles a missing state.
    // A failed request follows the ordinary synchronous fallback below.
    if (completion) dispatch_group_wait(completion, DISPATCH_TIME_FOREVER);
  }
  return CreateCachedPipeline(dev, p, error, archive_only, phase);
}

void MetalDrainPipelineWrites(bool stop_libraries) {
  // Requests own Metal objects, not a renderer pointer. Finish their publication
  // before draining the existing persistence queues at normal destruction/exit.
  for (size_t lane = 0; lane < 2; lane++) dispatch_sync(PipelineLane(lane), ^{});
  if (stop_libraries) stopping_libraries = true;
  // A group wait does not donate the caller's QoS to the serial queue.
  // Drain queued archives first so normal shutdown promotes work it needs,
  // instead of waiting for background compiler/archive work at background QoS.
  // Library continuations can enqueue more work; retain both waits below.
  dispatch_sync(WriteQueue(), ^{});
  dispatch_group_wait(library_group, DISPATCH_TIME_FOREVER);
  dispatch_sync(WriteQueue(), ^{});
}
MetalPipelineSource MetalPipelineOrigin(id<MTLRenderPipelineState> state) {
  std::lock_guard<std::mutex> lock(cache_mutex);
  auto it = pipeline_origins.find(state);
  return it == pipeline_origins.end() ? MetalPipelineSource::Runtime : it->second;
}
static void RestorePipelines(id<MTLDevice> dev, const std::map<std::string, NSDictionary*>& records,
                             std::chrono::steady_clock::time_point started, std::stop_token stop) {
  size_t ready = 0, skipped = 0;
  try {
    std::map<std::string, id<MTLLibrary>> libraries;
    std::set<std::string> attempted;
    std::vector<std::pair<std::string, NSDictionary*>> pipelines;
    for (const auto& [pipeline_key, record] : records) {
      if (stop.stop_requested()) break;
      @autoreleasepool {
        if (pipeline_key.size() != 64 || pipeline_key.find_first_not_of("0123456789abcdef") != std::string::npos ||
            !HasArtifact(ArchivePath(pipeline_key))) {
          skipped++; continue;
        }
        for (NSString* identity in @[record[@"vertex"] ?: @"", record[@"fragment"] ?: @""]) {
          if (![identity isKindOfClass:NSString.class]) continue;
          NSArray* parts = [identity componentsSeparatedByString:@"/"];
          if (parts.count != 2) continue;
          std::string key = [parts[0] UTF8String];
          if (key.size() != 64 || key.find_first_not_of("0123456789abcdef") != std::string::npos ||
              !attempted.insert(key).second) continue;
          // Preload must never turn a missing/rejected library into compilation.
          const auto library_started = std::chrono::steady_clock::now();
          id<MTLLibrary> lib = LoadArtifact(LibraryPath(key), [dev](const fs::path& candidate) {
            return LoadLibrary(dev, candidate);
          });
          if (!lib) continue;
          lib.label = Text(key); libraries[key] = lib;
          { std::lock_guard<std::mutex> lock(cache_mutex); loaded_libraries[key] = lib; }
          Report("library", "hit", library_started, "preload");
        }
        pipelines.emplace_back(pipeline_key, record);
      }
    }
    // Libraries are immutable and complete before workers read them. Restore
    // independent archives concurrently, retaining the same checksum and real
    // binary-hit gates, and join every lane before completing the task.
    std::atomic<size_t> next{0}, restored{0}, rejected{0};
    std::atomic<bool> failed{false};
    std::mutex error_mutex;
    std::string error;
    const size_t lanes = std::min(pipelines.size(), size_t(std::clamp(std::thread::hardware_concurrency(), 1u, 8u)));
    {
      std::vector<std::jthread> workers;
      for (size_t lane = 0; lane < lanes; lane++) workers.emplace_back([&] {
        while (!failed && !(stop.stop_requested())) {
          const size_t item = next.fetch_add(1);
          if (item >= pipelines.size()) break;
          @autoreleasepool {
            try {
              const auto& [key, record] = pipelines[item];
              auto descriptor = RecordedDescriptor(record, libraries);
              if (descriptor && PipelineKey(descriptor) == key &&
                  MetalCachedPipeline(dev, descriptor, nullptr, true, "preload")) restored++;
              else rejected++;
            } catch (const std::exception& e) {
              std::lock_guard<std::mutex> lock(error_mutex);
              error = e.what(); failed = true;
            }
          }
        }
      });
    }
    ready += restored.load(); skipped += rejected.load();
    if (failed) throw std::runtime_error(error);
  } catch (const std::exception& e) { std::fprintf(stderr, "gfx: shader preload: %s; runtime fallback enabled\n", e.what()); }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  std::fprintf(stderr, "gfx: shader-preload completed ready=%zu skipped=%zu %.3f ms%s\n", ready, skipped, ms,
               stop.stop_requested() ? " cancelled" : "");
}

std::function<void(std::stop_token)> MetalPreloadTask(id<MTLDevice> dev) {
  if (directory.empty()) return [](std::stop_token) {};
  const auto started = std::chrono::steady_clock::now();
  try {
    // The worker exports baseline and renderer descriptors, so first play
    // does not depend on an earlier gameplay recording. Observations augment
    // this catalog. Invalid metadata never enables compilation during preload.
    std::map<std::string, NSDictionary*> records;
    const auto descriptors = fs::path(directory) / "descriptors";
    if (fs::is_directory(descriptors))
      for (const auto& f : fs::directory_iterator(descriptors))
        if (f.is_regular_file() && f.path().extension() == ".json") {
          auto record = JsonRead(f.path());
          if (record.count) records[f.path().stem().string()] = record;
        }
    const bool observed = !records.empty();
    const auto catalog = fs::path(directory) / "prepared-descriptors.json";
    if (ValidBinary(catalog)) {
      auto entries = JsonRead(catalog);
      for (NSString* key in entries)
        if ([entries[key] isKindOfClass:NSDictionary.class] && (!observed || [entries[key][@"core"] boolValue]))
          records.emplace(key.UTF8String, entries[key]);
    }
    std::fprintf(stderr, "gfx: shader-preload policy=%s candidates=%zu\n", observed ? "observed" : "baseline", records.size());
    return [dev, records = std::move(records), started](std::stop_token stop) {
      @autoreleasepool { RestorePipelines(dev, records, started, stop); }
    };
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gfx: shader preload discovery: %s; runtime fallback enabled\n", e.what());
    return [](std::stop_token) {};
  }
}
void MetalPreloadPipelines(id<MTLDevice> dev) { MetalPreloadTask(dev)({}); }

id<MTLComputePipelineState> MetalCachedComputePipeline(id<MTLDevice> dev, id<MTLFunction> f, NSError** error) {
  const auto started = std::chrono::steady_clock::now();
  const bool runtime_ir = RuntimeFunction(f);
  auto p = [MTLComputePipelineDescriptor new]; p.computeFunction = f;
  if (!directory.empty()) {
    if (auto state = LoadComputeArtifact(dev, p, ArchivePath(ComputeKey(f), runtime_ir))) {
      Report("compute", "hit", started);
      if (runtime_ir) PersistCompute(dev, f, true);
      return state;
    }
  }
  id<MTLComputePipelineState> state = [dev newComputePipelineStateWithDescriptor:p options:MTLPipelineOptionNone reflection:nil error:error];
  Report("compute", "miss", started);
  if (state) PersistCompute(dev, f, runtime_ir);
  return state;
}
}  // namespace recomp::gfx

namespace recomp::gfx {
uint64_t MetalShaderIdentity(const hlsl::Program& p) {
  const std::string source = "#version 450\n" + MetalShaderPrelude(p.stage) + p.glsl;
  uint64_t hash = 0xCBF29CE484222325ull;
  for (unsigned char c : source) hash = (hash ^ c) * 0x100000001B3ull;
  return hash ^ uint64_t(p.stage);
}
namespace {
struct ShaderWork {
  std::string name;
  hlsl::Program program;
  uint64_t hash = 0;
  id<MTLFunction> full = nil, lean = nil;
  std::string msl;
};
struct RecordedPipelineWork {
  fs::path path;
  NSDictionary* record;
};
struct Preparation {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  std::string game, root;
  const MetalPreparationProfile& profile;
  fs::path scope, owned;
  FILE* log = nullptr;
  std::recursive_mutex work_mutex;
  unsigned failures = 0;
  size_t completed = 0, total = 0;
  std::vector<ShaderWork> shaders;
  std::map<uint64_t, size_t> shader_index;
  std::set<std::string> entry_files;
  std::set<std::string> pipeline_keys;
  std::vector<NSDictionary*> extra_sources;
  std::vector<RecordedPipelineWork> extra_descriptors;
  std::map<std::string, id<MTLLibrary>> extra_libraries;
  std::vector<MTLRenderPipelineDescriptor*> builtin_pipelines;
  std::vector<id<MTLFunction>> builtin_compute;
  std::map<std::string, id<MTLFunction>> smaa_functions;
  std::string phase = "Discovering";
  ~Preparation() { if (log) std::fclose(log); }
  fs::path State() const { return scope / "state.json"; }
  uintmax_t Size() const {
    uintmax_t n = 0; std::error_code ec;
    std::set<std::pair<dev_t, ino_t>> seen;
    if (fs::is_directory(owned, ec))
      for (const auto& f : fs::recursive_directory_iterator(owned, fs::directory_options::skip_permission_denied, ec))
        if (f.is_regular_file(ec) && !f.is_symlink(ec)) {
          struct stat info;
          if (lstat(f.path().c_str(), &info) == 0 && seen.emplace(info.st_dev, info.st_ino).second) n += info.st_size;
        }
    return n;
  }
  NSString* LogPath() const { return Text((scope / "errors.log").string()); }
  void Progress(bool known = true) {
    std::lock_guard<std::recursive_mutex> guard(work_mutex);
    Emit(@{@"status": @"Preparing", @"phase": Text(phase), @"completed": @(completed),
      @"total": known ? id(@(total)) : id(NSNull.null), @"failed": @(failures),
      @"cache": Text(owned.string()), @"log": LogPath()});
  }
  void Fail(const std::string& item, const std::string& message) {
    std::lock_guard<std::recursive_mutex> guard(work_mutex);
    failures++;
    if (log) { std::fprintf(log, "[%s, failure %u] %s: %s\n", phase.c_str(), failures, item.c_str(), message.c_str()); std::fflush(log); }
    std::fprintf(stderr, "[%s, failure %u] %s: %s\n", phase.c_str(), failures, item.c_str(), message.c_str());
  }
  void RememberEntry(const fs::path& p) {
    std::lock_guard<std::recursive_mutex> guard(work_mutex);
    entry_files.insert(fs::relative(p, scope).string());
    if (p.extension() == ".metallib") entry_files.insert(fs::relative(p.string() + ".sha256", scope).string());
  }
  std::string RecipeFingerprint() const {
    std::string data;
    for (const char* kind : {"descriptors", "sources"}) {
      fs::path path = scope / kind;
      if (!fs::is_directory(path)) continue;
      std::vector<fs::path> files;
      for (const auto& f : fs::directory_iterator(path)) if (f.is_regular_file()) files.push_back(f.path());
      std::sort(files.begin(), files.end());
      for (const auto& f : files) data += std::string(kind) + f.filename().string() + Hash(Read(f));
    }
    return Hash(data);
  }
  bool KnownComplete() const {
    for (const char* kind : {"sources", "descriptors"}) {
      fs::path path = scope / kind;
      if (!fs::is_directory(path)) continue;
      for (const auto& file : fs::directory_iterator(path)) {
        if (file.path().extension() != ".json") continue;
        auto record = JsonRead(file.path());
        if (std::strcmp(kind, "sources") == 0) {
          if (![record[@"source"] isKindOfClass:NSString.class] || ![record[@"macros"] isKindOfClass:NSDictionary.class]) return false;
          auto options = Options(false, true); options.preprocessorMacros = record[@"macros"];
          if (!ValidBinary(LibraryPath(LibraryKey(record[@"source"], options)))) return false;
        } else if (!record.count || !ValidBinary(ArchivePath(file.path().stem().string()))) return false;
      }
    }
    return true;
  }
  bool ValidEntries(NSDictionary* entries) const {
    if (![entries isKindOfClass:NSDictionary.class] || !entries.count) return false;
    NSArray* paths = entries.allKeys;
    std::atomic<size_t> next{0};
    std::atomic<bool> valid{true};
    // Independent reads/checksums must not serialize tens of thousands of
    // filesystem waits. Bound concurrency and stop scheduling after a failure.
    unsigned jobs = unsigned(std::min<size_t>(paths.count,
        std::clamp(std::thread::hardware_concurrency(), 1u, 16u)));
    std::vector<std::thread> workers;
    for (unsigned i = 0; i < jobs; i++) workers.emplace_back([&] {
      while (valid) {
        size_t item = next.fetch_add(1);
        if (item >= paths.count) break;
        @autoreleasepool {
          NSString* relative = paths[item];
          fs::path rel = relative.UTF8String;
          fs::path path = scope / rel;
          try {
            // Manifest paths are app-generated, but never follow arbitrary paths.
            if (std::string(relative.UTF8String).find("..") != std::string::npos || rel.is_absolute() ||
                !fs::is_regular_file(path) || ![entries[relative] isEqual:Text(Hash(Read(path)))]) valid = false;
          } catch (...) { valid = false; }
        }
      }
    });
    for (auto& worker : workers) worker.join();
    return valid;
  }
  void Status() {
    NSDictionary* state = JsonRead(State());
    NSString* status = state[@"status"];
    if (![status isKindOfClass:NSString.class]) status = fs::exists(owned / "v1") ? @"Needs preparation" : @"Not prepared";
    if ([status isEqualToString:@"Preparing"]) status = @"Incomplete";
    if ([status isEqualToString:@"Prepared"]) {
      if (![state[@"recipes"] isEqual:Text(RecipeFingerprint())] && !KnownComplete()) status = @"Needs preparation";
      if (!ValidEntries(state[@"entries"])) status = @"Needs preparation";
    }
    Emit(@{@"status": status, @"failed": state[@"failed"] ?: @0, @"cache": Text(owned.string()),
      @"size": @(Size()), @"log": LogPath(), @"message": state[@"message"] ?: @""});
  }
  void Save(NSString* status, NSString* message = @"") {
    NSMutableDictionary* entries = [NSMutableDictionary new];
    if ([status isEqualToString:@"Prepared"] || [status isEqualToString:@"Completed with errors"])
      for (const auto& relative : entry_files) entries[Text(relative)] = Text(Hash(Read(scope / relative)));
    JsonFile(State(), @{@"status": status, @"failed": @(failures), @"entries": entries,
      @"recipes": Text(RecipeFingerprint()), @"message": message}, true);
  }
  void Discover() {
    Progress(false);
    if (!profile.discover) throw std::runtime_error("Missing shader discovery provider");
    auto inputs = profile.discover(game);
    for (const auto& [name, error] : inputs.errors) Fail(name, error);
    for (auto& input : inputs.shaders) {
      ShaderWork work;
      work.name = std::move(input.name); work.program = std::move(input.program);
      work.hash = MetalShaderIdentity(work.program);
      if (shader_index.emplace(work.hash, shaders.size()).second) shaders.push_back(std::move(work));
    }
    for (const auto& recipe : inputs.recipes) Plan(recipe);
  }
  bool Compile(ShaderWork& work) {
    if (!work.hash) return false;
    const bool vertex = work.program.stage == hlsl::Stage::Vertex;
    std::string glsl = "#version 450\n" + MetalShaderPrelude(work.program.stage) + work.program.glsl, error;
    fs::path msl_path = scope / "translations" / (Hash(glsl) + ".msl");
    if (fs::exists(msl_path) && fs::exists(msl_path.string() + ".sha256")) {
      std::string saved = Read(msl_path);
      if (Read(msl_path.string() + ".sha256") == Hash(saved)) work.msl = std::move(saved);
    }
    if (work.msl.empty()) {
      std::vector<uint32_t> spirv;
      if (!GlslToSpirv(glsl, vertex ? ShaderStage::Vertex : ShaderStage::Fragment, &spirv, &error) ||
          !SpirvToMsl(spirv, &work.msl, &error)) { Fail(work.name, error); return false; }
    }
    bool lean = MetalLeanMacros(&work.msl, vertex);
    auto o = Options(lean, true);
    id<MTLLibrary> lib = CompileLibrary(dev, Text(work.msl), o, &error);
    if (!lib) { if (!cancelled) Fail(work.name, error); return false; }
    work.full = Function(lib, @"main0");
    if (!work.full) { Fail(work.name, "Metal library has no main0 function"); return false; }
    RememberEntry(LibraryPath(LibraryKey(Text(work.msl), o)));
    { std::lock_guard<std::recursive_mutex> guard(work_mutex); extra_libraries[lib.label.UTF8String] = lib; }
    if (lean && !cancelled) {
      auto lo = Options(true, false);
      id<MTLLibrary> ll = CompileLibrary(dev, Text(work.msl), lo, &error);
      work.lean = Function(ll, @"main0");
      if (!work.lean) { if (!cancelled) Fail(work.name + " (lean)", error); return false; }
      RememberEntry(LibraryPath(LibraryKey(Text(work.msl), lo)));
      { std::lock_guard<std::recursive_mutex> guard(work_mutex); extra_libraries[ll.label.UTF8String] = ll; }
    }
    fs::create_directories(msl_path.parent_path()); Atomic(msl_path, work.msl);
    Atomic(msl_path.string() + ".sha256", Hash(work.msl));
    RememberEntry(msl_path); RememberEntry(msl_path.string() + ".sha256");
    return true;
  }
  struct PipelineWork { MetalPipeKey key; uint64_t vs, ps; };
  std::vector<PipelineWork> pipeline_work;
  std::set<std::string> planned;
  void Plan(const MetalRecipe& r) {
    std::string bytes(reinterpret_cast<const char*>(&r), sizeof r);
    if (planned.insert(bytes).second) pipeline_work.push_back({r.key, r.vs, r.ps});
  }
  void ExtraSources() {
    extra_sources.push_back(@{@"source": @(kInternalMsl), @"macros": @{}});
    for (int pass = 0; pass < 3; pass++) for (int stage = 0; stage < 2; stage++) {
      std::string glsl = MetalSmaaGlsl(pass, stage == 0), msl, error;
      std::vector<uint32_t> spirv;
      fs::path path = scope / "translations" / (Hash(glsl) + ".msl");
      if (ValidBinary(path)) msl = Read(path);
      else if (!GlslToSpirv(glsl, stage == 0 ? ShaderStage::Vertex : ShaderStage::Fragment, &spirv, &error) ||
               !SpirvToMsl(spirv, &msl, &error)) { Fail("SMAA", error); continue; }
      extra_sources.push_back(@{@"source": Text(msl), @"macros": @{}, @"smaa": Text(std::to_string(pass) + "/" + std::to_string(stage))});
      fs::create_directories(path.parent_path()); Atomic(path, msl); Atomic(path.string() + ".sha256", Hash(msl));
      RememberEntry(path); RememberEntry(path.string() + ".sha256");
    }
    for (const char* kind : {"sources", "descriptors"}) {
      fs::path path = scope / kind;
      if (!fs::is_directory(path)) continue;
      for (const auto& f : fs::directory_iterator(path)) {
        if (f.path().extension() != ".json") continue;
        auto record = JsonRead(f.path());
        if (std::strcmp(kind, "sources") == 0) {
          if (![record[@"source"] isKindOfClass:NSString.class] || ![record[@"macros"] isKindOfClass:NSDictionary.class])
            Fail(f.path().string(), "Invalid shader source record; clean the shader cache");
          else {
            NSMutableDictionary* source = [record mutableCopy]; source[@"recordPath"] = Text(f.path().string());
            extra_sources.push_back(source);
          }
        } else extra_descriptors.push_back({f.path(), record});
      }
    }
  }
  void CompileExtra(NSDictionary* record) {
    auto options = Options(false, true); options.preprocessorMacros = record[@"macros"];
    NSString* source = record[@"source"];
    std::string error, key = LibraryKey(source, options);
    id<MTLLibrary> lib = CompileLibrary(dev, source, options, &error);
    if (lib) {
      extra_libraries[key] = lib; RememberEntry(LibraryPath(key));
      if ([source isEqualToString:@(kInternalMsl)]) Builtins(lib);
      if ([record[@"smaa"] isKindOfClass:NSString.class]) smaa_functions[[record[@"smaa"] UTF8String]] = Function(lib, @"main0");
    } else if (!cancelled) {
      const std::string path = [record[@"recordPath"] isKindOfClass:NSString.class] ? [record[@"recordPath"] UTF8String] : "built-in shader";
      Fail("Renderer library " + key + " (" + path + ")", error);
    }
  }
  void Builtins(id<MTLLibrary> lib) {
    for (NSString* name in @[@"taa_cs", @"taa_sharpen_cs"]) builtin_compute.push_back(Function(lib, name));
    // Exact descriptors used by clears, StretchRect, depth copy and MetalFX
    // motion generation. Additional formats are learned during gameplay.
    for (auto format : {MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGBA16Float}) {
      for (NSString* kind in @[@"clear", @"quad"]) {
        for (auto depth : {MTLPixelFormatInvalid, MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8}) {
          auto p = [MTLRenderPipelineDescriptor new];
          p.vertexFunction = Function(lib, [kind stringByAppendingString:@"_vs"]);
          p.fragmentFunction = Function(lib, [kind stringByAppendingString:@"_fs"]);
          p.colorAttachments[0].pixelFormat = format;
          p.depthAttachmentPixelFormat = depth;
          if (depth == MTLPixelFormatDepth32Float_Stencil8) p.stencilAttachmentPixelFormat = depth;
          builtin_pipelines.push_back(p);
          if ([kind isEqualToString:@"clear"]) {
            MTLRenderPipelineDescriptor* mask = [p copy]; mask.colorAttachments[0].writeMask = MTLColorWriteMaskNone; builtin_pipelines.push_back(mask);
          }
        }
      }
    }
    for (auto depth : {MTLPixelFormatDepth16Unorm, MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8}) {
      builtin_pipelines.push_back(MetalDepthCopyDescriptor(
          depth, Function(lib, @"tri_vs"), Function(lib, @"depth_copy_fs")));
    }
    auto motion = [MTLRenderPipelineDescriptor new]; motion.vertexFunction = Function(lib, @"tri_vs");
    motion.fragmentFunction = Function(lib, @"motion_fs"); motion.colorAttachments[0].pixelFormat = MTLPixelFormatRG16Float;
    motion.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float; builtin_pipelines.push_back(motion);
  }
  void Pipeline(MTLRenderPipelineDescriptor* pd, const std::string& label) {
    if (!pd) { Fail(label, "Pipeline descriptor is missing"); return; }
    if (!pd.vertexFunction || !pd.fragmentFunction) {
      Fail(label, std::string(!pd.vertexFunction ? "vertex" : "fragment") + " shader function is missing"); return;
    }
    std::string error, key = PipelineKey(pd);
    {
      std::lock_guard<std::recursive_mutex> guard(work_mutex);
      if (!pipeline_keys.insert(key).second) return;
    }
    {
      if (BuildPipeline(dev, pd, &error)) RememberEntry(ArchivePath(key));
      else if (!cancelled) {
        const auto record = PipelineRecord(pd);
        const auto details = Bytes([NSJSONSerialization dataWithJSONObject:record options:NSJSONWritingSortedKeys error:nil]);
        Fail(label + " [" + key + "]", error + "\n  Descriptor: " + details);
      }
    }
  }
  void RecordedPipeline(const RecordedPipelineWork& work) {
    // A recorded descriptor may outlive its source metadata. Restore verified
    // local/shared libraries by identity before requiring a source recompilation.
    for (NSString* stage in @[@"vertex", @"fragment"]) {
      NSString* identity = work.record[stage];
      if (![identity isKindOfClass:NSString.class]) continue;
      NSArray* parts = [identity componentsSeparatedByString:@"/"];
      if (parts.count != 2) continue;
      const std::string key = [parts[0] UTF8String];
      if (key.size() != 64 || key.find_first_not_of("0123456789abcdef") != std::string::npos || extra_libraries.contains(key)) continue;
      auto library = LoadArtifact(LibraryPath(key), [&](const fs::path& path) {
        return LoadLibrary(dev, path);
      }, true);
      if (library) {
        library.label = Text(key); extra_libraries[key] = library;
        RememberEntry(LibraryPath(key));
      }
    }
    std::string error;
    auto pd = RecordedDescriptor(work.record, extra_libraries, &error);
    const auto label = "Recorded pipeline " + work.path.string();
    if (pd) Pipeline(pd, label);
    else {
      const auto details = Bytes([NSJSONSerialization dataWithJSONObject:work.record options:NSJSONWritingSortedKeys error:nil]);
      Fail(label, error + "\n  Descriptor: " + details +
        "\n  Recovery: If a shader compilation failed above, fix it and retry preparation. Otherwise, play the affected scene and exit normally to record its shaders, then retry. For corrupt metadata, remove only this descriptor file and retry; completed entries are kept.");
    }
  }
  void Independent(size_t count, const std::function<void(size_t)>& work) {
    std::atomic<size_t> next{0};
    std::atomic<bool> fatal{false};
    std::string error;
    // Offline tools and Metal spend part of each job waiting on child processes,
    // compiler services and cache I/O. Keep those waits from leaving cores idle.
    unsigned n = std::clamp(2 * std::thread::hardware_concurrency(), 1u, 32u);
    if (const char* jobs = std::getenv("RECOMP_METAL_PREPARE_JOBS")) {
      char* end = nullptr;
      unsigned long value = std::strtoul(jobs, &end, 10);
      if (*jobs && !*end && value >= 1 && value <= 32) n = unsigned(value);
    }
    n = unsigned(std::min<size_t>(n, count));
    std::vector<std::thread> workers;
    for (unsigned i = 0; i < n; i++) workers.emplace_back([&] {
      while (!cancelled && !fatal) {
        size_t item = next.fetch_add(1);
        if (item >= count) break;
        @autoreleasepool {
          try {
            work(item);
            std::lock_guard<std::recursive_mutex> guard(work_mutex);
            if (!cancelled) { completed++; Progress(); }
          } catch (const std::exception& e) {
            std::lock_guard<std::recursive_mutex> guard(work_mutex);
            error = e.what(); fatal = true;
          }
        }
      }
    });
    for (auto& thread : workers) thread.join();
    if (fatal) throw std::runtime_error(error);
  }
  void SavePreparedRecipes() {
    // Only archives published successfully during this run are eligible for
    // prebuilding. Cancelled or failed work must not trigger automatic compiles.
    std::string bytes;
    NSMutableDictionary* descriptors = [NSMutableDictionary new];
    auto remember = [&](MTLRenderPipelineDescriptor* pd, bool core = false) {
      if (!pd || !pd.vertexFunction || !pd.fragmentFunction) return;
      const auto key = PipelineKey(pd);
      if (entry_files.contains(fs::relative(ArchivePath(key), scope).string())) {
        auto record = [PipelineRecord(pd) mutableCopy];
        if (core || [descriptors[Text(key)][@"core"] boolValue]) record[@"core"] = @YES;
        descriptors[Text(key)] = record;
      }
    };
    for (auto pd : builtin_pipelines) remember(pd, true);
    for (const auto& work : extra_descriptors) remember(RecordedDescriptor(work.record, extra_libraries));
    for (const auto& work : pipeline_work) {
      @autoreleasepool {
        auto vi = shader_index.find(work.vs), pi = shader_index.find(work.ps);
        if (vi == shader_index.end() || pi == shader_index.end()) continue;
        const auto& vs = shaders[vi->second]; const auto& ps = shaders[pi->second];
        id<MTLFunction> fragment = (work.key.lean & 1) ? ps.lean : ps.full;
        if (!vs.full || !fragment) continue;
        auto descriptor = MetalDescribe(work.key, vs.full, fragment);
        auto archive = ArchivePath(PipelineKey(descriptor));
        if (!entry_files.contains(fs::relative(archive, scope).string())) continue;
        remember(descriptor);
        MetalRecipe recipe{work.vs, work.ps, work.key};
        bytes.append(reinterpret_cast<const char*>(&recipe), sizeof recipe);
      }
    }
    auto catalog = scope / "prepared-descriptors.json";
    JsonFile(catalog, descriptors); Atomic(catalog.string() + ".sha256", Hash(Read(catalog)));
    RememberEntry(catalog); RememberEntry(catalog.string() + ".sha256");
    auto path = scope / "prepared-recipes-v1.bin";
    Atomic(path, bytes); Atomic(path.string() + ".sha256", Hash(bytes));
    RememberEntry(path); RememberEntry(path.string() + ".sha256");
  }
  void Run() {
    Save(@"Preparing");
    Discover();
    ExtraSources();
    phase = "Compiling shaders"; completed = 0; total = shaders.size() + extra_sources.size(); Progress();
    Independent(shaders.size(), [&](size_t i) { Compile(shaders[i]); });
    for (auto source : extra_sources) {
      if (cancelled) break;
      CompileExtra(source);
      if (!cancelled) { completed++; Progress(); }
    }
    for (int pass = 0; pass < 3; pass++) {
      auto p = [MTLRenderPipelineDescriptor new];
      p.vertexFunction = smaa_functions[std::to_string(pass) + "/0"];
      p.fragmentFunction = smaa_functions[std::to_string(pass) + "/1"];
      p.colorAttachments[0].pixelFormat = pass == 0 ? MTLPixelFormatRG8Unorm : pass == 1 ? MTLPixelFormatRGBA8Unorm : MTLPixelFormatBGRA8Unorm;
      if (pass < 2) p.stencilAttachmentPixelFormat = MTLPixelFormatStencil8;
      builtin_pipelines.push_back(p);
    }
    phase = "Building pipelines"; completed = 0;
    total = pipeline_work.size() + extra_descriptors.size() + builtin_pipelines.size() + builtin_compute.size();
    if (!cancelled) Progress();
    Independent(pipeline_work.size(), [&](size_t i) {
      const auto& p = pipeline_work[i];
      auto vs = shader_index.find(p.vs), ps = shader_index.find(p.ps);
      if (vs == shader_index.end() || ps == shader_index.end()) {
        Fail("Pipeline", "Missing shader source"); return;
      }
      auto& vertex = shaders[vs->second]; auto& pixel = shaders[ps->second];
      id<MTLFunction> pf = (p.key.lean & 1) ? pixel.lean : pixel.full;
      if ((p.key.lean & 1) && !pixel.lean && pixel.full) return;
      if (!vertex.full || !pf) Fail("Pipeline " + std::to_string(p.vs) + "/" + std::to_string(p.ps), "A required shader failed");
      else Pipeline(MetalDescribe(p.key, vertex.full, pf), "Pipeline " + std::to_string(p.vs) + "/" + std::to_string(p.ps));
    });
    for (auto pd : builtin_pipelines) {
      if (cancelled) break;
      Pipeline(pd, "Renderer pipeline");
      if (!cancelled) { completed++; Progress(); }
    }
    for (const auto& work : extra_descriptors) {
      if (cancelled) break;
      RecordedPipeline(work);
      if (!cancelled) { completed++; Progress(); }
    }
    for (auto function : builtin_compute) {
      if (cancelled) break;
      std::string error;
      if (BuildCompute(dev, function, &error)) RememberEntry(ArchivePath(ComputeKey(function)));
      else if (!cancelled) Fail("Renderer compute pipeline", error);
      if (!cancelled) { completed++; Progress(); }
    }
    SavePreparedRecipes();
    NSString* status = cancelled ? @"Incomplete" : failures ? @"Completed with errors" : @"Prepared";
    Save(status);
    Emit(@{@"status": status, @"failed": @(failures), @"phase": Text(phase), @"completed": @(completed),
      @"total": @(total), @"cache": Text(owned.string()), @"size": @(Size()), @"log": LogPath()});
  }
};
struct Lock {
  int fd = -1;
  ~Lock() { if (fd >= 0) { flock(fd, LOCK_UN); close(fd); } }
  void Acquire(const fs::path& owned) {
    fs::create_directories(owned.parent_path());
    fd = open((owned.string() + ".lock").c_str(), O_CREAT | O_RDWR | O_NOFOLLOW, 0600);
    if (fd < 0 || flock(fd, LOCK_EX | LOCK_NB) != 0) throw std::runtime_error("Shader cache is busy or unwritable. Close other preparation workers and check the cache folder permissions.");
  }
};
}

id<MTLFunction> MetalCachedFunction(id<MTLLibrary> lib, NSString* name) { return Function(lib, name); }
int MetalPreparationCommand(int argc, char** argv, const MetalPreparationProfile& profile) {
  if (argc < 2 || std::strcmp(argv[1], "--shader-preparation")) return -1;
  @autoreleasepool {
    Preparation p{.profile = profile};
    std::string action = argc > 2 ? argv[2] : "";
    try {
      for (int i = 3; i < argc; i++) {
        if (!std::strcmp(argv[i], "--game-dir") && i + 1 < argc) p.game = argv[++i];
        else if (!std::strcmp(argv[i], "--cache-root") && i + 1 < argc) p.root = argv[++i];
        else throw std::runtime_error("Expected --game-dir PATH and optional --cache-root PATH");
      }
      p.owned = Owned(p.root, profile);
      if (fs::is_symlink(p.owned)) throw std::runtime_error("The owned shader cache folder is a symbolic link. Choose another cache folder.");
      Lock lock;
      if (action == "clean") {
        lock.Acquire(p.owned);
        fs::remove_all(p.owned);
        Emit(@{@"status": @"Not prepared", @"cache": Text(p.owned.string()), @"size": @0}); return 0;
      }
      if (action != "prepare" && action != "status") throw std::runtime_error("Expected prepare, status, or clean");
      if (action == "prepare") {
        std::signal(SIGTERM, Cancel); std::signal(SIGINT, Cancel); cancelled = 0;
        lock.Acquire(p.owned);
        p.Progress(false);
      }
      if (action == "prepare") {
        fs::create_directories(p.owned);
        p.log = std::fopen((p.owned / "errors.log").c_str(), "w");
        if (!p.log) throw std::runtime_error("Cannot open the shader error log. Choose a writable cache folder.");
      }
      ConfigureCache(p.game, p.root, p.dev, profile); p.scope = directory;
      if (action == "status") { p.Status(); return 0; }
      fs::create_directories(p.scope);
      Atomic(p.scope / ".write-test", "ok"); fs::remove(p.scope / ".write-test");
      if (p.log) { std::fclose(p.log); p.log = nullptr; }
      p.log = std::fopen((p.scope / "errors.log").c_str(), "w");
      if (!p.log) throw std::runtime_error("Cannot open the shader error log. Choose a writable cache folder.");
      std::fprintf(p.log, "Shader preparation error log\nStarted: %s\nGame: %s\nCache scope: %s\nMetal device: %s\nOS: %s\nBuild: %s\n\n",
        NSDate.date.description.UTF8String, p.game.c_str(), p.scope.c_str(), p.dev.name.UTF8String,
        NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String, RECOMP_METAL_BUILD_ID);
      std::fflush(p.log);
      std::string compiler_log;
      if (!cancelled && !OfflineToolchainAvailable(&compiler_log)) throw std::runtime_error(compiler_log);
      p.Run();
      return cancelled ? 130 : p.failures ? 1 : 0;
    } catch (const std::exception& e) {
      p.Fail("Preparation stopped", e.what());
      if (!p.scope.empty()) { try { p.Save(@"Incomplete", Text(e.what())); } catch (...) {} }
      Emit(@{@"status": @"Incomplete", @"failed": @(p.failures), @"message": Text(e.what()),
        @"cache": Text(p.owned.string()), @"log": p.scope.empty() ? Text((p.owned / "errors.log").string()) : p.LogPath()});
      return 2;
    }
  }
}
}  // namespace recomp::gfx
