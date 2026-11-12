// bink_source_test: where BinkOpen's movie comes from (hle/bink_source.h),
// for its three kinds of first argument, in a 32-bit guest arena with
// files the test writes (the bytes are patterns behind a Bink header, but
// for the smallest movies the decoder opens, which BinkOpen is given):
//
// - a file name: the whole file;
// - the file in memory (BINKFROMMEMORY): the length its header gives;
// - a file handle (BINKFILEHANDLE) the guest opened with CreateFile and
//   moved with SetFilePointer: the movie at the handle's position inside a
//   larger file and nothing around it, the handle still open afterwards
//   and where the guest left it, and no movie from a handle that is at no
//   Bink header, at the end of its file, closed, not a file or not a
//   handle at all; no movie either if the read of the header fails or
//   stops short, or the file cannot be mapped, while a read that is
//   interrupted or comes in pieces is carried on.
//
// A handle's movie is mapped, not read. The calls made on the handle's file
// (pread, mmap, munmap) go through the test's, which count them: 8 bytes
// are read, the header; one mapping is made per movie and released when
// the movie is closed or turns out to be none. A movie of 512 MB in a
// sparse file (it takes no room on the disk, and is unlinked once open)
// opens without the process's peak memory growing by a fraction of that,
// and its mapping is still readable after the guest has closed the handle.
//
// A handle must never be taken for a name: the guest memory at the
// handle's value holds the name of another movie file, which must not be
// what is read. BinkOpen itself is called through its import thunk: on a
// handle at a movie inside a larger file, where the BINK it returns has
// that movie's size and neither of the two others' (the one at the start
// of the file, the one the name at the handle's value is of), and the
// handle is the guest's to read and close after BinkClose; then on handles
// that give no movie, with nothing mapped at their values, so reading a
// handle as a string there is a guest memory fault.

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <chrono>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

#include "runtime/hle/bink_source.h"
#include "runtime/hle/hle.h"
#include "runtime/hle/kernel32_state.h"
#include "runtime/runtime.h"

using namespace recomp;

namespace {

int g_failures = 0;
#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      g_failures++;                                                       \
    }                                                                     \
  } while (0)

using Bytes = std::vector<uint8_t>;

constexpr uint32_t kReturn = 0x00401000;  // never run: the handlers return here
constexpr uint32_t kNoFrameBuffers = 0x400, kSndTrack = 0x4000;
// What a game that keeps its movies in a resource file passes.
constexpr uint32_t kHandleFlags = hle::kBinkFileHandle | kNoFrameBuffers;
constexpr uint32_t kInvalidSetFilePointer = 0xFFFFFFFF;
constexpr uint32_t kNeverAHandle = 0x7FFC;

Ctx* g_ctx = nullptr;  // the guest thread's
uint32_t g_stack_top = 0, g_next = 0, g_scratch_end = 0;

uint32_t Alloc(uint32_t size) {
  uint32_t p = g_next;
  g_next += (size + 15) & ~15u;
  if (g_next > g_scratch_end) std::abort();
  std::memset(GPtr(p), 0, size);
  return p;
}

uint32_t Str(const std::string& s) {
  uint32_t p = Alloc(uint32_t(s.size() + 1));
  std::memcpy(GPtr(p), s.data(), s.size());
  return p;
}

// A stdcall to dll!name with `args`, as guest code makes it.
uint32_t Call(const char* dll, const char* name, const std::vector<uint32_t>& args) {
  uint32_t sp = g_stack_top - 4 * uint32_t(args.size());
  for (size_t i = 0; i < args.size(); i++) Wr32(sp + 4 * uint32_t(i), args[i]);
  sp -= 4;
  Wr32(sp, kReturn);
  g_ctx->esp = sp;
  hle::CallThunk(*g_ctx, uint32_t(hle::ImportThunk(dll, name)));
  CHECK(g_ctx->esp == g_stack_top);  // all arguments popped
  return g_ctx->eax;
}

// The guest's file calls.
uint32_t Open(const std::string& guest_path) {
  // GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING
  return Call("kernel32.dll", "CreateFileA", {Str(guest_path), 0x80000000, 1, 0, 3, 0, 0});
}
uint32_t Seek(uint32_t h, uint32_t pos) {
  return Call("kernel32.dll", "SetFilePointer", {h, pos, 0, 0});  // FILE_BEGIN
}
uint32_t Tell(uint32_t h) {
  return Call("kernel32.dll", "SetFilePointer", {h, 0, 0, 1});  // FILE_CURRENT
}
Bytes Read(uint32_t h, uint32_t n) {
  uint32_t buf = Alloc(n), got = Alloc(4);
  if (!Call("kernel32.dll", "ReadFile", {h, buf, n, got, 0})) return {};
  return Bytes(GPtr(buf), GPtr(buf) + Rd32(got));
}
uint32_t Close(uint32_t h) { return Call("kernel32.dll", "CloseHandle", {h}); }

uint32_t BinkOpen(uint32_t src, uint32_t flags) {
  return Call("binkw32.dll", "_BinkOpen@8", {src, flags});
}
void BinkClose(uint32_t bink) { Call("binkw32.dll", "_BinkClose@4", {bink}); }

// A Bink file: "BIKi", the length of what follows these 8 bytes (`claimed`,
// unless 0: `body`), then `body` bytes of a pattern that `seed` starts.
Bytes Movie(uint32_t body, uint8_t seed, uint32_t claimed = 0) {
  if (!claimed) claimed = body;
  Bytes b = {'B', 'I', 'K', 'i', uint8_t(claimed), uint8_t(claimed >> 8), uint8_t(claimed >> 16),
             uint8_t(claimed >> 24)};
  for (uint32_t i = 0; i < body; i++) b.push_back(uint8_t(seed + 7 * i));
  return b;
}

void Append(Bytes* to, const Bytes& b) { to->insert(to->end(), b.begin(), b.end()); }

void Put32(Bytes* to, uint32_t v) {
  for (int i = 0; i < 4; i++) to->push_back(uint8_t(v >> (8 * i)));
}

// The smallest Bink file the decoder opens, 80 bytes: one frame of `side` x
// `side` pixels at 30 frames a second, no audio. Its header (44 bytes),
// the frame index (the frame's offset, bit 0: a key frame) and the frame:
// 32 zero bytes.
constexpr uint32_t kTinyFrame = 32, kTinyIndexAt = 44, kTinyFrameAt = 48;
Bytes TinyMovie(uint32_t side) {
  Bytes b = {'B', 'I', 'K', 'i'};
  Put32(&b, kTinyFrameAt + kTinyFrame - 8);  // the length of the rest of the file
  Put32(&b, 1);                              // frames
  Put32(&b, kTinyFrame);                     // the largest frame
  Put32(&b, 0);
  Put32(&b, side);  // width
  Put32(&b, side);  // height
  Put32(&b, 30);    // frames per
  Put32(&b, 1);     // this many seconds
  Put32(&b, 0);     // video flags
  Put32(&b, 0);     // audio tracks
  if (b.size() != kTinyIndexAt) std::abort();
  Put32(&b, kTinyFrameAt | 1);
  b.insert(b.end(), kTinyFrame, 0);
  return b;
}

// The BINK struct's first fields.
uint32_t Width(uint32_t bink) { return Rd32(bink); }
uint32_t Height(uint32_t bink) { return Rd32(bink + 4); }
uint32_t Frames(uint32_t bink) { return Rd32(bink + 8); }

void WriteFile(const std::filesystem::path& path, const Bytes& b) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f || std::fwrite(b.data(), 1, b.size(), f) != b.size()) std::abort();
  std::fclose(f);
}

struct Source {
  bool ok;
  Bytes data;
  std::string what;
};
Source OpenSource(uint32_t src, uint32_t flags) {
  Source s;
  hle::BinkBytes bytes;
  bytes.Own({0xAA});  // what was there before must not survive
  s.ok = hle::OpenBinkSource(*g_ctx, src, flags, &bytes, &s.what);
  s.data.assign(bytes.data(), bytes.data() + bytes.size());
  return s;  // the movie's bytes are released here
}

// The files. pack.dat is a resource file: other data, a movie, other
// data, and a movie cut short by the end of the file.
const Bytes named = Movie(300, 1), packed = Movie(500, 2), decoy = Movie(200, 3),
            cut = Movie(40, 4, 1000), undecodable = Movie(56, 0);
constexpr uint32_t kPackedAt = 100, kGap = 64;
const uint32_t cut_at = kPackedAt + uint32_t(packed.size()) + kGap;
Bytes Pack() {
  Bytes pack(kPackedAt, 0xEE);
  Append(&pack, packed);
  pack.insert(pack.end(), kGap, 0xDD);
  Append(&pack, cut);
  return pack;
}
const Bytes pack = Pack();
constexpr uint32_t kUndecodableAt = 16;
// movies.dat is a resource file of movies the decoder opens: one at its
// start, other data, a smaller one, other data. other.bik is a third.
const Bytes first = TinyMovie(32), tiny = TinyMovie(16), other = TinyMovie(48);
constexpr uint32_t kTinyAt = 200;
Bytes Movies() {
  Bytes movies = first;
  movies.resize(kTinyAt, 0xEE);
  Append(&movies, tiny);
  movies.insert(movies.end(), kGap, 0xDD);
  return movies;
}

// A movie as large as the ones games keep in resource files, 512 MB, at an
// offset that is a multiple of no page size: 4096 frames of zero bytes. Only
// its header and frame index are written: the file is sparse.
constexpr uint32_t kLargeSize = 512u << 20, kLargeFrames = 4096, kLargeSide = 64;
constexpr uint32_t kLargeAt = 3 * 65536 + 1234;
constexpr uint32_t kSizeField = 0x28;  // of the BINK struct
Bytes LargeHead() {
  const uint32_t first = kTinyIndexAt + 4 * kLargeFrames;
  const uint32_t frame = ((kLargeSize - first) / kLargeFrames) & ~1u;
  Bytes b = {'B', 'I', 'K', 'i'};
  Put32(&b, kLargeSize - 8);
  Put32(&b, kLargeFrames);
  Put32(&b, frame + 2 * kLargeFrames);  // the largest frame: the last
  Put32(&b, 0);
  Put32(&b, kLargeSide);
  Put32(&b, kLargeSide);
  Put32(&b, 30);
  Put32(&b, 1);
  Put32(&b, 0);
  Put32(&b, 0);
  for (uint32_t i = 0; i < kLargeFrames; i++) Put32(&b, (first + i * frame) | 1);
  return b;
}

// False where files are not sparse: the file is not left there.
bool WriteLargeFile(const std::filesystem::path& path) {
  const Bytes head = LargeHead();
  const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
  if (fd < 0 || ftruncate(fd, off_t(kLargeAt) + kLargeSize + kGap) != 0 ||
      pwrite(fd, head.data(), head.size(), kLargeAt) != ssize_t(head.size()))
    std::abort();
  struct stat st;
  if (fstat(fd, &st) != 0) std::abort();
  ::close(fd);
  if (uint64_t(st.st_blocks) * 512 <= (4u << 20)) return true;
  std::filesystem::remove(path);
  return false;
}

// The process's peak resident memory, in bytes.
uint64_t PeakMemory() {
  struct rusage u;
  getrusage(RUSAGE_SELF, &u);
#ifdef __APPLE__
  return uint64_t(u.ru_maxrss);
#else
  return uint64_t(u.ru_maxrss) * 1024;
#endif
}

// The calls on a handle's file, counted. A read gives at most g_piece
// bytes, and one read (by number, from 0) may be interrupted, fail or find
// the end of the file; a mapping may fail.
enum class BadRead { None, Interrupted, Error, End };
BadRead g_bad_read = BadRead::None;
int g_bad_read_at = 0, g_reads = 0, g_maps = 0, g_unmaps = 0;
size_t g_piece = SIZE_MAX;
uint64_t g_read_bytes = 0;
bool g_fail_map = false;
void* g_map = nullptr;  // the last mapping
size_t g_map_size = 0;
ssize_t CountedPread(int fd, void* buf, size_t n, off_t at) {
  if (g_reads++ == g_bad_read_at) {
    switch (g_bad_read) {
      case BadRead::None: break;
      case BadRead::Interrupted: errno = EINTR; return -1;
      case BadRead::Error: errno = EIO; return -1;
      case BadRead::End: return 0;
    }
  }
  const ssize_t r = pread(fd, buf, std::min(n, g_piece), at);
  if (r > 0) g_read_bytes += uint64_t(r);
  return r;
}
void* CountedMmap(void* addr, size_t n, int prot, int flags, int fd, off_t at) {
  if (g_fail_map) {
    errno = ENODEV;
    return MAP_FAILED;
  }
  void* p = mmap(addr, n, prot, flags, fd, at);
  if (p != MAP_FAILED) {
    g_maps++;
    g_map = p;
    g_map_size = n;
  }
  return p;
}
int CountedMunmap(void* addr, size_t n) {
  if (addr == g_map && n == g_map_size) g_unmaps++;
  return munmap(addr, n);
}
void ResetCounts() {
  g_reads = g_maps = g_unmaps = 0;
  g_read_bytes = 0;
}

void WriteFiles(const std::filesystem::path& dir) {
  Bytes zeros(kUndecodableAt, 0);
  Append(&zeros, undecodable);
  WriteFile(dir / "name.bik", named);
  WriteFile(dir / "pack.dat", pack);
  WriteFile(dir / "decoy.bik", decoy);
  WriteFile(dir / "zeros.dat", zeros);
  WriteFile(dir / "movies.dat", Movies());
  WriteFile(dir / "other.bik", other);
}

void RunChecks(const std::filesystem::path& dir) {
  hle::g_bink_file = {CountedPread, CountedMmap, CountedMunmap};
  const size_t page = size_t(sysconf(_SC_PAGESIZE));

  // The kinds of source by open flags; the other flags do not matter.
  CHECK(hle::BinkSourceOf(0) == hle::BinkSource::Name);
  CHECK(hle::BinkSourceOf(kNoFrameBuffers | kSndTrack) == hle::BinkSource::Name);
  CHECK(hle::BinkSourceOf(kHandleFlags) == hle::BinkSource::Handle);
  CHECK(hle::BinkSourceOf(hle::kBinkFromMemory | kSndTrack) == hle::BinkSource::Memory);

  // BinkOpen itself on a handle at a movie. The first guest pages exist
  // here, and at the handle's value is the name of another movie's file.
  {
    const uint32_t h = Open("C:\\Test\\movies.dat");
    CHECK(h != 0 && h != 0xFFFFFFFF && h < 0x4000);
    g_proc->mem.Commit(0, 0x8000, Prot::RW);
    const std::string other_name = "C:\\Test\\other.bik";
    std::memcpy(GPtr(h), other_name.c_str(), other_name.size() + 1);
    {
      // The other movie is one the name form opens.
      const uint32_t b = BinkOpen(h, kNoFrameBuffers);
      CHECK(b != 0);
      if (b) {
        CHECK(Width(b) == 48 && Height(b) == 48);
        BinkClose(b);
      }
    }
    CHECK(g_maps == 0 && g_reads == 0);  // a name's file is not the handle's
    {
      // So is the one at the start of the file, for a handle there.
      const uint32_t b = BinkOpen(h, kHandleFlags);
      CHECK(b != 0);
      if (b) {
        CHECK(Width(b) == 32 && Height(b) == 32);
        BinkClose(b);
      }
    }
    CHECK(g_maps == 1 && g_unmaps == 1);
    CHECK(Seek(h, kTinyAt) == kTinyAt);
    ResetCounts();
    const uint32_t b = BinkOpen(h, kHandleFlags);
    CHECK(b != 0);
    if (b) {
      CHECK(Width(b) == 16);  // the movie at the handle's position
      CHECK(Height(b) == 16);
      CHECK(Frames(b) == 1);
      CHECK(Rd32(b + kSizeField) == tiny.size());
    }
    // Mapped from the page it is in, and only its header read.
    CHECK(g_maps == 1 && g_map_size == kTinyAt % page + tiny.size());
    CHECK(g_read_bytes == 8);
    CHECK(g_unmaps == 0);       // for as long as the movie is open
    CHECK(Tell(h) == kTinyAt);  // open, and where the guest left it
    if (b) BinkClose(b);
    CHECK(g_unmaps == 1);  // released with the movie
    CHECK(Tell(h) == kTinyAt);
    CHECK(Read(h, uint32_t(tiny.size())) == tiny);  // still the guest's to read
    // The read of the header fails, or the file cannot be mapped: no movie.
    CHECK(Seek(h, kTinyAt) == kTinyAt);
    ResetCounts();
    g_bad_read_at = 0, g_bad_read = BadRead::Error;
    CHECK(BinkOpen(h, kHandleFlags) == 0);
    g_bad_read = BadRead::None;
    CHECK(g_reads == 1 && g_maps == 0);
    g_fail_map = true;
    CHECK(BinkOpen(h, kHandleFlags) == 0);
    g_fail_map = false;
    CHECK(g_maps == 0 && g_unmaps == 0);
    CHECK(Tell(h) == kTinyAt);
    CHECK(Read(h, 8) == Bytes(tiny.begin(), tiny.begin() + 8));
    CHECK(Close(h) == 1);  // and to close
    CHECK(Close(h) == 0);
    g_proc->mem.Decommit(0, 0x8000);
  }

  // BinkOpen itself on handles that give no movie. Nothing is mapped at
  // the handles' values: taking one for a string faults.
  {
    const uint32_t h = Open("C:\\Test\\zeros.dat");
    CHECK(h != 0 && h != 0xFFFFFFFF && h < 0x4000);
    CHECK(Seek(h, kUndecodableAt) == kUndecodableAt);
    ResetCounts();
    CHECK(BinkOpen(h, kHandleFlags) == 0);  // a Bink header, but no movie behind it
    CHECK(g_maps == 1 && g_unmaps == 1);    // mapped to find that out, and released
    CHECK(Tell(h) == kUndecodableAt);
    CHECK(Read(h, 8) == Bytes(undecodable.begin(), undecodable.begin() + 8));
    CHECK(Close(h) == 1);  // still the guest's to close
    CHECK(BinkOpen(h, kHandleFlags) == 0);  // closed
    CHECK(BinkOpen(kNeverAHandle, kHandleFlags) == 0);
    CHECK(g_maps == 1 && g_reads == 1);  // nothing asked of what is no file handle
  }

  // BinkOpen itself on a handle at a large movie: mapped, so neither read
  // nor copied, whatever its size. The file is gone from the directory as
  // soon as the guest has it open.
  if (!WriteLargeFile(dir / "large.dat")) {
    std::fprintf(stderr, "bink_source_test: no sparse files in %s: the large movie is skipped\n",
                 dir.c_str());
  } else {
    const uint32_t h = Open("C:\\Test\\large.dat");
    CHECK(h != 0 && h != 0xFFFFFFFF);
    std::filesystem::remove(dir / "large.dat");
    CHECK(Seek(h, kLargeAt) == kLargeAt);
    ResetCounts();
    const uint64_t memory = PeakMemory();
    const auto start = std::chrono::steady_clock::now();
    const uint32_t b = BinkOpen(h, kHandleFlags);
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const uint64_t grown = PeakMemory() - memory;
    std::printf("bink_source_test: %u MB movie opened in %.2f ms: %llu bytes read, peak memory "
                "+%llu KiB\n",
                kLargeSize >> 20, ms, static_cast<unsigned long long>(g_read_bytes),
                static_cast<unsigned long long>(grown >> 10));
    CHECK(b != 0);
    if (b) {
      CHECK(Width(b) == kLargeSide && Height(b) == kLargeSide);
      CHECK(Frames(b) == kLargeFrames);
      CHECK(Rd32(b + kSizeField) == kLargeSize);
    }
    CHECK(g_maps == 1 && g_unmaps == 0);
    CHECK(g_map_size == kLargeAt % page + kLargeSize);  // the movie and no more of the file
    CHECK(g_read_bytes == 8);                           // its header
    CHECK(grown < kLargeSize / 8);                      // not held, and not gone through
    CHECK(ms < 2000);
    CHECK(Tell(h) == kLargeAt);
    // The mapping does not need the handle: the movie is there after the
    // guest closes it, from the mapping's page on.
    CHECK(Close(h) == 1);
    if (b && g_maps == 1 && g_unmaps == 0)
      CHECK(std::memcmp(static_cast<uint8_t*>(g_map) + kLargeAt % page, "BIKi", 4) == 0);
    if (b) BinkClose(b);
    CHECK(g_unmaps == 1);
  }

  // By name: the whole file, whatever the other flags.
  for (uint32_t flags : {0u, kNoFrameBuffers, kNoFrameBuffers | kSndTrack}) {
    const Source s = OpenSource(Str("C:\\Test\\name.bik"), flags);
    CHECK(s.ok && s.data == named);
    CHECK(s.what == "C:\\Test\\name.bik");
  }
  {
    const Source s = OpenSource(Str("C:\\Test\\missing.bik"), 0);
    CHECK(!s.ok && s.data.empty());
  }

  // From memory: the length the header gives, not what lies after it.
  {
    const uint32_t mem = Alloc(uint32_t(packed.size()) + 32);
    std::memcpy(GPtr(mem), packed.data(), packed.size());
    std::memset(GPtr(mem) + packed.size(), 0xDD, 32);
    const Source s = OpenSource(mem, hle::kBinkFromMemory | kSndTrack);
    CHECK(s.ok && s.data == packed);
    const uint32_t other = Alloc(64);
    std::memset(GPtr(other), 0xEE, 64);
    const Source bad = OpenSource(other, hle::kBinkFromMemory);
    CHECK(!bad.ok && bad.data.empty());
  }

  // From a file handle. The first guest pages now exist, and at the
  // handle's value is the name of the decoy movie.
  const uint32_t h = Open("C:\\Test\\pack.dat");
  CHECK(h != 0 && h != 0xFFFFFFFF && h < 0x4000);
  g_proc->mem.Commit(0, 0x8000, Prot::RW);
  const std::string decoy_name = "C:\\Test\\decoy.bik";
  std::memcpy(GPtr(h), decoy_name.c_str(), decoy_name.size() + 1);
  {
    // The decoy is a movie the name form would read.
    ResetCounts();
    const Source s = OpenSource(h, kNoFrameBuffers);
    CHECK(s.ok && s.data == decoy);
    CHECK(g_maps == 0 && g_reads == 0);
  }
  CHECK(Seek(h, kPackedAt) == kPackedAt);
  {
    const Source s = OpenSource(h, kHandleFlags);
    CHECK(s.ok);
    CHECK(s.data == packed);  // from the position on, and no further than the movie
    CHECK(s.data != decoy);   // the handle was not read as a name
    CHECK(s.what.find("decoy") == std::string::npos);
    CHECK(g_maps == 1 && g_unmaps == 1 && g_read_bytes == 8);
  }
  // The handle is where the guest left it and still reads.
  CHECK(Tell(h) == kPackedAt);
  CHECK(Read(h, 16) == Bytes(packed.begin(), packed.begin() + 16));
  CHECK(Tell(h) == kPackedAt + 16);
  // A movie the file ends in the middle of: what there is of it.
  CHECK(Seek(h, cut_at) == cut_at);
  {
    const Source s = OpenSource(h, kHandleFlags);
    CHECK(s.ok && s.data == cut);
    CHECK(Tell(h) == cut_at);
  }
  // The header read in pieces of 3 bytes, one of them interrupted: carried
  // on, the whole movie.
  g_piece = 3;
  CHECK(Seek(h, kPackedAt) == kPackedAt);
  for (int at : {-1, 0, 2}) {
    ResetCounts();
    g_bad_read_at = at, g_bad_read = BadRead::Interrupted;
    const Source s = OpenSource(h, kHandleFlags);
    CHECK(s.ok && s.data == packed);
    CHECK(g_reads == (at < 0 ? 3 : 4));  // 8 bytes by 3, the one again
  }
  // A read that fails, or finds the file ended before the bytes it has:
  // no movie, and nothing mapped.
  for (BadRead bad : {BadRead::Error, BadRead::End}) {
    for (int at : {0, 1, 2}) {
      ResetCounts();
      g_bad_read_at = at, g_bad_read = bad;
      const Source s = OpenSource(h, kHandleFlags);
      CHECK(!s.ok && s.data.empty());
      CHECK(g_reads == at + 1 && g_maps == 0);
      CHECK(Tell(h) == kPackedAt);
    }
  }
  g_bad_read = BadRead::None;
  // Nor if the file cannot be mapped.
  {
    g_fail_map = true;
    const Source s = OpenSource(h, kHandleFlags);
    g_fail_map = false;
    CHECK(!s.ok && s.data.empty());
    CHECK(Tell(h) == kPackedAt);
  }
  // The movie the file ends in the middle of is still one, in pieces too.
  CHECK(Seek(h, cut_at) == cut_at);
  {
    const Source s = OpenSource(h, kHandleFlags);
    CHECK(s.ok && s.data == cut);
  }
  g_piece = SIZE_MAX;
  // No Bink header at the position: the other data, the last bytes of the
  // file, its end.
  for (uint32_t pos : {0u, uint32_t(pack.size()) - 4, uint32_t(pack.size())}) {
    CHECK(Seek(h, pos) == pos);
    const Source s = OpenSource(h, kHandleFlags);
    CHECK(!s.ok && s.data.empty());
    CHECK(Tell(h) == pos);
  }
  CHECK(g_maps == g_unmaps);  // every mapping so far released
  // Still open: the guest closes it, once.
  CHECK(Close(h) == 1);
  CHECK(Close(h) == 0);
  CHECK(Tell(h) == kInvalidSetFilePointer);
  {
    // Closed. The decoy's name is still at its value.
    const Source s = OpenSource(h, kHandleFlags);
    CHECK(!s.ok && s.data.empty());
  }
  {
    const Source s = OpenSource(kNeverAHandle, kHandleFlags);
    CHECK(!s.ok && s.data.empty());
  }
  {
    // A handle, but not a file's.
    const uint32_t event = NewEventHandle(true, false);
    const Source s = OpenSource(event, kHandleFlags);
    CHECK(!s.ok && s.data.empty());
    CHECK(Close(event) == 1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  namespace fs = std::filesystem;
  if (argc < 2) {
    std::fprintf(stderr, "usage: bink_source_test <directory for its files>\n");
    return 2;
  }
  const fs::path dir = fs::path(argv[1]) / ("bink_source_test_" + std::to_string(::getpid()));
  fs::create_directories(dir / "data");
  ::setenv("XDG_DATA_HOME", (dir / "data").c_str(), 1);

  WriteFiles(dir);

  // A 32-bit process with its heap and scratch memory.
  g_proc = new Process;
  g_proc->game.id = "binktest";
  g_proc->game.guest_dir = "C:\\Test";
  g_proc->game.exe_name = "test.exe";
  g_proc->game_dir = dir.string();
  if (!g_proc->mem.Reserve(32)) return 1;
  g_proc->mem.Commit(kThunkBase, kThunkRegionSize, Prot::R);
  g_proc->mem.Commit(kPebVa, 0x1000, Prot::RW);
  g_proc->vm.Init(&g_proc->mem, 0x00010000, kThunkBase);
  InitKernel32State();
  g_next = uint32_t(g_proc->vm.Reserve(0, 0x100000));
  g_proc->vm.Commit(g_next, 0x100000, Prot::RW);
  g_scratch_end = g_next + 0x100000;

  // The checks run as a guest thread (the file calls set its last error).
  Thread* t = CreateGuestThread(0x100000);
  g_ctx = &t->ctx;
  g_stack_top = uint32_t(t->ctx.esp) - 0x100;
  std::promise<void> done;
  StartServiceThread(t, [&done, &dir] {
    RunChecks(dir);
    done.set_value();
  });
  done.get_future().wait();

  fs::remove_all(dir);
  if (g_failures) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("bink_source_test: ok\n");
  return 0;
}
