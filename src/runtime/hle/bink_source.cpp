#include "runtime/hle/bink_source.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>

#include "runtime/hle/hle.h"
#include "runtime/hle/kernel32_state.h"
#include "runtime/log.h"

namespace recomp::hle {
namespace {

// A Bink file starts with "BIK", a revision letter and the length of the
// rest of the file after these 8 bytes.
constexpr size_t kHeadSize = 8;

uint64_t FileSize(const uint8_t* head) {
  return kHeadSize + (head[4] | head[5] << 8 | head[6] << 16 | uint64_t(head[7]) << 24);
}

bool FromMemory(Ctx& c, uint32_t src, BinkBytes* bytes) {
  if (!EnsureReadable(c, src, kHeadSize) || std::memcmp(GPtr(src), "BIK", 3) != 0) {
    WarnF("binkw32: BinkOpen: no Bink header at %08X\n", src);
    return false;
  }
  const uint64_t size = FileSize(GPtr(src));
  if (!EnsureReadable(c, src, size)) {
    WarnF("binkw32: BinkOpen: cannot page in %llu bytes at %08X\n",
          static_cast<unsigned long long>(size), src);
    return false;
  }
  bytes->Own(std::vector<uint8_t>(GPtr(src), GPtr(src) + size));
  return true;
}

bool FromName(const std::string& name, BinkBytes* bytes) {
  FILE* f = std::fopen(HostPath(name).c_str(), "rb");
  if (!f) {
    WarnF("binkw32: BinkOpen(%s): cannot open\n", name.c_str());
    return false;
  }
  std::vector<uint8_t> data;
  uint8_t buf[65536];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) data.insert(data.end(), buf, buf + n);
  std::fclose(f);
  bytes->Own(std::move(data));
  return true;
}

// All of [at, at + n) of the file into p, in as many reads as it takes.
// Null, or why not: a read failed, or the file ended before `n` bytes.
const char* ReadAt(int fd, uint8_t* p, size_t n, off_t at) {
  size_t got = 0;
  while (got < n) {
    const ssize_t r = g_bink_file.pread(fd, p + got, n - got, at + off_t(got));
    if (r < 0 && errno == EINTR) continue;
    if (r < 0) return std::strerror(errno);
    if (r == 0) return "the file ends there";
    got += size_t(r);
  }
  return nullptr;
}

// The handle's descriptor is the game's: its header is read at an offset
// (pread) and its movie mapped, neither of which moves the file position,
// and it is never closed here. The object is held meanwhile, so a
// CloseHandle on another thread cannot close the descriptor under the
// calls; the mapping outlives both.
bool FromHandle(uint32_t handle, BinkBytes* bytes) {
  const std::shared_ptr<KObject> file = GetHandle(handle);
  const int fd = file && file->type == KObject::Type::File ? FileFd(file.get()) : -1;
  const off_t at = fd >= 0 ? lseek(fd, 0, SEEK_CUR) : -1;
  struct stat st;
  if (at < 0 || fstat(fd, &st) != 0) {
    WarnF("binkw32: BinkOpen: %08X is not an open file handle\n", handle);
    return false;
  }
  const uint64_t left = st.st_size > at ? uint64_t(st.st_size - at) : 0;
  uint8_t head[kHeadSize];
  if (left >= kHeadSize) {
    if (const char* error = ReadAt(fd, head, kHeadSize, at)) {
      WarnF("binkw32: BinkOpen: cannot read %zu bytes at offset %lld of file handle %08X: %s\n",
            kHeadSize, static_cast<long long>(at), handle, error);
      return false;
    }
  }
  if (left < kHeadSize || std::memcmp(head, "BIK", 3) != 0) {
    WarnF("binkw32: BinkOpen: no Bink header at offset %lld of file handle %08X (%llu bytes "
          "left)\n",
          static_cast<long long>(at), handle, static_cast<unsigned long long>(left));
    return false;
  }
  // A movie the file ends in the middle of goes as far as the file does
  // now, like a file opened by name: no page of the mapping lies wholly
  // past the end of the file. (A file made shorter while its movie plays
  // is not covered: reading a page it no longer has is a fault.)
  const size_t size = size_t(std::min(FileSize(head), left));
  // A mapping starts at a page of the file; the movie starts `delta` in.
  const size_t delta = size_t(at % off_t(sysconf(_SC_PAGESIZE)));
  void* map = g_bink_file.mmap(nullptr, delta + size, PROT_READ, MAP_PRIVATE, fd,
                               at - off_t(delta));
  if (map == MAP_FAILED) {
    WarnF("binkw32: BinkOpen: cannot map %zu bytes at offset %lld of file handle %08X: %s\n",
          size, static_cast<long long>(at), handle, std::strerror(errno));
    return false;
  }
  bytes->Map(map, delta, size);
  return true;
}

}  // namespace

BinkFileCalls g_bink_file = {pread, mmap, munmap};

void BinkBytes::Clear() {
  if (map_) g_bink_file.munmap(map_, map_size_);
  map_ = nullptr;
  map_size_ = 0;
  copy_.clear();
  copy_.shrink_to_fit();
  data_ = nullptr;
  size_ = 0;
}

void BinkBytes::Own(std::vector<uint8_t> copy) {
  Clear();
  copy_ = std::move(copy);
  data_ = copy_.data();
  size_ = copy_.size();
}

void BinkBytes::Map(void* map, size_t delta, size_t size) {
  Clear();
  map_ = map;
  map_size_ = delta + size;
  data_ = static_cast<const uint8_t*>(map) + delta;
  size_ = size;
}

BinkSource BinkSourceOf(uint32_t open_flags) {
  if (open_flags & kBinkFromMemory) return BinkSource::Memory;
  if (open_flags & kBinkFileHandle) return BinkSource::Handle;
  return BinkSource::Name;
}

bool OpenBinkSource(Ctx& c, uint32_t src, uint32_t open_flags, BinkBytes* bytes,
                    std::string* what) {
  bytes->Clear();
  switch (BinkSourceOf(open_flags)) {
    case BinkSource::Memory:
      *what = "memory";
      return FromMemory(c, src, bytes);
    case BinkSource::Handle: {
      char name[32];
      std::snprintf(name, sizeof name, "file handle %X", src);
      *what = name;
      return FromHandle(src, bytes);
    }
    case BinkSource::Name:
      break;
  }
  *what = GStr(src);
  return FromName(*what, bytes);
}

}  // namespace recomp::hle
