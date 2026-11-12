#pragma once
// Where BinkOpen's movie comes from. Its first argument is a file name, or
// with BINKFROMMEMORY a pointer to the whole file in memory, or with
// BINKFILEHANDLE a Win32 file handle the game opened and positioned at the
// movie (one of several in a resource file). Bink neither opens nor closes
// such a handle: it stays the game's.

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "runtime/runtime.h"

namespace recomp::hle {

inline constexpr uint32_t kBinkFileHandle = 0x00800000, kBinkFromMemory = 0x04000000;

enum class BinkSource { Name, Memory, Handle };
BinkSource BinkSourceOf(uint32_t open_flags);

// The bytes of an open movie, for as long as it is open. By name or from
// memory they are a copy. A handle's movie may be a gigabyte inside a
// resource file, so it is neither copied nor read when opened: its part of
// the file is mapped read-only, pages come in as the decoder gets to them,
// and the mapping goes with this object. A mapping does not depend on the
// handle: the game may move or close it while the movie plays.
class BinkBytes {
 public:
  BinkBytes() = default;
  ~BinkBytes() { Clear(); }
  BinkBytes(const BinkBytes&) = delete;
  BinkBytes& operator=(const BinkBytes&) = delete;

  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }

  void Clear();
  void Own(std::vector<uint8_t> copy);
  // `size` bytes that start `delta` bytes into the mapping at `map` (a
  // mapping starts at a page of the file, a movie anywhere).
  void Map(void* map, size_t delta, size_t size);

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
  std::vector<uint8_t> copy_;
  void* map_ = nullptr;
  size_t map_size_ = 0;
};

// The movie BinkOpen(src, open_flags) names into *bytes, and what to call it
// in the log into *what. A handle's movie starts at its file position (its
// length: the one in the Bink header there, or what the file has left of
// it); the position stays where the game put it. False, with a warning
// logged and *bytes empty, if there is no movie: a file that cannot be
// opened, no Bink header in memory or at the handle's position, a handle
// that is not an open file, a header that cannot be read, a file that
// cannot be mapped (no fallback to reading it whole: that is the cost
// mapping is there to avoid, and where it cannot be afforded it would end
// the process rather than the movie).
bool OpenBinkSource(Ctx& c, uint32_t src, uint32_t open_flags, BinkBytes* bytes,
                    std::string* what);

// The system calls made on a handle's file: pread for the 8 bytes of the
// Bink header, mmap and munmap for the movie. A test puts its own here to
// count them and to make them fail.
struct BinkFileCalls {
  ssize_t (*pread)(int fd, void* buf, size_t n, off_t at);
  void* (*mmap)(void* addr, size_t n, int prot, int flags, int fd, off_t at);
  int (*munmap)(void* addr, size_t n);
};
extern BinkFileCalls g_bink_file;

}  // namespace recomp::hle
