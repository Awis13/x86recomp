#pragma once
// The frame's camera (world to clip, D3D's row vectors) from the scene's
// draws: each draw's world^-1 * world-view-projection, taken from the
// registers a game names for them. A draw whose world register holds
// something else (a skinned mesh's bone matrix, a part's matrix relative to
// its parent, an identity a shader ignores) gives a matrix that is no
// camera at all, and which such draw comes first changes with what is in
// view. So a camera is taken only once two draws with different world
// matrices give the same one (to a relative 1e-4 of its largest entry);
// until then each distinct product is a candidate (kCandidates kept, a
// count each). Settle, where the frame needs a camera before two agreed,
// takes the candidate most draws gave (the first of them on a tie),
// marked unconfirmed.

#include <cmath>
#include <cstdint>
#include <cstring>

#include "runtime/gfx/matrix4.h"

namespace recomp::gfx {

class CameraVote {
 public:
  static constexpr int kCandidates = 8;
  enum class State : uint8_t { None, Confirmed, Unconfirmed };

  void Reset() {
    n_ = 0;
    state_ = State::None;
    rejected_ = 0;
    offered_ = 0;
    worst_ = 0;
  }
  // A draw's world (w) and world-view-projection (wvp) rows. True when this
  // draw confirms a camera (camera() then holds it); false otherwise
  // (including once a camera is taken: the vote is over for the frame).
  bool Offer(const float (*w)[4], const float (*wvp)[4]) {
    if (state_ != State::None) return false;
    double wd[16], m[16], iw[16], vp[16];
    for (int r = 0; r < 4; r++)
      for (int k = 0; k < 4; k++) {
        wd[r * 4 + k] = w[r][k];
        m[r * 4 + k] = wvp[r][k];
      }
    if (!Invert4(wd, iw)) return false;
    Mul4(iw, m, vp);
    for (double v : vp)
      if (!std::isfinite(v)) return false;
    offered_++;
    for (int i = 0; i < n_; i++) {
      Candidate& c = c_[i];
      if (!Same(c.vp, vp)) continue;
      c.count++;
      if (std::memcmp(c.world, w, sizeof c.world) != 0) {
        Take(i, State::Confirmed);
        return true;
      }
      return false;
    }
    if (n_ < kCandidates) {
      Candidate& c = c_[n_++];
      std::memcpy(c.vp, vp, sizeof vp);
      std::memcpy(c.world, w, sizeof c.world);
      c.count = 1;
    }
    return false;
  }
  // Where a camera is needed now: the confirmed one, else the candidate
  // most draws gave (unconfirmed), else none.
  State Settle() {
    if (state_ == State::None && n_ > 0) {
      int best = 0;
      for (int i = 1; i < n_; i++)
        if (c_[i].count > c_[best].count) best = i;
      Take(best, State::Unconfirmed);
    }
    return state_;
  }
  State state() const { return state_; }
  const double* camera() const { return vp_; }
  // Draws offered before the camera was taken whose product was not it.
  uint64_t rejected() const { return rejected_; }
  // Of those, the largest difference from the camera (relative to its
  // largest entry).
  double worst() const { return worst_; }
  // Whether two cameras are the same (to a relative 1e-4).
  static bool Same(const double* a, const double* b) {
    double big = 0, diff = 0;
    for (int k = 0; k < 16; k++) {
      big = std::fmax(big, std::fmax(std::fabs(a[k]), std::fabs(b[k])));
      diff = std::fmax(diff, std::fabs(a[k] - b[k]));
    }
    return diff <= 1e-4 * big;
  }
  static double Difference(const double* a, const double* b) {
    double big = 0, diff = 0;
    for (int k = 0; k < 16; k++) {
      big = std::fmax(big, std::fabs(a[k]));
      diff = std::fmax(diff, std::fabs(a[k] - b[k]));
    }
    return big > 0 ? diff / big : diff;
  }

 private:
  struct Candidate {
    double vp[16];
    float world[4][4];
    uint64_t count;
  };
  void Take(int i, State s) {
    std::memcpy(vp_, c_[i].vp, sizeof vp_);
    state_ = s;
    uint64_t agreed = c_[i].count;
    rejected_ = offered_ - agreed;
    for (int k = 0; k < n_; k++)
      if (k != i) worst_ = std::fmax(worst_, Difference(c_[i].vp, c_[k].vp));
  }
  Candidate c_[kCandidates];
  int n_ = 0;
  State state_ = State::None;
  double vp_[16] = {};
  uint64_t rejected_ = 0, offered_ = 0;
  double worst_ = 0;
};

}  // namespace recomp::gfx
