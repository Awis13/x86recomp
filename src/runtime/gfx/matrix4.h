#pragma once
// 4x4 matrices as 16 doubles, row after row (Direct3D's row vectors:
// p' = p * m).

#include <cmath>
#include <utility>

namespace recomp::gfx {

inline void Mul4(const double* a, const double* b, double* o) {
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < 4; k++) {
      double v = 0;
      for (int i = 0; i < 4; i++) v += a[r * 4 + i] * b[i * 4 + k];
      o[r * 4 + k] = v;
    }
}

// Gauss-Jordan with partial pivoting; false if singular (a pivot below
// 1e-12, or NaN).
inline bool Invert4(const double* m, double* o) {
  double a[4][8];
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < 8; k++) a[r][k] = k < 4 ? m[r * 4 + k] : (k - 4 == r ? 1.0 : 0.0);
  for (int col = 0; col < 4; col++) {
    int piv = col;
    for (int r = col + 1; r < 4; r++)
      if (std::fabs(a[r][col]) > std::fabs(a[piv][col])) piv = r;
    if (!(std::fabs(a[piv][col]) >= 1e-12)) return false;
    if (piv != col)
      for (int k = 0; k < 8; k++) std::swap(a[piv][k], a[col][k]);
    const double inv = 1.0 / a[col][col];
    for (int k = 0; k < 8; k++) a[col][k] *= inv;
    for (int r = 0; r < 4; r++)
      if (r != col) {
        const double f = a[r][col];
        for (int k = 0; k < 8; k++) a[r][k] -= f * a[col][k];
      }
  }
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < 4; k++) o[r * 4 + k] = a[r][k + 4];
  return true;
}

}  // namespace recomp::gfx
