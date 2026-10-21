// Target shapes for the stylization term  lambda * a_i * f(A^T R_i n_i).
//
// For the faceted targets f is the support function of the polytope
//     P = {y : d_k . y <= 1}
// whose face normals d_k are the preferred normal directions. On the unit
// sphere f is smallest (exactly 1) at the d_k, so surfaces snap to face them.
// The cube's P is [-1, 1]^3, whose support function is the paper's L1 norm
// and whose proximal step is soft-thresholding. For any support function
// the proximal step follows from Moreau's identity,
//     prox_{t f}(x) = x - t * proj_P(x / t),
// with proj_P computed exactly by testing P's faces and edges.
//
// The rounded cube uses f(y) = sum_c |y_c|^p, 1 < p < 2, instead: it pulls
// normals toward the cube axes without ever snapping them exactly, which
// rounds edges and corners (p near 1 is almost the cube, p near 2 almost a
// sphere). Roundness r in [0, 1] maps to p = 2 - 0.5 * 10^-r, which spaces
// the look evenly. Its proximal step solves w^q + t p w = |x| for
// w = |z|^(p-1), q = 1 / (p - 1): convex in w, so Newton's method from
// w = |x|^(p-1) descends monotonically onto the root.
//
// Must match cubic_stylization/solver.py (same directions, same formulas).

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cubify {

enum class TargetShape : int32_t {
  kCube = 0,
  kOctahedron = 1,
  kPyramid = 2,
  kHexColumn = 3,
  kRoundedCube = 4,
};

// Pyramid side-face normals sit this far above the horizon: the faces slope
// at ~52 degrees, like the Great Pyramid.
constexpr double kPyramidNormalElevationDeg = 38.0;

inline double rounded_exponent(double roundness) {
  return 2.0 - 0.5 * std::pow(10.0, -std::clamp(roundness, 0.0, 1.0));
}

class Target {
 public:
  // roundness only affects the rounded cube.
  explicit Target(int32_t code = 0, double roundness = 0.5) {
    if (code < 0 || code > 4) throw std::invalid_argument("unknown target shape " + std::to_string(code));
    shape_ = static_cast<TargetShape>(code);
    p_ = rounded_exponent(roundness);
    const double pi = std::acos(-1.0);
    switch (shape_) {
      case TargetShape::kCube:
      case TargetShape::kRoundedCube:
        for (int c = 0; c < 3; ++c)
          for (double s : {1.0, -1.0}) {
            Eigen::Vector3d d = Eigen::Vector3d::Zero();
            d[c] = s;
            dirs_.push_back(d);
          }
        break;
      case TargetShape::kOctahedron:
        for (double sx : {1.0, -1.0})
          for (double sy : {1.0, -1.0})
            for (double sz : {1.0, -1.0})
              dirs_.push_back(Eigen::Vector3d(sx, sy, sz) / std::sqrt(3.0));
        break;
      case TargetShape::kPyramid: {
        const double a = kPyramidNormalElevationDeg * pi / 180.0;
        const double c = std::cos(a), s = std::sin(a);
        dirs_ = {Eigen::Vector3d(c, 0, s), Eigen::Vector3d(-c, 0, s), Eigen::Vector3d(0, c, s),
                 Eigen::Vector3d(0, -c, s), Eigen::Vector3d(0, 0, -1)};
        break;
      }
      case TargetShape::kHexColumn:
        for (int k = 0; k < 6; ++k)
          dirs_.push_back(Eigen::Vector3d(std::cos(k * pi / 3.0), std::sin(k * pi / 3.0), 0.0));
        dirs_.push_back(Eigen::Vector3d(0, 0, 1));
        dirs_.push_back(Eigen::Vector3d(0, 0, -1));
        break;
    }
    if (shape_ != TargetShape::kCube && shape_ != TargetShape::kRoundedCube) build_edges();
  }

  TargetShape shape() const { return shape_; }

  // Proximal step of t * f at x (t >= 0).
  Eigen::Vector3d prox(const Eigen::Vector3d& x, double t) const {
    Eigen::Vector3d z;
    switch (shape_) {
      case TargetShape::kCube:
        for (int c = 0; c < 3; ++c) {
          const double mag = std::max(std::abs(x[c]) - t, 0.0);
          z[c] = x[c] > 0.0 ? mag : (x[c] < 0.0 ? -mag : 0.0);
        }
        return z;
      case TargetShape::kRoundedCube: {
        if (t <= 1e-12) return x;
        const double q = 1.0 / (p_ - 1.0);
        for (int c = 0; c < 3; ++c) {
          const double a = std::abs(x[c]);
          double w = std::pow(a, p_ - 1.0);
          for (int it = 0; it < 50; ++it) {
            const double step = (std::pow(w, q) + t * p_ * w - a) / (q * std::pow(w, q - 1.0) + t * p_);
            w -= step;
            if (std::abs(step) <= 1e-15 * std::max(1.0, w)) break;
          }
          const double mag = std::pow(std::max(w, 0.0), q);
          z[c] = x[c] < 0.0 ? -mag : mag;
        }
        return z;
      }
      default:
        if (t <= 1e-12) return x;
        return x - t * project(x / t);
    }
  }

  // Cosine between the unit vector y (target frame) and the nearest
  // preferred direction: 1 when y faces one exactly.
  double alignment(const Eigen::Vector3d& y) const {
    double best = -1.0;
    for (const auto& d : dirs_) best = std::max(best, d.dot(y));
    return best;
  }

 private:
  // Vertices of P from every triple of face planes; an edge joins two
  // vertices that share at least two tight faces.
  void build_edges() {
    const int K = static_cast<int>(dirs_.size());
    std::vector<Eigen::Vector3d> verts;
    std::vector<std::vector<int>> tight;
    for (int i = 0; i < K; ++i)
      for (int j = i + 1; j < K; ++j)
        for (int k = j + 1; k < K; ++k) {
          Eigen::Matrix3d M;
          M.row(0) = dirs_[i];
          M.row(1) = dirs_[j];
          M.row(2) = dirs_[k];
          if (std::abs(M.determinant()) < 1e-9) continue;
          const Eigen::Vector3d v = M.partialPivLu().solve(Eigen::Vector3d::Ones());
          bool inside = true;
          for (const auto& d : dirs_) inside = inside && d.dot(v) <= 1.0 + 1e-9;
          if (!inside) continue;
          bool dup = false;
          for (const auto& w : verts) dup = dup || (w - v).norm() < 1e-9;
          if (dup) continue;
          std::vector<int> act;
          for (int f = 0; f < K; ++f)
            if (std::abs(dirs_[f].dot(v) - 1.0) < 1e-9) act.push_back(f);
          verts.push_back(v);
          tight.push_back(act);
        }
    for (size_t a = 0; a < verts.size(); ++a)
      for (size_t b = a + 1; b < verts.size(); ++b) {
        int shared = 0;
        for (int f : tight[a]) shared += std::count(tight[b].begin(), tight[b].end(), f);
        if (shared >= 2) edges_.emplace_back(verts[a], verts[b]);
      }
  }

  // Euclidean projection onto P: the nearest of the face-plane projections
  // that land inside P and the clamped edge-segment projections.
  Eigen::Vector3d project(const Eigen::Vector3d& y) const {
    double worst = -std::numeric_limits<double>::infinity();
    for (const auto& d : dirs_) worst = std::max(worst, d.dot(y));
    if (worst <= 1.0) return y;

    Eigen::Vector3d best = y;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (const auto& d : dirs_) {
      const Eigen::Vector3d c = y - (d.dot(y) - 1.0) * d;
      const double lim = 1.0 + 1e-9 * std::max(1.0, c.norm());
      bool inside = true;
      for (const auto& e : dirs_) inside = inside && e.dot(c) <= lim;
      const double d2 = (c - y).squaredNorm();
      if (inside && d2 < best_d2) {
        best_d2 = d2;
        best = c;
      }
    }
    for (const auto& [a, b] : edges_) {
      const Eigen::Vector3d ab = b - a;
      const double s = std::clamp((y - a).dot(ab) / ab.squaredNorm(), 0.0, 1.0);
      const Eigen::Vector3d c = a + s * ab;
      const double d2 = (c - y).squaredNorm();
      if (d2 < best_d2) {
        best_d2 = d2;
        best = c;
      }
    }
    return best;
  }

  TargetShape shape_ = TargetShape::kCube;
  double p_ = 1.75;  // rounded-cube exponent
  std::vector<Eigen::Vector3d> dirs_;  // preferred normals (unit)
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> edges_;  // of P
};

}  // namespace cubify
