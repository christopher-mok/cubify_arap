#include "thin_walls.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace cubify {

namespace {

inline Eigen::Vector3d row3(const RowMatX3d& M, int i) { return M.row(i).transpose(); }

std::vector<Eigen::Vector3d> vertex_normals(const RowMatX3d& V, const RowMatX3i& F) {
  std::vector<Eigen::Vector3d> n(V.rows(), Eigen::Vector3d::Zero());
  for (Eigen::Index f = 0; f < F.rows(); ++f) {
    const int32_t a = F(f, 0), b = F(f, 1), c = F(f, 2);
    const Eigen::Vector3d fn = (row3(V, b) - row3(V, a)).cross(row3(V, c) - row3(V, a));
    n[a] += fn;
    n[b] += fn;
    n[c] += fn;
  }
  for (auto& v : n) {
    const double len = v.norm();
    v = len > 1e-20 ? Eigen::Vector3d(v / len) : Eigen::Vector3d::Zero();
  }
  return n;
}

// Moller-Trumbore; returns the ray parameter and the hit's (u, v) weights of
// corners b and c.
bool ray_triangle(const Eigen::Vector3d& o, const Eigen::Vector3d& d, const Eigen::Vector3d& a,
                  const Eigen::Vector3d& b, const Eigen::Vector3d& c, double& t, double& u,
                  double& v) {
  const Eigen::Vector3d e1 = b - a, e2 = c - a;
  const Eigen::Vector3d p = d.cross(e2);
  const double det = e1.dot(p);
  if (std::abs(det) < 1e-20) return false;
  const double inv = 1.0 / det;
  const Eigen::Vector3d s = o - a;
  u = s.dot(p) * inv;
  if (u < 0.0 || u > 1.0) return false;
  const Eigen::Vector3d q = s.cross(e1);
  v = d.dot(q) * inv;
  if (v < 0.0 || u + v > 1.0) return false;
  t = e2.dot(q) * inv;
  return t > 0.0;
}

// A vertex paired with an opposite-facing surface: behind it (sigma = +1,
// the two sides of a thin wall) or in front of it (sigma = -1, a narrow gap
// between touching parts or across a narrow opening).
struct Wall {
  int32_t v;       // vertex
  int32_t f;       // opposite face
  double b[3];     // barycentrics of the hit on f
  double t0;       // rest distance
  double sigma;    // +1 wall, -1 gap
};

// Uniform grid of triangles for short ray casts.
class TriangleGrid {
 public:
  TriangleGrid(const RowMatX3d& V, const RowMatX3i& F, double cell) : cell_(cell) {
    lo_ = V.colwise().minCoeff().transpose();
    for (Eigen::Index f = 0; f < F.rows(); ++f) {
      Eigen::Vector3d mn = row3(V, F(f, 0)), mx = mn;
      for (int k = 1; k < 3; ++k) {
        mn = mn.cwiseMin(row3(V, F(f, k)));
        mx = mx.cwiseMax(row3(V, F(f, k)));
      }
      const Eigen::Vector3i a = cell_of(mn), b = cell_of(mx);
      for (int x = a[0]; x <= b[0]; ++x)
        for (int y = a[1]; y <= b[1]; ++y)
          for (int z = a[2]; z <= b[2]; ++z)
            cells_[key(x, y, z)].push_back(static_cast<int32_t>(f));
    }
  }

  template <class Fn>
  void for_each_near_segment(const Eigen::Vector3d& p, const Eigen::Vector3d& q, Fn&& fn) const {
    const Eigen::Vector3i a = cell_of(p.cwiseMin(q)), b = cell_of(p.cwiseMax(q));
    for (int x = a[0]; x <= b[0]; ++x)
      for (int y = a[1]; y <= b[1]; ++y)
        for (int z = a[2]; z <= b[2]; ++z) {
          auto it = cells_.find(key(x, y, z));
          if (it == cells_.end()) continue;
          for (int32_t f : it->second) fn(f);
        }
  }

 private:
  Eigen::Vector3i cell_of(const Eigen::Vector3d& p) const {
    const Eigen::Vector3d c = ((p - lo_) / cell_).array().floor().matrix();
    return Eigen::Vector3i(static_cast<int>(c[0]), static_cast<int>(c[1]),
                           static_cast<int>(c[2]));
  }
  static int64_t key(int x, int y, int z) {
    return (static_cast<int64_t>(x) << 42) ^ (static_cast<int64_t>(y) << 21) ^
           static_cast<int64_t>(z);
  }

  double cell_;
  Eigen::Vector3d lo_;
  std::unordered_map<int64_t, std::vector<int32_t>> cells_;
};

// Signed thickness (walls) or clearance (gaps) along each vertex's normal;
// negative once the surfaces have crossed.
std::vector<double> thickness(const RowMatX3d& V, const RowMatX3i& F,
                              const std::vector<Wall>& walls,
                              const std::vector<Eigen::Vector3d>& n) {
  std::vector<double> d(walls.size());
  for (size_t k = 0; k < walls.size(); ++k) {
    const Wall& w = walls[k];
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    for (int c = 0; c < 3; ++c) p += w.b[c] * row3(V, F(w.f, c));
    d[k] = w.sigma * (row3(V, w.v) - p).dot(n[w.v]);
  }
  return d;
}

}  // namespace

ThinWallResult fix_thin_walls(const RowMatX3d& V_rest, const RowMatX3d& V_in, const RowMatX3i& F,
                              double min_thickness, double max_wall, int iterations,
                              ThreadPool& pool) {
  const int n = static_cast<int>(V_rest.rows());
  if (V_in.rows() != n) throw std::invalid_argument("rest and current vertex counts differ");
  if (n == 0 || F.rows() == 0) throw std::invalid_argument("mesh has no vertices or faces");
  if (F.minCoeff() < 0 || F.maxCoeff() >= n) throw std::invalid_argument("face index out of range");

  ThinWallResult res;
  res.V = V_in;
  const double diag = (V_rest.colwise().maxCoeff() - V_rest.colwise().minCoeff()).norm();
  const double tmax = max_wall * diag;
  if (!(tmax > 0.0)) return res;

  // ---- pair every vertex with the opposite-facing surface behind it (wall)
  // and in front of it (gap), in the rest pose
  const std::vector<Eigen::Vector3d> n0 = vertex_normals(V_rest, F);
  std::vector<Eigen::Vector3d> fn0(F.rows());
  for (Eigen::Index f = 0; f < F.rows(); ++f) {
    Eigen::Vector3d c = (row3(V_rest, F(f, 1)) - row3(V_rest, F(f, 0)))
                            .cross(row3(V_rest, F(f, 2)) - row3(V_rest, F(f, 0)));
    const double len = c.norm();
    fn0[f] = len > 1e-20 ? Eigen::Vector3d(c / len) : Eigen::Vector3d::Zero();
  }
  const TriangleGrid grid(V_rest, F, tmax);
  std::vector<Wall> found(2 * static_cast<size_t>(n));
  std::vector<char> has(2 * static_cast<size_t>(n), 0);
  const double eps = 1e-7 * diag;
  pool.parallel_for(n, 256, [&](int begin, int end) {
    for (int v = begin; v < end; ++v) {
      if (n0[v].isZero()) continue;
      for (int side = 0; side < 2; ++side) {
        const double sigma = side == 0 ? 1.0 : -1.0;
        const Eigen::Vector3d dir = -sigma * n0[v];
        const Eigen::Vector3d o = row3(V_rest, v) + eps * dir;
        double best = tmax;
        int32_t best_f = -1;
        double bu = 0.0, bv = 0.0;
        grid.for_each_near_segment(o, o + tmax * dir, [&](int32_t f) {
          if (F(f, 0) == v || F(f, 1) == v || F(f, 2) == v) return;
          double t, u, w;
          if (ray_triangle(o, dir, row3(V_rest, F(f, 0)), row3(V_rest, F(f, 1)),
                           row3(V_rest, F(f, 2)), t, u, w) &&
              t < best) {
            best = t;
            best_f = f;
            bu = u;
            bv = w;
          }
        });
        // only a surface facing back at us counts (not a fold or a side wall)
        if (best_f < 0 || fn0[best_f].dot(n0[v]) > -0.3) continue;
        found[2 * v + side] = Wall{v, best_f, {1.0 - bu - bv, bu, bv}, best + eps, sigma};
        has[2 * v + side] = 1;
      }
    }
  });
  std::vector<Wall> walls;
  for (size_t k = 0; k < found.size(); ++k)
    if (has[k]) walls.push_back(found[k]);
  res.walls = static_cast<int>(walls.size());
  if (walls.empty()) return res;

  // ---- one-ring adjacency for smoothing the correction
  std::vector<std::vector<int32_t>> nbr(n);
  for (Eigen::Index f = 0; f < F.rows(); ++f)
    for (int k = 0; k < 3; ++k) {
      nbr[F(f, k)].push_back(F(f, (k + 1) % 3));
      nbr[F(f, k)].push_back(F(f, (k + 2) % 3));
    }
  for (auto& l : nbr) {
    std::sort(l.begin(), l.end());
    l.erase(std::unique(l.begin(), l.end()), l.end());
  }

  // ---- push thin or crossed walls apart until none is left
  RowMatX3d& V = res.V;
  std::vector<double> d = thickness(V, F, walls, vertex_normals(V, F));
  for (double x : d) res.crossed_before += x < 0.0;

  RowMatX3d corr(n, 3), next(n, 3);
  std::vector<double> cnt(n);
  for (int pass = 0; pass < iterations; ++pass) {
    const std::vector<Eigen::Vector3d> nv = vertex_normals(V, F);
    d = thickness(V, F, walls, nv);
    corr.setZero();
    std::fill(cnt.begin(), cnt.end(), 0.0);
    bool any = false;
    for (size_t k = 0; k < walls.size(); ++k) {
      const Wall& w = walls[k];
      const double floor_t = min_thickness * w.t0;
      if (d[k] >= floor_t) continue;
      any = true;
      // split the missing distance between this vertex and the opposite side
      const Eigen::RowVector3d gap = (0.5 * w.sigma * (floor_t - d[k])) * nv[w.v].transpose();
      corr.row(w.v) += gap;
      cnt[w.v] += 1.0;
      for (int c = 0; c < 3; ++c) {
        corr.row(F(w.f, c)) -= w.b[c] * gap;
        cnt[F(w.f, c)] += w.b[c];
      }
    }
    if (!any) break;
    res.passes = pass + 1;
    for (int i = 0; i < n; ++i)
      if (cnt[i] > 1.0) corr.row(i) /= cnt[i];
    for (int s = 0; s < 3; ++s) {  // spread the correction so it stays smooth
      pool.parallel_for(n, 1024, [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
          Eigen::RowVector3d avg = Eigen::RowVector3d::Zero();
          for (int32_t j : nbr[i]) avg += corr.row(j);
          next.row(i) = nbr[i].empty() ? Eigen::RowVector3d(corr.row(i))
                                       : Eigen::RowVector3d(0.5 * corr.row(i) +
                                                            0.5 * avg / nbr[i].size());
        }
      });
      corr.swap(next);
    }
    V += corr;
  }

  d = thickness(V, F, walls, vertex_normals(V, F));
  for (double x : d) res.crossed_after += x < 0.0;
  const Eigen::VectorXd moved = (V - V_in).rowwise().norm();
  res.max_move = moved.maxCoeff();
  for (int i = 0; i < n; ++i) res.moved += moved[i] > 1e-5 * diag;
  return res;
}

}  // namespace cubify
