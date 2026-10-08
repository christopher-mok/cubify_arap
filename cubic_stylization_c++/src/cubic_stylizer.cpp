#include "cubic_stylizer.h"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace cubify {

namespace {

// ADMM constants from the paper
constexpr double kRhoInit = 1e-4;
constexpr double kMu = 10.0;
constexpr double kTau = 2.0;
constexpr double kEpsAbs = 1e-5;
constexpr double kEpsRel = 1e-3;

// Square flat regions: a vertex counts as aligned when the cosine between
// its normal and the nearest target direction reaches kFlatHi (~8 deg),
// ramping in from kFlatLo (~14 deg).
constexpr double kFlatLo = 0.97;
constexpr double kFlatHi = 0.99;

// Parts closer than this (fraction of the bounding-box diagonal) touch.
constexpr double kContactDistance = 0.005;

// Closest rotation to M (orthogonal Procrustes): R = V U^T for M = U S V^T,
// with the reflection case fixed by flipping the smallest singular vector.
inline Eigen::Matrix3d fit_rotation(const Eigen::Matrix3d& M) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  const Eigen::Matrix3d& Vs = svd.matrixV();
  Eigen::Matrix3d R = Vs * U.transpose();
  if (R.determinant() < 0.0) {
    U.col(2) *= -1.0;
    R = Vs * U.transpose();
  }
  return R;
}

inline Eigen::Vector3d row3(const RowMatX3d& M, int i) { return M.row(i).transpose(); }

struct UnionFind {
  std::vector<int32_t> parent;
  explicit UnionFind(int n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
  int32_t find(int32_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }
  void unite(int32_t a, int32_t b) {
    a = find(a);
    b = find(b);
    if (a != b) parent[std::max(a, b)] = std::min(a, b);
  }
};

}  // namespace

CubicStylizer::CubicStylizer(const RowMatX3d& V, const RowMatX3i& F, double cubeness,
                             const Eigen::Matrix3d& cube_axes,
                             const std::vector<int32_t>& pins, double flat_relax,
                             int32_t target, double roundness, bool keep_orientation,
                             const std::vector<Eigen::Vector3d>& custom_dirs)
    : V0_(V),
      F_(F),
      n_(static_cast<int>(V.rows())),
      lam_(cubeness),
      A_(cube_axes),
      target_(target, roundness, custom_dirs),
      keep_orientation_(keep_orientation),
      flat_relax_(std::clamp(flat_relax, 1e-4, 1.0)) {
  if (n_ == 0 || F_.rows() == 0) throw std::invalid_argument("mesh has no vertices or faces");
  if (F_.minCoeff() < 0 || F_.maxCoeff() >= n_)
    throw std::invalid_argument("face index out of range");
  if (!V0_.allFinite()) throw std::invalid_argument("mesh has non-finite vertex positions");

  pins_ = pins;
  std::sort(pins_.begin(), pins_.end());
  pins_.erase(std::unique(pins_.begin(), pins_.end()), pins_.end());
  if (!pins_.empty() && (pins_.front() < 0 || pins_.back() >= n_))
    throw std::invalid_argument("pin index out of range");

  build_edges();
  w_base_ = w_;
  build_normals_and_areas();
  build_solver();

  reset_state();
}

void CubicStylizer::reset_state() {
  // z = A^T n corresponds to the feasible start R = I.
  R_.assign(n_, Eigen::Matrix3d::Identity());
  z_.resize(n_);
  for (int i = 0; i < n_; ++i) z_[i] = A_.transpose() * nhat_[i];
  u_.assign(n_, Eigen::Vector3d::Zero());
  rho_.assign(n_, kRhoInit);
  has_iterated_ = false;
}

void CubicStylizer::set_style(int32_t target, double roundness,
                              const std::vector<Eigen::Vector3d>& custom_dirs, bool keep_orientation,
                              double flat_relax, const Eigen::Matrix3d& cube_axes) {
  target_ = Target(target, roundness, custom_dirs);  // validates before anything changes
  keep_orientation_ = keep_orientation;
  flat_relax_ = std::clamp(flat_relax, 1e-4, 1.0);
  A_ = cube_axes;
  restore_base_weights();
  reset_state();
}

// ---------------- precomputation ----------------

void CubicStylizer::build_edges() {
  const int64_t n = n_;
  const Eigen::Index m = F_.rows();

  // Half cotangent at each corner weights the opposite edge; summing over the
  // (one or two) faces of an edge gives its cotan weight.
  struct Entry {
    int64_t key;
    double c;
  };
  std::vector<Entry> ent;
  ent.reserve(static_cast<size_t>(3 * m));
  for (Eigen::Index f = 0; f < m; ++f) {
    for (int k = 0; k < 3; ++k) {
      int32_t a = F_(f, k), b = F_(f, (k + 1) % 3), c = F_(f, (k + 2) % 3);
      if (b == c) continue;  // degenerate face: no edge
      Eigen::Vector3d u = row3(V0_, b) - row3(V0_, a);
      Eigen::Vector3d v = row3(V0_, c) - row3(V0_, a);
      double cr = u.cross(v).norm();
      double cot = u.dot(v) / std::max(cr, 1e-12);
      int64_t lo = std::min(b, c), hi = std::max(b, c);
      ent.push_back({lo * n + hi, 0.5 * cot});
    }
  }
  std::sort(ent.begin(), ent.end(),
            [](const Entry& x, const Entry& y) { return x.key < y.key; });

  std::vector<int32_t> ua, ub;
  std::vector<double> uw;
  for (size_t s = 0; s < ent.size();) {
    size_t t = s;
    double sum = 0.0;
    while (t < ent.size() && ent[t].key == ent[s].key) sum += ent[t++].c;
    ua.push_back(static_cast<int32_t>(ent[s].key / n));
    ub.push_back(static_cast<int32_t>(ent[s].key % n));
    uw.push_back(std::abs(sum));
    s = t;
  }

  // directed edges, both ways, grouped by source vertex
  row_start_.assign(n_ + 1, 0);
  for (size_t e = 0; e < ua.size(); ++e) {
    ++row_start_[ua[e] + 1];
    ++row_start_[ub[e] + 1];
  }
  for (int i = 0; i < n_; ++i) row_start_[i + 1] += row_start_[i];

  const size_t E = static_cast<size_t>(row_start_[n_]);
  col_.resize(E);
  w_.resize(E);
  e0_.resize(E);
  std::vector<int32_t> fill(row_start_.begin(), row_start_.end() - 1);
  auto put = [&](int32_t i, int32_t j, double w) {
    int32_t s = fill[i]++;
    col_[s] = j;
    w_[s] = w;
    e0_[s] = row3(V0_, i) - row3(V0_, j);
  };
  for (size_t e = 0; e < ua.size(); ++e) {
    put(ua[e], ub[e], uw[e]);
    put(ub[e], ua[e], uw[e]);
  }

  deg_.assign(n_, 0.0);
  for (int i = 0; i < n_; ++i)
    for (int32_t s = row_start_[i]; s < row_start_[i + 1]; ++s) deg_[i] += w_[s];
}

void CubicStylizer::build_normals_and_areas() {
  area_.assign(n_, 0.0);
  std::vector<Eigen::Vector3d> nrm(n_, Eigen::Vector3d::Zero());
  for (Eigen::Index f = 0; f < F_.rows(); ++f) {
    int32_t a = F_(f, 0), b = F_(f, 1), c = F_(f, 2);
    Eigen::Vector3d fn = (row3(V0_, b) - row3(V0_, a)).cross(row3(V0_, c) - row3(V0_, a));
    double third = 0.5 * fn.norm() / 3.0;  // 2*area*normal -> area / 3
    for (int32_t v : {a, b, c}) {
      area_[v] += third;
      nrm[v] += fn;
    }
  }
  nhat_.resize(n_);
  for (int i = 0; i < n_; ++i) {
    double len = nrm[i].norm();
    // isolated/degenerate: the cubeness term vanishes
    nhat_[i] = len < 1e-12 ? Eigen::Vector3d::Zero() : Eigen::Vector3d(nrm[i] / len);
  }
}

void CubicStylizer::build_solver() {
  // Anchors: user pins, loose vertices (no incident face), and one vertex of
  // every connected part that has neither. The energy is translation-
  // invariant per part, so this keeps the system regular; the anchor's
  // position is arbitrary and overridden by place_floating_parts.
  anchored_.assign(n_, 0);
  for (int32_t p : pins_) anchored_[p] = 1;
  for (int i = 0; i < n_; ++i)
    if (deg_[i] <= 1e-12) anchored_[i] = 1;

  UnionFind uf(n_);
  for (int i = 0; i < n_; ++i)
    for (int32_t s = row_start_[i]; s < row_start_[i + 1]; ++s)
      if (col_[s] > i && w_[s] > 1e-12) uf.unite(i, col_[s]);

  // dense part ids; loose vertices belong to no part
  std::vector<int32_t> comp(n_, -1), id_of_root(n_, -1);
  int ncomp = 0;
  for (int i = 0; i < n_; ++i) {
    if (deg_[i] <= 1e-12) continue;
    int32_t r = uf.find(i);
    if (id_of_root[r] < 0) id_of_root[r] = ncomp++;
    comp[i] = id_of_root[r];
  }
  std::vector<char> pinned(ncomp, 0), seen(ncomp, 0);
  for (int32_t p : pins_)
    if (comp[p] >= 0) pinned[comp[p]] = 1;
  for (int i = 0; i < n_; ++i) {
    const int32_t c = comp[i];
    if (c >= 0 && !pinned[c] && !seen[c]) {
      anchored_[i] = 1;
      seen[c] = 1;
    }
  }

  build_part_placement(comp, pinned);
  factorize(true);
}

void CubicStylizer::build_part_placement(const std::vector<int32_t>& comp,
                                         const std::vector<char>& pinned) {
  const int ncomp = static_cast<int>(pinned.size());
  std::vector<std::vector<int32_t>> verts(ncomp);
  for (int i = 0; i < n_; ++i)
    if (comp[i] >= 0) verts[comp[i]].push_back(i);

  // Contacts: every vertex of an unpinned part paired with its nearest
  // vertex of another part within kContactDistance (uniform grid search).
  std::vector<std::vector<std::pair<int32_t, int32_t>>> contacts(ncomp);
  const double diag = (V0_.colwise().maxCoeff() - V0_.colwise().minCoeff()).norm();
  const double tau = kContactDistance * diag;
  if (ncomp > 1 && tau > 0.0) {
    const Eigen::RowVector3d lo = V0_.colwise().minCoeff();
    auto cell_of = [&](int i) {
      Eigen::RowVector3d c = ((V0_.row(i) - lo) / tau).array().floor().matrix();
      return Eigen::Vector3i(static_cast<int>(c[0]), static_cast<int>(c[1]),
                             static_cast<int>(c[2]));
    };
    auto key = [](int x, int y, int z) {
      return (static_cast<int64_t>(x) << 42) ^ (static_cast<int64_t>(y) << 21) ^
             static_cast<int64_t>(z);
    };
    std::unordered_map<int64_t, std::vector<int32_t>> grid;
    for (int i = 0; i < n_; ++i) {
      if (comp[i] < 0) continue;
      Eigen::Vector3i c = cell_of(i);
      grid[key(c[0], c[1], c[2])].push_back(i);
    }
    for (int i = 0; i < n_; ++i) {
      if (comp[i] < 0 || pinned[comp[i]]) continue;
      Eigen::Vector3i c = cell_of(i);
      int32_t best = -1;
      double best_d2 = tau * tau;
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dz = -1; dz <= 1; ++dz) {
            auto it = grid.find(key(c[0] + dx, c[1] + dy, c[2] + dz));
            if (it == grid.end()) continue;
            for (int32_t u : it->second) {
              if (comp[u] == comp[i]) continue;
              const double d2 = (V0_.row(u) - V0_.row(i)).squaredNorm();
              if (d2 < best_d2) {
                best_d2 = d2;
                best = u;
              }
            }
          }
      if (best >= 0) contacts[comp[i]].push_back({i, best});
    }
  }

  // touched_by[c]: parts with contacts into part c
  std::vector<std::vector<int32_t>> touched_by(ncomp);
  for (int c = 0; c < ncomp; ++c)
    for (const auto& vu : contacts[c]) touched_by[comp[vu.second]].push_back(c);
  for (auto& t : touched_by) {
    std::sort(t.begin(), t.end());
    t.erase(std::unique(t.begin(), t.end()), t.end());
  }

  // Placement order: pinned parts are held by their pins; a part touching
  // placed geometry follows it (breadth-first); when nothing left touches,
  // the largest remaining part keeps its rest centroid.
  std::vector<char> placed(pinned.begin(), pinned.end()), queued(ncomp, 0);
  std::deque<int32_t> queue;
  auto enqueue_touching = [&](int c) {
    for (int32_t d : touched_by[c])
      if (!placed[d] && !queued[d]) {
        queued[d] = 1;
        queue.push_back(d);
      }
  };
  auto place = [&](int c, bool by_contact) {
    Part part;
    part.verts = std::move(verts[c]);
    for (int32_t v : part.verts) part.rest_centroid += row3(V0_, v);
    part.rest_centroid /= static_cast<double>(part.verts.size());
    if (by_contact) {
      for (const auto& vu : contacts[c])
        if (placed[comp[vu.second]]) {
          part.contacts.push_back(vu);
          part.rest_offset += row3(V0_, vu.first) - row3(V0_, vu.second);
        }
      if (!part.contacts.empty())
        part.rest_offset /= static_cast<double>(part.contacts.size());
    }
    part.by_contact = !part.contacts.empty();
    placed[c] = 1;
    parts_.push_back(std::move(part));
    enqueue_touching(c);
  };

  std::vector<int32_t> by_size;
  for (int c = 0; c < ncomp; ++c) {
    if (pinned[c])
      enqueue_touching(c);
    else
      by_size.push_back(c);
  }
  std::stable_sort(by_size.begin(), by_size.end(), [&](int32_t a, int32_t b) {
    return verts[a].size() > verts[b].size();
  });
  size_t next_seed = 0;
  for (;;) {
    while (!queue.empty()) {
      const int32_t c = queue.front();
      queue.pop_front();
      if (!placed[c]) place(c, true);
    }
    while (next_seed < by_size.size() && placed[by_size[next_seed]]) ++next_seed;
    if (next_seed == by_size.size()) break;
    place(by_size[next_seed], false);
  }
}

void CubicStylizer::factorize(bool analyze) {
  // Free rows keep only free columns (edges into anchors go to the
  // right-hand side), anchor rows are identity: the matrix is
  // blockdiag(L_ff, I) up to permutation, hence SPD. The sparsity pattern
  // never changes, so reweighting only refactorizes numerically.
  std::vector<Eigen::Triplet<double>> T;
  T.reserve(col_.size() + n_);
  for (int i = 0; i < n_; ++i) {
    if (anchored_[i]) {
      T.emplace_back(i, i, 1.0);
      continue;
    }
    T.emplace_back(i, i, deg_[i]);
    for (int32_t s = row_start_[i]; s < row_start_[i + 1]; ++s)
      if (!anchored_[col_[s]]) T.emplace_back(i, col_[s], -w_[s]);
  }
  Eigen::SparseMatrix<double> L(n_, n_);
  L.setFromTriplets(T.begin(), T.end());
  L.makeCompressed();

  if (!lu_) {
    if (analyze) ldlt_.analyzePattern(L);
    ldlt_.factorize(L);
    if (ldlt_.info() == Eigen::Success) return;
    lu_ = std::make_unique<Eigen::SparseLU<Eigen::SparseMatrix<double>>>();
    analyze = true;
  }
  if (analyze) lu_->analyzePattern(L);
  lu_->factorize(L);
  if (lu_->info() != Eigen::Success)
    throw std::runtime_error("could not factorize the system matrix (degenerate mesh?)");
}

void CubicStylizer::reweight_flat_regions(const RowMatX3d& V) {
  std::vector<Eigen::Vector3d> nrm(n_, Eigen::Vector3d::Zero());
  for (Eigen::Index f = 0; f < F_.rows(); ++f) {
    const int32_t a = F_(f, 0), b = F_(f, 1), c = F_(f, 2);
    const Eigen::Vector3d fn = (row3(V, b) - row3(V, a)).cross(row3(V, c) - row3(V, a));
    nrm[a] += fn;
    nrm[b] += fn;
    nrm[c] += fn;
  }
  // g = 1 where the vertex normal faces a target direction, 0 where it does not
  std::vector<double> g(n_, 0.0);
  for (int i = 0; i < n_; ++i) {
    const double len = nrm[i].norm();
    if (len < 1e-12) continue;
    const double al = target_.alignment(A_.transpose() * nrm[i] / len);
    g[i] = std::clamp((al - kFlatLo) / (kFlatHi - kFlatLo), 0.0, 1.0);
  }
  for (int i = 0; i < n_; ++i) {
    deg_[i] = 0.0;
    for (int32_t s = row_start_[i]; s < row_start_[i + 1]; ++s) {
      const double f = 1.0 - (1.0 - flat_relax_) * std::min(g[i], g[col_[s]]);
      w_[s] = w_base_[s] * f;
      deg_[i] += w_[s];
    }
  }
  weights_relaxed_ = true;
  factorize(false);
}

void CubicStylizer::restore_base_weights() {
  if (!weights_relaxed_) return;
  w_ = w_base_;
  for (int i = 0; i < n_; ++i) {
    deg_[i] = 0.0;
    for (int32_t s = row_start_[i]; s < row_start_[i + 1]; ++s) deg_[i] += w_[s];
  }
  weights_relaxed_ = false;
  factorize(false);
}

// ARAP is unchanged by a rotation of the whole mesh, so without pins the
// stylization term alone decides the mesh's orientation and may turn it as a
// whole (Suzanne's head turns 10-14 degrees under the cube or octahedron).
// Undoing the area-weighted best-fit rotation from the rest pose
// costs no ARAP energy and makes the target shape come from reshaping.
void CubicStylizer::remove_net_rotation(RowMatX3d& V) const {
  double total = 0.0;
  Eigen::Vector3d c0 = Eigen::Vector3d::Zero(), c = Eigen::Vector3d::Zero();
  for (int i = 0; i < n_; ++i) {
    total += area_[i];
    c0 += area_[i] * row3(V0_, i);
    c += area_[i] * row3(V, i);
  }
  if (total <= 0.0) return;
  c0 /= total;
  c /= total;
  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  for (int i = 0; i < n_; ++i)
    H.noalias() += area_[i] * (row3(V0_, i) - c0) * (row3(V, i) - c).transpose();
  const Eigen::Matrix3d Rt = fit_rotation(H).transpose();  // fit_rotation(H): rest -> current
  for (int i = 0; i < n_; ++i) V.row(i) = (Rt * (row3(V, i) - c) + c).transpose();
}

void CubicStylizer::place_floating_parts(RowMatX3d& V) const {
  for (const Part& part : parts_) {
    Eigen::Vector3d t;
    if (part.by_contact) {
      // least-squares translation restoring the mean contact offset; the
      // touched vertices were placed earlier in this same pass
      Eigen::Vector3d cur = Eigen::Vector3d::Zero();
      for (const auto& vu : part.contacts) cur += row3(V, vu.first) - row3(V, vu.second);
      t = part.rest_offset - cur / static_cast<double>(part.contacts.size());
    } else {
      Eigen::Vector3d c = Eigen::Vector3d::Zero();
      for (int32_t v : part.verts) c += row3(V, v);
      t = part.rest_centroid - c / static_cast<double>(part.verts.size());
    }
    for (int32_t v : part.verts) V.row(v) += t.transpose();
  }
}

// ---------------- solver pieces ----------------

void CubicStylizer::local_step(const RowMatX3d& V, int admm_iters, ThreadPool& pool) {
  const bool admm = lam_ > 0.0;
  const double lam = lam_;
  const double sqrt3 = std::sqrt(3.0);

  pool.parallel_for(n_, 64, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      // ARAP covariance S_i = sum_j w_ij d_ij d'_ij^T over the one-ring
      Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
      const Eigen::Vector3d vi = row3(V, i);
      for (int32_t s = row_start_[i]; s < row_start_[i + 1]; ++s)
        S.noalias() += (w_[s] * e0_[s]) * (vi - row3(V, col_[s])).transpose();

      if (!admm) {  // classic ARAP: plain orthogonal Procrustes
        R_[i] = fit_rotation(S);
        continue;
      }

      const Eigen::Vector3d& nh = nhat_[i];
      Eigen::Vector3d& z = z_[i];
      Eigen::Vector3d& u = u_[i];
      double& rho = rho_[i];
      const double k = lam * area_[i];  // weight of the stylization term

      for (int it = 0; it < admm_iters; ++it) {
        // R-step: Procrustes on M = S + rho * n (A(z-u))^T
        Eigen::Matrix3d M = S + rho * nh * (A_ * (z - u)).transpose();
        const Eigen::Matrix3d R = fit_rotation(M);
        R_[i] = R;

        // z-step: proximal step of the target term at A^T R n + u
        // (soft-thresholding for the cube)
        const Eigen::Vector3d Rn = A_.transpose() * (R * nh);
        const Eigen::Vector3d x = Rn + u;
        const Eigen::Vector3d z_new = target_.prox(x, k / rho);

        // scaled dual update
        Eigen::Vector3d u_new = u + Rn - z_new;

        const double r_pri = (Rn - z_new).norm();
        const double s_dua = rho * (z_new - z).norm();

        // penalty update (Boyd et al. 2011)
        double rho_new = rho;
        if (r_pri > kMu * s_dua) {
          rho_new *= kTau;
          u_new /= kTau;
        } else if (s_dua > kMu * r_pri) {
          rho_new /= kTau;
          u_new *= kTau;
        }

        z = z_new;
        u = u_new;
        rho = rho_new;

        const double eps_pri = sqrt3 * kEpsAbs + kEpsRel * std::max(Rn.norm(), z_new.norm());
        const double eps_dua = sqrt3 * kEpsAbs + kEpsRel * rho_new * u_new.norm();
        if (r_pri < eps_pri && s_dua < eps_dua) break;
      }
    }
  });
}

RowMatX3d CubicStylizer::global_step(const RowMatX3d& ppos, ThreadPool& pool) {
  // b_i = sum_j w_ij/2 (R_i + R_j) e0_ij, plus edges into anchors moved over
  // from the left-hand side; anchor rows take their prescribed position.
  Eigen::MatrixXd b(n_, 3);
  pool.parallel_for(n_, 256, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      if (anchored_[i]) {
        b.row(i) = ppos.row(i);
        continue;
      }
      Eigen::Vector3d bi = Eigen::Vector3d::Zero();
      for (int32_t s = row_start_[i]; s < row_start_[i + 1]; ++s) {
        const int32_t j = col_[s];
        bi += (0.5 * w_[s]) * ((R_[i] + R_[j]) * e0_[s]);
        if (anchored_[j]) bi += w_[s] * row3(ppos, j);
      }
      b.row(i) = bi.transpose();
    }
  });
  Eigen::MatrixXd x = lu_ ? Eigen::MatrixXd(lu_->solve(b)) : Eigen::MatrixXd(ldlt_.solve(b));
  return RowMatX3d(x);
}

// ---------------- drivers ----------------

RowMatX3d CubicStylizer::solve(const RowMatX3d* pin_pos, const RowMatX3d* V_init,
                               int iterations, int admm_iters, ThreadPool& pool,
                               const ProgressFn& on_progress, int* iterations_done) {
  RowMatX3d ppos = V0_;
  if (pin_pos != nullptr && !pins_.empty()) {
    if (pin_pos->rows() != static_cast<Eigen::Index>(pins_.size()))
      throw std::invalid_argument("pin_pos has the wrong number of rows");
    for (size_t k = 0; k < pins_.size(); ++k) ppos.row(pins_[k]) = pin_pos->row(k);
  }

  if (V_init != nullptr && V_init->rows() != n_)
    throw std::invalid_argument("V_init has the wrong number of rows");
  RowMatX3d V = V_init != nullptr ? *V_init : V0_;
  const double bbox = (V0_.colwise().maxCoeff() - V0_.colwise().minCoeff()).norm();

  int done = 0;
  for (int it = 0; it < iterations; ++it) {
    // The first iteration of a session runs with plain weights: relaxing
    // regions flat in the *rest* pose (e.g. the tip of a knob, around a
    // high-valence pole) lets them collapse before anything has cubified.
    if (flat_relax_ < 1.0 && lam_ > 0.0 && has_iterated_)
      reweight_flat_regions(V);
    else
      restore_base_weights();
    local_step(V, admm_iters, pool);
    RowMatX3d V_new = global_step(ppos, pool);
    if (keep_orientation_ && pins_.empty()) remove_net_rotation(V_new);
    place_floating_parts(V_new);
    has_iterated_ = true;
    const double step = (V_new - V).rowwise().norm().maxCoeff();
    V = std::move(V_new);
    done = it + 1;
    if (on_progress) on_progress(done, iterations);
    if (step < 1e-6 * bbox) break;
  }
  if (iterations_done != nullptr) *iterations_done = done;
  return V;
}

}  // namespace cubify
