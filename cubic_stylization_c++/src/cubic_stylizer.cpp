#include "cubic_stylizer.h"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cubify {

namespace {

// ADMM constants from the paper
constexpr double kRhoInit = 1e-4;
constexpr double kMu = 10.0;
constexpr double kTau = 2.0;
constexpr double kEpsAbs = 1e-5;
constexpr double kEpsRel = 1e-3;

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
                             const std::vector<int32_t>& pins)
    : V0_(V), F_(F), n_(static_cast<int>(V.rows())), lam_(cubeness), A_(cube_axes) {
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
  build_normals_and_areas();
  build_solver();

  // z = A^T n corresponds to the feasible start R = I.
  R_.assign(n_, Eigen::Matrix3d::Identity());
  z_.resize(n_);
  for (int i = 0; i < n_; ++i) z_[i] = A_.transpose() * nhat_[i];
  u_.assign(n_, Eigen::Vector3d::Zero());
  rho_.assign(n_, kRhoInit);
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
  // every connected component that has neither — the energy is
  // translation-invariant per component, so this keeps the system regular.
  // Those auto-anchored components "float": after every global step they are
  // translated back to their rest centroid (see keep_floating_centroids), so
  // disconnected parts of one object keep their relative placement.
  anchored_.assign(n_, 0);
  for (int32_t p : pins_) anchored_[p] = 1;
  for (int i = 0; i < n_; ++i)
    if (deg_[i] <= 1e-12) anchored_[i] = 1;

  UnionFind uf(n_);
  for (int i = 0; i < n_; ++i)
    for (int32_t s = row_start_[i]; s < row_start_[i + 1]; ++s)
      if (col_[s] > i && w_[s] > 1e-12) uf.unite(i, col_[s]);
  std::vector<char> has_anchor(n_, 0);
  for (int i = 0; i < n_; ++i)
    if (anchored_[i]) has_anchor[uf.find(i)] = 1;
  std::vector<int32_t> float_of_root(n_, -1);
  floating_.assign(n_, -1);
  for (int i = 0; i < n_; ++i) {
    int32_t r = uf.find(i);
    if (!has_anchor[r]) {
      anchored_[i] = 1;
      has_anchor[r] = 1;
      float_of_root[r] = num_floating_++;
    }
    floating_[i] = float_of_root[r];
  }
  float_rest_centroid_ = centroids(V0_);

  // Free rows keep only free columns (edges into anchors go to the
  // right-hand side), anchor rows are identity: the matrix is
  // blockdiag(L_ff, I) up to permutation, hence SPD.
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

  ldlt_.compute(L);
  if (ldlt_.info() != Eigen::Success) {
    lu_ = std::make_unique<Eigen::SparseLU<Eigen::SparseMatrix<double>>>();
    lu_->analyzePattern(L);
    lu_->factorize(L);
    if (lu_->info() != Eigen::Success)
      throw std::runtime_error("could not factorize the system matrix (degenerate mesh?)");
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
      const double k = lam * area_[i];  // weight of the L1 term

      for (int it = 0; it < admm_iters; ++it) {
        // R-step: Procrustes on M = S + rho * n (A(z-u))^T
        Eigen::Matrix3d M = S + rho * nh * (A_ * (z - u)).transpose();
        const Eigen::Matrix3d R = fit_rotation(M);
        R_[i] = R;

        // z-step: soft-threshold A^T R n
        const Eigen::Vector3d Rn = A_.transpose() * (R * nh);
        const Eigen::Vector3d x = Rn + u;
        const double thr = k / rho;
        Eigen::Vector3d z_new;
        for (int c = 0; c < 3; ++c) {
          double mag = std::max(std::abs(x[c]) - thr, 0.0);
          z_new[c] = x[c] > 0.0 ? mag : (x[c] < 0.0 ? -mag : 0.0);
        }

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

std::vector<Eigen::Vector3d> CubicStylizer::centroids(const RowMatX3d& V) const {
  std::vector<Eigen::Vector3d> sum(num_floating_, Eigen::Vector3d::Zero());
  std::vector<double> count(num_floating_, 0.0);
  for (int i = 0; i < n_; ++i) {
    if (floating_[i] < 0) continue;
    sum[floating_[i]] += row3(V, i);
    count[floating_[i]] += 1.0;
  }
  for (int c = 0; c < num_floating_; ++c) sum[c] /= count[c];
  return sum;
}

void CubicStylizer::keep_floating_centroids(RowMatX3d& V) const {
  if (num_floating_ == 0) return;
  std::vector<Eigen::Vector3d> cur = centroids(V);
  for (int i = 0; i < n_; ++i) {
    const int32_t c = floating_[i];
    if (c >= 0) V.row(i) += (float_rest_centroid_[c] - cur[c]).transpose();
  }
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
    local_step(V, admm_iters, pool);
    RowMatX3d V_new = global_step(ppos, pool);
    keep_floating_centroids(V_new);
    const double step = (V_new - V).rowwise().norm().maxCoeff();
    V = std::move(V_new);
    done = it + 1;
    if (on_progress) on_progress(done, iterations);
    if (step < 1e-6 * bbox) break;
  }
  if (iterations_done != nullptr) *iterations_done = done;
  return V;
}

void CubicStylizer::recenter(RowMatX3d& V) const {
  const Eigen::RowVector3d shift = V0_.colwise().mean() - V.colwise().mean();
  V.rowwise() += shift;
}

}  // namespace cubify
