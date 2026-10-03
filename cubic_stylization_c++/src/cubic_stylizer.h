// Cubic Stylization / ARAP solver (Liu & Jacobson, SIGGRAPH Asia 2019), C++/Eigen.
//
// A port of cubic_stylization/solver.py. Minimizes the ARAP energy plus the
// L1 cubeness term
//     sum_i sum_{j in N(i)} (w_ij / 2) ||R_i d_ij - d'_ij||^2
//   + sum_i lambda * a_i * ||A^T R_i n_i||_1
// with local-global iterations. With cubeness = 0 this is classic ARAP. The
// local step runs the paper's per-vertex ADMM (Algorithm 1), multithreaded
// over vertices (a plain Procrustes fit when cubeness = 0); the global step
// is a prefactorized sparse Cholesky (LDLT) solve of the cotan Laplacian.
//
// Vertices may be pinned to prescribed positions ("handles"). The system
// matrix depends only on the pin *set*, so it is factorized once and pins can
// then be dragged interactively: each drag only rebuilds the right-hand side.

#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "thread_pool.h"

namespace cubify {

using RowMatX3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using RowMatX3i = Eigen::Matrix<int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;
using ProgressFn = std::function<void(int done, int total)>;

class CubicStylizer {
 public:
  // V: rest-pose positions; F: triangles (a virtual triangulation of a
  // quad/n-gon mesh works fine: only vertex positions are solved for);
  // cube_axes: rotation whose columns are the target cube axes;
  // pins: vertex indices to constrain (duplicates are fine).
  CubicStylizer(const RowMatX3d& V, const RowMatX3i& F, double cubeness,
                const Eigen::Matrix3d& cube_axes, const std::vector<int32_t>& pins);

  // Local-global iterations; returns the (n, 3) positions.
  //   pin_pos: (pins().size(), 3) targets for the pinned vertices, or null to
  //            hold them at rest
  //   V_init:  warm-start positions, or null to start from the rest pose
  RowMatX3d solve(const RowMatX3d* pin_pos, const RowMatX3d* V_init, int iterations,
                  int admm_iters, ThreadPool& pool, const ProgressFn& on_progress = nullptr,
                  int* iterations_done = nullptr);

  // Keep the result centred where the input was (one-shot stylization with
  // nothing pinned: the energy is translation-invariant).
  void recenter(RowMatX3d& V) const;

  int num_vertices() const { return n_; }
  const std::vector<int32_t>& pins() const { return pins_; }
  double cubeness() const { return lam_; }
  void set_cubeness(double lam) { lam_ = lam; }

 private:
  void build_edges();
  void build_normals_and_areas();
  void build_solver();
  void local_step(const RowMatX3d& V, int admm_iters, ThreadPool& pool);
  RowMatX3d global_step(const RowMatX3d& ppos, ThreadPool& pool);
  std::vector<Eigen::Vector3d> centroids(const RowMatX3d& V) const;
  void keep_floating_centroids(RowMatX3d& V) const;

  RowMatX3d V0_;
  RowMatX3i F_;
  int n_;
  double lam_;
  Eigen::Matrix3d A_;
  std::vector<int32_t> pins_;

  // directed one-ring edges in CSR order: row i holds every spoke i -> j
  std::vector<int32_t> row_start_, col_;
  std::vector<double> w_, deg_;
  std::vector<Eigen::Vector3d> e0_;  // rest-pose edge vectors V0_i - V0_j

  std::vector<Eigen::Vector3d> nhat_;
  std::vector<double> area_;

  // Global step: pinned rows become identity and edges into pins move to
  // the right-hand side, giving a symmetric positive definite system.
  std::vector<char> anchored_;
  // connected components with no pin: index per vertex (-1 = held by a pin
  // or loose) and rest-pose centroid per component
  std::vector<int32_t> floating_;
  int num_floating_ = 0;
  std::vector<Eigen::Vector3d> float_rest_centroid_;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_;
  std::unique_ptr<Eigen::SparseLU<Eigen::SparseMatrix<double>>> lu_;  // fallback

  // per-vertex rotations and ADMM state (warm-started across iterations)
  std::vector<Eigen::Matrix3d> R_;
  std::vector<Eigen::Vector3d> z_, u_;
  std::vector<double> rho_;
};

}  // namespace cubify
