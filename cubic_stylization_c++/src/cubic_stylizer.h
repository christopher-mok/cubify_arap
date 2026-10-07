// Cubic Stylization / ARAP solver (Liu & Jacobson, SIGGRAPH Asia 2019), C++/Eigen.
//
// A port of cubic_stylization/solver.py. Minimizes the ARAP energy plus the
// stylization term
//     sum_i sum_{j in N(i)} (w_ij / 2) ||R_i d_ij - d'_ij||^2
//   + sum_i lambda * a_i * f(A^T R_i n_i)
// with local-global iterations, where f is the target shape's term (the
// paper's L1 norm for the cube; see target_shape.h). With cubeness = 0 this is classic ARAP. The
// local step runs the paper's per-vertex ADMM (Algorithm 1), multithreaded
// over vertices (a plain Procrustes fit when cubeness = 0); the global step
// is a prefactorized sparse Cholesky (LDLT) solve of the cotan Laplacian.
//
// Vertices may be pinned to prescribed positions ("handles"). The system
// matrix depends only on the pin *set*, so it is factorized once and pins can
// then be dragged interactively: each drag only rebuilds the right-hand side.
//
// Two extensions beyond the paper:
//  - Disconnected parts. The energy does not couple separate parts, so each
//    part without pins is placed by translation after every iteration: a
//    part touching already-placed geometry keeps its contact offsets (a lid
//    stays seated on its pot), otherwise the largest part keeps its rest
//    centroid. Translations never change a part's shape.
//  - Square flat regions (optional, flat_relax < 1). Surfaces that already
//    face a target direction (a cube axis for the cube) count as perfectly
//    stylized, so a flat disc keeps its
//    round outline. Scaling the ARAP weight of edges inside such regions by
//    flat_relax (re-evaluated every iteration) lets them reshape in-plane,
//    so their rims can straighten into square outlines.

#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "target_shape.h"
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
  // flat_relax: edge-weight factor inside regions already facing a target
  // direction, in (0, 1]; 1 disables the square-flat-regions extension.
  // target: TargetShape code (0 = cube); roundness in [0, 1] shapes the
  // rounded cube only.
  CubicStylizer(const RowMatX3d& V, const RowMatX3i& F, double cubeness,
                const Eigen::Matrix3d& cube_axes, const std::vector<int32_t>& pins,
                double flat_relax = 1.0, int32_t target = 0, double roundness = 0.5);

  // Local-global iterations; returns the (n, 3) positions.
  //   pin_pos: (pins().size(), 3) targets for the pinned vertices, or null to
  //            hold them at rest
  //   V_init:  warm-start positions, or null to start from the rest pose
  RowMatX3d solve(const RowMatX3d* pin_pos, const RowMatX3d* V_init, int iterations,
                  int admm_iters, ThreadPool& pool, const ProgressFn& on_progress = nullptr,
                  int* iterations_done = nullptr);

  int num_vertices() const { return n_; }
  const std::vector<int32_t>& pins() const { return pins_; }
  double cubeness() const { return lam_; }
  void set_cubeness(double lam) { lam_ = lam; }

 private:
  void build_edges();
  void build_normals_and_areas();
  void build_solver();
  void build_part_placement(const std::vector<int32_t>& comp, const std::vector<char>& pinned);
  void factorize(bool analyze);
  void reweight_flat_regions(const RowMatX3d& V);
  void restore_base_weights();
  void place_floating_parts(RowMatX3d& V) const;
  void local_step(const RowMatX3d& V, int admm_iters, ThreadPool& pool);
  RowMatX3d global_step(const RowMatX3d& ppos, ThreadPool& pool);

  RowMatX3d V0_;
  RowMatX3i F_;
  int n_;
  double lam_;
  Eigen::Matrix3d A_;
  Target target_;
  std::vector<int32_t> pins_;

  // directed one-ring edges in CSR order: row i holds every spoke i -> j
  std::vector<int32_t> row_start_, col_;
  std::vector<double> w_, deg_;
  std::vector<double> w_base_;  // cotan weights before flat-region relaxing
  double flat_relax_;
  bool weights_relaxed_ = false;
  bool has_iterated_ = false;  // flat regions are only relaxed after one plain iteration
  std::vector<Eigen::Vector3d> e0_;  // rest-pose edge vectors V0_i - V0_j

  std::vector<Eigen::Vector3d> nhat_;
  std::vector<double> area_;

  // Global step: pinned rows become identity and edges into pins move to
  // the right-hand side, giving a symmetric positive definite system.
  std::vector<char> anchored_;

  // Parts without pins, in placement order (see place_floating_parts).
  struct Part {
    std::vector<int32_t> verts;
    bool by_contact = false;
    Eigen::Vector3d rest_centroid = Eigen::Vector3d::Zero();
    // (vertex of this part, nearest vertex of an earlier-placed part)
    std::vector<std::pair<int32_t, int32_t>> contacts;
    Eigen::Vector3d rest_offset = Eigen::Vector3d::Zero();  // mean V0_v - V0_u
  };
  std::vector<Part> parts_;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_;
  std::unique_ptr<Eigen::SparseLU<Eigen::SparseMatrix<double>>> lu_;  // fallback

  // per-vertex rotations and ADMM state (warm-started across iterations)
  std::vector<Eigen::Matrix3d> R_;
  std::vector<Eigen::Vector3d> z_, u_;
  std::vector<double> rho_;
};

}  // namespace cubify
