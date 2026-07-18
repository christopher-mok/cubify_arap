# GPU backend (CUDA / Apple Metal via MPS) for the Cubic Stylization solver,
# built on PyTorch. One code path serves both devices.
#
# The compute-heavy local step (per-vertex rotation fitting + ADMM) and the
# right-hand-side assembly run entirely on the GPU. Rotations are fitted with
# a scaled Newton polar-decomposition iteration (batched matmuls + closed-form
# 3x3 inverses) instead of SVD: SVD is unsupported/slow on MPS and slow for
# huge 3x3 batches on CUDA, while Newton-polar is a handful of fused matmul
# kernels. The rare degenerate/reflection cases fall back to CPU SVD.
#
# The global step reuses the prefactorized scipy LU from the base class when
# available (transferring only the (n,3) right-hand side); otherwise CG runs
# on the GPU.
#
# All tensors are float32: MPS has no float64, and float32 is also the fast
# path on CUDA. Positions are returned as float64 numpy arrays.

import numpy as np
import torch

try:
    from .solver import CubicStylizer, _RHO_INIT, _MU, _TAU, _EPS_ABS, _EPS_REL
except ImportError:  # standalone (outside the Blender package)
    from solver import CubicStylizer, _RHO_INIT, _MU, _TAU, _EPS_ABS, _EPS_REL


def _det3(X):
    return (X[:, 0, 0] * (X[:, 1, 1] * X[:, 2, 2] - X[:, 1, 2] * X[:, 2, 1])
            - X[:, 0, 1] * (X[:, 1, 0] * X[:, 2, 2] - X[:, 1, 2] * X[:, 2, 0])
            + X[:, 0, 2] * (X[:, 1, 0] * X[:, 2, 1] - X[:, 1, 1] * X[:, 2, 0]))


def _inv3(X):
    """Batched closed-form 3x3 inverse (adjugate / det)."""
    a, b, c = X[:, 0, 0], X[:, 0, 1], X[:, 0, 2]
    d, e, f = X[:, 1, 0], X[:, 1, 1], X[:, 1, 2]
    g, h, i = X[:, 2, 0], X[:, 2, 1], X[:, 2, 2]
    c00 = e * i - f * h
    c01 = f * g - d * i
    c02 = d * h - e * g
    det = a * c00 + b * c01 + c * c02
    det = torch.where(det.abs() < 1e-20, torch.full_like(det, 1e-20), det)
    adj = torch.stack([
        torch.stack([c00, c * h - b * i, b * f - c * e], dim=1),
        torch.stack([c01, a * i - c * g, c * d - a * f], dim=1),
        torch.stack([c02, b * g - a * h, a * e - b * d], dim=1),
    ], dim=1)
    return adj / det[:, None, None]


class TorchCubicStylizer(CubicStylizer):
    """Drop-in replacement for CubicStylizer running on 'cuda', 'mps' or
    'cpu' (torch). Same constructor plus `device`; same solve()/run() API
    (numpy in, numpy out)."""

    def __init__(self, V, F, cubeness=0.2, cube_axes=None, pins=None,
                 device='cuda'):
        super().__init__(V, F, cubeness=cubeness, cube_axes=cube_axes, pins=pins)
        self.dev = torch.device(device)
        dt = torch.float32
        self.dt = dt

        def T(x, dtype=dt):
            return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype,
                                   device=self.dev)

        self.t_ii = T(self.ii, torch.int64)
        self.t_jj = T(self.jj, torch.int64)
        self.t_w = T(self.w)
        self.t_e0 = T(self.e0)
        self.t_we0 = T(self.w[:, None] * self.e0)
        self.t_deg = T(self.deg)
        self.t_area = T(self.area)
        self.t_nhat = T(self.nhat)
        self.t_A = T(self.A)
        self.t_V0 = T(self.V0)
        self.t_eye = torch.eye(3, dtype=dt, device=self.dev)

        # ADMM state (warm-started across iterations/solves)
        self.t_z = self.t_nhat @ self.t_A
        self.t_u = torch.zeros(self.n, 3, dtype=dt, device=self.dev)
        self.t_rho = torch.full((self.n,), _RHO_INIT, dtype=dt, device=self.dev)

        if self._lu is not None:
            self.t_lu_anchors = T(self._lu_anchors, torch.int64)
            idx = np.flatnonzero(self._lu_edge_to_anchor)
            self.t_lu_pin_edges = T(idx, torch.int64)
        elif len(self._cg_anchors):
            self.t_cg_anchors = T(self._cg_anchors, torch.int64)
            idx = np.flatnonzero(self._cg_edge_to_anchor)
            self.t_cg_pin_edges = T(idx, torch.int64)

    # ---------------- rotation fitting ----------------

    def _t_polar_rotations(self, M):
        """argmax_{R in SO(3)} tr(R M), batched, GPU-only, no host sync.

        Scaled Newton iteration gives the orthogonal polar factor Q of M
        (M = Q H, H symmetric PSD). For det(Q) = +1 the optimum is R* = Q^T.
        For det(Q) = -1 (reflective fit — common in this ADMM, where the
        rank-1 rho-term can oppose orientation) the optimum flips the
        smallest singular direction: R* = (Q (I - 2 v3 v3^T))^T with v3 the
        smallest-eigenvalue eigenvector of H = Q^T M, which we get from the
        closed-form symmetric 3x3 eigenvalue formula + cross-product null
        space — still no SVD and no host sync. Returns (R, bad); `bad` flags
        only genuine non-convergence for the (rare) CPU rescue."""
        scale = M.flatten(1).norm(dim=1).clamp_min(1e-20)
        M0 = M / scale[:, None, None]
        X = M0
        for _ in range(8):
            det = _det3(X)
            mu = det.abs().clamp_min(1e-12).pow(-1.0 / 3.0).clamp(0.1, 10.0)
            X = 0.5 * (mu[:, None, None] * X
                       + _inv3(X).transpose(1, 2) / mu[:, None, None])
        Q = X

        # smallest-eigenvalue eigenvector of H = Q^T M0 (symmetric PSD)
        H = Q.transpose(1, 2) @ M0
        H = 0.5 * (H + H.transpose(1, 2))
        q = (H[:, 0, 0] + H[:, 1, 1] + H[:, 2, 2]) / 3.0
        Hq = H - q[:, None, None] * self.t_eye
        p = (Hq.flatten(1).square().sum(dim=1) / 6.0).sqrt().clamp_min(1e-20)
        B = Hq / p[:, None, None]
        r = (_det3(B) / 2.0).clamp(-1.0, 1.0)
        phi = torch.acos(r) / 3.0
        lam_min = q + 2.0 * p * torch.cos(phi + 2.0 * np.pi / 3.0)

        C = H - lam_min[:, None, None] * self.t_eye
        cands = torch.stack([
            torch.cross(C[:, 0], C[:, 1], dim=1),
            torch.cross(C[:, 0], C[:, 2], dim=1),
            torch.cross(C[:, 1], C[:, 2], dim=1)], dim=1)   # (n,3,3)
        norms = cands.norm(dim=2)
        best = norms.argmax(dim=1)
        v = cands[torch.arange(len(M), device=self.dev), best]
        # fully degenerate (H ~ isotropic): any flip direction is optimal
        v = torch.where(norms.max(dim=1).values[:, None] > 1e-12, v,
                        torch.tensor([1.0, 0.0, 0.0], dtype=self.dt,
                                     device=self.dev).expand_as(v))
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-20)

        Q_flip = Q @ (self.t_eye - 2.0 * v[:, :, None] * v[:, None, :])
        neg = (_det3(Q) < 0)[:, None, None]
        R = torch.where(neg, Q_flip, Q).transpose(1, 2)

        err = (Q @ Q.transpose(1, 2) - self.t_eye).flatten(1).norm(dim=1)
        # lam_min ~ sigma3/|M|; near-singular matrices (or a negative value,
        # meaning Newton grabbed a non-polar factor) can't be resolved in
        # float32 — route them to the CPU rescue as well
        bad = (err > 5e-3) | ~torch.isfinite(err) | (lam_min < 1e-4)
        return torch.nan_to_num(R), bad

    def _t_rescue_rotations(self, M, R, bad):
        """CPU SVD for flagged rotations (syncs; call sparingly)."""
        idx = bad.nonzero(as_tuple=False).squeeze(1)
        if idx.numel():
            Mc = M[idx].detach().cpu().double().numpy()
            U, _, Vt = np.linalg.svd(Mc)
            Rc = Vt.transpose(0, 2, 1) @ U.transpose(0, 2, 1)
            flip = np.linalg.det(Rc) < 0
            if np.any(flip):
                Uf = U[flip]
                Uf[:, :, 2] *= -1
                Rc[flip] = Vt[flip].transpose(0, 2, 1) @ Uf.transpose(0, 2, 1)
            R[idx] = torch.as_tensor(Rc, dtype=self.dt, device=self.dev)
        return R

    def _t_fit_rotations(self, M):
        R, bad = self._t_polar_rotations(M)
        return self._t_rescue_rotations(M, R, bad)

    # ---------------- local step ----------------

    def _t_local_step(self, Vt, admm_iters):
        n = self.n
        Ep = Vt[self.t_ii] - Vt[self.t_jj]
        K = torch.einsum('ei,ej->eij', self.t_we0, Ep)
        S = torch.zeros(n, 3, 3, dtype=self.dt, device=self.dev)
        S.index_add_(0, self.t_ii, K)

        if self.lam <= 0.0:
            return self._t_fit_rotations(S)

        # Full-batch ADMM with GPU-side per-vertex freezing: converged
        # vertices keep their state via `where` masks (freezing matters —
        # without it the Boyd rho update sees s=0 and doubles rho forever).
        # Host syncs (early-exit test + CPU-SVD rescue of rotations the
        # Newton iteration flagged as degenerate) happen only every 4th
        # iteration; a vertex is never frozen while its rotation is
        # unrescued, so all returned rotations are valid.
        z, u, rho = self.t_z, self.t_u, self.t_rho
        k = self.lam * self.t_area
        eps3 = float(np.sqrt(3.0)) * _EPS_ABS
        done = torch.zeros(n, dtype=torch.bool, device=self.dev)
        R_out = self.t_eye.expand(n, 3, 3).clone()

        for it in range(admm_iters):
            sync = (it & 3) == 3 or it == admm_iters - 1

            Azu = (z - u) @ self.t_A.T
            M = S + rho[:, None, None] * torch.einsum(
                'ni,nj->nij', self.t_nhat, Azu)
            R_new, bad = self._t_polar_rotations(M)
            if sync:
                R_new = self._t_rescue_rotations(M, R_new, bad)
                bad = torch.zeros_like(bad)
            R_out = torch.where(done[:, None, None], R_out, R_new)

            Rn = torch.einsum('nij,nj->ni', R_out, self.t_nhat) @ self.t_A
            x = Rn + u
            thr = (k / rho)[:, None]
            z_new = torch.sign(x) * (x.abs() - thr).clamp_min(0.0)
            u_new = u + Rn - z_new

            r_pri = (Rn - z_new).norm(dim=1)
            s_dua = rho * (z_new - z).norm(dim=1)

            inc = r_pri > _MU * s_dua
            dec = s_dua > _MU * r_pri
            rho_new = torch.where(inc, rho * _TAU,
                                  torch.where(dec, rho / _TAU, rho))
            u_new = torch.where(inc[:, None], u_new / _TAU,
                                torch.where(dec[:, None], u_new * _TAU, u_new))

            eps_pri = eps3 + _EPS_REL * torch.maximum(Rn.norm(dim=1),
                                                      z_new.norm(dim=1))
            eps_dua = eps3 + _EPS_REL * rho_new * u_new.norm(dim=1)
            conv = (r_pri < eps_pri) & (s_dua < eps_dua) & ~bad

            z = torch.where(done[:, None], z, z_new)
            u = torch.where(done[:, None], u, u_new)
            rho = torch.where(done, rho, rho_new)
            done = done | conv

            if sync and bool(done.all()):
                break

        self.t_z, self.t_u, self.t_rho = z, u, rho
        return R_out

    # ---------------- global step ----------------

    def _t_matvec(self, x):
        out = self.t_deg[:, None] * x
        out.index_add_(0, self.t_ii, -(self.t_w[:, None] * x[self.t_jj]))
        return out

    def _t_cg(self, b, x0, anchors, tol=1e-5, maxiter=400):
        x = x0.clone()
        if anchors is not None:
            x[anchors] = 0.0

        def matvec(v):
            out = self._t_matvec(v)
            if anchors is not None:
                out[anchors] = v[anchors]
            return out

        r = b - matvec(x)
        minv = 1.0 / self.t_deg.clamp_min(1e-12)[:, None]
        z = minv * r
        p = z.clone()
        rz = (r * z).sum(dim=0)
        bnorm = b.norm(dim=0) + 1e-30
        for it in range(maxiter):
            Ap = matvec(p)
            alpha = rz / ((p * Ap).sum(dim=0) + 1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            if (it & 7) == 7 and bool((r.norm(dim=0) < tol * bnorm).all()):
                break
            z = minv * r
            rz_new = (r * z).sum(dim=0)
            p = z + (rz_new / (rz + 1e-30)) * p
            rz = rz_new
        return x

    def _t_global_step(self, R, Vt, ppos_t):
        n = self.n
        Rsum = R[self.t_ii] + R[self.t_jj]
        contrib = 0.5 * self.t_w[:, None] * torch.einsum(
            'eij,ej->ei', Rsum, self.t_e0)
        b = torch.zeros(n, 3, dtype=self.dt, device=self.dev)
        b.index_add_(0, self.t_ii, contrib)

        if self._lu is not None:
            m = self.t_lu_pin_edges
            if m.numel():
                b.index_add_(0, self.t_ii[m],
                             self.t_w[m, None] * ppos_t[self.t_jj[m]])
            b[self.t_lu_anchors] = ppos_t[self.t_lu_anchors]
            b_np = b.detach().cpu().double().numpy()
            x = np.column_stack([self._lu.solve(b_np[:, c]) for c in range(3)])
            return torch.as_tensor(x, dtype=self.dt, device=self.dev)

        if len(self._cg_anchors):
            m = self.t_cg_pin_edges
            if m.numel():
                b.index_add_(0, self.t_ii[m],
                             self.t_w[m, None] * ppos_t[self.t_jj[m]])
            b[self.t_cg_anchors] = 0.0
            x = self._t_cg(b, Vt, self.t_cg_anchors)
            x[self.t_cg_anchors] = ppos_t[self.t_cg_anchors]
            return x

        x = self._t_cg(b, Vt, None)
        return x - x.mean(dim=0) + self.t_V0.mean(dim=0)

    # ---------------- driver ----------------

    def solve(self, pin_pos=None, V_init=None, iterations=30, admm_iters=100,
              on_progress=None):
        ppos = self.V0.copy()
        if pin_pos is not None and len(self.pins):
            ppos[self.pins] = np.asarray(pin_pos, dtype=np.float64).reshape(
                len(self.pins), 3)
        ppos_t = torch.as_tensor(ppos, dtype=self.dt, device=self.dev)

        V_start = (np.asarray(V_init, dtype=np.float64).reshape(-1, 3)
                   if V_init is not None else self.V0)
        Vt = torch.as_tensor(V_start, dtype=self.dt, device=self.dev)
        bbox = float(np.linalg.norm(self.V0.max(axis=0) - self.V0.min(axis=0)))

        for it in range(iterations):
            R = self._t_local_step(Vt, admm_iters)
            V_new = self._t_global_step(R, Vt, ppos_t)
            step = float((V_new - Vt).norm(dim=1).max())
            Vt = V_new
            if on_progress is not None:
                on_progress(it + 1, iterations)
            if step < 1e-6 * bbox:
                break
        return Vt.detach().cpu().double().numpy()
