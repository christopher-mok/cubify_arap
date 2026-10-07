# Cubic Stylization / ARAP solver (Liu & Jacobson, SIGGRAPH Asia 2019), pure numpy.
#
# Minimizes the ARAP energy plus the stylization term
#     sum_i sum_{j in N(i)} (w_ij / 2) ||R_i d_ij - d'_ij||^2
#   + sum_i lambda * a_i * f(A^T R_i n_i)
# with local-global iterations, where f is the target shape's term (the
# paper's L1 norm for the cube; see "target shapes" below). With cubeness = 0 this is classic ARAP. The
# local (rotation) step is a per-vertex ADMM (Algorithm 1 of the paper) batched
# over all vertices with numpy (a plain batched Procrustes when cubeness = 0);
# the global step is a cotan-Laplacian solve (scipy sparse LU when available,
# otherwise a Jacobi-preconditioned conjugate gradient).
#
# Vertices may be pinned to prescribed positions ("handles"). The system matrix
# depends only on the pin *set*, so it is factorized once and pins can then be
# dragged interactively: each drag only rebuilds the right-hand side.
#
# This module has no Blender dependencies and can be tested standalone.

import numpy as np

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import splu
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _det3(X):
    """Batched closed-form 3x3 determinant — much faster than the LAPACK
    round-trip of np.linalg.det on large (n, 3, 3) stacks."""
    return (X[:, 0, 0] * (X[:, 1, 1] * X[:, 2, 2] - X[:, 1, 2] * X[:, 2, 1])
            - X[:, 0, 1] * (X[:, 1, 0] * X[:, 2, 2] - X[:, 1, 2] * X[:, 2, 0])
            + X[:, 0, 2] * (X[:, 1, 0] * X[:, 2, 1] - X[:, 1, 1] * X[:, 2, 0]))


# ================== target shapes
#
# For the faceted targets f is the support function of the polytope
#     P = {y : d_k . y <= 1}
# whose face normals d_k are the preferred normal directions. On the unit
# sphere f is smallest (exactly 1) at the d_k, so surfaces snap to face them.
# The cube's P is [-1, 1]^3, whose support function is the paper's L1 norm
# and whose proximal step is soft-thresholding. For any support function the
# proximal step follows from Moreau's identity,
#     prox_{t f}(x) = x - t * proj_P(x / t),
# with proj_P computed exactly by testing P's faces and edges.
#
# The rounded cube uses f(y) = sum_c |y_c|^p, 1 < p < 2, instead: it pulls
# normals toward the cube axes without ever snapping them exactly, which
# rounds edges and corners (p near 1 is almost the cube, p near 2 almost a
# sphere). Roundness r in [0, 1] maps to p = 2 - 0.5 * 10^-r, which spaces
# the look evenly. Its proximal step solves w^q + t p w = |x| for
# w = |z|^(p-1), q = 1 / (p - 1): convex in w, so Newton's method from
# w = |x|^(p-1) descends monotonically onto the root.
#
# Must match cubic_stylization_c++/src/target_shape.h.

TARGETS = ('CUBE', 'OCTAHEDRON', 'PYRAMID', 'HEX_COLUMN', 'ROUNDED_CUBE')

# Pyramid side-face normals sit this far above the horizon: the faces slope
# at ~52 degrees, like the Great Pyramid.
_PYRAMID_NORMAL_ELEVATION_DEG = 38.0

_POLYTOPES = {}


def rounded_exponent(roundness):
    return 2.0 - 0.5 * 10.0 ** (-min(max(float(roundness), 0.0), 1.0))


def target_directions(target):
    """(K, 3) unit normals the target's surfaces snap to (target frame)."""
    if target in ('CUBE', 'ROUNDED_CUBE'):
        return np.concatenate([np.eye(3), -np.eye(3)])
    if target == 'OCTAHEDRON':
        s = np.array([1.0, -1.0])
        return np.stack(np.meshgrid(s, s, s, indexing='ij'), -1).reshape(-1, 3) / np.sqrt(3.0)
    if target == 'PYRAMID':
        a = np.radians(_PYRAMID_NORMAL_ELEVATION_DEG)
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [-c, 0, s], [0, c, s], [0, -c, s], [0, 0, -1.0]])
    if target == 'HEX_COLUMN':
        k = np.arange(6) * np.pi / 3.0
        side = np.stack([np.cos(k), np.sin(k), np.zeros(6)], -1)
        return np.concatenate([side, [[0, 0, 1.0], [0, 0, -1.0]]])
    raise ValueError(f"unknown target shape {target!r}")


def _polytope(target):
    """(D, edge_start, edge_end) of P for a faceted target: vertices from
    every triple of face planes; an edge joins two vertices sharing at least
    two tight faces."""
    if target not in _POLYTOPES:
        D = target_directions(target)
        verts, tight = [], []
        K = len(D)
        for i in range(K):
            for j in range(i + 1, K):
                for k in range(j + 1, K):
                    M = D[[i, j, k]]
                    if abs(np.linalg.det(M)) < 1e-9:
                        continue
                    v = np.linalg.solve(M, np.ones(3))
                    if np.any(D @ v > 1.0 + 1e-9):
                        continue
                    if any(np.linalg.norm(w - v) < 1e-9 for w in verts):
                        continue
                    verts.append(v)
                    tight.append(set(np.flatnonzero(np.abs(D @ v - 1.0) < 1e-9)))
        ea, eb = [], []
        for a in range(len(verts)):
            for b in range(a + 1, len(verts)):
                if len(tight[a] & tight[b]) >= 2:
                    ea.append(verts[a])
                    eb.append(verts[b])
        _POLYTOPES[target] = (D, np.array(ea), np.array(eb))
    return _POLYTOPES[target]


def _project_polytope(Y, D, ea, eb):
    """Euclidean projection of the rows of Y onto P = {y : D y <= 1}: the
    nearest of the face-plane projections that land inside P and the clamped
    edge-segment projections."""
    out = Y.copy()
    outside = (Y @ D.T).max(axis=1) > 1.0
    if not np.any(outside):
        return out
    y = Y[outside]
    C = y[:, None, :] - (y @ D.T - 1.0)[:, :, None] * D[None]          # (m, K, 3)
    lim = 1.0 + 1e-9 * np.maximum(1.0, np.linalg.norm(C, axis=2))
    inside = np.all(C @ D.T <= lim[:, :, None], axis=2)
    d2f = np.where(inside, ((C - y[:, None]) ** 2).sum(axis=2), np.inf)
    ab = eb - ea
    s = np.clip(((y[:, None] - ea[None]) * ab[None]).sum(axis=2) / (ab ** 2).sum(axis=1), 0.0, 1.0)
    CE = ea[None] + s[:, :, None] * ab[None]                           # (m, E, 3)
    d2e = ((CE - y[:, None]) ** 2).sum(axis=2)
    cand = np.concatenate([C, CE], axis=1)
    pick = np.argmin(np.concatenate([d2f, d2e], axis=1), axis=1)
    out[outside] = cand[np.arange(len(y)), pick]
    return out


def target_prox(target, X, t, p=1.75):
    """Proximal step of t_i * f at the rows x_i of X (t >= 0); p is the
    rounded cube's exponent."""
    if target == 'CUBE':
        return np.sign(X) * np.maximum(np.abs(X) - t[:, None], 0.0)
    if target == 'ROUNDED_CUBE':
        Z = X.copy()
        m = t > 1e-12
        a = np.abs(X[m])
        tp = t[m, None] * p
        q = 1.0 / (p - 1.0)
        w = a ** (p - 1.0)
        for _ in range(50):
            step = (w ** q + tp * w - a) / (q * w ** (q - 1.0) + tp)
            w -= step
            if np.all(np.abs(step) <= 1e-15 * np.maximum(1.0, w)):
                break
        Z[m] = np.sign(X[m]) * np.maximum(w, 0.0) ** q
        return Z
    D, ea, eb = _polytope(target)
    Z = X.copy()
    m = t > 1e-12
    tm = t[m, None]
    Z[m] = X[m] - tm * _project_polytope(X[m] / tm, D, ea, eb)
    return Z


def target_alignment(target, Y):
    """Cosine between each unit row of Y (target frame) and the nearest
    preferred direction: 1 when it faces one exactly."""
    return (Y @ target_directions(target).T).max(axis=1)


# ADMM constants from the paper
_RHO_INIT = 1e-4
_MU = 10.0
_TAU = 2.0
_EPS_ABS = 1e-5
_EPS_REL = 1e-3


class CubicStylizer:
    """Cubic stylization / ARAP deformation of a triangle mesh.

    V : (n, 3) float array of rest-pose vertex positions
    F : (m, 3) int array of triangle indices (a virtual triangulation of a
        quad/n-gon mesh works fine: only vertex positions are solved for)
    cubeness : lambda, strength of the stylization term (0 = classic ARAP)
    cube_axes : (3, 3) rotation matrix A whose columns are the target cube axes
    pins : optional iterable of vertex indices to constrain ("handles")
    target : one of TARGETS, the shape surfaces are stylized toward
    roundness : 0..1, rounded cube only (0 is nearly the cube)
    keep_orientation : without pins, undo the mesh's net rotation after every
                       global step (see _remove_net_rotation)
    """

    def __init__(self, V, F, cubeness=0.2, cube_axes=None, pins=None,
                 target='CUBE', roundness=0.5, keep_orientation=False):
        self.keep_orientation = bool(keep_orientation)
        self.target = str(target).upper()
        if self.target not in TARGETS:
            raise ValueError(f"unknown target shape {target!r}")
        self.p = rounded_exponent(roundness)
        self.V0 = np.asarray(V, dtype=np.float64).reshape(-1, 3)
        self.F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
        self.n = len(self.V0)
        self.lam = float(cubeness)
        self.A = np.eye(3) if cube_axes is None else np.asarray(cube_axes, dtype=np.float64)

        if self.n == 0 or len(self.F) == 0:
            raise ValueError("mesh has no vertices or faces")

        pin_list = [] if pins is None else list(pins)
        pins = (np.unique(np.asarray(pin_list, dtype=np.int64))
                if pin_list else np.zeros(0, np.int64))
        if len(pins) and (pins.min() < 0 or pins.max() >= self.n):
            raise ValueError("pin index out of range")
        self.pins = pins

        self._build_edges()
        self._build_normals_and_areas()
        self._build_solver()

        # ADMM state, warm-started across local-global iterations.
        # z = A^T n corresponds to the feasible start R = I.
        self.z = self.nhat @ self.A
        self.u = np.zeros((self.n, 3))
        self.rho = np.full(self.n, _RHO_INIT)

    # ---------------- precomputation ----------------

    def _build_edges(self):
        V, F, n = self.V0, self.F, self.n
        i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]

        def cot(a, b, c):
            u = V[b] - V[a]
            v = V[c] - V[a]
            cr = np.linalg.norm(np.cross(u, v), axis=1)
            return np.einsum('ij,ij->i', u, v) / np.maximum(cr, 1e-12)

        # cotangent at each corner weights the opposite edge
        I = np.concatenate([i1, i2, i0])
        J = np.concatenate([i2, i0, i1])
        C = 0.5 * np.concatenate([cot(i0, i1, i2), cot(i1, i2, i0), cot(i2, i0, i1)])

        lo = np.minimum(I, J)
        hi = np.maximum(I, J)
        key = lo * np.int64(n) + hi
        uniq, inv = np.unique(key, return_inverse=True)
        w = np.abs(np.bincount(inv, weights=C, minlength=len(uniq)))
        ua = uniq // n
        ub = uniq % n

        # directed edges, both ways: row i sees all its one-ring spokes
        self.ii = np.concatenate([ua, ub])
        self.jj = np.concatenate([ub, ua])
        self.w = np.concatenate([w, w])
        self.e0 = V[self.ii] - V[self.jj]           # rest-pose edge vectors
        self.deg = np.bincount(self.ii, weights=self.w, minlength=n)

        # weighted edge->vertex accumulator: (acc @ X)[i] = sum_e w_e X_e over
        # edges leaving i. One C sparse matmul replaces a python loop of
        # bincounts in the per-iteration S and rhs assemblies.
        self._acc = None
        if _HAS_SCIPY:
            E = len(self.ii)
            self._acc = csr_matrix(
                (self.w, (self.ii, np.arange(E))), shape=(n, E))

    def _build_normals_and_areas(self):
        V, F, n = self.V0, self.F, self.n
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])  # 2*area*normal
        fa = 0.5 * np.linalg.norm(fn, axis=1)

        self.area = np.zeros(n)
        nrm = np.zeros((n, 3))
        for k in range(3):
            self.area += np.bincount(F[:, k], weights=fa, minlength=n) / 3.0
            for c in range(3):
                nrm[:, c] += np.bincount(F[:, k], weights=fn[:, c], minlength=n)
        ln = np.linalg.norm(nrm, axis=1)
        self.nhat = nrm / np.maximum(ln, 1e-12)[:, None]
        self.nhat[ln < 1e-12] = 0.0  # isolated/degenerate: cubeness term vanishes

    def _build_solver(self):
        """Global-step solver for L p' = b with pinned vertices.

        Pinned rows are replaced by identity (as in the reference C++
        implementation) and edges into pins move to the right-hand side, so
        the matrix depends only on the pin *set* and is factorized once.

        scipy path: sparse LU. Vertex 0 is pinned automatically when the user
        pinned nothing (the energy is translation-invariant); loose vertices
        (no incident face) are always pinned so the system stays regular.
        Fallback path: matrix-free preconditioned CG. With pins it solves the
        (symmetric) system reduced to free vertices; without pins it solves
        the full singular-but-consistent Laplacian and recentres afterwards.
        """
        n = self.n
        lone = np.flatnonzero(self.deg <= 1e-12)

        # anchors used by the CG fallback (empty = singular solve + recentre)
        if len(self.pins):
            self._cg_anchors = np.unique(np.concatenate([self.pins, lone]))
        else:
            self._cg_anchors = np.zeros(0, np.int64)

        # anchors used by the LU path (never empty)
        parts = [self.pins, lone] + ([] if len(self.pins) else [np.array([0])])
        self._lu_anchors = np.unique(np.concatenate(parts).astype(np.int64))

        self._lu = None
        if _HAS_SCIPY:
            anchored = np.zeros(n, bool)
            anchored[self._lu_anchors] = True
            src_free = ~anchored[self.ii]
            both_free = src_free & ~anchored[self.jj]
            self._lu_edge_to_anchor = src_free & anchored[self.jj]

            rows = np.concatenate([self.ii[both_free], self.ii[src_free], self._lu_anchors])
            cols = np.concatenate([self.jj[both_free], self.ii[src_free], self._lu_anchors])
            vals = np.concatenate([-self.w[both_free], self.w[src_free],
                                   np.ones(len(self._lu_anchors))])
            try:
                L = csr_matrix((vals, (rows, cols)), shape=(n, n))
                self._lu = splu(L.tocsc())
            except Exception:
                self._lu = None  # e.g. disconnected mesh: fall back to CG

        if self._lu is None and len(self._cg_anchors):
            anchored = np.zeros(n, bool)
            anchored[self._cg_anchors] = True
            self._cg_edge_to_anchor = (~anchored[self.ii]) & anchored[self.jj]

    # ---------------- solver pieces ----------------

    def _matvec(self, x):
        """Full cotan Laplacian times x, matrix-free (x is (n, 3))."""
        out = self.deg[:, None] * x
        wx = self.w[:, None] * x[self.jj]
        for c in range(3):
            out[:, c] -= np.bincount(self.ii, weights=wx[:, c], minlength=self.n)
        return out

    def _cg_solve(self, b, x0, anchors, tol=1e-8, maxiter=1000):
        """CG on the Laplacian; rows in `anchors` are held at zero (the
        reduced free-vertex system), which keeps the operator symmetric."""
        x = x0.copy()
        if len(anchors):
            x[anchors] = 0.0

        def matvec(v):
            out = self._matvec(v)
            if len(anchors):
                out[anchors] = v[anchors]
            return out

        r = b - matvec(x)
        minv = 1.0 / np.maximum(self.deg, 1e-12)[:, None]
        z = minv * r
        p = z.copy()
        rz = np.einsum('ij,ij->j', r, z)
        bnorm = np.linalg.norm(b, axis=0) + 1e-30
        for _ in range(maxiter):
            Ap = matvec(p)
            alpha = rz / (np.einsum('ij,ij->j', p, Ap) + 1e-30)
            x += alpha * p
            r -= alpha * Ap
            if np.all(np.linalg.norm(r, axis=0) < tol * bnorm):
                break
            z = minv * r
            rz_new = np.einsum('ij,ij->j', r, z)
            p = z + (rz_new / (rz + 1e-30)) * p
            rz = rz_new
        return x

    def _local_step(self, V, admm_iters):
        """Batched per-vertex rotation fit. Returns (n, 3, 3) rotations."""
        n, A = self.n, self.A

        # ARAP covariance S_i = sum_j w_ij d_ij d'_ij^T over the one-ring
        Ep = V[self.ii] - V[self.jj]
        if self._acc is not None:
            outer = (self.e0[:, :, None] * Ep[:, None, :]).reshape(-1, 9)
            S = (self._acc @ outer).reshape(n, 3, 3)
        else:
            S = np.zeros((n, 3, 3))
            for a in range(3):
                for b in range(3):
                    S[:, a, b] = np.bincount(
                        self.ii, weights=self.w * self.e0[:, a] * Ep[:, b],
                        minlength=n)

        if self.lam <= 0.0:
            # classic ARAP: plain orthogonal Procrustes, no ADMM needed
            U, _, Vt = np.linalg.svd(S)
            R = Vt.transpose(0, 2, 1) @ U.transpose(0, 2, 1)
            flip = _det3(R) < 0
            if np.any(flip):
                Uf = U[flip]
                Uf[:, :, 2] *= -1
                R[flip] = Vt[flip].transpose(0, 2, 1) @ Uf.transpose(0, 2, 1)
            return R

        R_all = np.tile(np.eye(3), (n, 1, 1))
        act = np.arange(n)  # vertices whose ADMM has not converged yet
        z, u, rho = self.z, self.u, self.rho
        k = self.lam * self.area  # weight of the stylization term per vertex

        for _ in range(admm_iters):
            zc, uc, rc = z[act], u[act], rho[act]
            nh = self.nhat[act]

            # R-step: Procrustes on M = S + rho * n (A(z-u))^T
            Azu = (zc - uc) @ A.T
            M = S[act] + rc[:, None, None] * (nh[:, :, None] @ Azu[:, None, :])
            U, _, Vt = np.linalg.svd(M)
            R = Vt.transpose(0, 2, 1) @ U.transpose(0, 2, 1)
            flip = _det3(R) < 0
            if np.any(flip):
                Uf = U[flip]
                Uf[:, :, 2] *= -1
                R[flip] = Vt[flip].transpose(0, 2, 1) @ Uf.transpose(0, 2, 1)
            R_all[act] = R

            # z-step: proximal step of the target term at A^T R n + u
            # (soft-thresholding for the cube)
            Rn = np.einsum('nij,nj->ni', R, nh) @ A
            x = Rn + uc
            z_new = target_prox(self.target, x, k[act] / rc, self.p)

            # scaled dual update
            u_new = uc + Rn - z_new

            r_pri = np.linalg.norm(Rn - z_new, axis=1)
            s_dua = rc * np.linalg.norm(z_new - zc, axis=1)

            # penalty update (Boyd et al. 2011)
            inc = r_pri > _MU * s_dua
            dec = s_dua > _MU * r_pri
            rho_new = rc.copy()
            rho_new[inc] *= _TAU
            u_new[inc] /= _TAU
            rho_new[dec] /= _TAU
            u_new[dec] *= _TAU

            z[act], u[act], rho[act] = z_new, u_new, rho_new

            eps_pri = np.sqrt(3.0) * _EPS_ABS + _EPS_REL * np.maximum(
                np.linalg.norm(Rn, axis=1), np.linalg.norm(z_new, axis=1))
            eps_dua = np.sqrt(3.0) * _EPS_ABS + _EPS_REL * rho_new * np.linalg.norm(u_new, axis=1)
            act = act[~((r_pri < eps_pri) & (s_dua < eps_dua))]
            if len(act) == 0:
                break

        return R_all

    def _global_step(self, R, V, ppos):
        """Solve L p' = b for new positions given per-vertex rotations and
        prescribed pin positions ppos (an (n, 3) array; only rows of pinned
        vertices are read)."""
        n = self.n
        Rsum = R[self.ii] + R[self.jj]
        if self._acc is not None:
            b = self._acc @ (0.5 * np.einsum('eij,ej->ei', Rsum, self.e0))
        else:
            contrib = 0.5 * self.w[:, None] * np.einsum('eij,ej->ei', Rsum, self.e0)
            b = np.zeros((n, 3))
            for c in range(3):
                b[:, c] = np.bincount(self.ii, weights=contrib[:, c], minlength=n)

        if self._lu is not None:
            m = self._lu_edge_to_anchor
            if np.any(m):
                pc = self.w[m, None] * ppos[self.jj[m]]
                for c in range(3):
                    b[:, c] += np.bincount(self.ii[m], weights=pc[:, c], minlength=n)
            b[self._lu_anchors] = ppos[self._lu_anchors]
            return self._lu.solve(b)

        if len(self._cg_anchors):
            m = self._cg_edge_to_anchor
            if np.any(m):
                pc = self.w[m, None] * ppos[self.jj[m]]
                for c in range(3):
                    b[:, c] += np.bincount(self.ii[m], weights=pc[:, c], minlength=n)
            b[self._cg_anchors] = 0.0
            x = self._cg_solve(b, V, self._cg_anchors)
            x[self._cg_anchors] = ppos[self._cg_anchors]
            return x

        # unconstrained: singular but consistent; fix the nullspace by recentring
        x = self._cg_solve(b, V, self._cg_anchors)
        return x - x.mean(axis=0) + self.V0.mean(axis=0)

    # ---------------- drivers ----------------

    def solve(self, pin_pos=None, V_init=None, iterations=30, admm_iters=100,
              on_progress=None):
        """Run local-global iterations and return the (n, 3) positions.

        pin_pos : (len(self.pins), 3) target positions for the pinned
                  vertices (defaults to their rest positions)
        V_init : warm-start positions (defaults to the rest pose)
        """
        ppos = self.V0.copy()
        if pin_pos is not None and len(self.pins):
            ppos[self.pins] = np.asarray(pin_pos, dtype=np.float64).reshape(len(self.pins), 3)

        V = (np.asarray(V_init, dtype=np.float64).reshape(-1, 3).copy()
             if V_init is not None else self.V0.copy())
        bbox = np.linalg.norm(self.V0.max(axis=0) - self.V0.min(axis=0))

        for it in range(iterations):
            R = self._local_step(V, admm_iters)
            V_new = self._global_step(R, V, ppos)
            if self.keep_orientation and len(self.pins) == 0:
                V_new = self._remove_net_rotation(V_new)
            step = np.max(np.linalg.norm(V_new - V, axis=1))
            V = V_new
            if on_progress is not None:
                on_progress(it + 1, iterations)
            if step < 1e-6 * bbox:
                break
        return V

    def _remove_net_rotation(self, V):
        """Undo the area-weighted best-fit rotation from the rest pose.

        ARAP is unchanged by a rotation of the whole mesh, so without pins the
        stylization term alone decides the mesh's orientation and may turn it
        as a whole (Suzanne's head turns 10-14 degrees under the cube or
        octahedron). Undoing that rotation costs no ARAP energy and
        makes the target shape come from reshaping."""
        a = self.area
        total = a.sum()
        if total <= 0.0:
            return V
        c0 = a @ self.V0 / total
        c = a @ V / total
        H = (self.V0 - c0).T @ (a[:, None] * (V - c))
        U, _, Vt = np.linalg.svd(H)
        if np.linalg.det(Vt.T @ U.T) < 0:
            U[:, 2] *= -1
        R = Vt.T @ U.T                      # rest -> current
        return (V - c) @ R + c              # rows: R^T (v - c) + c

    def run(self, iterations=30, admm_iters=100, on_progress=None, pin_pos=None):
        """One-shot stylization. Holds pinned vertices (if any) in place;
        otherwise keeps the result centered where the input was."""
        V = self.solve(pin_pos=pin_pos, iterations=iterations,
                       admm_iters=admm_iters, on_progress=on_progress)
        if len(self.pins) == 0:
            V = V - V.mean(axis=0) + self.V0.mean(axis=0)
        return V


# ================== device selection

# Below this vertex count the numpy CPU path usually beats GPU kernel-launch
# and transfer overhead, so 'AUTO' only picks a GPU for larger meshes.
GPU_AUTO_MIN_VERTS = 20000

_TORCH_INFO = None


def torch_device_info_cached():
    """The cached probe result, or None if torch has not been probed yet
    (probing imports torch, which can take seconds — callers that must not
    block, like UI draw code, use this)."""
    return _TORCH_INFO


def reset_torch_info():
    """Forget the cached probe so the next torch_device_info() re-imports
    torch (used after installing torch into the running session)."""
    global _TORCH_INFO
    _TORCH_INFO = None
    import importlib
    importlib.invalidate_caches()


def torch_device_info():
    """(torch_available, cuda_available, mps_available), cached."""
    global _TORCH_INFO
    if _TORCH_INFO is None:
        try:
            import torch
            cuda = bool(torch.cuda.is_available())
            mps = bool(getattr(torch.backends, "mps", None) is not None
                       and torch.backends.mps.is_available())
            _TORCH_INFO = (True, cuda, mps)
        except Exception:
            _TORCH_INFO = (False, False, False)
    return _TORCH_INFO


def create_stylizer(V, F, cubeness=0.2, cube_axes=None, pins=None,
                    device='AUTO', target='CUBE', roundness=0.5,
                    keep_orientation=False):
    """Build a stylizer on the requested device.

    device : 'AUTO' | 'CPU' | 'CUDA' | 'MPS'  (case-insensitive)
    target : one of TARGETS; the GPU backend implements the cube only, so
             other targets always run on the CPU

    Returns (stylizer, resolved_device, warning). `resolved_device` is the
    device actually used; `warning` is a human-readable string when the
    request could not be honored (missing torch / unavailable device), else
    None. Always falls back to the numpy CPU solver rather than failing.
    """
    has_torch, cuda, mps = torch_device_info()
    req = str(device).upper()
    warning = None

    if req == 'CUDA' and not cuda:
        warning = ("CUDA unavailable" if has_torch
                   else "PyTorch not installed") + "; using CPU"
        resolved = 'CPU'
    elif req == 'MPS' and not mps:
        warning = ("MPS unavailable" if has_torch
                   else "PyTorch not installed") + "; using CPU"
        resolved = 'CPU'
    elif req in ('CUDA', 'MPS'):
        resolved = req
    elif req == 'AUTO':
        n = len(np.asarray(V).reshape(-1, 3))
        if n >= GPU_AUTO_MIN_VERTS and (cuda or mps):
            resolved = 'CUDA' if cuda else 'MPS'
        else:
            resolved = 'CPU'
    else:  # 'CPU' or anything unrecognized
        resolved = 'CPU'

    if resolved != 'CPU' and str(target).upper() != 'CUBE':
        if req != 'AUTO':
            warning = "the GPU solver supports the Cube target only; using CPU"
        resolved = 'CPU'

    if resolved == 'CPU':
        s = CubicStylizer(V, F, cubeness=cubeness, cube_axes=cube_axes,
                          pins=pins, target=target,
                          roundness=roundness, keep_orientation=keep_orientation)
        return s, 'CPU', warning

    try:
        try:
            from . import solver_torch
        except ImportError:
            import solver_torch
        s = solver_torch.TorchCubicStylizer(
            V, F, cubeness=cubeness, cube_axes=cube_axes, pins=pins,
            device=resolved.lower(), keep_orientation=keep_orientation)
        return s, resolved, warning
    except Exception as exc:
        s = CubicStylizer(V, F, cubeness=cubeness, cube_axes=cube_axes,
                          pins=pins, target=target,
                          roundness=roundness, keep_orientation=keep_orientation)
        return s, 'CPU', f"{resolved} backend failed ({exc}); using CPU"
