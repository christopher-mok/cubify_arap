# Preferred-direction ("Gauss map") helpers for the target shapes: preset
# directions, closing and drawing a custom direction set, extracting facet
# directions from a reference mesh, and finding the cube orientation that
# best fits a mesh as it is.
#
# A direction set D describes the polytope P = {y : d . y <= 1}; the target
# term is P's support function, smallest (exactly 1) at the d. The geometry
# here mirrors src/target_shape.h.
#
# No bpy imports: this module can be used (and tested) outside Blender.

import numpy as np

PYRAMID_NORMAL_ELEVATION_DEG = 38.0  # src/target_shape.h
MAX_CUSTOM_DIRECTIONS = 64     # src/target_shape.h kMaxCustomDirections
MAX_REFERENCE_DIRECTIONS = 32  # facets taken from one reference (keeps solves fast)


def rounded_exponent(roundness):
    """Rounded cube's p for a Roundness in [0, 1] (src/target_shape.h)."""
    return 2.0 - 0.5 * 10.0 ** (-min(max(float(roundness), 0.0), 1.0))


def preset_directions(target):
    """(k, 3) preferred directions of a preset target (target frame)."""
    if target in ('CUBE', 'ROUNDED_CUBE'):
        return np.concatenate([np.eye(3), -np.eye(3)])
    if target == 'OCTAHEDRON':
        s = np.array([1.0, -1.0])
        return np.stack(np.meshgrid(s, s, s, indexing='ij'), -1).reshape(-1, 3) / np.sqrt(3.0)
    if target == 'PYRAMID':
        a = np.radians(PYRAMID_NORMAL_ELEVATION_DEG)
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [-c, 0, s], [0, c, s], [0, -c, s], [0, 0, -1.0]])
    if target == 'HEX_COLUMN':
        k = np.arange(6) * np.pi / 3.0
        side = np.stack([np.cos(k), np.sin(k), np.zeros(6)], -1)
        return np.concatenate([side, [[0, 0, 1.0], [0, 0, -1.0]]])
    raise ValueError(f"no preset directions for {target!r}")


def normalized_unique(D, tol=1e-9):
    """Unit rows of D, dropping zero rows and near-duplicates (as the server
    does)."""
    out = []
    for d in np.asarray(D, dtype=np.float64).reshape(-1, 3):
        n = np.linalg.norm(d)
        if not np.isfinite(n) or n < 1e-9:
            continue
        u = d / n
        if all(u @ e <= 1.0 - tol for e in out):
            out.append(u)
    return np.array(out).reshape(-1, 3)


def open_directions(D):
    """Unit directions y that every row of D points away from (no surface
    would face y, so the shape is open); empty if D is closed. Exact: if the
    open region is not empty it contains some d_i x d_j or +-d_i."""
    D = normalized_unique(D)
    if len(D) == 0:
        return np.array([[0.0, 0.0, 1.0]])
    cand = [D, -D]
    i, j = np.triu_indices(len(D), 1)
    C = np.cross(D[i], D[j])
    n = np.linalg.norm(C, axis=1)
    C = C[n > 1e-9] / n[n > 1e-9, None]
    cand += [C, -C]
    Y = np.concatenate(cand)
    return Y[(Y @ D.T).max(axis=1) <= 1e-9]


def open_direction(D):
    """One direction nothing in D faces, or None if D is closed."""
    Y = open_directions(D)
    return Y[0] if len(Y) else None


def close_directions(D, max_added=8):
    """D plus the fewest directions (greedy) needed to make the shape
    closed: each added direction faces a side nothing else did. Returns
    (closed_D, number_added)."""
    D = normalized_unique(D)
    added = 0
    while added < max_added:
        Y = open_directions(D)
        if len(Y) == 0:
            break
        # the centre of the open region faces it squarely (a pyramid's four
        # sides get a flat base); fall back to one witness if it cancels out
        m = Y.sum(axis=0)
        y = m / np.linalg.norm(m) if np.linalg.norm(m) > 1e-6 else Y[0]
        D = np.concatenate([D, y[None]])
        added += 1
    return D, added


def polytope(D):
    """(vertices, edges) of P = {y : D y <= 1} for a closed direction set;
    edges index into vertices. Used for drawing the target."""
    D = normalized_unique(D)
    K = len(D)
    verts, tight = [], []
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
    edges = [(a, b) for a in range(len(verts)) for b in range(a + 1, len(verts))
             if len(tight[a] & tight[b]) >= 2]
    return np.array(verts).reshape(-1, 3), np.array(edges, dtype=np.int64).reshape(-1, 2)


def face_normals(V, F):
    """Unit face normals and areas of a triangle mesh."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = 0.5 * np.linalg.norm(n, axis=1)
    ok = a > 1e-20
    return n[ok] / (2.0 * a[ok, None]), a[ok]


def bin_normals(N, A, resolution=40):
    """Merge normals that fall in the same small cell (cells ~1/resolution
    wide): area-weighted mean direction and total area per cell."""
    key = np.round(N * resolution).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    S = np.zeros((inv.max() + 1, 3))
    np.add.at(S, inv, N * A[:, None])
    W = np.bincount(inv, weights=A)
    n = np.linalg.norm(S, axis=1)
    ok = n > 1e-20
    return S[ok] / n[ok, None], W[ok]


def cluster_normals(N, A, merge_angle_deg=10.0, min_area=0.005, max_dirs=32):
    """Dominant facet directions of a mesh from its face normals N and areas
    A: repeatedly seed at the heaviest remaining normal, mean-shift it to the
    area-weighted mean of everything within merge_angle, and take that
    cluster out. Clusters smaller than min_area (fraction of the total) are
    dropped. A flat-faceted (low-poly) mesh gives exactly its face normals.
    Returns (directions, area_fractions), heaviest first."""
    N, W = bin_normals(np.asarray(N, float), np.asarray(A, float))
    total = W.sum()
    cos_t = np.cos(np.radians(merge_angle_deg))
    alive = np.ones(len(N), bool)
    dirs, fracs = [], []
    while alive.any() and len(dirs) < 4 * max_dirs:
        idx = np.flatnonzero(alive)
        c = N[idx[np.argmax(W[idx])]]
        for _ in range(20):
            near = idx[N[idx] @ c >= cos_t]
            m = (N[near] * W[near, None]).sum(axis=0)
            c_new = m / np.linalg.norm(m)
            if c_new @ c > 1.0 - 1e-12:
                break
            c = c_new
        near = idx[N[idx] @ c >= cos_t]
        if len(near) == 0:  # cannot happen for a unit seed, but never loop forever
            break
        alive[near] = False
        dirs.append(c)
        fracs.append(W[near].sum() / total)
    order = np.argsort(fracs)[::-1]
    keep = [k for k in order if fracs[k] >= min_area][:max_dirs]
    return np.array([dirs[k] for k in keep]).reshape(-1, 3), np.array([fracs[k] for k in keep])


def _random_rotations(n, rng):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1)[:, None]
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def _small_rotations(n, angle, rng):
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    th = rng.normal(size=n) * angle
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    K = K - K.transpose(0, 2, 1)
    s, c = np.sin(th)[:, None, None], (1 - np.cos(th))[:, None, None]
    return np.eye(3)[None] + s * K + c * (K @ K)


def stylization_cost(N, W, Rs, target, D=None, p=1.8):
    """Area-weighted target term of normals N under each candidate axes
    matrix in Rs (columns = target axes): sum_i W_i f(R^T n_i)."""
    Y = np.einsum('mk,rkj->rmj', N, Rs)  # rows: (R^T n)^T
    if target == 'ROUNDED_CUBE':
        return (np.abs(Y) ** p).sum(axis=2) @ W
    verts, _ = polytope(preset_directions(target) if D is None else D)
    return (Y @ verts.T).max(axis=2) @ W  # support function = max over P's vertices


def auto_orientation(N, A, target, D=None, p=1.8, seed=0):
    """Cube Orientation (3x3, columns = target axes) under which the mesh
    with face normals N and areas A already best fits the target, so
    stylizing it changes the least. Global random search plus local
    refinement; among equally good fits the one closest to no rotation
    wins, so a symmetric target does not flip arbitrarily."""
    rng = np.random.default_rng(seed)
    Nb, Wb = bin_normals(np.asarray(N, float), np.asarray(A, float), resolution=12)
    Wb = Wb / Wb.sum()

    def cost(Rs):
        out = []
        for k in range(0, len(Rs), 256):
            out.append(stylization_cost(Nb, Wb, Rs[k:k + 256], target, D, p))
        return np.concatenate(out)

    Rs = np.concatenate([np.eye(3)[None], _random_rotations(4000, rng)])
    c = cost(Rs)
    seeds = Rs[np.argsort(c)[:12]]
    found = []
    for R in seeds:
        best, best_c = R, cost(R[None])[0]
        for angle in np.radians(np.geomspace(8.0, 0.02, 24)):
            cand = _small_rotations(64, angle, rng) @ best
            cc = cost(cand)
            k = np.argmin(cc)
            if cc[k] < best_c:
                best, best_c = cand[k], cc[k]
        found.append((best_c, best))
    top = min(f[0] for f in found)
    near = [R for cst, R in found if cst <= top * (1 + 1e-3) + 1e-12]
    angle = [np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) for R in near]
    return near[int(np.argmin(angle))]
