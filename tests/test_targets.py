# Target shapes and Keep Orientation (both solvers): run headless from the
# repo root:
#   blender --background --python-exit-code 1 --python tests/test_targets.py
import math
import sys
sys.path.insert(0, 'cubic_stylization')
sys.path.insert(0, 'cubic_stylization_c++')
import bmesh
import bpy
import numpy as np

import client
import solver

rng = np.random.default_rng(0)
ok = True


def check(cond, msg):
    global ok
    print(("  ok   " if cond else "  FAIL ") + msg)
    ok &= bool(cond)


# ---- projection math
X = rng.normal(size=(20000, 3)) * 2
t = rng.uniform(0, 1.5, size=20000)
D, ea, eb = solver._polytope('CUBE')
Zg = X - t[:, None] * solver._project_polytope(X / t[:, None], D, ea, eb)
check(np.abs(Zg - solver.target_prox('CUBE', X, t)).max() < 1e-12,
      "general projection with the cube's directions == soft-threshold")
for name in ('OCTAHEDRON', 'PYRAMID', 'HEX_COLUMN'):
    D, ea, eb = solver._polytope(name)
    verts = np.unique(np.round(np.concatenate([ea, eb]), 12), axis=0)
    Y = rng.normal(size=(20000, 3)) * 3
    P = solver._project_polytope(Y, D, ea, eb)
    worst = np.einsum('mk,mvk->mv', Y - P, verts[None] - P[:, None]).max()
    check((P @ D.T).max() <= 1 + 1e-9 and worst < 1e-9, f"{name}: projections inside P and optimal")
X = rng.normal(size=(20000, 3)) * 1.5
t = rng.uniform(0, 2, size=20000)
for r in (0.0, 0.5, 1.0):
    p = solver.rounded_exponent(r)
    a = np.abs(X)
    lo, hi = np.zeros_like(a), a.copy()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        g = mid - a + t[:, None] * p * mid ** (p - 1)
        hi = np.where(g > 0, mid, hi)
        lo = np.where(g > 0, lo, mid)
    dev = np.abs(solver.target_prox('ROUNDED_CUBE', X, t, p) - np.sign(X) * 0.5 * (lo + hi)).max()
    check(dev < 1e-12, f"rounded-cube step at roundness {r} matches bisection")

# ---- Suzanne's head (largest connected part)
bpy.ops.mesh.primitive_monkey_add()
ob = bpy.context.active_object
bpy.ops.object.modifier_add(type='SUBSURF')
ob.modifiers[-1].levels = 1
bpy.ops.object.modifier_apply(modifier=ob.modifiers[-1].name)
bm = bmesh.new()
bm.from_mesh(ob.data)
seen, parts = set(), []
for v in bm.verts:
    if v in seen:
        continue
    part, stack = [], [v]
    seen.add(v)
    while stack:
        u = stack.pop()
        part.append(u)
        for e in u.link_edges:
            w = e.other_vert(u)
            if w not in seen:
                seen.add(w)
                stack.append(w)
    parts.append(part)
for part in sorted(parts, key=len)[:-1]:
    bmesh.ops.delete(bm, geom=part, context='VERTS')
bmesh.ops.triangulate(bm, faces=bm.faces)
bm.verts.index_update()
V = np.array([v.co[:] for v in bm.verts])
F = np.array([[v.index for v in f.verts] for f in bm.faces])
size = np.ptp(V, axis=0).max()
area = solver.CubicStylizer(V, F, cubeness=0.0).area


def turned(Vo):
    c0, c = area @ V / area.sum(), area @ Vo / area.sum()
    U, _, Wt = np.linalg.svd((V - c0).T @ (area[:, None] * (Vo - c)))
    R = Wt.T @ U.T
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def aligned(Vo, tgt):
    Dt = solver.target_directions(tgt)
    n = np.cross(Vo[F[:, 1]] - Vo[F[:, 0]], Vo[F[:, 2]] - Vo[F[:, 0]])
    ar = np.linalg.norm(n, axis=1)
    n /= ar[:, None]
    return ar[(n @ Dt.T).max(1) > math.cos(math.radians(10))].sum() / ar.sum()


def cpp(tgt, keep=False, pins=None, iters=60, **kw):
    s, _, _ = client.create_stylizer(V, F, cubeness=0.6, target=tgt, pins=pins,
                                     keep_orientation=keep, **kw)
    with s:
        return s.run(iterations=iters)


for tgt in solver.TARGETS:
    py = solver.CubicStylizer(V, F, cubeness=0.6, target=tgt).run(iterations=20)
    check(np.abs(py - cpp(tgt, iters=20)).max() / size < 1e-6, f"{tgt}: python == c++")
    free, held = cpp(tgt), cpp(tgt, keep=True)
    check(aligned(free, tgt) > aligned(V, tgt) + 0.05, f"{tgt}: target shape forms "
          f"({aligned(V, tgt):.2f} -> {aligned(free, tgt):.2f} of surface within 10 deg)")
    check(turned(held) < 0.5 and aligned(held, tgt) >= 0.9 * aligned(free, tgt),
          f"{tgt}: Keep Orientation turns {turned(free):.1f} -> {turned(held):.2f} deg, shape still forms")
    # compared at 20 iterations: past that, numpy/Eigen rounding differences
    # get amplified while a solve is still settling (~1e-5 of the mesh size
    # by 60 iterations, with or without Keep Orientation)
    py = solver.CubicStylizer(V, F, cubeness=0.6, target=tgt, keep_orientation=True).run(iterations=20)
    check(np.abs(py - cpp(tgt, keep=True, iters=20)).max() / size < 1e-6,
          f"{tgt}: python == c++ with Keep Orientation")
pins = [0, len(V) // 4, len(V) // 2, 3 * len(V) // 4]
check(np.array_equal(cpp('OCTAHEDRON', pins=pins, iters=40), cpp('OCTAHEDRON', True, pins, 40)),
      "with pins, Keep Orientation is a no-op")
print("ALL-OK" if ok else "FAILED")
