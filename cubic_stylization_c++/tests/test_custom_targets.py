# Custom targets, SET_STYLE and the gauss.py helpers. Plain Python (numpy),
# against the built server:  python tests/test_custom_targets.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

import client
import gauss

ok = True


def check(cond, msg):
    global ok
    print(("  ok   " if cond else "  FAIL ") + msg)
    ok &= bool(cond)


def icosphere(level=3):
    t = (1 + 5 ** 0.5) / 2
    V = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
         [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    F = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
         [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
         [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    for _ in range(level):
        cache, F2 = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = V[a] + V[b]
                V.append(m / np.linalg.norm(m))
                cache[k] = len(V) - 1
            return cache[k]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            F2 += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        F = F2
    return np.array(V), np.array(F)


def run(V, F, iterations=30, **kw):
    s, _, _ = client.create_stylizer(V, F, cubeness=0.8, **kw)
    with s:
        return s.run(iterations=iterations)


V, F = icosphere(3)
V[:, 0] *= 1.3  # not perfectly symmetric

# ---- custom == presets (compared at 5 iterations: the presets' directions
# can differ from numpy's in the last bit, which long solves amplify)
for preset in ("CUBE", "OCTAHEDRON", "PYRAMID", "HEX_COLUMN"):
    a = run(V, F, iterations=5, target=preset)
    b = run(V, F, iterations=5, target="CUSTOM", directions=gauss.preset_directions(preset))
    check(np.abs(a - b).max() < 1e-8, f"custom with {preset.lower()} directions == {preset} preset "
                                      f"(max dev {np.abs(a - b).max():.1e})")

# ---- open sets are refused with a usable message, and close_directions fixes them
sides = gauss.preset_directions("PYRAMID")[:4]  # a pyramid without its base
try:
    run(V, F, target="CUSTOM", directions=sides)
    check(False, "open direction set refused")
except client.ServerError as exc:
    check("closed shape" in str(exc) and "nothing faces" in str(exc), f"open set refused: {exc}")
closed, added = gauss.close_directions(sides)
check(added == 1 and np.allclose(closed[-1], [0, 0, -1], atol=1e-9),
      f"pyramid sides closed with {added} direction {np.round(closed[-1], 3)} (its base)")
check(gauss.open_direction(closed) is None and gauss.open_direction(sides) is not None,
      "open_direction agrees with the server")
try:
    run(V, F, target="CUSTOM", directions=gauss.preset_directions("CUBE")[:3])
    check(False, "fewer than 4 directions refused")
except client.ServerError as exc:
    check("at least 4" in str(exc), "fewer than 4 directions refused")

# ---- SET_STYLE restarts a session exactly; chunked solving == one solve
s, _, _ = client.create_stylizer(V, F, cubeness=0.8, target="CUBE")
with s:
    s.run(iterations=10)
    s.set_style(target="CUSTOM", directions=gauss.preset_directions("OCTAHEDRON"),
                keep_orientation=True, cube_axes=gauss._random_rotations(1, np.random.default_rng(1))[0])
    restyled = s.run(iterations=30)
    s.set_style(target="CUSTOM", directions=gauss.preset_directions("OCTAHEDRON"),
                keep_orientation=True, cube_axes=gauss._random_rotations(1, np.random.default_rng(1))[0])
    chunked = s.solve(iterations=10)
    for _ in range(2):
        chunked = s.solve(iterations=10, warm_last=True)
fresh = run(V, F, target="CUSTOM", directions=gauss.preset_directions("OCTAHEDRON"),
            keep_orientation=True, cube_axes=gauss._random_rotations(1, np.random.default_rng(1))[0])
check(np.array_equal(restyled, fresh), "set_style + solve == fresh session (bit-identical)")
check(np.array_equal(chunked, fresh), "3 chunks of 10 iterations == one solve of 30 (bit-identical)")

# ---- facet directions from a reference mesh
k = 7
ang = np.arange(k) * 2 * np.pi / k
ring = np.stack([np.cos(ang), np.sin(ang)], -1)
PV = np.concatenate([np.c_[ring, -np.ones(k)], np.c_[ring, np.ones(k)], [[0, 0, -1], [0, 0, 1]]])
PF = []
for i in range(k):
    j = (i + 1) % k
    PF += [[i, j, k + j], [i, k + j, k + i], [2 * k, j, i], [2 * k + 1, k + i, k + j]]
PF = np.array(PF)
N, A = gauss.face_normals(PV, PF)
D, frac = gauss.cluster_normals(N, A, merge_angle_deg=10)
side = np.stack([np.cos(ang + np.pi / k), np.sin(ang + np.pi / k), np.zeros(k)], -1)
expect = np.concatenate([side, [[0, 0, 1], [0, 0, -1]]])
match = max(np.min(np.linalg.norm(expect - d, axis=1)) for d in D)
check(len(D) == k + 2 and match < 1e-9, f"7-sided prism -> {len(D)} directions, exact to {match:.1e}")

rng = np.random.default_rng(3)
bump = V * (1 + 0.02 * rng.normal(size=(len(V), 1)))
cubeish = np.sign(bump) * np.abs(bump) ** 0.25  # a lumpy rounded box
N, A = gauss.face_normals(cubeish, F)
D, frac = gauss.cluster_normals(N, A, merge_angle_deg=15, min_area=0.02)
axes = np.concatenate([np.eye(3), -np.eye(3)])
err = max(np.degrees(np.arccos(np.clip((D @ axes.T).max(1), -1, 1))))
check(len(D) == 6 and err < 6, f"lumpy box -> {len(D)} dominant directions, within {err:.1f} deg of the axes")

# ---- auto orientation recovers a known rotation of a box
R = gauss._random_rotations(1, np.random.default_rng(7))[0]
box_V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * [1.5, 1, 0.7]
box_F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
N, A = gauss.face_normals(box_V @ R.T, box_F)
Rf = gauss.auto_orientation(N, A, "CUBE")
miss = np.degrees(np.arccos(np.clip(np.abs(N @ Rf).max(axis=1), -1, 1))).max()
check(miss < 0.01, f"rotated box: auto orientation aligns every face (worst face {miss:.4f} deg off)")
check(gauss.stylization_cost(N, A / A.sum(), np.eye(3)[None], "CUBE")[0] > 1.05,
      "without it the rotated box is misaligned")
Rf = gauss.auto_orientation(*gauss.face_normals(box_V, box_F), "CUBE")
check(np.allclose(Rf, np.eye(3), atol=1e-6), "an aligned box keeps the identity (no arbitrary flip)")

print("ALL-OK" if ok else "FAILED")
