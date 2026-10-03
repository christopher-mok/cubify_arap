# Checks the C++ server against the reference numpy solver
# (../../cubic_stylization/solver.py) on the same inputs, and times both.
#
#   python tests/test_against_python.py
#
# Needs numpy (+ scipy for the reference solver's fast path) and a built
# server in bin/. No Blender required.

import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON = os.path.dirname(HERE)
REPO = os.path.dirname(ADDON)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


client = _load("cubify_client", os.path.join(ADDON, "client.py"))
ref = _load("cubify_ref_solver", os.path.join(REPO, "cubic_stylization", "solver.py"))


def icosphere(subdiv):
    t = (1.0 + 5 ** 0.5) / 2.0
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
         (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
         (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
         (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    for _ in range(subdiv):
        mid = {}

        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                p = V[a] + V[b]
                V.append(p / np.linalg.norm(p))
                mid[k] = len(V) - 1
            return mid[k]
        F2 = []
        for a, b, c in F:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    return np.array(V), np.array(F, dtype=np.int64)


def bumpy(V, seed=0):
    """Deterministic lumpy blob, so the result is not trivially symmetric."""
    rng = np.random.default_rng(seed)
    k = rng.normal(size=(6, 3))
    r = 1.0 + 0.15 * np.sin(V @ k.T).sum(axis=1) / 3.0
    return V * r[:, None] * np.array([1.3, 1.0, 0.8])


def rel_err(a, b, V0):
    bbox = np.linalg.norm(V0.max(0) - V0.min(0))
    return np.max(np.linalg.norm(a - b, axis=1)) / bbox


FAILED = []


def check(name, a, b, V0, tol):
    e = rel_err(a, b, V0)
    ok = e < tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max deviation {e:.2e} of bbox (tol {tol:.0e})")
    if not ok:
        FAILED.append(name)


def main():
    V, F = icosphere(4)
    V = bumpy(V)
    print(f"mesh: {len(V)} vertices, {len(F)} faces")
    A = np.array([[0.8660254, -0.5, 0.0], [0.5, 0.8660254, 0.0], [0.0, 0.0, 1.0]])

    # 1) one-shot cubify, no pins, tilted cube axes
    s_py = ref.CubicStylizer(V, F, cubeness=0.4, cube_axes=A)
    s_cc, label, _ = client.create_stylizer(V, F, cubeness=0.4, cube_axes=A)
    print("server:", client.get_server().version, "/", label)
    prog = []
    t0 = time.time(); a = s_py.run(iterations=20, admm_iters=100); t_py = time.time() - t0
    t0 = time.time(); b = s_cc.run(iterations=20, admm_iters=100,
                                    on_progress=lambda i, n: prog.append(i))
    t_cc = time.time() - t0
    print(f"cubify: numpy {t_py:.2f}s, C++ {t_cc:.2f}s ({t_py / t_cc:.1f}x)")
    check("cubify (lambda=0.4, rotated axes)", a, b, V, 1e-4)
    if prog != list(range(1, len(prog) + 1)) or not prog:
        FAILED.append("progress"); print("  FAIL progress:", prog)
    s_cc.close()

    # 2) ARAP with pins dragged, warm-started like the modal operator
    pins = list(range(0, len(V), 97))
    s_py = ref.CubicStylizer(V, F, cubeness=0.0, pins=pins)
    s_cc, _, _ = client.create_stylizer(V, F, cubeness=0.0, pins=pins)
    assert np.array_equal(s_cc.pins, s_py.pins)
    pp = V[s_py.pins].copy()
    Vp = Vc = V
    for step in range(5):
        pp[0] += np.array([0.05, 0.02, -0.03])
        Vp = s_py.solve(pin_pos=pp, V_init=Vp, iterations=2, admm_iters=100)
        Vc = s_cc.solve(pin_pos=pp, warm_last=True, iterations=2, admm_iters=100)
    check("ARAP drag (pinned, warm_last)", Vp, Vc, V, 1e-6)
    check("pins hit their targets", Vc[s_cc.pins], pp, V, 1e-9)
    s_cc.close()

    # 3) stylized drag + cubeness ramp (style-as-process bake path)
    s_py = ref.CubicStylizer(V, F, cubeness=0.6, pins=pins)
    s_cc, _, _ = client.create_stylizer(V, F, cubeness=0.6, pins=pins)
    Vp = Vc = V
    for k in range(1, 6):
        s_py.lam = s_cc.lam = 0.6 * k / 5
        Vp = s_py.solve(V_init=Vp, iterations=3, admm_iters=100)
        Vc = s_cc.solve(V_init=Vc, iterations=3, admm_iters=100)
    check("cubeness ramp with pins", Vp, Vc, V, 1e-4)
    s_cc.close()

    # 4) disconnected parts of one object (different sizes) + a loose vertex:
    #    every unpinned part keeps its centroid, so relative placement holds
    n = len(V)
    V2 = np.vstack([V, V + [3.0, 0, 0], 0.3 * V + [0, -4, 1], [[9.0, 9.0, 9.0]]])
    F2 = np.vstack([F, F + n, F + 2 * n])
    parts = [slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n)]
    s_cc, _, _ = client.create_stylizer(V2, F2, cubeness=1.0, threads=2)
    for name, b in (("run", s_cc.run(iterations=20)),
                    ("solve", s_cc.solve(iterations=20))):
        drift = max(np.linalg.norm(b[p].mean(0) - V2[p].mean(0)) for p in parts)
        d = b[parts[1]] - b[parts[0]]  # identical parts: identical up to translation
        same_shape = np.max(np.abs(d - d.mean(0))) < 1e-6
        ok = (np.all(np.isfinite(b)) and drift < 1e-12 and same_shape
              and np.allclose(b[-1], V2[-1]))
        print(f"  {'ok  ' if ok else 'FAIL'} 3 parts + loose vertex ({name}): "
              f"part centroid drift {drift:.1e}, identical parts match: {same_shape}")
        if not ok:
            FAILED.append(f"components ({name})")
    s_cc.close()

    #    with one part pinned, the pins hold it and the others still float
    pins2 = list(range(0, n, 50))
    s_cc, _, _ = client.create_stylizer(V2, F2, cubeness=1.0, pins=pins2)
    b = s_cc.run(iterations=20)
    drift = max(np.linalg.norm(b[p].mean(0) - V2[p].mean(0)) for p in parts[1:])
    ok = drift < 1e-12 and np.allclose(b[pins2], V2[pins2])
    print(f"  {'ok  ' if ok else 'FAIL'} one part pinned: pins held, unpinned "
          f"part drift {drift:.1e}")
    if not ok:
        FAILED.append("components (pinned)")
    s_cc.close()

    # 5) error reporting
    try:
        client.create_stylizer(V, F, pins=[len(V) + 5])
        FAILED.append("bad pin accepted"); print("  FAIL bad pin accepted")
    except client.ServerError as exc:
        print(f"  ok   bad pin rejected: {exc}")
    assert client.get_server().alive()

    # 6) larger mesh timing
    Vb, Fb = icosphere(6)
    Vb = bumpy(Vb, 1)
    s_py = ref.CubicStylizer(Vb, Fb, cubeness=0.4)
    t0 = time.time(); s_cc, _, _ = client.create_stylizer(Vb, Fb, cubeness=0.4)
    b = s_cc.run(iterations=10); t_cc = time.time() - t0
    t0 = time.time(); a = s_py.run(iterations=10); t_py = time.time() - t0
    print(f"{len(Vb)} vertices, 10 iterations incl. setup: numpy {t_py:.2f}s, "
          f"C++ {t_cc:.2f}s ({t_py / t_cc:.1f}x)")
    check("large mesh cubify", a, b, Vb, 1e-4)
    s_cc.close()

    client.shutdown_server()
    print("FAILED: " + ", ".join(FAILED) if FAILED else "all checks passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
