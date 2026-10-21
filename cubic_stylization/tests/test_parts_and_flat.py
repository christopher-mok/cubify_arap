# Checks the two extensions beyond the paper on the Utah teapot from
# ../../test_objs (body + separate lid in one mesh):
#   - contact seating: the lid stays on the rim it rests on
#   - Square Flat Regions: the lid's round outline turns square
#
#   python tests/test_parts_and_flat.py
#
# Needs numpy and a built server in bin/. No Blender required.

import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON = os.path.dirname(HERE)
REPO = os.path.dirname(ADDON)

spec = importlib.util.spec_from_file_location("cubify_client", os.path.join(ADDON, "client.py"))
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f" ({detail})" if detail else ""))
    if not cond:
        FAILED.append(name)


def load_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            p = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            F += [(p[0], p[i], p[i + 1]) for i in range(1, len(p) - 1)]
    return np.array(V), np.array(F)


def components(n, F):
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, c in F:
        for u, v in ((a, b), (b, c)):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)
    return np.array([find(i) for i in range(n)])


def squareness(P):
    """Top-view outline: 0 = circle, 1 = axis-aligned square."""
    P = P[:, :2] - P[:, :2].mean(0)
    ax = np.abs(P).max(0).mean()
    dg = np.abs(P @ (np.array([[1, 1], [1, -1]]).T / np.sqrt(2))).max(0).mean()
    return (dg / ax - 1) / (np.sqrt(2) - 1)


def main():
    V, F = load_obj(os.path.join(REPO, "test_objs", "utah_teapot.obj"))
    Vl, Fl = load_obj(os.path.join(REPO, "test_objs", "teapot_lid.obj"))
    roots = components(len(V), F)
    lid = roots != roots[0]
    body = ~lid
    print(f"teapot: {len(V)} vertices, lid {lid.sum()}")

    # contact pairs: lid vertices touching the body, and their nearest body vertex
    Vb = V[body]
    li, bi = [], []
    for i in np.flatnonzero(lid):
        d = np.linalg.norm(Vb - V[i], axis=1)
        k = np.argmin(d)
        if d[k] < 0.04:
            li.append(i)
            bi.append(np.flatnonzero(body)[k])
    li, bi = np.array(li), np.array(bi)

    results = {}
    for relax in (1.0, 0.01):
        t0 = time.time()
        s, _, _ = client.create_stylizer(V, F, cubeness=0.2, flat_relax=relax)
        o = s.run(iterations=30)
        s.close()
        dt = time.time() - t0
        results[relax] = o
        seat = ((o[li] - o[bi]) - (V[li] - V[bi]))[:, 2].mean()
        tag = "Square Flat Regions " + ("on" if relax < 1 else "off")
        print(f"{tag}: {dt:.2f}s")
        check(f"{tag}: lid stays seated", abs(seat) < 1e-3, f"seat height {seat:+.1e}")
        check(f"{tag}: body keeps its centroid",
              np.linalg.norm(o[body].mean(0) - V[body].mean(0)) < 1e-9)
        check(f"{tag}: finite", np.all(np.isfinite(o)))

    sq_off, sq_on = squareness(results[1.0][lid]), squareness(results[0.01][lid])
    check("off: lid stays round (paper behaviour)", sq_off < 0.15, f"squareness {sq_off:.2f}")
    check("on: lid outline squares", sq_on > 0.3, f"squareness {sq_on:.2f}")

    s, _, _ = client.create_stylizer(Vl, Fl, cubeness=0.2, flat_relax=0.01)
    sq_alone = squareness(s.run(iterations=30))
    s.close()
    check("on: lid file alone squares too", sq_alone > 0.3, f"squareness {sq_alone:.2f}")

    client.shutdown_server()
    print("FAILED: " + ", ".join(FAILED) if FAILED else "all checks passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
