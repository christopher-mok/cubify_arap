# Checks the thin-wall post step on the Utah teapot from ../../test_objs, a
# solid-shell mesh whose spout/rim walls cross each other after cubifying.
#
#   python tests/test_thin_walls.py
#
# Needs numpy and a built server in bin/. No Blender required.

import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("parts_test", os.path.join(HERE, "test_parts_and_flat.py"))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
client, check, load_obj, REPO = t.client, t.check, t.load_obj, t.REPO


def aligned(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = np.linalg.norm(fn, axis=1)
    ok = a > 1e-20
    return (a[ok] * (np.abs(fn[ok] / a[ok, None]).max(1) > 0.98)).sum() / a[ok].sum()


def main():
    V, F = load_obj(os.path.join(REPO, "test_objs", "utah_teapot.obj"))
    s, _, _ = client.create_stylizer(V, F, cubeness=0.2)
    C = s.run(iterations=30)
    s.close()

    t0 = time.time()
    Vf, st = client.fix_thin_walls(V, C, F)
    dt = time.time() - t0
    print(f"teapot: {st} in {dt:.3f}s")
    check("walls found (solid-shell mesh)", st["walls"] > 0.5 * len(V), f"{st['walls']} of {len(V)}")
    check("cubify crossed some walls", st["crossed_before"] > 100, f"{st['crossed_before']}")
    check("fix uncrosses them", st["crossed_after"] == 0, f"{st['crossed_after']} left")
    check("fix is local", st["moved"] < 0.15 * len(V) and st["max_move"] < 0.2,
          f"{st['moved']} vertices moved, max {st['max_move']:.3f}")
    a0, a1 = aligned(C, F), aligned(Vf, F)
    check("cubic style kept", a1 > a0 - 0.03, f"aligned area {a0:.2f} -> {a1:.2f}")

    V2, st2 = client.fix_thin_walls(V, Vf, F)
    check("running it again changes nothing much", st2["crossed_before"] == 0
          and np.abs(V2 - Vf).max() < 0.05, f"max move {np.abs(V2 - Vf).max():.3f}")

    V3, st3 = client.fix_thin_walls(V, V, F)
    check("uncubified mesh is left alone", st3["moved"] == 0 and np.array_equal(V3, V))

    try:
        client.fix_thin_walls(V, C[:-1], F)
        check("mismatched sizes rejected", False)
    except ValueError:
        check("mismatched sizes rejected", True)

    client.shutdown_server()
    print("FAILED: " + ", ".join(t.FAILED) if t.FAILED else "all checks passed")
    return 1 if t.FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
