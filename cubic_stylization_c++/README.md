# Cubic Stylization + ARAP — Blender Add-on (C++ solver)

The same tools as the Python add-on in `../cubic_stylization/` — cubify,
style-as-process bakes, cubify every frame, pins and interactive ARAP
manipulation — but every solve runs in a native **C++/Eigen** process
(`cubify_server`). Blender sends the mesh to the server and gets vertex
positions back; the add-on only handles UI and mesh I/O.

Results match the numpy solver to ~1e-12 (same algorithm, double
precision) and it is roughly 10–20× faster on CPU:

| vertices | numpy (Python add-on) | C++ (this add-on) |
|---------:|----------------------:|------------------:|
|    2,562 |                0.37 s |            0.02 s |
|   40,962 |                3.3 s  |            0.33 s |

(cubify, 20 resp. 10 iterations incl. setup, λ=0.4, 24-thread CPU.)
Interactive drags at 2 iterations take ~5 ms at 10k vertices and ~30 ms
at 41k.

## Install

1. Build the server once (below), or use a zip that already contains
   `bin/cubify_server(.exe)` for your platform.
2. **Edit → Preferences → Add-ons → Install…**, pick
   `cubic_stylization_c++.zip`, enable **Mesh: Cubic Stylization (C++)**.
3. Tools are in the 3D Viewport sidebar (**N**) → **Cubify C++** tab.

Both add-ons can be enabled together: they use separate operators,
settings and sidebar tabs. They share the `CubifyPins` vertex group, so
pins set in one are used by the other.

## Building the server

Requirements: CMake ≥ 3.18 and a C++17 compiler (Visual Studio 2019+ on
Windows, Xcode command line tools on macOS, gcc/clang on Linux). Eigen 3.4
is used from an installed package if CMake finds one, otherwise it is
downloaded automatically on the first build (or pass
`-DEIGEN3_INCLUDE_DIR=...`).

- **From Blender**: add-on preferences → **Build Server**. Runs in the
  background (log in the system console). On Windows the CMake bundled with
  Visual Studio is found automatically.
- **From a terminal**: `python builder.py` in this folder
  (`python builder.py --zip` also writes `../cubic_stylization_c++.zip`).
- **Manually**: `cmake -S src -B build && cmake --build build --config Release`.

The executable is written to `bin/`. It has no runtime dependencies (static
MSVC runtime, std::thread instead of OpenMP). Set **Server Executable** in
the preferences to use one from somewhere else; **Check Server** restarts
it and shows its version.

`bin/cubify_server --selftest` cubifies a generated sphere and prints
timings, without Blender.

## Using it

Identical to the Python add-on (see its README), with one change:

- **Threads** replaces **Device**: CPU threads for the solver, 0 = all
  cores. There is no GPU backend and no PyTorch install. For reference,
  the Python README's 163,842-vertex benchmark (10 iterations) takes 2.3 s
  here, against 4.0 s for its Metal GPU run (different machines).

Everything else — Cubeness, Cube Orientation, Iterations / ADMM
Iterations, Apply to Copy, Style as Process (Cubeness Ramp / Iterations,
Steps, Iterations / Step, Frame Step), Cubify Every Frame (to Copy), Set /
Add / Clear Pins, Stylized Drag, Drag Iterations, Start Manipulation — works
the same way.

## Architecture

```
Blender (Python)                          cubify_server (C++)
__init__.py  operators, UI                server.cpp        request loop, sessions
client.py    process + protocol  ──pipe──▶ cubic_stylizer.*  solver (Eigen)
builder.py   CMake build                  thread_pool.h     parallel_for
```

- One server process per Blender session, started on first use and
  stopped when the add-on is disabled (or when Blender exits, because
  stdin closes).
- Binary protocol over stdin/stdout, documented in `src/protocol.h`.
  Each mesh gets a **session** on the server holding its prefactorized
  system and ADMM state. Start Manipulation factorizes once, and each
  mouse move sends only the pin targets; the server warm-starts from its
  own copy of the previous result.
- Solve progress is streamed back and drives Blender's progress bar.

Solver differences from the numpy version (none change results on
normal meshes):

- The global step uses a sparse Cholesky (`SimplicialLDLT`) of the
  pinned Laplacian instead of LU / conjugate gradient. The pinned
  system is symmetric positive definite, so Cholesky applies.
- Loose parts of one object keep their relative placement. A connected
  part with no pins floats: after every iteration it is translated back
  to its rest-pose centroid. Parts with pins are held by the pins, and
  loose vertices stay put. The numpy solver fixes only vertex 0, so
  unpinned parts drift relative to each other. This also means a bake of
  an unpinned mesh keeps its centroid fixed, rather than vertex 0.
- The local step runs each vertex's ADMM to its own convergence on a
  thread pool, rather than batched numpy over the still-active vertices.
  The arithmetic is the same.
- Style-as-process bakes solve all steps before writing shape keys, so a
  failed solve leaves the mesh untouched.

## Tests

```sh
python tests/test_against_python.py      # C++ vs the numpy solver, timings
blender -b --factory-startup --python tests/blender_smoke_test.py
```

The Blender smoke test runs every operator except the modal drag, which
needs a viewport. It also checks that topology and UVs are preserved and
that both add-ons coexist. It passes on Blender 3.6 and 5.0.
