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

1. Use the all-in-one zip (it bundles server binaries for every platform:
   `cubify_server.exe` for Windows, `cubify_server-macos` as a
   universal2 arm64+x86_64 build, `cubify_server-linux`; the add-on
   launches the one matching your OS) — or build the server once (below).
   CI (`.github/workflows/build-cpp.yml`) builds all three and assembles
   the zip on every push; tagged `v*` releases get it attached.
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
  (`python builder.py --zip` also writes `../cubic_stylization_c++.zip`;
  `--universal` builds macOS arm64+x86_64). Without CMake, macOS/Linux
  fall back to a direct clang++/g++ build against an installed Eigen.
- **Manually**: `cmake -S src -B build && cmake --build build --config Release`.

The executable is written to `bin/`. It has no runtime dependencies (static
MSVC runtime, std::thread instead of OpenMP). Set **Server Executable** in
the preferences to use one from somewhere else; **Check Server** restarts
it and shows its version.

`bin/cubify_server --selftest` cubifies a generated sphere and prints
timings, without Blender.

## Using it

Identical to the Python add-on (see its README), with these changes:

- **Square Flat Regions** (off by default) plus a **Strength** slider,
  under Cubeness. See below.
- **Auto-fix Thin Walls** (off by default) and a **Fix Now** button, under
  Cubify Mesh. See below.
- **Repeat** (default 1), under ADMM Iterations: Cubify Mesh runs this
  many passes, each starting from the previous result — exactly the same
  as pressing Cubify again, so the style compounds. No existing parameter
  is equivalent: more Iterations barely changes a pass (teapot axis-aligned
  area 0.79 → 0.80 at 60 iterations), and Cubeness is the closest but not
  the same (teapot: 2 passes ≈ Cubeness 0.4, both 0.83; bunny: 2 passes
  0.40, more than Cubeness 0.6 at 0.38). Pins stay put across passes, Fix
  Thin Walls measures against the shape before the first pass, and time
  scales with the count. It applies to Cubify Mesh, not the bakes.
- **Custom** target, **Show Target**, **Auto Orient** and **Live Preview**
  (C++ add-on only). See below.
- **Threads** replaces **Device**: CPU threads for the solver, 0 = all
  cores. There is no GPU backend and no PyTorch install. For reference,
  the Python README's 163,842-vertex benchmark (10 iterations) takes 2.3 s
  here, against 4.0 s for its Metal GPU run (different machines).

Everything else — Target Shape (Cube, Octahedron, Pyramid, Hex Column,
Rounded Cube with Roundness), Keep Orientation, Cubeness, Cube Orientation,
Iterations / ADMM
Iterations, Apply to Copy, Style as Process (Cubeness Ramp / Iterations,
Steps, Iterations / Step, Frame Step), Cubify Every Frame (to Copy) with
its Cube Axes option (Object / World), Set /
Add / Clear Pins, Stylized Drag, Drag Iterations, Start Manipulation — works
the same way.

## Custom targets

Choose **Target Shape → Custom** to stylize toward your own set of
directions: every direction is one facet orientation surfaces will snap to.
The set describes a solid (the polytope whose faces point that way), and
the target term pulls the surface toward exactly those directions. The
list in the panel holds the directions; the ball under it edits the
active one. Ways to fill it:

- **Directions from Reference** — pick any mesh as **Reference**. A
  low-poly shape (a 7-sided prism, a hand-modeled gem) gives exactly its
  face directions; a detailed one gives its dominant planes, found by
  area-weighted clustering of its face normals (**Merge Angle**: normals
  closer than this become one facet; **Min Facet Area**: smaller facets
  are dropped; at most 32 from one reference). The Reference's rotation
  relative to the stylized object orients the style. Picking the object
  itself exaggerates its own planes.
- **Add Facet from Selection** — in Edit Mode, select a region and add its
  area-weighted average direction: that region becomes one flat facet
  facing the way it already roughly faces.
- **Start from Preset** — copy a preset's directions to edit from.
- **+ / − / ×** — add, remove, clear by hand.

Directions are stored in the target frame, so **Cube Orientation** still
rotates the whole set. A set must enclose a closed shape (something has to
face every side; Minkowski's condition) — the panel says when it does not,
and **Close Shape** adds the fewest directions that fix it (a pyramid's
four sides get exactly a flat base). Up to 64 directions.

**Show Target** (the eye next to Target Shape) draws the target solid as an
orange wireframe over the active object, oriented by Cube Orientation, for
every target.

## Auto Orient

The button under Cube Orientation sets it to the rotation under which the
active mesh, as it is, already fits the target best (an area-weighted fit
of its face normals: global search plus refinement). Stylizing then
reshapes the mesh least and does not turn it — e.g. a box rotated inside
its mesh data gets the target lined up with its faces (to ~0.03°), and
Cubify moves its vertices ~20× less. Among equally good fits it keeps the
one closest to the current orientation, so a symmetric target never flips.
It pairs well with Keep Orientation.

Note that this solver, like the paper, uses rotated *vertex* normals: an
8-vertex box has only diagonal corner normals, so even a perfectly
oriented one gets reshaped; subdivided flat faces behave as expected.

## Live Preview

Next to Cubify Mesh. While it runs, the active mesh re-cubifies whenever
any target setting changes in the panel — target, custom directions,
Roundness, Cube Orientation, Cubeness, iterations, Keep Orientation,
Square Flat Regions — and the status bar shows progress. **Enter** applies,
**Esc** restores the mesh. Viewport navigation and the panel keep working;
undo is held off until the preview ends.

Every change restarts from the original shape on the same server session
(no refactorization: changing the style never touches the system matrix),
solving in slices of ~25 ms per frame. So the result never depends on the
editing history, and Enter applies exactly what Cubify Mesh would give with
the final settings (bit-identical; Auto-fix Thin Walls and Apply to Copy
are honored, Repeat is not).

## Square Flat Regions

Cubic stylization rewards surface *normals* that face a cube axis; it never
looks at outlines. A flat, wide part such as a pot lid or a plate already
faces +Z almost everywhere, so its round outline costs nothing and stays
round, while the tall sides of the pot around it straighten into a box.
Raising Cubeness does not fix this: the part crumples before it squares.

With **Square Flat Regions** on, surfaces that already face a cube axis
(within about 8°, re-checked every iteration) become nearly free to
stretch in their own plane. Their rims can then pull the outline square.
On the Utah teapot the lid's top-view outline goes from round (squareness
0.07 on a 0 = circle, 1 = square scale) to 0.42, and the rim it sits in
from 0.32 to 0.61; the bunny looks practically unchanged.

- **Strength** scales how freely flat regions may stretch; 1 (default)
  scales their ARAP weights down to 1%, 0 has no effect.
- The first iteration of each solve setup runs without it, so regions
  that are flat only in the rest pose (a knob's tip) are not collapsed.
- It refactorizes the system every iteration: Cubify is about 3× slower
  (teapot, 17.8k vertices: 0.19 s → 0.62 s). It is not used while
  dragging pins.
- This goes beyond the paper; with it off, results match the paper's
  energy.

## Fix Thin Walls

Solid-shell meshes have walls with an inner and an outer surface (a
teapot's spout, the rim of a pot). Nothing in the stylization energy ties
the two sides together, so cubifying can push one through the other — the
inside shows through as backfaces. Parts that touch (a lid on its rim) can
likewise sink into each other.

**Fix Now** pushes them apart again. In the shape from before the last
Cubify, every vertex is paired with the opposite-facing surface behind it
(wall thickness) and in front of it (narrow gaps), up to 4% of the
bounding-box diagonal away. Wherever a pair got closer than **Min
Thickness** (default 10%) of its original distance, or crossed, both sides
are pushed apart along the normal with a smoothed correction, repeating
until none is left.

- Cubify stores the pre-cubify shape on the mesh (hidden attribute
  `.cubify_rest`, saved with the .blend), so the button works any time
  after a Cubify by this add-on; without it, the object is skipped.
  Running it twice is harmless.
- **Auto-fix Thin Walls** runs it after every Cubify, and on every step of
  both animation bakes (the solver itself keeps warm-starting from its
  unfixed result).
- **Min Thickness** and **Iterations** are in the button's redo panel.
- On the Utah teapot (λ 0.2): crossing triangle pairs 663 → 30 (the
  original mesh has 3), about 2.6k of 17.8k vertices move (at most 0.14),
  the cubic look is kept (axis-aligned area 0.79 → 0.78), in 0.05 s.

## Architecture

```
Blender (Python)                          cubify_server (C++)
__init__.py  operators, UI                server.cpp        request loop, sessions
client.py    process + protocol  ──pipe──▶ cubic_stylizer.*  solver (Eigen)
gauss.py     directions, outline,         target_shape.h    target terms, projection
             clustering, Auto Orient      thin_walls.*      Fix Thin Walls
builder.py   CMake build                  thread_pool.h     parallel_for
```

- One server process per Blender session, started on first use and
  stopped when the add-on is disabled (or when Blender exits, because
  stdin closes).
- Binary protocol over stdin/stdout, documented in `src/protocol.h`.
  Each mesh gets a **session** on the server holding its prefactorized
  system and ADMM state. Start Manipulation factorizes once, and each
  mouse move sends only the pin targets; the server warm-starts from its
  own copy of the previous result. Live Preview restyles a session in
  place (`SET_STYLE`), which restarts its state so the next solve equals a
  fresh session's.
- Solve progress is streamed back and drives Blender's progress bar.

Solver differences from the numpy version (none change results on
normal meshes):

- The global step uses a sparse Cholesky (`SimplicialLDLT`) of the
  pinned Laplacian instead of LU / conjugate gradient. The pinned
  system is symmetric positive definite, so Cholesky applies.
- Loose parts of one object keep their relative placement. The energy
  does not couple separate parts, so each part without pins is placed
  by translation after every iteration (its shape is never changed):
  - a part that touches already-placed geometry (closer than 0.5% of the
    bounding-box diagonal) keeps the average offset at its contact
    points, so a lid stays seated on the pot's rim even though both
    change shape;
  - otherwise the largest remaining part keeps its rest-pose centroid;
  - parts with pins are held by the pins, and loose vertices stay put.

  The numpy solver fixes only vertex 0, so its unpinned parts drift
  relative to each other. A bake of an unpinned mesh likewise keeps its
  centroid fixed here, rather than vertex 0.
- The local step runs each vertex's ADMM to its own convergence on a
  thread pool, rather than batched numpy over the still-active vertices.
  The arithmetic is the same.
- Style-as-process bakes solve all steps before writing shape keys, so a
  failed solve leaves the mesh untouched.

## Tests

```sh
python tests/test_against_python.py      # C++ vs the numpy solver, timings
python tests/test_parts_and_flat.py      # teapot: lid seating, Square Flat Regions
python tests/test_thin_walls.py          # teapot: thin-wall fix
python tests/test_custom_targets.py      # custom targets, SET_STYLE, clustering, Auto Orient
blender -b --factory-startup --python tests/blender_smoke_test.py
blender -b --factory-startup --python tests/blender_custom_test.py
```

`blender_custom_test.py` covers the authoring tools, Auto Orient, the
target outline and the live preview's solving path (compared with Cubify
Mesh); the live preview's modal loop itself needs a window.

The Blender smoke test runs every operator except the modal drag, which
needs a viewport. It also checks that topology and UVs are preserved and
that both add-ons coexist, and runs the teapot from `../test_objs` with
Square Flat Regions off and on, and with Fix Thin Walls (counting crossing
triangles). It passes on Blender 3.6 and 5.0.
