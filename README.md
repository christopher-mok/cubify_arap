# Cubic Stylization + ARAP — Blender Add-on (C++ server)

The primary add-on is `cubic_stylization_c++/`: Blender handles the UI and mesh I/O, while a native C++/Eigen server runs the solver over stdin/stdout. It provides two tools:

- **Cubic Stylization** (Hsueh-Ti Derek Liu & Alec Jacobson, SIGGRAPH Asia
  2019): an as-rigid-as-possible deformation with an L1 penalty on rotated
  vertex normals that drives the surface toward axis-aligned cube faces while
  preserving local detail.
- **Interactive ARAP manipulation**: pin vertices as handles, then grab and
  drag a pin in the viewport — the mesh follows as-rigidly-as-possible in
  real time.

Both tools work on **triangle and quad meshes** (and n-gons). Quad/n-gon
meshes are solved on an internal *virtual triangulation*: the real mesh
topology is never modified, so **UV maps and all loop data are preserved
exactly**. The panel shows whether the active mesh is a triangle, quad, or
mixed mesh.

## Install

1. In Blender: **Edit → Preferences → Add-ons → Install…**
2. Pick `cubic_stylization_c++.zip` and enable **Mesh: Cubic Stylization (C++)**.
3. Open the 3D Viewport sidebar (**N**) → **Cubify C++** tab.

The C++ server starts automatically when needed. Use **Threads** to choose
CPU parallelism (`0` uses all cores). PyTorch is not required. If a server
binary for your platform is unavailable, use **Build Server** in the add-on
preferences. See the [C++ add-on guide](cubic_stylization_c++/README.md) for
build requirements, platform packaging, and C++-specific controls such as
Repeat, Square Flat Regions, and Fix Thin Walls.

The Python add-on in `cubic_stylization/` remains available as a reference
implementation and optional alternative. Install `cubic_stylization.zip`
and enable **Mesh: Cubic Stylization** to use its **Cubify** tab. Its
**Device** selector supports CPU and optional PyTorch CUDA/Metal backends;
non-Cube targets use its CPU solver.

## Cubify

1. 3D Viewport sidebar (**N**) → **Cubify C++** tab.
2. Select one or more mesh objects (Object Mode) and adjust:
   - **Target Shape** — what surfaces are stylized toward (see below).
   - **Cubeness** (λ) — strength of the stylization. `0` is plain ARAP;
     `0.2` gives a soft look; `1.0+` gives a sharp target shape.
   - **Cube Orientation** — Euler rotation of the target shape's axes
     (object space), for stylizing against a tilted frame (and for pointing
     the pyramid's apex or the hex column's axis, both along +Z).
   - **Iterations** / **ADMM Iterations** — outer and inner solver budgets
     (defaults are fine).
   - **Apply to Copy** — keep the original; cubify a `<name>_cubified`
     duplicate.
3. Click **Cubify Mesh** (undo-supported). Pinned vertices, if any, are held
   in place during stylization.

### Target Shapes

| Target | Surfaces face… | Look |
|---|---|---|
| **Cube** | the 6 axis directions | the paper's cubic stylization |
| **Octahedron** | the 8 diagonal directions | crystal / gem facets |
| **Pyramid** | 4 sides sloping ~52° plus a flat base (apex +Z) | monumental, ancient |
| **Hex Column** | 6 vertical sides plus top and bottom (axis Z) | basalt columns |
| **Rounded Cube** | near the axes, without snapping | soft, pillowy cube |

**Roundness** (Rounded Cube only): `0` is almost a sharp cube, `1` a soft
pillow close to a sphere; `0.5` gives clearly rounded edges.

All targets keep the rest of the tool intact: pins, Style as Process,
Cubify Every Frame and Stylized Drag use the selected target.

**Keep Orientation** — the energy leaves a mesh's overall rotation free, so
without pins the target term can turn the whole mesh to line its large flat
areas up with the target's faces (Suzanne's head nods 10–14° about its
ear-to-ear axis under Cube, Octahedron or Hex Column, 4–5° under the
others). With Keep Orientation on, the
mesh's area-weighted best-fit rotation from the rest pose is undone after
every global step: that costs no ARAP energy, so the target shape has to
come from reshaping instead (it forms about as strongly as when the mesh
may turn). Off by default, so existing results don't change; it has no
effect with pins, which already hold the orientation. In Cubify Every Frame
it makes World mode follow the animated heading exactly.
The C++ server supports all target shapes on CPU. The optional Python GPU backend implements Cube only.

## Style as Process (animation bake)

Bake the cubification *process* itself as an animation: **Style as Process →
Bake Animation (Shape Keys)**. One absolute shape key is stored per step and
played across the timeline via a keyframed **Evaluation Time** (two linear
keyframes starting at the current frame) — scrub, retime, or ease it like any
f-curve. The base mesh, topology and UVs are untouched; delete the shape keys
to recover the original. Pinned vertices are held throughout.

- **Animate By**:
  - **Cubeness Ramp** (default) — λ ramps from 0 to the current Cubeness with
    a few warm-started iterations per step (**Iterations / Step**). Evenly
    paced; reads as a crystallization spreading at constant speed. Raise
    Iterations / Step if the final frames look unconverged.
  - **Iterations** — one solver iteration per step at full Cubeness: the raw
    convergence, where flat regions snap into place early and creases sharpen
    late (most of the change happens in the first frames).
- **Steps** — number of baked shape keys; **Frame Step** — timeline frames
  between them.

Meshes that already have shape keys are skipped (the bake would fight them).

### Cubify Every Frame (to Copy)

For an *animated* mesh (object transforms, shape keys, armatures, deforming
modifiers), **Cubify Every Frame (to Copy)** re-cubifies the evaluated
world-space mesh at every sampled frame of the scene range (**Frame Step**
sets the sampling) and bakes the results onto a **new** object named
`<name>_cubified_anim` — one absolute shape key per sampled frame, played
back via keyframed Evaluation Time. The original object and its animation
are untouched.

**Cube Axes** sets what the cube axes are attached to:

- **Object** (default) — the axes turn with the object (its world rotation
  times **Cube Orientation**), so a rotating object bakes as a rotating
  cubified shape, and deformations are re-cubified against the object's
  own frame.
- **World** — the axes stay fixed in the world, so a rotating object
  re-crystallizes against the world axes as it turns: the shape turns with
  the animation while its flat faces keep facing the same world directions.
  Like Cubify Mesh on a posed object, the cube term also pulls the shape
  partway toward the nearest axis-aligned heading.

Every sampled frame is an independent cubification of that frame's pose
(exactly what Cubify Mesh would give on it), with the same Iterations
budget. Frames are deliberately not warm-started from each other: ARAP
leaves the output's global rotation free, so starting from the previous
result lets the cube term turn the shape back to its earlier orientation
and the bake stops following the animation. Pins are honored when no
modifier changes the vertex count; animated topology-changing modifiers are
not supported. The copy sits at the world origin (identity transform) since
world space is baked into its keys.

## ARAP Manipulation

1. In Edit Mode, select the vertices you want as handles/anchors, then in the
   panel's **ARAP Manipulation** box click **Set Pins** (or **Add** to grow the
   set, **Clear** to remove). Pins are stored in a `CubifyPins` vertex group,
   so you can also edit them like any vertex group.
2. Back in Object Mode, click **Start Manipulation**. Pins are drawn as red
   points.
3. Left-click near a pin to grab it (turns green) and drag — the rest of the
   mesh deforms in the ARAP way while all other pins stay fixed. Orbit/zoom
   navigation still works while the tool is active.
4. **Enter/Space** confirms, **Esc/right-click** cancels and restores the mesh.

Options:
- **Stylized Drag** — use the current Cubeness during dragging so a cubified
  mesh keeps its cubic look while you deform it (off = classic ARAP).
- **Drag Iterations** — solver iterations per mouse move; raise it if the mesh
  visibly lags behind large drags.

The pose at the moment you press **Start Manipulation** is used as the ARAP
rest pose.

## Tests

Run headless from the repository root (both add-ons, Python and C++ solvers):

```sh
blender --background --python-exit-code 1 --python tests/test_targets.py
blender --background --python-exit-code 1 --python tests/test_style_bake.py
```

The C++ add-on has its own suite in `cubic_stylization_c++/tests/`.

## Notes

- Operates on local-space vertex data; object transforms and modifiers are not
  baked. Meshes with shape keys are skipped.
- Works best on connected, manifold meshes; boundaries are handled (one-sided
  cotangent weights), and loose vertices are kept fixed.
- Interactive dragging factorizes the system once at start; each mouse move
  only re-solves, so meshes in the 1k–50k vertex range stay interactive.

## Method

Minimizes `Σ_ij (w_ij/2)‖R_i d_ij − d′_ij‖² + Σ_i λ a_i f(Aᵀ R_i n̂_i)` by
local-global iteration: the global step is a cotan-Laplacian solve with pinned
vertices eliminated into the right-hand side, and the local rotation step is
the paper's per-vertex ADMM (soft-thresholding + orthogonal Procrustes,
penalty updates μ=10, τ=2, ρ₀=1e-4), batched over all vertices with numpy and
warm-started across iterations. With λ = 0 the local step reduces to plain
Procrustes and the method is exactly ARAP (Sorkine & Alexa 2007).

`f` is the target shape's term. For the cube it is the paper's L1 norm. The
other faceted targets generalize it: `f` is the support function of the
polytope `P = {y : d_k·y ≤ 1}` whose face normals `d_k` are the preferred
directions. On the unit sphere it is smallest (exactly 1) at the `d_k`, so
surfaces snap to face them; the cube's `P` is `[−1, 1]³`, whose support
function is L1. The ADMM's soft-threshold becomes the proximal step
`x − t·proj_P(x/t)` (Moreau's identity), with the projection computed
exactly from `P`'s faces and edges. The rounded cube uses
`f(y) = Σ |y_c|^p` with `p = 2 − 0.5·10^(−roundness)` ∈ [1.5, 1.95], whose
proximal step is a monotone Newton solve.
