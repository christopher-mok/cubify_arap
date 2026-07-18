# Cubic Stylization + ARAP — Blender Add-on

A Blender add-on with two tools built on the same solver:

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
2. Pick `cubic_stylization.zip` (or zip the `cubic_stylization/` folder yourself).
3. Enable **Mesh: Cubic Stylization** in the add-on list.

Requires Blender 2.93+ and numpy (bundled with Blender). If scipy is present
in Blender's Python it is used for a faster prefactorized sparse solve;
otherwise a dependency-free conjugate-gradient fallback is used automatically —
results are identical.

## Device (CPU / CUDA / Metal)

The **Device** dropdown selects where the solver runs:

- **Auto** (default) — GPU for meshes ≥ 20k vertices when one is available,
  CPU otherwise.
- **CPU** — the numpy solver, always available.
- **CUDA GPU** — NVIDIA GPUs, via PyTorch.
- **Metal GPU (MPS)** — Apple GPUs, via PyTorch.

GPU devices require PyTorch inside **Blender's own Python**. The easy way:
**Edit → Preferences → Add-ons → Cubic Stylization → Install PyTorch** — a
one-click background install into Blender's Python (macOS ~250 MB; Windows
~3 GB, CUDA build). **Check PyTorch** on the same panel reports what's
currently available. Manual alternative from a terminal:

```sh
# macOS (Apple GPU / MPS):
/Applications/Blender.app/Contents/Resources/<ver>/python/bin/python3.* -m pip install torch

# Linux (NVIDIA / CUDA — the default wheel bundles CUDA):
<blender>/python/bin/python -m pip install torch

# Windows (NVIDIA / CUDA): use the CUDA wheel index from pytorch.org, e.g.
<blender>\python\bin\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

If PyTorch or the requested device is missing, the add-on falls back to the
CPU solver and reports a warning — nothing breaks. The GPU backend runs the
whole local step (rotation fitting + ADMM) and right-hand-side assembly on
the device in float32; rotations are fitted with a batched Newton
polar-decomposition iteration (reflection cases handled exactly on-GPU via a
closed-form symmetric 3x3 eigensolve), so no SVD kernels are needed. Results
match the CPU solver to float32 precision.

Measured on an Apple M2 Max (cubify, 10 iterations, λ=0.4):

| vertices | CPU (numpy) | Metal (MPS) | speedup |
|---------:|------------:|------------:|--------:|
|   10,242 |       0.8 s |       1.5 s |    0.5× |
|   40,962 |       3.5 s |       2.4 s |    1.4× |
|  163,842 |      10.6 s |       4.0 s |    2.6× |

GPU pays off for large meshes; for small ones the CPU path wins (which is
what Auto does). CUDA uses the identical torch code path but could not be
benchmarked on this machine.

## Cubify

1. 3D Viewport sidebar (**N**) → **Cubify** tab.
2. Select one or more mesh objects (Object Mode) and adjust:
   - **Cubeness** (λ) — strength of the cube prior. `0` is plain ARAP;
     `0.2` gives a rounded-cube look; `1.0+` gives sharp cubes.
   - **Cube Orientation** — Euler rotation of the target cube axes (object
     space), for cubifying against a tilted frame.
   - **Iterations** / **ADMM Iterations** — outer and inner solver budgets
     (defaults are fine).
   - **Apply to Copy** — keep the original; cubify a `<name>_cubified`
     duplicate.
3. Click **Cubify Mesh** (undo-supported). Pinned vertices, if any, are held
   in place during stylization.

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

## Notes

- Operates on local-space vertex data; object transforms and modifiers are not
  baked. Meshes with shape keys are skipped.
- Works best on connected, manifold meshes; boundaries are handled (one-sided
  cotangent weights), and loose vertices are kept fixed.
- Interactive dragging factorizes the system once at start; each mouse move
  only re-solves, so meshes in the 1k–50k vertex range stay interactive.

## Method

Minimizes `Σ_ij (w_ij/2)‖R_i d_ij − d′_ij‖² + Σ_i λ a_i ‖Aᵀ R_i n̂_i‖₁` by
local-global iteration: the global step is a cotan-Laplacian solve with pinned
vertices eliminated into the right-hand side, and the local rotation step is
the paper's per-vertex ADMM (soft-thresholding + orthogonal Procrustes,
penalty updates μ=10, τ=2, ρ₀=1e-4), batched over all vertices with numpy and
warm-started across iterations. With λ = 0 the local step reduces to plain
Procrustes and the method is exactly ARAP (Sorkine & Alexa 2007).
