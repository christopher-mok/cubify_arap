# Cubic Stylization — Distribution

This branch provides a simplified Blender add-on backed by the C++/Eigen
solver. The full interface and development documentation are on `main`.

## Install

In Blender, open **Edit → Preferences → Add-ons → Install…**, select
`cubic_stylization_c++.zip`, and enable **Mesh: Cubic Stylization (C++)**.
Open the 3D Viewport sidebar (**N**) → **Cubify C++** tab.
The bundled server starts automatically on first use. Server checks and
building remain available in add-on preferences when needed.

## Controls

- **Cubeness** — strength of the selected style.
- **Roundness** — shown only for Rounded Cube; higher values soften edges.
- **Shape Type** — Cube by default; also Octahedron, Pyramid, Hex Column,
  and Rounded Cube.
- **Repeat** — number of consecutive passes when applying Cubify Mesh.
- **Square Flat Regions** — allow flat outlines to become more square.
- **Apply to Copy** — apply Cubify Mesh to a duplicate.
- **Cubify Mesh** — apply the selected style to the selected meshes.
- **Bake Animation** — bake the stylization process into shape keys.

Hover over a control for a short description. Advanced settings, pin tools,
thin-wall tools, and per-frame baking are omitted from the distribution UI.

The bake uses the built-in defaults: Cubeness Ramp, 20 steps, 3 solver
iterations per step, and 2 frames between steps, starting at the current
frame. It adds shape keys to the selected meshes; meshes with existing
shape keys are skipped. Apply to Copy and Repeat affect Cubify Mesh only.
Hidden settings retain their stored values in existing Blender scenes.
