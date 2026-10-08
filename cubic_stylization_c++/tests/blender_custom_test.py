# Headless test of the custom-target tools (reference object, facets from a
# selection, direction editing, Auto Orient, target drawing data, panel) and
# of the live preview's solving path:
#
#   blender -b --factory-startup --python-exit-code 1 --python tests/blender_custom_test.py
#
# The live preview's modal loop needs a window; here its restyle/step logic
# is driven directly and compared with Cubify Mesh.

import math
import os
import sys
import traceback

import addon_utils
import bmesh
import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MODULE = "cubic_stylization_c++"
sys.path.insert(0, REPO)

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f" ({detail})" if detail else ""))
    if not cond:
        FAILED.append(name)


def coords(me):
    a = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def facing(me, D, deg=10.0):
    """Share of face area whose normal is within deg of a direction in D."""
    me.update()
    n = np.empty(len(me.polygons) * 3)
    me.polygons.foreach_get("normal", n)
    a = np.empty(len(me.polygons))
    me.polygons.foreach_get("area", a)
    hit = (n.reshape(-1, 3) @ np.asarray(D).T).max(axis=1) > math.cos(math.radians(deg))
    return a[hit].sum() / a.sum()


def activate(ob):
    bpy.ops.object.mode_set(mode='OBJECT') if bpy.context.mode != 'OBJECT' else None
    bpy.ops.object.select_all(action='DESELECT')
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob


def sphere(name, loc=(0, 0, 0)):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, location=loc)
    ob = bpy.context.active_object
    ob.name = name
    return ob


def main():
    addon_utils.enable(MODULE, default_set=True, handle_error=lambda e: traceback.print_exc())
    addon = sys.modules[MODULE]
    scene = bpy.context.scene
    props = scene.cubify_cpp_settings
    props.cubeness, props.iterations = 1.0, 30
    gauss = addon.gauss

    # ---- reference object: a 7-sided prism, rotated 20 deg relative to the target
    bpy.ops.mesh.primitive_cylinder_add(vertices=7, location=(5, 0, 0), rotation=(0, 0, math.radians(20)))
    ref = bpy.context.active_object
    target = sphere("target")
    props.reference_object = ref
    activate(target)
    check("Directions from Reference runs", bpy.ops.cubify_cpp.dirs_from_object() == {'FINISHED'})
    D = addon.target_directions(props)
    n = np.empty(len(ref.data.polygons) * 3)
    ref.data.polygons.foreach_get("normal", n)
    rot = np.array(ref.matrix_world.to_3x3())
    expect = n.reshape(-1, 3) @ rot.T  # the prism's faces as seen from the (unrotated) target
    miss = max(np.min(np.linalg.norm(expect - d, axis=1)) for d in D)
    check("7-sided prism -> 9 directions, turned with the reference",
          props.target_shape == 'CUSTOM' and len(D) == 9 and miss < 1e-6, f"{len(D)} dirs, off by {miss:.1e}")
    before = facing(target.data, D)
    check("Cubify Mesh with the custom target", bpy.ops.object.cubify_cpp_mesh() == {'FINISHED'})
    after = facing(target.data, D)
    check("surface now faces the prism's directions", after > before + 0.4, f"{before:.2f} -> {after:.2f}")

    # ---- open sets: a pyramid without its base, closed by the Close Shape button
    props.custom_dirs.clear()
    for d in gauss.preset_directions("PYRAMID")[:4]:
        props.custom_dirs.add().direction = tuple(d)
    check("open set detected", gauss.open_direction(addon.target_directions(props)) is not None)
    s2 = sphere("open_set")
    activate(s2)
    V0 = coords(s2.data)
    try:
        r = bpy.ops.object.cubify_cpp_mesh()
    except RuntimeError as exc:  # headless Blender raises on an error report
        r = {'CANCELLED'}
        print("       ", exc)
    check("Cubify refuses an open set and leaves the mesh alone",
          r == {'CANCELLED'} and np.array_equal(coords(s2.data), V0))
    bpy.ops.cubify_cpp.dirs_edit(action='CLOSE')
    D = addon.target_directions(props)
    check("Close Shape adds the base", len(D) == 5 and np.allclose(D[-1], [0, 0, -1], atol=1e-6))

    # ---- edit buttons and presets
    bpy.ops.cubify_cpp.dirs_preset(preset='HEX_COLUMN')
    check("Start from Preset", len(props.custom_dirs) == 8)
    bpy.ops.cubify_cpp.dirs_edit(action='ADD')
    bpy.ops.cubify_cpp.dirs_edit(action='REMOVE')
    check("Add / Remove", len(props.custom_dirs) == 8)
    bpy.ops.cubify_cpp.dirs_edit(action='CLEAR')
    check("Clear", len(props.custom_dirs) == 0)

    # ---- facet from selection: the top faces of a sphere tilted 30 deg
    s3 = sphere("pick")
    activate(s3)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(s3.data)
    tilt = np.array([math.sin(math.radians(30)), 0, math.cos(math.radians(30))])
    for f in bm.faces:
        f.select = np.dot(np.array(f.normal), tilt) > 0.97
    bmesh.update_edit_mesh(s3.data)
    m = sum((np.array(f.normal) * f.calc_area() for f in bm.faces if f.select), np.zeros(3))
    m /= np.linalg.norm(m)
    check("Add Facet from Selection runs", bpy.ops.cubify_cpp.dirs_add_selection() == {'FINISHED'})
    d = np.array(props.custom_dirs[-1].direction)
    off = np.degrees(np.arccos(np.clip(d @ m, -1, 1)))
    check("picked direction is the selection's area-weighted normal", off < 0.02, f"{off:.1e} deg off")
    bpy.ops.object.mode_set(mode='OBJECT')

    # ---- Auto Orient: a box rotated in its mesh data
    # subdivided, so its faces are flat regions: this solver uses vertex
    # normals, and a bare 8-vertex box has only diagonal corner normals
    bpy.ops.mesh.primitive_cube_add(location=(0, 5, 0))
    box = bpy.context.active_object
    box.scale = (1.5, 1.0, 0.7)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.subdivide(number_cuts=8)
    bpy.ops.object.mode_set(mode='OBJECT')
    R = np.array(bpy.app.driver_namespace.get("_", None) or
                 __import__("mathutils").Euler((0.3, -0.5, 0.9)).to_matrix())
    box.data.transform(__import__("mathutils").Matrix(R.tolist()).to_4x4())
    props.target_shape, props.orientation = 'CUBE', (0, 0, 0)
    activate(box)
    V0 = coords(box.data)
    bpy.ops.object.cubify_cpp_mesh()
    moved_plain = np.linalg.norm(coords(box.data) - V0, axis=1).mean()
    box.data.vertices.foreach_set("co", V0.ravel())
    box.data.update()
    check("Auto Orient runs", bpy.ops.cubify_cpp.auto_orient() == {'FINISHED'})
    A = addon.cube_axes(props)
    n = np.empty(len(box.data.polygons) * 3)
    box.data.polygons.foreach_get("normal", n)
    miss = np.degrees(np.arccos(np.clip(np.abs(n.reshape(-1, 3) @ A).max(axis=1), -1, 1))).max()
    check("Auto Orient lines the target up with the box's faces", miss < 0.05, f"worst face {miss:.3f} deg")
    bpy.ops.object.cubify_cpp_mesh()
    moved_oriented = np.linalg.norm(coords(box.data) - V0, axis=1).mean()
    check("so Cubify reshapes the box far less", moved_oriented < 0.25 * moved_plain,
          f"mean vertex move {moved_plain:.3f} -> {moved_oriented:.4f}")

    # ---- target drawing data for every target (and none for an open set)
    for t in ('CUBE', 'OCTAHEDRON', 'PYRAMID', 'HEX_COLUMN', 'ROUNDED_CUBE'):
        props.target_shape = t
        geom = addon._target_geometry(props)
        nv = {'CUBE': 8, 'OCTAHEDRON': 6, 'PYRAMID': 5, 'HEX_COLUMN': 12, 'ROUNDED_CUBE': 8}[t]
        check(f"target outline for {t}", geom is not None and len(geom[0]) == nv)
    props.target_shape = 'CUSTOM'
    props.custom_dirs.clear()
    check("no outline for an empty custom set", addon._target_geometry(props) is None)

    # ---- panel draws in every state
    class Layout:
        def __getattr__(self, name):
            return lambda *a, **k: self
    for t in ('CUBE', 'ROUNDED_CUBE', 'CUSTOM'):
        props.target_shape = t
        for fill in (False, True):
            if fill:
                bpy.ops.cubify_cpp.dirs_preset(preset='OCTAHEDRON')
                props.target_shape = t
            fake = type("P", (), {"_draw_custom": staticmethod(addon.VIEW3D_PT_cubify_cpp._draw_custom)})()
            fake.layout = Layout()
            try:
                addon.VIEW3D_PT_cubify_cpp.draw(fake, bpy.context)
                ok = True
            except Exception:
                traceback.print_exc()
                ok = False
            check(f"panel draws ({t}, {'with' if fill else 'no'} directions)", ok)

    # ---- live preview solving path == Cubify Mesh, through restyles
    op_cls = addon.OBJECT_OT_cubify_cpp_live_preview
    s4 = sphere("live")
    activate(s4)
    V0 = coords(s4.data)
    props.target_shape, props.cubeness, props.keep_orientation = 'CUBE', 0.6, False
    op = type("Op", (), {})()
    for name in ("_signature", "_restyle", "_step"):
        setattr(op, name, getattr(op_cls, name).__get__(op))
    me = s4.data
    V, F = addon.read_mesh_arrays(me)
    op.stylizer, op.device, _ = addon.create_stylizer(bpy.context, V, F, props.cubeness,
                                                     addon.cube_axes(props), [])
    op.ob, op.V_cur, op.done, op.t_iter = s4, V.copy(), 0, None
    op._step(props, 1e-4)  # a slice, as one timer tick would
    partial = op.done
    # the user edits: octahedron, tilted, stronger
    bpy.ops.cubify_cpp.dirs_preset(preset='OCTAHEDRON')
    props.orientation = (0.2, 0.0, 0.4)
    props.cubeness = 0.9
    op._restyle(props)
    while op.done < props.iterations:
        op._step(props, 1e-3)  # many small ticks
    live = coords(me)
    op.stylizer.close()
    me.vertices.foreach_set("co", V0.ravel())
    me.update()
    bpy.ops.object.cubify_cpp_mesh()
    check("live preview after restyles == Cubify Mesh (bit-identical)",
          0 < partial < props.iterations and np.array_equal(live, coords(me)),
          f"first tick ran {partial} of {props.iterations} iterations; "
          f"max diff {np.abs(live - coords(me)).max():.1e}")


try:
    main()
except Exception:
    traceback.print_exc()
    FAILED.append("exception")
print("ALL-OK" if not FAILED else f"FAILED: {FAILED}")
if FAILED:
    sys.exit(1)
