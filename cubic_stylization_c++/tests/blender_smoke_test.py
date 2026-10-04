# Headless smoke test of the add-on's operators inside Blender:
#
#   blender -b --factory-startup --python tests/blender_smoke_test.py
#
# Enables the add-on from this repository (plus the Python add-on, to check
# that both coexist), then runs cubify, pins, both animation bakes and the
# solver path of the ARAP manipulator on generated meshes. The modal drag
# itself needs a live viewport and is not exercised here.

import os
import sys
import traceback

import addon_utils
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


def aligned_fraction(me):
    """Share of face area whose normal points along a local axis."""
    me.update()
    tot = al = 0.0
    for p in me.polygons:
        n = np.abs(np.array(p.normal))
        tot += p.area
        if n.max() > 0.98:
            al += p.area
    return al / tot


def eval_time_fcurves(key):
    ad = key.animation_data
    fcs = getattr(ad.action, "fcurves", None)
    if fcs is None:
        from bpy_extras import anim_utils
        fcs = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot).fcurves
    return [fc for fc in fcs if fc.data_path == "eval_time"]


def new_sphere(name, quads):
    if quads:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
    else:
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4)
    ob = bpy.context.active_object
    ob.name = name
    return ob


def select_only(ob):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob


def main():
    print("Blender", bpy.app.version_string)
    for mod in ("cubic_stylization", MODULE):
        addon_utils.enable(mod, default_set=True, handle_error=lambda e: traceback.print_exc())
    check("add-on enabled", hasattr(bpy.types.Scene, "cubify_cpp_settings"))
    check("python add-on coexists", hasattr(bpy.types.Scene, "cubify_settings"))
    addon = sys.modules[MODULE]

    # clear the default scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    scene = bpy.context.scene
    props = scene.cubify_cpp_settings
    props.cubeness = 0.5
    props.iterations = 20

    # ---- cubify: triangle mesh and quad mesh (UVs must survive)
    for quads in (False, True):
        ob = new_sphere("quad_sphere" if quads else "ico_sphere", quads)
        me = ob.data
        n_loops = len(me.loops)
        uv0 = np.empty(n_loops * 2)
        me.uv_layers.active.data.foreach_get("uv", uv0)
        before = aligned_fraction(me)
        select_only(ob)
        res = bpy.ops.object.cubify_cpp_mesh()
        after = aligned_fraction(me)
        uv1 = np.empty(n_loops * 2)
        me.uv_layers.active.data.foreach_get("uv", uv1)
        check(f"cubify {ob.name}", res == {'FINISHED'} and after > before + 0.3,
              f"axis-aligned area {before:.2f} -> {after:.2f}")
        check(f"{ob.name}: topology and UVs preserved",
              len(me.loops) == n_loops and np.array_equal(uv0, uv1))
        bpy.data.objects.remove(ob)

    # ---- loose parts of one object keep their relative placement
    a = new_sphere("parts", False)
    b = new_sphere("parts_small", True)
    b.scale = (0.4, 0.4, 0.4)
    b.location = (2.5, 0.0, 0.7)
    na = len(a.data.vertices)
    for o in (a, b):
        o.select_set(True)
    bpy.context.view_layer.objects.active = a
    bpy.ops.object.join()  # b's vertices are appended after a's (world space)
    V0 = coords(a.data).copy()
    select_only(a)
    res = bpy.ops.object.cubify_cpp_mesh()
    V1 = coords(a.data)
    drift = max(np.linalg.norm(V1[s].mean(0) - V0[s].mean(0))
                for s in (slice(0, na), slice(na, None)))
    check("loose parts keep their placement", res == {'FINISHED'} and drift < 1e-5,
          f"part centroid drift {drift:.1e}")
    bpy.data.objects.remove(a)

    # ---- teapot (body + separate lid): lid stays seated; Square Flat Regions
    check("Square Flat Regions off by default", not props.square_flat)
    bpy.ops.wm.obj_import(filepath=os.path.join(REPO, "test_objs", "utah_teapot.obj"),
                          forward_axis='Y', up_axis='Z')
    tea = bpy.context.selected_objects[0]
    V0 = coords(tea.data).copy()
    # the lid is the connected part that does not contain vertex 0
    E = np.empty(len(tea.data.edges) * 2, dtype=np.int64)
    tea.data.edges.foreach_get("vertices", E)
    parent = np.arange(len(V0))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in E.reshape(-1, 2):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(len(V0))])
    lid = roots != roots[0]
    lid_i, body_i = np.flatnonzero(lid), np.flatnonzero(~lid)
    near = np.array([body_i[np.argmin(np.linalg.norm(V0[body_i] - V0[i], axis=1))]
                     for i in lid_i[::8]])
    pairs = (lid_i[::8], near)
    keep = np.linalg.norm(V0[pairs[0]] - V0[pairs[1]], axis=1) < 0.04
    li, bi = pairs[0][keep], pairs[1][keep]

    def lid_square(V):
        P = V[lid, :2] - V[lid, :2].mean(0)
        dg = np.abs(P @ (np.array([[1, 1], [1, -1]]).T / np.sqrt(2))).max(0).mean()
        return (dg / np.abs(P).max(0).mean() - 1) / (np.sqrt(2) - 1)

    for on in (False, True):
        tea.data.vertices.foreach_set("co", V0.astype(np.float32).ravel())
        props.cubeness, props.square_flat = 0.2, on
        select_only(tea)
        res = bpy.ops.object.cubify_cpp_mesh()
        V1 = coords(tea.data)
        seat = ((V1[li] - V1[bi]) - (V0[li] - V0[bi]))[:, 2].mean()
        sq = lid_square(V1)
        # the solver seats the mean over *all* contacts; this subsample may
        # differ slightly (the unseated gap would be ~0.13-0.19)
        ok = res == {'FINISHED'} and abs(seat) < 5e-3 and (sq > 0.3 if on else sq < 0.15)
        check(f"teapot, Square Flat Regions {'on' if on else 'off'}: lid seated, "
              f"outline {'square' if on else 'round'}", ok,
              f"seat height {seat:+.1e}, lid squareness {sq:.2f}")
    props.square_flat, props.cubeness = False, 0.5
    bpy.data.objects.remove(tea)

    # ---- thin walls: the teapot is a solid shell whose walls cross when cubified
    from mathutils.bvhtree import BVHTree

    def crossing_pairs(me):
        me.calc_loop_triangles()
        tris = [tuple(t.vertices) for t in me.loop_triangles]
        tree = BVHTree.FromPolygons([v.co.copy() for v in me.vertices], tris,
                                    all_triangles=True)
        return sum(1 for a, b in tree.overlap(tree)
                   if a < b and not set(tris[a]) & set(tris[b]))

    bpy.ops.wm.obj_import(filepath=os.path.join(REPO, "test_objs", "utah_teapot.obj"),
                          forward_axis='Y', up_axis='Z')
    tea = bpy.context.selected_objects[0]
    select_only(tea)
    res = bpy.ops.object.cubify_cpp_fix_thin_walls()
    check("Fix Thin Walls needs a prior Cubify", res == {'CANCELLED'})
    V0 = coords(tea.data).copy()
    check("Auto-fix Thin Walls off by default", not props.auto_fix_walls)
    props.cubeness = 0.2
    bpy.ops.object.cubify_cpp_mesh()
    n_cross = crossing_pairs(tea.data)
    res = bpy.ops.object.cubify_cpp_fix_thin_walls()
    n_fixed = crossing_pairs(tea.data)
    check("Fix Thin Walls button", res == {'FINISHED'} and n_cross > 100 and n_fixed < 0.1 * n_cross,
          f"crossing triangle pairs {n_cross} -> {n_fixed}")
    tea.data.vertices.foreach_set("co", V0.astype(np.float32).ravel())
    tea.data.update()
    props.auto_fix_walls = True
    bpy.ops.object.cubify_cpp_mesh()
    n_auto = crossing_pairs(tea.data)
    check("Auto-fix after Cubify", n_auto < 0.1 * n_cross, f"crossing triangle pairs {n_auto}")
    bpy.data.objects.remove(tea)

    ob = new_sphere("bake_fix", True)  # bakes run the fix on every step
    select_only(ob)
    props.anim_mode, props.anim_samples = 'LAMBDA', 3
    res = bpy.ops.object.cubify_cpp_bake_animation()
    check("bake with Auto-fix", res == {'FINISHED'} and len(ob.data.shape_keys.key_blocks) == 4)
    bpy.data.objects.remove(ob)
    props.auto_fix_walls, props.cubeness = False, 0.5

    # ---- Repeat: N passes == pressing Cubify N times
    ob = new_sphere("repeat", False)
    select_only(ob)
    V0 = coords(ob.data).copy()
    check("Repeat defaults to 1", props.repeat == 1)
    bpy.ops.object.cubify_cpp_mesh()
    bpy.ops.object.cubify_cpp_mesh()
    twice = coords(ob.data).copy()
    ob.data.vertices.foreach_set("co", V0.astype(np.float32).ravel())
    ob.data.update()
    props.repeat = 2
    res = bpy.ops.object.cubify_cpp_mesh()
    props.repeat = 1
    rest = addon.read_rest(ob.data)
    check("Repeat 2 == Cubify twice", res == {'FINISHED'}
          and np.allclose(coords(ob.data), twice, atol=1e-5),
          f"max difference {np.abs(coords(ob.data) - twice).max():.1e}")
    check("Repeat keeps the original as the thin-wall rest pose",
          rest is not None and np.allclose(rest, V0, atol=1e-6))
    bpy.data.objects.remove(ob)

    # ---- apply to copy + multiple selection
    a = new_sphere("multi_a", False)
    b = new_sphere("multi_b", True)
    b.location.x = 3
    for o in (a, b):
        o.select_set(True)
    props.apply_to_copy = True
    Va = coords(a.data).copy()
    res = bpy.ops.object.cubify_cpp_mesh()
    props.apply_to_copy = False
    copies = [o for o in bpy.data.objects if o.name.endswith("_cubified")]
    check("apply to copy (2 objects)", res == {'FINISHED'} and len(copies) == 2
          and np.array_equal(coords(a.data), Va))
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

    # ---- pins: set from selection, then cubify holds them
    ob = new_sphere("pinned", False)
    select_only(ob)
    me = ob.data
    V0 = coords(me).copy()
    sel = np.flatnonzero(V0[:, 2] > 0.9)
    for v in me.vertices:
        v.select = v.index in set(sel.tolist())
    res = bpy.ops.object.cubify_cpp_pins(action='SET')
    check("set pins", res == {'FINISHED'} and len(addon.get_pin_indices(ob)) == len(sel),
          f"{len(sel)} pins")
    bpy.ops.object.cubify_cpp_mesh()
    V1 = coords(me)
    check("cubify holds pins", np.allclose(V1[sel], V0[sel], atol=1e-5)
          and not np.allclose(V1, V0, atol=1e-3))

    # ---- manipulator solver path: drag one pin, ARAP
    V, F = addon.read_mesh_arrays(me)
    s, device, _ = addon.create_stylizer(bpy.context, V, F, 0.0,
                                         addon.cube_axes(props), addon.get_pin_indices(ob))
    pp = V[s.pins].copy()
    pp[0] += [0.3, 0.0, 0.2]
    Vd = s.solve(pin_pos=pp, warm_last=True, iterations=props.drag_iterations)
    Vd = s.solve(pin_pos=pp, warm_last=True, iterations=props.drag_iterations)
    s.close()
    check("manipulator drag solve", np.allclose(Vd[s.pins], pp, atol=1e-9), device)
    bpy.ops.object.cubify_cpp_pins(action='CLEAR')
    check("clear pins", not addon.get_pin_indices(ob))
    bpy.data.objects.remove(ob)

    # ---- style-as-process bake, both modes
    for mode in ('LAMBDA', 'ITER'):
        ob = new_sphere(f"bake_{mode}", True)
        select_only(ob)
        props.anim_mode = mode
        props.anim_samples = 6
        V0 = coords(ob.data).copy()
        res = bpy.ops.object.cubify_cpp_bake_animation()
        key = ob.data.shape_keys
        nkeys = len(key.key_blocks) if key else 0
        fcs = eval_time_fcurves(key) if key else []
        linear = bool(fcs) and all(kp.interpolation == 'LINEAR'
                                   for fc in fcs for kp in fc.keyframe_points)
        last = np.empty(len(V0) * 3)
        key.key_blocks[-1].data.foreach_get("co", last)
        check(f"bake animation ({mode})",
              res == {'FINISHED'} and nkeys == 7 and linear
              and np.array_equal(coords(ob.data), V0)
              and not np.allclose(last.reshape(-1, 3), V0, atol=1e-3),
              f"{nkeys} keys, linear={linear}")
        bpy.data.objects.remove(ob)

    # ---- cubify every frame (object rotates over frames 1..9)
    ob = new_sphere("animated", False)
    select_only(ob)
    scene.frame_start, scene.frame_end = 1, 9
    props.anim_frame_step = 4
    ob.rotation_euler = (0, 0, 0)
    ob.keyframe_insert("rotation_euler", frame=1)
    ob.rotation_euler = (0, 0, 0.8)
    ob.keyframe_insert("rotation_euler", frame=9)
    scene.frame_set(5)
    res = bpy.ops.object.cubify_cpp_bake_frames()
    copy = bpy.data.objects.get("animated_cubified_anim")
    nkeys = len(copy.data.shape_keys.key_blocks) if copy else 0
    check("cubify every frame", res == {'FINISHED'} and nkeys == 4  # basis + 1,5,9
          and scene.frame_current == 5, f"{nkeys} keys")

    # ---- server is shut down with the add-on
    addon_utils.disable(MODULE, default_set=True)
    check("server stopped on disable", addon.client.server_cached() is None)

    print("FAILED: " + ", ".join(FAILED) if FAILED else "all checks passed")


try:
    main()
except Exception:
    traceback.print_exc()
    FAILED.append("exception")
sys.stdout.flush()
os._exit(1 if FAILED else 0)
