bl_info = {
    "name": "Cubic Stylization (C++)",
    "author": "Christopher Mok",
    "version": (1, 0, 0),
    "blender": (2, 93, 0),
    "location": "3D Viewport > Sidebar (N) > Cubify C++",
    "description": "Cubify meshes (Liu & Jacobson 2019) and ARAP handle deformation, "
                   "solved by a native C++/Eigen server process. Works on triangle "
                   "and quad meshes; topology and UVs are preserved",
    "category": "Mesh",
}

import os
import threading
import time

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Euler, Vector
from bpy_extras import view3d_utils

if "client" in locals():
    import importlib
    importlib.reload(client)  # noqa: F821
    importlib.reload(builder)  # noqa: F821
from . import builder, client

# Shared with the Python add-on, so pins carry over between the two.
PIN_GROUP = "CubifyPins"


# ================== Mesh helpers

def mesh_kind(me):
    """Classify polygon sizes. Returns (label, tris, quads, ngons)."""
    m = len(me.polygons)
    if m == 0:
        return "No faces", 0, 0, 0
    sizes = np.empty(m, dtype=np.int64)
    me.polygons.foreach_get("loop_total", sizes)
    tris = int(np.count_nonzero(sizes == 3))
    quads = int(np.count_nonzero(sizes == 4))
    ngons = m - tris - quads
    if ngons:
        label = "Mixed mesh (has n-gons)"
    elif tris and quads:
        label = "Mixed mesh (tris + quads)"
    elif quads:
        label = "Quad mesh"
    else:
        label = "Triangle mesh"
    return label, tris, quads, ngons


def read_mesh_arrays(me):
    """Vertex positions and a *virtual* triangulation of the polygons.

    The mesh itself is never modified, so quad/n-gon topology — and with it
    all UV/loop data — is preserved; only vertex positions are solved for.
    """
    n = len(me.vertices)
    V = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)

    me.calc_loop_triangles()
    m = len(me.loop_triangles)
    F = np.empty(m * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", F)
    return V, F.reshape(-1, 3)


def write_mesh_positions(me, V):
    me.vertices.foreach_set("co", np.asarray(V, dtype=np.float32).ravel())
    me.update()


def get_pin_indices(ob):
    vg = ob.vertex_groups.get(PIN_GROUP)
    if vg is None:
        return []
    gi = vg.index
    return [v.index for v in ob.data.vertices
            if any(g.group == gi for g in v.groups)]


# Shape before the last Cubify, kept on the mesh (hidden: leading dot) so
# Fix Thin Walls can run later — it measures walls in the pre-cubify shape.
REST_ATTR = ".cubify_rest"


def store_rest(me, V):
    a = me.attributes.get(REST_ATTR)
    if a is not None and (a.domain != 'POINT' or a.data_type != 'FLOAT_VECTOR'):
        me.attributes.remove(a)
        a = None
    if a is None:
        me.attributes.new(REST_ATTR, 'FLOAT_VECTOR', 'POINT')
    me.attributes[REST_ATTR].data.foreach_set(
        "vector", np.asarray(V, dtype=np.float32).ravel())


def read_rest(me):
    """The stored pre-cubify positions, or None if missing or stale."""
    a = me.attributes.get(REST_ATTR)
    if (a is None or a.domain != 'POINT' or a.data_type != 'FLOAT_VECTOR'
            or len(a.data) != len(me.vertices)):
        return None
    V = np.empty(len(me.vertices) * 3, dtype=np.float32)
    a.data.foreach_get("vector", V)
    return V.reshape(-1, 3).astype(np.float64)


def fix_walls(context, V_rest, V, F, min_thickness=0.1, iterations=60):
    props = context.scene.cubify_cpp_settings
    return client.fix_thin_walls(V_rest, V, F, min_thickness=min_thickness,
                                 iterations=iterations, threads=props.threads,
                                 server_path=server_path(context))


def walls_note(st):
    if st["crossed_before"] == 0 and st["moved"] == 0:
        return "no thin-wall crossings"
    return (f"thin walls: {st['crossed_before']} crossed points -> "
            f"{st['crossed_after']} ({st['moved']} vertices moved, "
            f"max {st['max_move']:.3g})")


def addon_prefs(context=None):
    context = context or bpy.context
    addon = context.preferences.addons.get(__package__)
    return addon.preferences if addon is not None else None


def server_path(context=None):
    prefs = addon_prefs(context)
    return client.resolve_server_path(prefs.server_path if prefs else "")


def flat_relax(props):
    """Edge-weight factor for Square Flat Regions (1 = off). Strength 1
    gives 0.01, the value the feature was tuned with."""
    if not props.square_flat:
        return 1.0
    return 1.0 - 0.99 * props.square_flat_strength


def create_stylizer(context, V, F, cubeness, A, pins, allow_square_flat=True):
    props = context.scene.cubify_cpp_settings
    return client.create_stylizer(V, F, cubeness=cubeness, cube_axes=A, pins=pins,
                                  threads=props.threads,
                                  server_path=server_path(context),
                                  flat_relax=flat_relax(props) if allow_square_flat else 1.0)


def cube_axes(props):
    return np.array(Euler(props.orientation, 'XYZ').to_matrix(), dtype=np.float64)


def rotation_part(M):
    """Nearest proper rotation to the 3x3 part of a 4x4 matrix (drops
    scale and shear)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64)[:3, :3])
    if np.linalg.det(U @ Vt) < 0:
        U[:, 2] *= -1
    return U @ Vt


# ================== Settings

class CubifyCppSettings(bpy.types.PropertyGroup):
    cubeness: bpy.props.FloatProperty(
        name="Cubeness",
        description="Strength of the L1 cubeness term (lambda). 0 is plain ARAP; "
                    "0.2-1.0 gives increasingly sharp cubes",
        default=0.2, min=0.0, soft_max=5.0, max=20.0, step=1, precision=2)
    orientation: bpy.props.FloatVectorProperty(
        name="Cube Orientation",
        description="Rotation of the target cube axes (in the object's local space)",
        subtype='EULER', default=(0.0, 0.0, 0.0))
    iterations: bpy.props.IntProperty(
        name="Iterations",
        description="Local-global solver iterations",
        default=30, min=1, max=500)
    admm_iterations: bpy.props.IntProperty(
        name="ADMM Iterations",
        description="Maximum inner ADMM iterations per rotation fit "
                    "(stops early on convergence)",
        default=100, min=1, max=300)
    repeat: bpy.props.IntProperty(
        name="Repeat",
        description="Cubify this many times in a row, each pass starting from "
                    "the previous result (same as pressing Cubify again). The "
                    "style compounds: stronger than raising Cubeness on "
                    "detailed shapes. Cubify Mesh only; time scales with it",
        default=1, min=1, max=20)
    apply_to_copy: bpy.props.BoolProperty(
        name="Apply to Copy",
        description="Cubify a duplicate and keep the original object unchanged",
        default=False)
    stylized_drag: bpy.props.BoolProperty(
        name="Stylized Drag",
        description="Use the current Cubeness during ARAP manipulation, so "
                    "dragging preserves the cubic style (0 = classic ARAP)",
        default=False)
    drag_iterations: bpy.props.IntProperty(
        name="Drag Iterations",
        description="Local-global iterations per mouse move while dragging "
                    "(higher is stiffer/more converged but slower)",
        default=2, min=1, max=20)
    anim_mode: bpy.props.EnumProperty(
        name="Animate By",
        description="What advances from one baked step to the next",
        items=[
            ('LAMBDA', "Cubeness Ramp",
             "Ramp Cubeness from 0 to its current value with a few "
             "warm-started iterations per step — evenly paced, eased "
             "transformation"),
            ('ITER', "Iterations",
             "One solver iteration per step at full Cubeness — the raw "
             "convergence: flat regions snap first, creases sharpen late"),
        ], default='LAMBDA')
    anim_samples: bpy.props.IntProperty(
        name="Steps",
        description="Number of baked steps (one shape key each)",
        default=20, min=2, max=200)
    anim_iters: bpy.props.IntProperty(
        name="Iterations / Step",
        description="Warm-started solver iterations per step (Cubeness Ramp "
                    "mode). Raise it if the final frames look unconverged",
        default=3, min=1, max=50)
    anim_frame_step: bpy.props.IntProperty(
        name="Frame Step",
        description="Timeline frames between baked steps. Style-as-process "
                    "bakes start at the current frame; Cubify Every Frame "
                    "samples the scene frame range at this step",
        default=2, min=1, max=50)
    anim_axes: bpy.props.EnumProperty(
        name="Cube Axes",
        description="What the cube axes are attached to in Cubify Every Frame",
        items=[
            ('OBJECT', "Object",
             "Cube axes turn with the object: a rotating object bakes as a "
             "rotating cubified shape"),
            ('WORLD', "World",
             "Cube axes stay fixed in the world: a rotating object "
             "re-crystallizes against the world axes as it turns"),
        ], default='OBJECT')
    square_flat: bpy.props.BoolProperty(
        name="Square Flat Regions",
        description="Let surfaces that already face a cube axis (lids, plates, "
                    "flat tops) reshape in-plane so their round outlines turn "
                    "square. Goes beyond the paper's method; about 3x slower. "
                    "Not used while dragging pins",
        default=False)
    square_flat_strength: bpy.props.FloatProperty(
        name="Strength",
        description="How freely flat regions may reshape (0 = no effect)",
        default=1.0, min=0.0, max=1.0, subtype='FACTOR')
    auto_fix_walls: bpy.props.BoolProperty(
        name="Auto-fix Thin Walls",
        description="After Cubify and for every baked step, push apart thin "
                    "walls (e.g. the inner and outer surface of a spout) that "
                    "cubification made cross each other",
        default=False)
    threads: bpy.props.IntProperty(
        name="Threads",
        description="CPU threads the C++ solver uses (0 = all cores)",
        default=0, min=0, max=256)


# ================== Cubify operator

class OBJECT_OT_cubify_cpp(bpy.types.Operator):
    """Apply Cubic Stylization to the selected mesh objects (C++ solver).
    Quad meshes are solved on a virtual triangulation: topology and UVs
    are preserved. Pinned vertices are held in place"""
    bl_idname = "object.cubify_cpp_mesh"
    bl_label = "Cubify Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (context.mode == 'OBJECT' and ob is not None and ob.type == 'MESH')

    def execute(self, context):
        props = context.scene.cubify_cpp_settings
        targets = [ob for ob in context.selected_objects if ob.type == 'MESH']
        if not targets and context.active_object and context.active_object.type == 'MESH':
            targets = [context.active_object]
        if not targets:
            self.report({'ERROR'}, "Select at least one mesh object")
            return {'CANCELLED'}

        wm = context.window_manager
        wm.progress_begin(0, 100)
        done = 0
        try:
            for ob in targets:
                ok = self._cubify_object(context, ob, props,
                                         base=done / len(targets),
                                         span=1.0 / len(targets))
                if ok:
                    done += 1
        finally:
            wm.progress_end()

        if done == 0:
            return {'CANCELLED'}
        return {'FINISHED'}

    def _cubify_object(self, context, ob, props, base, span):
        if ob.data.shape_keys is not None:
            self.report({'WARNING'}, f"{ob.name}: skipped (has shape keys)")
            return False
        if len(ob.data.polygons) == 0:
            self.report({'WARNING'}, f"{ob.name}: skipped (no faces)")
            return False

        if props.apply_to_copy:
            dup = ob.copy()
            dup.data = ob.data.copy()
            dup.name = ob.name + "_cubified"
            context.collection.objects.link(dup)
            ob = dup

        me = ob.data
        kind, _, _, _ = mesh_kind(me)
        V, F = read_mesh_arrays(me)
        pins = get_pin_indices(ob)

        wm = context.window_manager
        t0 = time.time()
        passes = props.repeat
        V_out = V
        for p in range(passes):
            # each pass cubifies the previous result as its new rest pose —
            # the same as pressing Cubify again, so the style compounds
            def progress(i, total, p=p):
                wm.progress_update(int(100 * (base + span * (p + i / total) / passes)))
            stylizer = None
            try:
                stylizer, device, _ = create_stylizer(context, V_out, F, props.cubeness,
                                                      cube_axes(props), pins)
                V_out = stylizer.run(iterations=props.iterations,
                                     admm_iters=props.admm_iterations,
                                     on_progress=progress)
            except Exception as exc:
                self.report({'ERROR'}, f"{ob.name}: solver failed ({exc})")
                return False
            finally:
                if stylizer is not None:
                    stylizer.close()

            if not np.all(np.isfinite(V_out)):
                self.report({'ERROR'}, f"{ob.name}: solver produced invalid positions")
                return False

        fix_note = ""
        if props.auto_fix_walls:
            try:
                V_out, st = fix_walls(context, V, V_out, F)
                fix_note = "; " + walls_note(st)
            except Exception as exc:
                self.report({'WARNING'}, f"{ob.name}: thin-wall fix failed ({exc})")

        store_rest(me, V)  # for a later Fix Thin Walls
        write_mesh_positions(me, V_out)

        note = f", {len(pins)} pinned" if pins else ""
        if passes > 1:
            note += f", x{passes} passes"
        self.report({'INFO'},
                    f"{ob.name} ({kind.lower()}): cubified {len(V)} vertices in "
                    f"{time.time() - t0:.2f}s on {device} "
                    f"(lambda={props.cubeness:.2f}{note}){fix_note}")
        return True


class OBJECT_OT_cubify_cpp_fix_thin_walls(bpy.types.Operator):
    """Push apart thin walls that the last Cubify made cross each other
    (e.g. the inner surface of a hollow spout poking through the outer one).
    Walls are measured in the shape from before that Cubify, which Cubify
    stores on the mesh"""
    bl_idname = "object.cubify_cpp_fix_thin_walls"
    bl_label = "Fix Thin Walls"
    bl_options = {'REGISTER', 'UNDO'}

    min_thickness: bpy.props.FloatProperty(
        name="Min Thickness",
        description="Walls thinner than this fraction of their original "
                    "thickness (or crossed) are pushed back to it",
        default=0.1, min=0.0, max=1.0, subtype='FACTOR')
    iterations: bpy.props.IntProperty(
        name="Iterations", description="Maximum push-apart passes",
        default=60, min=1, max=500)

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (context.mode == 'OBJECT' and ob is not None and ob.type == 'MESH')

    def execute(self, context):
        targets = [ob for ob in context.selected_objects if ob.type == 'MESH']
        if not targets and context.active_object and context.active_object.type == 'MESH':
            targets = [context.active_object]
        done = 0
        for ob in targets:
            me = ob.data
            V_rest = read_rest(me)
            if V_rest is None:
                self.report({'WARNING'},
                            f"{ob.name}: skipped (no pre-cubify shape stored — "
                            "cubify it with this add-on first)")
                continue
            V, F = read_mesh_arrays(me)
            try:
                V_out, st = fix_walls(context, V_rest, V, F,
                                      min_thickness=self.min_thickness,
                                      iterations=self.iterations)
            except Exception as exc:
                self.report({'ERROR'}, f"{ob.name}: thin-wall fix failed ({exc})")
                continue
            write_mesh_positions(me, V_out)
            self.report({'INFO'}, f"{ob.name}: {walls_note(st)}")
            done += 1
        return {'FINISHED'} if done else {'CANCELLED'}


# ================== Style-as-process animation bake

class OBJECT_OT_cubify_cpp_bake_anim(bpy.types.Operator):
    """Bake the cubification process as an animation over the timeline:
    one absolute shape key per step, driven by keyframed Evaluation Time.
    The base mesh, topology and UVs are untouched — delete the shape keys
    to get the original back"""
    bl_idname = "object.cubify_cpp_bake_animation"
    bl_label = "Bake Animation (Shape Keys)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (context.mode == 'OBJECT' and ob is not None and ob.type == 'MESH')

    def execute(self, context):
        props = context.scene.cubify_cpp_settings
        if props.cubeness <= 0.0:
            self.report({'ERROR'}, "Cubeness is 0 — nothing to animate")
            return {'CANCELLED'}
        targets = [ob for ob in context.selected_objects if ob.type == 'MESH']
        if not targets and context.active_object and context.active_object.type == 'MESH':
            targets = [context.active_object]
        if not targets:
            self.report({'ERROR'}, "Select at least one mesh object")
            return {'CANCELLED'}

        wm = context.window_manager
        wm.progress_begin(0, 100)
        done = 0
        try:
            for ob in targets:
                ok = self._bake_object(context, ob, props,
                                       base=done / len(targets),
                                       span=1.0 / len(targets))
                if ok:
                    done += 1
        finally:
            wm.progress_end()

        if done == 0:
            return {'CANCELLED'}
        return {'FINISHED'}

    def _bake_object(self, context, ob, props, base, span):
        me = ob.data
        if me.shape_keys is not None:
            self.report({'WARNING'}, f"{ob.name}: skipped (already has shape keys)")
            return False
        if len(me.polygons) == 0:
            self.report({'WARNING'}, f"{ob.name}: skipped (no faces)")
            return False

        V0, F = read_mesh_arrays(me)
        pins = get_pin_indices(ob)

        try:
            stylizer, device, _ = create_stylizer(context, V0, F, props.cubeness,
                                                  cube_axes(props), pins)
        except Exception as exc:
            self.report({'ERROR'}, f"{ob.name}: solver setup failed ({exc})")
            return False

        try:
            # Solve every step first, so a failure leaves the mesh untouched.
            # Each step is warm-started from the previous one, which stays on
            # the server (warm_last) instead of being re-sent.
            wm = context.window_manager
            steps = props.anim_samples
            results = []
            t0 = time.time()
            for k in range(1, steps + 1):
                if props.anim_mode == 'LAMBDA':
                    stylizer.lam = props.cubeness * k / steps
                    iters = props.anim_iters
                else:
                    iters = 1
                try:
                    V = stylizer.solve(warm_last=True, iterations=iters,
                                       admm_iters=props.admm_iterations)
                except Exception as exc:
                    self.report({'ERROR'}, f"{ob.name}: solver failed at step {k} ({exc})")
                    return False
                if not np.all(np.isfinite(V)):
                    self.report({'ERROR'}, f"{ob.name}: invalid positions at step {k}")
                    return False
                if props.auto_fix_walls:
                    # only the baked key is fixed; the solver keeps
                    # warm-starting from its own unfixed result
                    try:
                        V, _ = fix_walls(context, V0, V, F)
                    except Exception as exc:
                        self.report({'ERROR'},
                                    f"{ob.name}: thin-wall fix failed at step {k} ({exc})")
                        return False
                results.append(V)
                wm.progress_update(int(100 * (base + span * k / steps)))
        finally:
            stylizer.close()

        # Basis = untouched original; every step is its own absolute key.
        # Only shape-key point positions are written: base mesh vertices,
        # topology, UVs and all loop data stay exactly as they are.
        ob.shape_key_add(name="Basis", from_mix=False)
        key = me.shape_keys
        key.use_relative = False
        for k, V in enumerate(results, start=1):
            kb = ob.shape_key_add(name=f"Cubify {k:03d}", from_mix=False)
            kb.interpolation = 'KEY_LINEAR'
            kb.data.foreach_set("co", np.asarray(V, dtype=np.float32).ravel())

        # Play the key sequence across the timeline: eval_time runs linearly
        # from the Basis key's position to the last key's.
        f0 = context.scene.frame_current
        f1 = f0 + steps * props.anim_frame_step
        key.eval_time = key.key_blocks[0].frame
        key.keyframe_insert(data_path="eval_time", frame=f0)
        key.eval_time = key.key_blocks[-1].frame
        key.keyframe_insert(data_path="eval_time", frame=f1)
        _linear_eval_time(key)

        mode = ("cubeness ramp" if props.anim_mode == 'LAMBDA'
                else "iterations")
        self.report({'INFO'},
                    f"{ob.name}: baked {steps} steps ({mode}) over frames "
                    f"{f0}-{f1} in {time.time() - t0:.2f}s on {device}")
        return True


def _linear_eval_time(key):
    """Linear interpolation on the eval_time keyframes. Blender 4.4+ stores
    f-curves in layered actions (channelbags), older versions on the action."""
    ad = key.animation_data
    if not ad or not ad.action:
        return
    fcurves = getattr(ad.action, "fcurves", None)
    if fcurves is None:
        try:
            from bpy_extras import anim_utils
            bag = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
            fcurves = bag.fcurves if bag else []
        except Exception:
            fcurves = []
    for fc in fcurves:
        if fc.data_path == "eval_time":
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'


class OBJECT_OT_cubify_cpp_bake_frames(bpy.types.Operator):
    """Cubify the animated mesh at every sampled frame of the scene range
    and bake the results onto a NEW copy object as shape keys over the
    timeline. The copy follows the original's animation (object transforms,
    shape keys, armatures, deforming modifiers — evaluated in world space)
    with each frame re-cubified; the original is untouched"""
    bl_idname = "object.cubify_cpp_bake_frames"
    bl_label = "Cubify Every Frame (to Copy)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (context.mode == 'OBJECT' and ob is not None and ob.type == 'MESH')

    def execute(self, context):
        props = context.scene.cubify_cpp_settings
        if props.cubeness <= 0.0:
            self.report({'ERROR'}, "Cubeness is 0 — nothing to cubify")
            return {'CANCELLED'}
        ob = context.active_object
        scene = context.scene
        frames = list(range(scene.frame_start, scene.frame_end + 1,
                            props.anim_frame_step))
        if frames[-1] != scene.frame_end:
            frames.append(scene.frame_end)

        frame_restore = scene.frame_current
        wm = context.window_manager
        wm.progress_begin(0, 100)
        new_ob = None
        try:
            # ---- setup from the first sampled frame
            scene.frame_set(frames[0])
            dg = context.evaluated_depsgraph_get()
            ob_eval = ob.evaluated_get(dg)
            me0 = bpy.data.meshes.new_from_object(
                ob_eval, preserve_all_data_layers=True, depsgraph=dg)
            n = len(me0.vertices)
            if n == 0 or len(me0.polygons) == 0:
                bpy.data.meshes.remove(me0)
                self.report({'ERROR'}, f"{ob.name}: evaluated mesh has no geometry")
                return {'CANCELLED'}
            _, F = read_mesh_arrays(me0)

            # pins only translate to the evaluated mesh when no modifier
            # changed the vertex count/order
            pins = []
            if n == len(ob.data.vertices):
                pins = get_pin_indices(ob)
            elif get_pin_indices(ob):
                self.report({'WARNING'},
                            f"{ob.name}: pins ignored (modifiers change the "
                            "vertex count)")

            A = cube_axes(props)

            new_ob = bpy.data.objects.new(ob.name + "_cubified_anim", me0)
            context.collection.objects.link(new_ob)
            # world space is baked into the keys; the copy stays at identity
            new_ob.shape_key_add(name="Basis", from_mix=False)
            key = me0.shape_keys
            key.use_relative = False

            # ---- per-frame solve
            device = None
            t0 = time.time()
            for i, f in enumerate(frames):
                scene.frame_set(f)
                dg = context.evaluated_depsgraph_get()
                ob_eval = ob.evaluated_get(dg)
                me_ev = ob_eval.data
                if len(me_ev.vertices) != n:
                    raise RuntimeError(
                        f"vertex count changed at frame {f} "
                        f"({len(me_ev.vertices)} vs {n}) — animated "
                        "topology-changing modifiers are not supported")
                V = np.empty(n * 3, dtype=np.float64)
                me_ev.vertices.foreach_get("co", V)
                M = np.array(ob_eval.matrix_world, dtype=np.float64)
                Vw = V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]

                A_f = rotation_part(M) @ A if props.anim_axes == 'OBJECT' else A

                # Every frame is solved from its own pose, never warm-started
                # from the previous result: ARAP leaves the output's global
                # rotation free, so the cube term would turn an already
                # cubified shape back to its earlier orientation and the
                # bake would stop following the animation. The rest pose
                # changes every frame, so each frame also needs its own
                # system (cotan weights depend on it).
                stylizer, device, _ = create_stylizer(context, Vw, F, props.cubeness,
                                                      A_f, pins)
                with stylizer:
                    V_out = stylizer.run(iterations=props.iterations,
                                         admm_iters=props.admm_iterations)
                if not np.all(np.isfinite(V_out)):
                    raise RuntimeError(f"invalid positions at frame {f}")
                if props.auto_fix_walls:
                    V_out, _ = fix_walls(context, Vw, V_out, F)

                kb = new_ob.shape_key_add(name=f"Frame {f:04d}", from_mix=False)
                kb.interpolation = 'KEY_LINEAR'
                kb.data.foreach_set("co", np.asarray(V_out, dtype=np.float32).ravel())
                key.eval_time = kb.frame
                key.keyframe_insert(data_path="eval_time", frame=f)
                wm.progress_update(int(100 * (i + 1) / len(frames)))

            _linear_eval_time(key)

            self.report({'INFO'},
                        f"{new_ob.name}: {len(frames)} frames "
                        f"({frames[0]}-{frames[-1]}, step {props.anim_frame_step}) "
                        f"in {time.time() - t0:.2f}s on {device}")
            return {'FINISHED'}
        except Exception as exc:
            if new_ob is not None:
                me = new_ob.data
                bpy.data.objects.remove(new_ob)
                bpy.data.meshes.remove(me)
            self.report({'ERROR'}, f"{ob.name}: {exc}")
            return {'CANCELLED'}
        finally:
            scene.frame_set(frame_restore)
            wm.progress_end()


# ================== Pin management

class OBJECT_OT_cubify_cpp_pins(bpy.types.Operator):
    """Manage the pinned (handle) vertices stored in the 'CubifyPins'
    vertex group"""
    bl_idname = "object.cubify_cpp_pins"
    bl_label = "Cubify Pins"
    bl_options = {'REGISTER', 'UNDO'}

    action: bpy.props.EnumProperty(items=[
        ('SET', "Set From Selection", "Replace pins with the selected vertices"),
        ('ADD', "Add Selection", "Add the selected vertices to the pins"),
        ('CLEAR', "Clear", "Remove all pins"),
    ], default='SET')

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob is not None and ob.type == 'MESH'
                and context.mode in {'OBJECT', 'EDIT_MESH'})

    def execute(self, context):
        ob = context.active_object
        prev_mode = ob.mode
        if prev_mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')  # sync edit-mode selection
        try:
            me = ob.data
            if self.action == 'CLEAR':
                vg = ob.vertex_groups.get(PIN_GROUP)
                if vg is not None:
                    ob.vertex_groups.remove(vg)
                self.report({'INFO'}, f"{ob.name}: pins cleared")
                return {'FINISHED'}

            sel = [v.index for v in me.vertices if v.select]
            if not sel:
                self.report({'ERROR'}, "No vertices selected "
                            "(select vertices in Edit Mode first)")
                return {'CANCELLED'}

            vg = ob.vertex_groups.get(PIN_GROUP)
            if vg is None:
                vg = ob.vertex_groups.new(name=PIN_GROUP)
            elif self.action == 'SET':
                vg.remove(list(range(len(me.vertices))))
            vg.add(sel, 1.0, 'REPLACE')
            self.report({'INFO'}, f"{ob.name}: {len(get_pin_indices(ob))} pins")
            return {'FINISHED'}
        finally:
            if prev_mode == 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')


# ================== Interactive ARAP manipulation

class OBJECT_OT_arap_cpp_manipulate(bpy.types.Operator):
    """Interactive ARAP deformation (C++ solver): drag pinned vertices and
    the mesh follows as-rigidly-as-possible. Click near a pin to grab it,
    drag to deform, Enter/Space to confirm, Esc/right-click to cancel"""
    bl_idname = "object.arap_cpp_manipulate"
    bl_label = "Start Manipulation"
    bl_options = {'REGISTER', 'UNDO'}

    _PICK_RADIUS_PX = 30.0

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (context.mode == 'OBJECT' and ob is not None and ob.type == 'MESH')

    def invoke(self, context, event):
        ob = context.active_object
        me = ob.data
        if me.shape_keys is not None:
            self.report({'ERROR'}, "Meshes with shape keys are not supported")
            return {'CANCELLED'}
        pins = get_pin_indices(ob)
        if not pins:
            self.report({'ERROR'},
                        "No pins: select vertices and use 'Set From Selection' first")
            return {'CANCELLED'}
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Must be run from a 3D Viewport")
            return {'CANCELLED'}

        props = context.scene.cubify_cpp_settings
        V, F = read_mesh_arrays(me)
        lam = props.cubeness if props.stylized_drag else 0.0

        try:
            # factorized once here; every drag only re-solves on the server
            # (Square Flat Regions would refactorize every iteration, so it
            # stays off for interactive drags)
            self.solver, device, _ = create_stylizer(context, V, F, lam,
                                                     cube_axes(props), pins,
                                                     allow_square_flat=False)
        except Exception as exc:
            self.report({'ERROR'}, f"Solver setup failed ({exc})")
            return {'CANCELLED'}

        self.ob = ob
        self.pins = np.asarray(self.solver.pins)          # sorted unique
        self.pin_pos = V[self.pins].copy()                # drag targets
        self.V_orig = V.copy()
        self.V_cur = V.copy()
        self.grab = None                                  # index into self.pins
        self.drag_iters = props.drag_iterations
        self.admm_iters = props.admm_iterations

        self.area = context.area
        self.region = next(r for r in self.area.regions if r.type == 'WINDOW')
        self.rv3d = self.area.spaces.active.region_3d

        try:
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        except Exception:
            shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
        self._shader = shader
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_pins, (), 'WINDOW', 'POST_VIEW')

        context.workspace.status_text_set(
            f"ARAP Manipulation ({device})  |  Left-drag a pin to deform  |  "
            "Enter/Space: confirm  |  Esc/Right-click: cancel")
        context.window_manager.modal_handler_add(self)
        self.area.tag_redraw()
        return {'RUNNING_MODAL'}

    # ---- drawing

    def _draw_pins(self):
        mw = self.ob.matrix_world
        coords = [tuple(mw @ Vector(self.V_cur[p])) for p in self.pins]
        gpu.state.depth_test_set('NONE')
        gpu.state.point_size_set(12.0)
        self._shader.bind()
        self._shader.uniform_float("color", (1.0, 0.15, 0.15, 1.0))
        batch_for_shader(self._shader, 'POINTS', {"pos": coords}).draw(self._shader)
        if self.grab is not None:
            gpu.state.point_size_set(16.0)
            self._shader.uniform_float("color", (0.2, 1.0, 0.3, 1.0))
            g = [tuple(mw @ Vector(self.pin_pos[self.grab]))]
            batch_for_shader(self._shader, 'POINTS', {"pos": g}).draw(self._shader)
        gpu.state.point_size_set(1.0)
        gpu.state.depth_test_set('LESS_EQUAL')

    # ---- interaction helpers

    def _mouse_region_coords(self, event):
        return (event.mouse_x - self.region.x, event.mouse_y - self.region.y)

    def _pick_pin(self, coord):
        """Nearest pin to the 2D mouse position, or None."""
        mw = self.ob.matrix_world
        best, best_d = None, self._PICK_RADIUS_PX
        for k, p in enumerate(self.pins):
            world = mw @ Vector(self.V_cur[p])
            p2d = view3d_utils.location_3d_to_region_2d(self.region, self.rv3d, world)
            if p2d is None:
                continue
            d = (Vector(coord) - p2d).length
            if d < best_d:
                best, best_d = k, d
        return best

    def _drag_update(self, coord):
        mw = self.ob.matrix_world
        cur_world = mw @ Vector(self.pin_pos[self.grab])
        target_world = view3d_utils.region_2d_to_location_3d(
            self.region, self.rv3d, coord, cur_world)
        target_local = mw.inverted() @ target_world
        self.pin_pos[self.grab] = np.array(target_local, dtype=np.float64)

        # warm-start from the server's copy of the previous result (identical
        # to self.V_cur) so only the pin targets travel over the pipe
        self.V_cur = self.solver.solve(
            pin_pos=self.pin_pos, warm_last=True,
            iterations=self.drag_iters, admm_iters=self.admm_iters)
        write_mesh_positions(self.ob.data, self.V_cur)
        self.area.tag_redraw()

    def _exit(self, context):
        bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
        context.workspace.status_text_set(None)
        self.area.tag_redraw()
        self.solver.close()

    # ---- modal loop

    def modal(self, context, event):
        # let viewport navigation through
        if (event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                           'TRACKPADPAN', 'TRACKPADZOOM'}
                or event.type.startswith('NUMPAD')):
            return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.grab = self._pick_pin(self._mouse_region_coords(event))
                self.area.tag_redraw()
            elif event.value == 'RELEASE':
                self.grab = None
                self.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE':
            if self.grab is not None:
                try:
                    self._drag_update(self._mouse_region_coords(event))
                except Exception as exc:
                    self.report({'ERROR'}, f"Solve failed ({exc})")
                    write_mesh_positions(self.ob.data, self.V_orig)
                    self._exit(context)
                    return {'CANCELLED'}
            return {'RUNNING_MODAL'}

        if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            self._exit(context)
            return {'FINISHED'}

        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            write_mesh_positions(self.ob.data, self.V_orig)
            self._exit(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}


# ================== Panel

class VIEW3D_PT_cubify_cpp(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Cubify C++"
    bl_label = "Cubic Stylization (C++)"

    def draw(self, context):
        layout = self.layout
        props = context.scene.cubify_cpp_settings
        ob = context.active_object

        if not os.path.isfile(server_path(context)):
            box = layout.box()
            box.label(text="C++ server not built", icon='ERROR')
            op = box.operator("preferences.addon_show",
                              text="Build in Add-on Preferences",
                              icon='PREFERENCES')
            op.module = __package__

        if ob is not None and ob.type == 'MESH' and ob.data is not None:
            me = ob.data
            kind, tris, quads, ngons = mesh_kind(me)
            box = layout.box()
            box.label(text=f"{ob.name}: {len(me.vertices)} verts", icon='MESH_DATA')
            box.label(text=f"{kind} ({tris} tris, {quads} quads"
                           + (f", {ngons} n-gons)" if ngons else ")"),
                      icon='MESH_GRID' if quads and not tris else 'MESH_ICOSPHERE')
            if quads or ngons:
                box.label(text="Solved on virtual triangulation;", icon='INFO')
                box.label(text="topology and UVs are preserved")
            n_pins = (len(get_pin_indices(ob)) if len(me.vertices) <= 100000
                      else None)
            if n_pins is None:
                box.label(text=f"Pins: group '{PIN_GROUP}'", icon='PINNED')
            else:
                box.label(text=f"Pins: {n_pins}", icon='PINNED')

        col = layout.column(align=True)
        col.prop(props, "cubeness")
        col.prop(props, "iterations")
        col.prop(props, "admm_iterations")
        col.prop(props, "repeat")
        row = layout.row(align=True)
        row.prop(props, "square_flat")
        sub = row.row(align=True)
        sub.active = props.square_flat
        sub.prop(props, "square_flat_strength")
        layout.prop(props, "threads")
        layout.prop(props, "orientation")
        layout.prop(props, "apply_to_copy")
        layout.operator(OBJECT_OT_cubify_cpp.bl_idname, icon='MESH_CUBE')
        row = layout.row(align=True)
        row.prop(props, "auto_fix_walls")
        row.operator(OBJECT_OT_cubify_cpp_fix_thin_walls.bl_idname, text="Fix Now",
                     icon='MOD_SOLIDIFY')

        layout.separator()
        box = layout.box()
        box.label(text="Style as Process", icon='SHAPEKEY_DATA')
        row = box.row(align=True)
        row.prop(props, "anim_mode", expand=True)
        col = box.column(align=True)
        col.prop(props, "anim_samples")
        if props.anim_mode == 'LAMBDA':
            col.prop(props, "anim_iters")
        col.prop(props, "anim_frame_step")
        box.operator(OBJECT_OT_cubify_cpp_bake_anim.bl_idname, icon='RENDER_ANIMATION')
        box.prop(props, "anim_axes")
        box.operator(OBJECT_OT_cubify_cpp_bake_frames.bl_idname, icon='DUPLICATE')

        layout.separator()
        box = layout.box()
        box.label(text="ARAP Manipulation", icon='VIEW_PAN')
        row = box.row(align=True)
        op = row.operator(OBJECT_OT_cubify_cpp_pins.bl_idname, text="Set Pins")
        op.action = 'SET'
        op = row.operator(OBJECT_OT_cubify_cpp_pins.bl_idname, text="Add")
        op.action = 'ADD'
        op = row.operator(OBJECT_OT_cubify_cpp_pins.bl_idname, text="Clear")
        op.action = 'CLEAR'
        box.prop(props, "stylized_drag")
        box.prop(props, "drag_iterations")
        box.operator(OBJECT_OT_arap_cpp_manipulate.bl_idname, icon='ORIENTATION_GIMBAL')


# ================== Preferences: server status and one-click build

# Build state shared between the worker thread, the watcher timer and the
# preferences UI. Only the worker writes ok/msg; the UI only reads.
_BUILD = {"thread": None, "ok": None, "msg": ""}


def _build_worker():
    """Runs in a background thread so Blender stays responsive during the
    build (the first one also downloads Eigen). No bpy access here."""
    try:
        ok, msg = builder.build(log=print)  # full CMake log to the system console
        _BUILD["ok"], _BUILD["msg"] = ok, msg
    except Exception as exc:
        _BUILD["ok"], _BUILD["msg"] = False, f"build failed: {exc}"


def _redraw_preferences():
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'PREFERENCES':
                    area.tag_redraw()
    except Exception:
        pass


def _watch_build():
    """bpy.app.timers callback: poll the worker, then probe the result."""
    thread = _BUILD["thread"]
    if thread is not None and thread.is_alive():
        _redraw_preferences()
        return 0.5
    _BUILD["thread"] = None
    if _BUILD["ok"]:
        try:
            server = client.get_server(server_path())
            _BUILD["msg"] = f"Built — {server.version}"
        except Exception as exc:
            _BUILD["ok"] = False
            _BUILD["msg"] = f"built, but the server does not start: {exc}"
    _redraw_preferences()
    return None


class CUBIFY_CPP_OT_check_server(bpy.types.Operator):
    """Start the C++ server (restarting it if running) and report its version"""
    bl_idname = "cubify_cpp.check_server"
    bl_label = "Check Server"

    def execute(self, context):
        client.shutdown_server()
        try:
            server = client.get_server(server_path(context))
            _BUILD.update(ok=True, msg=f"{server.version} — "
                          f"{server.hardware_threads} hardware threads")
        except Exception as exc:
            _BUILD.update(ok=False, msg=str(exc))
        return {'FINISHED'}


class CUBIFY_CPP_OT_build_server(bpy.types.Operator):
    """Compile the C++ server from the bundled source with CMake. Needs CMake
    and a C++17 compiler; the first build downloads Eigen. Runs in the
    background; Blender stays usable"""
    bl_idname = "cubify_cpp.build_server"
    bl_label = "Build Server"

    def execute(self, context):
        if _BUILD["thread"] is not None:
            self.report({'WARNING'}, "Build already running")
            return {'CANCELLED'}
        # a running server locks its executable on Windows
        client.shutdown_server()
        _BUILD.update(ok=None, msg="Building… (progress in the system console)")
        thread = threading.Thread(target=_build_worker, daemon=True)
        _BUILD["thread"] = thread
        thread.start()
        bpy.app.timers.register(_watch_build, first_interval=0.5)
        self.report({'INFO'}, "C++ server build started in the background")
        return {'FINISHED'}


class CubifyCppPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    server_path: bpy.props.StringProperty(
        name="Server Executable",
        description="Path to a cubify_server executable. Leave empty to use "
                    "the one built into the add-on's bin folder",
        subtype='FILE_PATH', default="")

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="C++ solver server", icon='SYSTEM')

        building = _BUILD["thread"] is not None
        path = client.resolve_server_path(self.server_path)
        running = client.server_cached()
        if building:
            box.label(text=_BUILD["msg"], icon='SORTTIME')
        elif _BUILD["msg"]:
            icon = 'ERROR' if _BUILD["ok"] is False else 'CHECKMARK'
            box.label(text=_BUILD["msg"], icon=icon)
        elif running is not None:
            box.label(text=f"Running — {running.version}", icon='CHECKMARK')
        elif os.path.isfile(path):
            box.label(text="Server built (starts on first use)", icon='CHECKMARK')
        else:
            box.label(text="Server not built yet", icon='ERROR')

        box.prop(self, "server_path")
        box.label(text="Using: " + path)

        row = box.row(align=True)
        row.enabled = not building
        row.operator(CUBIFY_CPP_OT_check_server.bl_idname, icon='VIEWZOOM')
        row.operator(CUBIFY_CPP_OT_build_server.bl_idname, icon='TOOL_SETTINGS')

        cmake = builder.find_cmake()
        box.label(text="CMake: " + (cmake or "not found — install CMake and a "
                                     "C++17 compiler to build"))
        box.label(text="Terminal alternative: python builder.py (in the add-on folder)")


# ================== Registration

_classes = (CubifyCppSettings, OBJECT_OT_cubify_cpp, OBJECT_OT_cubify_cpp_fix_thin_walls,
            OBJECT_OT_cubify_cpp_bake_anim,
            OBJECT_OT_cubify_cpp_bake_frames, OBJECT_OT_cubify_cpp_pins,
            OBJECT_OT_arap_cpp_manipulate, VIEW3D_PT_cubify_cpp,
            CUBIFY_CPP_OT_check_server, CUBIFY_CPP_OT_build_server,
            CubifyCppPreferences)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.cubify_cpp_settings = bpy.props.PointerProperty(type=CubifyCppSettings)


def unregister():
    client.shutdown_server()
    del bpy.types.Scene.cubify_cpp_settings
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
