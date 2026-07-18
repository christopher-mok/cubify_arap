bl_info = {
    "name": "Cubic Stylization",
    "author": "Christopher Mok",
    "version": (1, 1, 0),
    "blender": (2, 93, 0),
    "location": "3D Viewport > Sidebar (N) > Cubify",
    "description": "Cubify meshes (Liu & Jacobson 2019) and ARAP handle deformation. "
                   "Works on triangle and quad meshes; topology and UVs are preserved",
    "category": "Mesh",
}

import time

import bpy
import bmesh  # noqa: F401  (kept for users extending the addon)
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Euler, Vector
from bpy_extras import view3d_utils

if "solver" in locals():
    import importlib
    importlib.reload(solver)  # noqa: F821
from . import solver

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
    F = np.empty(m * 3, dtype=np.int64)
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


# ================== Settings

class CubifySettings(bpy.types.PropertyGroup):
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
    device: bpy.props.EnumProperty(
        name="Device",
        description="Where the solver runs. GPU devices need PyTorch installed "
                    "in Blender's Python (see the add-on README)",
        items=[
            ('AUTO', "Auto", "GPU for large meshes when available, else CPU"),
            ('CPU', "CPU", "numpy solver (always available)"),
            ('CUDA', "CUDA GPU", "NVIDIA GPU via PyTorch"),
            ('MPS', "Metal GPU (MPS)", "Apple GPU via PyTorch"),
        ],
        default='AUTO')


# ================== Cubify operator

class OBJECT_OT_cubify(bpy.types.Operator):
    """Apply Cubic Stylization to the selected mesh objects.
    Quad meshes are solved on a virtual triangulation: topology and UVs
    are preserved. Pinned vertices are held in place"""
    bl_idname = "object.cubify_mesh"
    bl_label = "Cubify Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (context.mode == 'OBJECT' and ob is not None and ob.type == 'MESH')

    def execute(self, context):
        props = context.scene.cubify_settings
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

        A = np.array(Euler(props.orientation, 'XYZ').to_matrix(), dtype=np.float64)

        wm = context.window_manager
        t0 = time.time()
        try:
            stylizer, device, warn = solver.create_stylizer(
                V, F, cubeness=props.cubeness, cube_axes=A, pins=pins,
                device=props.device)
            if warn:
                self.report({'WARNING'}, f"{ob.name}: {warn}")
            V_out = stylizer.run(
                iterations=props.iterations,
                admm_iters=props.admm_iterations,
                on_progress=lambda i, total: wm.progress_update(
                    int(100 * (base + span * i / total))))
        except Exception as exc:
            self.report({'ERROR'}, f"{ob.name}: solver failed ({exc})")
            return False

        if not np.all(np.isfinite(V_out)):
            self.report({'ERROR'}, f"{ob.name}: solver produced invalid positions")
            return False

        write_mesh_positions(me, V_out)

        note = f", {len(pins)} pinned" if pins else ""
        self.report({'INFO'},
                    f"{ob.name} ({kind.lower()}): cubified {len(V)} vertices in "
                    f"{time.time() - t0:.2f}s on {device} "
                    f"(lambda={props.cubeness:.2f}{note})")
        return True


# ================== Pin management

class OBJECT_OT_cubify_pins(bpy.types.Operator):
    """Manage the pinned (handle) vertices stored in the 'CubifyPins'
    vertex group"""
    bl_idname = "object.cubify_pins"
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

class OBJECT_OT_arap_manipulate(bpy.types.Operator):
    """Interactive ARAP deformation: drag pinned vertices and the mesh
    follows as-rigidly-as-possible. Click near a pin to grab it, drag to
    deform, Enter/Space to confirm, Esc/right-click to cancel"""
    bl_idname = "object.arap_manipulate"
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

        props = context.scene.cubify_settings
        V, F = read_mesh_arrays(me)
        lam = props.cubeness if props.stylized_drag else 0.0
        A = np.array(Euler(props.orientation, 'XYZ').to_matrix(), dtype=np.float64)

        try:
            self.solver, device, warn = solver.create_stylizer(
                V, F, cubeness=lam, cube_axes=A, pins=pins,
                device=props.device)
            if warn:
                self.report({'WARNING'}, warn)
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
            "ARAP Manipulation  |  Left-drag a pin to deform  |  "
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

        self.V_cur = self.solver.solve(
            pin_pos=self.pin_pos, V_init=self.V_cur,
            iterations=self.drag_iters, admm_iters=self.admm_iters)
        write_mesh_positions(self.ob.data, self.V_cur)
        self.area.tag_redraw()

    def _exit(self, context):
        bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
        context.workspace.status_text_set(None)
        self.area.tag_redraw()

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

class VIEW3D_PT_cubify(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Cubify"
    bl_label = "Cubic Stylization"

    def draw(self, context):
        layout = self.layout
        props = context.scene.cubify_settings
        ob = context.active_object

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
        layout.prop(props, "device")
        info = solver.torch_device_info_cached()
        if info is not None:
            has_torch, cuda, mps = info
            if props.device in {'CUDA', 'MPS'} and not has_torch:
                layout.label(text="PyTorch not installed: will use CPU",
                             icon='ERROR')
            elif props.device == 'CUDA' and not cuda:
                layout.label(text="CUDA not available: will use CPU",
                             icon='ERROR')
            elif props.device == 'MPS' and not mps:
                layout.label(text="MPS not available: will use CPU",
                             icon='ERROR')
        layout.prop(props, "orientation")
        layout.prop(props, "apply_to_copy")
        layout.operator(OBJECT_OT_cubify.bl_idname, icon='MESH_CUBE')

        layout.separator()
        box = layout.box()
        box.label(text="ARAP Manipulation", icon='VIEW_PAN')
        row = box.row(align=True)
        op = row.operator(OBJECT_OT_cubify_pins.bl_idname, text="Set Pins")
        op.action = 'SET'
        op = row.operator(OBJECT_OT_cubify_pins.bl_idname, text="Add")
        op.action = 'ADD'
        op = row.operator(OBJECT_OT_cubify_pins.bl_idname, text="Clear")
        op.action = 'CLEAR'
        box.prop(props, "stylized_drag")
        box.prop(props, "drag_iterations")
        box.operator(OBJECT_OT_arap_manipulate.bl_idname, icon='ORIENTATION_GIMBAL')


# ================== Registration

_classes = (CubifySettings, OBJECT_OT_cubify, OBJECT_OT_cubify_pins,
            OBJECT_OT_arap_manipulate, VIEW3D_PT_cubify)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.cubify_settings = bpy.props.PointerProperty(type=CubifySettings)


def unregister():
    del bpy.types.Scene.cubify_settings
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
