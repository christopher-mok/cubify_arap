# Style as Process bake (both add-ons): run headless from the repo root:
#   blender --background --python-exit-code 1 --python tests/test_style_bake.py
import sys
sys.path.insert(0, '.')
import importlib.util
import numpy as np
import bpy

import cubic_stylization
cubic_stylization.register()
spec = importlib.util.spec_from_file_location("cubic_stylization_cpp", "cubic_stylization_c++/__init__.py")
cpp = importlib.util.module_from_spec(spec)
sys.modules["cubic_stylization_cpp"] = cpp
spec.loader.exec_module(cpp)
cpp.register()

scene = bpy.context.scene
ok = True


def check(cond, msg):
    global ok
    print(("  ok   " if cond else "  FAIL ") + msg)
    ok &= bool(cond)


def coords(me):
    x = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", x)
    return x


for label, op, settings in [("python", "cubify_bake_animation", "cubify_settings"),
                            ("cpp", "cubify_cpp_bake_animation", "cubify_cpp_settings")]:
    props = getattr(scene, settings)
    props.cubeness, props.anim_samples, props.anim_frame_step = 0.5, 8, 2
    for mode in ("LAMBDA", "ITER"):
        for target in ("CUBE", "OCTAHEDRON"):
            props.anim_mode, props.target_shape = mode, target
            scene.frame_set(1)
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3)
            ob = bpy.context.active_object
            me = ob.data
            co0 = coords(me)
            uv0 = np.empty(len(me.uv_layers.active.data) * 2)
            me.uv_layers.active.data.foreach_get("uv", uv0)
            check(getattr(bpy.ops.object, op)() == {'FINISHED'}, f"{label} {mode} {target}: bake runs")
            key = me.shape_keys
            kbs = key.key_blocks
            uv1 = np.empty_like(uv0)
            me.uv_layers.active.data.foreach_get("uv", uv1)
            first, last = np.empty_like(co0), np.empty_like(co0)
            kbs[0].data.foreach_get("co", first)
            kbs[-1].data.foreach_get("co", last)
            fc = [f for f in key.animation_data.action.fcurves if f.data_path == "eval_time"]
            check(len(kbs) == props.anim_samples + 1 and not key.use_relative
                  and np.array_equal(coords(me), co0) and np.array_equal(uv0, uv1)
                  and np.allclose(first, co0, atol=1e-6) and np.abs(last - first).max() > 1e-3
                  and len(fc) == 1 and len(fc[0].keyframe_points) == 2,
                  f"{label} {mode} {target}: {len(kbs)} absolute keys, base mesh + UVs untouched, "
                  f"last key moved {np.abs(last - first).max():.3f}")
            dg = bpy.context.evaluated_depsgraph_get()
            f0, f1 = (int(k.co[0]) for k in fc[0].keyframe_points)
            seen = []
            for f in (f0, (f0 + f1) // 2, f1):
                scene.frame_set(f)
                dg.update()
                seen.append(coords(ob.evaluated_get(dg).data))
            check(np.allclose(seen[0], co0, atol=1e-5) and np.abs(seen[1] - seen[0]).max() > 1e-4
                  and np.allclose(seen[2], last, atol=1e-5),
                  f"{label} {mode} {target}: timeline plays original -> stylized")
            scene.frame_set(1)
    check(getattr(bpy.ops.object, op)() == {'CANCELLED'}, f"{label}: refuses a mesh that already has keys")

print("ALL-OK" if ok else "FAILED")
