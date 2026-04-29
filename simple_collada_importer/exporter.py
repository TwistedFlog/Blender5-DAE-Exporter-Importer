# SPDX-License-Identifier: GPL-3.0-or-later
"""COLLADA (.dae) exporter — meshes, materials, armatures, skin weights.

Writes a COLLADA 1.4.1 document that round-trips cleanly with the Simple
COLLADA importer (same sid / id conventions).
"""

import math
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

import bpy
from mathutils import Matrix, Vector


# ─────────────────────────── XML HELPERS ────────────────────────────────────

def _el(parent, tag, attrib=None, text=None):
    e = ET.SubElement(parent, tag, attrib or {})
    if text is not None:
        e.text = text
    return e


def _mat4_text(m):
    """Blender Matrix → COLLADA row-major float string."""
    rows = []
    for row in m:
        rows.extend(f"{v:.6g}" for v in row)
    return " ".join(rows)


def _floats(arr):
    return " ".join(f"{v:.6g}" for v in arr)


def _ints(arr):
    return " ".join(str(v) for v in arr)


def _prettify(root):
    raw = ET.tostring(root, encoding="unicode")
    dom = minidom.parseString(raw)
    return dom.toprettyxml(indent="  ", encoding="utf-8")


# ─────────────────────────── GEOMETRY WRITER ────────────────────────────────

def _gather_geometry_data(obj, global_matrix):
    """Evaluate *obj* once and collect the mesh data needed for COLLADA export."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()

    try:
        # corner_normals (Blender 4.1+) is the replacement for the removed
        # calc_normals_split() API. It gives the definitive per-loop normal for
        # every corner: custom split normals if baked in, otherwise the smooth
        # vertex normal or flat face normal — whichever applies.
        # calc_loop_triangles() triangulates in-place while preserving each loop's
        # index into mesh.loops / corner_normals so the right normal is always used.
        corner_normals = eval_mesh.corner_normals
        eval_mesh.calc_loop_triangles()

        world = global_matrix @ obj.matrix_world
        world_rot = world.to_3x3().normalized()

        positions = [world @ v.co for v in eval_mesh.vertices]

        has_uv = bool(eval_mesh.uv_layers)
        uv_layer = eval_mesh.uv_layers.active.data if has_uv else None

        # Build deduplicated normal and UV tables in one pass over triangles.
        # Deduplication keeps the float arrays proportional to unique values rather
        # than one entry per loop, which prevents the ~6x bloat on re-export cycles.
        unique_norms = []
        norm_to_idx = {}
        unique_uvs = []
        uv_to_idx = {}

        face_rows = []  # [(v0,n0,uv0, v1,n1,uv1, v2,n2,uv2), mat_idx]
        for tri in eval_mesh.loop_triangles:
            tri_data = []
            for loop_idx in tri.loops:
                loop = eval_mesh.loops[loop_idx]
                vert_idx = loop.vertex_index

                # corner_normals[loop_idx] gives the custom split normal, smooth
                # vertex normal, or flat face normal — whichever applies.
                n = (world_rot @ corner_normals[loop_idx].vector).normalized()
                nk = (round(n.x, 5), round(n.y, 5), round(n.z, 5))
                if nk not in norm_to_idx:
                    norm_to_idx[nk] = len(unique_norms)
                    unique_norms.append(n)
                ni = norm_to_idx[nk]

                if uv_layer:
                    uv = uv_layer[loop_idx].uv
                    uk = (round(uv.x, 5), round(uv.y, 5))
                    if uk not in uv_to_idx:
                        uv_to_idx[uk] = len(unique_uvs)
                        unique_uvs.append((uv.x, uv.y))
                    ui = uv_to_idx[uk]
                else:
                    ui = 0

                tri_data.extend([vert_idx, ni, ui])
            face_rows.append((tri_data, tri.material_index))

        return {
            "positions": positions,
            "normals": unique_norms,
            "uvs": unique_uvs,
            "face_rows": face_rows,
            "materials": obj.data.materials or [],
            "vertex_count": len(eval_mesh.vertices),
            "has_uv": has_uv,
            "world_mat": world,
        }
    finally:
        eval_obj.to_mesh_clear()


def _write_morph_target_geometry(lib_geom, geom_id, shape_key, vertex_count, world_mat, geom_name=None):
    """Write a positions-only <geometry> for a morph target shape key.

    Morph target geometries only need a positions source — normals, UVs, and
    face topology are never consumed from them by Unity's DAE importer, so
    writing the full geometry would duplicate that data identically for every
    shape key for no benefit.
    """
    positions = [world_mat @ shape_key.data[i].co for i in range(vertex_count)]
    n_pos = len(positions)

    geom_el = _el(lib_geom, "geometry", {"id": geom_id, "name": geom_name or geom_id})
    mesh_el = _el(geom_el, "mesh")

    pos_src_id = f"{geom_id}-positions"
    src = _el(mesh_el, "source", {"id": pos_src_id})
    _el(src, "float_array",
        {"id": f"{pos_src_id}-array", "count": str(n_pos * 3)},
        text=_floats(v[i] for v in positions for i in range(3)))
    tc = _el(src, "technique_common")
    acc = _el(tc, "accessor",
              {"source": f"#{pos_src_id}-array", "count": str(n_pos), "stride": "3"})
    for axis in ("X", "Y", "Z"):
        _el(acc, "param", {"name": axis, "type": "float"})

    verts_id = f"{geom_id}-vertices"
    verts_el = _el(mesh_el, "vertices", {"id": verts_id})
    _el(verts_el, "input", {"semantic": "POSITION", "source": f"#{pos_src_id}"})


def _write_geometry_from_data(lib_geom, obj, geom_id, geom_data, positions_override=None, geom_name=None):
    """Append a <geometry> element for *obj* into *lib_geom*."""
    world_positions = positions_override if positions_override is not None else geom_data["positions"]

    n_pos = len(world_positions)
    n_norm = len(geom_data["normals"])
    n_uv = len(geom_data["uvs"])

    # ---- XML structure ----
    geom_el = _el(lib_geom, "geometry", {"id": geom_id, "name": geom_name or geom_id})
    mesh_el = _el(geom_el, "mesh")

    # positions source
    pos_src_id = f"{geom_id}-positions"
    src = _el(mesh_el, "source", {"id": pos_src_id})
    fa = _el(src, "float_array",
             {"id": f"{pos_src_id}-array", "count": str(n_pos * 3)},
             text=_floats(v[i] for v in world_positions for i in range(3)))
    tc = _el(src, "technique_common")
    acc = _el(tc, "accessor",
              {"source": f"#{pos_src_id}-array", "count": str(n_pos), "stride": "3"})
    for axis in ("X", "Y", "Z"):
        _el(acc, "param", {"name": axis, "type": "float"})

    # normals source
    nrm_src_id = f"{geom_id}-normals"
    src = _el(mesh_el, "source", {"id": nrm_src_id})
    _el(src, "float_array",
        {"id": f"{nrm_src_id}-array", "count": str(n_norm * 3)},
        text=_floats(v[i] for v in geom_data["normals"] for i in range(3)))
    tc = _el(src, "technique_common")
    acc = _el(tc, "accessor",
              {"source": f"#{nrm_src_id}-array", "count": str(n_norm), "stride": "3"})
    for axis in ("X", "Y", "Z"):
        _el(acc, "param", {"name": axis, "type": "float"})

    # uv source (optional)
    uv_src_id = f"{geom_id}-texcoord"
    if geom_data["uvs"]:
        src = _el(mesh_el, "source", {"id": uv_src_id})
        _el(src, "float_array",
            {"id": f"{uv_src_id}-array", "count": str(n_uv * 2)},
            text=_floats(v[i] for v in geom_data["uvs"] for i in range(2)))
        tc = _el(src, "technique_common")
        acc = _el(tc, "accessor",
                  {"source": f"#{uv_src_id}-array", "count": str(n_uv), "stride": "2"})
        for axis in ("S", "T"):
            _el(acc, "param", {"name": axis, "type": "float"})

    # vertices element
    verts_id = f"{geom_id}-vertices"
    verts_el = _el(mesh_el, "vertices", {"id": verts_id})
    _el(verts_el, "input", {"semantic": "POSITION", "source": f"#{pos_src_id}"})

    # Group faces by material index
    mat_faces = {}
    for tri, mat_idx in geom_data["face_rows"]:
        mat_faces.setdefault(mat_idx, []).append(tri)

    for mat_idx, tris in sorted(mat_faces.items()):
        mat = geom_data["materials"][mat_idx] if (
            geom_data["materials"] and mat_idx < len(geom_data["materials"])
        ) else None
        mat_symbol = _safe_id(mat.name) if mat else "Material"

        tri_el = _el(mesh_el, "triangles",
                     {"count": str(len(tris)), "material": mat_symbol})
        _el(tri_el, "input",
            {"semantic": "VERTEX", "source": f"#{verts_id}", "offset": "0"})
        _el(tri_el, "input",
            {"semantic": "NORMAL", "source": f"#{nrm_src_id}", "offset": "1"})
        if geom_data["uvs"]:
            _el(tri_el, "input",
                {"semantic": "TEXCOORD", "source": f"#{uv_src_id}",
                 "offset": "2", "set": "0"})

        p_data = []
        for tri in tris:
            if geom_data["uvs"]:
                p_data.extend(tri)            # v, n, uv per corner
            else:
                # drop every third index (uv)
                for k in range(3):
                    p_data.extend(tri[k * 3: k * 3 + 2])
        _el(tri_el, "p", text=_ints(p_data))

    return geom_id


def _write_geometry(lib_geom, obj, geom_id, global_matrix, positions_override=None):
    """Append a <geometry> element for *obj* into *lib_geom*."""
    geom_data = _gather_geometry_data(obj, global_matrix)
    return _write_geometry_from_data(lib_geom, obj, geom_id, geom_data, positions_override=positions_override)

# ─────────────────────────── MATERIAL / EFFECT WRITER ───────────────────────

def _safe_id(name):
    """Turn a Blender name into a safe COLLADA id while preserving colons."""
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() or ch in "-_.:" else "_")
    return "".join(out) or "Material"


def _write_materials(root, ns, obj, dae_dir, export_textures, written_materials, written_effects):
    """Write <library_effects> and <library_materials> for *obj*'s materials.

    Returns a dict  mat_name → (effect_id, material_id, symbol).
    """
    lib_eff = root.find("library_effects")
    lib_mat = root.find("library_materials")

    result = {}
    for mat in (obj.data.materials or []):
        if mat is None:
            continue
        if mat.name in result:
            continue

        safe = _safe_id(mat.name)
        eff_id = f"{safe}-effect"
        mat_id = f"{safe}-material"
        symbol = safe

        if mat.name in written_materials:
            result[mat.name] = written_materials[mat.name]
            continue

        diff_color = (0.8, 0.8, 0.8, 1.0)
        diff_tex_path = None

        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    base = node.inputs.get("Base Color")
                    if base:
                        if base.is_linked:
                            from_node = base.links[0].from_node
                            if from_node.type == "TEX_IMAGE" and from_node.image:
                                img = from_node.image
                                raw = bpy.path.abspath(img.filepath)
                                if export_textures and os.path.isfile(raw):
                                    diff_tex_path = raw
                                else:
                                    diff_tex_path = raw
                        else:
                            diff_color = tuple(base.default_value)
                    break
        else:
            diff_color = (*mat.diffuse_color[:3], 1.0)

        if eff_id not in written_effects:
            eff_el = _el(lib_eff, "effect", {"id": eff_id})
            prof = _el(eff_el, "profile_COMMON")

            tex_ref = None
            if diff_tex_path:
                img_id = f"{safe}-image"
                try:
                    rel = os.path.relpath(diff_tex_path, dae_dir)
                except ValueError:
                    rel = diff_tex_path
                rel = rel.replace("\\", "/")

                surf_sid = f"{safe}-surface"
                samp_sid = f"{safe}-sampler"

                np_el = _el(prof, "newparam", {"sid": surf_sid})
                surf = _el(np_el, "surface", {"type": "2D"})
                _el(surf, "init_from", text=img_id)

                np_el = _el(prof, "newparam", {"sid": samp_sid})
                samp = _el(np_el, "sampler2D")
                _el(samp, "source", text=surf_sid)

                tex_ref = samp_sid
                mat_entry = (eff_id, mat_id, symbol, img_id, rel)
            else:
                mat_entry = (eff_id, mat_id, symbol, None, None)

            tech = _el(prof, "technique", {"sid": "common"})
            lambert = _el(tech, "lambert")
            emission = _el(lambert, "emission")
            _el(emission, "color", {"sid": "emission"}, text="0 0 0 1")
            diffuse_el = _el(lambert, "diffuse")
            if tex_ref:
                _el(diffuse_el, "texture", {"texture": tex_ref, "texcoord": "UVMap"})
            else:
                rgba = " ".join(f"{v:.6g}" for v in diff_color)
                _el(diffuse_el, "color", {"sid": "diffuse"}, text=rgba)

            written_effects[eff_id] = True
        else:
            if diff_tex_path:
                try:
                    rel = os.path.relpath(diff_tex_path, dae_dir)
                except ValueError:
                    rel = diff_tex_path
                rel = rel.replace("\\", "/")
                img_id = f"{safe}-image"
                mat_entry = (eff_id, mat_id, symbol, img_id, rel)
            else:
                mat_entry = (eff_id, mat_id, symbol, None, None)

        mat_el = _el(lib_mat, "material", {"id": mat_id, "name": mat.name})
        _el(mat_el, "instance_effect", {"url": f"#{eff_id}"})

        written_materials[mat.name] = mat_entry
        result[mat.name] = mat_entry

    return result

# ─────────────────────────── ARMATURE WRITER ────────────────────────────────

def _pose_bone_world(arm_obj, bone):
    """World-space matrix of *bone* in rest pose."""
    return arm_obj.matrix_world @ bone.matrix_local


def _write_joint_node(parent_el, arm_obj, bone, bone_node_map=None):
    """Recursively write <node type='JOINT'> for *bone* and its children."""
    safe_sid = _safe_id(bone.name)
    node_id = f"Armature_{safe_sid}"

    node_el = _el(parent_el, "node",
                  {"id": node_id, "name": bone.name,
                   "sid": safe_sid, "type": "JOINT"})
    if bone_node_map is not None:
        bone_node_map[(arm_obj, bone.name)] = node_el

    # Local transform relative to parent (or armature root)
    if bone.parent:
        parent_world = _pose_bone_world(arm_obj, bone.parent)
        local = parent_world.inverted() @ _pose_bone_world(arm_obj, bone)
    else:
        local = arm_obj.matrix_world.inverted() @ _pose_bone_world(arm_obj, bone)

    _el(node_el, "matrix", {"sid": "transform"}, text=_mat4_text(local))

    for child in bone.children:
        _write_joint_node(node_el, arm_obj, child, bone_node_map)


# ─────────────────────────── SKIN CONTROLLER WRITER ─────────────────────────

def _write_controller(lib_ctrl, obj, arm_obj, geom_id, ctrl_id):
    """Write a <controller><skin> for *obj* skinned to *arm_obj*.

    Returns list of joint sid names used, or [] if no skin data.
    """
    if arm_obj is None:
        return []

    arm_mod = next(
        (m for m in obj.modifiers if m.type == "ARMATURE" and m.object == arm_obj),
        None,
    )
    if arm_mod is None:
        return []

    bones = arm_obj.data.bones
    bone_names = [b.name for b in bones]
    if not bone_names:
        return []

    # Compute inverse bind matrices (world-space rest pose, inverted)
    inv_binds = []
    for bone in bones:
        world_mat = _pose_bone_world(arm_obj, bone)
        try:
            inv = world_mat.inverted()
        except Exception:
            inv = Matrix.Identity(4)
        inv_binds.append(inv)

    # Gather vertex weights per bone
    vg_index = {vg.name: vg.index for vg in obj.vertex_groups}
    bone_vg_indices = [vg_index.get(b.name, -1) for b in bones]

    mesh = obj.data
    # vertex index → [(bone_idx, weight), ...]
    vert_weights = {v.index: [] for v in mesh.vertices}
    for v in mesh.vertices:
        for ge in v.groups:
            for bi, bvi in enumerate(bone_vg_indices):
                if bvi == ge.group and ge.weight > 0.0:
                    vert_weights[v.index].append((bi, ge.weight))

    # Collect flat weight list
    weight_list = []
    weight_index = {}
    for vidx, pairs in vert_weights.items():
        for bi, w in pairs:
            key = round(w, 6)
            if key not in weight_index:
                weight_index[key] = len(weight_list)
                weight_list.append(w)

    # Build vcount / v arrays
    vcounts = []
    v_data = []
    for v in mesh.vertices:
        pairs = vert_weights.get(v.index, [])
        vcounts.append(len(pairs))
        for bi, w in pairs:
            v_data.append(bi)
            v_data.append(weight_index[round(w, 6)])

    # ── XML ──────────────────────────────────────────────────────────────
    ctrl_el = _el(lib_ctrl, "controller", {"id": ctrl_id, "name": ctrl_id})
    skin_el = _el(ctrl_el, "skin", {"source": f"#{geom_id}"})

    # bind_shape_matrix (identity — mesh verts already in world space)
    _el(skin_el, "bind_shape_matrix",
        text="1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1")

    n_bones = len(bone_names)

    # joints source
    j_src_id = f"{ctrl_id}-joints"
    src = _el(skin_el, "source", {"id": j_src_id})
    _el(src, "Name_array",
        {"id": f"{j_src_id}-array", "count": str(n_bones)},
        text=" ".join(_safe_id(b) for b in bone_names))
    tc = _el(src, "technique_common")
    acc = _el(tc, "accessor",
              {"source": f"#{j_src_id}-array",
               "count": str(n_bones), "stride": "1"})
    _el(acc, "param", {"name": "JOINT", "type": "name"})

    # inv_bind_matrix source
    ibm_src_id = f"{ctrl_id}-bind_poses"
    src = _el(skin_el, "source", {"id": ibm_src_id})
    ibm_floats = []
    for inv in inv_binds:
        for row in inv:
            ibm_floats.extend(row)
    _el(src, "float_array",
        {"id": f"{ibm_src_id}-array", "count": str(n_bones * 16)},
        text=_floats(ibm_floats))
    tc = _el(src, "technique_common")
    acc = _el(tc, "accessor",
              {"source": f"#{ibm_src_id}-array",
               "count": str(n_bones), "stride": "16"})
    _el(acc, "param", {"name": "TRANSFORM", "type": "float4x4"})

    # weights source
    w_src_id = f"{ctrl_id}-weights"
    src = _el(skin_el, "source", {"id": w_src_id})
    _el(src, "float_array",
        {"id": f"{w_src_id}-array", "count": str(len(weight_list))},
        text=_floats(weight_list))
    tc = _el(src, "technique_common")
    acc = _el(tc, "accessor",
              {"source": f"#{w_src_id}-array",
               "count": str(len(weight_list)), "stride": "1"})
    _el(acc, "param", {"name": "WEIGHT", "type": "float"})

    # joints element
    joints_el = _el(skin_el, "joints")
    _el(joints_el, "input",
        {"semantic": "JOINT", "source": f"#{j_src_id}"})
    _el(joints_el, "input",
        {"semantic": "INV_BIND_MATRIX", "source": f"#{ibm_src_id}"})

    # vertex_weights element
    n_verts = len(mesh.vertices)
    vw_el = _el(skin_el, "vertex_weights",
                {"count": str(n_verts)})
    _el(vw_el, "input",
        {"semantic": "JOINT", "source": f"#{j_src_id}", "offset": "0"})
    _el(vw_el, "input",
        {"semantic": "WEIGHT", "source": f"#{w_src_id}", "offset": "1"})
    _el(vw_el, "vcount", text=_ints(vcounts))
    _el(vw_el, "v", text=_ints(v_data))

    return [_safe_id(b) for b in bone_names]

def _write_morph_controller(lib_ctrl, source_id, ctrl_id, target_geom_ids, target_names, weights):
    """Write a COLLADA morph controller for a mesh's shape keys."""
    ctrl_el = _el(lib_ctrl, "controller", {"id": ctrl_id, "name": ctrl_id})
    morph_el = _el(ctrl_el, "morph", {"source": f"#{source_id}", "method": "NORMALIZED"})

    tgt_src_id = f"{ctrl_id}-targets"
    src = _el(morph_el, "source", {"id": tgt_src_id})
    _el(src, "IDREF_array",
        {"id": f"{tgt_src_id}-array", "count": str(len(target_geom_ids))},
        text=" ".join(target_geom_ids))
    tc = _el(src, "technique_common")
    acc = _el(tc, "accessor",
              {"source": f"#{tgt_src_id}-array", "count": str(len(target_geom_ids)), "stride": "1"})
    _el(acc, "param", {"name": "MORPH_TARGET", "type": "IDREF"})

    w_src_id = f"{ctrl_id}-weights"
    src = _el(morph_el, "source", {"id": w_src_id})
    _el(src, "float_array",
        {"id": f"{w_src_id}-array", "count": str(len(weights))},
        text=_floats(weights))
    tc = _el(src, "technique_common")
    acc = _el(tc, "accessor",
              {"source": f"#{w_src_id}-array", "count": str(len(weights)), "stride": "1"})
    _el(acc, "param", {"name": "MORPH_WEIGHT", "type": "float"})

    targets_el = _el(morph_el, "targets")
    _el(targets_el, "input", {"semantic": "MORPH_TARGET", "source": f"#{tgt_src_id}"})
    _el(targets_el, "input", {"semantic": "MORPH_WEIGHT", "source": f"#{w_src_id}"})

    return ctrl_id



def _write_empty_node(parent_el, obj, global_matrix, empty_node_map, empty_world_map,
                      arm_obj=None, bone_node_map=None):
    """Recursively write EMPTY objects as COLLADA nodes preserving hierarchy."""
    world = global_matrix @ obj.matrix_world

    parent_el_for_node = parent_el
    parent_world = None

    if obj.parent is not None:
        if obj.parent.type == "EMPTY" and obj.parent in empty_node_map:
            parent_el_for_node = empty_node_map[obj.parent]
            parent_world = empty_world_map[obj.parent]
        elif (
            obj.parent.type == "ARMATURE"
            and obj.parent_type == "BONE"
            and bone_node_map is not None
        ):
            parent_el_for_node = bone_node_map.get((obj.parent, obj.parent_bone), parent_el)
            if obj.parent_bone:
                try:
                    bone = obj.parent.data.bones.get(obj.parent_bone)
                    if bone is not None:
                        parent_world = global_matrix @ _pose_bone_world(obj.parent, bone)
                except Exception:
                    parent_world = None
        elif obj.parent.type == "ARMATURE":
            parent_world = global_matrix @ obj.parent.matrix_world

    if parent_world is None:
        local = world
    else:
        try:
            local = parent_world.inverted() @ world
        except Exception:
            local = world

    safe_name = _safe_id(obj.name)
    node_el = _el(parent_el_for_node, "node",
                  {"id": safe_name, "name": obj.name, "type": "NODE"})
    _el(node_el, "matrix", {"sid": "transform"}, text=_mat4_text(local))

    empty_node_map[obj] = node_el
    empty_world_map[obj] = world

    for child in obj.children:
        if child.type == "EMPTY":
            _write_empty_node(node_el, child, global_matrix, empty_node_map,
                              empty_world_map, arm_obj=arm_obj,
                              bone_node_map=bone_node_map)
# ─────────────────────────── TOP-LEVEL EXPORT ────────────────────────────────

def export_dae(
    filepath,
    context,
    selected_only=False,
    export_rig=True,
    export_textures=True,
    global_scale=1.0,
    forward_axis="-Y",
):
    """Export the scene (or selection) to a COLLADA .dae file.

    Returns ``(exported_count, error_msg_or_None)``.
    """
    from bpy_extras.io_utils import axis_conversion

    if forward_axis == "-Y":
        global_matrix = Matrix.Scale(global_scale, 4)
    else:
        global_matrix = (
            Matrix.Scale(global_scale, 4)
            @ axis_conversion(
                from_forward="-Y", from_up="Z",
                to_forward=forward_axis, to_up="Z",
            ).to_4x4()
        )

    dae_dir = os.path.dirname(os.path.abspath(filepath))

    objects = (
        [o for o in context.selected_objects]
        if selected_only
        else list(context.scene.objects)
    )
    mesh_objects = [o for o in objects if o.type == "MESH"]
    arm_objects  = [o for o in objects if o.type == "ARMATURE"]
    empty_objects = [o for o in objects if o.type == "EMPTY"]

    if not mesh_objects and not arm_objects and not empty_objects:
        return 0, "No mesh, armature, or empty objects to export"

    # ── Build XML skeleton ────────────────────────────────────────────────
    root = ET.Element("COLLADA", {
        "xmlns": "http://www.collada.org/2005/11/COLLADASchema",
        "version": "1.4.1",
    })

    # <asset>
    asset = _el(root, "asset")
    contrib = _el(asset, "contributor")
    _el(contrib, "authoring_tool", text="Simple COLLADA Exporter (Blender)")
    _el(asset, "up_axis", text="Z_UP")

    lib_img  = _el(root, "library_images")
    lib_eff  = _el(root, "library_effects")
    lib_mat  = _el(root, "library_materials")
    lib_geom = _el(root, "library_geometries")
    lib_ctrl = _el(root, "library_controllers")
    lib_vs   = _el(root, "library_visual_scenes")

    vs_el    = _el(lib_vs, "visual_scene", {"id": "Scene", "name": "Scene"})

    # Track written images to avoid duplicates
    written_images = {}
    written_materials = {}
    written_effects = {}

    exported = 0

    # ── Armatures first (joints live inside the scene node) ───────────────
    arm_node_ids = {}   # arm_obj → node id in visual scene
    arm_node_map = {}
    bone_node_map = {}
    for arm_obj in arm_objects:
        if not export_rig:
            break
        arm_node_id = _safe_id(arm_obj.name)
        arm_node = _el(vs_el, "node",
                       {"id": arm_node_id, "name": arm_obj.name, "type": "NODE"})
        local = global_matrix @ arm_obj.matrix_world
        _el(arm_node, "matrix", {"sid": "transform"},
            text=_mat4_text(local))

        arm_node_map[arm_obj] = arm_node
        for bone in arm_obj.data.bones:
            if bone.parent is None:
                _write_joint_node(arm_node, arm_obj, bone, bone_node_map)

        arm_node_ids[arm_obj] = arm_node_id
        exported += 1

    # ── Empties ───────────────────────────────────────────────────────────
    empty_set = set(empty_objects)
    empty_node_map = {}
    empty_world_map = {}

    def _empty_roots():
        roots = []
        for obj in empty_objects:
            parent = obj.parent
            if parent is None or parent.type != "EMPTY" or parent not in empty_set:
                roots.append(obj)
        return roots

    for obj in _empty_roots():
        parent_el = vs_el
        if obj.parent is not None:
            if obj.parent.type == "EMPTY" and obj.parent in empty_node_map:
                parent_el = empty_node_map[obj.parent]
            elif (
                obj.parent.type == "ARMATURE"
                and obj.parent_type == "BONE"
                and bone_node_map
            ):
                parent_el = bone_node_map.get((obj.parent, obj.parent_bone), vs_el)
            elif obj.parent.type == "ARMATURE" and obj.parent in arm_node_map:
                parent_el = arm_node_map[obj.parent]
        _write_empty_node(parent_el, obj, global_matrix, empty_node_map,
                          empty_world_map, bone_node_map=bone_node_map)
        exported += 1

    # ── Meshes ────────────────────────────────────────────────────────────
    for obj in mesh_objects:
        safe_name = _safe_id(obj.name)
        geom_id   = f"{safe_name}-mesh"
        ctrl_id   = f"{safe_name}-skin"

        # Find parent armature (if any and if we exported it)
        arm_obj = None
        if export_rig:
            for mod in obj.modifiers:
                if mod.type == "ARMATURE" and mod.object in arm_node_ids:
                    arm_obj = mod.object
                    break

        shape_keys = getattr(obj.data, "shape_keys", None)
        basis_key = None
        morph_keys = []
        if shape_keys is not None and getattr(shape_keys, "key_blocks", None):
            key_blocks = list(shape_keys.key_blocks)
            if key_blocks:
                basis_key = key_blocks[0]
                morph_keys = [kb for kb in key_blocks[1:] if not kb.mute]

        # Evaluate the geometry once and reuse it for the base mesh and every
        # morph target. This avoids paying the full depsgraph/to_mesh cost for
        # each key block.
        geom_data = _gather_geometry_data(obj, global_matrix)

        # Keep only shape keys that actually move vertices. Blender files can
        # sometimes carry shape key datablocks that are effectively identical to
        # the Basis key; those should not force morph export.
        real_morph_keys = list(morph_keys)
        if morph_keys and basis_key is not None:
            basis_count = len(basis_key.data)
            if basis_count:
                real_morph_keys = []
                for kb in morph_keys:
                    if len(kb.data) != basis_count:
                        real_morph_keys.append(kb)
                        continue
                    is_different = False
                    for i in range(basis_count):
                        if (kb.data[i].co - basis_key.data[i].co).length_squared > 1e-12:
                            is_different = True
                            break
                    if is_different:
                        real_morph_keys.append(kb)

        has_morph = bool(real_morph_keys)
        morph_ctrl_id = f"{safe_name}-morph" if has_morph else None

        # Write controller (skin) early so morph controllers can reference it.
        has_skin = False
        if arm_obj is not None:
            skin_source_id = morph_ctrl_id if morph_ctrl_id is not None else geom_id
            joint_names = _write_controller(lib_ctrl, obj, arm_obj, skin_source_id, ctrl_id)
            has_skin = bool(joint_names)

        # Write geometry
        _write_geometry_from_data(lib_geom, obj, geom_id, geom_data, geom_name=obj.name)

        if has_morph:
            target_geom_ids = []
            target_names = []
            target_weights = []
            vertex_count = geom_data["vertex_count"]
            world = geom_data["world_mat"]
            for kb in real_morph_keys:
                target_geom_id = f"{safe_name}-morph-{_safe_id(kb.name)}"
                if len(kb.data) != vertex_count:
                    continue
                # Write a positions-only geometry for this morph target.
                # Normals, UVs, and face topology are never read by Unity's DAE
                # importer from morph target geometries, so writing the full
                # geometry (as _write_geometry_from_data does) just bloats the
                # file with identical duplicated data for every shape key.
                _write_morph_target_geometry(
                    lib_geom, target_geom_id, kb, vertex_count, world, geom_name=kb.name,
                )
                target_geom_ids.append(target_geom_id)
                target_names.append(kb.name)
                target_weights.append(float(kb.value))
            if target_geom_ids:
                _write_morph_controller(lib_ctrl, geom_id, morph_ctrl_id, target_geom_ids, target_names, target_weights)
                has_morph = True
            else:
                has_morph = False

        # Write materials
        mat_info = _write_materials(root, "ns", obj, dae_dir, export_textures, written_materials, written_effects)

        # Register images
        for mat_name, info in mat_info.items():
            eff_id, mat_id, symbol, img_id, img_rel = info
            if img_id and img_id not in written_images:
                img_el = _el(lib_img, "image", {"id": img_id, "name": img_id})
                _el(img_el, "init_from", text=img_rel)
                written_images[img_id] = True

        # Write visual scene node
        mesh_node = _el(vs_el, "node",
                        {"id": safe_name, "name": obj.name, "type": "NODE"})

        if has_skin:
            # Instance the skin controller. When morphs are also present the
            # skin controller's source is already the morph controller
            # (geometry → morph → skin), so instancing the skin is enough to
            # activate the full chain. Instancing only the morph here would
            # silently drop all skinning data.
            ic_el = _el(mesh_node, "instance_controller",
                        {"url": f"#{ctrl_id}"})
            # skeleton root
            arm_bones = arm_obj.data.bones
            roots = [b for b in arm_bones if b.parent is None]
            for root_bone in roots:
                _el(ic_el, "skeleton",
                    text=f"#Armature_{_safe_id(root_bone.name)}")
            # bind_material
            if mat_info:
                bm_el = _el(ic_el, "bind_material")
                tc_el = _el(bm_el, "technique_common")
                for mat_name, info in mat_info.items():
                    eff_id, mat_id, symbol, img_id, img_rel = info
                    _el(tc_el, "instance_material",
                        {"symbol": symbol, "target": f"#{mat_id}"})
        elif morph_ctrl_id is not None:
            # Morph-only (no armature): instance the morph controller directly.
            ic_el = _el(mesh_node, "instance_controller",
                        {"url": f"#{morph_ctrl_id}"})
            if mat_info:
                bm_el = _el(ic_el, "bind_material")
                tc_el = _el(bm_el, "technique_common")
                for mat_name, info in mat_info.items():
                    eff_id, mat_id, symbol, img_id, img_rel = info
                    _el(tc_el, "instance_material",
                        {"symbol": symbol, "target": f"#{mat_id}"})
        else:
            ig_el = _el(mesh_node, "instance_geometry",
                        {"url": f"#{geom_id}"})
            if mat_info:
                bm_el = _el(ig_el, "bind_material")
                tc_el = _el(bm_el, "technique_common")
                for mat_name, info in mat_info.items():
                    eff_id, mat_id, symbol, img_id, img_rel = info
                    _el(tc_el, "instance_material",
                        {"symbol": symbol, "target": f"#{mat_id}"})

        exported += 1

    # ── scene instance ────────────────────────────────────────────────────
    scene_el = _el(root, "scene")
    _el(scene_el, "instance_visual_scene", {"url": "#Scene"})

    # ── Prune empty library sections ──────────────────────────────────────
    for lib in (lib_img, lib_ctrl):
        if len(lib) == 0:
            root.remove(lib)

    # ── Write file ────────────────────────────────────────────────────────
    try:
        os.makedirs(dae_dir, exist_ok=True)
        pretty = _prettify(root)
        with open(filepath, "wb") as f:
            f.write(pretty)
    except Exception as e:
        return exported, f"Failed to write file: {e}"

    return exported, None