# SPDX-License-Identifier: GPL-3.0-or-later
"""Operators for the Blender COLLADA extension."""

import os

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    StringProperty,
)
from bpy.types import Operator, OperatorFileListElement
from bpy_extras.io_utils import ImportHelper, ExportHelper

from .importer import import_dae
from .exporter import export_dae
from .preferences import get_prefs


class IMPORT_OT_blender_collada_full(Operator, ImportHelper):
    """Import a COLLADA (.dae) mesh with full features"""

    bl_idname = "import_scene.blender_collada_full"
    bl_label = "Import Blender COLLADA (.dae)"
    bl_options = {"REGISTER", "UNDO", "PRESET"}
    filename_ext = ".dae"

    filter_glob: StringProperty(default="*.dae", options={"HIDDEN"})

    # Multi-file + drag-and-drop support
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN", "SKIP_SAVE"})
    files: CollectionProperty(
        type=OperatorFileListElement,
        options={"HIDDEN", "SKIP_SAVE"},
    )

    import_rig: BoolProperty(
        name="Import Rig",
        description="Import armature and skin weights if present in the DAE file",
        default=True,
    )

    split_by_material: BoolProperty(
        name="Split by Material",
        description=(
            "Split each imported geometry into one object per material so each "
            "piece appears separately in the Outliner"
        ),
        default=False,
    )

    use_default_material: BoolProperty(
        name="Use Blender Default Material",
        description=(
            "Ignore the DAE's diffuse/specular colors and textures and assign "
            "each material Blender's stock Principled BSDF defaults. Useful "
            "when a DAE has no usable materials or textures, to avoid "
            "splotchy chrome-like viewport shading"
        ),
        default=False,
    )

    recalculate_normals: BoolProperty(
        name="Recalculate Normals (Outside)",
        description=(
            "Discard any per-corner normals from the DAE and recompute "
            "consistent outward-facing normals after import. Fixes DAE files "
            "with inconsistent face winding that show up as dark / flipped "
            "patches on the model"
        ),
        default=False,
    )

    global_scale: FloatProperty(
        name="Scale",
        description="Uniform scale applied to imported geometry and transforms",
        default=1.0,
        min=1e-6,
        soft_min=0.001,
        soft_max=1000.0,
    )

    forward_axis: EnumProperty(
        name="Forward",
        description="Which DAE axis (after up-axis correction) maps to Blender's forward (-Y)",
        items=(
            ("-Y", "-Y Forward", ""),
            ("Y", "Y Forward", ""),
            ("-X", "-X Forward", ""),
            ("X", "X Forward", ""),
            ("-Z", "-Z Forward", ""),
            ("Z", "Z Forward", ""),
        ),
        default="-Y",
    )

    use_file_name_for_armature: BoolProperty(
        name="Use File Name for Armature",
        description=(
            "Use the imported .dae file name for the armature object name. "
            "When disabled, the importer falls back to the COLLADA visual scene name "
            "or id if available."
        ),
        default=True,
    )

    empty_display_type: EnumProperty(
        name="Empty Display",
        description="Display style used for imported empties",
        items=(
            ("SPHERE", "Sphere", ""),
            ("PLAIN_AXES", "Plain Axes", ""),
        ),
        default="SPHERE",
    )

    def invoke(self, context, event):
        # Seed defaults from add-on preferences so the user's chosen defaults
        # appear pre-selected in both the file dialog and drag-and-drop path.
        prefs = get_prefs(context)
        if prefs is not None:
            if not self.properties.is_property_set("import_rig"):
                self.import_rig = prefs.default_import_rig
            if not self.properties.is_property_set("split_by_material"):
                self.split_by_material = prefs.default_split_by_material
            if not self.properties.is_property_set("use_default_material"):
                self.use_default_material = prefs.default_use_default_material
            if not self.properties.is_property_set("recalculate_normals"):
                self.recalculate_normals = prefs.default_recalculate_normals
            if not self.properties.is_property_set("global_scale"):
                self.global_scale = prefs.default_global_scale
            if not self.properties.is_property_set("forward_axis"):
                self.forward_axis = prefs.default_forward_axis
            if not self.properties.is_property_set("use_file_name_for_armature"):
                self.use_file_name_for_armature = prefs.default_use_file_name_for_armature
            if not self.properties.is_property_set("empty_display_type"):
                self.empty_display_type = prefs.default_empty_display_type

        # Drag-and-drop populates `files`/`directory` directly; skip the dialog.
        if self.files:
            return self.execute(context)
        return super().invoke(context, event)

    def execute(self, context):
        # Build the list of paths to import.
        if self.files:
            paths = [
                os.path.join(self.directory, f.name)
                for f in self.files
                if f.name.lower().endswith(".dae")
            ]
        elif self.filepath:
            paths = [self.filepath]
        else:
            self.report({"ERROR"}, "No file given")
            return {"CANCELLED"}

        total_imported = 0
        last_arm = None
        errors = []

        # When importing more than one file, isolate each file's objects in
        # its own collection so the Outliner shows them as distinct groups.
        per_file_collections = len(paths) > 1
        scene_collection = context.scene.collection

        wm = context.window_manager
        wm.progress_begin(0, max(len(paths), 1))
        try:
            for i, path in enumerate(paths):
                wm.progress_update(i)
                target_coll = None
                if per_file_collections:
                    coll_name = os.path.splitext(os.path.basename(path))[0]
                    target_coll = bpy.data.collections.new(coll_name)
                    scene_collection.children.link(target_coll)
                count, arm_obj, err = import_dae(
                    path,
                    context,
                    import_rig=self.import_rig,
                    global_scale=self.global_scale,
                    forward_axis=self.forward_axis,
                    split_by_material=self.split_by_material,
                    use_default_material=self.use_default_material,
                    recalculate_normals=self.recalculate_normals,
                    use_file_name_for_armature=self.use_file_name_for_armature,
                    empty_display_type=self.empty_display_type,
                    target_collection=target_coll,
                    wm=wm if len(paths) == 1 else None,
                )
                if err:
                    errors.append(f"{os.path.basename(path)}: {err}")
                total_imported += count
                if arm_obj is not None:
                    last_arm = arm_obj
                # Drop empty per-file collection if nothing was imported.
                if (
                    target_coll is not None
                    and not target_coll.objects
                    and not target_coll.children
                ):
                    scene_collection.children.unlink(target_coll)
                    bpy.data.collections.remove(target_coll)
                wm.progress_update(i + 1)
        finally:
            wm.progress_end()

        if total_imported == 0:
            msg = "; ".join(errors) if errors else "No objects imported"
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        rig_msg = f" + armature ({last_arm.name})" if last_arm else ""
        file_msg = f" from {len(paths)} file(s)" if len(paths) > 1 else ""
        self.report(
            {"INFO"},
            f"Imported {total_imported} object(s){rig_msg}{file_msg}.",
        )
        if errors:
            self.report({"WARNING"}, " | ".join(errors))
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "import_rig")
        layout.prop(self, "use_file_name_for_armature")
        layout.prop(self, "split_by_material")
        layout.prop(self, "use_default_material")
        layout.prop(self, "recalculate_normals")
        layout.prop(self, "global_scale")
        layout.prop(self, "forward_axis")
        layout.prop(self, "empty_display_type")


class OBJECT_OT_assign_textures_by_name(Operator):
    """Assign textures based on material names matching image file names"""

    bl_idname = "object.assign_textures_by_name"
    bl_label = "Assign Textures by Name"
    bl_options = {"REGISTER", "UNDO"}

    directory: StringProperty(
        name="Texture Folder",
        description="Folder containing texture images",
        subtype="DIR_PATH",
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        folder = bpy.path.abspath(self.directory)
        if not os.path.isdir(folder):
            self.report({"ERROR"}, f"Not a directory: {folder}")
            return {"CANCELLED"}

        exts = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".dds"}
        images = {}
        for f in os.listdir(folder):
            name, ext = os.path.splitext(f)
            if ext.lower() in exts:
                full = os.path.join(folder, f)
                try:
                    img = bpy.data.images.load(full, check_existing=True)
                    images[name] = img
                except Exception:
                    pass

        assigned = 0
        for obj in context.selected_objects:
            if not hasattr(obj.data, "materials"):
                continue
            for mat in obj.data.materials:
                if not mat or str(mat.name).strip() not in images:
                    continue
                img = images[str(mat.name).strip()]
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                while nodes:
                    nodes.remove(nodes[0])
                out_n = nodes.new("ShaderNodeOutputMaterial")
                out_n.location = (300, 0)
                bsdf_n = nodes.new("ShaderNodeBsdfPrincipled")
                bsdf_n.location = (0, 0)
                img_n = nodes.new("ShaderNodeTexImage")
                img_n.location = (-300, 0)
                img_n.image = img
                links.new(img_n.outputs["Color"], bsdf_n.inputs["Base Color"])
                links.new(bsdf_n.outputs["BSDF"], out_n.inputs["Surface"])
                assigned += 1

        self.report({"INFO"}, f"Assigned textures to {assigned} materials.")
        return {"FINISHED"}


class EXPORT_OT_blender_collada(Operator, ExportHelper):
    """Export selected objects to a COLLADA (.dae) file"""

    bl_idname = "export_scene.blender_collada"
    bl_label = "Export Blender COLLADA (.dae)"
    bl_options = {"REGISTER", "UNDO", "PRESET"}
    filename_ext = ".dae"

    filter_glob: StringProperty(default="*.dae", options={"HIDDEN"})

    selected_only: BoolProperty(
        name="Selected Only",
        description="Export only selected objects (uncheck to export entire scene)",
        default=False,
    )

    export_rig: BoolProperty(
        name="Export Rig",
        description="Export armature and skin weights",
        default=True,
    )

    export_textures: BoolProperty(
        name="Reference Textures",
        description="Include texture image paths in the exported file",
        default=True,
    )

    global_scale: FloatProperty(
        name="Scale",
        description="Uniform scale applied to exported geometry and transforms",
        default=1.0,
        min=1e-6,
        soft_min=0.001,
        soft_max=1000.0,
    )

    forward_axis: EnumProperty(
        name="Forward",
        description="Which Blender axis maps to the DAE forward direction",
        items=(
            ("-Y", "-Y Forward", ""),
            ("Y",  "Y Forward",  ""),
            ("-X", "-X Forward", ""),
            ("X",  "X Forward",  ""),
            ("-Z", "-Z Forward", ""),
            ("Z",  "Z Forward",  ""),
        ),
        default="-Y",
    )

    def execute(self, context):
        count, err = export_dae(
            self.filepath,
            context,
            selected_only=self.selected_only,
            export_rig=self.export_rig,
            export_textures=self.export_textures,
            global_scale=self.global_scale,
            forward_axis=self.forward_axis,
        )
        if err and count == 0:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}
        msg = f"Exported {count} object(s) to {os.path.basename(self.filepath)}"
        if err:
            self.report({"WARNING"}, err)
        self.report({"INFO"}, msg)
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "selected_only")
        layout.prop(self, "export_rig")
        layout.prop(self, "export_textures")
        layout.prop(self, "global_scale")
        layout.prop(self, "forward_axis")
        layout.prop(self, "use_file_name_for_armature")
        layout.prop(self, "empty_display_type")
