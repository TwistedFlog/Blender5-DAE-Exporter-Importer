# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender COLLADA (.dae) Importer — Blender 5 extension."""

import bpy

from . import importer  # noqa: F401  (kept for side-effect-free module import)
from .operators import (
    IMPORT_OT_blender_collada_full,
    EXPORT_OT_blender_collada,
    OBJECT_OT_assign_textures_by_name,
)
from .file_handler import IO_FH_blender_collada
from .preferences import BlenderColladaPreferences


classes = (
    BlenderColladaPreferences,
    IMPORT_OT_blender_collada_full,
    EXPORT_OT_blender_collada,
    OBJECT_OT_assign_textures_by_name,
    IO_FH_blender_collada,
)


def menu_func_import(self, context):
    self.layout.operator(
        IMPORT_OT_blender_collada_full.bl_idname,
        text="Blender COLLADA (.dae)",
    )


def menu_func_export(self, context):
    self.layout.operator(
        EXPORT_OT_blender_collada.bl_idname,
        text="Blender COLLADA (.dae)",
    )


def menu_func_assign_textures(self, context):
    self.layout.operator(
        OBJECT_OT_assign_textures_by_name.bl_idname,
        text="Assign Textures by Name",
    )


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
    bpy.types.VIEW3D_MT_object.append(menu_func_assign_textures)


def unregister():
    bpy.types.VIEW3D_MT_object.remove(menu_func_assign_textures)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
