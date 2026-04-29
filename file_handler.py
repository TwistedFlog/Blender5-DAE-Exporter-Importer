# SPDX-License-Identifier: GPL-3.0-or-later
"""Drag-and-drop FileHandler for .dae files."""

import bpy

from .operators import IMPORT_OT_blender_collada_full


class IO_FH_blender_collada(bpy.types.FileHandler):
    """Allow dragging .dae files from the OS into Blender's 3D viewport / outliner."""

    bl_idname = "IO_FH_blender_collada"
    bl_label = "Blender COLLADA"
    bl_import_operator = IMPORT_OT_blender_collada_full.bl_idname
    bl_file_extensions = ".dae"

    @classmethod
    def poll_drop(cls, context):
        # Accept drops in the 3D View and Outliner.
        return context.area is not None and context.area.type in {"VIEW_3D", "OUTLINER"}
