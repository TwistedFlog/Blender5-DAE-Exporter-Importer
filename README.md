# Blender COLLADA (.dae) Import & Export — Blender 5

A lightweight Blender 5 **extension** that restores support for importing and exporting `.dae`
(COLLADA) files after the official importer was removed in Blender 5. It also
handles textures, polylist geometry, armatures, skin weights, and per-file
up-axis correction.

This version of the Addon aims to work and behave similarly to the original DAE addon, but
obviously with improvements.

> **Upstream project:** <https://github.com/Tzeentchnet/Blender-v5-DAE-Importer-Add-on>

## Credits

This extension builds directly on prior community work:

- **[Tzeentchnet](https://github.com/Tzeentchnet)** — the immediate code base this
  extension is built on
- **[RebeccaNod1](https://github.com/RebeccaNod1)** — the next to immediate code base this
  extension is built on
- **[MilesExilium](https://github.com/MilesExilium)** — extended materials, armature, skin
  weights, polylist support
- **[ekztal](https://github.com/ekztal)** — original Blender 5 add-on
- Original concept by **/u/varyingopinions** on Reddit

Huge thanks to the authors above for the core importer logic. On top of that
foundation, this fork repackages everything as a proper Blender 5 extension,
adds drag-and-drop and multi-file import, a status-bar progress indicator,
configurable scale and forward axis, a correctness fix for skin-weight
assignment, and NumPy-accelerated parsing and mesh upload

## Features

- Supports both **COLLADA 1.4.1** and **COLLADA 1.5.0** schemas — namespace is
  detected at parse time so files from any compliant exporter load correctly.
- Imports COLLADA `.dae` geometry — `<triangles>` **and** `<polylist>` (auto fan-triangulated).
- Builds Principled BSDF materials with diffuse, normal, alpha/transparent, AO,
  and specular channels.
- Resolves scene-level material bindings, including the common case where a
  primitive material symbol maps to a different COLLADA material id.
- Supports both **FCOLLADA** and **OpenCOLLADA3dsMax** normal-map conventions.
- Imports skinned meshes instanced through `<instance_controller>` as well as
  direct `<instance_geometry>` nodes.
- Heuristic "bad diffuse" correction — substitutes the `_alb` variant when an
  exporter wires an AO/normal/spec map into the diffuse channel.
- Imports armatures from joint hierarchy + `INV_BIND_MATRIX` (toggleable).
- Imports skin weights as vertex groups + `Armature` modifier.
- Per-file `up_axis` correction (Y_UP / Z_UP / X_UP).
- **Configurable global scale** and **forward-axis** remap on the import operator.
- **Progress bar** in Blender's status bar — per-geometry updates for single
  files and per-file updates for batched imports.
- **Drag-and-drop** `.dae` files from your OS file manager into the 3D viewport.
- **NumPy-accelerated** parsing and bulk mesh data transfer for fast imports of
  large meshes.
- "Assign Textures by Name" helper operator (Object menu) for batch texture wiring.

## Installation

### Blender 5.0+ (extensions panel)

1. Download code as zip
2. In Blender open **Edit → Preferences → Get Extensions → Install from Disk…**
3. Select the `.zip` file.
4. The extension is enabled automatically. You will find it under the **Add-ons**
   tab as **Blender COLLADA (.dae) Importer**.

## Usage
Same as old Blender

### File menu

**File → Import → Blender COLLADA (.dae)** — opens a file dialog. The sidebar
(and F9 redo panel) exposes:

- **Import Rig** — Import armature/skin weights.
- **Use File Name for Armature** — Uses File Name for the Armature name 
  instead of the .dae one
- **Split by Material** — split each imported geometry into one object per
  material, so each piece appears separately in the Outliner. Off by default.
- **Use Blender Default Material** — ignore the DAE's diffuse / specular
  colors and texture references and assign each material Blender's stock
  Principled BSDF defaults instead. Useful when a DAE has no usable materials
  or its textures cannot be located, to avoid splotchy chrome-like viewport
  shading. Off by default.
- **Recalculate Normals (Outside)** — discard any per-corner normals from the
  DAE and recompute consistent outward-facing normals after import (same as
  Mesh → Normals → Recalculate Outside in edit mode). All polygons are also
  marked smooth. Fixes DAE files with inconsistent face winding that show up
  as dark / flipped patches on the model. On by default.
- **Scale** — uniform scale applied to imported geometry and transforms
  (default `1.0`).
- **Forward** — which DAE axis (after up-axis correction) maps to Blender's
  forward (`-Y`). Default `-Y` keeps existing behavior.
- **Empty Display** — Changes the way empties are displayed.

The defaults for every option above are also configurable in
**Edit → Preferences → Add-ons → Blender COLLADA (.dae) Import & Export**, so your
preferred values are pre-selected in both the file dialog and drag-and-drop
imports.

### Drag and drop

Drag one or more `.dae` files from your OS file manager (Explorer, Finder,
Nautilus, …) onto the **3D Viewport** or **Outliner**. The files are imported
immediately using the defaults set in the add-on preferences. Use the **F9
redo panel** to adjust any option after the drop.

### Assign Textures by Name

After import, in the **Object menu** select **Assign Textures by Name**, then
pick a folder. Materials whose name matches an image file (e.g. material
`brick_wall` and image `brick_wall.png`) get a fresh Principled BSDF + image
texture.

## Suggested future improvements

These are documented for contributors; they are **not** implemented yet.

- Use `xml.etree.ElementTree.iterparse` + `clear()` to reduce peak memory on
  very large `.dae` files.
- Optional preservation of vertex order for round-trip workflows.

## Notes

- Tested on Blender 5.1+. Skin weights require the mesh and armature to be
  exported together in the same `.dae` file.
- Normal maps are connected automatically for the imported UV set. Models that
  rely on additional UV channels may still need manual shader adjustment.
- Schema coverage: COLLADA 1.4.1 and 1.5.0. Only `<triangles>` and `<polylist>`
  primitives are imported — `<polygons>`, `<lines>`, `<linestrips>`, `<tristrips>`,
  `<trifans>`, and 1.5-only B-rep / NURBS / kinematics / physics elements are
  ignored.

## License

GPL-3.0-or-later — matches Blender's add-on licensing requirements. See
[`LICENSE`](LICENSE) for the full text.
