"""Before/after figures for PR #108. Run from the branch worktree with PYTHONPATH=."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk
from PIL import Image, ImageDraw, ImageFont

from cadjoint.meshing import SurfaceOptions, extract_surface, surface_grid, wall_thickness
from tests.meshing.fixtures import clevis_bracket, space_frame_node

OUT = Path(sys.argv[1])
OUT.mkdir(exist_ok=True)
SCRATCH = Path(__file__).parent
vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
vtk.vtkMapper.SetResolveCoincidentTopologyPolygonOffsetParameters(1.5, 4.0)

W, H = 900, 640
GREY = (0.78, 0.81, 0.85)
FONT = "/System/Library/Fonts/Helvetica.ttc"


def font(size, bold=False):
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


# ── geometry ────────────────────────────────────────────────────────────────


def surface_of(scene, cell):
    return extract_surface(scene, SurfaceOptions(feature_edges=True), grid=surface_grid(scene, cell=cell))


def polydata(vertices, faces, normals=None):
    faces = np.asarray(faces, dtype=np.int64)
    cells = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    mesh = pv.PolyData(np.asarray(vertices, dtype=np.float64), cells)
    if normals is not None:
        mesh.point_data["Normals"] = np.asarray(normals, dtype=np.float64)
        mesh.point_data.active_normals_name = "Normals"
    return mesh


def segments(chords):
    chords = np.asarray(chords, dtype=np.float64).reshape(-1, 2, 3)
    points = chords.reshape(-1, 3)
    lines = np.hstack([np.full((len(chords), 1), 2), np.arange(len(points)).reshape(-1, 2)]).ravel()
    return pv.PolyData(points, lines=lines)


def fold_edges(surface, degrees=16.0):
    """Mesh edges whose two triangles fold past ``degrees``: what a mesh viewer guesses."""
    vertices = np.asarray(surface.mesh.vertices, dtype=np.float64)
    faces = np.asarray(surface.mesh.faces, dtype=np.int64)
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    normal = np.cross(b - a, c - a)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owner = np.tile(np.arange(len(faces)), 3)
    key = np.sort(edges, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, owner = key[order], owner[order]
    same = (key[1:] == key[:-1]).all(axis=1)
    first, second = owner[:-1][same], owner[1:][same]
    folded = (normal[first] * normal[second]).sum(axis=1) < np.cos(np.radians(degrees))
    return vertices[key[:-1][same][folded]]


def corner_error(surface):
    """Median angle between shading normal and facet normal at split vertices: field vs split."""
    mesh = surface.mesh
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    facet = np.cross(b - a, c - a)
    facet /= np.maximum(np.linalg.norm(facet, axis=1, keepdims=True), 1e-30)
    rendered = surface.render_mesh()
    split = (np.bincount(rendered.rows, minlength=len(vertices)) > 1)[faces]

    def angle(normals):
        cosine = np.einsum("fki,fi->fk", normals, facet)
        return np.degrees(np.arccos(np.clip(cosine, -1, 1)))[split]

    field = angle(np.asarray(mesh.normals, dtype=np.float64)[faces])
    return float(np.median(field)), float(np.median(angle(surface.corner_normals)))


# ── rendering ───────────────────────────────────────────────────────────────


def render(meshes, camera, size=(W, H)):
    """meshes: list of (PolyData, kwargs)."""
    plotter = pv.Plotter(off_screen=True, window_size=size)
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    for mesh, kwargs in meshes:
        actor = plotter.add_mesh(mesh, **kwargs)
        if kwargs.get("phong"):
            actor.prop.interpolation = "phong"
    plotter.camera_position = camera
    plotter.camera.view_angle = camera_angle(camera)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return Image.fromarray(image)


def camera_angle(camera):
    return 30.0


def shaded(mesh, color=GREY):
    return (mesh, dict(color=color, phong=True, smooth_shading=False, specular=0.35, specular_power=30, diffuse=0.85, ambient=0.15))


def lines(chords, color="#1a1a1a", width=2.5):
    return (segments(chords), dict(color=color, line_width=width, render_lines_as_tubes=False))


def camera_on(center, direction, distance, up=(0, 0, 1)):
    center = np.asarray(center, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    return [tuple(center + distance * direction), tuple(center), up]


# ── composition ─────────────────────────────────────────────────────────────


def panel(image, label):
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 44], fill="white")
    draw.text((18, 12), label, fill="#111111", font=font(24, bold=True))
    return image


def annotate(image, text, xy, color="#111111", size=22):
    draw = ImageDraw.Draw(image)
    box = draw.textbbox(xy, text, font=font(size))
    pad = 6
    draw.rectangle([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], fill=(255, 255, 255), outline=color, width=2)
    draw.text(xy, text, fill=color, font=font(size))


def inset(image, zoomed, box_on_image, at, color="#d1495b"):
    """Paste ``zoomed`` at ``at`` and frame the region ``box_on_image`` it magnifies."""
    draw = ImageDraw.Draw(image)
    draw.rectangle(box_on_image, outline=color, width=3)
    framed = zoomed.copy()
    ImageDraw.Draw(framed).rectangle([0, 0, framed.width - 1, framed.height - 1], outline=color, width=4)
    image.paste(framed, at)


def ring(image, center, radius, color="#d1495b", width=4):
    x, y = center
    ImageDraw.Draw(image).ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=width)


def figure(title, caption, left, right, name, key=None):
    gap, margin = 16, 24
    title_h, caption_h = 64, 100
    width = left.width + right.width + gap + 2 * margin
    height = title_h + max(left.height, right.height) + caption_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 18), title, fill="#111111", font=font(30, bold=True))
    sheet.paste(left, (margin, title_h))
    sheet.paste(right, (margin + left.width + gap, title_h))
    draw.rectangle([margin, title_h, margin + left.width - 1, title_h + left.height - 1], outline="#cccccc", width=2)
    draw.rectangle(
        [margin + left.width + gap, title_h, margin + left.width + gap + right.width - 1, title_h + right.height - 1],
        outline="#cccccc",
        width=2,
    )
    draw.text((margin, title_h + left.height + 18), caption, fill="#333333", font=font(21))
    if key is not None:
        sheet.paste(key, (width - margin - key.width, title_h + left.height + 14))
    sheet.save(OUT / name)
    print("wrote", name, sheet.size)


SCALE = 3


def render_hi(meshes, camera, points=()):
    """Render at SCALE x; return (full-res image, projected pixel positions of ``points`` in full-res)."""
    plotter = pv.Plotter(off_screen=True, window_size=(W * SCALE, H * SCALE))
    plotter.set_background("white")
    for mesh, kwargs in meshes:
        kwargs = dict(kwargs)
        phong = kwargs.pop("phong", False)
        if "line_width" in kwargs:
            kwargs["line_width"] *= SCALE
        actor = plotter.add_mesh(mesh, **kwargs)
        if phong:
            actor.prop.interpolation = "phong"
    plotter.camera_position = camera
    plotter.camera.view_angle = 30.0
    image = Image.fromarray(plotter.screenshot(return_img=True))
    projected = []
    coordinate = vtk.vtkCoordinate()
    coordinate.SetCoordinateSystemToWorld()
    for point in points:
        coordinate.SetValue(*point)
        x, y = coordinate.GetComputedDoubleDisplayValue(plotter.renderer)
        projected.append((x, image.height - y))
    plotter.close()
    return image, projected


def small(image):
    return image.resize((W, H), Image.LANCZOS)


def with_inset(big, focus, label, zoom_box=150, inset_size=300, at=(W - 318, H - 318)):
    """Main panel (downscaled) plus an inset cropped from the full-res render around ``focus``."""
    fx, fy = focus
    half = zoom_box * SCALE / 2
    crop = big.crop((int(fx - half), int(fy - half), int(fx + half), int(fy + half))).resize((inset_size, inset_size), Image.LANCZOS)
    main = small(big)
    box = [fx / SCALE - zoom_box / 2, fy / SCALE - zoom_box / 2, fx / SCALE + zoom_box / 2, fy / SCALE + zoom_box / 2]
    inset(main, crop, box, at)
    return panel(main, label)


def main():
    clevis = surface_of(clevis_bracket.scene, 1.0)
    node = surface_of(space_frame_node.scene, 0.825)
    print("surfaces", clevis.mesh.faces.shape, node.mesh.faces.shape)

    clevis_cam = camera_on((0, 0, 24), (0.67, 0.47, 0.57), 250)
    node_cam = camera_on((0, 0, 0), (-0.75, -0.30, 0.25), 210)
    rim = (9.0, 26.0, 47.0)  # the +y lug's crown rim, outer face
    tube_end = node.mesh.vertices[np.argmax(np.asarray(node.mesh.vertices) @ np.array([-0.8, -0.3, 0.2]))]

    # 1. crease shading --------------------------------------------------------
    for name, surface, camera, focus_point in (
        ("clevis", clevis, clevis_cam, rim),
        ("node", node, node_cam, tuple(np.asarray(tube_end, dtype=float))),
    ):
        before_err, after_err = corner_error(surface)
        welded = polydata(surface.mesh.vertices, surface.mesh.faces, surface.mesh.normals)
        rendered = surface.render_mesh()
        split = polydata(rendered.vertices, rendered.faces, rendered.normals)
        big_before, (focus,) = render_hi([shaded(welded)], camera, [focus_point])
        big_after, _ = render_hi([shaded(split)], camera)
        left = with_inset(big_before, focus, "Before: one averaged normal per vertex")
        right = with_inset(big_after, focus, "After: surface.render_mesh()")
        annotate(left, f"median normal error at crease vertices: {before_err:.0f}°", (18, 60))
        annotate(right, f"median normal error at crease vertices: {after_err:.0f}°", (18, 60))
        figure(
            f"1. Crease shading — {'clevis bracket, 1 mm cell' if name == 'clevis' else 'space-frame node, 0.825 mm cell'}",
            "A crease vertex's single normal is the mean of both sides, so rims shade grainy. render_mesh() gives one vertex per\n"
            "smooth side, with that side's Hermite normal. Error = angle between a corner's shading normal and its triangle's normal.",
            left,
            right,
            f"1-crease-shading-{name}.png",
        )

    # 2. feature edges ----------------------------------------------------------
    bolt = (45.0, 20.0, 10.0)
    for name, surface, camera, focus_point in (
        ("clevis", clevis, clevis_cam, bolt),
        ("node", node, node_cam, tuple(np.asarray(tube_end, dtype=float))),
    ):
        welded = polydata(surface.mesh.vertices, surface.mesh.faces, surface.mesh.normals)
        rendered = surface.render_mesh()
        split = polydata(rendered.vertices, rendered.faces, rendered.normals)
        guessed = fold_edges(surface)
        big_before, (focus,) = render_hi([shaded(split), lines(guessed, width=2.0)], camera, [focus_point])
        big_after, _ = render_hi([shaded(split), lines(surface.feature_edges, width=2.0)], camera)
        left = with_inset(big_before, focus, "Before: mesh edges folding > 16°", zoom_box=110)
        right = with_inset(big_after, focus, "After: surface.feature_edges", zoom_box=110)
        annotate(left, f"{len(guessed):,} mesh edges", (18, 60))
        annotate(right, f"{len(surface.feature_edges):,} chords", (18, 60))
        figure(
            f"2. Feature edges — {'clevis bracket, 1 mm cell' if name == 'clevis' else 'space-frame node, 0.825 mm cell'}",
            "Dual-contour vertices near a crease land on either side of it from cell to cell, so fold-angle edges draw stubs and\n"
            "zig-zags. feature_edges links the feature cells along each crease and hard CSG seam: chords of the true curve\n"
            "(they still leave short gaps, visible here on the node's outer rim).",
            left,
            right,
            f"2-feature-edges-{name}.png",
        )

    # 3. viewer overlay seams ---------------------------------------------------
    before = np.load(SCRATCH / "overlay_before.npy")
    after = np.load(SCRATCH / "overlay_after.npy")
    rendered = clevis.render_mesh()
    split = polydata(rendered.vertices, rendered.faces, rendered.normals)
    def key(chords):
        return {tuple(np.round(np.sort(c, axis=0).ravel(), 4)) for c in chords}

    kept = key(after)
    removed = np.array([c for c in before if tuple(np.round(np.sort(c, axis=0).ravel(), 4)) not in kept])
    still = np.array([c for c in before if tuple(np.round(np.sort(c, axis=0).ravel(), 4)) in kept])
    root = np.abs(removed[:, :, 2].mean(axis=1) - 10.0) < 2.0
    print("overlay: removed", len(removed), "root", int(root.sum()), "flush", int((~root).sum()), "kept", len(still))
    cam = camera_on((0, 0, 22), (0.55, -0.62, 0.56), 200)
    big_before, _ = render_hi(
        [shaded(split), lines(still, width=2.0), lines(removed[~root], color="#d1495b", width=4.0)],
        cam,
    )
    # The root seam lies inside the fillet's material, so the surface hides it:
    # draw it through the part, from a lines-only pass of the same camera.
    seam_only, _ = render_hi([lines(removed[root], color="#2e86ab", width=4.0)], cam)
    ink = np.asarray(seam_only).astype(int)
    mask = Image.fromarray(((255 * 3 - ink.sum(axis=2)) > 60).astype(np.uint8) * 255)
    big_before.paste(seam_only, (0, 0), mask)
    big_after, _ = render_hi([shaded(split), lines(after, width=2.0)], cam)
    left, right = panel(small(big_before), "Before (main)"), panel(small(big_after), "After (this PR)")
    annotate(left, "1,918 segments", (18, 60))
    annotate(right, "1,665 segments", (18, 60))
    annotate(left, f"red: flush seam, lug stem meets crown ({int((~root).sum())})", (18, 104), color="#d1495b", size=19)
    annotate(left, f"blue: hard seam under the root fillet, drawn through it ({int(root.sum())})", (18, 142), color="#2e86ab", size=19)
    figure(
        "3. Viewer sharp-edge overlay — clevis bracket, 1 mm cell",
        "Two kinds of seam chord no longer draw: operands meeting flush on one face (no fold), and the operands' hard seam under\n"
        "a blended Boolean (the projected point is off the scene's surface). Hole, cutout, lug and rib rims are kept.",
        left,
        right,
        "3-viewer-overlay-seams.png",
    )

    # 4. thin walls -------------------------------------------------------------
    coarse = surface_of(space_frame_node.scene, 1.5)
    panels = []
    for surface, label in ((coarse, "1.5 mm cell"), (node, "0.825 mm cell")):
        walls = wall_thickness(space_frame_node.scene, surface.mesh, surface.cell)
        cells = walls.per_vertex / surface.cell
        rendered = surface.render_mesh()
        mesh = polydata(rendered.vertices, rendered.faces, rendered.normals)
        values = cells[rendered.rows]
        mesh.point_data["cells"] = np.where(np.isnan(values), 5.5, np.minimum(values, 4.99))
        cmap = _wall_colours()
        big, _ = render_hi(
            [(mesh, dict(scalars="cells", cmap=cmap, clim=(0, 6), show_scalar_bar=False, phong=True, specular=0.25, interpolate_before_map=False))],
            node_cam,
        )
        image = panel(small(big), f"{label}: thinnest {walls.thinnest:.2f} mm = {walls.cells:.1f} cells")
        verdict = "not resolved (< 2 cells)" if not walls.resolved else "resolved (≥ 2 cells)"
        annotate(image, verdict, (18, 60), color="#d1495b" if not walls.resolved else "#2a7a3b")
        panels.append(image)
    legend = _legend()
    figure(
        "4. Wall thickness — space-frame node (2 mm shell)",
        "wall_thickness() sends a ray inward from every vertex through the field and reports the wall in cells. Under 2 cells the\n"
        "mesh closes through the wall or falls apart (here: Euler characteristic −313 at 1.5 mm); about 4 cells give a clean wall.",
        panels[0],
        panels[1],
        "4-wall-thickness-node.png",
        key=legend,
    )


BANDS = [(0, 1, "#8e1b2c"), (1, 2, "#d1495b"), (2, 3, "#edae49"), (3, 4, "#f4e3a1"), (4, 5, "#7fa7c9"), (5, 6, "#e6e8eb")]


def _wall_colours():
    from matplotlib.colors import ListedColormap

    return ListedColormap([colour for _, _, colour in BANDS])


def _legend():
    labels = ["< 1", "1–2", "2–3", "3–4", "≥ 4", "none"]
    image = Image.new("RGB", (560, 70), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width - 1, image.height - 1], outline="#999999", width=2)
    draw.text((10, 6), "cells across the wall (none = no wall measured)", fill="#111111", font=font(17, bold=True))
    for column, ((_, _, colour), text) in enumerate(zip(BANDS, labels)):
        x = 10 + 91 * column
        draw.rectangle([x, 36, x + 24, 58], fill=colour, outline="#666666")
        draw.text((x + 30, 37), text, fill="#111111", font=font(17))
    return image


if __name__ == "__main__":
    main()
