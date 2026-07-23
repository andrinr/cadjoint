# Interactive UI: Draggable Object Manipulation

> Research into adding interactive, draggable manipulation of jaxCAD objects from Python/Jupyter.

## Current state

- All rendering is matplotlib-only (static images).
- No interactive UI — everything is scripted.
- `extract_parameters(sdf)` already returns `{path: Parameter}` — this is the bridge to any UI.
- `solve_constraints(sdf)` re-solves after parameter changes — drag → re-solve → re-render loop is architecturally ready.

## The core interaction loop

```
user drags handle
    → new parameter value
    → (optionally) solve_constraints(scene)  # snap to constraint manifold
    → re-render
    → update display
```

This loop is already fully supported by jaxCAD's existing code. The UI frameworks below just need to trigger it.

---

## Approach 1: 2D sketch editor via ipympl (recommended first step)

### What it is

`%matplotlib widget` (ipympl) makes matplotlib figures interactive in Jupyter. Mouse events (click, drag, release) fire Python callbacks with data-space coordinates. No new dependencies beyond what is already installed.

### Why it's the right first step

- **Zero new dependencies** — matplotlib is already in `pyproject.toml`.
- Directly exercises the existing constraint solver.
- `Line`, `Circle`, and `Rectangle` already exist as parameter-bearing objects.
- Delivers the most useful interactive experience for 2D sketch building.

### Implementation pattern

```python
%matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jaxcad.geometry.parameters import Vector
from jaxcad.constraints.solve import newton_raphson
from jaxcad.constraints.graph import ConstraintGraph

# Scene
p1 = Vector([0.0, 0.0, 0.0], free=True, name="p1")
p2 = Vector([2.0, 0.0, 0.0], free=True, name="p2")
p3 = Vector([1.0, 1.5, 0.0], free=True, name="p3")
params = {"p1": p1, "p2": p2, "p3": p3}
# ... add constraints to graph ...

# --- Drag state ---
dragging = {"name": None}
PICK_RADIUS = 0.15

fig, ax = plt.subplots(figsize=(7, 5))

def draw():
    ax.cla()
    pts = {n: p.value[:2] for n, p in params.items()}
    for a, b in [("p1","p2"),("p2","p3"),("p3","p1")]:
        ax.plot([pts[a][0],pts[b][0]], [pts[a][1],pts[b][1]], 'b-', lw=2)
    for name, (x,y) in pts.items():
        ax.add_patch(plt.Circle((x,y), 0.08, color='orange', zorder=5))
    ax.set_xlim(-1,4); ax.set_ylim(-1,3); ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.canvas.draw_idle()

def on_press(event):
    if event.inaxes != ax: return
    mouse = np.array([event.xdata, event.ydata])
    for name, param in params.items():
        if np.linalg.norm(mouse - np.array(param.value[:2])) < PICK_RADIUS:
            dragging["name"] = name; break

def on_move(event):
    if not dragging["name"] or not event.inaxes: return
    params[dragging["name"]].value = jnp.array(
        [event.xdata, event.ydata, 0.0], dtype=jnp.float32
    )
    # Run a few Newton steps to track constraints during drag
    try:
        _re_solve(graph, list(params.values()), max_iter=3)
    except Exception:
        pass
    draw()

def on_release(event):
    dragging["name"] = None
    try:
        _re_solve(graph, list(params.values()), max_iter=100)  # full solve on release
    except Exception:
        pass
    draw()

fig.canvas.mpl_connect('button_press_event',   on_press)
fig.canvas.mpl_connect('motion_notify_event',  on_move)
fig.canvas.mpl_connect('button_release_event', on_release)
draw()
plt.show()
```

### Constraint-aware dragging

When dragging a constrained point, project the mouse move onto the constraint manifold using the null-space already computed by `ConstraintGraph.extract_free_dof`:

```python
def drag_on_manifold(delta_mouse, null_space):
    """Project mouse displacement onto constraint manifold null-space."""
    # null_space: (n_params, n_free_dof)
    # Project delta into the free-DOF subspace
    coeffs = null_space.T @ delta_mouse    # (n_free_dof,)
    return null_space @ coeffs             # projected full-space move
```

---

## Approach 2: Marimo reactive sliders (best for Scalar parameters)

### What it is

[Marimo](https://marimo.io) (`pip install marimo`) is a reactive Python notebook. When a `mo.ui.slider` value changes, all dependent cells re-run automatically. No callbacks needed.

### Why it's the best for scalar exploration

- `radius`, `angle`, `smoothness` parameters map perfectly to sliders.
- Re-render fires automatically on every slider move.
- Works today with jaxCAD's existing `render()`.

### Pattern

```python
# Cell 1 — define scene
import marimo as mo, jaxcad as jc, jax.numpy as jnp
from jaxcad.geometry.parameters import Scalar

radius_p = Scalar(1.0, free=True, name="radius")
sphere = jc.Sphere(radius=radius_p)

# Cell 2 — slider
radius_slider = mo.ui.slider(0.1, 3.0, value=1.0, step=0.05, label="Radius")
radius_slider

# Cell 3 — render (re-runs on every slider change)
radius_p.value = jnp.array(radius_slider.value)
ax = jc.render(sphere, method="raymarch")
mo.mpl.interactive(ax.figure)
```

### Auto-generating sliders from free parameters

```python
def parameter_sliders(sdf):
    """Generate one slider per free Scalar parameter."""
    from jaxcad.extraction import extract_parameters
    from jaxcad.geometry.parameters import Scalar
    free, _ = extract_parameters(sdf)
    sliders = {}
    for path, param in free.items():
        if isinstance(param, Scalar):
            lo, hi = param.bounds or (0.0, 5.0)
            sliders[param.name] = mo.ui.slider(lo, hi, value=float(param.value))
    return mo.vstack([mo.hstack([mo.text(n), s]) for n, s in sliders.items()]), sliders
```

---

## Approach 3: anywidget + three.js drag handles (best Jupyter 3D)

### What it is

[anywidget](https://anywidget.dev) (`pip install anywidget`) lets you write custom Jupyter widgets as an ES module + Python class. Synced state via traitlets. Works in JupyterLab 4, VS Code, and Marimo.

Pair with **three.js `DragControls`** for genuine 3D drag handles on free `Vector` parameters.

When rasterized three.js helpers or meshes are layered over the SDF shader,
write the raymarched intersection into the shared depth buffer instead of
manually ordering passes. [Compositing raymarched and polygonal
geometry](https://iquilezles.org/articles/raypolys/) derives the required
eye-space to clip-space depth mapping and is the implementation reference for
that viewer work.

### Architecture

```
Python (anywidget)                    JavaScript (three.js)
─────────────────────────────────     ──────────────────────────
SceneWidget.mesh_vertices  ────────>  rebuild THREE.Mesh
SceneWidget.mesh_faces     ────────>
SceneWidget.handles        ────────>  spawn orange handle spheres
                           <────────  SceneWidget.dragged = {name, pos}
on_drag: update param.value
         → rebuild mesh
         → update mesh_vertices ──>
```

### Python side

```python
import anywidget, traitlets, jax, jax.numpy as jnp, numpy as np
from skimage import measure
from jaxcad.extraction import extract_parameters
from jaxcad.geometry.parameters import Vector

class SceneWidget(anywidget.AnyWidget):
    handles        = traitlets.List([]).tag(sync=True)
    dragged        = traitlets.Dict({}).tag(sync=True)
    mesh_vertices  = traitlets.List([]).tag(sync=True)
    mesh_faces     = traitlets.List([]).tag(sync=True)

    _esm = "..."  # JavaScript ES module (see below)

    def __init__(self, sdf, **kwargs):
        super().__init__(**kwargs)
        self._sdf = sdf
        self._update_mesh(); self._update_handles()
        self.observe(self._on_drag, names=["dragged"])

    def _on_drag(self, change):
        d = change["new"]
        if not d: return
        free, _ = extract_parameters(self._sdf)
        for param in free.values():
            if param.name == d["name"] and isinstance(param, Vector):
                param.value = jnp.array(d["pos"])
        self._update_mesh()

    def _update_handles(self):
        free, _ = extract_parameters(self._sdf)
        self.handles = [
            {"name": p.name, "pos": list(map(float, p.value))}
            for p in free.values()
            if isinstance(p, Vector) and p.name
        ]

    def _update_mesh(self, res=50):
        x = jnp.linspace(-4, 4, res)
        X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
        pts = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        vol = np.array(jax.vmap(self._sdf)(pts).reshape(res, res, res))
        try:
            verts, faces, _, _ = measure.marching_cubes(vol, 0.0, spacing=(8/res,)*3)
            verts -= 4.0
            self.mesh_vertices = verts.tolist()
            self.mesh_faces = faces.tolist()
        except ValueError:
            pass
```

### JavaScript side (ES module)

```javascript
import * as THREE from 'https://esm.sh/three@0.164';
import { OrbitControls } from 'https://esm.sh/three@0.164/examples/jsm/controls/OrbitControls.js';
import { DragControls } from 'https://esm.sh/three@0.164/examples/jsm/controls/DragControls.js';

export function render({ model, el }) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(600, 400);
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1.5, 0.1, 100);
  camera.position.set(6, 6, 6);
  const orbit = new OrbitControls(camera, renderer.domElement);

  let meshObj = null, dragControls = null;
  const handleObjects = [];

  function rebuildMesh() {
    if (meshObj) scene.remove(meshObj);
    const verts = model.get('mesh_vertices');
    const faces = model.get('mesh_faces');
    if (!verts.length) return;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts.flat(), 3));
    geo.setIndex(faces.flat());
    geo.computeVertexNormals();
    meshObj = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x4fc3f7, roughness: 0.4 }));
    scene.add(meshObj);
  }

  function rebuildHandles() {
    handleObjects.forEach(h => scene.remove(h));
    handleObjects.length = 0;
    if (dragControls) dragControls.dispose();

    for (const h of model.get('handles')) {
      const geo = new THREE.SphereGeometry(0.12, 16, 16);
      const mat = new THREE.MeshStandardMaterial({ color: 0xff6600 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(...h.pos);
      mesh.userData.paramName = h.name;
      scene.add(mesh);
      handleObjects.push(mesh);
    }

    dragControls = new DragControls(handleObjects, camera, renderer.domElement);
    dragControls.addEventListener('drag', e => {
      orbit.enabled = false;
      const { x, y, z } = e.object.position;
      model.set('dragged', { name: e.object.userData.paramName, pos: [x, y, z] });
      model.save_changes();
    });
    dragControls.addEventListener('dragend', () => { orbit.enabled = true; });
  }

  scene.add(new THREE.AmbientLight(0xffffff, 0.4));
  scene.add(Object.assign(new THREE.DirectionalLight(0xffffff, 0.8), { position: { x:5,y:5,z:5 } }));

  model.on('change:mesh_vertices', rebuildMesh);
  model.on('change:handles', rebuildHandles);
  rebuildMesh(); rebuildHandles();

  function animate() {
    requestAnimationFrame(animate);
    orbit.update();
    renderer.render(scene, camera);
  }
  animate();
}
```

### Key libraries

- anywidget: https://anywidget.dev / https://github.com/manzt/anywidget
- three.js DragControls: https://threejs.org/docs/#examples/en/controls/DragControls

---

## Approach 4: pyvista mesh picking (3D, Jupyter-compatible)

### What it is

[pyvista](https://docs.pyvista.org) (`pip install pyvista`) is a VTK-based Python 3D viewer. In Jupyter (via trame or panel backend), it supports mesh picking: clicking the mesh returns the 3D world coordinate, which can be used to update a `Vector` parameter.

### When to use it

Good for workflows where you want to **click a point on the mesh** to set a parameter value — e.g., "click where the sphere center should be". Less suitable for live drag (high latency due to VTK-Python round-trip).

### Pattern

```python
import pyvista as pv
import numpy as np

pl = pv.Plotter(notebook=True)
mesh = make_pv_mesh(scene)  # marching cubes → pv.PolyData
actor = pl.add_mesh(mesh, color="cyan")

def on_pick(point):
    sphere_center.value = jnp.array(point)
    # Optionally: solve_constraints(scene)
    pl.remove_actor(actor)
    pl.add_mesh(make_pv_mesh(scene), color="cyan")
    pl.render()

pl.enable_point_picking(callback=on_pick, use_mesh=True)
pl.show()
```

---

## Approach 5: FastAPI + three.js web app (shareable URL)

### When to use it

Best for shipping jaxCAD as a tool others can use in a browser — no local Python install required on the client side.

### Architecture

```
Browser (three.js)                 Python server (FastAPI + JAX)
──────────────────                 ──────────────────────────────
GET /mesh          ─────────────>  marching cubes → glTF JSON
DragControls drag  ─────────────>  POST /update_param {name, pos}
                   <─────────────  re-mesh → updated glTF
```

### Server sketch

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import jax, jax.numpy as jnp, numpy as np
from skimage import measure

app = FastAPI()
center = Vector([0.,0.,0.], free=True, name="center")
scene  = Sphere(radius=1.5).translate(center)

@app.get("/mesh")
def get_mesh():
    x = jnp.linspace(-4, 4, 60)
    X,Y,Z = jnp.meshgrid(x,x,x, indexing="ij")
    pts = jnp.stack([X.ravel(),Y.ravel(),Z.ravel()], axis=1)
    vol = np.array(jax.vmap(scene)(pts).reshape(60,60,60))
    verts, faces, _, _ = measure.marching_cubes(vol, 0.0, spacing=(8/60,)*3)
    verts -= 4.0
    return JSONResponse({"vertices": verts.tolist(), "faces": faces.tolist()})

@app.post("/update_param")
def update_param(body: dict):
    if body["name"] == center.name:
        center.value = jnp.array(body["pos"])
    return {"ok": True}

@app.get("/handles")
def get_handles():
    free, _ = extract_parameters(scene)
    return [{"name": p.name, "pos": list(map(float, p.value))}
            for p in free.values() if isinstance(p, Vector) and p.name]
```

---

## Approach 6: imgui-bundle + ImGuizmo (desktop app)

### What it is

[imgui-bundle](https://github.com/pthom/imgui_bundle) (`pip install imgui-bundle`) exposes **ImGuizmo** — the canonical open-source 3D gizmo (translate/rotate/scale arrows). Desktop-only (no Jupyter), but gives professional-quality handles.

### Pattern

```python
from imgui_bundle import imgui, imguizmo, hello_imgui
import numpy as np

camera_view = np.eye(4, dtype=np.float32)
camera_proj = make_projection(fov=50, aspect=1.5, near=0.1, far=100)
object_matrix = np.eye(4, dtype=np.float32)

def gui_loop():
    global object_matrix
    changed, object_matrix = imguizmo.manipulate(
        camera_view, camera_proj,
        imguizmo.OPERATION.translate,
        imguizmo.MODE.world,
        object_matrix,
    )
    if changed:
        new_pos = object_matrix[:3, 3]
        selected_param.value = jnp.array(new_pos)
        # solve_constraints(scene)
        rerender(scene)

hello_imgui.run(gui_loop)
```

### Key library

- imgui-bundle (includes ImGuizmo): https://github.com/pthom/imgui_bundle

---

## Comparison table

| Approach | Jupyter? | Effort | Best for |
|---|---|---|---|
| **ipympl + matplotlib callbacks** | Yes | 1–2 days | 2D sketch editor with draggable constrained points |
| **Marimo reactive sliders** | Marimo | ~half day | Scalar parameters (radius, angle, scale) |
| **anywidget + three.js** | Yes | 3–5 days | 3D drag handles in notebook |
| **pyvista picking** | Yes (iframe) | 1–2 days | Click-to-set Vector parameter from 3D mesh |
| **FastAPI + three.js** | No (browser) | 3–5 days | Shareable web app |
| **imgui-bundle + ImGuizmo** | No (desktop) | 4–7 days | Professional-quality desktop gizmo |

---

## Recommended implementation order

### Phase 1 — 2D sketch editor (ipympl)

**Zero new dependencies.** Implement drag events on matplotlib canvas for `Line`/`Circle`/`Rectangle` geometry. Wire to `newton_raphson` (or the new LM solver) for live constraint re-solving during drag. ~100 lines of code.

Key file to add: `jaxcad/ui/sketch_editor.py`

### Phase 2 — Marimo parameter sliders

Add a `jaxcad.ui.sliders(sdf)` helper:
```python
def sliders(sdf):
    """Return (mo.vstack of sliders, dict of {name: slider}) for all free Scalars."""
```
Works with Marimo's reactive re-execution — no callbacks needed. ~20 lines.

### Phase 3 — anywidget + three.js 3D handles

The code sketches above are essentially complete (~200 lines Python + ~100 lines JavaScript). Add as `jaxcad/ui/scene_widget.py` with the JavaScript inlined as a string. Requires `pip install anywidget scikit-image`.

### Phase 4 (optional)

FastAPI web app for sharing, or imgui-bundle for a desktop application with production-quality gizmos.

---

## Key architectural insight

The bridge from jaxCAD's parameter system to any UI is already there:

```python
from jaxcad.extraction import extract_parameters
from jaxcad.geometry.parameters import Vector, Scalar

free, _ = extract_parameters(scene)

# 3D drag handle positions:
vector_handles = [(p.name, p.value) for p in free.values() if isinstance(p, Vector)]

# Scalar slider ranges:
scalar_params  = [(p.name, p.value, p.bounds) for p in free.values() if isinstance(p, Scalar)]
```

When the user interacts:
```python
p.value = new_value              # update parameter
solved  = solve_constraints(scene)  # snap to constraint manifold (optional)
image   = render(scene)             # re-render
```

This three-line loop is the entire UI backend.
