"""Offscreen GLSL renderer using moderngl.

Install the optional dep with:  pip install jaxcad[glsl]
"""

from __future__ import annotations

import numpy as np

try:
    import moderngl

    _HAS_MODERNGL = True
except ImportError:
    _HAS_MODERNGL = False

_VERT = """\
#version 430
in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }
"""

_QUAD = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)


class GLSLRenderer:
    """Renders a compiled GLSL fragment shader offscreen via moderngl."""

    def __init__(self) -> None:
        if not _HAS_MODERNGL:
            raise ImportError(
                "moderngl is required for GLSL rendering.\n"
                "Install it with:  pip install jaxcad[glsl]"
            )
        self._ctx = moderngl.create_standalone_context()

    def render(
        self,
        fragment_shader: str,
        camera_pos: np.ndarray,
        camera_target: np.ndarray,
        resolution: tuple[int, int],
        light_dir: np.ndarray | None = None,
        bg_color: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render *fragment_shader* offscreen and return an ``(H, W, 3)`` image.

        Args:
            fragment_shader: Complete GLSL 4.30 fragment shader source.
            camera_pos: Camera position in world space, shape ``(3,)``.
            camera_target: Camera look-at point, shape ``(3,)``.
            resolution: Output ``(height, width)``.
            light_dir: World-space light direction, defaults to ``[1, 2, 3]``.
            bg_color: Background RGB, defaults to black.

        Returns:
            ``float32`` numpy array of shape ``(H, W, 3)`` in ``[0, 1]``.
        """
        h, w = resolution
        ctx = self._ctx

        prog = ctx.program(vertex_shader=_VERT, fragment_shader=fragment_shader)

        def _set(name: str, *vals):
            if name in prog:
                prog[name].value = tuple(vals) if len(vals) > 1 else vals[0]

        _set("iResolution", float(w), float(h))
        _set("uCameraPos", *[float(v) for v in camera_pos])
        _set("uCameraTarget", *[float(v) for v in camera_target])

        ld = light_dir if light_dir is not None else np.array([1.0, 2.0, 3.0])
        _set("uLightDir", *[float(v) for v in ld])

        bg = bg_color if bg_color is not None else np.zeros(3)
        _set("uBgColor", *[float(v) for v in bg])

        vbo = ctx.buffer(_QUAD.tobytes())
        vao = ctx.simple_vertex_array(prog, vbo, "position")

        tex = ctx.texture((w, h), 3, dtype="f4")
        fbo = ctx.framebuffer(color_attachments=[tex])
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0)
        vao.render(moderngl.TRIANGLE_STRIP)

        raw = np.frombuffer(fbo.read(components=3, dtype="f4"), dtype=np.float32)
        image = raw.reshape(h, w, 3)[::-1]  # flip Y (OpenGL origin = bottom-left)
        return np.clip(image, 0.0, 1.0)
