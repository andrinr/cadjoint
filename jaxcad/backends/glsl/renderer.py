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
        self._ctx = moderngl.create_standalone_context(require=430)

    def __enter__(self) -> GLSLRenderer:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying OpenGL context."""
        if self._ctx is not None:
            self._ctx.release()
            self._ctx = None

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
        if h <= 0 or w <= 0:
            raise ValueError("resolution dimensions must be positive")
        camera_pos = np.asarray(camera_pos, dtype=np.float32)
        camera_target = np.asarray(camera_target, dtype=np.float32)
        if camera_pos.shape != (3,) or camera_target.shape != (3,):
            raise ValueError("camera_pos and camera_target must each have shape (3,)")
        if not np.isfinite(camera_pos).all() or not np.isfinite(camera_target).all():
            raise ValueError("camera_pos and camera_target must contain only finite values")
        if np.array_equal(camera_pos, camera_target):
            raise ValueError("camera_pos and camera_target must differ")

        ld = np.asarray(light_dir if light_dir is not None else [1.0, 2.0, 3.0], dtype=np.float32)
        if ld.shape != (3,) or not np.isfinite(ld).all() or not np.any(ld):
            raise ValueError("light_dir must be a finite, non-zero vector with shape (3,)")

        bg = np.asarray(bg_color if bg_color is not None else np.zeros(3), dtype=np.float32)
        if bg.shape != (3,) or not np.isfinite(bg).all() or np.any((bg < 0) | (bg > 1)):
            raise ValueError("bg_color must have shape (3,) with finite values between 0 and 1")

        ctx = self._ctx
        if ctx is None:
            raise RuntimeError("GLSLRenderer is closed")

        prog = ctx.program(vertex_shader=_VERT, fragment_shader=fragment_shader)

        def _set(name: str, *vals):
            if name in prog:
                prog[name].value = tuple(vals) if len(vals) > 1 else vals[0]

        _set("iResolution", float(w), float(h))
        _set("uCameraPos", *[float(v) for v in camera_pos])
        _set("uCameraTarget", *[float(v) for v in camera_target])

        _set("uLightDir", *[float(v) for v in ld])

        _set("uBgColor", *[float(v) for v in bg])

        vbo = ctx.buffer(_QUAD.tobytes())
        vao = ctx.simple_vertex_array(prog, vbo, "position")
        tex = ctx.texture((w, h), 3, dtype="f4")
        fbo = ctx.framebuffer(color_attachments=[tex])
        try:
            fbo.use()
            ctx.clear(0.0, 0.0, 0.0)
            vao.render(moderngl.TRIANGLE_STRIP)

            raw = np.frombuffer(fbo.read(components=3, dtype="f4"), dtype=np.float32)
            image = raw.reshape(h, w, 3)[::-1]  # flip Y (OpenGL origin = bottom-left)
            return np.clip(image, 0.0, 1.0)
        finally:
            fbo.release()
            tex.release()
            vao.release()
            vbo.release()
            prog.release()
