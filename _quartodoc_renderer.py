"""A quartodoc renderer that can render a ``Yields:`` section.

quartodoc 0.11's markdown renderer dispatches on griffe's docstring section
types and implements every one cadjoint uses except
:class:`~griffe.DocstringSectionYields`, so ``quartodoc build`` dies with
``NotImplementedError: Unsupported type: DocstringSectionYields`` the moment
it reaches a generator or a context manager.  Six modules here have one,
including :func:`cadjoint.cache.enable_compilation_cache`'s neighbours and
the source-map capture helpers.

The alternative was to relabel those sections ``Returns:``, which would be
wrong: a context manager does not return the thing ``with`` binds, it yields
it, and the distinction is exactly what a reader of a fixture-shaped helper
needs.  So this fills the gap instead, mirroring quartodoc's own Returns
implementation — the item types have the same shape (name, annotation,
description), and reusing its table keeps the output identical in style.

Wired up through ``_quarto.yml``::

    quartodoc:
      renderer:
        style: _quartodoc_renderer.py

quartodoc resolves a ``.py`` style by importing it as a module from the
working directory, so this file lives at the repository root rather than
under ``tools/``: a path with a directory in it is not an importable module
name and ``quartodoc build`` fails before it renders anything.

Delete this when quartodoc renders Yields itself.
"""

from __future__ import annotations

from plum import dispatch
from quartodoc._griffe_compat import docstrings as ds
from quartodoc.renderers.md_renderer import MdRenderer, ParamRow


class Renderer(MdRenderer):
    """quartodoc's markdown renderer, plus the two Yields dispatches."""

    style = "yields"

    @dispatch
    def render(self, el: ds.DocstringSectionYields):
        """One table, the same shape Returns and Raises already produce."""
        rows = list(map(self.render, el.value))
        return self._render_table(rows, ["Name", "Type", "Description"], "returns")

    # plum dispatches on the annotation, so the repeated name is the point,
    # not a mistake; quartodoc's own renderer is written the same way.
    @dispatch
    def render(self, el: ds.DocstringYield):  # noqa: F811
        """One row: a yielded value has no default, like a returned one."""
        return ParamRow(
            el.name,
            el.description,
            annotation=self.render_annotation(el.annotation),
        )
