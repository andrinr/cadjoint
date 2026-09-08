"""Markdown tables for the experiment scripts."""

from __future__ import annotations

import math


def table(headers: list[str], rows: list[list]) -> str:
    """A GitHub-flavoured markdown table; floats are formatted by the caller."""
    cells = [[str(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |" for r in cells]
    return "\n".join([line, rule, *body])


def sci(x: float, digits: int = 2) -> str:
    """Signed scientific notation, e.g. ``-1.03e-03``."""
    return f"{x:+.{digits}e}"


def order(errors: list[float], ratios: list[float] | None = None) -> list[str]:
    """Observed convergence orders between successive levels, the first blank.

    Args:
        errors: One error per level, coarse to fine.
        ratios: h_coarse / h_fine between successive levels; halvings by default.
    """
    ratios = [2.0] * (len(errors) - 1) if ratios is None else ratios
    out = ["-"]
    for a, b, r in zip(errors[:-1], errors[1:], ratios):
        out.append("-" if a == 0 or b == 0 else f"{math.log(abs(a) / abs(b)) / math.log(r):.2f}")
    return out
