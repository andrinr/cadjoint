"""Everything that runs *inside* the compile worker, and nothing that does not.

The viewer is two programs. This one is the child: it executes the user's
scene program, and turns it into whatever the request asked for — the
shaders and construction tree (``main``), a feature-edge overlay, a
simulation (``fem``), an optimizer run (``optimize``), a file (``export``).
It is spawned per request, or kept and fed several, and either way it is the
only place a scene's arbitrary Python is ever executed.

The other program is the server, in :mod:`cadjoint.viewer.playground` and
:mod:`cadjoint.viewer._worker_client`, which spawns this one, supervises it
and cancels it. The two share :mod:`cadjoint.viewer.worker.protocol` and
nothing else of their own.

Run as ``python -m cadjoint.viewer.worker``; ``--serve`` answers NDJSON
requests until stdin closes instead of one JSON request and exiting.
"""

from __future__ import annotations

from cadjoint.viewer.worker.protocol import MODULE, RETIRE_FLAG

__all__ = ["MODULE", "RETIRE_FLAG"]
