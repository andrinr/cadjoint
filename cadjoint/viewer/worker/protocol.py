"""The wire between the server and the worker it spawns.

Both sides need these and neither owns them, so they live here rather than
in one importing the other's implementation to read a constant.
"""

from __future__ import annotations

__all__ = ["MODULE", "RETIRE_FLAG"]

#: What the server runs as a child: ``python -m <MODULE>``, or with
#: ``--serve`` for a worker that answers more than one request.
MODULE = "cadjoint.viewer.worker"

#: Response field a served worker sets on its final answer, telling the
#: client this process is finished and must not receive another request.
#:
#: Told rather than inferred: polling for the child's exit races it, and a
#: request sent to a process on its way out is lost.
RETIRE_FLAG = "__retire__"
