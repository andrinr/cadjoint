# zero-set protocol harness

Proves `research/zero-set.proto` end to end on cadjoint scenes: a cadjoint
model is lowered in the cadjoint venv, crosses as proto JSON into a second
venv that has never heard of cadjoint, and a reference backend there serves
`Discover`, `Solve`, `Evaluate` and `Vjp` over gRPC while the host runs the
schema's admission checks across the socket.

| file | side | what it does |
| --- | --- | --- |
| `lower.py` | cadjoint venv | lowers a scene to the node table; emits `model.json` and a JAX reference of values and ∂f/∂θ; `--per-node` lowers every instance alone |
| `backend.py` | scratch venv | numpy-only backend with its own forward-mode derivatives; grid discovery, Newton projection, gRPC service |
| `host.py` | scratch venv | the thirteen checks: strict parse, fidelity vs JAX, discover, solve, certification, Vjp/Jvp, finite differences, refresh |
| `check_nodes.py` | scratch venv | per-instance fidelity, naming any class whose lowering is wrong |

## Setup

```sh
uv venv /tmp/proto-venv && uv pip install --python /tmp/proto-venv/bin/python grpcio-tools numpy
mkdir -p /tmp/zs-stubs && /tmp/proto-venv/bin/python -m grpc_tools.protoc \
    --proto_path=research --python_out=/tmp/zs-stubs --grpc_python_out=/tmp/zs-stubs research/zero-set.proto
```

The cadjoint venv's protobuf runtime is older than that compiler's generated
code, so the cadjoint side emits proto3 JSON and the other side parses it
strictly with `json_format`; that is a feature, since it exercises the
language-neutral encoding.

## Run

```sh
# the sample bracket-like scene
.venv/bin/python research/zeroset_test/lower.py /tmp/zs
PYTHONPATH=/tmp/zs-stubs:research/zeroset_test /tmp/proto-venv/bin/python research/zeroset_test/host.py /tmp/zs

# any scene, e.g. the motor shield (its discovery is slow on this naive backend: use a coarse grid)
.venv/bin/python research/zeroset_test/lower.py /tmp/zs-motor scenes/motor_shield.py
PYTHONPATH=/tmp/zs-stubs:research/zeroset_test /tmp/proto-venv/bin/python research/zeroset_test/host.py /tmp/zs-motor --cells 16

# which node class lowers wrongly, if any
.venv/bin/python research/zeroset_test/lower.py /tmp/zs-motor scenes/motor_shield.py --per-node
PYTHONPATH=/tmp/zs-stubs:research/zeroset_test /tmp/proto-venv/bin/python research/zeroset_test/check_nodes.py /tmp/zs-motor
```

## What it has shown (2026-09-07)

- The sample scene passes all thirteen checks: values within 5e-8 and ∂f/∂θ within 2e-6 of JAX at every point, 3526 certified points, backend Vjp identical to the host's, forward derivative within 1e-6 of central differences, topology hash stable under a parameter change.
- The motor shield's 154 node instances all lower exactly; the whole model matches JAX to 4e-7 in value and 3e-6 in ∂θ over 77 parameters, on 140 KB of wire.
- The backend is deliberately naive. Its grid discovery takes minutes on the motor shield and over-seeds edges; a real backend replaces it, the protocol does not change.
