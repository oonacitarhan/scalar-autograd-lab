# Architecture

Value nodes store parents and local backward closures. A topological traversal propagates gradients once per graph node. The MLP composes scalar operations into tanh layers; full-batch squared-error training updates weights and biases.

## Data flow

- `scalargrad/core.py` — algorithm primitives.
- `scalargrad/features.py` — experiment and analysis APIs.
- `scalargrad/cli.py` — argument parsing and JSON output.
- `examples/` — small reproducible inputs.
- `tests/` — API, numerical, and CLI checks.
- `benchmarks/run.py` — timed representative workload.

## Limits

This is an educational scalar engine, not a tensor library. It has no GPU, batching kernels, or optimizer state. XOR is a training demonstration, not a held-out ML benchmark.
