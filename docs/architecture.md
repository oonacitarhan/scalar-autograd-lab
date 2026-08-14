# Architecture

Value nodes store parents and local backward closures. A topological traversal propagates gradients once per graph node. The MLP composes scalar operations into tanh layers; full-batch squared-error training updates weights and biases.

## Data flow

- `scalargrad/core.py` — autodiff primitives.
- `scalargrad/features.py` — network and training APIs.
- `scalargrad/cli.py` — command-line experiment.
- `tests/` — numerical gradient, training and CLI checks.
- `examples/` — XOR fixture.
- `benchmarks/` — timed training workload.

## Limits

This is an educational scalar engine, not a tensor library. It has no GPU, batching kernels, or optimizer state. XOR is a training demonstration, not a held-out ML benchmark. Neural network training is sensitive to initialization; the example uses seed 0.
