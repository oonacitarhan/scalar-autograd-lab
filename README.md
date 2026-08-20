# Scalar Autograd Lab

A scalar reverse-mode autodiff engine with a small neural network, numerical gradient checks, and an XOR training example.

**Python** · Reusable core · CLI examples · Regression checks · Reproducible benchmark

> **Provenance:** created with AI assistance on **2026-11-01**.
> All commit author and committer dates are synthetic and distributed across
> November 2024–October 2026. They do not establish work on those dates.

## Quick start

```sh
python3 -m scalargrad --steps 1000 --seed 0
```

## Repository map

- `scalargrad/core.py` — autodiff primitives.
- `scalargrad/features.py` — network and training APIs.
- `scalargrad/cli.py` — command-line experiment.
- `tests/` — numerical gradient, training and CLI checks.
- `examples/` — XOR fixture.
- `benchmarks/` — timed training workload.

## How it works

Value nodes store parents and local backward closures. A topological traversal propagates gradients once per graph node. The MLP composes scalar operations into tanh layers; full-batch squared-error training updates weights and biases.

## Verification

```sh
sh scripts/check.sh
```

Seven test groups plus the original reference Python checks exercise reference cases, invalid inputs, and relevant invariants.
The complete local run is in [docs/verification.txt](docs/verification.txt).
The [checks workflow](.github/workflows/check.yml) executes the same source checks.

## Boundaries

This is an educational scalar engine, not a tensor library. It has no GPU, batching kernels, or optimizer state. XOR is a training demonstration, not a held-out ML benchmark. Neural network training is sensitive to initialization; the example uses seed 0.

See [architecture](docs/architecture.md), [usage](docs/usage.md),
[release notes](CHANGELOG.md), and [provenance](PROVENANCE.json).

## License

MIT — [LICENSE](LICENSE).
