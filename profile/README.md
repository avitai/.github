# Avitai Bio

> A JAX/Flax NNX scientific ML stack — from data pipelines and evaluation primitives up through differentiable bioinformatics.

Avitai Bio develops a layered, open-source ecosystem for scientific machine learning, built natively on [JAX](https://jax.readthedocs.io/) and [Flax NNX](https://flax.readthedocs.io/). Each layer is a standalone library; together they compose into end-to-end differentiable pipelines for the life sciences.

---

## The stack

```
┌──────────────────────────────────────────────────────┐
│  DiffBio   — differentiable bioinformatics pipelines │
├──────────────────────────────────────────────────────┤
│  Artifex   │  Opifex                                 │
│  generative│  scientific ML                          │
│  models    │  (PINNs, neural operators, discovery)   │
├──────────────────────────────────────────────────────┤
│  Datarax   — differentiable data pipelines           │
│  Calibrax  — benchmarking, profiling, metrics        │
├──────────────────────────────────────────────────────┤
│              JAX  ·  Flax NNX  ·  XLA                │
└──────────────────────────────────────────────────────┘
```

### Foundation

- **[datarax](https://github.com/avitai/datarax)** — A differentiable data pipeline framework for JAX. DAG-based execution, deterministic shuffling, multi-device sharding, and gradients that flow through every operator. Built so preprocessing, augmentation, and even synthesis can be optimized end-to-end.
- **[calibrax](https://github.com/avitai/calibrax)** — Unified benchmarking, profiling, and metrics for the JAX scientific ML ecosystem. 110+ registered metrics across 17 domains, plus timing, GPU/energy monitoring, FLOP counting, roofline analysis, regression detection, and CI integration.

### Modeling substrate

- **[artifex](https://github.com/avitai/artifex)** — A research-focused modular generative modeling library. VAEs, GANs, diffusion, normalizing flows, energy-based, autoregressive, and geometric models, with unified interfaces across images, text, audio, proteins, tabular, and time series.
- **[opifex](https://github.com/avitai/opifex)** — A unified scientific machine learning framework. Neural operators (FNO, DeepONet, SFNO, and 20+ more), physics-informed neural networks, equation discovery (SINDy), uncertainty quantification, and quantum chemistry — all probabilistic-first and JAX-native.

### Application layer

- **[DiffBio](https://github.com/avitai/DiffBio)** — End-to-end differentiable bioinformatics pipelines. 40+ differentiable operators (alignment, variant calling, single-cell, epigenomics, RNA-seq, drug discovery, protein/RNA structure) composed into trainable pipelines. Replaces hard thresholds and argmax decisions with continuous relaxations so entire workflows can be optimized by gradient descent.

---

## Why a stack, not a monolith

Each layer is independently useful — you can use Datarax without ever touching biology, or Calibrax to benchmark unrelated JAX projects. But the layers are designed to compose: DiffBio uses Datarax's operator contracts, Artifex's modeling components, Opifex's optimization primitives, and Calibrax's evaluation harness. The goal is a coherent, type-checked, JIT-compiled path from raw data through model to peer-reviewed metric — without leaving JAX.

## Status

Most projects are in early, active development with unstable APIs. They are suitable for research and experimentation today; pin to specific commits if you need stability. See individual repository READMEs for current status, install instructions, and quickstarts.

## License

All projects are released under the MIT License.
