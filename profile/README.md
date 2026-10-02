# Avitai

**A differentiable, JAX-native foundation for scientific machine learning.**

Avitai Bio builds virtual-cell software. The **Avitai stack** is the open, domain-agnostic
foundation we built in order to do it, and it works anywhere, not only in biology. Five
foundation libraries, plus two end-to-end applications in domains chosen to be as different from
each other as possible: a cell, and a self-driving car.

Everything here is MIT, pre-1.0, and actively developed in the open.

---

## The stack

```
┌──────────────────────────────────────────────────────────────┐
│  DiffBio                    │  DiffAV                        │
│  differentiable genomics    │  differentiable AV safety      │
├──────────────────────────────────────────────────────────────┤
│  opifex     scientific ML: PDEs, neural operators, PINNs,    │
│             quantum chemistry, UQ            (the capstone)  │
├──────────────────────────────────────────────────────────────┤
│  artifex    generative modeling: diffusion, VAE, flows,      │
│             energy-based, across images, audio, proteins     │
├──────────────────────────────────────────────────────────────┤
│  datarax    differentiable data pipelines                    │
├──────────────────────────────────────────────────────────────┤
│  calibrax   metrics, benchmarking, profiling   (measurement) │
├──────────────────────────────────────────────────────────────┤
│  substrax   training plumbing: devices, sharding, restarts   │
├──────────────────────────────────────────────────────────────┤
│              built on JAX, Flax NNX and XLA                  │
└──────────────────────────────────────────────────────────────┘
```

Each layer depends only on the layers below it. `substrax` is the bottom of the chain, depends on
none of the others, and is the cleanest place to start.

## Install

Versions are the latest releases as of 2026-10-02, and every count below was measured against
them rather than copied forward.

| Repo | Install | Latest | What it is |
|---|---|---|---|
| [substrax](https://github.com/avitai/substrax) | `pip install substrax` | 0.1.20 | Training plumbing: process setup, device discovery and sharding, checkpointing, stopping rules, run logging. The bottom of the chain. |
| [calibrax](https://github.com/avitai/calibrax) | `pip install calibrax` | 0.1.14 | 140 metrics across 20 domains, plus FLOPs, roofline, and energy profiling. The measurement layer. |
| [datarax](https://github.com/avitai/datarax) | `pip install datarax` | 0.1.17 | Differentiable data pipelines. Gradients flow through preprocessing. |
| [artifex](https://github.com/avitai/artifex) | `pip install avitai-artifex` | 0.1.15 | Modular generative modeling. Note the package name: bare `artifex` on PyPI is an unrelated project. |
| [opifex](https://github.com/avitai/opifex) | `pip install opifex` | 0.2.10 | Unified scientific ML. The capstone. |
| [DiffBio](https://github.com/avitai/DiffBio) | `pip install diffbio` | 0.1.9 | End-to-end differentiable bioinformatics pipelines. |
| [DiffAV](https://github.com/avitai/DiffAV) | `pip install git+https://github.com/avitai/DiffAV` | unreleased | Physics-informed, RL-aligned autonomous-driving safety evaluation. |

The whole foundation in one line:

```bash
pip install substrax calibrax datarax avitai-artifex opifex
```

## Foundation

- **[substrax](https://github.com/avitai/substrax)** is the training plumbing every other library
  here would otherwise write for itself: JAX process configuration, device discovery and array
  sharding, checkpoint save and restore, stopping rules, run logging, and job submission to remote
  compute. It is the only library here that depends on none of the others, which makes it the
  bottom of the chain.

- **[calibrax](https://github.com/avitai/calibrax)** is the shared measurement layer, built on
  `substrax`. It registers **140 metrics across 20
  domains** (audio, calibration, classification, clustering, distance, divergence, fairness,
  forecasting, general, generative, geometric, graph, image, information, manifold, ranking,
  segmentation, statistical, text, uncertainty), with representative ones checked against scikit-learn and SciPy references at 1e-6. It
  also carries timing, GPU and energy monitoring, FLOP counting, roofline analysis and regression
  detection. Do not take the count on trust:

  ```python
  from calibrax.metrics import MetricRegistry
  print(len(MetricRegistry().list_names()))   # 140
  ```

- **[datarax](https://github.com/avitai/datarax)** is a differentiable data-pipeline framework for
  JAX. Every stage is a Flax NNX module, so gradients flow through preprocessing and augmentation
  rather than stopping at the dataloader. DAG-based execution with caching and differentiable
  rebatching, multi-device sharding, deterministic O(1)-memory shuffling, and exact mid-epoch
  resume.

- **[artifex](https://github.com/avitai/artifex)** is a modular generative-modeling library. VAEs,
  GANs, diffusion, normalizing flows, energy-based, autoregressive and geometric models behind one
  typed interface, across images, text, audio, proteins, tabular data and time series.

- **[opifex](https://github.com/avitai/opifex)** is the scientific-ML capstone: neural operators
  (FNO, DeepONet, SFNO and more), physics-informed neural networks, equation discovery, quantum
  chemistry, atomistic molecular dynamics, and uncertainty quantification, probabilistic-first
  throughout.

## Applications

Two repos take the same four libraries all the way into finished domains. They exist to answer
one question honestly: does the foundation actually generalise?

- **[DiffBio](https://github.com/avitai/DiffBio)** makes a genomics pipeline trainable. Hard
  thresholds, argmax and Smith-Waterman all block gradients, which is why bioinformatics pipelines
  get hand-tuned instead of learned. DiffBio replaces them with differentiable relaxations (soft
  quality filtering, temperature-softmax pileup, continuous Smith-Waterman), so variant calling,
  single-cell, RNA-seq, multi-omics and perturbation pipelines become one differentiable function
  you can optimize against a downstream objective. 40+ operators, 6 named pipelines.

- **[DiffAV](https://github.com/avitai/DiffAV)** makes autonomous-driving safety evaluation
  differentiable. A physics-constrained (bicycle-model) diffusion world-model predicts multi-agent
  trajectories, then RL-aligned adversarial steering pushes a scenario's adversary toward its
  victim while keeping the result feasible. In the measured result it moves the adversary from
  7.30 m to 2.88 m from its victim while off-road feasibility **improves**, 0.21 to 0.08.

## What this is not

Being specific here is more useful than being impressive.

- **Pre-1.0.** APIs will change without deprecation cycles. Pin a version if you need stability.
- **DiffAV is a research scaffold, not a leaderboard entry.** Its trajectory minADE6 is about
  5.6 m, scored with JAX-native proxies for the WOSAC metrics rather than the official
  implementation, so it is not comparable with leaderboard entries. The adversarial-steering
  behaviour is the result. It is not state of the art at trajectory prediction and we do not claim
  it is.
- **The domain examples are worked tutorials, not hardened solutions.** The foundation ships
  example suites spanning physics and PDEs, neural operators, quantum chemistry, molecular
  dynamics, vision, audio and generative modeling. They demonstrate reach; they are not
  production pipelines.

## Bring your domain

The foundation carries physics, chemistry, molecular dynamics, vision and generative examples out
of the box, and two repos here take biology and autonomous driving all the way. If the same
substrate would be useful in your field, open an issue and tell us what breaks. Early feedback
genuinely steers what gets built next.

## License

MIT, across every repository.
