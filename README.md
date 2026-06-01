# Onion

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Onion.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Onion.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Onion.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Onion.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Onion.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Onion.jl)

Composable deep-learning layers for Julia — the building blocks we keep reaching for, in one place.

Onion collects modern neural-network components (transformers, attention, normalization, rotary embeddings)
alongside structural-biology layers (invariant point attention, pairwise and triangle operations, ESMFold-style folding)
and exposes them through a small, swappable backend system so the same model can run on a portable fallback or a tuned CUDA kernel.

## Quick start

```julia
using Onion

dim, n_heads, n_kv_heads, seqlen = 64, 8, 4, 10

block = TransformerBlock(dim, n_heads, n_kv_heads)
rope  = RoPE(dim ÷ n_heads, 1000)

h = randn(Float32, dim, seqlen, 1)

h = block(h, 1, rope[1:seqlen])
```

## Overview

- **Attention & transformers** — `Attention` (GQA, self/cross), `TransformerBlock`, `AdaTransformerBlock`, `KVCache`, `RoPE` / `MultidimRoPE` / `STRINGRoPE`.
- **Norms & feed-forward** — `RMSNorm`, `LayerNorm`, `AdaLN`, `DyT`, `L2Norm`, `StarGLU`, and more.
- **Composability** — `Composed`, `SkipConnection`, `ResidualConnection`, `GeneralizedHyperConnection`, `VirtualWidthNetwork`.
- **Structural biology** — `IPAblock`, `CrossFrameIPA`, `Framemover`, plus pairwise/triangle layers (`PairformerLayer`, `TriangleAttention`, `TriangleMultiplicativeUpdate`, `OuterProductMean`, `AttentionPairBias`) for AlphaFold/ESMFold-style models.

## Backends

Core operations are written as **primitives** that dispatch on a backend, so you can move work to a faster implementation without touching your model code:

```julia
withbackend(cuTileBackend()) do
    block(h, 1, rope[1:seqlen])
end
```

| Backend | Notes |
| --- | --- |
| `DefaultBackend` | Portable CPU/GPU implementations (the default). |
| `NNopBackend` | KernelAbstractions-based kernels via [NNkernels.jl](https://github.com/FluxML/NNkernels.jl). |
| `cuTileBackend` | cuTile-generated CUDA kernels. |

Backends only implement the primitives they specialize; everything else falls through to `DefaultBackend`. See [`src/primitives/README.md`](src/primitives/README.md) for the dispatch model and how to add your own.

## Installation

```julia
using Pkg
Pkg.Registry.add(url="https://github.com/MurrellGroup/MurrellGroupRegistry")
Pkg.add("Onion")
```
