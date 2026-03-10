# Onion

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Onion.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Onion.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Onion.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Onion.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Onion.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Onion.jl)

## Installation

```julia
using Pkg
Registry.add("https://github.com/MurrellGroup/MurrellGroupRegistry")
Pkg.add("Onion")
```

## Release Process

This repository contains three packages registered in `MurrellGroupRegistry`:

- `Onion` from the repository root
- `OnionCore` from `lib/OnionCore`
- `OnionStyle` from `lib/OnionStyle`

To release one of them:

1. Bump the version in that package's `Project.toml`.
2. Merge the release commit to `main`.
3. Trigger Registrator for the package you are releasing.

Examples:

```text
@JuliaRegistrator register registry=MurrellGroup/MurrellGroupRegistry
@JuliaRegistrator register registry=MurrellGroup/MurrellGroupRegistry subdir=lib/OnionCore
@JuliaRegistrator register registry=MurrellGroup/MurrellGroupRegistry subdir=lib/OnionStyle
```

TagBot creates the matching GitHub release after the registry PR is merged.
