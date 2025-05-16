```@meta
CurrentModule = Onion
```

# Onion.jl

Documentation for [Onion](https://github.com/MurrellGroup/Onion.jl), a Julia package providing various neural network components.

## Package Organization

Onion is now organized into modules:

- **Main module (`Onion`)**: Contains general-purpose neural network components like attention mechanisms, normalization layers, and other utilities.

- **UNet submodule (`Onion.Unet`)**: Contains all UNet-related components organized in a separate namespace. This includes:
  - Basic building blocks: `ResidualBlock`, `EncoderBlock`, `DecoderBlock`, `Bottleneck`, etc.
  - `FlexibleUNet`: A configurable UNet implementation 
  - Various utility functions for UNet construction

## Usage

To use components from the main module:

```julia
using Onion

# Using components from the main module
attention = Onion.Attention(dim=64, heads=8)
```

To use UNet components:

```julia
using Onion

# UNet components are accessed through the Unet submodule
model = Onion.Unet.FlexibleUNet(
    in_channels=3,
    out_channels=3,
    base_channels=64,
    channel_multipliers=[1, 2, 4, 8]
)
```

```@index
```

```@autodocs
Modules = [Onion]
```
