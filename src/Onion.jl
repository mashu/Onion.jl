module Onion

using Flux, LinearAlgebra, BatchedTransformations

include("shared.jl")
include("AdaLN.jl")
include("RMSNorm.jl")
include("DyT.jl")
include("StarGLU.jl")
include("GQAttention.jl")
include("RoPE.jl")
include("Unet/Unet.jl")
include("FSQ.jl")
include("IPAHelpers.jl")

# Export Unet as a submodule
export Unet

export
    #shared:
    glut,
    #layers:
    AdaLN,
    RMSNorm,
    DyT,
    StarGLU,
    GQAttention,
    RoPE,
    TransformerBlock,
    Attention,
    FSQ,
    Framemover,
    IPAblock,
    pair_encode,
    chunk,
    unchunk,
    self_att_padding_mask,
    cross_att_padding_mask
    
# UNet components are now accessed via Onion.Unet namespace
# This keeps documentation organized by module structure
end
