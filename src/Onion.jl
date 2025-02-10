module Onion

using Flux, LinearAlgebra

include("shared.jl")
include("AdaLN.jl")
include("RMSNorm.jl")
include("StarGLU.jl")
include("GQAttention.jl")
include("RoPE.jl")

export
    #shared:
    glut,
    #layers:
    AdaLN,
    RMSNorm,
    StarGLU,
    GQAttention,
    RoPE,
    TransformerBlock,
    Attention

end
