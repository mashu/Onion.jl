module Utils

using ArrayInterface: restructure
using ChainRulesCore
using Einops
using LinearAlgebra

include("glut.jl")
export glut

include("reshapable.jl")
export reshapable, hardreshape

include("like.jl")
export like, zeros_like, ones_like, falses_like, trues_like

include("watmul.jl")
export watmul, ⨝

include("add_something.jl")
export ⊞

include("ofeltype.jl")
export ofeltype

include("masks.jl")
export self_att_padding_mask
export cross_att_padding_mask
export causal_mask

end
