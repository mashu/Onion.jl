module Onion

using Flux

include("shared.jl")
include("AdaLN.jl")
include("RMSNorm.jl")
include("StarGLU.jl")

export
    #shared:
    glut,
    #layers:
    AdaLN,
    RMSNorm,
    StarGLU

end
