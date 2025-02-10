module Onion

using Flux

include("shared.jl")
include("AdaLN.jl")

export
    #shared:
    glut,
    #layers:
    AdaLN
end

end
