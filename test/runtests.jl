using Onion
using Test
using Flux

const ONION_TEST_UNET = get(ENV, "ONION_TEST_UNET", "false") == "true"

@testset "Onion.jl" verbose=true begin

    include("BlockLinear.jl")
    include("Modulator.jl")
    include("DyT.jl")

    include("embedding/embedding.jl")
    include("norm/norm.jl")
    include("transformers/transformers.jl")
    
    if ONION_TEST_UNET
        include("UNet/test_unet.jl")
    else
        @info "Skipping UNet tests"
    end

end
