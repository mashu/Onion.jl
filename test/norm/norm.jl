@testset "norm.jl" begin

    @testset "LpNorm" begin
        v = randn(Float32, 10, 20)
        rtol = 1f-2
        @testset for p in 1:5, dims in [(1,2), 1, ()]
            s = sum(x -> abs(x)^p, LpNorm(p; dims)(v); dims)
            @test all(x -> isapprox(x, 1f0; rtol), s)
        end
    end

end
