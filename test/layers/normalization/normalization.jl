@testset "norm.jl" begin

    @testset "Derf" begin
        derf = Derf(256)
        x = randn(Float32, 256, 5)
        y = derf(x)
        @test size(y) == size(x)
    end

    @testset "DyT" begin
        dyt = DyT(256)
        x = randn(Float32, 256, 2)
        y = dyt(x)
        @test size(y) == size(x)
    end

    @testset "LpNorm" begin
        v = randn(Float32, 40, 50)
        atol = 1f-2
        @testset for p in 1:5, dims in [(1,2), 1, ()]
            s = sum(x -> abs(x)^p, LpNorm(p; dims)(v); dims)
            @test all(x -> isapprox(abs(x), 1f0; atol), s)
        end
    end

end
