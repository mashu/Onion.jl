@testset "Modulator.jl" begin

    @testset "Single output dimension" begin
        modulator = Modulator(6 => 7)
        Y = rand(Float32, 7, 5)
        X = rand(Float32, 6, 5)
        @test modulator(Y, X) |> size == (7, 5)
    end

    @testset "Multiple output dimensions" begin
        Y = rand(Float32, 5, 7, 5)
        X = rand(Float32, 6, 5)

        @test Modulator(6 => (5, 7))(Y, X) |> size == (5, 7, 5)
        @test Modulator(6 => (1, 7))(Y, X) |> size == (5, 7, 5)
        @test Modulator(6 => (5, 1))(Y, X) |> size == (5, 7, 5)
    end

end
