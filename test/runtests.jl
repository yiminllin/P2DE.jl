using P2DE
using Test

@testset "P2DE.jl" begin
    @testset "Smoke Test - Single Thread" begin
        @time begin
            @test true
            run(`$(Base.julia_cmd()) --threads=1 $(abspath("test_smoke.jl"))`)
        end
    end
    @testset "Smoke Test - Single Thread" begin
        @time begin
            @test true
            run(`$(Base.julia_cmd()) --threads=2 $(abspath("test_smoke.jl"))`)
        end
    end
end
