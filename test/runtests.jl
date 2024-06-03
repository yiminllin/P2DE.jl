using P2DE
using Test

@testset "P2DE.jl" begin
    # Write your tests here.
    @time begin
        @test true
        run(`$(Base.julia_cmd()) --threads=2 $(abspath("../examples/convergence/isentropic-vortex-convergence.jl"))`)
    end
end
