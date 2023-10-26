# Overview

Simple PDE solvers in 2D. Only for research and educational purposes, and only DG for compressible flow now.

# Run an example
```console
julia --project=@. --threads=24 examples/2D/astro-jet.jl
```
or anything in ```examples/```

# Configure an example
by changing ```param``` in scripts:
```julia
struct Param
    N     # Degree of approximation polynomial
    K     # Number of elements in 1D.
          # In 2D, it is a tuple (Kx,Ky), number of elements along
          # each direction

    xL    # In 1D, the domain is [xL,xR]
    xR    # In 2D, the domain is [xL[1],xR[1]] x [xL[2],xR[2]]

    global_constants
    timestepping_param
    limiting_param
    postprocessing_param

    equation                    # CompressibleEulerIdealGas,
                                # CompressibleNavierStokesIdealGas
    approximation_basis_type    # GaussCollocation,
                                # LobattoCollocation
    rhs_type                    # LowOrderPositivity
                                # EntropyStable
                                # StandardDG
                                # ESLimitedLowOrderPos
                                # StdDGLimitedLowOrderPos
    entropyproj_limiter_type    # NoEntropyProjectionLimiter
                                # NodewiseScaledExtrapolation
    rhs_limiter_type            # NoRHSLimiter
                                # SubcellLimiter
                                # ZhangShuLimiter

                                # rhs_limiter_type.bound_type
                                # PositivityBound
                                # PositivityAndMinEntropyBound
                                # PositivityAndRelaxedMinEntropyBound
                                # PositivityAndCellEntropyBound
                                # PositivityAndRelaxedCellEntropyBound
                                # TVDBound
                                # TVDAndMinEntropyBound
                                # TVDAndRelaxedMinEntropyBound
                                # TVDAndCellEntropyBound
                                # TVDAndRelaxedCellEntropyBound
end
```