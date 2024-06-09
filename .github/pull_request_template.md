## Description

## Issue number

## Checklist before landing
- [ ] Run Integration/Convergence Tests
    - [ ] `julia --project=@. --threads=30 examples/convergence/isentropic-vortex-convergence.jl`
    - [ ] `julia --project=@. --threads=30 examples/convergence/leblanc-convergence.jl`
    - [ ] `julia --project=@. --threads=30 examples/convergence/wall-convergence.jl`
