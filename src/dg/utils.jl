function check_conservation(state, solver)
    (; Uq) = state.preallocation
    (; K, Nq) = solver.discrete_data.sizes

    total = zero(Uq[1])
    for k = 1:K
        for i = 1:Nq
            total += solver.discrete_data.geom.J[i, k] * solver.discrete_data.ops.wq[i] * Uq[i, k]
        end
    end
    return sum(total)
end
