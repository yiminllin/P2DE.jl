"""
    neighbor_index(i, Nfp)

Compute the neighboring face (node, element) index `(iP, kP)` of the face node with global face node index `i`.
"""
function neighbor_index(i, Nfp)
    iP = mod1(i, Nfp)
    kP = div(i - 1, Nfp) + 1
    return iP, kP
end

"""
    neighbor_index(i, k, Nfp, mapP)

Given the face map `mapP`, of size `Nfp`x`K`, compute the neighboring face (node, element) index `(iP, kP)` of the face node with node index `i` and element index `k`.
"""
function neighbor_index(i, k, Nfp, mapP)
    return neighbor_index(mapP[i, k], Nfp)
end

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
