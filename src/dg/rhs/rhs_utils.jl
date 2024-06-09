function reference_to_physical(dim::Dim1, U, GJ)
    Ur = U[1]
    rxJ = GJ[1]
    return SVector(rxJ * Ur,)
end

function reference_to_physical(dim::Dim2, U, GJ)
    Ur, Us = U
    rxJ, sxJ, ryJ, syJ = GJ
    return SVector(rxJ * Ur + sxJ * Us, ryJ * Ur + syJ * Us)
end

function Bx(dim::Dim1, i, k, solver)
    (; Br) = solver.discrete_data.ops
    (; rxJh) = solver.discrete_data.geom

    iface = i + solver.discrete_data.sizes.Nq
    return reference_to_physical((Br[i, i],), (rxJh[iface, k],), dim)
end

function Bx(dim::Dim2, i, k, solver)
    (; Br, Bs) = solver.discrete_data.ops
    (; rxJh, ryJh, sxJh, syJh) = solver.discrete_data.geom

    iface = i + solver.discrete_data.sizes.Nq
    return reference_to_physical(dim, (Br[i, i], Bs[i, i]), (rxJh[iface, k], sxJh[iface, k], ryJh[iface, k], syJh[iface, k]))
end

function Bx_with_n(dim::Dim1, i, k, solver)
    Bx_i = Bx(dim, i, k, solver)
    return Bx_i, abs(Bx_i[1])
end

function Bx_with_n(dim::Dim2, i, k, solver)
    Bx_i, By_i = Bx(dim, i, k, solver)
    return SVector(Bx_i, By_i), norm((Bx_i, By_i))
end

function Sx(dim::Dim1, i, j, k, solver)
    (; Srh_db) = solver.discrete_data.ops
    (; rxJh) = solver.discrete_data.geom

    return reference_to_physical(dim, (Srh_db[i, j],), (rxJh[i, k],))
end

function Sx(dim::Dim2, i, j, k, solver)
    (; Srh_db, Ssh_db) = solver.discrete_data.ops
    (; rxJh, ryJh, sxJh, syJh) = solver.discrete_data.geom

    return reference_to_physical(dim, (Srh_db[i, j], Ssh_db[i, j]), (rxJh[i, k], sxJh[i, k], ryJh[i, k], syJh[i, k]))
end

function Sx0(dim::Dim1, i, j, k, solver)
    (; Sr0) = solver.discrete_data.ops
    (; rxJh) = solver.discrete_data.geom

    return reference_to_physical(dim, (Sr0[i, j],), (rxJh[i, k],))
end

function Sx0(dim::Dim2, i, j, k, solver)
    (; Sr0, Ss0) = solver.discrete_data.ops
    (; rxJh, ryJh, sxJh, syJh) = solver.discrete_data.geom

    return reference_to_physical(dim, (Sr0[i, j], Ss0[i, j]), (rxJh[i, k], sxJh[i, k], ryJh[i, k], syJh[i, k]))
end

function Sx0_with_n(dim::Dim1, i, j, k, solver)
    Sx0_ij = Sx0(dim, i, j, k, solver)
    return (Sx0_ij), abs(Sx0_ij[1])
end

function Sx0_with_n(dim::Dim2, i, j, k, solver)
    Sx0_ij, Sy0_ij = Sx0(dim, i, j, k, solver)
    return SVector(Sx0_ij, Sy0_ij), norm((Sx0_ij, Sy0_ij))
end

# TODO: hardcoded
function apply_LF_dissipation_to_BF(dim::Dim1, BF, i, k, lf, solver)
    BF[i, k] = SVector{1}(BF[i, k][1] - lf)
end

function apply_LF_dissipation_to_BF(dim::Dim2, BF, i, k, lf, solver)
    N1D = solver.param.N + 1
    if i <= 2 * N1D
        BF[i, k] = SVector(BF[i, k][1] - lf, BF[i, k][2])
    else
        BF[i, k] = SVector(BF[i, k][1], BF[i, k][2] - lf)
    end
end

function apply_LF_dissipation_to_fstar(dim::Dim1, fstar, i, k, Bxy_i, lf, solver)
    fstar[i, k] = SVector{1}(fstar[i, k][1] - lf / Bxy_i[1])
end

function apply_LF_dissipation_to_fstar(dim::Dim2, fstar, i, k, Bxy_i, lf, solver)
    N1D = solver.discrete_data.sizes.N1D
    if i <= 2 * N1D
        fstar[i, k] = SVector(fstar[i, k][1] - lf / Bxy_i[1], fstar[i, k][2])
    else
        fstar[i, k] = SVector(fstar[i, k][1], fstar[i, k][2] - lf / Bxy_i[2])
    end
end

# TODO: hardcoded
function graph_viscosity(dim::Dim1, i, j, k, Sxy0J_ij, state, solver)
    # TODO: better pass in lambda_arr
    (; λarr) = low_order_cache(state)
    (; Uq) = state.preallocation

    return SVector{1}(λarr[i, j, k] * (Uq[j, k] - Uq[i, k]))
end

function graph_viscosity(dim::Dim2, i, j, k, Sxy0J_ij, state, solver)
    # TODO: better pass in lambda_arr
    (; λarr) = low_order_cache(state)
    (; Uq) = state.preallocation

    Nc = num_components(solver)
    Sx0J_ij, _ = Sxy0J_ij
    visc_term = λarr[i, j, k] * (Uq[j, k] - Uq[i, k])
    # If it is the dissipation in x-direction
    # TODO: hardcoded tolerance
    if abs(Sx0J_ij) > 1e-10
        return SVector(visc_term, zero(SVector{Nc,Float64}))
    else
        return SVector(zero(SVector{Nc,Float64}), visc_term)
    end
end
