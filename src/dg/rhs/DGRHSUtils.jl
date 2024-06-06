function reference_to_physical(U, GJ, dim::Dim1)
    Ur = U[1]
    rxJ = GJ[1]
    return SVector(rxJ * Ur,)
end

function reference_to_physical(U, GJ, dim::Dim2)
    Ur, Us = U
    rxJ, sxJ, ryJ, syJ = GJ
    return SVector(rxJ * Ur + sxJ * Us, ryJ * Ur + syJ * Us)
end

function Bx(i, k, discrete_data, dim::Dim1)
    (; Br) = discrete_data.ops
    (; rxJh) = discrete_data.geom

    iface = i + discrete_data.sizes.Nq
    return reference_to_physical((Br[i, i],), (rxJh[iface, k],), dim)
end

function Bx(i, k, discrete_data, dim::Dim2)
    (; Br, Bs) = discrete_data.ops
    (; rxJh, ryJh, sxJh, syJh) = discrete_data.geom

    iface = i + discrete_data.sizes.Nq
    return reference_to_physical((Br[i, i], Bs[i, i]), (rxJh[iface, k], sxJh[iface, k], ryJh[iface, k], syJh[iface, k]), dim)
end

function Bx_with_n(i, k, discrete_data, dim::Dim1)
    Bx_i = Bx(i, k, discrete_data, dim)
    return Bx_i, abs(Bx_i[1])
end

function Bx_with_n(i, k, discrete_data, dim::Dim2)
    Bx_i, By_i = Bx(i, k, discrete_data, dim)
    return SVector(Bx_i, By_i), norm((Bx_i, By_i))
end

function Sx(i, j, k, discrete_data, dim::Dim1)
    (; Srh_db) = discrete_data.ops
    (; rxJh) = discrete_data.geom

    return reference_to_physical((Srh_db[i, j],), (rxJh[i, k],), dim)
end

function Sx(i, j, k, discrete_data, dim::Dim2)
    (; Srh_db, Ssh_db) = discrete_data.ops
    (; rxJh, ryJh, sxJh, syJh) = discrete_data.geom

    return reference_to_physical((Srh_db[i, j], Ssh_db[i, j]), (rxJh[i, k], sxJh[i, k], ryJh[i, k], syJh[i, k]), dim)
end

function Sx0(i, j, k, discrete_data, dim::Dim1)
    (; Sr0) = discrete_data.ops
    (; rxJh) = discrete_data.geom

    return reference_to_physical((Sr0[i, j],), (rxJh[i, k],), dim)
end

function Sx0(i, j, k, discrete_data, dim::Dim2)
    (; Sr0, Ss0) = discrete_data.ops
    (; rxJh, ryJh, sxJh, syJh) = discrete_data.geom

    return reference_to_physical((Sr0[i, j], Ss0[i, j]), (rxJh[i, k], sxJh[i, k], ryJh[i, k], syJh[i, k]), dim)
end

function Sx0_with_n(i, j, k, discrete_data, dim::Dim1)
    Sx0_ij = Sx0(i, j, k, discrete_data, dim)
    return (Sx0_ij), abs(Sx0_ij[1])
end

function Sx0_with_n(i, j, k, discrete_data, dim::Dim2)
    Sx0_ij, Sy0_ij = Sx0(i, j, k, discrete_data, dim)
    return SVector(Sx0_ij, Sy0_ij), norm((Sx0_ij, Sy0_ij))
end

# TODO: hardcoded
function apply_LF_dissipation_to_BF(BF, param, i, k, lf, dim::Dim1)
    BF[i, k] = SVector{1}(BF[i, k][1] - lf)
end

function apply_LF_dissipation_to_BF(BF, param, i, k, lf, dim::Dim2)
    N1D = param.N + 1
    if i <= 2 * N1D
        BF[i, k] = SVector(BF[i, k][1] - lf, BF[i, k][2])
    else
        BF[i, k] = SVector(BF[i, k][1], BF[i, k][2] - lf)
    end
end

function apply_LF_dissipation_to_fstar(fstar, param, i, k, Bxy_i, lf, dim::Dim1)
    fstar[i, k] = SVector{1}(fstar[i, k][1] - lf / Bxy_i[1])
end

function apply_LF_dissipation_to_fstar(fstar, param, i, k, Bxy_i, lf, dim::Dim2)
    N1D = param.N + 1
    if i <= 2 * N1D
        fstar[i, k] = SVector(fstar[i, k][1] - lf / Bxy_i[1], fstar[i, k][2])
    else
        fstar[i, k] = SVector(fstar[i, k][1], fstar[i, k][2] - lf / Bxy_i[2])
    end
end

# TODO: hardcoded
function graph_viscosity(cache, prealloc, param, i, j, k, Sxy0J_ij, dim::Dim1)
    (; λarr) = cache
    (; Uq) = prealloc

    return SVector{1}(λarr[i, j, k] * (Uq[j, k] - Uq[i, k]))
end

function graph_viscosity(cache, prealloc, param, i, j, k, Sxy0J_ij, dim::Dim2)
    (; λarr) = cache
    (; Uq) = prealloc

    Nc = num_components(param.equation)
    Sx0J_ij, Sy0J_ij = Sxy0J_ij
    visc_term = λarr[i, j, k] * (Uq[j, k] - Uq[i, k])
    # If it is the dissipation in x-direction
    # TODO: hardcoded tolerance
    if abs(Sx0J_ij) > 1e-10
        return SVector(visc_term, zero(SVector{Nc,Float64}))
    else
        return SVector(zero(SVector{Nc,Float64}), visc_term)
    end
end
