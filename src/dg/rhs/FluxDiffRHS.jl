#########################
### RHS of modal ESDG ###
#########################
function rhs_fluxdiff!(prealloc, rhs_cache, param, discrete_data, bcdata, nstage, timer, need_proj=true)
    (; entropyproj_limiter_type, equation) = param

    cache = high_order_cache(rhs_cache)
    @timeit_debug timer "entropy projection" begin
        if (need_proj)
            entropy_projection!(prealloc, param, entropyproj_limiter_type, discrete_data, nstage, timer)
        end
    end

    @timeit_debug timer "calculate primitive variables" begin
        calculate_primitive_variables!(cache, prealloc, param, equation, bcdata)
    end
    @timeit_debug timer "calculate interface dissipation coefficients" begin
        calculate_interface_dissipation_coeff!(cache, prealloc, param, bcdata, discrete_data)
    end
    @timeit_debug timer "enforce boundary conditions" begin
        enforce_BC!(cache, prealloc, param, bcdata, discrete_data)
    end

    # Flux differencing
    @timeit_debug timer "clear cache" begin
        clear_flux_differencing_cache!(cache)
    end
    @timeit_debug timer "flux differencing volume kernel" begin
        flux_differencing_volume!(cache, prealloc, param, equation, discrete_data)
    end
    @timeit_debug timer "flux differencing surface kernel" begin
        flux_differencing_surface!(cache, prealloc, param, discrete_data)
    end

    # Assemble RHS
    @timeit_debug timer "assemble rhs" begin
        assemble_rhs!(cache, prealloc, param, discrete_data, nstage)
    end

    # check_flux_diff_entropy_stability(cache,prealloc,param,discrete_data,dim_type(equation))
end

function calculate_primitive_variables!(cache, prealloc, param, equation::CompressibleIdealGas, bcdata)
    (; equation) = param
    (; mapP) = bcdata
    (; u_tilde) = prealloc
    (; psi_tilde) = prealloc
    (; beta, rholog, betalog, uP, betaP, rhologP, betalogP) = cache
    K = num_elements(param)
    Nfp = size(mapP, 1)

    @batch for k = 1:K
        for i = 1:size(beta, 1)
            psi_tilde[i, k] = psi_ufun(equation, u_tilde[i, k])
            beta[i, k] = betafun(equation, u_tilde[i, k])
            rholog[i, k] = log(u_tilde[i, k][1])
            betalog[i, k] = log(beta[i, k])
        end
    end

    # Boundary contributions
    # TODO: refactor
    Nq = size(prealloc.Uq, 1)
    Nh = size(u_tilde, 1)
    uf = @view u_tilde[Nq+1:Nh, :]
    betaf = @view beta[Nq+1:Nh, :]
    rhologf = @view rholog[Nq+1:Nh, :]
    betalogf = @view betalog[Nq+1:Nh, :]
    @batch for k = 1:K
        for i = 1:Nfp
            iP = mod1(mapP[i, k], Nfp)
            kP = div(mapP[i, k] - 1, Nfp) + 1
            uP[i, k] = uf[iP, kP]
            betaP[i, k] = betaf[iP, kP]
            rhologP[i, k] = rhologf[iP, kP]
            betalogP[i, k] = betalogf[iP, kP]
        end
    end
end

function calculate_primitive_variables!(cache, prealloc, param, equation::KPP{Dim2}, bcdata)
    (; equation) = param
    (; mapP) = bcdata
    (; u_tilde) = prealloc
    (; uP) = cache
    K = num_elements(param)
    Nfp = size(mapP, 1)

    # Boundary contributions
    # TODO: refactor
    Nq = size(prealloc.Uq, 1)
    Nh = size(u_tilde, 1)
    uf = @view u_tilde[Nq+1:Nh, :]
    @batch for k = 1:K
        for i = 1:Nfp
            iP = mod1(mapP[i, k], Nfp)
            kP = div(mapP[i, k] - 1, Nfp) + 1
            uP[i, k] = uf[iP, kP]
        end
    end
end

function calculate_interface_dissipation_coeff!(cache, prealloc, param, bcdata, discrete_data)
    (; lam, LFc) = cache
    (; u_tilde) = prealloc
    (; equation) = param
    (; mapP) = bcdata

    # TODO: refactor
    K = num_elements(param)
    Nq = size(prealloc.Uq, 1)
    Nh = size(prealloc.u_tilde, 1)
    Nfp = Nh - Nq
    uf = @view u_tilde[Nq+1:Nh, :]
    dim = dim_type(equation)

    # Lax Friedrichs dissipation
    @batch for k = 1:K
        for i = 1:Nfp
            Bxy_i, n_i_norm = Bx_with_n(i, k, discrete_data, dim)
            n_i = @. Bxy_i / n_i_norm
            lam[i, k] = wavespeed_estimate(equation, uf[i, k], n_i)
            LFc[i, k] = 0.5 * n_i_norm
        end
    end

    @batch for k = 1:K
        for i = 1:Nfp
            LFc[i, k] = LFc[i, k] * max(lam[i, k], lam[mapP[i, k]])
        end
    end
end

function enforce_BC!(cache, prealloc, param, bcdata, discrete_data)
    (; Uq, u_tilde) = prealloc
    (; equation) = param
    (; mapP, mapI, mapO, Ival) = bcdata
    (; LFc, uP, betaP, rhologP, betalogP) = cache
    (; fq2q) = discrete_data.ops

    Nfp = size(mapP, 1)
    # zero dissipation on the boundary
    @batch for i in mapI
        LFc[i] = 0.0
    end
    @batch for i in mapO
        LFc[i] = 0.0
    end

    # Enforce inflow BC
    Nq = size(Uq, 1)
    Nh = size(u_tilde, 1)
    uf = @view u_tilde[Nq+1:Nh, :]
    @batch for i = 1:size(mapI, 1)
        ii = mapI[i]
        uP[ii] = Ival[i]
        betaP[ii] = betafun(equation, uP[ii])
        rhologP[ii] = log(uP[ii][1])
        betalogP[ii] = log(betaP[ii])
    end

    # Enforce outflow BC
    @batch for i = 1:size(mapO, 1)
        io = mapO[i]
        iP = mod1(io, Nfp)
        kP = div(io - 1, Nfp) + 1
        iq = fq2q[iP]
        uP[io] = Uq[iq, kP]
        betaP[io] = betafun(equation, uP[io])
        rhologP[io] = log(uP[io][1])
        betalogP[io] = log(betaP[io])
    end
end

function clear_flux_differencing_cache!(cache)
    (; QF1) = cache

    @batch for k = 1:size(QF1, 2)
        for i = 1:size(QF1, 1)
            QF1[i, k] = zero(QF1[i, k])
        end
    end
end

function flux_differencing_volume!(cache, prealloc, param, equation, discrete_data)
    (; equation) = param
    (; QF1) = cache
    (; Srsh_nnz) = discrete_data.ops

    dim = dim_type(equation)
    vol_flux_type = high_order_volume_flux_type(param.rhs_type)
    K = num_elements(param)
    Nq = discrete_data.sizes.Nq
    Nh = size(QF1, 1)
    @batch for k = 1:K
        for (i, j) in Srsh_nnz
            Ui = U(vol_flux_type, i, k, cache, prealloc, param.equation, dim)
            Uj = U(vol_flux_type, j, k, cache, prealloc, param.equation, dim)
            accumulate_QF1!(QF1, i, Ui, j, Uj, k, param, discrete_data, equation)
        end
    end
end

# Get (utilde, beta, log(rho), log(beta)) at index i,element k
function U(vol_flux_type::ChandrashekarFlux, idx, k, cache, prealloc, equation::CompressibleIdealGas, dim::Dim1)
    (; u_tilde) = prealloc
    (; beta, rholog, betalog) = cache

    return SVector(u_tilde[idx, k][1], u_tilde[idx, k][2] / u_tilde[idx, k][1], beta[idx, k], rholog[idx, k], betalog[idx, k])
end

function U(vol_flux_type::ChandrashekarFlux, idx, k, cache, prealloc, equation::CompressibleIdealGas, dim::Dim2)
    (; u_tilde) = prealloc
    (; beta, rholog, betalog) = cache

    return SVector(u_tilde[idx, k][1], u_tilde[idx, k][2] / u_tilde[idx, k][1], u_tilde[idx, k][3] / u_tilde[idx, k][1], beta[idx, k], rholog[idx, k], betalog[idx, k])
end

function U(vol_flux_type::CentralFlux, idx, k, cache, prealloc, equation::CompressibleIdealGas, dim)
    return prealloc.u_tilde[idx, k]
end

function U(vol_flux_type, idx, k, cache, prealloc, equation::KPP, dim)
    return prealloc.u_tilde[idx, k]
end

function accumulate_QF1!(QF1, i, Ui, j, Uj, k, param, discrete_data, equation)
    ϵ = param.global_constants.ZEROTOL
    dim = dim_type(equation)
    # TODO: assume Sxyh_db_ij = -Sxyh_db_ji
    #              Sxyh_db_ji = Sx(j,i,k,discrete_data,dim)
    Sxyh_db_ij = Sx(i, j, k, discrete_data, dim)
    fxy = high_order_volume_flux_type(high_order_volume_flux_type(param.rhs_type), equation, Ui, Uj)
    Sfxy_ij = @. Sxyh_db_ij * fxy
    Sfxy_ji = -Sfxy_ij
    QF1[i, k] += Sfxy_ij
    QF1[j, k] += Sfxy_ji
end

function high_order_volume_flux_type(vol_flux_type::ChandrashekarFlux, equation, Ui, Uj)
    return fS(equation, Ui, Uj)
end

function high_order_volume_flux_type(vol_flux_type::CentralFlux, equation, Ui, Uj)
    fi = fluxes(equation, Ui)
    fj = fluxes(equation, Uj)
    return @. 0.5 * (fi + fj)
end

function flux_differencing_surface!(cache, prealloc, param, discrete_data)
    (; equation) = param

    K = num_elements(param)
    @batch for k = 1:K
        accumulate_numerical_flux!(prealloc, cache, k, param, discrete_data, equation)
    end
end

function accumulate_numerical_flux!(prealloc, cache, k, param, discrete_data, equation)
    (; BF_H, fstar_H, u_tilde) = prealloc
    (; uP, LFc) = cache

    # Boundary contributions (B F)1
    Nq = size(prealloc.Uq, 1)
    Nh = size(u_tilde, 1)
    Nfp = size(BF_H, 1)
    dim = dim_type(equation)
    uf = @view u_tilde[Nq+1:Nh, :]
    for i = 1:Nfp
        fstar_H[i, k] = evaluate_high_order_surface_flux(prealloc, cache, param, i, k, high_order_surface_flux_type(param.rhs_type))
        Bxy_i = Bx(i, k, discrete_data, dim)
        BF_H[i, k] = @. Bxy_i * fstar_H[i, k]
        # Apply LF dissipation
        lf = LFc[i, k] * (uP[i, k] - uf[i, k])
        apply_LF_dissipation_to_BF(BF_H, param, i, k, lf, dim)
        apply_LF_dissipation_to_fstar(fstar_H, param, i, k, Bxy_i, lf, dim)
    end
end

function evaluate_high_order_surface_flux(prealloc, cache, param, i, k, surface_flux_type::ChandrashekarOnProjectedVal)
    (; equation) = param
    (; u_tilde) = prealloc
    (; beta, rholog, betalog, uP, betaP, rhologP, betalogP) = cache

    Nq = size(prealloc.Uq, 1)
    Nh = size(u_tilde, 1)
    Nd = dim(equation)
    uf = @view u_tilde[Nq+1:Nh, :]
    betaf = @view beta[Nq+1:Nh, :]
    rhologf = @view rholog[Nq+1:Nh, :]
    betalogf = @view betalog[Nq+1:Nh, :]
    return fS(equation, (uf[i, k][1], (uf[i, k][c] / uf[i, k][1] for c in 2:2+Nd-1)..., betaf[i, k], rhologf[i, k], betalogf[i, k]),
        (uP[i, k][1], (uP[i, k][c] / uP[i, k][1] for c in 2:2+Nd-1)..., betaP[i, k], rhologP[i, k], betalogP[i, k]))
end

function evaluate_high_order_surface_flux(prealloc, cache, param, i, k, surface_flux_type::LaxFriedrichsOnProjectedVal)
    (; equation) = param
    (; u_tilde) = prealloc
    (; uP) = cache

    Nq = size(prealloc.Uq, 1)
    Nh = size(u_tilde, 1)
    uf = @view u_tilde[Nq+1:Nh, :]
    fxyf = fluxes(equation, uf[i, k])
    fxyP = fluxes(equation, uP[i, k])

    return @. 0.5 * (fxyf + fxyP)
end

function project_flux_difference_to_quad_unlimited!(k, cache, prealloc, discrete_data, tid)
    (; BF_H) = prealloc
    (; MinvVhTQF1, MinvVfTBF1, QF1) = cache
    (; MinvVhT, MinvVfT) = discrete_data.ops

    @views mul!(MinvVhTQF1[:, k], MinvVhT, QF1[:, k])
    @views mul!(MinvVfTBF1[:, k], MinvVfT, BF_H[:, k])
end

function project_flux_difference_to_quad!(cache, prealloc, param, entropyproj_limiter_type::NoEntropyProjectionLimiter, discrete_data, k, nstage, tid)
    project_flux_difference_to_quad_unlimited!(k, cache, prealloc, discrete_data, tid)
end

# TODO: hardcoded, only for gauss
function project_flux_difference_to_quad!(cache, prealloc, param, entropyproj_limiter_type::ScaledExtrapolation, discrete_data, k, nstage, tid)
    (; BF_H) = prealloc
    (; MinvVhTQF1, MinvVfTBF1, QF1, VhT_new, MinvVhT_new, Vf_new) = cache
    (; Nq, Nh) = discrete_data.sizes

    # TODO: only consider Gauss and LGL for now, so Vq = I
    #       and we assume VqT_new = I
    #       VqT_new = @views VhT_new[:,1:Nq]
    #       VqT_new .= transpose(discrete_data.ops.Vq)
    update_limited_extrapolation!(cache, prealloc, param, param.entropyproj_limiter_type, discrete_data, k, nstage, tid)
    Vf_new = @views Vf_new[:, :, tid]
    VhT_new = @views VhT_new[:, :, tid]
    MinvVhT_new = @views MinvVhT_new[:, :, tid]
    VfT_new = @views VhT_new[:, Nq+1:Nh]
    @. VfT_new = transpose(Vf_new)
    @. MinvVhT_new = (1 / discrete_data.ops.wq) * VhT_new
    MinvVfT_new = @views MinvVhT_new[:, Nq+1:Nh]
    @views mul!(MinvVhTQF1[:, k], MinvVhT_new, QF1[:, k])
    @views mul!(MinvVfTBF1[:, k], MinvVfT_new, BF_H[:, k])
end

function update_limited_extrapolation!(cache, prealloc, param, entropyproj_limiter_type::NodewiseScaledExtrapolation, discrete_data, k, nstage, tid)
    (; Vf_new) = cache
    (; Vf, Vf_low) = discrete_data.ops

    for i = 1:discrete_data.sizes.Nfp
        l_k_i = prealloc.θ_local_arr[i, k, nstage]
        @views @. Vf_new[i, :, tid] = l_k_i * Vf[i, :] + (1 - l_k_i) * Vf_low[i, :]
    end
end

# Return whether on element k, the extrapolation is limited
function is_Vf_limited(prealloc, k, nstage, entropyproj_limiter_type::NoEntropyProjectionLimiter)
    return false
end

function is_Vf_limited(prealloc, k, nstage, entropyproj_limiter_type::NodewiseScaledExtrapolation)
    return minimum(view(prealloc.θ_local_arr, :, k, nstage)) < 1.0
end

# TODO: dispatch
function assemble_rhs!(cache, prealloc, param, discrete_data, nstage)
    (; entropyproj_limiter_type) = param
    (; QF1, MinvVhTQF1, MinvVfTBF1) = cache
    (; BF_H, rhsH, rhsxyH) = prealloc
    (; Jq) = discrete_data.geom
    (; MinvVhT, MinvVfT, Vq) = discrete_data.ops

    K = num_elements(param)
    # Assemble RHS
    @batch for k = 1:K
        tid = Threads.threadid()
        # If 1. LGL
        #    2. Gauss with no entropy proj limiter, adaptive filter
        #                  scaled extrapolation with l_k = 1 (elementwise)
        #                                            l_k_i = 1 for all i (nodewise)
        # Apply precomputed matrix 
        # otherwise, if on Gauss with scaled extrapolation and nonzero limiting param
        #            apply limited Vf
        if !is_Vf_limited(prealloc, k, nstage, entropyproj_limiter_type)
            project_flux_difference_to_quad_unlimited!(k, cache, prealloc, discrete_data, tid)
        else
            project_flux_difference_to_quad!(cache, prealloc, param, entropyproj_limiter_type, discrete_data, k, nstage, tid)
        end
        # TODO: assume collocation scheme, so Nq = Np
        for i = 1:size(rhsH, 1)
            rhsxyH[i, k] = -(MinvVhTQF1[i, k] + MinvVfTBF1[i, k]) / Jq[i, k]
            rhsH[i, k] = sum(rhsxyH[i, k])
        end
        # TODO: Assume Vq = I for both LGL and Gauss
        # @views mul!(rhsH[:,k],Vq,sum.(rhsxyH[:,k]))
    end
end

function check_flux_diff_entropy_stability(cache, prealloc, param, discrete_data, dim::Dim2)
    (; equation) = param
    (; v_tilde, psi_tilde, Uq, vq) = prealloc
    (; fstar_H, rhsxyH) = prealloc
    (; QF1, MinvVhTQF1, MinvVfTBF1) = cache
    (; wq) = discrete_data.ops
    (; Jq) = discrete_data.geom
    (; Nq, Nh, Nfp) = discrete_data.sizes

    # Check entropy stability
    K = num_elements(param)
    Nd = dim(equation)
    dim = dim_type(equation)
    @batch for k = 1:K
        entropy_estimate_vol = zero(SVector{Nd,Float64})   # vT rhs_vol
        entropy_estimate_surf = zero(SVector{Nd,Float64})   # vT rhs_surf
        entropy_estimate = zero(SVector{Nd,Float64})   # vT rhs
        for i = 1:Nq
            m_i = wq[i]
            entropy_estimate_vol += -m_i * SVector(sum(vq[i, k] .* MinvVhTQF1[i, k][1]), sum(vq[i, k] .* MinvVhTQF1[i, k][2]))
            entropy_estimate_surf += -m_i * SVector(sum(vq[i, k] .* MinvVfTBF1[i, k][1]), sum(vq[i, k] .* MinvVfTBF1[i, k][2]))
            entropy_estimate += m_i * Jq[i, k] * SVector(sum(vq[i, k] .* rhsxyH[i, k][1]), sum(vq[i, k] .* rhsxyH[i, k][2]))
        end
        sum_Bpsi = zero(SVector{Nd,Float64})   # 1T B psi
        vTBfstar = zero(SVector{Nd,Float64})   # vT B f*
        for i = 1:Nfp
            Bxy_i = Bx(i, k, discrete_data, dim)
            sum_Bpsi += Bxy_i .* psi_tilde[Nq+i, k]
            vTBfstar += Bxy_i .* SVector(sum(v_tilde[i+Nq, k] .* fstar_H[i, k][1]), sum(v_tilde[i+Nq, k] .* fstar_H[i, k][2]))
        end
        diff_vol = entropy_estimate_vol - sum_Bpsi
        diff_surf = entropy_estimate_surf + vTBfstar
        diff = entropy_estimate - sum_Bpsi + vTBfstar
        tol = 1e-12
        if max(abs(diff_vol[1]), abs(diff_vol[2]), abs(diff_surf[1]), abs(diff_surf[2]), abs(diff[1]), abs(diff[2])) > tol
            @show k, diff_vol, diff_surf, diff
        end
    end
end
