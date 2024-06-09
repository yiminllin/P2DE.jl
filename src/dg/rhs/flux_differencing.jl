#########################
### RHS of modal ESDG ###
#########################
function rhs_fluxdiff!(state, solver, state_param, time_param, need_proj=true)

    @timeit_debug time_param.timer "entropy projection" begin
        if (need_proj)
            entropy_projection!(entropyproj_limiter(solver), state, solver, time_param)
        end
    end

    @timeit_debug time_param.timer "calculate primitive variables" begin
        calculate_primitive_variables!(equation(solver), state, solver, state_param)
    end
    @timeit_debug time_param.timer "calculate interface dissipation coefficients" begin
        calculate_interface_dissipation_coeff!(state, solver, state_param)
    end
    @timeit_debug time_param.timer "enforce boundary conditions" begin
        enforce_BC!(state, solver, state_param)
    end

    # Flux differencing
    @timeit_debug time_param.timer "clear cache" begin
        clear_flux_differencing_cache!(state, solver)
    end
    @timeit_debug time_param.timer "flux differencing volume kernel" begin
        flux_differencing_volume!(state, solver)
    end
    @timeit_debug time_param.timer "flux differencing surface kernel" begin
        flux_differencing_surface!(state, solver)
    end

    # Assemble RHS
    @timeit_debug time_param.timer "assemble rhs" begin
        assemble_rhs!(state, solver, time_param)
    end
end

function calculate_primitive_variables!(equation::CompressibleIdealGas, state, solver, state_param)
    (; mapP) = state_param.bcdata
    (; u_tilde, psi_tilde) = state.preallocation
    (; beta, rholog, betalog, uP, betaP, rhologP, betalogP) = high_order_cache(state)
    (; K, Nq, Nh, Nfp) = solver.discrete_data.sizes

    @batch for k = 1:K
        for i = 1:Nh
            psi_tilde[i, k] = psi_ufun(equation, u_tilde[i, k])
            beta[i, k] = betafun(equation, u_tilde[i, k])
            rholog[i, k] = log(u_tilde[i, k][1])
            betalog[i, k] = log(beta[i, k])
        end
    end

    # Boundary contributions
    # TODO: refactor
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

function calculate_primitive_variables!(equation::KPP{Dim2}, state, solver, state_param)
    (; mapP) = state_param.bcdata
    (; u_tilde) = state.preallocation
    (; uP) = high_order_cache(state)
    (; K, Nfp, Nq, Nh) = solver.discrete_data.sizes

    # Boundary contributions
    # TODO: refactor
    uf = @view u_tilde[Nq+1:Nh, :]
    @batch for k = 1:K
        for i = 1:Nfp
            iP = mod1(mapP[i, k], Nfp)
            kP = div(mapP[i, k] - 1, Nfp) + 1
            uP[i, k] = uf[iP, kP]
        end
    end
end

function calculate_interface_dissipation_coeff!(state, solver, state_param)
    (; lam, LFc) = high_order_cache(state)
    (; u_tilde) = state.preallocation
    (; mapP) = state_param.bcdata
    (; K, Nfp, Nq, Nh) = solver.discrete_data.sizes

    # TODO: refactor
    uf = @view u_tilde[Nq+1:Nh, :]

    # Lax Friedrichs dissipation
    @batch for k = 1:K
        for i = 1:Nfp
            Bxy_i, n_i_norm = Bx_with_n(dim_type(solver), i, k, solver)
            n_i = @. Bxy_i / n_i_norm
            lam[i, k] = wavespeed_estimate(equation(solver), uf[i, k], n_i)
            LFc[i, k] = 0.5 * n_i_norm
        end
    end

    @batch for k = 1:K
        for i = 1:Nfp
            LFc[i, k] = LFc[i, k] * max(lam[i, k], lam[mapP[i, k]])
        end
    end
end

function enforce_BC!(state, solver, state_param)
    (; Uq) = state.preallocation
    (; LFc, uP, betaP, rhologP, betalogP) = high_order_cache(state)
    (; mapI, mapO, Ival) = state_param.bcdata
    (; fq2q) = solver.discrete_data.ops
    (; Nfp) = solver.discrete_data.sizes

    # zero dissipation on the boundary
    @batch for i in mapI
        LFc[i] = 0.0
    end
    @batch for i in mapO
        LFc[i] = 0.0
    end

    # Enforce inflow BC
    @batch for i = 1:size(mapI, 1)
        ii = mapI[i]
        uP[ii] = Ival[i]
        betaP[ii] = betafun(equation(solver), uP[ii])
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
        betaP[io] = betafun(equation(solver), uP[io])
        rhologP[io] = log(uP[io][1])
        betalogP[io] = log(betaP[io])
    end
end

function clear_flux_differencing_cache!(state, solver)
    (; QF1) = high_order_cache(state)
    (; Nh, K) = solver.discrete_data.sizes

    @batch for k = 1:K
        for i = 1:Nh
            QF1[i, k] = zero(QF1[i, k])
        end
    end
end

function flux_differencing_volume!(state, solver)
    (; QF1) = high_order_cache(state)
    (; Srsh_nnz) = solver.discrete_data.ops
    (; K) = solver.discrete_data.sizes

    @batch for k = 1:K
        for (i, j) in Srsh_nnz
            Ui = U(dim_type(solver), equation(solver), high_order_volume_flux_type(solver), i, k, state)
            Uj = U(dim_type(solver), equation(solver), high_order_volume_flux_type(solver), j, k, state)
            accumulate_QF1!(QF1, i, Ui, j, Uj, k, solver)
        end
    end
end

# Get (utilde, beta, log(rho), log(beta)) at index i,element k
function U(dim::Dim1, equation::CompressibleIdealGas, vol_flux_type::ChandrashekarFlux, idx, k, state)
    (; u_tilde) = state.preallocation
    (; beta, rholog, betalog) = high_order_cache(state)

    return SVector(u_tilde[idx, k][1], u_tilde[idx, k][2] / u_tilde[idx, k][1], beta[idx, k], rholog[idx, k], betalog[idx, k])
end

function U(dim::Dim2, equation::CompressibleIdealGas, vol_flux_type::ChandrashekarFlux, idx, k, state)
    (; u_tilde) = state.preallocation
    (; beta, rholog, betalog) = high_order_cache(state)

    return SVector(u_tilde[idx, k][1], u_tilde[idx, k][2] / u_tilde[idx, k][1], u_tilde[idx, k][3] / u_tilde[idx, k][1], beta[idx, k], rholog[idx, k], betalog[idx, k])
end

function U(dim, equation::CompressibleIdealGas, vol_flux_type::CentralFlux, idx, k, state)
    return state.preallocation.u_tilde[idx, k]
end

function U(dim, equation::KPP, vol_flux_type, idx, k, state)
    return state.preallocation.u_tilde[idx, k]
end

function accumulate_QF1!(QF1, i, Ui, j, Uj, k, solver)
    ϵ = solver.param.global_constants.ZEROTOL
    # TODO: assume Sxyh_db_ij = -Sxyh_db_ji
    #              Sxyh_db_ji = Sx(j,i,k,discrete_data,dim)
    Sxyh_db_ij = Sx(dim_type(solver), i, j, k, solver)
    fxy = eval_high_order_volume_flux(high_order_volume_flux_type(solver), equation(solver), Ui, Uj)
    Sfxy_ij = @. Sxyh_db_ij * fxy
    Sfxy_ji = -Sfxy_ij
    QF1[i, k] += Sfxy_ij
    QF1[j, k] += Sfxy_ji
end

function eval_high_order_volume_flux(vol_flux_type::ChandrashekarFlux, equation, Ui, Uj)
    return fS(equation, Ui, Uj)
end

function eval_high_order_volume_flux(vol_flux_type::CentralFlux, equation, Ui, Uj)
    fi = fluxes(equation, Ui)
    fj = fluxes(equation, Uj)
    return @. 0.5 * (fi + fj)
end

function flux_differencing_surface!(state, solver)
    (; K) = solver.discrete_data.sizes

    @batch for k = 1:K
        accumulate_numerical_flux!(state, k, solver)
    end
end

function accumulate_numerical_flux!(state, k, solver)
    (; BF_H, fstar_H, u_tilde) = state.preallocation
    (; uP, LFc) = high_order_cache(state)
    (; Nq, Nh, Nfp) = solver.discrete_data.sizes

    # Boundary contributions (B F)1
    uf = @view u_tilde[Nq+1:Nh, :]
    for i = 1:Nfp
        fstar_H[i, k] = eval_high_order_surface_flux(high_order_surface_flux_type(solver), i, k, state, solver)
        Bxy_i = Bx(dim_type(solver), i, k, solver)
        BF_H[i, k] = @. Bxy_i * fstar_H[i, k]
        # Apply LF dissipation
        lf = LFc[i, k] * (uP[i, k] - uf[i, k])
        apply_LF_dissipation_to_BF(dim_type(solver), BF_H, i, k, lf, solver)
        apply_LF_dissipation_to_fstar(dim_type(solver), fstar_H, i, k, Bxy_i, lf, solver)
    end
end

function eval_high_order_surface_flux(surface_flux_type::ChandrashekarOnProjectedVal, i, k, state, solver)
    (; u_tilde) = state.preallocation
    (; beta, rholog, betalog, uP, betaP, rhologP, betalogP) = high_order_cache(state)
    (; Nq, Nh, Nd) = solver.discrete_data.sizes

    uf = @view u_tilde[Nq+1:Nh, :]
    betaf = @view beta[Nq+1:Nh, :]
    rhologf = @view rholog[Nq+1:Nh, :]
    betalogf = @view betalog[Nq+1:Nh, :]
    return fS(equation(solver), (uf[i, k][1], (uf[i, k][c] / uf[i, k][1] for c in 2:2+Nd-1)..., betaf[i, k], rhologf[i, k], betalogf[i, k]),
        (uP[i, k][1], (uP[i, k][c] / uP[i, k][1] for c in 2:2+Nd-1)..., betaP[i, k], rhologP[i, k], betalogP[i, k]))
end

function eval_high_order_surface_flux(surface_flux_type::LaxFriedrichsOnProjectedVal, i, k, state, solver)
    (; u_tilde) = state.preallocation
    (; uP) = high_order_cache(state)
    (; Nq, Nh) = solver.discrete_data.sizes

    uf = @view u_tilde[Nq+1:Nh, :]
    fxyf = fluxes(equation(solver), uf[i, k])
    fxyP = fluxes(equation(solver), uP[i, k])

    return @. 0.5 * (fxyf + fxyP)
end

function project_flux_difference_to_quad_unlimited!(state, k, solver)
    (; BF_H) = state.preallocation
    (; MinvVhTQF1, MinvVfTBF1, QF1) = high_order_cache(state)
    (; MinvVhT, MinvVfT) = solver.discrete_data.ops

    @views mul!(MinvVhTQF1[:, k], MinvVhT, QF1[:, k])
    @views mul!(MinvVfTBF1[:, k], MinvVfT, BF_H[:, k])
end

function project_flux_difference_to_quad!(entropyproj_limiter_type::NoEntropyProjectionLimiter, state, k, nstage, tid, solver)
    project_flux_difference_to_quad_unlimited!(state, k, solver)
end

# TODO: hardcoded, only for gauss
function project_flux_difference_to_quad!(entropyproj_limiter_type::ScaledExtrapolation, state, k, nstage, tid, solver)
    (; BF_H) = state.preallocation
    (; MinvVhTQF1, MinvVfTBF1, QF1, VhT_new, MinvVhT_new, Vf_new) = high_order_cache(state)
    (; Nq, Nh) = solver.discrete_data.sizes
    (; wq) = solver.discrete_data.ops

    # TODO: only consider Gauss and LGL for now, so Vq = I
    #       and we assume VqT_new = I
    #       VqT_new = @views VhT_new[:,1:Nq]
    #       VqT_new .= transpose(discrete_data.ops.Vq)
    update_limited_extrapolation!(entropyproj_limiter(solver), state, k, nstage, tid, solver)
    Vf_new = @views Vf_new[:, :, tid]
    VhT_new = @views VhT_new[:, :, tid]
    MinvVhT_new = @views MinvVhT_new[:, :, tid]
    VfT_new = @views VhT_new[:, Nq+1:Nh]
    @. VfT_new = transpose(Vf_new)
    @. MinvVhT_new = (1 / wq) * VhT_new
    MinvVfT_new = @views MinvVhT_new[:, Nq+1:Nh]
    @views mul!(MinvVhTQF1[:, k], MinvVhT_new, QF1[:, k])
    @views mul!(MinvVfTBF1[:, k], MinvVfT_new, BF_H[:, k])
end

function update_limited_extrapolation!(entropyproj_limiter_type::NodewiseScaledExtrapolation, state, k, nstage, tid, solver)
    (; Vf_new) = high_order_cache(state)
    (; Vf, Vf_low) = solver.discrete_data.ops
    (; Nfp) = solver.discrete_data.sizes

    for i = 1:Nfp
        l_k_i = state.preallocation.θ_local_arr[i, k, nstage]
        @views @. Vf_new[i, :, tid] = l_k_i * Vf[i, :] + (1 - l_k_i) * Vf_low[i, :]
    end
end

# Return whether on element k, the extrapolation is limited
function is_Vf_limited(entropyproj_limiter_type::NoEntropyProjectionLimiter, state, k, nstage)
    return false
end

function is_Vf_limited(entropyproj_limiter_type::NodewiseScaledExtrapolation, state, k, nstage)
    return minimum(view(state.preallocation.θ_local_arr, :, k, nstage)) < 1.0
end

# TODO: dispatch
function assemble_rhs!(state, solver, time_param)
    (; MinvVhTQF1, MinvVfTBF1) = high_order_cache(state)
    (; rhsH, rhsxyH) = state.preallocation
    (; Jq) = solver.discrete_data.geom
    (; K, Nq) = solver.discrete_data.sizes
    (; nstage) = time_param

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
        if !is_Vf_limited(entropyproj_limiter(solver), state, k, nstage)
            project_flux_difference_to_quad_unlimited!(state, k, solver)
        else
            project_flux_difference_to_quad!(entropyproj_limiter(solver), state, k, nstage, tid, solver)
        end
        # TODO: assume collocation scheme, so Nq = Np
        for i = 1:Nq
            rhsxyH[i, k] = -(MinvVhTQF1[i, k] + MinvVfTBF1[i, k]) / Jq[i, k]
            rhsH[i, k] = sum(rhsxyH[i, k])
        end
        # TODO: Assume Vq = I for both LGL and Gauss
        # @views mul!(rhsH[:,k],Vq,sum.(rhsxyH[:,k]))
    end
end

