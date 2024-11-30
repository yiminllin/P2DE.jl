##########################################
### RHS of positivity preserving Gauss ###
##########################################
function rhs_low_graph_visc!(state, solver, state_param, time_param, need_proj=true)
    @timeit_debug time_param.timer "entropy projection" begin
        if (need_proj)
            entropy_projection!(entropyproj_limiter(solver), state, solver, time_param)
        end
    end
    @timeit_debug time_param.timer "calculate wavespeed and inviscid flux" begin
        calculate_wavespeed_and_inviscid_flux!(state, solver)
    end
    @timeit_debug time_param.timer "enforce boundary condition" begin
        get_uP_and_enforce_BC!(state, solver, state_param)
    end

    # Assemble RHS
    @timeit_debug time_param.timer "clear cache" begin
        clear_low_order_rhs!(state, solver)
    end
    @timeit_debug time_param.timer "Low order volume kernel" begin
        accumulate_low_order_rhs_volume!(state, solver)
    end
    @timeit_debug time_param.timer "Low order surface kernel" begin
        accumulate_low_order_rhs_surface!(state, solver, state_param)
    end
    @timeit_debug time_param.timer "Scale low order solution by mass" begin
        scale_low_order_rhs_by_mass!(state, solver)
    end


    dt = time_param.dt
    @timeit_debug time_param.timer "Calculate low order positivity CFL" begin
        # Determine positivity CFL
        if (time_param.nstage == 1)
            dt = calculate_lambda_and_low_order_CFL!(state, solver, time_param)
        end
    end

    return dt
end

function calculate_wavespeed_and_inviscid_flux!(state, solver)
    (; K) = solver.discrete_data.sizes
    @batch for k = 1:K
        update_face_values!(low_order_surface_flux(solver), state, k, solver)
        update_wavespeed_and_inviscid_flux!(state, k, solver)
    end
end

function update_face_values!(surface_flux::LaxFriedrichsOnNodalVal, state, k, solver)
    (; Uf) = low_order_cache(state)
    (; Uq) = state.preallocation
    (; fq2q) = solver.discrete_data.ops
    (; Nfp) = solver.discrete_data.sizes

    for i = 1:Nfp
        iq = fq2q[i]
        Uf[i, k] = Uq[iq, k]
    end
end

function update_face_values!(surface_flux::LaxFriedrichsOnProjectedVal, state, k, solver)
    (; Uf) = low_order_cache(state)
    (; u_tilde) = state.preallocation
    (; Nq, Nfp) = solver.discrete_data.sizes

    for i = 1:Nfp
        Uf[i, k] = u_tilde[i+Nq, k]
    end
end

function update_wavespeed_and_inviscid_flux!(state, k, solver)
    (; Uq) = state.preallocation
    (; Uf, wavespeed_f, flux) = low_order_cache(state)
    (; Nq, Nfp) = solver.discrete_data.sizes

    # Volume inviscid flux
    for i = 1:Nq
        u_i = Uq[i, k]
        flux[i, k] = fluxes(equation(solver), u_i)
    end

    # Surface wavespeed and inviscid flux
    for i = 1:Nfp
        u_i = Uf[i, k]
        Bxy_i, n_i_norm = Bx_with_n(dim(solver), i, k, solver)
        n_i = @. Bxy_i / n_i_norm
        wavespeed_f[i, k] = wavespeed_estimate(equation(solver), u_i, n_i)
        flux[i+Nq, k] = fluxes(equation(solver), u_i)
    end
end

function get_uP_and_enforce_BC!(state, solver, state_param)
    (; Uf, uP) = low_order_cache(state)
    (; Uq, u_tilde) = state.preallocation
    (; mapP, mapI, mapO, Ival) = state_param.bcdata
    (; fq2q) = solver.discrete_data.ops
    (; K, Nfp) = solver.discrete_data.sizes

    # Initialize uP
    @batch for k = 1:K
        for i = 1:Nfp
            iP = mod1(mapP[i, k], Nfp)
            kP = div(mapP[i, k] - 1, Nfp) + 1
            uP[i, k] = Uf[iP, kP]
        end
    end

    # TODO: we still apply LF dissipation on Dirichlet boundary condition now
    # Enforce inflow BC
    @batch for i = 1:size(mapI, 1)
        ii = mapI[i]
        uP[ii] = Ival[i]
    end

    @batch for i = 1:size(mapO, 1)
        io = mapO[i]
        iP = mod1(io, Nfp)
        kP = div(io - 1, Nfp) + 1
        iq = fq2q[iP]
        uP[io] = Uq[iq, kP]
    end
end

function clear_low_order_rhs!(state, solver)
    (; rhsxyL) = state.preallocation
    (; Q0F1) = low_order_cache(state)
    (; K, Nq) = solver.discrete_data.sizes

    @batch for k = 1:K
        for i = 1:Nq
            rhsxyL[i, k] = zero(rhsxyL[i, k])
            Q0F1[i, k] = zero(Q0F1[i, k])
        end
    end
end

function accumulate_low_order_rhs_volume!(state, solver)
    (; rhsxyL, Uq) = state.preallocation
    (; flux, lambdaarr, Q0F1) = low_order_cache(state)
    (; Srs0_nnz) = solver.discrete_data.ops
    (; K, Nq) = solver.discrete_data.sizes

    @batch for k = 1:K
        # Volume contributions
        for (i, j) in Srs0_nnz
            u_i = Uq[i, k]
            u_j = Uq[j, k]
            Fxyij = @. 0.5 * (flux[i, k] + flux[j, k])
            # TODO: assume Sxy0J_ij = -Sxy0J_ji
            #              n_ij_norm = n_ji_norm
            #              Sxy0J_ji,n_ji_norm = Sx0_with_n(j,i,k,discrete_data,dim(equation))
            Sxy0J_ij, n_ij_norm = Sx0_with_n(dim(solver), i, j, k, solver)
            n_ij = @. Sxy0J_ij / n_ij_norm
            n_ji = -n_ij
            wavespeed_ij = max(wavespeed_estimate(equation(solver), u_i, n_ij), wavespeed_estimate(equation(solver), u_j, n_ji))
            lambdaarr[i, j, k] = n_ij_norm * wavespeed_ij
            lambdaarr[j, i, k] = lambdaarr[i, j, k]
            lambdaD_ij = graph_viscosity(dim(solver), i, j, k, Sxy0J_ij, state, solver)
            SFxy_lambdaD_ij = @. 2.0 * Sxy0J_ij * Fxyij - lambdaD_ij
            SFxy_lambdaD_ji = -SFxy_lambdaD_ij
            Q0F1[i, k] += SFxy_lambdaD_ij
            Q0F1[j, k] += SFxy_lambdaD_ji
        end
    end

    @batch for k = 1:K
        for i = 1:Nq
            rhsxyL[i, k] -= Q0F1[i, k]
        end
    end
end

function accumulate_low_order_rhs_surface!(state, solver, state_param)
    (; rhsxyL, BF_L, fstar_L) = state.preallocation
    (; Uf, uP, flux, wavespeed_f, lambdaBarr) = low_order_cache(state)
    (; mapP) = state_param.bcdata
    (; fq2q) = solver.discrete_data.ops
    (; K, Nq, Nfp) = solver.discrete_data.sizes

    @batch for k = 1:K
        # Surface contributions
        for i = 1:Nfp
            # TODO: refactor
            iP = mod1(mapP[i, k], Nfp)
            kP = div(mapP[i, k] - 1, Nfp) + 1

            Bxy_i, n_i_norm = Bx_with_n(dim(solver), i, k, solver)
            lambdaBarr[i, k] = 0.5 * n_i_norm * max(wavespeed_f[i, k], wavespeed_f[iP, kP])

            flux_xy_P = fluxes(equation(solver), uP[i, k])
            fstar_L[i, k] = @. 0.5 * (flux[i+Nq, k] + flux_xy_P)
            BF_L[i, k] = @. Bxy_i * fstar_L[i, k]

            lf = lambdaBarr[i, k] * (uP[i, k] - Uf[i, k])
            apply_LF_dissipation_to_BF(dim(solver), BF_L, i, k, lf, solver)
            apply_LF_dissipation_to_fstar(dim(solver), fstar_L, i, k, Bxy_i, lf, solver)

            iq = fq2q[i]
            rhsxyL[iq, k] -= BF_L[i, k]
        end
    end
end

function scale_low_order_rhs_by_mass!(state, solver)
    (; rhsL, rhsxyL) = state.preallocation
    (; Jq) = solver.discrete_data.geom
    (; wq) = solver.discrete_data.ops
    (; K, Nq) = solver.discrete_data.sizes

    @batch for k = 1:K
        # Divide by mass
        for i = 1:Nq
            wJq_i = Jq[i, k] * wq[i]
            rhsxyL[i, k] = rhsxyL[i, k] / wJq_i
            rhsL[i, k] = sum(rhsxyL[i, k])
        end
    end
end

function calculate_lambda_and_low_order_CFL!(state, solver, time_param)
    (; dtarr) = low_order_cache(state)
    (; CFL, dt0, T) = solver.param.timestepping_param
    (; Jq) = solver.discrete_data.geom
    (; wq) = solver.discrete_data.ops
    (; K, Nq) = solver.discrete_data.sizes
    (; t) = time_param

    @. dtarr = min(CFL * dt0, T - t)
    @batch for k = 1:K
        tid = Threads.threadid()
        dt = dtarr[tid]
        accumulate_alpha!(low_order_surface_flux(solver), state, k, solver)
        for i = 1:Nq
            wq_i = wq[i]
            wJq_i = Jq[i, k] * wq_i
            dt = min(dt, CFL * 0.5 * wJq_i / lambda_i(i, k, state, solver))
        end
        dtarr[tid] = dt
    end
    return minimum(dtarr)
end

function accumulate_alpha!(surface_flux::LaxFriedrichsOnNodalVal, state, k, solver)
    return nothing
end

function accumulate_alpha!(surface_flux::LaxFriedrichsOnProjectedVal, state, k, solver)
    (; alpha) = low_order_cache(state)
    (; Uq, u_tilde) = state.preallocation
    (; fq2q) = solver.discrete_data.ops
    (; Nq, Nfp, Nh) = solver.discrete_data.sizes

    utilde_f = @view u_tilde[Nq+1:Nh, :]
    for i = 1:Nfp
        # TODO: preallocate into Fmask, refactor
        iq = fq2q[i]
        alpha[i, k] = find_alpha(equation(solver), Uq[iq, k], utilde_f[i, k], solver)
    end
end

function lambda_i(i, k, state, solver)
    cache = low_order_cache(state)
    (; lambdaarr) = cache
    (; q2fq) = solver.discrete_data.ops
    (; Nq) = solver.discrete_data.sizes

    lambda_i = 0.0
    # TODO: can only consider upper diagonal
    for j = 1:Nq
        lambda_i += lambdaarr[i, j, k]
    end

    for j in q2fq[i]
        _, n_j_norm = Bx_with_n(dim(solver), j, k, solver)    # TODO: redundant
        lambda_i += lambda_B_CFL(low_order_surface_flux(solver), cache, j, n_j_norm, k)
    end

    return lambda_i
end

function lambda_B_CFL(surface_flux::LaxFriedrichsOnNodalVal, cache, i, n_i_norm, k)
    return cache.lambdaBarr[i, k]
end

function lambda_B_CFL(surface_flux::LaxFriedrichsOnProjectedVal, cache, i, n_i_norm, k)
    (; lambdaBarr, alpha, wavespeed_f) = cache

    return alpha[i, k] * lambdaBarr[i, k] + 0.5 * n_i_norm * wavespeed_f[i, k]
end

###############
###  Utils  ###
###############

# TODO: refactor with bisection
# Find alpha s.t. alpha*ui - uitilde >= 0
function find_alpha(equation::CompressibleIdealGas, ui, uitilde, solver)
    POSTOL = solver.param.global_constants.POSTOL
    alphaL = 0.0
    alphaR = 1.0
    substate = alphaR * ui - uitilde
    while (true)
        if (substate[1] > POSTOL && rhoe_ufun(equation, substate) > POSTOL)
            break
        end
        alphaR = 2 * alphaR
        substate = alphaR * ui - uitilde
    end

    maxit = 50
    iter = 0.0
    tolerance = 1e-8
    while (iter < maxit || (alphaL - alphaR) > tolerance)
        alphaM = (alphaL + alphaR) / 2
        substate = alphaM * ui - uitilde
        if (substate[1] > POSTOL && rhoe_ufun(equation, substate) > POSTOL)
            alphaR = alphaM
        else
            alphaL = alphaM
        end
        iter = iter + 1
    end

    return alphaR
end

# TODO: hardcoded
function find_alpha(equation::KPP, ui, uitilde, solver)
    return 1.0
end

