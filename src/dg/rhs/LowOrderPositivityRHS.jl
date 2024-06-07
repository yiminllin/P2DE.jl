##########################################
### RHS of positivity preserving Gauss ###
##########################################
function rhs_pos_Gauss!(prealloc, rhs_cache, param, discrete_data, bcdata, t, dt, nstage, timer, need_proj=true)
    (; entropyproj_limiter_type) = param

    cache = low_order_cache(rhs_cache)
    @timeit_debug timer "entropy projection" begin
        if (need_proj)
            entropy_projection!(prealloc, param, entropyproj_limiter_type, discrete_data, nstage, timer)
        end
    end
    @timeit_debug timer "calculate wavespeed and inviscid flux" begin
        calculate_wavespeed_and_inviscid_flux!(cache, prealloc, param, discrete_data)
    end
    @timeit_debug timer "enforce boundary condition" begin
        get_uP_and_enforce_BC!(cache, prealloc, param, bcdata, discrete_data)
    end

    # Assemble RHS
    @timeit_debug timer "clear cache" begin
        clear_low_order_rhs!(cache, prealloc, param, discrete_data)
    end
    @timeit_debug timer "Low order volume kernel" begin
        accumulate_low_order_rhs_volume!(cache, prealloc, param, discrete_data)
    end
    @timeit_debug timer "Low order surface kernel" begin
        accumulate_low_order_rhs_surface!(cache, prealloc, param, discrete_data, bcdata)
    end
    @timeit_debug timer "Scale low order solution by mass" begin
        scale_low_order_rhs_by_mass!(prealloc, param, discrete_data)
    end

    @timeit_debug timer "Calculate low order positivity CFL" begin
        # Determine positivity CFL
        if (nstage == 1)
            dt = calculate_lambda_and_low_order_CFL!(cache, prealloc, param, discrete_data, bcdata, t)
        end
    end

    # check_low_order_entropy_stability(cache,prealloc,param,discrete_data,dim_type(equation))

    return dt
end

function calculate_wavespeed_and_inviscid_flux!(cache, prealloc, param, discrete_data)
    (; K) = discrete_data.sizes
    @batch for k = 1:K
        update_face_values!(cache, prealloc, k, discrete_data, low_order_surface_flux_type(param.rhs_type))
        update_wavespeed_and_inviscid_flux!(cache, prealloc, k, param, discrete_data)
    end
end

function update_face_values!(cache, prealloc, k, discrete_data, surface_flux_type::LaxFriedrichsOnNodalVal)
    (; Uf) = cache
    (; Uq) = prealloc
    (; fq2q) = discrete_data.ops
    (; Nfp) = discrete_data.sizes

    for i = 1:Nfp
        iq = fq2q[i]
        Uf[i, k] = Uq[iq, k]
    end
end

function update_face_values!(cache, prealloc, k, discrete_data, surface_flux_type::LaxFriedrichsOnProjectedVal)
    (; Uf) = cache
    (; u_tilde) = prealloc
    (; Nq, Nfp) = discrete_data.sizes

    for i = 1:Nfp
        Uf[i, k] = u_tilde[i+Nq, k]
    end
end

function update_wavespeed_and_inviscid_flux!(cache, prealloc, k, param, discrete_data)
    (; equation) = param
    (; Uq) = prealloc
    (; Uf, wavespeed_f, flux) = cache
    (; Nq, Nfp) = discrete_data.sizes

    # Volume inviscid flux
    for i = 1:Nq
        u_i = Uq[i, k]
        flux[i, k] = fluxes(equation, u_i)
    end

    # Surface wavespeed and inviscid flux
    for i = 1:Nfp
        u_i = Uf[i, k]
        Bxy_i, n_i_norm = Bx_with_n(i, k, discrete_data, dim_type(equation))
        n_i = @. Bxy_i / n_i_norm
        wavespeed_f[i, k] = wavespeed_estimate(equation, u_i, n_i)
        flux[i+Nq, k] = fluxes(equation, u_i)
    end
end

function get_uP_and_enforce_BC!(cache, prealloc, param, bcdata, discrete_data)
    (; Uf, uP) = cache
    (; Uq, u_tilde) = prealloc
    (; mapP, mapI, mapO, Ival) = bcdata
    (; fq2q) = discrete_data.ops
    (; K, Nfp) = discrete_data.sizes

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

function clear_low_order_rhs!(cache, prealloc, param, discrete_data)
    (; rhsxyL) = prealloc
    (; Q0F1) = cache
    (; K, Nq) = discrete_data.sizes

    @batch for k = 1:K
        for i = 1:Nq
            rhsxyL[i, k] = zero(rhsxyL[i, k])
            Q0F1[i, k] = zero(Q0F1[i, k])
        end
    end
end

function accumulate_low_order_rhs_volume!(cache, prealloc, param, discrete_data)
    (; equation) = param
    (; rhsxyL, Uq) = prealloc
    (; flux, λarr, Q0F1) = cache
    (; Srs0_nnz) = discrete_data.ops
    (; K, Nq) = discrete_data.sizes

    @batch for k = 1:K
        # Volume contributions
        for (i, j) in Srs0_nnz
            u_i = Uq[i, k]
            u_j = Uq[j, k]
            Fxyij = @. 0.5 * (flux[i, k] + flux[j, k])
            # TODO: assume Sxy0J_ij = -Sxy0J_ji
            #              n_ij_norm = n_ji_norm
            #              Sxy0J_ji,n_ji_norm = Sx0_with_n(j,i,k,discrete_data,dim_type(equation))
            Sxy0J_ij, n_ij_norm = Sx0_with_n(i, j, k, discrete_data, dim_type(equation))
            n_ij = @. Sxy0J_ij / n_ij_norm
            n_ji = -n_ij
            wavespeed_ij = max(wavespeed_estimate(equation, u_i, n_ij), wavespeed_estimate(equation, u_j, n_ji))
            λarr[i, j, k] = n_ij_norm * wavespeed_ij
            λarr[j, i, k] = λarr[i, j, k]
            ΛD_ij = graph_viscosity(cache, prealloc, param, i, j, k, Sxy0J_ij, dim_type(equation))
            SFxy_ΛD_ij = @. 2.0 * Sxy0J_ij * Fxyij - ΛD_ij
            SFxy_ΛD_ji = -SFxy_ΛD_ij
            Q0F1[i, k] += SFxy_ΛD_ij
            Q0F1[j, k] += SFxy_ΛD_ji
        end
    end

    @batch for k = 1:K
        for i = 1:Nq
            rhsxyL[i, k] -= Q0F1[i, k]
        end
    end
end

function accumulate_low_order_rhs_surface!(cache, prealloc, param, discrete_data, bcdata)
    (; equation) = param
    (; rhsxyL, BF_L, fstar_L) = prealloc
    (; Uf, uP, flux, wavespeed_f, λBarr) = cache
    (; mapP) = bcdata
    (; fq2q) = discrete_data.ops
    (; K, Nq, Nfp) = discrete_data.sizes

    @batch for k = 1:K
        # Surface contributions
        for i = 1:Nfp
            # TODO: refactor
            iP = mod1(mapP[i, k], Nfp)
            kP = div(mapP[i, k] - 1, Nfp) + 1

            Bxy_i, n_i_norm = Bx_with_n(i, k, discrete_data, dim_type(equation))
            λBarr[i, k] = 0.5 * n_i_norm * max(wavespeed_f[i, k], wavespeed_f[iP, kP])

            flux_xy_P = fluxes(equation, uP[i, k])
            fstar_L[i, k] = @. 0.5 * (flux[i+Nq, k] + flux_xy_P)
            BF_L[i, k] = @. Bxy_i * fstar_L[i, k]

            lf = λBarr[i, k] * (uP[i, k] - Uf[i, k])
            apply_LF_dissipation_to_BF(BF_L, param, i, k, lf, dim_type(equation))
            apply_LF_dissipation_to_fstar(fstar_L, param, i, k, Bxy_i, lf, dim_type(equation))

            iq = fq2q[i]
            rhsxyL[iq, k] -= BF_L[i, k]
        end
    end
end

function scale_low_order_rhs_by_mass!(prealloc, param, discrete_data)
    (; Jq) = discrete_data.geom
    (; wq) = discrete_data.ops
    (; rhsL, rhsxyL) = prealloc
    (; K, Nq) = discrete_data.sizes

    @batch for k = 1:K
        # Divide by mass
        for i = 1:Nq
            wJq_i = Jq[i, k] * wq[i]
            rhsxyL[i, k] = rhsxyL[i, k] / wJq_i
            rhsL[i, k] = sum(rhsxyL[i, k])
        end
    end
end

function calculate_lambda_and_low_order_CFL!(cache, prealloc, param, discrete_data, bcdata, t)
    (; CFL, dt0, T) = param.timestepping_param
    (; Jq) = discrete_data.geom
    (; wq) = discrete_data.ops
    (; dtarr) = cache
    (; K, Nq) = discrete_data.sizes

    @. dtarr = min(CFL * dt0, T - t)
    @batch for k = 1:K
        tid = Threads.threadid()
        dt = dtarr[tid]
        accumulate_alpha!(cache, prealloc, k, param, discrete_data, low_order_surface_flux_type(param.rhs_type))
        for i = 1:Nq
            wq_i = wq[i]
            wJq_i = Jq[i, k] * wq_i
            dt = min(dt, CFL * 0.5 * wJq_i / lambda_i(i, k, cache, prealloc, param, discrete_data, bcdata))
        end
        dtarr[tid] = dt
    end
    return minimum(dtarr)
end

function accumulate_alpha!(cache, prealloc, k, param, discrete_data, surface_flux_type::LaxFriedrichsOnNodalVal)
    return nothing
end

function accumulate_alpha!(cache, prealloc, k, param, discrete_data, surface_flux_type::LaxFriedrichsOnProjectedVal)
    (; αarr) = cache
    (; Uq, u_tilde) = prealloc
    (; fq2q) = discrete_data.ops
    (; Nq, Nfp, Nh) = discrete_data.sizes
    (; equation) = param

    utilde_f = @view u_tilde[Nq+1:Nh, :]
    for i = 1:Nfp
        # TODO: preallocate into Fmask, refactor
        iq = fq2q[i]
        αarr[i, k] = find_alpha(equation, param, Uq[iq, k], utilde_f[i, k])
    end
end

function lambda_i(i, k, cache, prealloc, param, discrete_data, bcdata)
    (; equation) = param
    (; λarr) = cache
    (; q2fq) = discrete_data.ops
    (; Nq) = discrete_data.sizes

    lambda_i = 0.0
    # TODO: can only consider upper diagonal
    for j = 1:Nq
        lambda_i += λarr[i, j, k]
    end

    surface_flux_type = low_order_surface_flux_type(param.rhs_type)
    for j in q2fq[i]
        _, n_j_norm = Bx_with_n(j, k, discrete_data, dim_type(equation))    # TODO: redundant
        lambda_i += lambda_B_CFL(cache, j, n_j_norm, k, surface_flux_type)
    end

    return lambda_i
end

function lambda_B_CFL(cache, i, n_i_norm, k, surface_flux_type::LaxFriedrichsOnNodalVal)
    return cache.λBarr[i, k]
end

function lambda_B_CFL(cache, i, n_i_norm, k, surface_flux_type::LaxFriedrichsOnProjectedVal)
    (; λBarr, αarr, wavespeed_f) = cache

    return αarr[i, k] * λBarr[i, k] + 0.5 * n_i_norm * wavespeed_f[i, k]
end

###############
###  Utils  ###
###############

# TODO: refactor with bisection
# Find alpha s.t. alpha*ui - uitilde >= 0
function find_alpha(equation::CompressibleIdealGas, param, ui, uitilde)
    (; equation) = param
    POSTOL = param.global_constants.POSTOL
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
function find_alpha(equation::KPP, param, ui, uitilde)
    return 1.0
end

function check_low_order_entropy_stability(cache, prealloc, param, discrete_data, dim::Dim2)
    (; equation) = param
    (; rhsxyL, fstar_L, vq) = prealloc
    (; Uf) = cache
    (; Jq) = discrete_data.geom
    (; wq) = discrete_data.ops
    (; K, Nq, Nfp) = discrete_data.sizes

    @batch for k = 1:K
        entropy_estimate = zero(SVector{dim(equation),Float64})   # vT rhs
        # TODO: hardcoded
        update_face_values!(cache, prealloc, k, discrete_data, LaxFriedrichsOnNodalVal())
        for i = 1:Nq
            m_i = wq[i]
            entropy_estimate += m_i * Jq[i, k] * SVector(sum(vq[i, k] .* rhsxyL[i, k][1]), sum(vq[i, k] .* rhsxyL[i, k][2]))
        end
        sum_Bpsi = zero(SVector{dim(equation),Float64})   # 1T B psi
        vTBfstar = zero(SVector{dim(equation),Float64})   # vT B f*
        for i = 1:Nfp
            Bxy_i = Bx(i, k, discrete_data, dim_type(equation))
            sum_Bpsi += Bxy_i .* psi_ufun(equation, Uf[i, k])
            vi = v_ufun(equation, Uf[i, k])
            vTBfstar += Bxy_i .* SVector(sum(vi .* fstar_L[i, k][1]), sum(vi .* fstar_L[i, k][2]))
        end
        diff = entropy_estimate - sum_Bpsi + vTBfstar
        tol = 1e-12
        if diff[1] > tol || diff[2] > tol
            @show k, diff
        end
    end
end
