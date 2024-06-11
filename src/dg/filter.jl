function compute_entropyproj_limiting_param!(approx_basis_type::LobattoCollocation, state, solver, time_param)
    clear_entropyproj_limiting_parameter_cache!(prealloc, param.entropyproj_limiter_type, nstage)
end

# TODO: put into entropy projection to avoid an extra projection step
function compute_entropyproj_limiting_param!(approx_basis_type::GaussCollocation, state, solver, time_param)
    (; K) = solver.discrete_data.sizes
    clear_entropyproj_limiting_parameter_cache!(entropyproj_limiter(solver), state, time_param)
    # TODO: possible redundant calculation, only used for calculation of bounds on the fly
    calc_face_values!(equation(solver), state, solver)
    @batch for k = 1:K
        tid = Threads.threadid()
        solve_theta!(entropyproj_limiter(solver), equation(solver), state, k, solver, time_param, tid)
    end
end

# TODO: unnecessary?
function clear_entropyproj_limiting_parameter_cache!(entropyproj_limiter_type::NodewiseScaledExtrapolation, state, time_param)
    view(state.preallocation.theta_local_arr, :, :, time_param.nstage) .= 1.0
end

function clear_entropyproj_limiting_parameter_cache!(entropyproj_limiter_type::NoEntropyProjectionLimiter, state, time_param)
    # Do nothing
end

function calc_face_values!(equation::CompressibleEulerIdealGas, state, solver)
    (; Uq, vq) = state.preallocation
    (; Uf, VUf, rhoef) = state.cache.entropyproj_limiter_cache
    (; Vf) = solver.discrete_data.ops
    (; K, Nq, Nfp) = solver.discrete_data.sizes

    @batch for k = 1:K
        @views mul!(Uf[:, k], Vf, Uq[:, k])
        for i = 1:Nq
            vq[i, k] = v_ufun(equation(solver), Uq[i, k])
        end
        for i = 1:Nfp
            rhoef[i, k] = rhoe_ufun(equation(solver), Uf[i, k])
        end
        @views mul!(VUf[:, k], Vf, vq[:, k])
    end
end

# TODO: do nothing for KPP now
function calc_face_values!(equation::KPP, state, solver)
    return nothing
end

function solve_theta!(entropyproj_limiter_type::NodewiseScaledExtrapolation, equation::CompressibleIdealGas, state, k, solver, time_param, tid)
    (; theta_arr, theta_local_arr) = state.preallocation
    (; Nfp) = solver.discrete_data.sizes
    (; nstage) = time_param
    for i = 1:Nfp
        f(theta_i) = update_and_check_bound_limited_entropyproj_var_on_face_node!(state, theta_i, i, k, tid, solver)
        theta_local_arr[i, k, nstage] = bisection(f, 0.0, 1.0)
    end
    # TODO: hardcode for post postprocessing
    theta_arr[k, nstage] = sum(view(theta_local_arr, :, k, nstage)) / Nfp
end

function solve_theta!(entropyproj_limiter_type::NoEntropyProjectionLimiter, equation::CompressibleIdealGas, state, k, solver, time_param, tid)
    return 1.0
end

# TODO: do nothing for KPP now
function solve_theta!(entropyproj_limiter_type, equation::KPP, state, k, solver, time_param, tid)
    return nothing
end

function update_limited_entropyproj_vars!(entropyproj_limiter_type::NoEntropyProjectionLimiter, state, theta_i, i, k, tid, solver)
    # Do nothing
end

# TODO: better to define custom index type for hybridized, quad, and surface nodes...
function calculate_limited_entropyproj_vars_on_node!(state, i, tid, solver)
    (; v3tilde, rhotilde, rhoetilde, v_tilde_k, u_tilde_k) = state.cache.entropyproj_limiter_cache
    v3tilde[i, tid] = v_tilde_k[i, tid][end]
    rhotilde[i, tid] = u_tilde_k[i, tid][1]
    rhoetilde[i, tid] = rhoe_ufun(equation(solver), u_tilde_k[i, tid])
end

function calculate_limited_entropyproj_vars_on_face_node!(state, i, tid, solver)
    calculate_limited_entropyproj_vars_on_node!(state, i + solver.discrete_data.sizes.Nq, tid, solver)
end

# TODO: skip volume quadrature points
function check_bound_on_face_node(state, i, k, tid, solver)
    (; v3tilde, rhotilde, rhoetilde, VUf, rhoef, Uf) = state.cache.entropyproj_limiter_cache
    (; Nq) = solver.discrete_data.sizes
    ϵ = solver.param.global_constants.POSTOL
    zeta = solver.param.limiting_param.zeta
    eta = solver.param.limiting_param.eta
    return (v3tilde[Nq+i, tid] < min(zeta * VUf[i, k][end], -ϵ)
            && rhotilde[Nq+i, tid] > max((1 - eta) * Uf[i, k][1], ϵ)
            && rhotilde[Nq+i, tid] < (1 + eta) * Uf[i, k][1]
            && rhoetilde[Nq+i, tid] > max((1 - eta) * rhoef[i, k], ϵ)
            && rhoetilde[Nq+i, tid] < (1 + eta) * rhoef[i, k])
end

# TODO: Refactor. element versus. node - use multiple dispatch
function update_and_check_bound_limited_entropyproj_var_on_face_node!(state, theta_i, i, k, tid, solver)
    if update_limited_entropyproj_vars_on_face_node!(entropyproj_limiter(solver), state, theta_i, i, k, tid, solver)
        return check_bound_on_face_node(state, i, k, tid, solver)
    else
        return false
    end
end

function update_limited_entropyproj_vars_on_face_node!(entropyproj_limiter_type::NodewiseScaledExtrapolation, state, theta_i, i, k, tid, solver)
    (; vq) = state.preallocation
    (; v_tilde_k, u_tilde_k) = state.cache.entropyproj_limiter_cache
    (; Nq) = solver.discrete_data.sizes
    (; Vf, Vf_low) = solver.discrete_data.ops
    ϵ = solver.param.global_constants.POSTOL

    # TODO: applying the function directly results in allocations
    # entropy_projection_face_node!(view(v_tilde_k,:,tid),view(u_tilde_k,:,tid),view(vq_k,:,tid),i,theta_i,param,discrete_data,prealloc)
    v_tilde_k[i+Nq, tid] = zero(v_tilde_k[i+Nq, tid])
    for j = 1:Nq
        v_tilde_k[i+Nq, tid] += (theta_i * Vf[i, j] + (1 - theta_i) * Vf_low[i, j]) * vq[j, k]
    end
    # Check well-definedness
    if v_tilde_k[i+Nq, tid][end] < -ϵ
        u_tilde_k[i+Nq, tid] = u_vfun(equation(solver), v_tilde_k[i+Nq, tid])
        calculate_limited_entropyproj_vars_on_face_node!(state, i, tid, solver)
        return true
    else
        return false
    end
end
