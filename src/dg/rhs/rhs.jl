include("./rhs_utils.jl")
include("./flux_differencing.jl")
include("./low_order_graph_viscosity.jl")

function rhs!(state, solver, state_param, time_param)
    @timeit_debug time_param.timer "initialize rhs" begin
        init_rhs!(entropyproj_limiter(solver), state, solver, time_param)
    end
    @timeit_debug time_param.timer "rhs calculation" begin
        dt = rhs!(rhs(solver), state, solver, state_param, time_param)
    end
    return dt
end

function init_rhs!(entropyproj_limiter_type::NoEntropyProjectionLimiter, state, solver, time_param)
    return nothing
end

function init_rhs!(entropyproj_limiter_type::ScaledExtrapolation, state, solver, time_param)
    @timeit_debug time_param.timer "compute entropy projection limiting parameters" begin
        compute_entropyproj_limiting_param!(basis(solver), state, solver, time_param)
    end
end

function rhs!(rhs_type::LowOrderPositivity, state, solver, state_param, time_param)
    @timeit_debug time_param.timer "low order positivity" begin
        dt = rhs_low_graph_visc!(state, solver, state_param, time_param, true)
    end
    copyto!(state.preallocation.rhsU, state.preallocation.rhsL)
    return dt
end

function rhs!(rhs_type::FluxDiffRHS, state, solver, state_param, time_param)
    @timeit_debug time_param.timer "high order ESDG" begin
        rhs_fluxdiff!(state, solver, state_param, time_param, true)
    end
    copyto!(state.preallocation.rhsU, state.preallocation.rhsH)
    return time_param.dt
end

function rhs!(rhs_type::LimitedDG, state, solver, state_param, time_param)
    @timeit_debug time_param.timer "entropy projection" begin
        entropy_projection!(entropyproj_limiter(solver), state, solver, time_param)
    end
    @timeit_debug time_param.timer "low order positivity" begin
        dt = rhs_low_graph_visc!(state, solver, state_param, time_param, false)
    end
    @timeit_debug time_param.timer "high order ESDG" begin
        rhs_fluxdiff!(state, solver, state_param, time_param, false)
    end
    @timeit_debug time_param.timer "apply positivity limiter" begin
        apply_rhs_limiter!(limiter(solver), state, solver, state_param, time_param)
    end
    return dt
end

# TODO: dispatch on element type instead of the passed in discrete data
# TODO: refactor with NodewiseScaledExtrapolation
function entropy_projection_element!(vq_k, v_tilde_k, u_tilde_k, Uq_k, l_k, solver)
    (; Nq, Nfp) = solver.discrete_data.sizes

    calculate_entropy_var!(vq_k, Uq_k, solver)    # TODO: move calculating entropy var out of entropy projection
    # For nodal collocation, quad values are the same
    for i = 1:Nq
        entropy_projection_volume_node!(v_tilde_k, u_tilde_k, vq_k, Uq_k, i)
    end
    for i = 1:Nfp
        entropy_projection_face_node!(v_tilde_k, u_tilde_k, vq_k, i, l_k, solver)
    end
end

function calculate_entropy_var!(vq_k, Uq_k, solver)
    (; Nq) = solver.discrete_data.sizes
    for i = 1:Nq
        vq_k[i] = v_ufun(equation(solver), Uq_k[i])
    end
end

function entropy_projection_volume_node!(v_tilde_k, u_tilde_k, vq_k, Uq_k, i)
    v_tilde_k[i] = vq_k[i]
    u_tilde_k[i] = Uq_k[i]
end

function entropy_projection_face_node!(v_tilde_k, u_tilde_k, vq_k, i, l_k_i, solver)
    (; Nq) = solver.discrete_data.sizes
    (; Vf, Vf_low) = solver.discrete_data.ops
    # TODO: v_tilde_k[i+Nq] = @views sum(Vf_new[i,:].*vq_k)
    #       requires allocation... why?
    v_tilde_k[i+Nq] = zero(v_tilde_k[i+Nq])
    for j = 1:Nq
        v_tilde_k[i+Nq] += (l_k_i * Vf[i, j] + (1 - l_k_i) * Vf_low[i, j]) * vq_k[j]
    end
    u_tilde_k[i+Nq] = u_vfun(equation(solver), v_tilde_k[i+Nq])
end

# TODO: ugly dispatch
function entropy_projection!(entropyproj_limiter_type::NoEntropyProjectionLimiter, state, solver, time_param)
    (; Uq, vq, v_tilde, u_tilde) = state.preallocation
    (; K) = solver.discrete_data.sizes

    @batch for k = 1:K
        vq_k = view(vq, :, k)
        v_tilde_k = view(v_tilde, :, k)
        u_tilde_k = view(u_tilde, :, k)
        Uq_k = view(Uq, :, k)
        l_k = 1.0
        # TODO: we can skip LGL instead of applying identity
        entropy_projection_element!(vq_k, v_tilde_k, u_tilde_k, Uq_k, l_k, solver)
    end
end

# TODO: ugly dispatch
function entropy_projection!(entropyproj_limiter_type::NodewiseScaledExtrapolation, state, solver, time_param)
    (; Uq, vq, v_tilde, u_tilde) = state.preallocation
    (; K, Nq, Nfp) = solver.discrete_data.sizes

    @batch for k = 1:K
        vq_k = view(vq, :, k)
        v_tilde_k = view(v_tilde, :, k)
        u_tilde_k = view(u_tilde, :, k)
        Uq_k = view(Uq, :, k)
        # TODO: we can skip LGL instead of applying identity
        # TODO: refactor
        calculate_entropy_var!(vq_k, Uq_k, solver)
        for i = 1:Nq
            entropy_projection_volume_node!(v_tilde_k, u_tilde_k, vq_k, Uq_k, i)
        end
        for i = 1:Nfp
            l_k_i = state.preallocation.theta_local_arr[i, k, time_param.nstage]
            entropy_projection_face_node!(v_tilde_k, u_tilde_k, vq_k, i, l_k_i, solver)
        end
    end
end
