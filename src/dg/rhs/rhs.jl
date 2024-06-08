include("./rhs_utils.jl")
include("./flux_differencing.jl")
include("./low_order_graph_viscosity.jl")

function rhs!(param, discrete_data, bcdata, prealloc, caches, t, dt, nstage, timer)
    @timeit_debug timer "initialize rhs" begin
        init_rhs!(param, param.entropyproj_limiter_type, discrete_data, bcdata, prealloc, caches, t, dt, nstage, timer)
    end
    @timeit_debug timer "rhs calculation" begin
        dt = rhs!(param.rhs_type, param, discrete_data, bcdata, prealloc, caches, t, dt, nstage, timer)
    end
    return dt
end

function init_rhs!(param, entropyproj_limiter_type::NoEntropyProjectionLimiter, discrete_data, bcdata, prealloc, caches, t, dt, nstage, timer)
    return nothing
end

function init_rhs!(param, entropyproj_limiter_type::ScaledExtrapolation, discrete_data, bcdata, prealloc, caches, t, dt, nstage, timer)
    (; approximation_basis_type) = param
    (; entropyproj_limiter_cache) = caches

    @timeit_debug timer "compute entropy projection limiting parameters" begin
        compute_entropyproj_limiting_param!(param, discrete_data, prealloc, entropyproj_limiter_cache, approximation_basis_type, nstage)
    end
end

function rhs!(rhs_type::LowOrderPositivity, param, discrete_data, bcdata, prealloc, caches, t, dt, nstage, timer)
    (; rhsL, rhsU) = prealloc
    (; rhs_cache) = caches

    @timeit_debug timer "low order positivity" begin
        dt = rhs_pos_Gauss!(prealloc, rhs_cache, param, discrete_data, bcdata, t, dt, nstage, timer, true)
    end
    copyto!(rhsU, rhsL)
    return dt
end

function rhs!(rhs_type::FluxDiffRHS, param, discrete_data, bcdata, prealloc, caches, t, dt, nstage, timer)
    (; rhsH, rhsU) = prealloc
    (; rhs_cache) = caches

    @timeit_debug timer "high order ESDG" begin
        rhs_fluxdiff!(prealloc, rhs_cache, param, discrete_data, bcdata, nstage, timer, true)
    end
    copyto!(rhsU, rhsH)
    return dt
end

function rhs!(rhs_type::LimitedDG, param, discrete_data, bcdata, prealloc, caches, t, dt, nstage, timer)
    (; rhs_cache) = caches

    @timeit_debug timer "entropy projection" begin
        entropy_projection!(prealloc, param, param.entropyproj_limiter_type, discrete_data, nstage, timer)
    end
    @timeit_debug timer "low order positivity" begin
        dt = rhs_pos_Gauss!(prealloc, rhs_cache, param, discrete_data, bcdata, t, dt, nstage, timer, false)
    end
    @timeit_debug timer "high order ESDG" begin
        rhs_fluxdiff!(prealloc, rhs_cache, param, discrete_data, bcdata, nstage, timer, false)
    end
    @timeit_debug timer "apply positivity limiter" begin
        apply_rhs_limiter!(prealloc, param, discrete_data, bcdata, caches, t, dt, nstage, param.rhs_limiter_type, timer)
    end
    return dt
end

# TODO: dispatch on element type instead of the passed in discrete data
# TODO: refactor with NodewiseScaledExtrapolation
function entropy_projection_element!(vq_k, v_tilde_k, u_tilde_k, Uq_k, l_k, param, discrete_data, prealloc)
    (; Nq, Nfp) = discrete_data.sizes

    calculate_entropy_var!(vq_k, Uq_k, param, discrete_data)    # TODO: move calculating entropy var out of entropy projection
    # For nodal collocation, quad values are the same
    for i = 1:Nq
        entropy_projection_volume_node!(v_tilde_k, u_tilde_k, vq_k, Uq_k, i, param, discrete_data, prealloc)
    end
    for i = 1:Nfp
        entropy_projection_face_node!(v_tilde_k, u_tilde_k, vq_k, i, l_k, param, discrete_data, prealloc)
    end
end

function calculate_entropy_var!(vq_k, Uq_k, param, discrete_data)
    (; Nq) = discrete_data.sizes
    for i = 1:Nq
        vq_k[i] = v_ufun(param.equation, Uq_k[i])
    end
end

function entropy_projection_volume_node!(v_tilde_k, u_tilde_k, vq_k, Uq_k, i, param, discrete_data, prealloc)
    v_tilde_k[i] = vq_k[i]
    u_tilde_k[i] = Uq_k[i]
end

function entropy_projection_face_node!(v_tilde_k, u_tilde_k, vq_k, i, l_k_i, param, discrete_data, prealloc)
    (; Nq) = discrete_data.sizes
    (; Vf, Vf_low) = discrete_data.ops
    # TODO: v_tilde_k[i+Nq] = @views sum(Vf_new[i,:].*vq_k)
    #       requires allocation... why?
    v_tilde_k[i+Nq] = zero(v_tilde_k[i+Nq])
    for j = 1:Nq
        v_tilde_k[i+Nq] += (l_k_i * Vf[i, j] + (1 - l_k_i) * Vf_low[i, j]) * vq_k[j]
    end
    u_tilde_k[i+Nq] = u_vfun(param.equation, v_tilde_k[i+Nq])
end

# TODO: ugly dispatch
function entropy_projection!(prealloc, param, entropyproj_limiter_type::NoEntropyProjectionLimiter, discrete_data, nstage, timer)
    (; Uq, vq, v_tilde, u_tilde) = prealloc
    (; K) = discrete_data.sizes

    @batch for k = 1:K
        vq_k = view(vq, :, k)
        v_tilde_k = view(v_tilde, :, k)
        u_tilde_k = view(u_tilde, :, k)
        Uq_k = view(Uq, :, k)
        l_k = 1.0
        # TODO: we can skip LGL instead of applying identity
        entropy_projection_element!(vq_k, v_tilde_k, u_tilde_k, Uq_k, l_k, param, discrete_data, prealloc)
    end
end

# TODO: ugly dispatch
function entropy_projection!(prealloc, param, entropyproj_limiter_type::NodewiseScaledExtrapolation, discrete_data, nstage, timer)
    (; Uq, vq, v_tilde, u_tilde) = prealloc
    (; K, Nq, Nfp) = discrete_data.sizes

    @batch for k = 1:K
        vq_k = view(vq, :, k)
        v_tilde_k = view(v_tilde, :, k)
        u_tilde_k = view(u_tilde, :, k)
        Uq_k = view(Uq, :, k)
        # TODO: we can skip LGL instead of applying identity
        # TODO: refactor
        calculate_entropy_var!(vq_k, Uq_k, param, discrete_data)
        for i = 1:Nq
            entropy_projection_volume_node!(v_tilde_k, u_tilde_k, vq_k, Uq_k, i, param, discrete_data, prealloc)
        end
        for i = 1:Nfp
            l_k_i = prealloc.θ_local_arr[i, k, nstage]
            entropy_projection_face_node!(v_tilde_k, u_tilde_k, vq_k, i, l_k_i, param, discrete_data, prealloc)
        end
    end
end
