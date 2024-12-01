struct ElementwiseScaledExtrapolation <: ScaledExtrapolation end

function Base.show(io::IO, entropyproj_limiter::ElementwiseScaledExtrapolation)
    text = print(io, "ElemOpBlend")
end

# TODO: ugly dispatch
function entropy_projection!(prealloc, param, entropyproj_limiter::ElementwiseScaledExtrapolation, discrete_data, nstage, timer)
    (; Uq, vq, v_tilde, u_tilde) = prealloc
    (; Nh, Nq, Nfp) = discrete_data.sizes
    K = num_elements(param)

    @batch for k = 1:K
        vq_k = view(vq, :, k)
        v_tilde_k = view(v_tilde, :, k)
        u_tilde_k = view(u_tilde, :, k)
        Uq_k = view(Uq, :, k)
        l_k = prealloc.theta[k, nstage]
        # TODO: we can skip LGL instead of applying identity
        entropy_projection_element!(vq_k, v_tilde_k, u_tilde_k, Uq_k, l_k, param, discrete_data, prealloc)
    end
end

# TODO: unnecessary?
function clear_entropyproj_limiting_parameter_cache!(prealloc, entropyproj_limiter::ElementwiseScaledExtrapolation, nstage)
    view(prealloc.theta, :, nstage) .= 1.0
end

function solve_theta!(prealloc, cache, k, nstage, entropyproj_limiter::ElementwiseScaledExtrapolation, equation::CompressibleIdealGas, param, discrete_data, tid)
    f(theta) = update_and_check_bound_limited_entropyproj_var_on_element!(prealloc, cache, theta, k, param, discrete_data, tid)
    prealloc.theta[k, nstage] = bisection(f, 0.0, 1.0)
end

function update_limited_extrapolation!(cache, prealloc, param, entropyproj_limiter::ElementwiseScaledExtrapolation, discrete_data, k, nstage, tid)
    (; Vf_new) = cache
    (; Vf, Vf_low) = discrete_data.ops

    l_k = prealloc.theta[k, nstage]
    @. @views Vf_new[:, :, tid] = l_k * Vf + (1.0 - l_k) * Vf_low
end

function is_Vf_limited(prealloc, k, nstage, entropyproj_limiter::ElementwiseScaledExtrapolation)
    return prealloc.theta[k, nstage] < 1.0
end

function update_and_check_bound_limited_entropyproj_var_on_element!(prealloc, cache, theta, k, param, discrete_data, tid)
    try
        update_limited_entropyproj_vars_on_element!(prealloc, cache, theta, k, param.entropyproj_limiter, param, discrete_data, tid)
        return check_bound_on_element(k, cache, param, discrete_data.sizes, tid)
    catch err
        if isa(err, DomainError)
            return false
        else
            throw(err)
        end
    end
    return false
end

function update_limited_entropyproj_vars_on_element!(prealloc, cache, theta, k, entropyproj_limiter::ScaledExtrapolation, param, discrete_data, tid)
    (; Uq) = prealloc
    (; v_tilde_k, u_tilde_k, vq_k) = cache

    entropy_projection_element!(view(vq_k, :, tid), view(v_tilde_k, :, tid), view(u_tilde_k, :, tid), view(Uq, :, k), theta, param, discrete_data, prealloc)
    calculate_limited_entropyproj_vars_on_element!(cache, param, tid)
end

# TODO: hardcoded for 1D
function calculate_limited_entropyproj_vars_on_element!(cache, param, tid)
    for i = 1:size(cache.v3tilde, 1)
        calculate_limited_entropyproj_vars_on_node!(cache, i, param, tid)
    end
end

function check_bound_on_element(k, cache, param, sizes, tid)
    for i = 1:sizes.Nfp
        if !check_bound_on_face_node(i, k, cache, param, sizes, tid)
            return false
        end
    end
    return true
end
