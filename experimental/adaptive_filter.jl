abstract type AdaptiveFilter <: EntropyProjectionLimiterType end
struct ExponentialFilter <: AdaptiveFilter end
struct ZhangShuFilter <: AdaptiveFilter end

function entropy_projection!(prealloc, param, entropyproj_limiter_type::Union{AdaptiveFilter,NoEntropyProjectionLimiter}, discrete_data, nstage, timer)
    (; Uq, vq, v_tilde, u_tilde) = prealloc
    (; Nh, Nq, Nfp) = discrete_data.sizes
    K = get_num_elements(param)

    for k = 1:K
        vq_k = view(vq, :, k)
        v_tilde_k = view(v_tilde, :, k)
        u_tilde_k = view(u_tilde, :, k)
        Uq_k = view(Uq, :, k)
        l_k = 1.0
        # TODO: we can skip LGL instead of applying identity
        entropy_projection_element!(vq_k, v_tilde_k, u_tilde_k, Uq_k, l_k, param, discrete_data, prealloc)
    end
end

function init_get_rhs!(param, entropyproj_limiter_type::AdaptiveFilter, discrete_data, bcdata, prealloc, caches, t, dt, nstage, timer)
    (; approximation_basis_type) = param
    (; entropyproj_limiter_cache) = caches

    @timeit_debug timer "compute modal coefficients" begin
        compute_modal_coefficients!(prealloc, param, discrete_data, entropyproj_limiter_cache)
    end
    @timeit_debug timer "compute entropy projection limiting parameters" begin
        compute_entropyproj_limiting_param!(param, discrete_data, prealloc, entropyproj_limiter_cache, approximation_basis_type, nstage)
    end
    @timeit_debug timer "apply entropy projection limiting" begin
        apply_entropyproj_filtering!(prealloc, param, param.entropyproj_limiter_type, discrete_data, nstage)
    end
end

function project_flux_difference_to_quad!(cache, prealloc, param, entropyproj_limiter_type::AdaptiveFilter, discrete_data, k, nstage)
    project_flux_difference_to_quad_unlimited!(k, cache, prealloc, discrete_data)
end

function get_entropyproj_limiter_cache(entropyproj_limiter_type::AdaptiveFilter, param, sizes)
    (; Np, Nh, Nq, Nfp, Nc, Ns) = sizes
    K = get_num_elements(param)
    Nd = get_dim(param.equation)

    return EntropyProjectionLimiterCache{Nd,Nc}(K=K, Np=Np, Nq=Nq, Nh=Nh, Nfp=Nfp)
end

# TODO: unnecessary?
function clear_entropyproj_limiting_parameter_cache!(prealloc, entropyproj_limiter_type::ExponentialFilter, nstage)
    view(prealloc.θ_arr, :, nstage) .= 0.0
end

# TODO: unnecessary?
function clear_entropyproj_limiting_parameter_cache!(prealloc, entropyproj_limiter_type::ZhangShuFilter, nstage)
    view(prealloc.θ_arr, :, nstage) .= 1.0
end

function solve_theta!(prealloc, cache, k, nstage, entropyproj_limiter_type::ExponentialFilter, param, discrete_data)
    f(θ) = update_and_check_bound_limited_entropyproj_var_on_element!(prealloc, cache, θ, k, param, discrete_data)
    prealloc.θ_arr[k, nstage] = bisection(f, -log(param.global_constants.ZEROTOL), 0.0)
end

function solve_theta!(prealloc, cache, k, nstage, entropyproj_limiter_type::ZhangShuFilter, param, discrete_data)
    f(θ) = update_and_check_bound_limited_entropyproj_var_on_element!(prealloc, cache, θ, k, param, discrete_data)
    prealloc.θ_arr[k, nstage] = bisection(f, 0.0, 1.0)
end

function update_limited_entropyproj_vars_on_element!(prealloc, cache, θ, k, entropyproj_limiter_type::AdaptiveFilter, param, discrete_data)
    (; VqVDM) = discrete_data.ops
    (; U_modal, U_k, Uq_k, vq_k, v_tilde_k, u_tilde_k) = cache

    U_k .= @views U_modal[:, k]
    apply_filter!(U_k, param.entropyproj_limiter_type, param.equation, θ)
    mul!(Uq_k, VqVDM, U_k)

    # TODO: only project to Gauss element
    entropy_projection_element!(vq_k, v_tilde_k, u_tilde_k, Uq_k, 1.0, param, discrete_data, prealloc)
    calculate_limited_entropyproj_vars_on_element!(cache, param)
end

#########################
### Filtering methods ###
#########################
function compute_modal_coefficients!(prealloc, param, discrete_data, cache)
    (; Uq) = prealloc
    (; U_modal) = cache
    (; VDMinvPq) = discrete_data.ops

    K = get_num_elements(param)
    for k = 1:K
        @views mul!(U_modal[:, k], VDMinvPq, Uq[:, k])      # TODO: why there is allocation when remove k = 1:K loop?
    end
end

function apply_entropyproj_filtering!(prealloc, param, entropyproj_limiter_type::AdaptiveFilter, discrete_data, nstage)
    (; Uq, θ_arr, U_modal) = prealloc
    (; VqVDM) = discrete_data.ops

    K = get_num_elements(param)
    for k = 1:K
        U_modal_k = @views U_modal[:, k]
        apply_filter!(U_modal_k, param.entropyproj_limiter_type, param.equation, θ_arr[k, nstage])
    end
    for k = 1:K
        @views mul!(Uq[:, k], VqVDM, U_modal[:, k])      # TODO: why there is allocation when remove k = 1:K loop?
    end
end

function apply_entropyproj_filtering!(prealloc, param, entropyproj_limiter_type::ScaledExtrapolation, discrete_data, nstage)
    # Do nothing
end

function apply_filter!(U, entropyproj_limiter_type::ExponentialFilter, equation::EquationType{Dim1}, θ)
    for i = 1:size(U, 1)
        U[i] = exp(-θ * (i - 1) * (i - 1)) * U[i]
    end
end

function apply_filter!(U, entropyproj_limiter_type::ZhangShuFilter, equation::EquationType{Dim1}, θ)
    for i = 2:size(U, 1)
        U[i] = θ * U[i]
    end
end


# Post processing
function plot_limiting_bar!(entropyproj_limiter_type::ExponentialFilter, x, barL, normalization_factor, k)
    plot!(x, barL[k] * ones(2) / normalization_factor, st=:bar, alpha=0.2)
end

function plot_limiting_bar!(entropyproj_limiter_type::ZhangShuFilter, x, barL, normalization_factor, k)
    plot!(x, (1 - barL[k]) * ones(2), st=:bar, alpha=0.2)
end

