# TODO: put into entropy projection to avoid an extra projection step
function compute_entropyproj_limiting_param!(param,discrete_data_gauss,prealloc,cache,nstage)
    @unpack LGLind = prealloc

    K  = get_num_elements(param)
    clear_entropyproj_limiting_parameter_cache!(prealloc,param.entropyproj_limiter_type,nstage)
    # TODO: possible redundant calculation, only used for calculation of bounds on the fly
    calc_face_values!(prealloc,cache,param,discrete_data_gauss)
    for k = 1:K
        if (!LGLind[k])
            solve_theta!(prealloc,cache,k,nstage,param.entropyproj_limiter_type,param,discrete_data_gauss)
        end
    end
end

# TODO: unnecessary?
function clear_entropyproj_limiting_parameter_cache!(prealloc,entropyproj_limiter_type::ExponentialFilter,nstage)
    view(prealloc.Farr,:,nstage) .= 0.0
end

# TODO: unnecessary?
function clear_entropyproj_limiting_parameter_cache!(prealloc,entropyproj_limiter_type::ZhangShuFilter,nstage)
    view(prealloc.Farr,:,nstage) .= 1.0
end

# TODO: unnecessary?
function clear_entropyproj_limiting_parameter_cache!(prealloc,entropyproj_limiter_type::ElementwiseScaledExtrapolation,nstage)
    view(prealloc.Farr,:,nstage) .= 1.0
end

# TODO: unnecessary?
function clear_entropyproj_limiting_parameter_cache!(prealloc,entropyproj_limiter_type::NodewiseScaledExtrapolation,nstage)
    view(prealloc.θ_local_arr,:,:,nstage) .= 1.0
end

function clear_entropyproj_limiting_parameter_cache!(prealloc,entropyproj_limiter_type::NoEntropyProjectionLimiter,nstage)
    # Do nothing
end

# TODO: refactor, only work for gauss
function calc_face_values!(prealloc,cache,param,discrete_data_gauss)
    @unpack Uq,vq        = prealloc
    @unpack Uf,VUf,rhoef = cache
    @unpack Vf           = discrete_data_gauss.ops
    
    K  = get_num_elements(param)
    for k = 1:K
        @views mul!(Uf[:,k],Vf,Uq[:,k])
        for i = 1:size(vq,1)
            vq[i,k] = v_ufun(param.equation,Uq[i,k])
        end
        for i = 1:size(VUf,1)
            rhoef[i,k] = rhoe_ufun(param.equation,Uf[i,k])
        end
        @views mul!(VUf[:,k],Vf,vq[:,k])
    end
end

function solve_theta!(prealloc,cache,k,nstage,entropyproj_limiter_type::ExponentialFilter,param,discrete_data_gauss)
    f(θ) = update_and_check_bound_limited_entropyproj_var_on_element!(prealloc,cache,θ,k,param,discrete_data_gauss)
    prealloc.Farr[k,nstage] = bisection(f,-log(param.global_constants.ZEROTOL),0.0)
end

function solve_theta!(prealloc,cache,k,nstage,entropyproj_limiter_type::ZhangShuFilter,param,discrete_data_gauss)
    f(θ) = update_and_check_bound_limited_entropyproj_var_on_element!(prealloc,cache,θ,k,param,discrete_data_gauss)
    prealloc.Farr[k,nstage] = bisection(f,0.0,1.0)
end

function solve_theta!(prealloc,cache,k,nstage,entropyproj_limiter_type::ElementwiseScaledExtrapolation,param,discrete_data_gauss)
    f(θ) = update_and_check_bound_limited_entropyproj_var_on_element!(prealloc,cache,θ,k,param,discrete_data_gauss)
    prealloc.Farr[k,nstage] = bisection(f,0.0,1.0)
end

function solve_theta!(prealloc,cache,k,nstage,entropyproj_limiter_type::NodewiseScaledExtrapolation,param,discrete_data_gauss)
    @unpack vq_k = cache

    calculate_entropy_var!(vq_k,view(prealloc.Uq,:,k),param,discrete_data_gauss)    # TODO: calculation of vq seems duplicate with entropy projection step
    for i = 1:discrete_data_gauss.sizes.Nfp
        f(θ_i) = update_and_check_bound_limited_entropyproj_var_on_face_node!(prealloc,cache,θ_i,i,k,param,discrete_data_gauss)
        prealloc.θ_local_arr[i,k,nstage] = bisection(f,0.0,1.0)
    end
    # TODO: hardcode for post postprocessing
    prealloc.Farr[k,nstage] = sum(view(prealloc.θ_local_arr,:,k,nstage))/discrete_data_gauss.sizes.Nfp
end

function solve_theta!(prealloc,cache,k,entropyproj_limiter_type::NoEntropyProjectionLimiter,param,discrete_data_gauss)
    return 0.0
end

function update_and_check_bound_limited_entropyproj_var_on_element!(prealloc,cache,θ,k,param,discrete_data_gauss)
    try
        update_limited_entropyproj_vars_on_element!(prealloc,cache,θ,k,param.entropyproj_limiter_type,param,discrete_data_gauss)
        return check_bound_on_element(k,cache,param,discrete_data_gauss.sizes)
    catch err
        if isa(err, DomainError)
            return false
        else
            throw(err)
        end
    end
    return false
end

function update_limited_entropyproj_vars_on_element!(prealloc,cache,θ,k,entropyproj_limiter_type::AdaptiveFilter,param,discrete_data_gauss)
    @unpack VqVDM                                     = discrete_data_gauss.ops
    @unpack U_modal,U_k,Uq_k,vq_k,v_tilde_k,u_tilde_k = cache
    
    U_k .= @views U_modal[:,k]
    apply_filter!(U_k,param.entropyproj_limiter_type,param.equation,θ)
    mul!(Uq_k,VqVDM,U_k)
    
    # TODO: only project to Gauss element
    entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,1.0,param,discrete_data_gauss,prealloc)
    calculate_limited_entropyproj_vars_on_element!(cache,param)
end

function update_limited_entropyproj_vars_on_element!(prealloc,cache,θ,k,entropyproj_limiter_type::ScaledExtrapolation,param,discrete_data_gauss)
    @unpack VqVDM = discrete_data_gauss.ops
    @unpack Uq    = prealloc
    @unpack v3tilde,rhotilde,rhoetilde,v_tilde_k,u_tilde_k,vq_k,U_k,Uq_k = cache
    
    entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,view(Uq,:,k),θ,param,discrete_data_gauss,prealloc)
    calculate_limited_entropyproj_vars_on_element!(cache,param)
end

function update_limited_entropyproj_vars!(prealloc,cache,θ,k,entropyproj_limiter_type::NoEntropyProjectionLimiter,param,discrete_data_gauss)
    # Do nothing
end

# TODO: hardcoded for 1D
function calculate_limited_entropyproj_vars_on_element!(cache,param)
    for i = 1:size(cache.v3tilde,1)
        calculate_limited_entropyproj_vars_on_node!(cache,i,param)
    end
end

# TODO: better to define custom index type for hybridized, quad, and surface nodes...
function calculate_limited_entropyproj_vars_on_node!(cache,i,param)
    @unpack v3tilde,rhotilde,rhoetilde,v_tilde_k,u_tilde_k = cache
    v3tilde[i]   = v_tilde_k[i][end]
    rhotilde[i]  = u_tilde_k[i][1]
    rhoetilde[i] = rhoe_ufun(param.equation,u_tilde_k[i])
end

function calculate_limited_entropyproj_vars_on_face_node!(cache,i,param)
    Nq = size(cache.Uq_k,1)
    calculate_limited_entropyproj_vars_on_node!(cache,i+Nq,param)
end

function check_bound_on_element(k,cache,param,sizes) 
    @unpack v3tilde,rhotilde,rhoetilde = cache

    # TODO: only works for Gauss now
    for i = 1:sizes.Nfp
        if !check_bound_on_face_node(i,k,cache,param,sizes)
            return false
        end
    end
    return true
end

# TODO: skip volume quadrature points
function check_bound_on_face_node(i,k,cache,param,sizes) 
    @unpack v3tilde,rhotilde,rhoetilde,VUf,rhoef,Uf = cache
    ϵ = param.global_constants.POSTOL
    ζ = param.limiting_param.ζ
    η = param.limiting_param.η
    Nq = sizes.Nq
    return (v3tilde[Nq+i] < min(ζ*VUf[i,k][end],-ϵ)
         && rhotilde[Nq+i] > max((1-η)*Uf[i,k][1], ϵ)
         && rhotilde[Nq+i] < (1+η)*Uf[i,k][1]
         && rhoetilde[Nq+i] > max((1-η)*rhoef[i,k], ϵ)
         && rhoetilde[Nq+i] < (1+η)*rhoef[i,k])
end

# TODO: Refactor. element versus. node - use multiple dispatch
function update_and_check_bound_limited_entropyproj_var_on_face_node!(prealloc,cache,θ_i,i,k,param,discrete_data_gauss)
    try
        update_limited_entropyproj_vars_on_face_node!(prealloc,cache,θ_i,i,k,param.entropyproj_limiter_type,param,discrete_data_gauss)
        return check_bound_on_face_node(i,k,cache,param,discrete_data_gauss.sizes)
    catch err
        if isa(err, DomainError)
            return false
        else
            throw(err)
        end
    end
    return false
end

function update_limited_entropyproj_vars_on_face_node!(prealloc,cache,θ_i,i,k,entropyproj_limiter_type::NodewiseScaledExtrapolation,param,discrete_data_gauss)
    @unpack Uq = prealloc
    @unpack v3tilde,rhotilde,rhoetilde,v_tilde_k,u_tilde_k,vq_k,U_k,Uq_k = cache
    
    entropy_projection_face_node!(v_tilde_k,u_tilde_k,vq_k,i,θ_i,param,discrete_data_gauss,prealloc)
    calculate_limited_entropyproj_vars_on_face_node!(cache,i,param)
end

#########################
### Filtering methods ###
#########################
function compute_modal_coefficients!(prealloc,param,discrete_data_gauss,cache)
    @unpack Uq       = prealloc
    @unpack U_modal  = cache
    @unpack VDMinvPq = discrete_data_gauss.ops
    
    K  = get_num_elements(param)
    for k = 1:K
        @views mul!(U_modal[:,k],VDMinvPq,Uq[:,k])      # TODO: why there is allocation when remove k = 1:K loop?
    end
end

function apply_entropyproj_filtering!(prealloc,param,entropyproj_limiter_type::AdaptiveFilter,discrete_data_gauss,nstage)
    @unpack Uq,Farr,U_modal,LGLind = prealloc
    @unpack VqVDM                  = discrete_data_gauss.ops
    
    K  = get_num_elements(param)
    for k = 1:K
        if (!LGLind[k])
            U_modal_k = @views U_modal[:,k]
            apply_filter!(U_modal_k,param.entropyproj_limiter_type,param.equation,Farr[k,nstage])
        end
    end
    for k = 1:K
        @views mul!(Uq[:,k],VqVDM,U_modal[:,k])      # TODO: why there is allocation when remove k = 1:K loop?
    end
end

function apply_entropyproj_filtering!(prealloc,param,entropyproj_limiter_type::ScaledExtrapolation,discrete_data_gauss,nstage)
    # Do nothing
end

function apply_filter!(U,entropyproj_limiter_type::ExponentialFilter,equation::EquationType{Dim1},θ)
    for i = 1:size(U,1)
        U[i] = exp(-θ*(i-1)*(i-1))*U[i]
    end
end

function apply_filter!(U,entropyproj_limiter_type::ZhangShuFilter,equation::EquationType{Dim1},θ)
    for i = 2:size(U,1)
        U[i] = θ*U[i]
    end
end