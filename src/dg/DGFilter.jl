function compute_entropyproj_limiting_param!(param,discrete_data,prealloc,cache,approx_basis_type::LobattoCollocation,nstage)
    clear_entropyproj_limiting_parameter_cache!(prealloc,param.entropyproj_limiter_type,nstage)
end

# TODO: put into entropy projection to avoid an extra projection step
function compute_entropyproj_limiting_param!(param,discrete_data,prealloc,cache,approx_basis_type::GaussCollocation,nstage)
    K  = get_num_elements(param)
    clear_entropyproj_limiting_parameter_cache!(prealloc,param.entropyproj_limiter_type,nstage)
    # TODO: possible redundant calculation, only used for calculation of bounds on the fly
    calc_face_values!(prealloc,cache,param,discrete_data)
    for k = 1:K
        solve_theta!(prealloc,cache,k,nstage,param.entropyproj_limiter_type,param,discrete_data)
    end
end

# TODO: unnecessary?
function clear_entropyproj_limiting_parameter_cache!(prealloc,entropyproj_limiter_type::ElementwiseScaledExtrapolation,nstage)
    view(prealloc.θ_arr,:,nstage) .= 1.0
end

# TODO: unnecessary?
function clear_entropyproj_limiting_parameter_cache!(prealloc,entropyproj_limiter_type::NodewiseScaledExtrapolation,nstage)
    view(prealloc.θ_local_arr,:,:,nstage) .= 1.0
end

function clear_entropyproj_limiting_parameter_cache!(prealloc,entropyproj_limiter_type::NoEntropyProjectionLimiter,nstage)
    # Do nothing
end

function calc_face_values!(prealloc,cache,param,discrete_data)
    @unpack Uq,vq        = prealloc
    @unpack Uf,VUf,rhoef = cache
    @unpack Vf           = discrete_data.ops
    
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

function solve_theta!(prealloc,cache,k,nstage,entropyproj_limiter_type::ElementwiseScaledExtrapolation,param,discrete_data)
    f(θ) = update_and_check_bound_limited_entropyproj_var_on_element!(prealloc,cache,θ,k,param,discrete_data)
    prealloc.θ_arr[k,nstage] = bisection(f,0.0,1.0)
end

function solve_theta!(prealloc,cache,k,nstage,entropyproj_limiter_type::NodewiseScaledExtrapolation,param,discrete_data)
    @unpack vq_k = cache

    calculate_entropy_var!(vq_k,view(prealloc.Uq,:,k),param,discrete_data)    # TODO: calculation of vq seems duplicate with entropy projection step
    for i = 1:discrete_data.sizes.Nfp
        f(θ_i) = update_and_check_bound_limited_entropyproj_var_on_face_node!(prealloc,cache,θ_i,i,k,param,discrete_data)
        prealloc.θ_local_arr[i,k,nstage] = bisection(f,0.0,1.0)
    end
    # TODO: hardcode for post postprocessing
    prealloc.θ_arr[k,nstage] = sum(view(prealloc.θ_local_arr,:,k,nstage))/discrete_data.sizes.Nfp
end

function solve_theta!(prealloc,cache,k,entropyproj_limiter_type::NoEntropyProjectionLimiter,param,discrete_data)
    return 0.0
end

function update_and_check_bound_limited_entropyproj_var_on_element!(prealloc,cache,θ,k,param,discrete_data)
    try
        update_limited_entropyproj_vars_on_element!(prealloc,cache,θ,k,param.entropyproj_limiter_type,param,discrete_data)
        return check_bound_on_element(k,cache,param,discrete_data.sizes)
    catch err
        if isa(err, DomainError)
            return false
        else
            throw(err)
        end
    end
    return false
end

function update_limited_entropyproj_vars_on_element!(prealloc,cache,θ,k,entropyproj_limiter_type::ScaledExtrapolation,param,discrete_data)
    @unpack VqVDM = discrete_data.ops
    @unpack Uq    = prealloc
    @unpack v3tilde,rhotilde,rhoetilde,v_tilde_k,u_tilde_k,vq_k,U_k,Uq_k = cache
    
    entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,view(Uq,:,k),θ,param,discrete_data,prealloc)
    calculate_limited_entropyproj_vars_on_element!(cache,param)
end

function update_limited_entropyproj_vars!(prealloc,cache,θ,k,entropyproj_limiter_type::NoEntropyProjectionLimiter,param,discrete_data)
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
function update_and_check_bound_limited_entropyproj_var_on_face_node!(prealloc,cache,θ_i,i,k,param,discrete_data)
    try
        update_limited_entropyproj_vars_on_face_node!(prealloc,cache,θ_i,i,k,param.entropyproj_limiter_type,param,discrete_data)
        return check_bound_on_face_node(i,k,cache,param,discrete_data.sizes)
    catch err
        if isa(err, DomainError)
            return false
        else
            throw(err)
        end
    end
    return false
end

function update_limited_entropyproj_vars_on_face_node!(prealloc,cache,θ_i,i,k,entropyproj_limiter_type::NodewiseScaledExtrapolation,param,discrete_data)
    @unpack Uq = prealloc
    @unpack v3tilde,rhotilde,rhoetilde,v_tilde_k,u_tilde_k,vq_k,U_k,Uq_k = cache
    
    entropy_projection_face_node!(v_tilde_k,u_tilde_k,vq_k,i,θ_i,param,discrete_data,prealloc)
    calculate_limited_entropyproj_vars_on_face_node!(cache,i,param)
end
