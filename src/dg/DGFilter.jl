# TODO: put into entropy projection to avoid an extra projection step
function compute_entropyproj_limiting_param!(param,discrete_data_gauss,prealloc,nstage)
    @unpack Farr,U_modal,LGLind = prealloc

    clear_entropyproj_limiting_parameter_cache!(prealloc,param.entropyproj_limiter_type,nstage)
    for k = 1:param.K
        if (!LGLind[k])
            bound = get_relaxed_bound(param,prealloc,k)
            solve_theta!(prealloc,k,nstage,bound,param.entropyproj_limiter_type,param,discrete_data_gauss)
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

function get_relaxed_bound(param,prealloc,k)
    Uq = prealloc.Uq
    ϵ  = param.global_constants.POSTOL
    η  = param.limiting_param.η
    ζ  = param.limiting_param.ζ

    v3max   = typemin(Float64)
    rhomax  = typemin(Float64)
    rhoemax = typemin(Float64)
    rhomin  = typemax(Float64)
    rhoemin = typemax(Float64)
    for i = 1:size(Uq,1)
        u_i    = Uq[i,k]
        rho_i  = u_i[1]
        rhoe_i = rhoe_ufun(param.equation,u_i)
        v3_i   = v3_ufun(param.equation,u_i)
        rhomax  = max(rhomax,rho_i)
        rhoemax = max(rhoemax,rhoe_i)
        rhomin  = min(rhomin,rho_i)
        rhoemin = min(rhoemin,rhoe_i)
        v3max   = max(v3max,v3_i)
    end 
    bound = (ζ*v3max,max((1-η)*rhomin,ϵ),(1+η)*rhomax,max((1-η)*rhoemin,ϵ),(1+η)*rhoemax)
    return bound
end

function solve_theta!(prealloc,k,nstage,bound,entropyproj_limiter_type::ExponentialFilter,param,discrete_data_gauss)
    f(θ) = update_and_check_bound_limited_entropyproj_var_on_element!(prealloc,θ,k,bound,param,discrete_data_gauss)
    prealloc.Farr[k,nstage] = bisection(f,-log(param.global_constants.ZEROTOL),0.0)
end

function solve_theta!(prealloc,k,nstage,bound,entropyproj_limiter_type::ZhangShuFilter,param,discrete_data_gauss)
    f(θ) = update_and_check_bound_limited_entropyproj_var_on_element!(prealloc,θ,k,bound,param,discrete_data_gauss)
    prealloc.Farr[k,nstage] = bisection(f,0.0,1.0)
end

function solve_theta!(prealloc,k,nstage,bound,entropyproj_limiter_type::ElementwiseScaledExtrapolation,param,discrete_data_gauss)
    f(θ) = update_and_check_bound_limited_entropyproj_var_on_element!(prealloc,θ,k,bound,param,discrete_data_gauss)
    prealloc.Farr[k,nstage] = bisection(f,0.0,1.0)
end

function solve_theta!(prealloc,k,nstage,bound,entropyproj_limiter_type::NodewiseScaledExtrapolation,param,discrete_data_gauss)
    calculate_entropy_var!(prealloc.vq_k,view(prealloc.Uq,:,k),param,discrete_data_gauss)    # TODO: calculation of vq seems duplicate with entropy projection step
    for i = 1:discrete_data_gauss.sizes.Nfp
        f(θ_i) = update_and_check_bound_limited_entropyproj_var_on_face_node!(prealloc,θ_i,i,k,bound,param,discrete_data_gauss)
        prealloc.θ_local_arr[i,k,nstage] = bisection(f,0.0,1.0)
    end
    # TODO: hardcode for post postprocessing
    prealloc.Farr[k,nstage] = sum(view(prealloc.θ_local_arr,:,k,nstage))/discrete_data_gauss.sizes.Nfp
end

function solve_theta!(prealloc,k,bound,entropyproj_limiter_type::NoEntropyProjectionLimiter,param,discrete_data_gauss)
    return 0.0
end

function update_and_check_bound_limited_entropyproj_var_on_element!(prealloc,θ,k,bound,param,discrete_data_gauss)
    try
        update_limited_entropyproj_vars_on_element!(prealloc,θ,k,param.entropyproj_limiter_type,param,discrete_data_gauss)
        if check_bound_on_element(prealloc,bound,param,discrete_data_gauss.sizes)
            return true
        end
    catch err
        if isa(err, DomainError)
            return false
        else
            throw(err)
        end
    end
    return false
end

function update_limited_entropyproj_vars_on_element!(prealloc,θ,k,entropyproj_limiter_type::AdaptiveFilter,param,discrete_data_gauss)
    @unpack VqVDM                    = discrete_data_gauss.ops
    @unpack U_modal,U_k,Uq_k         = prealloc
    @unpack vq_k,v_tilde_k,u_tilde_k = prealloc
    
    U_k .= @views U_modal[:,k]
    apply_filter!(U_k,param.entropyproj_limiter_type,param.equation,θ)
    mul!(Uq_k,VqVDM,U_k)
    
    # TODO: only project to Gauss element
    entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,1.0,param,discrete_data_gauss,prealloc)
    calculate_limited_entropyproj_vars_on_element!(prealloc,param)
end

function update_limited_entropyproj_vars_on_element!(prealloc,θ,k,entropyproj_limiter_type::ScaledExtrapolation,param,discrete_data_gauss)
    @unpack VqVDM = discrete_data_gauss.ops
    @unpack Uq,v3tilde,rhotilde,rhoetilde,v_tilde_k,u_tilde_k,vq_k,U_k,Uq_k = prealloc
    
    entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,view(Uq,:,k),θ,param,discrete_data_gauss,prealloc)
    calculate_limited_entropyproj_vars_on_element!(prealloc,param)
end

function update_limited_entropyproj_vars!(prealloc,θ,k,entropyproj_limiter_type::NoEntropyProjectionLimiter,param,discrete_data_gauss)
    # Do nothing
end

# TODO: hardcoded for 1D
function calculate_limited_entropyproj_vars_on_element!(prealloc,param)
    for i = 1:size(prealloc.v3tilde,1)
        calculate_limited_entropyproj_vars_on_node!(prealloc,i,param)
    end
end

function calculate_limited_entropyproj_vars_on_node!(prealloc,i,param)
    @unpack v3tilde,rhotilde,rhoetilde,v_tilde_k,u_tilde_k = prealloc
    v3tilde[i]   = v_tilde_k[i][end]
    rhotilde[i]  = u_tilde_k[i][1]
    rhoetilde[i] = rhoe_ufun(param.equation,u_tilde_k[i])
end

function check_bound_on_element(prealloc,bound,param,sizes) 
    @unpack v3tilde,rhotilde,rhoetilde = prealloc

    v3max,rhomin,rhomax,rhoemin,rhoemax = bound
    for i = 1:sizes.Nh                 # TODO: change to Nq?
        if !check_bound_on_node(i,prealloc,bound,param,sizes)
            return false
        end
    end
    return true
end

# TODO: skip volume quadrature points
function check_bound_on_node(i,prealloc,bound,param,sizes) 
    @unpack v3tilde,rhotilde,rhoetilde = prealloc
    v3max,rhomin,rhomax,rhoemin,rhoemax = bound
    if ((v3tilde[i] > v3max)
        || (rhotilde[i] < rhomin)
        || (rhotilde[i] > rhomax)
        || (rhoetilde[i] < rhoemin)
        || (rhoetilde[i] > rhoemax))
        return false
    end
    return true
end

# TODO: Refactor. element versus. node - use multiple dispatch
function update_and_check_bound_limited_entropyproj_var_on_face_node!(prealloc,θ_i,i,k,bound,param,discrete_data_gauss)
    try
        update_limited_entropyproj_vars_on_face_node!(prealloc,θ_i,i,k,param.entropyproj_limiter_type,param,discrete_data_gauss)
        if check_bound_on_face_node(i,prealloc,bound,param,discrete_data_gauss.sizes)
            return true
        end
    catch err
        if isa(err, DomainError)
            return false
        else
            throw(err)
        end
    end
    return false
end

function update_limited_entropyproj_vars_on_face_node!(prealloc,θ_i,i,k,entropyproj_limiter_type::NodewiseScaledExtrapolation,param,discrete_data_gauss)
    @unpack Uq,v3tilde,rhotilde,rhoetilde,v_tilde_k,u_tilde_k,vq_k,U_k,Uq_k = prealloc
    
    entropy_projection_face_node!(v_tilde_k,u_tilde_k,vq_k,i,θ_i,param,discrete_data_gauss,prealloc)
    calculate_limited_entropyproj_vars_on_face_node!(prealloc,i,param)
end

function calculate_limited_entropyproj_vars_on_face_node!(prealloc,i,param)
    Nq = size(prealloc.Uq,1)
    calculate_limited_entropyproj_vars_on_node!(prealloc,i+Nq,param)
end

function check_bound_on_face_node(i,prealloc,bound,param,sizes)
    return check_bound_on_node(i+sizes.Nq,prealloc,bound,param,sizes)  
end

#########################
### Filtering methods ###
#########################
function compute_modal_coefficients!(prealloc,param,discrete_data_gauss)
    @unpack Uq,U_modal = prealloc
    @unpack VDMinvPq   = discrete_data_gauss.ops
    for k = 1:param.K
        @views mul!(U_modal[:,k],VDMinvPq,Uq[:,k])      # TODO: why there is allocation when remove k = 1:K loop?
    end
end

function apply_entropyproj_filtering!(prealloc,param,entropyproj_limiter_type::AdaptiveFilter,discrete_data_gauss,nstage)
    @unpack Uq,Farr,U_modal,LGLind = prealloc
    @unpack VqVDM                  = discrete_data_gauss.ops
    for k = 1:param.K
        if (!LGLind[k])
            U_modal_k = @views U_modal[:,k]
            apply_filter!(U_modal_k,param.entropyproj_limiter_type,param.equation,Farr[k,nstage])
        end
    end
    for k = 1:param.K
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