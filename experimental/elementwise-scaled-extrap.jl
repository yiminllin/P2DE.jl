struct ElementwiseScaledExtrapolation <: ScaledExtrapolation end

function Base.show(io::IO,entropyproj_limiter_type::ElementwiseScaledExtrapolation)
    text = print(io,"ElemOpBlend")
end

# TODO: ugly dispatch
function entropy_projection!(prealloc,param,entropyproj_limiter_type::ElementwiseScaledExtrapolation,discrete_data,nstage,timer)
    @unpack Uq,vq,v_tilde,u_tilde = prealloc
    @unpack Nh,Nq,Nfp             = discrete_data.sizes
    K = get_num_elements(param)
    
    @batch for k = 1:K
        vq_k      = view(vq,:,k)
        v_tilde_k = view(v_tilde,:,k)
        u_tilde_k = view(u_tilde,:,k)
        Uq_k      = view(Uq,:,k)
        l_k       = prealloc.θ_arr[k,nstage]
        # TODO: we can skip LGL instead of applying identity
        entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data,prealloc)
    end
end

# TODO: unnecessary?
function clear_entropyproj_limiting_parameter_cache!(prealloc,entropyproj_limiter_type::ElementwiseScaledExtrapolation,nstage)
    view(prealloc.θ_arr,:,nstage) .= 1.0
end

function solve_theta!(prealloc,cache,k,nstage,entropyproj_limiter_type::ElementwiseScaledExtrapolation,equation::CompressibleIdealGas,param,discrete_data,tid)
    f(θ) = update_and_check_bound_limited_entropyproj_var_on_element!(prealloc,cache,θ,k,param,discrete_data,tid)
    prealloc.θ_arr[k,nstage] = bisection(f,0.0,1.0)
end

function update_limited_extrapolation!(cache,prealloc,param,entropyproj_limiter_type::ElementwiseScaledExtrapolation,discrete_data,k,nstage,tid)
    @unpack Vf_new    = cache
    @unpack Vf,Vf_low = discrete_data.ops

    l_k = prealloc.θ_arr[k,nstage]
    @. @views Vf_new[:,:,tid] = l_k*Vf+(1.0-l_k)*Vf_low
end

function is_Vf_limited(prealloc,k,nstage,entropyproj_limiter_type::ElementwiseScaledExtrapolation)
    return prealloc.θ_arr[k,nstage] < 1.0
end
