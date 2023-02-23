include("./DGRHSUtils.jl")
include("./EntropyStableRHS.jl")
include("./LowOrderPositivityRHS.jl")

function rhs!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    init_get_rhs!(param,param.entropyproj_limiter_type,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    dt = get_rhs!(param.rhs_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    return dt
end

function init_get_rhs!(param,entropyproj_limiter_type::NoEntropyProjectionLimiter,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    if (nstage == 1)
        update_indicator!(prealloc,param.approximation_basis_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops)
    end
end

function init_get_rhs!(param,entropyproj_limiter_type::AdaptiveFilter,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    compute_modal_coefficients!(prealloc,param,discrete_data_gauss)
    compute_entropyproj_limiting_param!(param,discrete_data_gauss,prealloc,nstage)
    apply_entropyproj_filtering!(prealloc,param,param.entropyproj_limiter_type,discrete_data_gauss,nstage)
    if (nstage == 1)
        update_indicator!(prealloc,param.approximation_basis_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops)
    end
end

function init_get_rhs!(param,entropyproj_limiter_type::ScaledExtrapolation,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    compute_entropyproj_limiting_param!(param,discrete_data_gauss,prealloc,nstage)
    if (nstage == 1)
        update_indicator!(prealloc,param.approximation_basis_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops)
    end
end

function get_rhs!(rhs_type::LowOrderPositivity,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    @unpack rhsL,rhsU = prealloc
    dt = rhs_pos_Gauss!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,t,dt,nstage,true)
    copyto!(rhsU,rhsL)
    return dt
end

function get_rhs!(rhs_type::EntropyStable,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    @unpack rhsH,rhsU = prealloc
    rhs_modalESDG!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,nstage,true)
    copyto!(rhsU,rhsH)
    return dt
end

function get_rhs!(rhs_type::ESLimitedLowOrderPos,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    @unpack rhsH,rhsL,rhsU = prealloc
    entropy_projection!(prealloc,param,param.entropyproj_limiter_type,discrete_data_gauss,discrete_data_LGL,nstage)
    dt = rhs_pos_Gauss!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,t,dt,nstage,false)
    rhs_modalESDG!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,nstage,false)
    apply_positivity_limiter!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,dt,nstage,param.positivity_limiter_type)
    return dt
end

# TODO: dispatch on element type instead of the passed in discrete data
# TODO: refactor with NodewiseScaledExtrapolation
function entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data,prealloc)
    @unpack VhPq_new,Vf_new   = prealloc
    @unpack Nh,Nq,Nfp         = discrete_data.sizes
    @unpack Vf,Vf_low,Pq,VhPq = discrete_data.ops

    calculate_entropy_var!(vq_k,Uq_k,param,discrete_data)    # TODO: move calculating entropy var out of entropy projection
    # For nodal collocation, quad values are the same
    for i = 1:Nq
        entropy_projection_volume_node!(v_tilde_k,u_tilde_k,vq_k,Uq_k,i,param,discrete_data,prealloc)
    end
    for i = 1:Nfp
        entropy_projection_face_node!(v_tilde_k,u_tilde_k,vq_k,i,l_k,param,discrete_data,prealloc)
    end
end

function calculate_entropy_var!(vq_k,Uq_k,param,discrete_data)
    @unpack Nq = discrete_data.sizes
    for i = 1:Nq
        vq_k[i] = v_ufun(param.equation,Uq_k[i])
    end
end

function entropy_projection_volume_node!(v_tilde_k,u_tilde_k,vq_k,Uq_k,i,param,discrete_data,prealloc)
    v_tilde_k[i] = vq_k[i]
    u_tilde_k[i] = Uq_k[i]
end

function entropy_projection_face_node!(v_tilde_k,u_tilde_k,vq_k,i,l_k_i,param,discrete_data,prealloc)
    @unpack Nh,Nq     = discrete_data.sizes
    @unpack Vf,Vf_low = discrete_data.ops
    @unpack Vf_new    = prealloc
    if (l_k_i != 1.0)   # TODO: require l_k ∈ [0,1]
        @views Vf_new[i,:] = l_k_i*Vf[i,:]+(1-l_k_i)*Vf_low[i,:]   # TODO: Vf_new only allocate 1D vector instead of 2D matrix?
        # TODO: v_tilde_k[i+Nq] = @views sum(Vf_new[i,:].*vq_k)
        #       requires allocation... why?
        v_tilde_k[i+Nq] = zero(v_tilde_k[i+Nq])
        for j = 1:Nq
            v_tilde_k[i+Nq] += Vf_new[i,j]*vq_k[j]
        end
    else
        # Nothing is applied if l_k == 1
        # TODO: v_tilde_k[i+Nq] = @views sum(Vf[i,:].*vq_k)
        #       requires allocation... why?
        v_tilde_k[i+Nq] = zero(v_tilde_k[i+Nq])
        for j = 1:Nq
            v_tilde_k[i+Nq] += Vf[i,j]*vq_k[j]
        end
    end
    u_tilde_k[i+Nq] = u_vfun(param.equation,v_tilde_k[i+Nq])
end

# TODO: ugly dispatch
function entropy_projection!(prealloc,param,entropyproj_limiter_type::Union{AdaptiveFilter,NoEntropyProjectionLimiter},discrete_data_gauss,discrete_data_LGL,nstage)
    @unpack Uq,vq,v_tilde,u_tilde,LGLind,L_Vf_arr,Vf_new,VhPq_new = prealloc
    @unpack Nh,Nq,Nfp                                             = discrete_data_gauss.sizes
    K = get_num_elements(param)
    
    for k = 1:K
        vq_k      = view(vq,:,k)
        v_tilde_k = view(v_tilde,:,k)
        u_tilde_k = view(u_tilde,:,k)
        Uq_k      = view(Uq,:,k)
        l_k       = 1.0
        # TODO: we can skip LGL instead of applying identity
        discrete_data = (LGLind[k]) ? discrete_data_LGL : discrete_data_gauss
        entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data,prealloc)
    end
end

# TODO: ugly dispatch
function entropy_projection!(prealloc,param,entropyproj_limiter_type::ElementwiseScaledExtrapolation,discrete_data_gauss,discrete_data_LGL,nstage)
    @unpack Uq,vq,v_tilde,u_tilde,LGLind,L_Vf_arr,Vf_new,VhPq_new = prealloc
    @unpack Nh,Nq,Nfp                                             = discrete_data_gauss.sizes
    K = get_num_elements(param)
    
    for k = 1:K
        vq_k      = view(vq,:,k)
        v_tilde_k = view(v_tilde,:,k)
        u_tilde_k = view(u_tilde,:,k)
        Uq_k      = view(Uq,:,k)
        l_k       = prealloc.Farr[k,nstage]
        # TODO: we can skip LGL instead of applying identity
        discrete_data = (LGLind[k]) ? discrete_data_LGL : discrete_data_gauss
        entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data,prealloc)
    end
end

# TODO: ugly dispatch
function entropy_projection!(prealloc,param,entropyproj_limiter_type::NodewiseScaledExtrapolation,discrete_data_gauss,discrete_data_LGL,nstage)
    @unpack Uq,vq,v_tilde,u_tilde,LGLind,L_Vf_arr,Vf_new,VhPq_new = prealloc
    @unpack Nh,Nq,Nfp                                             = discrete_data_gauss.sizes
    K = get_num_elements(param)
    
    for k = 1:K
        vq_k      = view(vq,:,k)
        v_tilde_k = view(v_tilde,:,k)
        u_tilde_k = view(u_tilde,:,k)
        Uq_k      = view(Uq,:,k)
        # TODO: we can skip LGL instead of applying identity
        if (LGLind[k])
            entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data_LGL,prealloc)
        else
            # TODO: refactor
            calculate_entropy_var!(vq_k,Uq_k,param,discrete_data_gauss)
            for i = 1:Nq
                entropy_projection_volume_node!(v_tilde_k,u_tilde_k,vq_k,Uq_k,i,param,discrete_data_gauss,prealloc)
            end
            for i = 1:Nfp
                l_k_i = prealloc.θ_local_arr[i,k,nstage]
                entropy_projection_face_node!(v_tilde_k,u_tilde_k,vq_k,i,l_k_i,param,discrete_data_gauss,prealloc)
            end
        end
    end
end
