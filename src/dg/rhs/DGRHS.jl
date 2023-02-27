include("./DGRHSUtils.jl")
include("./EntropyStableRHS.jl")
include("./LowOrderPositivityRHS.jl")

function rhs!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,caches,t,dt,nstage,timer)
    @timeit timer "initialize rhs" begin
    init_get_rhs!(param,param.entropyproj_limiter_type,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,caches,t,dt,nstage,timer)
    end
    @timeit timer "rhs calculation" begin
    dt = get_rhs!(param.rhs_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,caches,t,dt,nstage,timer)
    end
    return dt
end

function init_get_rhs!(param,entropyproj_limiter_type::NoEntropyProjectionLimiter,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,caches,t,dt,nstage,timer)
    @timeit timer "update indicator" begin
    if (nstage == 1)
        update_indicator!(prealloc,param.approximation_basis_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops)
    end
end
end

function init_get_rhs!(param,entropyproj_limiter_type::AdaptiveFilter,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,caches,t,dt,nstage,timer)
    @unpack entropyproj_limiter_cache = caches

    @timeit timer "compute modal coefficients" begin
    compute_modal_coefficients!(prealloc,param,discrete_data_gauss,entropyproj_limiter_cache)
    end
    @timeit timer "compute entropy projection limiting parameters" begin
    compute_entropyproj_limiting_param!(param,discrete_data_gauss,prealloc,entropyproj_limiter_cache,nstage)
    end
    @timeit timer "apply entropy projection limiting" begin
    apply_entropyproj_filtering!(prealloc,param,param.entropyproj_limiter_type,discrete_data_gauss,nstage)
    end
    @timeit timer "update indicator" begin
    if (nstage == 1)
        update_indicator!(prealloc,param.approximation_basis_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops)
    end
    end
end

function init_get_rhs!(param,entropyproj_limiter_type::ScaledExtrapolation,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,caches,t,dt,nstage,timer)
    @unpack entropyproj_limiter_cache = caches

    @timeit timer "compute entropy projection limiting parameters" begin
    compute_entropyproj_limiting_param!(param,discrete_data_gauss,prealloc,entropyproj_limiter_cache,nstage)
    end
    @timeit timer "update indicator" begin
    if (nstage == 1)
        update_indicator!(prealloc,param.approximation_basis_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops)
    end
    end
end

function get_rhs!(rhs_type::LowOrderPositivity,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,caches,t,dt,nstage,timer)
    @unpack rhsL,rhsU = prealloc
    @unpack rhs_cache = caches

    @timeit timer "low order positivity" begin
    dt = rhs_pos_Gauss!(prealloc,rhs_cache,param,discrete_data_gauss,discrete_data_LGL,bcdata,t,dt,nstage,timer,true)
    end
    copyto!(rhsU,rhsL)
    return dt
end

function get_rhs!(rhs_type::EntropyStable,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,caches,t,dt,nstage,timer)
    @unpack rhsH,rhsU = prealloc
    @unpack rhs_cache = caches

    @timeit timer "high order ESDG" begin
    rhs_modalESDG!(prealloc,rhs_cache,param,discrete_data_gauss,discrete_data_LGL,bcdata,nstage,timer,true)
    end
    copyto!(rhsU,rhsH)
    return dt
end

function get_rhs!(rhs_type::ESLimitedLowOrderPos,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,caches,t,dt,nstage,timer)
    @unpack rhs_cache,limiter_cache = caches

    @timeit timer "entropy projection" begin
    entropy_projection!(prealloc,param,param.entropyproj_limiter_type,discrete_data_gauss,discrete_data_LGL,nstage,timer)
    end
    @timeit timer "low order positivity" begin
    dt = rhs_pos_Gauss!(prealloc,rhs_cache,param,discrete_data_gauss,discrete_data_LGL,bcdata,t,dt,nstage,timer,false)
    end
    @timeit timer "high order ESDG" begin
    rhs_modalESDG!(prealloc,rhs_cache,param,discrete_data_gauss,discrete_data_LGL,bcdata,nstage,timer,false)
    end
    @timeit timer "apply positivity limiter" begin
    apply_positivity_limiter!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,limiter_cache,dt,nstage,param.positivity_limiter_type,timer)
    end
    return dt
end

# TODO: dispatch on element type instead of the passed in discrete data
# TODO: refactor with NodewiseScaledExtrapolation
function entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data,prealloc)
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
    if (l_k_i != 1.0)   # TODO: require l_k ∈ [0,1]
        # TODO: v_tilde_k[i+Nq] = @views sum(Vf_new[i,:].*vq_k)
        #       requires allocation... why?
        v_tilde_k[i+Nq] = zero(v_tilde_k[i+Nq])
        for j = 1:Nq
            v_tilde_k[i+Nq] += (l_k_i*Vf[i,j]+(1-l_k_i)*Vf_low[i,j])*vq_k[j]
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
function entropy_projection!(prealloc,param,entropyproj_limiter_type::Union{AdaptiveFilter,NoEntropyProjectionLimiter},discrete_data_gauss,discrete_data_LGL,nstage,timer)
    @unpack Uq,vq,v_tilde,u_tilde,LGLind = prealloc
    @unpack Nh,Nq,Nfp                    = discrete_data_gauss.sizes
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
function entropy_projection!(prealloc,param,entropyproj_limiter_type::ElementwiseScaledExtrapolation,discrete_data_gauss,discrete_data_LGL,nstage,timer)
    @unpack Uq,vq,v_tilde,u_tilde,LGLind = prealloc
    @unpack Nh,Nq,Nfp                    = discrete_data_gauss.sizes
    K = get_num_elements(param)
    
    for k = 1:K
        vq_k      = view(vq,:,k)
        v_tilde_k = view(v_tilde,:,k)
        u_tilde_k = view(u_tilde,:,k)
        Uq_k      = view(Uq,:,k)
        l_k       = prealloc.θ_arr[k,nstage]
        # TODO: we can skip LGL instead of applying identity
        discrete_data = (LGLind[k]) ? discrete_data_LGL : discrete_data_gauss
        entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data,prealloc)
    end
end

# TODO: ugly dispatch
function entropy_projection!(prealloc,param,entropyproj_limiter_type::NodewiseScaledExtrapolation,discrete_data_gauss,discrete_data_LGL,nstage,timer)
    @unpack Uq,vq,v_tilde,u_tilde,LGLind = prealloc
    @unpack Nh,Nq,Nfp                    = discrete_data_gauss.sizes
    K = get_num_elements(param)
    
    for k = 1:K
        vq_k      = view(vq,:,k)
        v_tilde_k = view(v_tilde,:,k)
        u_tilde_k = view(u_tilde,:,k)
        Uq_k      = view(Uq,:,k)
        # TODO: we can skip LGL instead of applying identity
        if (LGLind[k])
            entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,1.0,param,discrete_data_LGL,prealloc)
        else
            # TODO: refactor
            @timeit timer "calculate entropy variables" begin
            calculate_entropy_var!(vq_k,Uq_k,param,discrete_data_gauss)
            end
            @timeit timer "entropy projection volume nodes" begin
            for i = 1:Nq
                entropy_projection_volume_node!(v_tilde_k,u_tilde_k,vq_k,Uq_k,i,param,discrete_data_gauss,prealloc)
            end
            end
            @timeit timer "entropy projection face nodes" begin
            for i = 1:Nfp
                l_k_i = prealloc.θ_local_arr[i,k,nstage]
                entropy_projection_face_node!(v_tilde_k,u_tilde_k,vq_k,i,l_k_i,param,discrete_data_gauss,prealloc)
            end
            end
        end
    end
end
