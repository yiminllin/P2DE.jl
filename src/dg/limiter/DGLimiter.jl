include("./DGLimiterUtils.jl")
include("./ZhangShuLimiter.jl")
include("./SubcellLimiter.jl")

############################################
### Appy positivity limiter to limit RHS ###
############################################
function apply_positivity_limiter!(prealloc,param,discrete_data_gauss,discrete_data_LGL,dt,nstage,positivity_limiter_type::ZhangShuLimiter)
    @unpack rhsL,rhsH,rhsU = prealloc
    @unpack Uq,uL_k,P_k    = prealloc
    
    K  = get_num_elements(param)
    ζ = param.limiting_param.ζ
    for k = 1:K
        @views @. uL_k = Uq[:,k] + dt*rhsL[:,k]
        @views @. P_k  = dt*(rhsH[:,k]-rhsL[:,k])
        Lrho(uL_i)  = ζ*uL_i[1]
        Lrhoe(uL_i) = ζ*rhoe_ufun(param.equation,uL_i)
        Urho  = Inf
        Urhoe = Inf
        zhang_shu_bound_limiter!(prealloc.Larr,param,uL_k,P_k,k,Lrho,Lrhoe,Urho,Urhoe,nstage)
        l = prealloc.Larr[k,nstage]
        @views @. rhsU[:,k] = (1-l)*rhsL[:,k] + l*(rhsH[:,k])
    end
end

function apply_positivity_limiter!(prealloc,param,discrete_data_gauss,discrete_data_LGL,dt,nstage,positivity_limiter_type::SubcellLimiter)
    @unpack rhsL,rhsH,rhsU    = prealloc
    @unpack Uq,uL_k,f_bar_lim = prealloc
    @unpack L_local_arr       = prealloc

    accumulate_f_bar!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    subcell_bound_limiter!(prealloc,param,discrete_data_gauss,discrete_data_LGL,dt,nstage)
    accumulate_f_bar_limited!(prealloc,param,nstage)
    apply_subcell_limiter!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
end

####################################################
### Appy Zhang-Shu limiter to LGL-Gauss transfer ###
####################################################
function clear_transfer_cache!(prealloc)
    prealloc.L_G2L_arr .= 1.0
    prealloc.L_L2G_arr .= 1.0
end

function transfer_indicator(prealloc,discrete_data_gauss,k)
    # TODO: a "nearly do-nothing" indicator
    ind = false
    for i = 1:discrete_data_gauss.sizes.Nq
        if (prealloc.Uq[i,k][1] < 1e-2)
            ind = true
            break
        end
    end
    return ind
end

function update_indicator!(prealloc,approximation_basis_type::GaussCollocation,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,firststep=false)
    if (firststep)
        clear_transfer_cache!(prealloc)
        prealloc.LGLind .= false
    end
    # Else, do nothing
end

function update_indicator!(prealloc,approximation_basis_type::LobattoCollocation,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,firststep=false)
    if (firststep)
        clear_transfer_cache!(prealloc)
        prealloc.LGLind .= true
    end
    # Else, do nothing
end

function update_indicator!(prealloc,approximation_basis_type::HybridGaussLGL,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,firststep=false)
    @unpack LGLind,L_G2L_arr,L_L2G_arr = prealloc
    
    K  = get_num_elements(param)
    clear_transfer_cache!(prealloc)
    for k = 1:K
        ind = transfer_indicator(prealloc,discrete_data_gauss,k)
        if (firststep)
            prealloc.LGLind[k] = ind
        else
            # If switch from LGL to Gauss or Gauss to LGL, we need to cast the
            # solution to Gauss/LGL
            if (LGLind[k] != ind)
                LGLind[k] = ind
                if (LGLind[k])
                    # Gauss to LGL
                    apply_transfer_limiter!(prealloc,param,discrete_data_gauss,transfer_ops.T_g2l,L_G2L_arr,k)
                else
                    # LGL to Gauss
                    apply_transfer_limiter!(prealloc,param,discrete_data_LGL,transfer_ops.T_l2g,L_L2G_arr,k)
                end
            end
        end
    end
end

function apply_transfer_limiter!(prealloc,param,discrete_data,T,L,k)
    @unpack uL_k,P_k = prealloc

    Uk    = view(prealloc.Uq,:,k)
    avg_k = get_average(Uk,discrete_data.ops.wq)
    for i = 1:size(uL_k,1)
        uL_k[i] = avg_k
    end
    mul!(P_k,T,Uk)
    @. P_k = P_k-uL_k

    Lrho  = param.global_constants.POSTOL
    Lrhoe = param.global_constants.POSTOL
    Urho,Urhoe = get_upper_bound(Uk,param)

    zhang_shu_bound_limiter!(L,param,uL_k,P_k,k,Lrho,Lrhoe,Urho,Urhoe,1)
    for i = 1:size(Uk,1)
        Uk[i] = uL_k[i] + L[k]*P_k[i]
    end
end

function get_average(Uk,wq)
    avg_k  = zero(SVector{3,Float64})
    for i = 1:size(Uk,1)
        avg_k = avg_k + wq[i]*Uk[i]
    end
    avg_k = avg_k/2         # TODO: hardcoded reference element size
    return avg_k
end

function get_upper_bound(Uk,param)
    @unpack η = param.limiting_param

    Urho  = -Inf
    Urhoe = -Inf
    for i = 1:size(Uk,1)
        Urho  = max(Urho,Uk[i][1])
        Urhoe = max(Urhoe,rhoe_ufun(param.equation,Uk[i]))
    end
    Urho  = (1+η)*Urho
    Urhoe = (1+η)*Urhoe

    return Urho,Urhoe
end