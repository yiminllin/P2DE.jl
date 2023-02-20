###############################
### Subcell limiter methods ###
###############################
function accumulate_f_bar!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    @unpack rhsL,rhsH,f_bar_H,f_bar_L,flux_H,flux_L = prealloc
    
    K  = get_num_elements(param)
    Nq = size(prealloc.Uq,1)
    # TODO: f_bar_H, f_bar_L could be combine into a single cache?
    for k = 1:K
        wq = prealloc.LGLind[k] ? discrete_data_LGL.ops.wq : discrete_data_gauss.ops.wq
        Jq = prealloc.LGLind[k] ? discrete_data_LGL.geom.Jq : discrete_data_gauss.geom.Jq
        f_bar_H[1,k] = flux_H[1,k][1]
        f_bar_L[1,k] = flux_L[1,k][1]
        for i = 2:Nq+1
            f_bar_H[i,k] = f_bar_H[i-1,k]+Jq[i-1,k]*wq[i-1]*rhsH[i-1,k]
            f_bar_L[i,k] = f_bar_L[i-1,k]+Jq[i-1,k]*wq[i-1]*rhsL[i-1,k]
        end
    end
end

function subcell_bound_limiter!(prealloc,param,discrete_data_gauss,discrete_data_LGL,dt,nstage)
    @unpack Uq,uL_k,L_local_arr,f_bar_H,f_bar_L,rhsL = prealloc
    
    K  = get_num_elements(param)
    Nq = size(Uq,1)
    ζ = param.limiting_param.ζ
    @views @. L_local_arr[:,:,nstage] = 1.0
    # Calculate limiting parameter
    for k = 1:K
        wq = prealloc.LGLind[k] ? discrete_data_LGL.ops.wq : discrete_data_gauss.ops.wq
        Jq = prealloc.LGLind[k] ? discrete_data_LGL.geom.Jq : discrete_data_gauss.geom.Jq
        @views @. uL_k = Uq[:,k] + dt*rhsL[:,k]
        Lrho(uL_i)  = ζ*uL_i[1]
        Lrhoe(uL_i) = ζ*rhoe_ufun(param.equation,uL_i)
        Urho  = Inf
        Urhoe = Inf
        # TODO: ugly...
        for i = 1:Nq
            wJq_i = (wq[i]*Jq[i,k])
            L_local_arr[i,k,nstage] = min(L_local_arr[i,k,nstage], get_limiting_param(param,uL_k[i],-2*dt*(f_bar_H[i,k]-f_bar_L[i,k])/wJq_i,Lrho(uL_k[i]),Lrhoe(uL_k[i]),Urho,Urhoe))
        end
        for i = 2:Nq+1
            wJq_im1 = (wq[i-1]*Jq[i-1,k])
            L_local_arr[i,k,nstage] = min(L_local_arr[i,k,nstage], get_limiting_param(param,uL_k[i-1],2*dt*(f_bar_H[i,k]-f_bar_L[i,k])/wJq_im1,Lrho(uL_k[i-1]),Lrhoe(uL_k[i-1]),Urho,Urhoe))
        end
    end

    # Symmetrize limiting parameter TODO: hardcoded, should use mapP
    for k = 1:K
        l = min(L_local_arr[1,k,nstage], L_local_arr[end,mod1(k-1,K),nstage])
        L_local_arr[1,k,nstage] = l
        L_local_arr[end,mod1(k-1,K),nstage] = l
    end
end

# TODO: not necessary
function accumulate_f_bar_limited!(prealloc,param,nstage)
    @unpack L_local_arr,f_bar_H,f_bar_L,f_bar_lim = prealloc
    
    K  = get_num_elements(param)
    Nq = size(prealloc.Uq,1)
    # TODO: f_bar_H, f_bar_L could be combine into a single cache? df_bar?
    for k = 1:K
        for i = 1:Nq+1
            f_bar_lim[i,k] = L_local_arr[i,k,nstage]*f_bar_H[i,k] + (1-L_local_arr[i,k,nstage])*f_bar_L[i,k]
        end
    end
end

function apply_subcell_limiter!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    @unpack rhsU,f_bar_lim = prealloc
    
    K  = get_num_elements(param)
    Nq = size(prealloc.Uq,1)
    # Update step
    for k = 1:K
        wq = prealloc.LGLind[k] ? discrete_data_LGL.ops.wq : discrete_data_gauss.ops.wq
        Jq = prealloc.LGLind[k] ? discrete_data_LGL.geom.Jq : discrete_data_gauss.geom.Jq
        for i = 1:Nq
            wJq_i     = (wq[i]*Jq[i,k])
            rhsU[i,k] = (f_bar_lim[i+1,k]-f_bar_lim[i,k])/wJq_i
        end
    end
end

