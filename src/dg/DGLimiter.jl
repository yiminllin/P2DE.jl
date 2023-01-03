#################################
### Zhang-Shu limiter methods ###
#################################

# Elementwise zhang-shu limiter
# Given states uL, uH at the element k, and state uL satisfy convex bound
# Find largest l ∈ [0,1] s.t. uL + l(P) satifies convex bound, P = uH-uL
#                          or (l uH + (1-l) uL)
# Save the limiting parameter to L[k,nstage]
# TODO: replace the upper bound on E with the upper bound on rhoe
# TODO: generalize to arbitrary convex bound
# TODO: generalize to any equation
# TODO: pack bounds as a struct
function zhang_shu_bound_limiter!(L,param,uL,P,k,Lrho::Function,Lrhoe::Function,Urho,UE,nstage)
    l = 1.0
    for i = 1:size(uL,1)
        uL_i = uL[i]
        l = min(l, get_limiting_param(param,uL[i],P[i],Lrho(uL_i),Lrhoe(uL_i),Urho,UE))
    end
    L[k,nstage] = l
end

function zhang_shu_bound_limiter!(L,param,uL,P,k,Lrho::Real,Lrhoe::Real,Urho,UE,nstage)
    l = 1.0
    for i = 1:size(uL,1)
        uL_i = uL[i]
        l = min(l, get_limiting_param(param,uL[i],P[i],Lrho,Lrhoe,Urho,UE))
    end
    L[k,nstage] = l
end

# Find l s.t. rho(UL + lP)  ∈ [Lrho, Urho]
#             rhoe(UL + lP) ∈ [Lrhoe, inf)
#             E(UL + lP)    ∈ (-inf, UE]
function get_limiting_param(param,UL,P,Lrho,Lrhoe,Urho,UE)
    l = min(1.0,
            get_limiting_param_lowerbound_rho_rhoe(param,UL,P,Lrho,Lrhoe),
            get_limiting_param_bound_component(param,UL[1],P[1],param.global_constants.POSTOL,Urho),
            get_limiting_param_bound_component(param,UL[3],P[3],param.global_constants.POSTOL,UE))
    return l
end

function get_limiting_param_lowerbound_rho_rhoe(param,U_low,P_ij,Lrho,Lrhoe)
    @unpack ZEROTOL = param.global_constants
    l = 1.0
    # Limit density
    if U_low[1] + P_ij[1] < Lrho
        l = max((Lrho-U_low[1])/P_ij[1],0.0)
    end

    # limiting internal energy (via quadratic function)
    a = P_ij[1]*P_ij[3]-1.0/2.0*P_ij[2]^2
    b = U_low[3]*P_ij[1]+U_low[1]*P_ij[3]-U_low[2]*P_ij[2]-P_ij[1]*Lrhoe
    c = U_low[3]*U_low[1]-1.0/2.0*U_low[2]^2-U_low[1]*Lrhoe

    l_eps_ij = 1.0
    if b^2-4*a*c >= 0
        r1 = (-b+sqrt(b^2-4*a*c))/(2*a)
        r2 = (-b-sqrt(b^2-4*a*c))/(2*a)
        if r1 > ZEROTOL && r2 > ZEROTOL
            l_eps_ij = min(r1,r2)
        elseif r1 > ZEROTOL && r2 < -ZEROTOL
            l_eps_ij = r1
        elseif r2 > ZEROTOL && r1 < -ZEROTOL
            l_eps_ij = r2
        end
    end

    l = min(l,l_eps_ij)
    return l
end

function get_limiting_param_bound_component(param,rhoL,rhoP,Lrho,Urho)
    @unpack ZEROTOL = param.global_constants
    if ((rhoL + rhoP >= Lrho + ZEROTOL) && (rhoL + rhoP <= Urho - ZEROTOL))
        return 1.0
    else
        if ((rhoL < Lrho - ZEROTOL) || (rhoL > Urho + ZEROTOL))
            @show "low order rho not in bound",rhoL,rhoP,Lrho,Urho
            if (abs(rhoL-Lrho) > 1e-8 && abs(rhoL-Urho) > 1E-8)
                @show "Difference large:",abs(rhoL-Lrho),abs(rhoL-Urho)
            end
        end
        if (rhoL + rhoP < Lrho - ZEROTOL)
            return (Lrho-rhoL)/rhoP
        elseif (rhoL + rhoP > Urho + ZEROTOL)
            return (Urho-rhoL)/rhoP
        else
            return 1.0
        end
    end
end

###############################
### Subcell limiter methods ###
###############################
function accumulate_f_bar!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    @unpack rhsL,rhsH,f_bar_H,f_bar_L,flux_H,flux_L = prealloc
    Nq = size(prealloc.Uq,1)
    # TODO: f_bar_H, f_bar_L could be combine into a single cache?
    for k = 1:param.K
        wq = prealloc.LGLind[k] ? discrete_data_LGL.ops.wq : discrete_data_gauss.ops.wq
        Jq = prealloc.LGLind[k] ? discrete_data_LGL.geom.Jq : discrete_data_gauss.geom.Jq
        f_bar_H[1,k] = flux_H[1,k]
        f_bar_L[1,k] = flux_L[1,k]
        for i = 2:Nq+1
            f_bar_H[i,k] = f_bar_H[i-1,k]+Jq[i-1,k]*wq[i-1]*rhsH[i-1,k]
            f_bar_L[i,k] = f_bar_L[i-1,k]+Jq[i-1,k]*wq[i-1]*rhsL[i-1,k]
        end
    end
end

function subcell_bound_limiter!(prealloc,param,discrete_data_gauss,discrete_data_LGL,dt,nstage)
    @unpack Uq,uL_k,L_local_arr,f_bar_H,f_bar_L,rhsL = prealloc
    K = param.K
    Nq = size(Uq,1)
    ζ = param.limiting_param.ζ
    # Calculate limiting parameter
    for k = 1:param.K
        wq = prealloc.LGLind[k] ? discrete_data_LGL.ops.wq : discrete_data_gauss.ops.wq
        Jq = prealloc.LGLind[k] ? discrete_data_LGL.geom.Jq : discrete_data_gauss.geom.Jq
        @views @. uL_k = Uq[:,k] + dt*rhsL[:,k]
        Lrho(uL_i)  = ζ*uL_i[1]
        Lrhoe(uL_i) = ζ*rhoe_ufun(param.equation,uL_i)
        Urho = Inf
        UE   = Inf
        # TODO: ugly...
        L_local_arr[1] = 1.0
        for i = 2:Nq+1
            wJq_im1 = (wq[i-1]*Jq[i-1,k])
            L_local_arr[i,k,nstage] = get_limiting_param(param,uL_k[i-1],2*dt*(f_bar_H[i-1,k]-f_bar_L[i-1,k])/wJq_im1,Lrho(uL_k[i-1]),Lrhoe(uL_k[i-1]),Urho,UE)
        end
        for i = 1:Nq
            wJq_i = (wq[i]*Jq[i,k])
            L_local_arr[i,k,nstage] = min(L_local_arr[i],get_limiting_param(param,uL_k[i],-2*dt*(f_bar_H[i,k]-f_bar_L[i,k])/wJq_i,Lrho(uL_k[i]),Lrhoe(uL_k[i]),Urho,UE))
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
    Nq = size(prealloc.Uq,1)
    # TODO: f_bar_H, f_bar_L could be combine into a single cache? df_bar?
    for k = 1:param.K
        for i = 1:Nq+1
            f_bar_lim[i,k] = L_local_arr[i,k,nstage]*f_bar_H[i,k] + (1-L_local_arr[i,k,nstage])*f_bar_L[i,k]
        end
    end
end

function apply_subcell_limiter!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    @unpack rhsU,f_bar_lim = prealloc
    Nq = size(prealloc.Uq,1)
    # Update step
    for k = 1:param.K
        wq = prealloc.LGLind[k] ? discrete_data_LGL.ops.wq : discrete_data_gauss.ops.wq
        Jq = prealloc.LGLind[k] ? discrete_data_LGL.geom.Jq : discrete_data_gauss.geom.Jq
        for i = 1:Nq
            wJq_i     = (wq[i]*Jq[i,k])
            rhsU[i,k] = (f_bar_lim[i+1,k]-f_bar_lim[i,k])/wJq_i
        end
    end
end

############################################
### Appy positivity limiter to limit RHS ###
############################################
function apply_positivity_limiter!(prealloc,param,discrete_data_gauss,discrete_data_LGL,dt,nstage,positivity_limiter_type::ZhangShuLimiter)
    @unpack rhsL,rhsH,rhsU = prealloc
    @unpack Uq,uL_k,P_k    = prealloc
    ζ = param.limiting_param.ζ
    for k = 1:param.K
        @views @. uL_k = Uq[:,k] + dt*rhsL[:,k]
        @views @. P_k  = dt*(rhsH[:,k]-rhsL[:,k])
        Lrho(uL_i)  = ζ*uL_i[1]
        Lrhoe(uL_i) = ζ*rhoe_ufun(param.equation,uL_i)
        Urho = Inf
        UE   = Inf
        zhang_shu_bound_limiter!(prealloc.Larr,param,uL_k,P_k,k,Lrho,Lrhoe,Urho,UE,nstage)
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
    clear_transfer_cache!(prealloc)
    for k = 1:param.K
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
    Urho,UE = get_upper_bound(Uk,param)

    zhang_shu_bound_limiter!(L,param,uL_k,P_k,k,Lrho,Lrhoe,Urho,UE,1)
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

    Urho = -Inf
    UE   = -Inf
    for i = 1:size(Uk,1)
        Urho = max(Urho,Uk[i][1])
        UE   = max(UE,Uk[i][3])
    end
    Urho = (1+η)*Urho
    UE   = (1+η)*UE

    return Urho,UE
end