#################################
### Zhang-Shu limiter methods ###
#################################
function apply_zhang_shu_limiter!(prealloc,limiter_cache,shockcapture_cache,param,dt,nstage)
    @unpack equation          = param
    @unpack Uq,rhsL,rhsH,rhsU = prealloc
    @unpack uL_k,P_k          = limiter_cache
    @unpack blending_factor   = shockcapture_cache

    K  = get_num_elements(param)
    ζ = param.limiting_param.ζ
    Lrho(uL_i)  = ζ*uL_i[1]
    Lrhoe(uL_i) = ζ*rhoe_ufun(param.equation,uL_i)
    @batch for k = 1:K
        tid = Threads.threadid()
        @views @. uL_k[:,tid] = Uq[:,k] + dt*rhsL[:,k]
        @views @. P_k[:,tid]  = dt*(rhsH[:,k]-rhsL[:,k])
        Urho  = Inf
        Urhoe = Inf
        zhang_shu_bound_limiter!(equation,prealloc.Larr,param,view(uL_k,:,tid),view(P_k,:,tid),k,Lrho,Lrhoe,Urho,Urhoe,nstage)
        l = min(prealloc.Larr[k,nstage],blending_factor[k,nstage])
        @views @. rhsU[:,k] = (1-l)*rhsL[:,k] + l*(rhsH[:,k])
    end
end

# Elementwise zhang-shu limiter
# Given states uL, uH at the element k, and state uL satisfy convex bound
# Find largest l ∈ [0,1] s.t. uL + l(P) satifies convex bound, P = uH-uL
#                          or (l uH + (1-l) uL)
# Save the limiting parameter to L[k,nstage]
# TODO: replace the upper bound on E with the upper bound on rhoe
# TODO: generalize to arbitrary convex bound
# TODO: generalize to any equation
# TODO: pack bounds as a struct
function zhang_shu_bound_limiter!(equation::CompressibleIdealGas,L,param,uL,P,k,Lrho::Function,Lrhoe::Function,Urho,Urhoe,nstage)
    @unpack rhs_limiter_type = param
    bound_type = get_bound_type(param)
    l = 1.0
    for i = 1:size(uL,1)
        uL_i = uL[i]
        bound = (Lrho(uL_i),Lrhoe(uL_i),Urho,Urhoe)
        l = min(l, get_limiting_param(rhs_limiter_type,bound_type,param,uL[i],P[i],bound))
    end
    L[k,nstage] = l
end

function zhang_shu_bound_limiter!(equation::CompressibleIdealGas,L,param,uL,P,k,Lrho::Real,Lrhoe::Real,Urho,Urhoe,nstage)
    @unpack rhs_limiter_type = param
    bound_type = get_bound_type(param)
    l = 1.0
    for i = 1:size(uL,1)
        uL_i = uL[i]
        bound = (Lrho,Lrhoe,Urho,Urhoe)
        l = min(l, get_limiting_param(rhs_limiter_type,bound_type,param,uL[i],P[i],bound))
    end
    L[k,nstage] = l
end

function zhang_shu_bound_limiter!(equation::KPP,L,param,uL,P,k,Lrho,Lrhoe,Urho,Urhoe,nstage)
    L[k,nstage] = 1.0
end