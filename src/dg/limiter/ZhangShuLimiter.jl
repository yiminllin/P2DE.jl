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
function zhang_shu_bound_limiter!(L,param,uL,P,k,Lrho::Function,Lrhoe::Function,Urho,Urhoe,nstage)
    @unpack rhs_limiter_type = param
    bound_type = get_bound_type(rhs_limiter_type)
    l = 1.0
    for i = 1:size(uL,1)
        uL_i = uL[i]
        bound = (Lrho(uL_i),Lrhoe(uL_i),Urho,Urhoe)
        l = min(l, get_limiting_param(rhs_limiter_type,bound_type,param,uL[i],P[i],bound))
    end
    L[k,nstage] = l
end

function zhang_shu_bound_limiter!(L,param,uL,P,k,Lrho::Real,Lrhoe::Real,Urho,Urhoe,nstage)
    @unpack rhs_limiter_type = param
    bound_type = get_bound_type(rhs_limiter_type)
    l = 1.0
    for i = 1:size(uL,1)
        uL_i = uL[i]
        bound = (Lrho,Lrhoe,Urho,Urhoe)
        l = min(l, get_limiting_param(rhs_limiter_type,bound_type,param,uL[i],P[i],bound))
    end
    L[k,nstage] = l
end