#################################
### Zhang-Shu limiter methods ###
#################################
function apply_zhang_shu_limiter!(state, solver, time_param)
    (; Uq, rhsL, rhsH, rhsU, L) = state.preallocation
    (; uL_k, P_k) = state.cache.limiter_cache
    (; blending_factor) = state.cache.shockcapture_cache
    (; K) = solver.discrete_data.sizes
    (; dt, nstage) = time_param

    zeta = solver.param.limiting_param.zeta
    Lrho(uL_i) = zeta * uL_i[1]
    Lrhoe(uL_i) = zeta * rhoe_ufun(equation(solver), uL_i)
    @batch for k = 1:K
        tid = Threads.threadid()
        @views @. uL_k[:, tid] = Uq[:, k] + dt * rhsL[:, k]
        @views @. P_k[:, tid] = dt * (rhsH[:, k] - rhsL[:, k])
        Urho = Inf
        Urhoe = Inf
        zhang_shu_bound_limiter!(equation(solver), solver, L, view(uL_k, :, tid), view(P_k, :, tid), k, Lrho, Lrhoe, Urho, Urhoe, nstage)
        l = min(L[k, nstage], blending_factor[k, nstage])
        @views @. rhsU[:, k] = (1 - l) * rhsL[:, k] + l * (rhsH[:, k])
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
function zhang_shu_bound_limiter!(equation::CompressibleIdealGas, solver, L, uL, P, k, Lrho::Function, Lrhoe::Function, Urho, Urhoe, nstage)
    (; Nq) = solver.discrete_data.sizes

    l = 1.0
    for i = 1:Nq
        uL_i = uL[i]
        bound_val = (Lrho(uL_i), Lrhoe(uL_i), Urho, Urhoe)
        l = min(l, limiting_param(limiter(solver), bound(solver), solver, uL[i], P[i], bound_val))
    end
    L[k, nstage] = l
end

function zhang_shu_bound_limiter!(equation::CompressibleIdealGas, solver, L, uL, P, k, Lrho::Real, Lrhoe::Real, Urho, Urhoe, nstage)
    (; Nq) = solver.discrete_data.sizes

    l = 1.0
    for i = 1:Nq
        bound_val = (Lrho, Lrhoe, Urho, Urhoe)
        l = min(l, limiting_param(limiter(solver), bound(solver), solver, uL[i], P[i], bound_val))
    end
    L[k, nstage] = l
end

function zhang_shu_bound_limiter!(equation::KPP, L, P, k, Lrho, Lrhoe, Urho, Urhoe, nstage, solver)
    L[k, nstage] = 1.0
end
