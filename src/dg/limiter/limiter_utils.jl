# Find l s.t. rho(UL + lP)  ∈ [Lrho, Urho]
#             rhoe(UL + lP) ∈ [Lrhoe, Urhoe]
# TODO: refactor
function limiting_param(rhs_limiter_type::ZhangShuLimiter, bound_type, solver, UL, P, bound)
    Lrho, Lrhoe, Urho, Urhoe = bound
    l = min(1.0, limiting_param_bound_rho_rhoe(solver, UL, P, Lrho, Lrhoe, Urho, Urhoe))
    return l
end

function limiting_param(rhs_limiter_type::SubcellLimiter, bound_type::Union{PositivityBound,PositivityAndCellEntropyBound,PositivityAndRelaxedCellEntropyBound,TVDBound,TVDAndCellEntropyBound,TVDAndRelaxedCellEntropyBound}, solver, UL, P, bound)
    Lrho, Lrhoe, Urho, Urhoe = bound
    l = limiting_param_bound_rho_rhoe(solver, UL, P, Lrho, Lrhoe, Urho, Urhoe)
    return l
end

# Find l s.t. rho(UL + lP)  ∈ [Lrho, Urho]
#             rhoe(UL + lP) ∈ [Lrhoe, Urhoe]
#             phi(UL + lP)  ∈ [Lphi, inf),    phi = rhoe rho^{-\gamma}, modified specific entropy
function limiting_param(rhs_limiter_type::SubcellLimiter, bound_type::Union{PositivityAndMinEntropyBound,PositivityAndRelaxedMinEntropyBound,TVDAndMinEntropyBound,TVDAndRelaxedMinEntropyBound}, solver, UL, P, bound)
    Lrho, Lrhoe, Lphi, Urho, Urhoe = bound
    lpos = limiting_param_bound_rho_rhoe(solver, UL, P, Lrho, Lrhoe, Urho, Urhoe)
    l = limiting_param_bound_phi(solver, UL, P, Lphi, lpos)   # TODO: assume for l \in [0,lpos] gives positive quantities
    return l
end

function limiting_param_bound_rho_rhoe(solver, U, P, Lrho, Lrhoe, Urho, Urhoe)
    l = 1.0
    # Limit density, lower bound
    if U[1] + P[1] < Lrho
        l = max((Lrho - U[1]) / P[1], 0.0)
    end
    # Limit density, upper bound
    if U[1] + P[1] > Urho
        l = min(l, max((Urho - U[1]) / P[1], 0.0))
    end

    l = min(l, rhoe_quadratic_solve(solver, U, P, Lrhoe),
        rhoe_quadratic_solve(solver, U, P, Urhoe))
    return l
end

function limiting_param_bound_phi(solver, U, P, Lphi, lpos)
    (; equation) = solver.param
    (; POSTOL) = solver.param.global_constants

    # TODO: refactor
    f(l) = s_modified_ufun(equation, U + l * P) >= Lphi - POSTOL
    l = bisection(f, 0, lpos)
    return l
end

function rhoe_quadratic_solve(solver, UL, P, Lrhoe)
    (; ZEROTOL) = solver.param.global_constants

    if Lrhoe == Inf
        return 1.0
    end

    # limiting internal energy (via quadratic function) lower bound
    a, b, c = rhoe_quadratic_coefficients(dim_type(solver), UL, P, Lrhoe)

    l_eps_ij = 1.0
    if b^2 - 4 * a * c >= 0
        r1 = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a)
        r2 = (-b - sqrt(b^2 - 4 * a * c)) / (2 * a)
        if r1 > ZEROTOL && r2 > ZEROTOL
            l_eps_ij = min(r1, r2)
        elseif r1 > ZEROTOL && r2 < -ZEROTOL
            l_eps_ij = r1
        elseif r2 > ZEROTOL && r1 < -ZEROTOL
            l_eps_ij = r2
        end
    end

    return l_eps_ij
end

function rhoe_quadratic_coefficients(dim::Dim1, U, P, Lrhoe)
    a = P[1] * P[3] - 1.0 / 2.0 * P[2]^2
    b = U[3] * P[1] + U[1] * P[3] - U[2] * P[2] - P[1] * Lrhoe
    c = U[3] * U[1] - 1.0 / 2.0 * U[2]^2 - U[1] * Lrhoe
    return a, b, c
end

function rhoe_quadratic_coefficients(dim::Dim2, U, P, Lrhoe)
    a = P[1] * P[4] - 1.0 / 2.0 * (P[2]^2 + P[3]^2)
    b = U[4] * P[1] + U[1] * P[4] - U[2] * P[2] - U[3] * P[3] - P[1] * Lrhoe
    c = U[4] * U[1] - 1.0 / 2.0 * (U[2]^2 + U[3]^2) - U[1] * Lrhoe
    return a, b, c
end


#############################
### Subcell limiter Utils ###
#############################
# TODO: refactor
function subcell_face_idx_to_quad_face_index_x(si, sj, solver)
    (; N1D) = solver.discrete_data.sizes
    iface = 0
    if (si == 1)
        iface = sj
    elseif (si == N1D + 1)
        iface = sj + N1D
    end

    return iface
end

# TODO: refactor
function subcell_face_idx_to_quad_face_index_y(si, sj, solver)
    (; N1D) = solver.discrete_data.sizes
    iface = 0
    if (sj == 1)
        iface = si + 2 * N1D
    elseif (sj == N1D + 1)
        iface = si + 3 * N1D
    end

    return iface
end

# TODO: refactor
function subcell_index_P_x(si, sj, k, solver, state_param)
    (; mapP) = state_param.bcdata
    (; N1D) = solver.discrete_data.sizes

    N1Dp1 = N1D + 1
    Nfp = 4 * N1D

    # map subcell index to face quad index
    iface = 0
    if (si == 1)
        iface = sj
    elseif (si == N1Dp1)
        iface = sj + N1D
    end

    ifaceP = mapP[iface, k]
    iP = mod1(ifaceP, Nfp)
    kP = div(ifaceP - 1, Nfp) + 1

    # map face quad index to subcell index
    sjP = mod1(iP, N1D)
    if div(iP - 1, N1D) == 0
        siP = 1
    else
        siP = N1Dp1
    end

    return siP, sjP, kP
end

# TODO: refactor
function subcell_index_P_y(si, sj, k, solver, state_param)
    (; mapP) = state_param.bcdata
    (; N1D) = solver.discrete_data.sizes

    N1Dp1 = N1D + 1
    Nfp = 4 * N1D

    iface = 0
    if (sj == 1)
        iface = si + 2 * N1D
    elseif (sj == N1Dp1)
        iface = si + 3 * N1D
    end

    ifaceP = mapP[iface, k]
    iP = mod1(ifaceP, Nfp)
    kP = div(ifaceP - 1, Nfp) + 1

    # map face quad index to subcell index
    siP = mod1(iP, N1D)
    if div(iP - 1, N1D) == 2
        sjP = 1
    else
        sjP = N1Dp1
    end

    return siP, sjP, kP
end

# TODO: refactor
function quad_index_to_quad_index_P(dim::Dim1, idx, k, solver, state_param)
    (; mapP) = state_param.bcdata
    (; Nfp) = solver.discrete_data.sizes
    (; q2fq, fq2q) = solver.discrete_data.ops

    i = idx
    iface = q2fq[i][1]
    iP = mod1(mapP[iface, k], Nfp)
    iP = fq2q[iP]
    kP = div(mapP[iface, k] - 1, Nfp) + 1
    return iP, kP
end

# TODO: hardcoding... Assume q2fq come in with order: first vertical face nodes
#                     then horizontal face nodes
function quad_index_to_quad_index_P(dim::Dim2, idx, k, direction, solver, state_param)
    (; mapP) = state_param.bcdata
    (; N1D, Nfp) = solver.discrete_data.sizes
    (; q2fq, fq2q) = solver.discrete_data.ops

    i = idx
    iface = length(q2fq[i]) == 1 ? q2fq[i][1] : q2fq[i][direction]
    iP = mod1(mapP[iface, k], Nfp)
    iP = fq2q[iP]
    kP = div(mapP[iface, k] - 1, Nfp) + 1
    jP = div(iP - 1, N1D) + 1
    iP = mod1(iP, N1D)
    return iP, jP, kP
end

function low_order_stencil(dim::Dim1, idx, k, solver, state_param)
    (; N1D) = solver.discrete_data.sizes
    i = idx
    sl = i - 1 >= 1 ? (i - 1, k) : quad_index_to_quad_index_P(dim, i, k, solver, state_param)
    sr = i + 1 <= N1D ? (i + 1, k) : quad_index_to_quad_index_P(dim, i, k, solver, state_param)
    return (sl, sr)
end

function low_order_stencil(dim::Dim2, idx, k, solver, state_param)
    (; N1D) = solver.discrete_data.sizes
    i, j = idx
    idxq = i + (j - 1) * N1D
    sl = i - 1 >= 1 ? (i - 1, j, k) : quad_index_to_quad_index_P(dim, idxq, k, 1, solver, state_param)
    sr = i + 1 <= N1D ? (i + 1, j, k) : quad_index_to_quad_index_P(dim, idxq, k, 1, solver, state_param)
    sb = j - 1 >= 1 ? (i, j - 1, k) : quad_index_to_quad_index_P(dim, idxq, k, 2, solver, state_param)
    st = j + 1 <= N1D ? (i, j + 1, k) : quad_index_to_quad_index_P(dim, idxq, k, 2, solver, state_param)
    return (sl, sr, sb, st)
end
