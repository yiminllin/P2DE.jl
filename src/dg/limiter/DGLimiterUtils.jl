# Find l s.t. rho(UL + lP)  ∈ [Lrho, Urho]
#             rhoe(UL + lP) ∈ [Lrhoe, Urhoe]
function get_limiting_param(param,UL,P,Lrho,Lrhoe,Urho,Urhoe)
    l = min(1.0, get_limiting_param_bound_rho_rhoe(param,UL,P,Lrho,Lrhoe,Urho,Urhoe))
    return l
end

function get_limiting_param_bound_rho_rhoe(param,U,P,Lrho,Lrhoe,Urho,Urhoe)
    l = 1.0
    # Limit density, lower bound
    if U[1] + P[1] < Lrho
        l = max((Lrho-U[1])/P[1],0.0)
    end
    # Limit density, upper bound
    if U[1] + P[1] > Urho
        l = min(l, max((Urho-U[1])/P[1],0.0))
    end

    l = min(l,rhoe_quadratic_solve(param,U,P,Lrhoe),
              rhoe_quadratic_solve(param,U,P,Urhoe))
    return l
end

function rhoe_quadratic_solve(param,UL,P,Lrhoe)
    @unpack ZEROTOL = param.global_constants

    dim = get_dim_type(param.equation)
    if Lrhoe == Inf
        return 1.0
    end

    # limiting internal energy (via quadratic function) lower bound
    a,b,c = get_rhoe_quadratic_coefficients(UL,P,Lrhoe,dim)

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

    return l_eps_ij
end

function get_rhoe_quadratic_coefficients(U,P,Lrhoe,dim::Dim1)
    a = P[1]*P[3]-1.0/2.0*P[2]^2
    b = U[3]*P[1]+U[1]*P[3]-U[2]*P[2]-P[1]*Lrhoe
    c = U[3]*U[1]-1.0/2.0*U[2]^2-U[1]*Lrhoe
    return a,b,c
end

function get_rhoe_quadratic_coefficients(U,P,Lrhoe,dim::Dim2)
    a = P[1]*P[4]-1.0/2.0*(P[2]^2+P[3]^2)
    b = U[4]*P[1]+U[1]*P[4]-U[2]*P[2]-U[3]*P[3]-P[1]*Lrhoe
    c = U[4]*U[1]-1.0/2.0*(U[2]^2+U[3]^2)-U[1]*Lrhoe
    return a,b,c
end
