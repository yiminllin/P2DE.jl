@inline function wavespeed_estimate(equation::KPP{Dim2},U,n)
    return 1.0
end

@inline function v_ufun(equation::KPP{Dim2},U)
    return SVector(U)
end

@inline function u_vfun(equation::KPP{Dim2},V)
    return SVector(V)
end

@inline function fluxes(equation::KPP{Dim2},U)
    u = U[1]
    fx1 = sin(u)
    fy1 = cos(u)
    return (SVector(fx1),SVector(fy1))
end
