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

@inline function fS(equation::KPP{Dim2},UL,UR)
    uL = UL[1]
    uR = UR[1]

    psixL = -cos(uL)
    psixR = -cos(uR)
    psiyL = sin(uL)
    psiyR = sin(uR)

    # TODO: hardcoded
    if abs(uR-uL) < 1e-12
        return (SVector(.5*(sin(uL)+sin(uR))),SVector(.5*(cos(uL)+cos(uR))))
    else
        return (SVector((psixR-psixL)/(uR-uL)),SVector((psiyR-psiyL)/(uR-uL)))
    end
end
