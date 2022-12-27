function primitive_to_conservative(equation::CompressibleIdealGas{Dim1},U)
    γ = get_γ(equation)
    rho,u,p = U
    rhou = rho*u 
    E = p/(γ-1) + .5*rho*u^2
    return SVector{3,Float64}(rho,rhou,E)
end

@inline function pfun(equation::CompressibleIdealGas{Dim1},U)
    γ = get_γ(equation)
    rho,rhou,E = U
    return (γ-1.0)*(E-.5*rhou^2/rho)
end

@inline function E_ufun(equation::CompressibleIdealGas{Dim1},U)
    γ = get_γ(equation)
    rho,u,p = U
    return p/(γ-1) + .5*rho*(u^2)
end

@inline function betafun(equation::CompressibleIdealGas{Dim1},U)
    p = pfun(equation,U)
    return U[1]/(2*p)
end

@inline function wavespeed_davis_estimate(equation::CompressibleIdealGas{Dim1},U)
    γ = get_γ(equation)
    p = pfun(equation,U)
    return abs(U[2]/U[1]) + sqrt(γ*p/U[1])
end

@inline function sfun(equation::CompressibleIdealGas{Dim1},U)
    γ = get_γ(equation)
    p = pfun(equation,U)
    return log(p/(U[1]^γ))
end

@inline function rhoe_ufun(equation::CompressibleIdealGas{Dim1},U)
    rho,rhou,E = U
    return E-.5*rhou*rhou/rho
end

@inline function s_vfun(equation::CompressibleIdealGas{Dim1},V)
    γ = get_γ(equation)
    v1,vu,vE = V
    return γ-v1+vu^2/(2*vE)
end

@inline function rhoe_vfun(equation::CompressibleIdealGas{Dim1},V)
    γ = get_γ(equation)
    s = s_vfun(equation,V)
    v1,vu,vE = V
    return ((γ-1)/((-vE)^γ))^(1/(γ-1)) * exp(-s/(γ-1))
end

@inline function v3_ufun(equation::CompressibleIdealGas{Dim1},U)
    γ = get_γ(equation)
    p = pfun(equation,U)
    rho,rhou,E = U
    return -rho*(γ-1)/p
end

@inline function v_ufun(equation,U)
    γ = get_γ(equation)
    s = sfun(equation,U)
    p = pfun(equation,U)
    rho,rhou,E = U
    v1 = (γ+1-s)-(γ-1)*E/p
    vu = rhou*(γ-1)/p
    vE = -rho*(γ-1)/p
    return SVector{3,Float64}(v1,vu,vE)
end

@inline function u_vfun(equation::CompressibleIdealGas{Dim1},V)
    v1,vu,vE = V
    rhoeV = rhoe_vfun(equation,V)
    rho   = -rhoeV*vE
    rhou  = rhoeV*vu
    E     = rhoeV*(1-vu^2/(2*vE))
    return SVector{3,Float64}(rho,rhou,E)
end

@inline function euler_fluxes(equation::CompressibleIdealGas{Dim1},U)
    rho,rhou,E = U
    p  = pfun(equation,U)
    u  = rhou/rho
    f1 = rhou
    f2 = rhou*u+p
    f3 = u*(E+p)
    return SVector{3,Float64}(f1,f2,f3);
end

@inline function fS_prim_log(equation::CompressibleIdealGas{Dim1},UL,UR)
    γ = get_γ(equation)

    rhoL,uL,betaL,rhologL,betalogL = UL
    rhoR,uR,betaR,rhologR,betalogR = UR

    rholog  = logmean.(rhoL,rhoR,rhologL,rhologR)
    betalog = logmean.(betaL,betaR,betalogL,betalogR)

    # arithmetic avgs
    rhoavg = .5*(rhoL+rhoR)
    uavg   = .5*(uL+uR)

    unorm = uL*uR
    pa    = rhoavg/(betaL+betaR)
    f4aux = rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm

    FxS1 = rholog*uavg
    FxS2 = FxS1*uavg + pa
    FxS3 = f4aux*uavg

    return SVector(FxS1,FxS2,FxS3)
end

@inline function logmean(aL,aR)
    return logmean(aL,aR,log(aL),log(aR))
end

@inline function logmean(aL,aR,logL,logR)

    # From "Entropy stable num. approx. for the isothermal and polytropic Euler"
    # by Winters et al 2019.
    da = aR-aL
    aavg = .5*(aR+aL)
    f = da/aavg
    v = f^2
    if abs(f)<1e-4
        # numbers assume γ = 1.4 (Winters provides formulas for general γ)
        return aavg*(1 + v*(-.2-v*(.0512 - v*0.026038857142857)))
    else
        return -da/(logL-logR)
    end
end