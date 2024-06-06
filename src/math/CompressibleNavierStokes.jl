function primitive_to_conservative(equation::CompressibleIdealGas{Dim1}, U)
    γ = get_γ(equation)
    rho, u, p = U
    rhou = rho * u
    E = p / (γ - 1) + 0.5 * rho * u^2
    return SVector{3,Float64}(rho, rhou, E)
end

function primitive_to_conservative(equation::CompressibleIdealGas{Dim2}, U)
    γ = get_γ(equation)
    rho, u, v, p = U
    rhou = rho * u
    rhov = rho * v
    E = p / (γ - 1) + 0.5 * rho * (u^2 + v^2)
    return SVector{4,Float64}(rho, rhou, rhov, E)
end

@inline function pfun(equation::CompressibleIdealGas{Dim1}, U)
    γ = get_γ(equation)
    rho, rhou, E = U
    return (γ - 1.0) * (E - 0.5 * rhou^2 / rho)
end

@inline function pfun(equation::CompressibleIdealGas{Dim2}, U)
    γ = get_γ(equation)
    rho, rhou, rhov, E = U
    return (γ - 1.0) * (E - 0.5 * (rhou^2 + rhov^2) / rho)
end

@inline function E_ufun(equation::CompressibleIdealGas{Dim1}, U)
    γ = get_γ(equation)
    rho, u, p = U
    return p / (γ - 1) + 0.5 * rho * (u^2)
end

@inline function E_ufun(equation::CompressibleIdealGas{Dim2}, U)
    γ = get_γ(equation)
    rho, u, v, p = U
    return p / (γ - 1) + 0.5 * rho * (u^2 + v^2)
end

@inline function betafun(equation::CompressibleIdealGas{Dim}, U) where {Dim}
    p = pfun(equation, U)
    return U[1] / (2 * p)
end

# TODO: dispatch on wavespeed estimate type
@inline function wavespeed_estimate(equation::CompressibleIdealGas{Dim1}, U)
    γ = get_γ(equation)
    p = pfun(equation, U)
    return abs(U[2] / U[1]) + sqrt(γ * p / U[1])
end

@inline function wavespeed_estimate(equation::CompressibleIdealGas{Dim1}, U, n)
    return wavespeed_estimate(equation, U)
end

# n - normalized normal
@inline function wavespeed_estimate(equation::CompressibleIdealGas{Dim2}, U, n)
    Un = SVector(U[1], n[1] * U[2] + n[2] * U[3], U[4])
    return wavespeed_estimate(to_equation_1D(equation), Un)
end

@inline function sfun(equation::CompressibleIdealGas{Dim}, U) where {Dim}
    γ = get_γ(equation)
    p = pfun(equation, U)
    return log(p / (U[1]^γ))
end

@inline function rhoe_ufun(equation::CompressibleIdealGas{Dim1}, U)
    rho, rhou, E = U
    return E - 0.5 * rhou * rhou / rho
end

@inline function rhoe_ufun(equation::CompressibleIdealGas{Dim2}, U)
    rho, rhou, rhov, E = U
    return E - 0.5 * (rhou * rhou + rhov * rhov) / rho
end

@inline function s_modified_ufun(equation::CompressibleIdealGas, U)
    rho = U[1]
    γ = get_γ(equation)
    rhoe = rhoe_ufun(equation, U)
    return rhoe * rho^(-γ)
end

@inline function s_vfun(equation::CompressibleIdealGas{Dim1}, V)
    γ = get_γ(equation)
    v1, vu, vE = V
    return γ - v1 + vu^2 / (2 * vE)
end

@inline function s_vfun(equation::CompressibleIdealGas{Dim2}, V)
    γ = get_γ(equation)
    v1, vu, vv, vE = V
    return γ - v1 + (vu^2 + vv^2) / (2 * vE)
end

@inline function rhoe_vfun(equation::CompressibleIdealGas{Dim}, V) where {Dim}
    γ = get_γ(equation)
    s = s_vfun(equation, V)
    vE = V[end]
    return ((γ - 1) / ((-vE)^γ))^(1 / (γ - 1)) * exp(-s / (γ - 1))
end

@inline function v3_ufun(equation::CompressibleIdealGas{Dim}, U) where {Dim}
    γ = get_γ(equation)
    p = pfun(equation, U)
    rho = U[1]
    return -rho * (γ - 1) / p
end

@inline function v_ufun(equation::CompressibleIdealGas{Dim1}, U)
    γ = get_γ(equation)
    s = sfun(equation, U)
    p = pfun(equation, U)
    rho, rhou, E = U
    v1 = (γ + 1 - s) - (γ - 1) * E / p
    vu = rhou * (γ - 1) / p
    vE = -rho * (γ - 1) / p
    return SVector{3,Float64}(v1, vu, vE)
end

@inline function psi_ufun(equation::CompressibleIdealGas{Dim1}, U)
    γm1 = get_γ(equation) - 1.0
    return γm1 * SVector(U[2])
end

@inline function psi_ufun(equation::CompressibleIdealGas{Dim2}, U)
    γm1 = get_γ(equation) - 1.0
    return γm1 * SVector(U[2], U[3])
end

@inline function v_ufun(equation::CompressibleIdealGas{Dim2}, U)
    γ = get_γ(equation)
    s = sfun(equation, U)
    p = pfun(equation, U)
    rho, rhou, rhov, E = U
    v1 = (γ + 1 - s) - (γ - 1) * E / p
    vu = rhou * (γ - 1) / p
    vv = rhov * (γ - 1) / p
    vE = -rho * (γ - 1) / p
    return SVector{4,Float64}(v1, vu, vv, vE)
end

@inline function u_vfun(equation::CompressibleIdealGas{Dim1}, V)
    v1, vu, vE = V
    rhoeV = rhoe_vfun(equation, V)
    rho = -rhoeV * vE
    rhou = rhoeV * vu
    E = rhoeV * (1 - vu^2 / (2 * vE))
    return SVector{3,Float64}(rho, rhou, E)
end

@inline function u_vfun(equation::CompressibleIdealGas{Dim2}, V)
    v1, vu, vv, vE = V
    rhoeV = rhoe_vfun(equation, V)
    rho = -rhoeV * vE
    rhou = rhoeV * vu
    rhov = rhoeV * vv
    E = rhoeV * (1 - (vu^2 + vv^2) / (2 * vE))
    return SVector{4,Float64}(rho, rhou, rhov, E)
end

@inline function fluxes(equation::CompressibleIdealGas{Dim1}, U)
    rho, rhou, E = U
    p = pfun(equation, U)
    u = rhou / rho
    f1 = rhou
    f2 = rhou * u + p
    f3 = u * (E + p)
    return (SVector(f1, f2, f3),)
end

@inline function fluxes(equation::CompressibleIdealGas{Dim2}, U)
    rho, rhou, rhov, E = U
    p = pfun(equation, U)
    u = rhou / rho
    v = rhov / rho
    rhouv = rho * u * v
    Ep = E + p

    fx1 = rhou
    fx2 = rhou * u + p
    fx3 = rhouv
    fx4 = u * Ep

    fy1 = rhov
    fy2 = rhouv
    fy3 = rhov * v + p
    fy4 = v * Ep

    return SVector(fx1, fx2, fx3, fx4), SVector(fy1, fy2, fy3, fy4)
end

@inline function fS(equation::CompressibleIdealGas{Dim1}, UL, UR)
    γ = get_γ(equation)

    rhoL, uL, betaL, rhologL, betalogL = UL
    rhoR, uR, betaR, rhologR, betalogR = UR

    rholog = logmean.(rhoL, rhoR, rhologL, rhologR)
    betalog = logmean.(betaL, betaR, betalogL, betalogR)

    # arithmetic avgs
    rhoavg = 0.5 * (rhoL + rhoR)
    uavg = 0.5 * (uL + uR)

    unorm = uL * uR
    pa = rhoavg / (betaL + betaR)
    f4aux = rholog / (2 * (γ - 1) * betalog) + pa + 0.5 * rholog * unorm

    FxS1 = rholog * uavg
    FxS2 = FxS1 * uavg + pa
    FxS3 = f4aux * uavg

    return (SVector(FxS1, FxS2, FxS3),)
end

@inline function fS(equation::CompressibleIdealGas{Dim2}, UL, UR)
    γ = get_γ(equation)

    rhoL, uL, vL, betaL, rhologL, betalogL = UL
    rhoR, uR, vR, betaR, rhologR, betalogR = UR

    rholog = logmean.(rhoL, rhoR, rhologL, rhologR)
    betalog = logmean.(betaL, betaR, betalogL, betalogR)

    # arithmetic avgs
    rhoavg = 0.5 * (rhoL + rhoR)
    uavg = 0.5 * (uL + uR)
    vavg = 0.5 * (vL + vR)

    unorm = uL * uR + vL * vR
    pa = rhoavg / (betaL + betaR)
    f4aux = rholog / (2 * (γ - 1) * betalog) + pa + 0.5 * rholog * unorm

    FxS1 = rholog * uavg
    FxS2 = FxS1 * uavg + pa
    FxS3 = FxS1 * vavg
    FxS4 = f4aux * uavg

    FyS1 = rholog * vavg
    FyS2 = FxS3
    FyS3 = FyS1 * vavg + pa
    FyS4 = f4aux * vavg

    return SVector(FxS1, FxS2, FxS3, FxS4), SVector(FyS1, FyS2, FyS3, FyS4)
end

@inline function fS_x(equation::CompressibleIdealGas{Dim2}, UL, UR)
    γ = get_γ(equation)

    rhoL, uL, vL, betaL, rhologL, betalogL = UL
    rhoR, uR, vR, betaR, rhologR, betalogR = UR

    rholog = logmean.(rhoL, rhoR, rhologL, rhologR)
    betalog = logmean.(betaL, betaR, betalogL, betalogR)

    # arithmetic avgs
    rhoavg = 0.5 * (rhoL + rhoR)
    uavg = 0.5 * (uL + uR)
    vavg = 0.5 * (vL + vR)

    unorm = uL * uR + vL * vR
    pa = rhoavg / (betaL + betaR)
    f4aux = rholog / (2 * (γ - 1) * betalog) + pa + 0.5 * rholog * unorm

    FxS1 = rholog * uavg
    FxS2 = FxS1 * uavg + pa
    FxS3 = FxS1 * vavg
    FxS4 = f4aux * uavg

    return SVector(FxS1, FxS2, FxS3, FxS4), SVector(0.0, 0.0, 0.0, 0.0)
end

@inline function fS_y(equation::CompressibleIdealGas{Dim2}, UL, UR)
    γ = get_γ(equation)

    rhoL, uL, vL, betaL, rhologL, betalogL = UL
    rhoR, uR, vR, betaR, rhologR, betalogR = UR

    rholog = logmean.(rhoL, rhoR, rhologL, rhologR)
    betalog = logmean.(betaL, betaR, betalogL, betalogR)

    # arithmetic avgs
    rhoavg = 0.5 * (rhoL + rhoR)
    uavg = 0.5 * (uL + uR)
    vavg = 0.5 * (vL + vR)

    unorm = uL * uR + vL * vR
    pa = rhoavg / (betaL + betaR)
    f4aux = rholog / (2 * (γ - 1) * betalog) + pa + 0.5 * rholog * unorm

    FyS1 = rholog * vavg
    FyS2 = FyS1 * uavg
    FyS3 = FyS1 * vavg + pa
    FyS4 = f4aux * vavg

    return SVector(0.0, 0.0, 0.0, 0.0), SVector(FyS1, FyS2, FyS3, FyS4)
end

@inline function logmean(aL, aR)
    return logmean(aL, aR, log(aL), log(aR))
end

@inline function logmean(aL, aR, logL, logR)

    # From "Entropy stable num. approx. for the isothermal and polytropic Euler"
    # by Winters et al 2019.
    da = aR - aL
    aavg = 0.5 * (aR + aL)
    f = da / aavg
    v = f^2
    if abs(f) < 1e-4
        # numbers assume γ = 1.4 (Winters provides formulas for general γ)
        return aavg * (1 + v * (-0.2 - v * (0.0512 - v * 0.026038857142857)))
    else
        return -da / (logL - logR)
    end
end
