function primitive_to_conservative(equation::CompressibleIdealGas{Dim1},U)
    γ = get_γ(equation)
    rho,u,p = U
    rhou = rho*u 
    E = p/(γ-1) + .5*rho*u^2
    return SVector{3,Float64}(rho,rhou,E)
end

function primitive_to_conservative(equation::CompressibleIdealGas{Dim2},U)
    γ = get_γ(equation)
    rho,u,v,p = U
    rhou = rho*u
    rhov = rho*v
    E = p/(γ-1) + .5*rho*(u^2+v^2)
    return SVector{4,Float64}(rho,rhou,rhov,E)
end

@inline function pfun(equation::CompressibleIdealGas{Dim1},U)
    γ = get_γ(equation)
    rho,rhou,E = U
    return (γ-1.0)*(E-.5*rhou^2/rho)
end

@inline function pfun(equation::CompressibleIdealGas{Dim2},U)
    γ = get_γ(equation)
    rho,rhou,rhov,E = U
    return (γ-1.0)*(E-.5*(rhou^2+rhov^2)/rho)
end

@inline function E_ufun(equation::CompressibleIdealGas{Dim1},U)
    γ = get_γ(equation)
    rho,u,p = U
    return p/(γ-1) + .5*rho*(u^2)
end

@inline function E_ufun(equation::CompressibleIdealGas{Dim2},U)
    γ = get_γ(equation)
    rho,u,v,p = U
    return p/(γ-1) + .5*rho*(u^2+v^2)
end

@inline function betafun(equation::CompressibleIdealGas{Dim},U) where {Dim}
    p = pfun(equation,U)
    return U[1]/(2*p)
end

# TODO: dispatch on wavespeed estimate type
@inline function wavespeed_estimate(equation::CompressibleIdealGas{Dim1},U)
    γ = get_γ(equation)
    p = pfun(equation,U)
    return abs(U[2]/U[1]) + sqrt(γ*p/U[1])
end

@inline function wavespeed_estimate(equation::CompressibleIdealGas{Dim1},U,n)
    return wavespeed_estimate(equation,U)
end

# n - normalized normal
@inline function wavespeed_estimate(equation::CompressibleIdealGas{Dim2},U,n)
    Un = SVector(U[1],n[1]*U[2]+n[2]*U[3],U[4])
    return wavespeed_estimate(get_equation_1D(equation),Un)
end

function zhang_positivity_estimate(equation::CompressibleNavierStokesIdealGas{Dim1},U,sigma,n)
    rho,rhou,E = U
    _,sigma2,sigma3 = sigma[1]
    p   = pfun(equation,U)
    u   = rhou/rho
    e   = (E - .5*rho*u^2)/rho
    tau = sigma2
    q   = u*tau-sigma3
    return abs(u)+1/(2*rho^2*e)*(sqrt(rho^2*q^2+2*rho^2*e*abs(tau-p)^2)+rho*abs(q))
end

@inline function zhang_positivity_estimate(equation::CompressibleNavierStokesIdealGas{Dim2},U,sigma,n)
    rho,rhou,rhov,E = U
    _,sigmax2,sigmax3,sigmax4 = sigma[1]
    _,sigmay2,sigmay3,sigmay4 = sigma[2]
    nx,ny = n

    p = pfun(equation,U)
    u = rhou/rho
    v = rhov/rho
    e = (E-.5*rho*(u^2+v^2))/rho
    tau_xx = sigmax2
    tau_yx = sigmax3
    tau_xy = sigmay2
    tau_yy = sigmay3
    q_x = u*tau_xx+v*tau_yx-sigmax4
    q_y = u*tau_xy+v*tau_yy-sigmay4

    v_vec = u*nx+v*ny
    q_vec = q_x*nx+q_y*ny
    tau_vec_x = nx*tau_xx+ny*tau_yx
    tau_vec_y = nx*tau_xy+ny*tau_yy

    return abs(v_vec)+1/(2*rho^2*e)*(sqrt(rho^2*q_vec^2+2*rho^2*e*((tau_vec_x-p*nx)^2+(tau_vec_y-p*ny)^2))+rho*abs(q_vec))
end

@inline function sfun(equation::CompressibleIdealGas{Dim},U) where {Dim}
    γ = get_γ(equation)
    p = pfun(equation,U)
    return log(p/(U[1]^γ))
end

@inline function rhoe_ufun(equation::CompressibleIdealGas{Dim1},U)
    rho,rhou,E = U
    return E-.5*rhou*rhou/rho
end

@inline function rhoe_ufun(equation::CompressibleIdealGas{Dim2},U)
    rho,rhou,rhov,E = U
    return E-.5*(rhou*rhou+rhov*rhov)/rho
end

@inline function s_modified_ufun(equation::CompressibleIdealGas,U)
    rho = U[1]
    γ = get_γ(equation)
    rhoe = rhoe_ufun(equation,U)
    return rhoe*rho^(-γ)
end

@inline function s_vfun(equation::CompressibleIdealGas{Dim1},V)
    γ = get_γ(equation)
    v1,vu,vE = V
    return γ-v1+vu^2/(2*vE)
end

@inline function s_vfun(equation::CompressibleIdealGas{Dim2},V)
    γ = get_γ(equation)
    v1,vu,vv,vE = V
    return γ-v1+(vu^2+vv^2)/(2*vE)
end

@inline function rhoe_vfun(equation::CompressibleIdealGas{Dim},V) where {Dim}
    γ = get_γ(equation)
    s = s_vfun(equation,V)
    vE = V[end]
    return ((γ-1)/((-vE)^γ))^(1/(γ-1)) * exp(-s/(γ-1))
end

@inline function v3_ufun(equation::CompressibleIdealGas{Dim},U) where {Dim}
    γ = get_γ(equation)
    p = pfun(equation,U)
    rho = U[1]
    return -rho*(γ-1)/p
end

@inline function v_ufun(equation::CompressibleIdealGas{Dim1},U)
    γ = get_γ(equation)
    s = sfun(equation,U)
    p = pfun(equation,U)
    rho,rhou,E = U
    v1 = (γ+1-s)-(γ-1)*E/p
    vu = rhou*(γ-1)/p
    vE = -rho*(γ-1)/p
    return SVector{3,Float64}(v1,vu,vE)
end

@inline function psi_ufun(equation::CompressibleIdealGas{Dim1},U)
    γm1 = get_γ(equation)-1.0
    return γm1*SVector(U[2])
end

@inline function psi_ufun(equation::CompressibleIdealGas{Dim2},U)
    γm1 = get_γ(equation)-1.0
    return γm1*SVector(U[2],U[3])
end

@inline function v_ufun(equation::CompressibleIdealGas{Dim2},U)
    γ = get_γ(equation)
    s = sfun(equation,U)
    p = pfun(equation,U)
    rho,rhou,rhov,E = U
    v1 = (γ+1-s)-(γ-1)*E/p
    vu = rhou*(γ-1)/p
    vv = rhov*(γ-1)/p
    vE = -rho*(γ-1)/p
    return SVector{4,Float64}(v1,vu,vv,vE)
end

@inline function u_vfun(equation::CompressibleIdealGas{Dim1},V)
    v1,vu,vE = V
    rhoeV = rhoe_vfun(equation,V)
    rho   = -rhoeV*vE
    rhou  = rhoeV*vu
    E     = rhoeV*(1-vu^2/(2*vE))
    return SVector{3,Float64}(rho,rhou,E)
end

@inline function u_vfun(equation::CompressibleIdealGas{Dim2},V)
    v1,vu,vv,vE = V
    rhoeV = rhoe_vfun(equation,V)
    rho   = -rhoeV*vE
    rhou  = rhoeV*vu
    rhov  = rhoeV*vv
    E     = rhoeV*(1-(vu^2+vv^2)/(2*vE))
    return SVector{4,Float64}(rho,rhou,rhov,E)
end

@inline function get_Kvisc(equation::CompressibleNavierStokesIdealGas{Dim1},VU)
    lambda = get_λ(equation)
    mu = get_μ(equation)
    kappa = get_κ(equation)
    cv = get_cv(equation)
    _,v2,v3 = VU

    return SMatrix{3,3}(0.0, 0.0, 0.0,
                        0.0, -(2*mu-lambda)/v3, (2*mu-lambda)*v2/v3^2,
                        0.0, (2*mu-lambda)*v2/v3^2, -(2*mu-lambda)*v2^2/v3^3+kappa/cv/v3^2)
end

@inline function get_Kvisc(equation::CompressibleNavierStokesIdealGas{Dim2},VU)
    γ = get_γ(equation)
    lambda = get_λ(equation)
    mu = get_μ(equation)
    Pr = get_Pr(equation)
    _,v2,v3,v4 = VU

    # TODO: Clean up...
    λ = -lambda
    μ   = mu
    v2_sq = v2^2
    v3_sq = v3^2
    v4_sq = v4^2
    λ2μ = (λ+2.0*μ)
    iv4c = 1/(v4^3)

    Kxx = SMatrix{4,4}(0.0, 0.0, 0.0, 0.0,
                       0.0, iv4c*-λ2μ*v4_sq, 0.0, iv4c*λ2μ*v2*v4,
                       0.0, 0.0, iv4c*-μ*v4_sq, iv4c*μ*v3*v4,
                       0.0, iv4c*λ2μ*v2*v4, iv4c*μ*v3*v4, iv4c*-(λ2μ*v2_sq + μ*v3_sq - γ*μ*v4/Pr))
    Kxy = SMatrix{4,4}(0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, iv4c*-λ*v4_sq, iv4c*λ*v3*v4,
                       0.0, iv4c*-μ*v4_sq, 0.0, iv4c*μ*v2*v4,
                       0.0, iv4c*μ*v3*v4, iv4c*λ*v2*v4, iv4c*(λ+μ)*(-v2*v3))
    Kyx = Kxy'
    Kyy = SMatrix{4,4}(0.0, 0.0, 0.0, 0.0,
                       0.0, iv4c*-μ*v4_sq, 0.0, iv4c*μ*v2*v4,
                       0.0, 0.0, iv4c*-λ2μ*v4_sq, iv4c*λ2μ*v3*v4,
                       0.0, iv4c*μ*v2*v4, iv4c*λ2μ*v3*v4, iv4c*-(λ2μ*v3_sq + μ*v2_sq - γ*μ*v4/Pr))

    return (Kxx,Kxy,Kyx,Kyy)
end

@inline function euler_inviscid_fluxes(dim::Dim1, U, p)
    rho,rhou,E = U
    u  = rhou/rho
    f1 = rhou
    f2 = rhou*u+p
    f3 = u*(E+p)
    return (SVector(f1,f2,f3),)
end

@inline function euler_inviscid_fluxes(dim::Dim2, U, p)
    rho,rhou,rhov,E = U
    u  = rhou/rho
    v  = rhov/rho
    rhouv = rho*u*v
    Ep    = E+p

    fx1 = rhou
    fx2 = rhou*u+p
    fx3 = rhouv
    fx4 = u*Ep

    fy1 = rhov
    fy2 = rhouv
    fy3 = rhov*v+p
    fy4 = v*Ep

    return SVector(fx1,fx2,fx3,fx4), SVector(fy1,fy2,fy3,fy4)
end

@inline function fluxes(equation::CompressibleEulerIdealGas,U)
    return euler_inviscid_fluxes(get_dim_type(equation), U, pfun(equation, U)), SVector(zero(U), zero(U))
end

@inline function fluxes(equation::CompressibleNavierStokesIdealGas,U)
    u,sigma = U
    return euler_inviscid_fluxes(get_dim_type(equation), u, pfun(equation, u)), sigma
end

# https://arxiv.org/pdf/1901.04924.pdf
@inline function noslip_flux(equation::CompressibleIdealGas,rhoM,uM,vM,pM,nx,ny)
    γ = get_γ(equation)
    # Assume n = (n1,n2) normalized normal
    c     = sqrt(γ*pM/rhoM)
    vn    = uM*nx+vM*ny
    Ma_n  = vn/c
    Pstar = pM
    if (vn > 0)
        Pstar = (1+γ*Ma_n*((γ+1)/4*Ma_n + sqrt(((γ+1)/4*Ma_n)^2+1)))*pM
    else
        Pstar = max((1+1/2*(γ-1)*Ma_n)^(2*γ/(γ-1)), 1e-4)*pM
    end

    return 0.0, Pstar*nx, Pstar*ny, 0.0
end

@inline function fS(equation::CompressibleIdealGas{Dim1},UL,UR)
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

    return (SVector(FxS1,FxS2,FxS3),)
end

@inline function fS(equation::CompressibleIdealGas{Dim2},UL,UR)
    γ = get_γ(equation)

    rhoL,uL,vL,betaL,rhologL,betalogL = UL
    rhoR,uR,vR,betaR,rhologR,betalogR = UR

    rholog  = logmean.(rhoL,rhoR,rhologL,rhologR)
    betalog = logmean.(betaL,betaR,betalogL,betalogR)

    # arithmetic avgs
    rhoavg = .5*(rhoL+rhoR)
    uavg   = .5*(uL+uR)
    vavg   = .5*(vL+vR)

    unorm = uL*uR + vL*vR
    pa    = rhoavg/(betaL+betaR)
    f4aux = rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm

    FxS1 = rholog*uavg
    FxS2 = FxS1*uavg + pa
    FxS3 = FxS1*vavg
    FxS4 = f4aux*uavg

    FyS1 = rholog*vavg
    FyS2 = FxS3
    FyS3 = FyS1*vavg + pa
    FyS4 = f4aux*vavg

    return SVector(FxS1,FxS2,FxS3,FxS4), SVector(FyS1,FyS2,FyS3,FyS4)
end

@inline function fS_x(equation::CompressibleIdealGas{Dim2},UL,UR)
    γ = get_γ(equation)

    rhoL,uL,vL,betaL,rhologL,betalogL = UL
    rhoR,uR,vR,betaR,rhologR,betalogR = UR

    rholog  = logmean.(rhoL,rhoR,rhologL,rhologR)
    betalog = logmean.(betaL,betaR,betalogL,betalogR)

    # arithmetic avgs
    rhoavg = .5*(rhoL+rhoR)
    uavg   = .5*(uL+uR)
    vavg   = .5*(vL+vR)

    unorm = uL*uR + vL*vR
    pa    = rhoavg/(betaL+betaR)
    f4aux = rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm

    FxS1 = rholog*uavg
    FxS2 = FxS1*uavg + pa
    FxS3 = FxS1*vavg
    FxS4 = f4aux*uavg

    return SVector(FxS1,FxS2,FxS3,FxS4), SVector(0.0,0.0,0.0,0.0)
end

@inline function fS_y(equation::CompressibleIdealGas{Dim2},UL,UR)
    γ = get_γ(equation)

    rhoL,uL,vL,betaL,rhologL,betalogL = UL
    rhoR,uR,vR,betaR,rhologR,betalogR = UR

    rholog  = logmean.(rhoL,rhoR,rhologL,rhologR)
    betalog = logmean.(betaL,betaR,betalogL,betalogR)

    # arithmetic avgs
    rhoavg = .5*(rhoL+rhoR)
    uavg   = .5*(uL+uR)
    vavg   = .5*(vL+vR)

    unorm = uL*uR + vL*vR
    pa    = rhoavg/(betaL+betaR)
    f4aux = rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm

    FyS1 = rholog*vavg
    FyS2 = FyS1*uavg
    FyS3 = FyS1*vavg + pa
    FyS4 = f4aux*vavg

    return SVector(0.0,0.0,0.0,0.0), SVector(FyS1,FyS2,FyS3,FyS4)
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