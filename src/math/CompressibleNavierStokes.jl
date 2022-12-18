function primitive_to_conservative(eqn,U)
    @unpack γ = eqn
    rho,u,p = U
    rhou = rho*u 
    E = p/(γ-1) + .5*rho*u^2
    return SVector{3,Float64}(rho,rhou,E)
end

# TODO: only 1D
@inline function pfun_1D(eqn,U)
    rho,rhou,E = U
    return (eqn.γ-1.0)*(E-.5*rhou^2/rho)
end

@inline function E_ufun_1D(eqn,U)
    @unpack γ = eqn
    rho,u,p = U
    return p/(γ-1) + .5*rho*(u^2)
end

@inline function betafun_1D(eqn,U)
    p = pfun_1D(eqn,U)
    return U[1]/(2*p)
end

@inline function wavespeed_davis_estimate_1D(eqn,U)
    p = pfun_1D(eqn,U)
    return abs(U[2]/U[1]) + sqrt(eqn.γ*p/U[1])
end

@inline function sfun_1D(eqn,U)
    p = pfun_1D(eqn,U)
    return log(p/(U[1]^eqn.γ))
end

@inline function rhoe_ufun_1D(eqn,U)
    @unpack γ = eqn
    rho,rhou,E = U
    return E-.5*rhou*rhou/rho
end

@inline function s_vfun_1D(eqn,V)
    @unpack γ = eqn
    v1,vu,vE = V
    return γ-v1+vu^2/(2*vE)
end

@inline function rhoe_vfun_1D(eqn,V)
    @unpack γ = eqn
    s = s_vfun_1D(eqn,V)
    v1,vu,vE = V
    return ((γ-1)/((-vE)^γ))^(1/(γ-1)) * exp(-s/(γ-1))
end

@inline function v3_ufun_1D(eqn,U)
    @unpack γ = eqn
    p = pfun_1D(eqn,U)
    rho,rhou,E = U
    return -rho*(γ-1)/p
end

@inline function v_ufun_1D(eqn,U)
    @unpack γ = eqn
    s = sfun_1D(eqn,U)
    p = pfun_1D(eqn,U)
    rho,rhou,E = U
    v1 = (γ+1-s)-(γ-1)*E/p
    vu = rhou*(γ-1)/p
    vE = -rho*(γ-1)/p
    return SVector{3,Float64}(v1,vu,vE)
end

@inline function u_vfun_1D(eqn,V)
    v1,vu,vE = V
    rhoeV = rhoe_vfun_1D(eqn,V)
    rho   = -rhoeV*vE
    rhou  = rhoeV*vu
    E     = rhoeV*(1-vu^2/(2*vE))
    return SVector{3,Float64}(rho,rhou,E)
end

@inline function euler_fluxes_1D(eqn,U)
    rho,rhou,E = U
    p  = pfun_1D(eqn,U)
    u  = rhou/rho
    f1 = rhou
    f2 = rhou*u+p
    f3 = u*(E+p)
    return SVector{3,Float64}(f1,f2,f3);
end