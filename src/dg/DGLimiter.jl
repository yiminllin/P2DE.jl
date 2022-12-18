function limiting_param(param, U_low, P_ij, Lrho, Lrhoe)
    @unpack ZEROTOL = param
    l = 1.0
    # Limit density
    if U_low[1] + P_ij[1] < Lrho
        l = max((Lrho-U_low[1])/P_ij[1],0.0)
    end

    # limiting internal energy (via quadratic function)
    a = P_ij[1]*P_ij[3]-1.0/2.0*P_ij[2]^2
    b = U_low[3]*P_ij[1]+U_low[1]*P_ij[3]-U_low[2]*P_ij[2]-P_ij[1]*Lrhoe
    c = U_low[3]*U_low[1]-1.0/2.0*U_low[2]^2-U_low[1]*Lrhoe

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

    l = min(l,l_eps_ij)
    return l
end

function limiting_param_rhobound(param,rhoL,rhoP,Lrho,Urho)
    @unpack ZEROTOL = param
    TOL = ZEROTOL # TODO
    if ((rhoL + rhoP >= Lrho + TOL) && (rhoL + rhoP <= Urho - TOL))
        return 1.0
    else
        if ((rhoL < Lrho - ZEROTOL) || (rhoL > Urho + ZEROTOL))
            @show "low order rho not in bound",rhoL,rhoP,Lrho,Urho
            if (abs(rhoL-Lrho) > 1e-8 && abs(rhoL-Urho) > 1E-8)
                @show "Difference large:",abs(rhoL-Lrho),abs(rhoL-Urho)
            end
        end
        if (rhoL + rhoP < Lrho - TOL)
            return (Lrho-rhoL)/rhoP
        elseif (rhoL + rhoP > Urho + TOL)
            return (Urho-rhoL)/rhoP
        else
            return 1.0
        end
    end
end

# Find l s.t. rho(UL + lP)  ∈ [Lrho, Urho]
#             rhoe(UL + lP) ∈ [Lrhoe, inf)
#             E(UL + lP)    ∈ (-inf, UE]
function zhang_shu_limiting_param(param,UL,P,Lrho,Lrhoe,Urho,UE)
    @unpack POSTOL = param
    l       = 1.0
    l       = min(l, limiting_param(param,UL,P,Lrho,Lrhoe))
    l       = min(l, limiting_param_rhobound(param,UL[1],P[1],POSTOL,Urho))
    # TODO: bound on E
    l       = min(l, limiting_param_rhobound(param,UL[3],P[3],POSTOL,UE))
    return l
end

# rhsH, rhsL - low and high order right hand side
# i.e. d u^H / dt = r_i^H
#      d u^L / dt = r_i^L
# Then u_i^{n+1} = u_i^{L,n+1} + dt l (r^H_i - r^L_i)
#                  u_i^{L,n+1} = u_i^n + dt r^L_i
#                = u_i^n + dt[ (1-l)r^L_i + l r^H_i ]
# Return the new rhs r_i^limited = (1-l)r^L_i + l r^H_i
function zhang_shu_limiter(U,param,discrete_data,prealloc,dt,nstage)
    @unpack ZETA,K,EQN,POSTOL   = param
    @unpack Nq                  = discrete_data.sizes
    @unpack Larr,rhsU,rhsH,rhsL = prealloc

    for k = 1:K
        l = 1.0
        for i = 1:Nq
            U_low_i = U[i,k] + dt*rhsL[i,k]
            P_i     = dt*(rhsH[i,k]-rhsL[i,k])
            Lrho    = ZETA*U_low_i[1]
            Lrhoe   = ZETA*rhoe_ufun_1D(EQN,U_low_i)
            Urho    = Inf
            UE      = Inf
            l = min(l, zhang_shu_limiting_param(param,U_low_i,P_i,Lrho,Lrhoe,Urho,UE))
        end
        Larr[k,nstage] = l
        @views @. rhsU[:,k] = (1-l)*rhsL[:,k] + l*(rhsH[:,k])
    end
end

function get_average(Uk,Nq,wq)
    avg_k  = zero(SVector{3,Float64})
    for i = 1:Nq
        avg_k = avg_k + wq[i]*Uk[i]
    end
    avg_k = avg_k/2         # TODO: hardcoded reference element size
    return avg_k
end

function get_upper_bound(Uk,param,Nq)
    @unpack ETA = param

    Urho    = -Inf
    UE      = -Inf
    for i = 1:Nq
        Urho = max(Urho,Uk[i][1])
        UE   = max(UE,Uk[i][3])
    end
    Urho    = (1+ETA)*Urho
    UE      = (1+ETA)*UE

    return Urho,UE
end

function zhang_shu_limiter_transfer(Uk,Uk_new,avg_k,param,Urho,UE,Nq)
    @unpack POSTOL = param
    l = 1.0
    for i = 1:Nq
        U_low_i = avg_k
        P_i     = Uk_new[i]-avg_k
        Lrho    = POSTOL
        Lrhoe   = POSTOL 
        l = min(l, zhang_shu_limiting_param(param,U_low_i,P_i,Lrho,Lrhoe,Urho,UE))
    end
    return l
end

# TODO: refactor
function limit_Gauss_to_LGL!(Uk,param,discrete_data_gauss,transfer_ops,prealloc)
    @unpack POSTOL,K,XL,XR,EQN,ETA,ZETA = param
    @unpack LGLind                      = prealloc
    @unpack T_g2l  = transfer_ops
    @unpack Nq     = discrete_data_gauss.sizes
    @unpack wq     = discrete_data_gauss.ops

    avg_k   = get_average(Uk,Nq,wq) 
    Urho,UE = get_upper_bound(Uk,param,Nq)

    Uk_LGL = T_g2l*Uk
    l = zhang_shu_limiter_transfer(Uk,Uk_LGL,avg_k,param,Urho,UE,Nq)

    for i = 1:Nq
        Uk[i] = avg_k + l*(Uk_LGL[i]-avg_k)
    end

    return l
end

# TODO: refactor
function limit_LGL_to_Gauss!(Uk,param,discrete_data_LGL,transfer_ops,prealloc)
    @unpack POSTOL,K,XL,XR,EQN,ETA = param
    @unpack LGLind                 = prealloc
    @unpack T_l2g  = transfer_ops
    @unpack Nq     = discrete_data_LGL.sizes
    @unpack wq     = discrete_data_LGL.ops

    avg_k   = get_average(Uk,Nq,wq) 
    Urho,UE = get_upper_bound(Uk,param,Nq)

    Uk_Gauss = T_l2g*Uk
    l = zhang_shu_limiter_transfer(Uk,Uk_Gauss,avg_k,param,Urho,UE,Nq)

    for i = 1:Nq
        Uk[i] = avg_k + l*(Uk_Gauss[i]-avg_k)
    end

    return l
end

function solve_l_vf(Uq_k,bound,param,discrete_data_gauss,prealloc)
    @unpack ZEROTOL = param
    @unpack Nh      = discrete_data_gauss.sizes

    # If l = 1 satisfy the bound, return
    l = 1.0
    try
        update_filtered_values_Vf!(prealloc,Uq_k,l,param,discrete_data_gauss)
        if satisfy_bound(prealloc,bound,param,discrete_data_gauss.sizes)
            return l
        end
    catch
    end

    # Bisection
    l_l = 0.0
    l_r = 1.0

    maxit = 20
    iter = 0
    while (iter <= maxit)
        l_m = (l_l+l_r)/2
        try
            update_filtered_values_Vf!(prealloc,Uq_k,l_m,param,discrete_data_gauss)
            if satisfy_bound(prealloc,bound,param,discrete_data_gauss.sizes)
                l_l = l_m
            else
                l_r = l_m
            end
        catch err
            if (isa(err, DomainError))
                l_r = l_m 
            end
        end
        iter = iter + 1
    end

    return l_l
end

function entropy_projection_limitedVf!(Uq_k,l_k,param,discrete_data_gauss,prealloc)
    @unpack EQN                                      = param
    @unpack vq_k,v_tilde_k,u_tilde_k,VhPq_new,Vf_new = prealloc
    @unpack Nh,Nq                                    = discrete_data_gauss.sizes
    @unpack Vf,Vf_low,Pq                             = discrete_data_gauss.ops

    for i = 1:Nq
        vq_k[i] = v_ufun_1D(EQN,Uq_k[i])
    end
    @. Vf_new = l_k*Vf+(1.0-l_k)*Vf_low
    VfPq_new = @views VhPq_new[Nq+1:Nh,:]
    mul!(VfPq_new,Vf_new,Pq)
    mul!(v_tilde_k,VhPq_new,vq_k)
    for i = 1:Nh
        u_tilde_k[i] = u_vfun_1D(EQN,v_tilde_k[i])
    end
end

function update_filtered_values_Vf!(prealloc,Uq_k,l_k,param,discrete_data_gauss)
    @unpack EQN,FILTERTYPE = param
    @unpack Np,Nh          = discrete_data_gauss.sizes
    @unpack VqVDM          = discrete_data_gauss.ops
    @unpack v3tilde,rhotilde,rhoetilde,v_tilde_k,u_tilde_k = prealloc
    
    entropy_projection_limitedVf!(Uq_k,l_k,param,discrete_data_gauss,prealloc)

    for i = 1:Nh
        v3tilde[i]   = v_tilde_k[i][end]
        rhotilde[i]  = u_tilde_k[i][1]
        rhoetilde[i] = rhoe_ufun_1D(EQN,u_tilde_k[i])
    end
end

function satisfy_bound(prealloc,bound,param,sizes) 
    @unpack v3tilde,rhotilde,rhoetilde = prealloc
    @unpack POSTOL                     = param
    @unpack Nq,Nh                      = sizes

    v3max,rhomin,rhomax,rhoemin,rhoemax = bound
    for i = 1:Nh
        if ((v3tilde[i] > v3max)
         || (rhotilde[i] < rhomin)
         || (rhotilde[i] > rhomax)
         || (rhoetilde[i] < rhoemin)
         || (rhoetilde[i] > rhoemax))
            return false
        end
    end
    return true
end

function limit_Vf!(Uq,param,discrete_data_gauss,prealloc,nstage)
    @unpack EQN,POSTOL,ZETA,ETA,K = param
    @unpack LGLind,L_Vf_arr       = prealloc
    @unpack Nq                    = discrete_data_gauss.sizes

    view(L_Vf_arr,:,nstage) .= 1.0
    for k = 1:K
        l = 1.0
        if (!LGLind[k])
            v3max   = typemin(Float64)
            rhomax  = typemin(Float64)
            rhoemax = typemin(Float64)
            rhomin  = typemax(Float64)
            rhoemin = typemax(Float64)
            for i = 1:Nq
                u_i    = Uq[i,k]
                rho_i  = u_i[1]
                rhoe_i = rhoe_ufun_1D(EQN,u_i)
                v3_i   = v3_ufun_1D(EQN,u_i)
                rhomax  = max(rhomax,rho_i)
                rhoemax = max(rhoemax,rhoe_i)
                rhomin  = min(rhomin,rho_i)
                rhoemin = min(rhoemin,rhoe_i)
                v3max   = max(v3max,v3_i)
            end 
            bound = (ZETA*v3max,max((1-ETA)*rhomin,POSTOL),(1+ETA)*rhomax,max((1-ETA)*rhoemin,POSTOL),(1+ETA)*rhoemax)
            Uq_k = view(Uq,:,k)
            L_Vf_arr[k,nstage] = solve_l_vf(Uq_k,bound,param,discrete_data_gauss,prealloc)
        end
    end
end