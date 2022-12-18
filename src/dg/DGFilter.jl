function exponential_filter(eta,Np)
    return diagm([exp(-eta*(i-1)*(i-1)) for i = 1:Np])
end

function zhang_shu_filter(eta,Np)
    F = eta*diagm(ones(Np))
    F[1,1] = 1.0
    return F
end

function apply_exponential_filter!(U,eta,Np)
    for i = 1:Np
        U[i] = exp(-eta*(i-1)*(i-1))*U[i]
    end
end

function apply_zhang_shu_filter!(U,eta,Np)
    for i = 2:Np
        U[i] = eta*U[i]
    end
end

function update_filtered_values!(prealloc,Umodal_k,eta,param,discrete_data_gauss)
    @unpack EQN,FILTERTYPE = param
    @unpack Np,Nh          = discrete_data_gauss.sizes
    @unpack VqVDM          = discrete_data_gauss.ops
    @unpack v3tilde,rhotilde,rhoetilde,v_tilde_k,u_tilde_k,U_k,Uq_k = prealloc
    
    copyto!(U_k,Umodal_k)
    if (FILTERTYPE < 0)
        apply_zhang_shu_filter!(U_k,eta,Np)
    else
        apply_exponential_filter!(U_k,eta,Np)
    end
    mul!(Uq_k,VqVDM,U_k)
    
    entropy_projection_element(Uq_k,param,discrete_data_gauss,prealloc)

    for i = 1:Nh
        v3tilde[i]   = v_tilde_k[i][end]
        rhotilde[i]  = u_tilde_k[i][1]
        rhoetilde[i] = rhoe_ufun_1D(EQN,u_tilde_k[i])
    end
end

function satisfy_bound(prealloc,bound,param,sizes,FILTERTYPE) 
    @unpack v3tilde,rhotilde,rhoetilde = prealloc
    @unpack POSTOL                     = param
    @unpack Nq,Nh                      = sizes

    v3max,rhomin,rhomax,rhoemin,rhoemax = bound
    if (abs(FILTERTYPE) == 1)
        # return (reduce(&, v3tilde_q .<= v3max) && reduce(&, v3tilde_f .<= -POSTOL))
        for i = 1:Nh
            upperbound = i <= Nq ? v3max : -POSTOL
            if (v3tilde[i] > upperbound)
                return false
            end
        end
        return true
    elseif (abs(FILTERTYPE) == 2)
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
    else
        return true
    end
end

function solve_eta(Umodal_k,bound,param,discrete_data_gauss,prealloc)
    @unpack ZEROTOL,FILTERTYPE = param
    @unpack Nh                 = discrete_data_gauss.sizes

    # If eta = 0 satisfy the bound, return 0
    eta = 0.0
    if (FILTERTYPE < 0) 
        eta = 1.0
    end
    try
        update_filtered_values!(prealloc,Umodal_k,eta,param,discrete_data_gauss)
        if satisfy_bound(prealloc,bound,param,discrete_data_gauss.sizes,FILTERTYPE)
            return eta
        end
    catch
    end

    # Bisection
    eta_l = 0.0
    eta_r = -log(ZEROTOL)
    if (FILTERTYPE < 0)
        eta_l = 1.0
        eta_r = 0.0
    end

    maxit = 20
    iter = 0
    while (iter <= maxit)
        eta_m = (eta_l+eta_r)/2
        try
            update_filtered_values!(prealloc,Umodal_k,eta_m,param,discrete_data_gauss)
            if satisfy_bound(prealloc,bound,param,discrete_data_gauss.sizes,FILTERTYPE)
                eta_r = eta_m
            else
                eta_l = eta_m
            end
        catch err
            if (isa(err, DomainError))
                eta_l = eta_m 
            end
        end
        iter = iter + 1
    end

    return eta_r
end

function apply_filter!(Uq,param,discrete_data_gauss,prealloc,nstage)
    @unpack ZETA,ETA,POSTOL,EQN,K,FILTERTYPE = param
    @unpack Np,Nq                            = discrete_data_gauss.sizes
    @unpack VDMinvPq,VqVDM                   = discrete_data_gauss.ops
    @unpack Farr,U_modal,LGLind              = prealloc

    if (FILTERTYPE > 0)
        view(Farr,:,nstage) .= 0.0
    end
    if (FILTERTYPE < 0)
        view(Farr,:,nstage) .= 1.0
    end
    for k = 1:K
        if (!LGLind[k])
            @views mul!(U_modal[:,k],VDMinvPq,Uq[:,k])
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
            U_modal_k = @views U_modal[:,k]
            Farr[k,nstage] = solve_eta(U_modal_k,bound,param,discrete_data_gauss,prealloc)
            if (FILTERTYPE > 0)
                apply_exponential_filter!(U_modal_k,Farr[k,nstage],Np)
            elseif (FILTERTYPE < 0)
                apply_zhang_shu_filter!(U_modal_k,Farr[k,nstage],Np)
            end

            @views mul!(Uq[:,k],VqVDM,U_modal[:,k])
        end
    end
end