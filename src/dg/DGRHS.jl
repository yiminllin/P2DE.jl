function update_indicator(Uq,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,prealloc,firststep=false)
    @unpack K,DISCRETIZATION_TYPE      = param
    @unpack Nq                         = discrete_data_gauss.sizes
    @unpack LGLind,L_L2G_arr,L_G2L_arr = prealloc
    @unpack T_g2l,T_l2g                = transfer_ops

    L_G2L_arr .= 1.0
    L_L2G_arr .= 1.0
    for k = 1:K
        ind = false
        if (DISCRETIZATION_TYPE == 0)
            ind = false
        elseif (DISCRETIZATION_TYPE == 1)
            ind = true
        else
            # TODO: a "nearly do-nothing" indicator
            ind = false
            for i = 1:Nq
                if (Uq[i,k][1] < 1e-2 || Uq[i,k][3] < 1e-2)
                    ind = true
                    break
                end
            end
        end

        if (firststep)
            LGLind[k] = ind
        else
            # If switch from LGL to Gauss or Gauss to LGL, we need to cast the
            # solution to Gauss/LGL
            if (LGLind[k] != ind)
                LGLind[k] = ind
                # Gauss to LGL
                @views Uk = Uq[:,k]
                if (LGLind[k])
                    L_G2L_arr[k] = limit_Gauss_to_LGL!(Uk,param,discrete_data_gauss,transfer_ops,prealloc)
                else
                    L_L2G_arr[k] = limit_LGL_to_Gauss!(Uk,param,discrete_data_LGL,transfer_ops,prealloc)
                end
            end
        end
    end
end

function entropy_projection_element(U_k,param,discrete_data_gauss,prealloc)
    @unpack EQN                      = param
    @unpack vq_k,v_tilde_k,u_tilde_k = prealloc
    @unpack Nh,Nq                    = discrete_data_gauss.sizes
    @unpack VhPq                     = discrete_data_gauss.ops

    for i = 1:Nq
        vq_k[i] = v_ufun_1D(EQN,U_k[i])
    end
    mul!(v_tilde_k,VhPq,vq_k)
    for i = 1:Nh
        u_tilde_k[i] = u_vfun_1D(EQN,v_tilde_k[i])
    end
end

function entropy_projection(Uq,param,discrete_data_gauss,discrete_data_LGL,prealloc,nstage)
    @unpack K,EQN,MODIFYVF                                     = param
    @unpack vq,v_tilde,u_tilde,LGLind,L_Vf_arr,Vf_new,VhPq_new = prealloc
    @unpack Nh,Nq,Nfp                                          = discrete_data_gauss.sizes
    
    for k = 1:K
        for i = 1:Nq
            vq[i,k] = v_ufun_1D(EQN,Uq[i,k])
        end
        if LGLind[k]
            @views mul!(v_tilde[:,k],discrete_data_LGL.ops.VhPq,vq[:,k])
        else
            if (MODIFYVF)
                l_k = L_Vf_arr[k,nstage]
                @. Vf_new = l_k*discrete_data_gauss.ops.Vf+(1-l_k)*discrete_data_gauss.ops.Vf_low
                VfPq_new = @views VhPq_new[Nq+1:Nh,:]
                mul!(VfPq_new,Vf_new,discrete_data_gauss.ops.Pq)
                @views mul!(v_tilde[:,k],VhPq_new,vq[:,k])
            else
                @views mul!(v_tilde[:,k],discrete_data_gauss.ops.VhPq,vq[:,k])
            end
        end
        for i = 1:Nh
            u_tilde[i,k] = u_vfun_1D(EQN,v_tilde[i,k])
        end
    end
end

# RHS of modal ESDG
function rhs_modalESDG(Uq,param,discrete_data_gauss,discrete_data_LGL,bcdata,prealloc,nstage,need_proj=true)
    @unpack K,EQN,MODIFYVF           = param
    @unpack mapP,mapI,mapO,inflowarr = bcdata
    @unpack Nh,Np,Nq,Nfp                = discrete_data_gauss.sizes
    # TODO: Assume uniform mesh, Need geom for LGL
    @unpack J,rxJh                      = discrete_data_gauss.geom
    @unpack vq,v_tilde,u_tilde,beta,rholog,betalog,lam,LFc,rhsH,Ui,Uj,QF1,BF1,uP,betaP,rhologP,betalogP,spatial,boundary,LGLind,L_Vf_arr,Vf_new,VhT_new,MinvVhT_new = prealloc

    if (need_proj)
        entropy_projection(Uq,param,discrete_data_gauss,discrete_data_LGL,prealloc,nstage)
    end

    for k = 1:K
        for i = 1:Nh
            beta[i,k]    = betafun_1D(EQN,u_tilde[i,k])
            rholog[i,k]  = log(u_tilde[i,k][1])
            betalog[i,k] = log(beta[i,k])
        end
    end

    # Boundary contributions
    uf       = @view u_tilde[Nq+1:Nh,:]
    betaf    = @view beta[Nq+1:Nh,:]
    rhologf  = @view rholog[Nq+1:Nh,:]
    betalogf = @view betalog[Nq+1:Nh,:]
    for k = 1:K
        for i = 1:Nfp
            iP = mapP[i,k]
            uP[i,k]       = uf[iP]
            betaP[i,k]    = betaf[iP]
            rhologP[i,k]  = rhologf[iP]
            betalogP[i,k] = betalogf[iP]
        end
    end

    # Lax Friedrichs dissipation
    for k = 1:K
        for i = 1:Nfp
            lam[i,k] = wavespeed_davis_estimate_1D(EQN,uf[i,k])
        end
    end

    for k = 1:K
        for i = 1:Nfp
            LFc[i,k] = 0.5*max(lam[i,k],lam[mapP[i,k]])
        end
    end

    # zero dissipation on the boundary
    for i = mapI
        LFc[i] = 0.0
    end 
    for i in mapO
        LFc[i] = 0.0
    end 

    # Enforce Inflow BC
    for i = 1:size(mapI,1)
        ii = mapI[i]
        uP[ii]       = inflowarr[i]
        betaP[ii]    = betafun(EQN,uP[ii])
        rhologP[ii]  = log(uP[ii][1])
        betalogP[ii] = log(betaP[ii])
    end

    # Enforce outflow BC
    for i = 1:size(mapO,1)
        io = mapO[i]
        # TODO: hardcoded
        uP[io]       = Uq[end,end]
        betaP[io]    = betafun(EQN,uP[io])
        rhologP[io]  = log(uP[io][1])
        betalogP[io] = log(betaP[io])
    end

    # Flux differencing
    @. QF1 = zero(QF1)
    for k = 1:K
        for j = 1:Nh
            Uj[1] = u_tilde[j,k][1]
            Uj[2] = u_tilde[j,k][2]/Uj[1]
            Uj[3] = beta[j,k]
            Uj[4] = rholog[j,k]
            Uj[5] = betalog[j,k]
            for i = j+1:Nh
                Ui[1] = u_tilde[i,k][1]
                Ui[2] = u_tilde[i,k][2]/Ui[1]
                Ui[3] = beta[i,k]
                Ui[4] = rholog[i,k]
                Ui[5] = betalog[i,k]
                if (MODIFYVF)
                    l_k = L_Vf_arr[k,nstage]
                    Qrh_skew_db_ij = (LGLind[k]) ? discrete_data_LGL.ops.Qrh_skew_db[i,j] : l_k*discrete_data_gauss.ops.Qrh_skew_db[i,j] + (1-l_k)*discrete_data_gauss.ops.Qrh_skew_low_db[i,j]
                else
                    Qrh_skew_db_ij = (LGLind[k]) ? discrete_data_LGL.ops.Qrh_skew_db[i,j] : discrete_data_gauss.ops.Qrh_skew_db[i,j]
                end
                QFij = Qrh_skew_db_ij*fS_prim_log(EQN,Ui,Uj)
                QF1[i,k] += rxJh[i,k]*QFij
                QF1[j,k] -= rxJh[j,k]*QFij
            end
        end

        # Boundary contributions (B F)1
        for i = 1:Nfp
            # BF1[i,k] = fS_prim_log(EQN,(uf[i,k][1],uf[i,k][2]/uf[i,k][1],betaf[i,k],rhologf[i,k],betalogf[i,k]),
            #                            (uP[i,k][1],uP[i,k][2]/uP[i,k][1],betaP[i,k],rhologP[i,k],betalogP[i,k]))
            BF1[i,k] = .5*(euler_fluxes_1D(EQN,uf[i,k])+euler_fluxes_1D(EQN,uP[i,k]))
        end
        BF1[1,k] = -BF1[1,k]     # TODO: hardcode scale by normal
        # LF dissipation
        for i = 1:Nfp
            BF1[i,k] -= LFc[i,k]*(uP[i,k]-uf[i,k])
        end
    end

    for k = 1:K
        if LGLind[k]
            @views mul!(spatial[:,k],discrete_data_LGL.ops.MinvVhT,QF1[:,k])
            @views mul!(boundary[:,k],discrete_data_LGL.ops.LIFT,BF1[:,k])
        else
            if (MODIFYVF)
                l_k = L_Vf_arr[k,nstage]
                @. Vf_new = l_k*discrete_data_gauss.ops.Vf+(1.0-l_k)*discrete_data_gauss.ops.Vf_low
                VqT_new = @views VhT_new[:,1:Nq]
                VfT_new = @views VhT_new[:,Nq+1:Nh]
                VqT_new .= transpose(discrete_data_gauss.ops.Vq)
                VfT_new .= transpose(Vf_new)
                @. MinvVhT_new = (1/discrete_data_gauss.ops.wq)*VhT_new
                MinvVfT_new = @views MinvVhT_new[:,Nq+1:Nh]
                @views mul!(spatial[:,k],MinvVhT_new,QF1[:,k])
                @views mul!(boundary[:,k],MinvVfT_new,BF1[:,k])
            else
                @views mul!(spatial[:,k],discrete_data_gauss.ops.MinvVhT,QF1[:,k])
                @views mul!(boundary[:,k],discrete_data_gauss.ops.LIFT,BF1[:,k])
            end
        end
        for i = 1:Np
            spatial[i,k] = -(spatial[i,k]+boundary[i,k])/J[i,k]
        end
        if LGLind[k]
            @views mul!(rhsH[:,k],discrete_data_LGL.ops.Vq,spatial[:,k])
        else
            @views mul!(rhsH[:,k],discrete_data_gauss.ops.Vq,spatial[:,k])
        end
    end
end

function find_alpha(param,ui,uitilde)
    @unpack EQN,POSTOL = param
    alphaL = 0.0
    alphaR = 1.0
    substate = alphaR*ui-uitilde
    while (true)
        if (substate[1] > POSTOL && rhoe_ufun_1D(EQN,substate) > POSTOL)
            break
        end
        alphaR = 2*alphaR
        substate = alphaR*ui-uitilde
    end

    maxit = 50
    iter = 0.0
    tolerance = 1e-8
    while (iter < maxit || (alphaL-alphaR) > tolerance)
        alphaM = (alphaL+alphaR)/2
        substate = alphaM*ui-uitilde
        if (substate[1] > POSTOL && rhoe_ufun_1D(EQN,substate) > POSTOL)
            alphaR = alphaM
        else
            alphaL = alphaM
        end
        iter = iter + 1
    end

    return alphaR
end

function rhs_pos_Gauss(Uq,param,discrete_data_gauss,discrete_data_LGL,bcdata,prealloc,t,dt,nstage,need_proj=true)
    @unpack K,T,EQN,CFL,dt0                      = param
    @unpack mapP,mapI,mapO,inflowarr             = bcdata
    @unpack Jq                                   = discrete_data_gauss.geom
    @unpack Nh,Nq,Nfp                            = discrete_data_gauss.sizes
    @unpack u_tilde,flux,wavespeed,alphaarr,rhsL = prealloc
    @unpack αarr,LGLind                          = prealloc

    if (need_proj)
        entropy_projection(Uq,param,discrete_data_gauss,discrete_data_LGL,prealloc,nstage)
    end
    
    for k = 1:K
        for i = 1:Nh
            ui = (i <= Nq) ? Uq[i,k] : u_tilde[i,k]
            wavespeed[i,k] = wavespeed_davis_estimate_1D(EQN,ui)
            flux[i,k]      = euler_fluxes_1D(EQN,ui)
        end
    end

    utilde_f    = @view u_tilde[Nq+1:Nh,:]
    flux_f      = @view flux[Nq+1:Nh,:]
    wavespeed_f = @view wavespeed[Nq+1:Nh,:]

    # Determine positivity CFL
    if (nstage == 1)
        dt = min(CFL*dt0,T-t)
        for k = 1:K
            αarr[1,k]   = find_alpha(param,Uq[1,k],u_tilde[1,k])
            αarr[end,k] = find_alpha(param,Uq[end,k],u_tilde[end,k])
            for i = 1:Nq
                wq_i = LGLind[k] ? discrete_data_LGL.ops.wq[i] : discrete_data_gauss.ops.wq[i]
                wJq_i    = Jq[i,k]*wq_i
                lambda_i = 0.0
                for j = 1:Nq
                    Sr0_ij = LGLind[k] ? discrete_data_LGL.ops.Sr0[i,j] : discrete_data_gauss.ops.Sr0[i,j]
                    if Sr0_ij != 0
                        lambda_i += abs(Sr0_ij)*max(wavespeed[i,k], wavespeed[j,k])
                    end
                end
                if i == 1
                    lambda_i += LGLind[k] ? .5*max(wavespeed_f[1,k],wavespeed_f[mapP[1+(k-1)*Nfp]]) : αarr[1,k]*.5*max(wavespeed_f[1,k],wavespeed_f[mapP[1+(k-1)*Nfp]]) + .5*wavespeed[i,k]
                end
                if i == Nq
                    lambda_i += LGLind[k] ? .5*max(wavespeed_f[2,k],wavespeed_f[mapP[2+(k-1)*Nfp]]) : αarr[2,k]*.5*max(wavespeed_f[2,k],wavespeed_f[mapP[2+(k-1)*Nfp]]) + .5*wavespeed[i,k]
                end

                dt = min(dt, CFL*.5*wJq_i/lambda_i)
            end
        end
    end

    # Assemble RHS
    for k = 1:K
        for i = 1:Nq
            rhsL[i,k] = zero(SVector{3,Float64})
        end
    end
    for k = 1:K
        # Volume contributions
        for j = 1:Nq
            for i = j+1:Nq
                Sr0_ij = LGLind[k] ? discrete_data_LGL.ops.Sr0[i,j] : discrete_data_gauss.ops.Sr0[i,j]
                if Sr0_ij != 0
                    Fij      = .5*(flux[i,k]+flux[j,k])
                    lambdaij = abs(Sr0_ij)*max(wavespeed[i,k], wavespeed[j,k])
                    LFij     = (2.0*Sr0_ij*Fij) - lambdaij*(Uq[j,k]-Uq[i,k])
                    rhsL[i,k] -= LFij
                    rhsL[j,k] += LFij
                end
            end
        end

        # Surface contributions
        for i = 1:Nfp
            idx = i+Nfp*(k-1)
            Iidx = findfirst(x->(x==idx), mapI)
            Oidx = findfirst(x->(x==idx), mapO)
            fluxP   = !isnothing(Iidx) ? euler_fluxes_1D(EQN,inflowarr[Iidx]) : flux_f[mapP[idx]]
            utildeP = !isnothing(Iidx) ? inflowarr[Iidx] : utilde_f[mapP[idx]]
            lambdaD = (!isnothing(Iidx) || !isnothing(Oidx)) ? 0.0 : .5*max(wavespeed_f[idx], wavespeed_f[mapP[idx]])
            if i == 1
                rhsL[1,k]   -= -.5*(flux_f[idx]+fluxP)-lambdaD*(utildeP-utilde_f[idx])
            elseif i == 2
                rhsL[end,k] -=  .5*(flux_f[idx]+fluxP)-lambdaD*(utildeP-utilde_f[idx])
            end
        end

        # Divide by mass
        for i = 1:Nq
            wq_i = LGLind[k] ? discrete_data_LGL.ops.wq[i] : discrete_data_gauss.ops.wq[i]
            wJq_i    = Jq[i,k]*wq_i
            rhsL[i,k] = rhsL[i,k]/wJq_i
        end
    end

    return dt
end