# TODO: dispatch on element type instead of the passed in discrete data
function entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data,prealloc)
    @unpack VhPq_new,Vf_new   = prealloc
    @unpack Nh,Nq             = discrete_data.sizes
    @unpack Vf,Vf_low,Pq,VhPq = discrete_data.ops

    for i = 1:Nq
        vq_k[i] = v_ufun(param.equation,Uq_k[i])
    end
    # TODO: use dispatch for this if-else?
    if (l_k == 1.0)
        mul!(v_tilde_k,VhPq,vq_k)
    else
        # TODO: precompute VfPq and VflowPq, and simply take linear combination
        @. Vf_new = l_k*Vf+(1.0-l_k)*Vf_low
        VfPq_new = @views VhPq_new[Nq+1:Nh,:]
        mul!(VfPq_new,Vf_new,Pq)
        mul!(v_tilde_k,VhPq_new,vq_k)
    end
    for i = 1:Nh
        u_tilde_k[i] = u_vfun(param.equation,v_tilde_k[i])
    end
end

# TODO: ugly dispatch
function entropy_projection!(prealloc,param,entropyproj_limiter_type::Union{AdaptiveFilter,NoEntropyProjectionLimiter},discrete_data_gauss,discrete_data_LGL,nstage)
    @unpack K                                                     = param
    @unpack Uq,vq,v_tilde,u_tilde,LGLind,L_Vf_arr,Vf_new,VhPq_new = prealloc
    @unpack Nh,Nq,Nfp                                             = discrete_data_gauss.sizes
    
    for k = 1:K
        vq_k      = view(vq,:,k)
        v_tilde_k = view(v_tilde,:,k)
        u_tilde_k = view(u_tilde,:,k)
        Uq_k      = view(Uq,:,k)
        l_k       = 1.0
        # TODO: we can skip LGL instead of applying identity
        if (LGLind[k])
            entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data_LGL,prealloc)
        else
            entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data_gauss,prealloc)
        end
    end
end

# TODO: ugly dispatch
function entropy_projection!(prealloc,param,entropyproj_limiter_type::ScaledExtrapolation,discrete_data_gauss,discrete_data_LGL,nstage)
    @unpack K = param
    @unpack Uq,vq,v_tilde,u_tilde,LGLind,L_Vf_arr,Vf_new,VhPq_new = prealloc
    @unpack Nh,Nq,Nfp                                             = discrete_data_gauss.sizes
    
    for k = 1:K
        vq_k      = view(vq,:,k)
        v_tilde_k = view(v_tilde,:,k)
        u_tilde_k = view(u_tilde,:,k)
        Uq_k      = view(Uq,:,k)
        l_k       = Farr[k,nstage]
        # TODO: we can skip LGL instead of applying identity
        if (LGLind[k])
            entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data_LGL,prealloc)
        else
            entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data_gauss,prealloc)
        end
    end
end

#########################
### RHS of modal ESDG ###
#########################
function rhs_modalESDG!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,nstage,need_proj=true)
    @unpack entropyproj_limiter_type = param
    # TODO: Assume uniform mesh, Need geom for LGL
    @unpack J,rxJh      = discrete_data_gauss.geom

    if (need_proj)
        entropy_projection!(prealloc,param,entropyproj_limiter_type,discrete_data_gauss,discrete_data_LGL,nstage)
    end

    calculate_primitive_variables!(prealloc,param,bcdata)
    calculate_interface_dissipation_coeff!(prealloc,param,bcdata)
    enforce_BC!(prealloc,param,bcdata)

    # Flux differencing
    clear_flux_differencing_cache!(prealloc)
    flux_differencing_volume!(prealloc,param,discrete_data_LGL,discrete_data_gauss,rxJh)
    flux_differencing_surface!(prealloc,param)

    # Assemble RHS
    assemble_rhs!(prealloc,param,discrete_data_gauss,discrete_data_LGL,nstage,J)
end

function calculate_primitive_variables!(prealloc,param,bcdata)
    @unpack K,equation = param
    @unpack mapP       = bcdata
    @unpack beta,rholog,betalog,u_tilde,uP,betaP,rhologP,betalogP = prealloc
    for k = 1:K
        for i = 1:size(beta,1)
            beta[i,k]    = betafun(equation,u_tilde[i,k])
            rholog[i,k]  = log(u_tilde[i,k][1])
            betalog[i,k] = log(beta[i,k])
        end
    end

    # Boundary contributions
    # TODO: refactor
    Nq = size(prealloc.Uq,1)
    Nh = size(u_tilde,1)
    uf       = @view u_tilde[Nq+1:Nh,:]
    betaf    = @view beta[Nq+1:Nh,:]
    rhologf  = @view rholog[Nq+1:Nh,:]
    betalogf = @view betalog[Nq+1:Nh,:]
    for k = 1:K
        for i = 1:size(mapP,1)
            iP = mapP[i,k]
            uP[i,k]       = uf[iP]
            betaP[i,k]    = betaf[iP]
            rhologP[i,k]  = rhologf[iP]
            betalogP[i,k] = betalogf[iP]
        end
    end
end

function calculate_interface_dissipation_coeff!(prealloc,param,bcdata)
    @unpack lam,LFc,u_tilde = prealloc
    @unpack K,equation      = param
    @unpack mapP            = bcdata

    # TODO: refactor
    Nq = size(prealloc.Uq,1)
    Nh = size(prealloc.u_tilde,1)
    uf = @view u_tilde[Nq+1:Nh,:]
    # Lax Friedrichs dissipation
    for k = 1:K
        for i = 1:size(lam,1)
            lam[i,k] = wavespeed_davis_estimate(equation,uf[i,k])
        end
    end

    for k = 1:K
        for i = 1:size(LFc,1)
            LFc[i,k] = 0.5*max(lam[i,k],lam[mapP[i,k]])
        end
    end
end

function enforce_BC!(prealloc,param,bcdata)
    @unpack equation                      = param
    @unpack mapP,mapI,mapO,inflowarr      = bcdata
    @unpack LFc,uP,betaP,rhologP,betalogP = prealloc

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
        betaP[ii]    = betafun(equation,uP[ii])
        rhologP[ii]  = log(uP[ii][1])
        betalogP[ii] = log(betaP[ii])
    end

    # Enforce outflow BC
    for i = 1:size(mapO,1)
        io = mapO[i]
        # TODO: hardcoded
        uP[io]       = Uq[end,end]
        betaP[io]    = betafun(equation,uP[io])
        rhologP[io]  = log(uP[io][1])
        betalogP[io] = log(betaP[io])
    end
end

function clear_flux_differencing_cache!(prealloc)
    @unpack QF1,BF1 = prealloc
    @. QF1 = zero(QF1)
    @. BF1 = zero(BF1)
end

function flux_differencing_volume!(prealloc,param,discrete_data_LGL,discrete_data_gauss,rxJh)
    @unpack K,equation = param
    @unpack QF1,u_tilde,beta,rholog,betalog,Ui,Uj,LGLind = prealloc
    Nh = size(u_tilde,1)
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
                Qrh_skew_db_ij = (LGLind[k]) ? discrete_data_LGL.ops.Qrh_skew_db[i,j] : discrete_data_gauss.ops.Qrh_skew_db[i,j]
                QFij = Qrh_skew_db_ij*fS_prim_log(equation,Ui,Uj)
                QF1[i,k] += rxJh[i,k]*QFij
                QF1[j,k] -= rxJh[j,k]*QFij
            end
        end
    end
end

function flux_differencing_surface!(prealloc,param)
    @unpack equation           = param
    @unpack BF1,u_tilde,uP,LFc = prealloc

    Nq = size(prealloc.Uq,1)
    Nh = size(u_tilde,1)
    Nfp = size(BF1,1)
    uf = @view u_tilde[Nq+1:Nh,:]
    for k  = 1:param.K
        # Boundary contributions (B F)1
        for i = 1:Nfp
            # BF1[i,k] = fS_prim_log(equation,(uf[i,k][1],uf[i,k][2]/uf[i,k][1],betaf[i,k],rhologf[i,k],betalogf[i,k]),
            #                                 (uP[i,k][1],uP[i,k][2]/uP[i,k][1],betaP[i,k],rhologP[i,k],betalogP[i,k]))
            BF1[i,k] = .5*(euler_fluxes(equation,uf[i,k])+euler_fluxes(equation,uP[i,k]))
        end
        BF1[1,k] = -BF1[1,k]     # TODO: hardcode scale by normal
        # LF dissipation
        for i = 1:Nfp
            BF1[i,k] -= LFc[i,k]*(uP[i,k]-uf[i,k])
        end
    end
end

function project_flux_difference_to_quad!(prealloc,param,entropyproj_limiter_type::Union{AdaptiveFilter,NoEntropyProjectionLimiter},discrete_data_gauss,k,nstage)
    @unpack spatial,boundary,QF1,BF1 = prealloc
    @views mul!(spatial[:,k],discrete_data_gauss.ops.MinvVhT,QF1[:,k])
    @views mul!(boundary[:,k],discrete_data_gauss.ops.LIFT,BF1[:,k])
end

function project_flux_difference_to_quad!(prealloc,param,entropyproj_limiter_type::ScaledExtrapolation,discrete_data_gauss,k,nstage)
    @unpack spatial,boundary,QF1,BF1,Vf_new,VhT_new,MinvVhT_new = prealloc

    l_k = Farr[k,nstage]
    @. Vf_new = l_k*discrete_data_gauss.ops.Vf+(1.0-l_k)*discrete_data_gauss.ops.Vf_low
    VqT_new = @views VhT_new[:,1:Nq]
    VfT_new = @views VhT_new[:,Nq+1:Nh]
    VqT_new .= transpose(discrete_data_gauss.ops.Vq)
    VfT_new .= transpose(Vf_new)
    @. MinvVhT_new = (1/discrete_data_gauss.ops.wq)*VhT_new
    MinvVfT_new = @views MinvVhT_new[:,Nq+1:Nh]
    @views mul!(spatial[:,k],MinvVhT_new,QF1[:,k])
    @views mul!(boundary[:,k],MinvVfT_new,BF1[:,k])
end

# TODO: dispatch
function assemble_rhs!(prealloc,param,discrete_data_gauss,discrete_data_LGL,nstage,J)
    @unpack spatial,boundary,QF1,BF1,LGLind,rhsH = prealloc
    # Assemble RHS
    for k = 1:param.K
        if LGLind[k]
            @views mul!(spatial[:,k],discrete_data_LGL.ops.MinvVhT,QF1[:,k])
            @views mul!(boundary[:,k],discrete_data_LGL.ops.LIFT,BF1[:,k])
        else
            project_flux_difference_to_quad!(prealloc,param,param.entropyproj_limiter_type,discrete_data_gauss,k,nstage)
        end
        for i = 1:size(spatial,1)
            spatial[i,k] = -(spatial[i,k]+boundary[i,k])/J[i,k]
        end
        if LGLind[k]
            @views mul!(rhsH[:,k],discrete_data_LGL.ops.Vq,spatial[:,k])
        else
            @views mul!(rhsH[:,k],discrete_data_gauss.ops.Vq,spatial[:,k])
        end
    end
end


##########################################
### RHS of positivity preserving Gauss ###
##########################################
function rhs_pos_Gauss(param,discrete_data_gauss,discrete_data_LGL,bcdata,prealloc,t,dt,nstage,need_proj=true)
    @unpack entropyproj_limiter_type             = param
    @unpack mapP,mapI,mapO,inflowarr             = bcdata
    @unpack Jq                                   = discrete_data_gauss.geom
    @unpack Nh,Nq,Nfp                            = discrete_data_gauss.sizes
    @unpack u_tilde,flux,wavespeed,alphaarr,rhsL = prealloc
    @unpack αarr,LGLind                          = prealloc

    if (need_proj)
        entropy_projection!(prealloc,param,entropyproj_limiter_type,discrete_data_gauss,discrete_data_LGL,nstage)
    end
    calculate_wavespeed_and_inviscid_flux!(prealloc,param)

    # Determine positivity CFL
    if (nstage == 1)
        dt = calculate_low_order_CFL(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,t)
    end

    # Assemble RHS
    clear_low_order_rhs!(prealloc,param)
    # TODO: write routine in K loop?
    accumulate_low_order_rhs_volume!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    accumulate_low_order_rhs_surface!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata)
    scale_low_order_rhs_by_mass!(prealloc,param,discrete_data_gauss,discrete_data_LGL)

    return dt
end

function calculate_wavespeed_and_inviscid_flux!(prealloc,param)
    @unpack equation = param
    @unpack Uq,u_tilde,wavespeed,flux = prealloc
    for k = 1:param.K
        for i = 1:size(u_tilde,1)
            ui = (i <= size(Uq,1)) ? Uq[i,k] : u_tilde[i,k]
            wavespeed[i,k] = wavespeed_davis_estimate(equation,ui)
            flux[i,k]      = euler_fluxes(equation,ui)
        end
    end
end

function calculate_low_order_CFL(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,t)
    @unpack CFL,dt0,T = param.timestepping_param
    @unpack Jq        = discrete_data_gauss.geom
    @unpack mapP      = bcdata
    @unpack αarr,LGLind,wavespeed,Uq,u_tilde = prealloc

    Nq  = size(prealloc.Uq,1)
    Nh  = size(prealloc.u_tilde,1)
    Nfp = Nh-Nq
    wavespeed_f = @view wavespeed[Nq+1:Nh,:]
    dt = min(CFL*dt0,T-t)
    for k = 1:param.K
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
    return dt
end

# TODO: refactor with bisection
function find_alpha(param,ui,uitilde)
    @unpack equation = param
    POSTOL = param.global_constants.POSTOL
    alphaL = 0.0
    alphaR = 1.0
    substate = alphaR*ui-uitilde
    while (true)
        if (substate[1] > POSTOL && rhoe_ufun(equation,substate) > POSTOL)
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
        if (substate[1] > POSTOL && rhoe_ufun(equation,substate) > POSTOL)
            alphaR = alphaM
        else
            alphaL = alphaM
        end
        iter = iter + 1
    end

    return alphaR
end

function clear_low_order_rhs!(prealloc,param)
    rhsL = prealloc.rhsL
    # Assemble RHS
    for k = 1:param.K
        for i = 1:size(rhsL,1)
            rhsL[i,k] = zero(SVector{3,Float64})
        end
    end
end

function accumulate_low_order_rhs_volume!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    @unpack Uq,rhsL,flux,wavespeed,LGLind = prealloc
    Nq = size(Uq,1)
    for k = 1:param.K
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
    end
end

function accumulate_low_order_rhs_surface!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata)
    @unpack equation = param
    @unpack Uq,rhsL,flux,wavespeed,LGLind,u_tilde = prealloc
    @unpack mapP,mapI,mapO,inflowarr              = bcdata

    Nq  = size(Uq,1)
    Nh  = size(u_tilde,1)
    Nfp = size(bcdata.mapP,1)
    utilde_f    = @view u_tilde[Nq+1:Nh,:]
    flux_f      = @view flux[Nq+1:Nh,:]
    wavespeed_f = @view wavespeed[Nq+1:Nh,:]
    for k = 1:param.K
        # Surface contributions
        for i = 1:Nfp
            idx = i+Nfp*(k-1)
            Iidx = findfirst(x->(x==idx), mapI)
            Oidx = findfirst(x->(x==idx), mapO)
            fluxP   = !isnothing(Iidx) ? euler_fluxes(equation,inflowarr[Iidx]) : flux_f[mapP[idx]]
            utildeP = !isnothing(Iidx) ? inflowarr[Iidx] : utilde_f[mapP[idx]]
            lambdaD = (!isnothing(Iidx) || !isnothing(Oidx)) ? 0.0 : .5*max(wavespeed_f[idx], wavespeed_f[mapP[idx]])
            if i == 1
                rhsL[1,k]   -= -.5*(flux_f[idx]+fluxP)-lambdaD*(utildeP-utilde_f[idx])
            elseif i == 2
                rhsL[end,k] -=  .5*(flux_f[idx]+fluxP)-lambdaD*(utildeP-utilde_f[idx])
            end
        end
    end
end

function scale_low_order_rhs_by_mass!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    @unpack rhsL,LGLind = prealloc
    @unpack Jq = discrete_data_gauss.geom
    for k = 1:param.K
        # Divide by mass
        for i = 1:size(rhsL,1)
            wq_i = LGLind[k] ? discrete_data_LGL.ops.wq[i] : discrete_data_gauss.ops.wq[i]
            wJq_i    = Jq[i,k]*wq_i
            rhsL[i,k] = rhsL[i,k]/wJq_i
        end
    end
end