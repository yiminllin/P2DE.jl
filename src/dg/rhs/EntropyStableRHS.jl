#########################
### RHS of modal ESDG ###
#########################
function rhs_modalESDG!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,nstage,need_proj=true)
    @unpack entropyproj_limiter_type = param
    @unpack J = discrete_data_gauss.geom

    if (need_proj)
        entropy_projection!(prealloc,param,entropyproj_limiter_type,discrete_data_gauss,discrete_data_LGL,nstage)
    end

    calculate_primitive_variables!(prealloc,param,bcdata)
    calculate_interface_dissipation_coeff!(prealloc,param,bcdata,discrete_data_gauss,discrete_data_LGL)
    enforce_BC!(prealloc,param,bcdata)

    # Flux differencing
    clear_flux_differencing_cache!(prealloc)
    flux_differencing_volume!(prealloc,param,discrete_data_LGL,discrete_data_gauss)
    flux_differencing_surface!(prealloc,param,discrete_data_LGL,discrete_data_gauss)

    # Assemble RHS
    assemble_rhs!(prealloc,param,discrete_data_gauss,discrete_data_LGL,nstage,J)
end

function calculate_primitive_variables!(prealloc,param,bcdata)
    @unpack equation   = param
    @unpack mapP       = bcdata
    @unpack beta,rholog,betalog,u_tilde,uP,betaP,rhologP,betalogP = prealloc
    K = get_num_elements(param)

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

function calculate_interface_dissipation_coeff!(prealloc,param,bcdata,discrete_data_gauss,discrete_data_LGL)
    @unpack lam,LFc,u_tilde,LGLind = prealloc
    @unpack equation               = param
    @unpack mapP                   = bcdata

    # TODO: refactor
    K  = get_num_elements(param)
    Nq = size(prealloc.Uq,1)
    Nh = size(prealloc.u_tilde,1)
    Nfp = Nh-Nq
    uf = @view u_tilde[Nq+1:Nh,:]
    dim = get_dim_type(equation)

    # Lax Friedrichs dissipation
    for k = 1:K
        discrete_data = LGLind[k] ? discrete_data_LGL : discrete_data_gauss
        for i = 1:Nfp
            Bxy_i,n_i_norm = get_Bx_with_n(i,k,discrete_data,dim)
            n_i = Bxy_i./n_i_norm 
            lam[i,k] = wavespeed_davis_estimate(equation,uf[i,k],n_i)
            LFc[i,k] = .5*n_i_norm
        end
    end

    for k = 1:K
        discrete_data = LGLind[k] ? discrete_data_LGL : discrete_data_gauss
        for i = 1:Nfp
            LFc[i,k] = LFc[i,k]*max(lam[i,k],lam[mapP[i,k]])
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
        uP[io]       = prealloc.Uq[end,end]
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

# TODO: Assume uniform mesh, Need geom for LGL
function flux_differencing_volume!(prealloc,param,discrete_data_LGL,discrete_data_gauss)
    @unpack equation = param
    @unpack QF1,u_tilde,beta,rholog,betalog,Ui,Uj,LGLind = prealloc
    @unpack rxJh = discrete_data_gauss.geom

    K  = get_num_elements(param)
    Nh = size(u_tilde,1)
    for k = 1:K
        discrete_data = LGLind[k] ? discrete_data_LGL : discrete_data_gauss
        for j = 1:Nh
            accumulate_U_beta!(Uj,j,k,prealloc,param.equation)
            for i = j+1:Nh
                accumulate_U_beta!(Ui,i,k,prealloc,param.equation)
                accumulate_QF1!(prealloc,i,Ui,j,Uj,k,discrete_data,equation)
            end
        end
    end
end

# Accumulate U with (utilde, beta, log(rho), log(beta)) at index i,element k
function accumulate_U_beta!(U,idx,k,prealloc,equation::EquationType{Dim1})
    @unpack u_tilde,beta,rholog,betalog = prealloc

    U[1] = u_tilde[idx,k][1]
    U[2] = u_tilde[idx,k][2]/U[1]
    U[3] = beta[idx,k]
    U[4] = rholog[idx,k]
    U[5] = betalog[idx,k]
end

function accumulate_U_beta!(U,idx,k,prealloc,equation::EquationType{Dim2})
    @unpack u_tilde,beta,rholog,betalog = prealloc

    U[1] = u_tilde[idx,k][1]
    U[2] = u_tilde[idx,k][2]/U[1]
    U[3] = u_tilde[idx,k][3]/U[1]
    U[4] = beta[idx,k]
    U[5] = rholog[idx,k]
    U[6] = betalog[idx,k]
end

function accumulate_QF1!(prealloc,i,Ui,j,Uj,k,discrete_data,equation::EquationType{Dim1})
    @unpack QF1 = prealloc
    @unpack rxJh = discrete_data.geom

    fx = fS_prim_log(equation,Ui,Uj)
    Sxh_db_ij = get_Sx(i,j,k,discrete_data,Dim1())
    Sxh_db_ji = get_Sx(j,i,k,discrete_data,Dim1())
    QF1[i,k] += Sxh_db_ij[1]*fx
    QF1[j,k] += Sxh_db_ji[1]*fx
end

function accumulate_QF1!(prealloc,i,Ui,j,Uj,k,discrete_data,equation::EquationType{Dim2})
    @unpack QF1 = prealloc

    fx,fy = fS_prim_log(equation,Ui,Uj)
    Sxh_db_ij,Syh_db_ij = get_Sx(i,j,k,discrete_data,Dim2())
    Sxh_db_ji,Syh_db_ji = get_Sx(j,i,k,discrete_data,Dim2())
    QF1[i,k] += Sxh_db_ij*fx+Syh_db_ij*fy
    QF1[j,k] += Sxh_db_ji*fx+Syh_db_ji*fy
end

function flux_differencing_surface!(prealloc,param,discrete_data_LGL,discrete_data_gauss)
    @unpack LGLind   = prealloc
    @unpack equation = param
    @unpack BF1,u_tilde,uP,LFc,flux_H = prealloc

    K  = get_num_elements(param)
    for k  = 1:K
        discrete_data = (LGLind[k]) ? discrete_data_LGL : discrete_data_gauss
        accumulate_BF1!(prealloc,k,param,discrete_data,equation)
        apply_LF_dissipation!(prealloc,k)
    end
end

function accumulate_BF1!(prealloc,k,param,discrete_data,equation::EquationType{Dim1})
    @unpack BF1,flux_H = prealloc

    # Boundary contributions (B F)1
    Nfp = size(BF1,1)
    for i = 1:Nfp
        BF1[i,k] = evaluate_high_order_surface_flux(prealloc,param,i,k,equation,get_high_order_surface_flux(param.rhs_type))
        flux_H[i,k] = BF1[i,k]   # TODO: refactor
    end
    BF1[1,k] = -BF1[1,k]     # TODO: hardcode scale by normal
    flux_H[1,k] = -flux_H[1,k]
end

function accumulate_BF1!(prealloc,k,param,discrete_data,equation::EquationType{Dim2})
    @unpack Br,Bs      = discrete_data.ops
    @unpack BF1,flux_H = prealloc
    @unpack rxJh,ryJh,sxJh,syJh = discrete_data.geom
    
    # Boundary contributions (B F)1
    Nfp = size(BF1,1)
    Nq  = size(prealloc.Uq,1)
    Nh  = size(rxJh,1)
    rxJf,ryJf,sxJf,syJf = (x->view(x,Nq+1:Nh,k)).((rxJh,ryJh,sxJh,syJh))
    for i = 1:Nfp
        fx,fy = evaluate_high_order_surface_flux(prealloc,param,i,k,equation,get_high_order_surface_flux(param.rhs_type))
        Bx_i,By_i = get_Bx(i,k,discrete_data,Dim2())     # TODO: redundant calculation
        BF1[i,k] = Bx_i*fx+By_i*fy
        flux_H[i,k] = BF1[i,k]
    end
end

function evaluate_high_order_surface_flux(prealloc,param,i,k,equation::EquationType{Dim1},surface_flux_type::ChandrashekarOnProjectedVal)
    @unpack equation                                              = param
    @unpack u_tilde,beta,rholog,betalog,uP,betaP,rhologP,betalogP = prealloc
    Nq = size(prealloc.Uq,1)
    Nh = size(u_tilde,1)
    uf       = @view u_tilde[Nq+1:Nh,:]
    betaf    = @view beta[Nq+1:Nh,:]
    rhologf  = @view rholog[Nq+1:Nh,:]
    betalogf = @view betalog[Nq+1:Nh,:]
    return fS_prim_log(equation,(uf[i,k][1],uf[i,k][2]/uf[i,k][1],betaf[i,k],rhologf[i,k],betalogf[i,k]),
                                (uP[i,k][1],uP[i,k][2]/uP[i,k][1],betaP[i,k],rhologP[i,k],betalogP[i,k]))
end

function evaluate_high_order_surface_flux(prealloc,param,i,k,equation::EquationType{Dim2},surface_flux_type::ChandrashekarOnProjectedVal)
    @unpack equation                                              = param
    @unpack u_tilde,beta,rholog,betalog,uP,betaP,rhologP,betalogP = prealloc
    Nq = size(prealloc.Uq,1)
    Nh = size(u_tilde,1)
    uf       = @view u_tilde[Nq+1:Nh,:]
    betaf    = @view beta[Nq+1:Nh,:]
    rhologf  = @view rholog[Nq+1:Nh,:]
    betalogf = @view betalog[Nq+1:Nh,:]
    return fS_prim_log(equation,(uf[i,k][1],uf[i,k][2]/uf[i,k][1],uf[i,k][3]/uf[i,k][1],betaf[i,k],rhologf[i,k],betalogf[i,k]),
                                (uP[i,k][1],uP[i,k][2]/uP[i,k][1],uP[i,k][3]/uP[i,k][1],betaP[i,k],rhologP[i,k],betalogP[i,k]))
end

function evaluate_high_order_surface_flux(prealloc,param,i,k,equation::EquationType{Dim1},surface_flux_type::LaxFriedrichsOnProjectedVal)
    @unpack equation   = param
    @unpack uP,u_tilde = prealloc
    Nq = size(prealloc.Uq,1)
    Nh = size(u_tilde,1)
    uf = @view u_tilde[Nq+1:Nh,:]
    return .5*(euler_fluxes(equation,uf[i,k])+euler_fluxes(equation,uP[i,k]))
end

function evaluate_high_order_surface_flux(prealloc,param,i,k,equation::EquationType{Dim2},surface_flux_type::LaxFriedrichsOnProjectedVal)
    @unpack equation   = param
    @unpack uP,u_tilde = prealloc
    Nq = size(prealloc.Uq,1)
    Nh = size(u_tilde,1)
    uf = @view u_tilde[Nq+1:Nh,:]
    fxf,fyf = euler_fluxes(equation,uf[i,k])
    fxP,fyP = euler_fluxes(equation,uP[i,k])
    return (.5*(fxf+fxP),.5*(fyf+fyP))
end

function apply_LF_dissipation!(prealloc,k)
    @unpack BF1,flux_H,LFc,uP,u_tilde = prealloc

    # LF dissipation
    Nq = size(prealloc.Uq,1)
    Nh = size(u_tilde,1)
    Nfp = size(BF1,1)
    uf = @view u_tilde[Nq+1:Nh,:]
    for i = 1:Nfp
        lf = LFc[i,k]*(uP[i,k]-uf[i,k])
        BF1[i,k] -= lf
        flux_H[i,k] -= lf   # TODO: refactor
    end
end

function project_flux_difference_to_quad!(prealloc,param,entropyproj_limiter_type::Union{AdaptiveFilter,NoEntropyProjectionLimiter},discrete_data_gauss,k,nstage)
    @unpack spatial,boundary,QF1,BF1 = prealloc
    @views mul!(spatial[:,k],discrete_data_gauss.ops.MinvVhT,QF1[:,k])
    @views mul!(boundary[:,k],discrete_data_gauss.ops.MinvVfT,BF1[:,k])
end

function project_flux_difference_to_quad!(prealloc,param,entropyproj_limiter_type::ScaledExtrapolation,discrete_data_gauss,k,nstage)
    @unpack spatial,boundary,QF1,BF1,Vf_new,VhT_new,MinvVhT_new = prealloc
    @unpack Nq,Nh = discrete_data_gauss.sizes

    update_limited_extrapolation!(prealloc,param,param.entropyproj_limiter_type,discrete_data_gauss,k,nstage)
    VqT_new = @views VhT_new[:,1:Nq]
    VfT_new = @views VhT_new[:,Nq+1:Nh]
    VqT_new .= transpose(discrete_data_gauss.ops.Vq)
    VfT_new .= transpose(Vf_new)
    @. MinvVhT_new = (1/discrete_data_gauss.ops.wq)*VhT_new
    MinvVfT_new = @views MinvVhT_new[:,Nq+1:Nh]
    @views mul!(spatial[:,k],MinvVhT_new,QF1[:,k])
    @views mul!(boundary[:,k],MinvVfT_new,BF1[:,k])
end

function update_limited_extrapolation!(prealloc,param,entropyproj_limiter_type::ElementwiseScaledExtrapolation,discrete_data_gauss,k,nstage)
    @unpack Vf_new    = prealloc
    @unpack Vf,Vf_low = discrete_data_gauss.ops
    l_k = prealloc.Farr[k,nstage]
    @. Vf_new = l_k*Vf+(1.0-l_k)*Vf_low
end

function update_limited_extrapolation!(prealloc,param,entropyproj_limiter_type::NodewiseScaledExtrapolation,discrete_data_gauss,k,nstage)
    @unpack Vf_new    = prealloc
    @unpack Vf,Vf_low = discrete_data_gauss.ops
    l_k = prealloc.Farr[k,nstage]
    for i = 1:discrete_data_gauss.sizes.Nfp
        l_k_i = prealloc.θ_local_arr[i,k,nstage]
        @views Vf_new[i,:] = l_k_i*Vf[i,:]+(1-l_k_i)*Vf_low[i,:]
    end
end

# TODO: dispatch
function assemble_rhs!(prealloc,param,discrete_data_gauss,discrete_data_LGL,nstage,J)
    @unpack spatial,boundary,QF1,BF1,LGLind,rhsH = prealloc

    K  = get_num_elements(param)
    # Assemble RHS
    for k = 1:K
        if LGLind[k]
            @views mul!(spatial[:,k],discrete_data_LGL.ops.MinvVhT,QF1[:,k])
            @views mul!(boundary[:,k],discrete_data_LGL.ops.MinvVfT,BF1[:,k])
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

