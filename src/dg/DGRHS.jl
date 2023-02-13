function rhs!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    init_get_rhs!(param,param.entropyproj_limiter_type,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    dt = get_rhs!(param.rhs_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    return dt
end

function init_get_rhs!(param,entropyproj_limiter_type::NoEntropyProjectionLimiter,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    if (nstage == 1)
        update_indicator!(prealloc,param.approximation_basis_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops)
    end
end

function init_get_rhs!(param,entropyproj_limiter_type::AdaptiveFilter,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    compute_modal_coefficients!(prealloc,param,discrete_data_gauss)
    compute_entropyproj_limiting_param!(param,discrete_data_gauss,prealloc,nstage)
    apply_entropyproj_filtering!(prealloc,param,param.entropyproj_limiter_type,discrete_data_gauss,nstage)
    if (nstage == 1)
        update_indicator!(prealloc,param.approximation_basis_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops)
    end
end

function init_get_rhs!(param,entropyproj_limiter_type::ScaledExtrapolation,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    compute_entropyproj_limiting_param!(param,discrete_data_gauss,prealloc,nstage)
    if (nstage == 1)
        update_indicator!(prealloc,param.approximation_basis_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops)
    end
end

function get_rhs!(rhs_type::LowOrderPositivity,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    @unpack rhsL,rhsU = prealloc
    dt = rhs_pos_Gauss!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,t,dt,nstage,true)
    copyto!(rhsU,rhsL)
    return dt
end

function get_rhs!(rhs_type::EntropyStable,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    @unpack rhsH,rhsU = prealloc
    rhs_modalESDG!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,nstage,true)
    copyto!(rhsU,rhsH)
    return dt
end

function get_rhs!(rhs_type::ESLimitedLowOrderPos,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    @unpack rhsH,rhsL,rhsU = prealloc
    entropy_projection!(prealloc,param,param.entropyproj_limiter_type,discrete_data_gauss,discrete_data_LGL,nstage)
    dt = rhs_pos_Gauss!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,t,dt,nstage,false)
    rhs_modalESDG!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,nstage,false)
    apply_positivity_limiter!(prealloc,param,discrete_data_gauss,discrete_data_LGL,dt,nstage,param.positivity_limiter_type)
    return dt
end

# TODO: dispatch on element type instead of the passed in discrete data
# TODO: refactor with NodewiseScaledExtrapolation
function entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data,prealloc)
    @unpack VhPq_new,Vf_new   = prealloc
    @unpack Nh,Nq,Nfp         = discrete_data.sizes
    @unpack Vf,Vf_low,Pq,VhPq = discrete_data.ops

    calculate_entropy_var!(vq_k,Uq_k,param,discrete_data)    # TODO: move calculating entropy var out of entropy projection
    # For nodal collocation, quad values are the same
    for i = 1:Nq
        entropy_projection_volume_node!(v_tilde_k,u_tilde_k,vq_k,Uq_k,i,param,discrete_data,prealloc)
    end
    for i = 1:Nfp
        entropy_projection_face_node!(v_tilde_k,u_tilde_k,vq_k,i,l_k,param,discrete_data,prealloc)
    end
end

function calculate_entropy_var!(vq_k,Uq_k,param,discrete_data)
    @unpack Nq = discrete_data.sizes
    for i = 1:Nq
        vq_k[i] = v_ufun(param.equation,Uq_k[i])
    end
end

function entropy_projection_volume_node!(v_tilde_k,u_tilde_k,vq_k,Uq_k,i,param,discrete_data,prealloc)
    v_tilde_k[i] = vq_k[i]
    u_tilde_k[i] = Uq_k[i]
end

function entropy_projection_face_node!(v_tilde_k,u_tilde_k,vq_k,i,l_k_i,param,discrete_data,prealloc)
    @unpack Nh,Nq     = discrete_data.sizes
    @unpack Vf,Vf_low = discrete_data.ops
    @unpack Vf_new    = prealloc
    if (l_k_i != 1.0)   # TODO: require l_k ∈ [0,1]
        @views Vf_new[i,:] = l_k_i*Vf[i,:]+(1-l_k_i)*Vf_low[i,:]   # TODO: Vf_new only allocate 1D vector instead of 2D matrix?
        # TODO: v_tilde_k[i+Nq] = @views sum(Vf_new[i,:].*vq_k)
        #       requires allocation... why?
        v_tilde_k[i+Nq] = zero(v_tilde_k[i+Nq])
        for j = 1:Nq
            v_tilde_k[i+Nq] += Vf_new[i,j]*vq_k[j]
        end
    else
        # Nothing is applied if l_k == 1
        # TODO: v_tilde_k[i+Nq] = @views sum(Vf[i,:].*vq_k)
        #       requires allocation... why?
        v_tilde_k[i+Nq] = zero(v_tilde_k[i+Nq])
        for j = 1:Nq
            v_tilde_k[i+Nq] += Vf[i,j]*vq_k[j]
        end
    end
    u_tilde_k[i+Nq] = u_vfun(param.equation,v_tilde_k[i+Nq])
end

# TODO: ugly dispatch
function entropy_projection!(prealloc,param,entropyproj_limiter_type::Union{AdaptiveFilter,NoEntropyProjectionLimiter},discrete_data_gauss,discrete_data_LGL,nstage)
    @unpack Uq,vq,v_tilde,u_tilde,LGLind,L_Vf_arr,Vf_new,VhPq_new = prealloc
    @unpack Nh,Nq,Nfp                                             = discrete_data_gauss.sizes
    K = get_num_elements(param)
    
    for k = 1:K
        vq_k      = view(vq,:,k)
        v_tilde_k = view(v_tilde,:,k)
        u_tilde_k = view(u_tilde,:,k)
        Uq_k      = view(Uq,:,k)
        l_k       = 1.0
        # TODO: we can skip LGL instead of applying identity
        discrete_data = (LGLind[k]) ? discrete_data_LGL : discrete_data_gauss
        entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data,prealloc)
    end
end

# TODO: ugly dispatch
function entropy_projection!(prealloc,param,entropyproj_limiter_type::ElementwiseScaledExtrapolation,discrete_data_gauss,discrete_data_LGL,nstage)
    @unpack Uq,vq,v_tilde,u_tilde,LGLind,L_Vf_arr,Vf_new,VhPq_new = prealloc
    @unpack Nh,Nq,Nfp                                             = discrete_data_gauss.sizes
    K = get_num_elements(param)
    
    for k = 1:K
        vq_k      = view(vq,:,k)
        v_tilde_k = view(v_tilde,:,k)
        u_tilde_k = view(u_tilde,:,k)
        Uq_k      = view(Uq,:,k)
        l_k       = prealloc.Farr[k,nstage]
        # TODO: we can skip LGL instead of applying identity
        discrete_data = (LGLind[k]) ? discrete_data_LGL : discrete_data_gauss
        entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data,prealloc)
    end
end

# TODO: ugly dispatch
function entropy_projection!(prealloc,param,entropyproj_limiter_type::NodewiseScaledExtrapolation,discrete_data_gauss,discrete_data_LGL,nstage)
    @unpack Uq,vq,v_tilde,u_tilde,LGLind,L_Vf_arr,Vf_new,VhPq_new = prealloc
    @unpack Nh,Nq,Nfp                                             = discrete_data_gauss.sizes
    K = get_num_elements(param)
    
    for k = 1:K
        vq_k      = view(vq,:,k)
        v_tilde_k = view(v_tilde,:,k)
        u_tilde_k = view(u_tilde,:,k)
        Uq_k      = view(Uq,:,k)
        # TODO: we can skip LGL instead of applying identity
        if (LGLind[k])
            entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data_LGL,prealloc)
        else
            # TODO: refactor
            calculate_entropy_var!(vq_k,Uq_k,param,discrete_data_gauss)
            for i = 1:Nq
                entropy_projection_volume_node!(v_tilde_k,u_tilde_k,vq_k,Uq_k,i,param,discrete_data_gauss,prealloc)
            end
            for i = 1:Nfp
                l_k_i = prealloc.θ_local_arr[i,k,nstage]
                entropy_projection_face_node!(v_tilde_k,u_tilde_k,vq_k,i,l_k_i,param,discrete_data_gauss,prealloc)
            end
        end
    end
end

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
    calculate_interface_dissipation_coeff!(prealloc,param,bcdata,discrete_data_gauss,discrete_data_LGL,param.equation)
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

# TODO: refactor
function calculate_interface_dissipation_coeff!(prealloc,param,bcdata,discrete_data_gauss,discrete_data_LGL,equation::EquationType{Dim1})
    @unpack lam,LFc,u_tilde = prealloc
    @unpack equation        = param
    @unpack mapP            = bcdata

    # TODO: refactor
    K  = get_num_elements(param)
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

function calculate_interface_dissipation_coeff!(prealloc,param,bcdata,discrete_data_gauss,discrete_data_LGL,equation::EquationType{Dim2})
    @unpack lam,LFc,u_tilde,n_i,LGLind = prealloc
    @unpack equation                   = param
    @unpack mapP                       = bcdata
    @unpack rxJh,ryJh,sxJh,syJh        = discrete_data_gauss.geom   # TODO: hardcoded

    # TODO: refactor
    K  = get_num_elements(param)
    Nq = size(prealloc.Uq,1)
    Nh = size(prealloc.u_tilde,1)
    uf = @view u_tilde[Nq+1:Nh,:]
    rxJf,ryJf,sxJf,syJf = (x->view(x,Nq+1:Nh,:)).((rxJh,ryJh,sxJh,syJh))
    # Lax Friedrichs dissipation
    for k = 1:K
        Br = LGLind[k] ? discrete_data_LGL.ops.Br : discrete_data_gauss.ops.Br
        Bs = LGLind[k] ? discrete_data_LGL.ops.Bs : discrete_data_gauss.ops.Bs
        for i = 1:size(lam,1)
            Bx_i = rxJf[i,k]*Br[i,i]+sxJf[i,k]*Bs[i,i]      # TODO: refactor geometric factor calculation
            By_i = ryJf[i,k]*Br[i,i]+syJf[i,k]*Bs[i,i]
            n_i_norm = sqrt(Bx_i^2+By_i^2)
            n_i[1] = Bx_i/n_i_norm
            n_i[2] = By_i/n_i_norm
            lam[i,k] = wavespeed_davis_estimate(equation,uf[i,k],n_i)
        end
    end

    for k = 1:K
        Br = LGLind[k] ? discrete_data_LGL.ops.Br : discrete_data_gauss.ops.Br
        Bs = LGLind[k] ? discrete_data_LGL.ops.Bs : discrete_data_gauss.ops.Bs
        for i = 1:size(LFc,1)
            Bx_i = rxJf[i,k]*Br[i,i]+sxJf[i,k]*Bs[i,i]      # TODO: refactor geometric factor calculation
            By_i = ryJf[i,k]*Br[i,i]+syJf[i,k]*Bs[i,i]
            n_i_norm = sqrt(Bx_i^2+By_i^2)
            LFc[i,k] = 0.5*n_i_norm*max(lam[i,k],lam[mapP[i,k]])
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
                QFxyrs_ij = get_QFrs_ij(i,j,Ui,Uj,discrete_data,equation)
                accumulate_QF1!(prealloc,i,j,k,QFxyrs_ij,discrete_data,equation)
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

function get_QFrs_ij(i,j,Ui,Uj,discrete_data,equation::EquationType{Dim1})
    @unpack Srh_db = discrete_data.ops

    Srh_db_ij = Srh_db[i,j]
    QFxr_ij = Srh_db_ij*fS_prim_log(equation,Ui,Uj)

    return (QFxr_ij,)
end

function get_QFrs_ij(i,j,Ui,Uj,discrete_data,equation::EquationType{Dim2})
    @unpack Srh_db,Ssh_db = discrete_data.ops

    Srh_db_ij = Srh_db[i,j]
    Ssh_db_ij = Ssh_db[i,j]
    fx,fy = fS_prim_log(equation,Ui,Uj)
    QFxr_ij = Srh_db_ij*fx
    QFxs_ij = Ssh_db_ij*fx
    QFyr_ij = Srh_db_ij*fy
    QFys_ij = Ssh_db_ij*fy

    # dfx/dr,dfx/ds,dfy/dr,dfy/ds
    return (QFxr_ij,QFxs_ij,QFyr_ij,QFys_ij)
end

function accumulate_QF1!(prealloc,i,j,k,QFxyrs_ij,discrete_data,equation::EquationType{Dim1})
    @unpack QF1 = prealloc
    @unpack rxJh = discrete_data.geom

    QFxr_ij = QFxyrs_ij[1]
    QF1[i,k] += rxJh[i,k]*QFxr_ij
    QF1[j,k] -= rxJh[j,k]*QFxr_ij
end

function accumulate_QF1!(prealloc,i,j,k,QFxyrs_ij,discrete_data,equation::EquationType{Dim2})
    @unpack QF1 = prealloc
    @unpack rxJh,ryJh,sxJh,syJh = discrete_data.geom

    QFxr_ij,QFxs_ij,QFyr_ij,QFys_ij = QFxyrs_ij
    dfxdx(idx) = rxJh[idx,k]*QFxr_ij+sxJh[idx,k]*QFxs_ij
    dfydy(idx) = ryJh[idx,k]*QFyr_ij+syJh[idx,k]*QFys_ij
    dfxdx_i = dfxdx(i)
    dfydy_i = dfydy(i)
    dfxdx_j = dfxdx(j)
    dfydy_j = dfydy(j)
    QF1[i,k] += dfxdx_i+dfydy_i
    QF1[j,k] -= dfxdx_j+dfydy_j
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
        BF1[i,k] = (rxJf[i]*Br[i,i]+sxJf[i]*Bs[i,i])*fx+(ryJf[i]*Br[i,i]+syJf[i]*Bs[i,i])*fy
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


##########################################
### RHS of positivity preserving Gauss ###
##########################################
function rhs_pos_Gauss!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,t,dt,nstage,need_proj=true)
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
    # check_bar_states!(dt,prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata)

    return dt
end

function calculate_wavespeed_and_inviscid_flux!(prealloc,param)
    @unpack equation = param
    @unpack Uq,Uf,u_tilde,wavespeed,flux = prealloc
    
    K  = get_num_elements(param)
    Nq = size(Uq,1)
    Nh  = size(prealloc.u_tilde,1)
    Nfp = Nh-Nq
    for k = 1:K
        for i = 1:Nfp
            update_face_value!(prealloc,i,k,get_low_order_surface_flux(param.rhs_type))
        end
        for i = 1:size(u_tilde,1)
            ui = (i <= Nq) ? Uq[i,k] : Uf[i-Nq,k]
            wavespeed[i,k] = wavespeed_davis_estimate(equation,ui)
            flux[i,k]      = euler_fluxes(equation,ui)
        end
    end
end

# TODO: hardcoded for 1D
function update_face_value!(prealloc,i,k,surface_flux_type::LaxFriedrichsOnNodalVal)
    @unpack Uq,Uf = prealloc
    if (i == 1)
        Uf[i,k] = Uq[1,k]
    elseif (i == 2)
        Uf[i,k] = Uq[end,k]
    end
end

function update_face_value!(prealloc,i,k,surface_flux_type::LaxFriedrichsOnProjectedVal)
    @unpack u_tilde,Uf = prealloc
    Uf[i,k] = u_tilde[i+size(prealloc.Uq,1),k]
end

function calculate_low_order_CFL(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,t)
    @unpack CFL,dt0,T = param.timestepping_param
    @unpack Jq        = discrete_data_gauss.geom
    @unpack mapP      = bcdata
    @unpack αarr,LGLind,wavespeed,Uq,u_tilde = prealloc

    K  = get_num_elements(param)
    Nq  = size(prealloc.Uq,1)
    Nh  = size(prealloc.u_tilde,1)
    Nfp = Nh-Nq
    wavespeed_f = @view wavespeed[Nq+1:Nh,:]
    utilde_f    = @view u_tilde[Nq+1:Nh,:]
    dt = min(CFL*dt0,T-t)
    for k = 1:K
        αarr[1,k]   = find_alpha(param,Uq[1,k],utilde_f[1,k])
        αarr[end,k] = find_alpha(param,Uq[end,k],utilde_f[end,k])
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
                lambda_i += LGLind[k] ? get_lambda_B(prealloc,mapP,1,k) : get_lambda_B(prealloc,mapP,1,k,get_low_order_surface_flux(param.rhs_type))
            end
            if i == Nq
                lambda_i += LGLind[k] ? get_lambda_B(prealloc,mapP,2,k) : get_lambda_B(prealloc,mapP,2,k,get_low_order_surface_flux(param.rhs_type))
            end

            dt = min(dt, CFL*.5*wJq_i/lambda_i)
        end
    end
    return dt
end

# TODO: refactor with bisection
# Find alpha s.t. alpha*ui - uitilde >= 0
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

function get_lambda_B(prealloc,mapP,i,k)
    @unpack wavespeed = prealloc
    Nq  = size(prealloc.Uq,1)
    Nh  = size(prealloc.u_tilde,1)
    Nfp = size(mapP,1)
    wavespeed_f = @view wavespeed[Nq+1:Nh,:]
    return .5*max(wavespeed_f[i,k],wavespeed_f[mapP[i+(k-1)*Nfp]])
end

function get_lambda_B(prealloc,mapP,i,k,surface_flux_type::LaxFriedrichsOnNodalVal)
    return get_lambda_B(prealloc,mapP,i,k) 
end

function get_lambda_B(prealloc,mapP,i,k,surface_flux_type::LaxFriedrichsOnProjectedVal)
    @unpack αarr,wavespeed = prealloc
    Nq  = size(prealloc.Uq,1)
    Nh  = size(prealloc.u_tilde,1)
    Nfp = size(mapP,1)
    wavespeed_f = @view wavespeed[Nq+1:Nh,:]
    return αarr[i,k]*.5*max(wavespeed_f[i,k],wavespeed_f[mapP[i+(k-1)*Nfp]]) + .5*wavespeed[i,k]
end

function clear_low_order_rhs!(prealloc,param)
    K  = get_num_elements(param)
    rhsL = prealloc.rhsL
    # Assemble RHS
    for k = 1:K
        for i = 1:size(rhsL,1)
            rhsL[i,k] = zero(SVector{3,Float64})
        end
    end
end

function accumulate_low_order_rhs_volume!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    @unpack Uq,rhsL,flux,wavespeed,LGLind = prealloc
    
    K  = get_num_elements(param)
    Nq = size(Uq,1)
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
    end
end

function accumulate_low_order_rhs_surface!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata)
    @unpack equation = param
    @unpack Uq,Uf,rhsL,flux,flux_L,wavespeed,LGLind,u_tilde = prealloc
    @unpack mapP,mapI,mapO,inflowarr                     = bcdata

    K  = get_num_elements(param)
    Nq  = size(Uq,1)
    Nh  = size(u_tilde,1)
    Nfp = size(bcdata.mapP,1)
    flux_f      = @view flux[Nq+1:Nh,:]
    wavespeed_f = @view wavespeed[Nq+1:Nh,:]
    for k = 1:K
        # Surface contributions
        for i = 1:Nfp
            idx = i+Nfp*(k-1)
            Iidx = findfirst(x->(x==idx), mapI)
            Oidx = findfirst(x->(x==idx), mapO)
            fluxP   = !isnothing(Iidx) ? euler_fluxes(equation,inflowarr[Iidx]) : flux_f[mapP[idx]]
            uP      = !isnothing(Iidx) ? inflowarr[Iidx] : Uf[mapP[idx]]
            lambdaD = (!isnothing(Iidx) || !isnothing(Oidx)) ? 0.0 : .5*max(wavespeed_f[idx], wavespeed_f[mapP[idx]])
            # TODO: hardcoded scale by normal
            if i == 1
                flux_L[i,k] = -.5*(flux_f[idx]+fluxP)-lambdaD*(uP-Uf[idx])
                rhsL[1,k]  -= flux_L[i,k] 
            elseif i == 2
                flux_L[i,k]  = .5*(flux_f[idx]+fluxP)-lambdaD*(uP-Uf[idx])
                rhsL[end,k] -= flux_L[i,k]
            end
        end
    end
end

function scale_low_order_rhs_by_mass!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    @unpack rhsL,LGLind = prealloc
    @unpack Jq = discrete_data_gauss.geom

    K  = get_num_elements(param)
    for k = 1:K
        # Divide by mass
        for i = 1:size(rhsL,1)
            wq_i = LGLind[k] ? discrete_data_LGL.ops.wq[i] : discrete_data_gauss.ops.wq[i]
            wJq_i    = Jq[i,k]*wq_i
            rhsL[i,k] = rhsL[i,k]/wJq_i
        end
    end
end

function check_bar_states!(dt,prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata)
    @unpack Uq,rhsL,flux,flux_L,wavespeed,LGLind,u_tilde = prealloc
    @unpack Jq = discrete_data_gauss.geom
    @unpack mapP = bcdata

    K  = get_num_elements(param)
    Nq  = size(Uq,1)
    Nh  = size(u_tilde,1)
    Nfp = size(bcdata.mapP,1)
    flux_f      = @view flux[Nq+1:Nh,:]
    wavespeed_f = @view wavespeed[Nq+1:Nh,:]
    for k = 1:K
        for i = 1:Nq
            rhsL_i = zero(Uq[i,k])
            rhsL_bar_i = zero(Uq[i,k])
            lambda_i = 0.0
            wq_i  = LGLind[k] ? discrete_data_LGL.ops.wq[i] : discrete_data_gauss.ops.wq[i]
            wJq_i = Jq[i,k]*wq_i

            # Volume
            for j = 1:Nq
                Sr0_ij = LGLind[k] ? discrete_data_LGL.ops.Sr0[i,j] : discrete_data_gauss.ops.Sr0[i,j]
                if (Sr0_ij != 0)
                    lambda_ij = abs(Sr0_ij)*max(wavespeed[i,k], wavespeed[j,k])
                    lambda_i += lambda_ij
                    rhsL_i -= (Sr0_ij*(flux[j,k]+flux[i,k])-lambda_ij*(Uq[j,k]-Uq[i,k]))/wJq_i
                end
            end
            # boundary 
            if i == 1
                lambda_ij = .5*max(wavespeed[1,k],wavespeed[Nq,mod1(k-1,K)])
                lambda_i += lambda_ij
                rhsL_i -= (-.5*(flux[1,k]+flux[Nq,mod1(k-1,K)])-lambda_ij*(Uq[Nq,mod1(k-1,K)]-Uq[1,k]))/wJq_i
            end
            if i == Nq
                lambda_ij = .5*max(wavespeed[Nq,k],wavespeed[1,mod1(k+1,K)])
                lambda_i += lambda_ij
                rhsL_i -= (.5*(flux[Nq,k]+flux[1,mod1(k+1,K)])-lambda_ij*(Uq[1,mod1(k+1,K)]-Uq[Nq,k]))/wJq_i
            end

            # Using bar states
            for j = 1:Nq
                Sr0_ij = discrete_data_gauss.ops.Sr0[i,j]
                if (Sr0_ij != 0)
                    lambda_ij = abs(Sr0_ij)*max(wavespeed[i,k], wavespeed[j,k])
                    ubar_ij = .5*(Uq[i,k]+Uq[j,k])-.5*Sr0_ij/lambda_ij*(flux[j,k]-flux[i,k])
                    is_positive,_,_ = check_positivity_node(ubar_ij,param)
                    if !is_positive
                        @show k,i,j
                    end
                    rhsL_bar_i += 2*lambda_ij*ubar_ij/wJq_i
                    rhsL_bar_i -= 2*lambda_ij*Uq[i,k]/wJq_i
                end
            end
            if i == 1
                Br0_i = -.5
                lambda_ij = .5*max(wavespeed[1,k],wavespeed[Nq,mod1(k-1,K)])
                ubar_ij = .5*(Uq[1,k]+Uq[Nq,mod1(k-1,K)])-.5*Br0_i/lambda_ij*(flux[Nq,mod1(k-1,K)]-flux[1,k])
                is_positive,_,_ = check_positivity_node(ubar_ij,param)
                if !is_positive
                    @show k,i,lambda_ij
                end
                rhsL_bar_i += 2*lambda_ij*ubar_ij/wJq_i
                rhsL_bar_i -= 2*lambda_ij*Uq[1,k]/wJq_i
            end
            if i == Nq
                Br0_i = .5
                lambda_ij = .5*max(wavespeed[Nq,k],wavespeed[1,mod1(k+1,K)])
                ubar_ij = .5*(Uq[Nq,k]+Uq[1,mod1(k+1,K)])-.5*Br0_i/lambda_ij*(flux[1,mod1(k+1,K)]-flux[Nq,k])
                ubar_ij_L = .5*Uq[Nq,k]+.5*Br0_i/lambda_ij*flux[Nq,k]
                ubar_L    = 2*lambda_ij*Uq[Nq,k] + flux[Nq,k]
                ubar_ij_R = .5*Uq[1,mod1(k+1,K)]-.5*Br0_i/lambda_ij*flux[1,mod1(k+1,K)]
                check_positivity_node(ubar_ij,param)
                check_positivity_node(ubar_ij_L,param)
                check_positivity_node(ubar_ij_R,param)
                rhsL_bar_i += 2*lambda_ij*ubar_ij/wJq_i
                rhsL_bar_i -= 2*lambda_ij*Uq[end,k]/wJq_i
            end

            if (norm(rhsL_i-rhsL_bar_i, Inf) > 1e-6)
                @show k,i,rhsL_i-rhsL_bar_i
            end

            if (norm(rhsL_i-rhsL[i,k]) > 1e-10)
                @show k,i,rhsL_i-rhsL[i,k]
            end
        end
    end
end