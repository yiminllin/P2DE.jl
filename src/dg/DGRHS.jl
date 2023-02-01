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
        v_tilde_k[i+Nq] = @views sum(Vf_new[i,:].*vq_k)
    else
        # Nothing is applied if l_k == 1
        v_tilde_k[i+Nq] = @views sum(Vf[i,:].*vq_k)
    end
    u_tilde_k[i+Nq] = u_vfun(param.equation,v_tilde_k[i+Nq])
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
function entropy_projection!(prealloc,param,entropyproj_limiter_type::ElementwiseScaledExtrapolation,discrete_data_gauss,discrete_data_LGL,nstage)
    @unpack K = param
    @unpack Uq,vq,v_tilde,u_tilde,LGLind,L_Vf_arr,Vf_new,VhPq_new = prealloc
    @unpack Nh,Nq,Nfp                                             = discrete_data_gauss.sizes
    
    for k = 1:K
        vq_k      = view(vq,:,k)
        v_tilde_k = view(v_tilde,:,k)
        u_tilde_k = view(u_tilde,:,k)
        Uq_k      = view(Uq,:,k)
        l_k       = prealloc.Farr[k,nstage]
        # TODO: we can skip LGL instead of applying identity
        if (LGLind[k])
            entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data_LGL,prealloc)
        else
            entropy_projection_element!(vq_k,v_tilde_k,u_tilde_k,Uq_k,l_k,param,discrete_data_gauss,prealloc)
        end
    end
end

# TODO: ugly dispatch
function entropy_projection!(prealloc,param,entropyproj_limiter_type::NodewiseScaledExtrapolation,discrete_data_gauss,discrete_data_LGL,nstage)
    @unpack K = param
    @unpack Uq,vq,v_tilde,u_tilde,LGLind,L_Vf_arr,Vf_new,VhPq_new = prealloc
    @unpack Nh,Nq,Nfp                                             = discrete_data_gauss.sizes
    
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
    @unpack BF1,u_tilde,uP,LFc,flux_H = prealloc

    Nq = size(prealloc.Uq,1)
    Nh = size(u_tilde,1)
    Nfp = size(BF1,1)
    uf = @view u_tilde[Nq+1:Nh,:]
    for k  = 1:param.K
        # Boundary contributions (B F)1
        for i = 1:Nfp
            BF1[i,k] = evaluate_high_order_surface_flux(prealloc,param,i,k,get_high_order_surface_flux(param.rhs_type))
            flux_H[i,k] = BF1[i,k]   # TODO: refactor
        end
        BF1[1,k] = -BF1[1,k]     # TODO: hardcode scale by normal
        flux_H[1,k] = -flux_H[1,k]
        # LF dissipation
        for i = 1:Nfp
            lf = LFc[i,k]*(uP[i,k]-uf[i,k])
            BF1[i,k] -= lf
            flux_H[i,k] -= lf   # TODO: refactor
        end
    end
end

function evaluate_high_order_surface_flux(prealloc,param,i,k,surface_flux_type::ChandrashekarOnProjectedVal)
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

function evaluate_high_order_surface_flux(prealloc,param,i,k,surface_flux_type::LaxFriedrichsOnProjectedVal)
    @unpack equation   = param
    @unpack uP,u_tilde = prealloc
    Nq = size(prealloc.Uq,1)
    Nh = size(u_tilde,1)
    uf = @view u_tilde[Nq+1:Nh,:]
    return .5*(euler_fluxes(equation,uf[i,k])+euler_fluxes(equation,uP[i,k]))
end

function project_flux_difference_to_quad!(prealloc,param,entropyproj_limiter_type::Union{AdaptiveFilter,NoEntropyProjectionLimiter},discrete_data_gauss,k,nstage)
    @unpack spatial,boundary,QF1,BF1 = prealloc
    @views mul!(spatial[:,k],discrete_data_gauss.ops.MinvVhT,QF1[:,k])
    @views mul!(boundary[:,k],discrete_data_gauss.ops.LIFT,BF1[:,k])
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
    Nq = size(Uq,1)
    for k = 1:param.K
        for i = 1:size(u_tilde,1)
            update_face_value!(prealloc,i-Nq,k,get_low_order_surface_flux(param.rhs_type))
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

    Nq  = size(prealloc.Uq,1)
    Nh  = size(prealloc.u_tilde,1)
    Nfp = Nh-Nq
    wavespeed_f = @view wavespeed[Nq+1:Nh,:]
    utilde_f    = @view u_tilde[Nq+1:Nh,:]
    dt = min(CFL*dt0,T-t)
    for k = 1:param.K
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
    @unpack Uq,Uf,rhsL,flux,flux_L,wavespeed,LGLind,u_tilde = prealloc
    @unpack mapP,mapI,mapO,inflowarr                     = bcdata

    Nq  = size(Uq,1)
    Nh  = size(u_tilde,1)
    Nfp = size(bcdata.mapP,1)
    flux_f      = @view flux[Nq+1:Nh,:]
    wavespeed_f = @view wavespeed[Nq+1:Nh,:]
    for k = 1:param.K
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
    for k = 1:param.K
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

    Nq  = size(Uq,1)
    Nh  = size(u_tilde,1)
    Nfp = size(bcdata.mapP,1)
    flux_f      = @view flux[Nq+1:Nh,:]
    wavespeed_f = @view wavespeed[Nq+1:Nh,:]
    for k = 1:param.K
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
                lambda_ij = .5*max(wavespeed[1,k],wavespeed[Nq,mod1(k-1,param.K)])
                lambda_i += lambda_ij
                rhsL_i -= (-.5*(flux[1,k]+flux[Nq,mod1(k-1,param.K)])-lambda_ij*(Uq[Nq,mod1(k-1,param.K)]-Uq[1,k]))/wJq_i
            end
            if i == Nq
                lambda_ij = .5*max(wavespeed[Nq,k],wavespeed[1,mod1(k+1,param.K)])
                lambda_i += lambda_ij
                rhsL_i -= (.5*(flux[Nq,k]+flux[1,mod1(k+1,param.K)])-lambda_ij*(Uq[1,mod1(k+1,param.K)]-Uq[Nq,k]))/wJq_i
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
                lambda_ij = .5*max(wavespeed[1,k],wavespeed[Nq,mod1(k-1,param.K)])
                ubar_ij = .5*(Uq[1,k]+Uq[Nq,mod1(k-1,param.K)])-.5*Br0_i/lambda_ij*(flux[Nq,mod1(k-1,param.K)]-flux[1,k])
                is_positive,_,_ = check_positivity_node(ubar_ij,param)
                if !is_positive
                    @show k,i,lambda_ij
                end
                rhsL_bar_i += 2*lambda_ij*ubar_ij/wJq_i
                rhsL_bar_i -= 2*lambda_ij*Uq[1,k]/wJq_i
            end
            if i == Nq
                Br0_i = .5
                lambda_ij = .5*max(wavespeed[Nq,k],wavespeed[1,mod1(k+1,param.K)])
                ubar_ij = .5*(Uq[Nq,k]+Uq[1,mod1(k+1,param.K)])-.5*Br0_i/lambda_ij*(flux[1,mod1(k+1,param.K)]-flux[Nq,k])
                ubar_ij_L = .5*Uq[Nq,k]+.5*Br0_i/lambda_ij*flux[Nq,k]
                ubar_L    = 2*lambda_ij*Uq[Nq,k] + flux[Nq,k]
                ubar_ij_R = .5*Uq[1,mod1(k+1,param.K)]-.5*Br0_i/lambda_ij*flux[1,mod1(k+1,param.K)]
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