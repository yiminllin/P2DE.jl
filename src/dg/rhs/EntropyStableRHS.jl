#########################
### RHS of modal ESDG ###
#########################
function rhs_modalESDG!(prealloc,rhs_cache,param,discrete_data,bcdata,nstage,timer,need_proj=true)
    @unpack entropyproj_limiter_type = param

    cache = get_high_order_cache(rhs_cache)
    @timeit_debug timer "entropy projection" begin
    if (need_proj)
        entropy_projection!(prealloc,param,entropyproj_limiter_type,discrete_data,nstage,timer)
    end
    end

    @timeit_debug timer "calculate primitive variables" begin
    calculate_primitive_variables!(cache,prealloc,param,bcdata)
    end
    @timeit_debug timer "calculate interface dissipation coefficients" begin
    calculate_interface_dissipation_coeff!(cache,prealloc,param,bcdata,discrete_data)
    end
    @timeit_debug timer "enforce boundary conditions" begin
    enforce_BC!(cache,prealloc,param,bcdata)
    end

    # Flux differencing
    @timeit_debug timer "clear cache" begin
    clear_flux_differencing_cache!(cache)
    end
    @timeit_debug timer "flux differencing volume kernel" begin
    flux_differencing_volume!(cache,prealloc,param,discrete_data)
    end
    @timeit_debug timer "flux differencing surface kernel" begin
    flux_differencing_surface!(cache,prealloc,param,discrete_data)
    end

    # Assemble RHS
    @timeit_debug timer "assemble rhs" begin
    assemble_rhs!(cache,prealloc,param,discrete_data,nstage)
    end
end

function calculate_primitive_variables!(cache,prealloc,param,bcdata)
    @unpack equation   = param
    @unpack mapP       = bcdata
    @unpack u_tilde    = prealloc
    @unpack beta,rholog,betalog,uP,betaP,rhologP,betalogP = cache
    K = get_num_elements(param)
    Nfp = size(mapP,1)

    @batch for k = 1:K
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
    @batch for k = 1:K
        for i = 1:Nfp
            iP = mapP[i,k]
            # TODO: why direct access uf[iP] fails on obelix?
            uP[i,k]       = uf[iP:iP][1]
            betaP[i,k]    = betaf[iP:iP][1]
            rhologP[i,k]  = rhologf[iP:iP][1]
            betalogP[i,k] = betalogf[iP:iP][1]
        end
    end
end

function calculate_interface_dissipation_coeff!(cache,prealloc,param,bcdata,discrete_data)
    @unpack lam,LFc  = cache
    @unpack u_tilde  = prealloc
    @unpack equation = param
    @unpack mapP     = bcdata

    # TODO: refactor
    K  = get_num_elements(param)
    Nq = size(prealloc.Uq,1)
    Nh = size(prealloc.u_tilde,1)
    Nfp = Nh-Nq
    uf = @view u_tilde[Nq+1:Nh,:]
    dim = get_dim_type(equation)

    # Lax Friedrichs dissipation
    @batch for k = 1:K
        for i = 1:Nfp
            Bxy_i,n_i_norm = get_Bx_with_n(i,k,discrete_data,dim)
            n_i = Bxy_i./n_i_norm 
            lam[i,k] = wavespeed_davis_estimate(equation,uf[i,k],n_i)
            LFc[i,k] = .5*n_i_norm
        end
    end

    @batch for k = 1:K
        for i = 1:Nfp
            LFc[i,k] = LFc[i,k]*max(lam[i,k],lam[mapP[i,k]])
        end
    end
end

function enforce_BC!(cache,prealloc,param,bcdata)
    @unpack Uq                            = prealloc
    @unpack equation                      = param
    @unpack mapP,mapI,mapO,inflowarr      = bcdata
    @unpack LFc,uP,betaP,rhologP,betalogP = cache

    # zero dissipation on the boundary
    @batch for i in mapI
        LFc[i] = 0.0
    end 
    @batch for i in mapO
        LFc[i] = 0.0
    end 

    # Enforce Inflow BC
    @batch for i = 1:size(mapI,1)
        ii = mapI[i]
        uP[ii]       = inflowarr[i]
        betaP[ii]    = betafun(equation,uP[ii])
        rhologP[ii]  = log(uP[ii][1])
        betalogP[ii] = log(betaP[ii])
    end

    # Enforce outflow BC
    @batch for i = 1:size(mapO,1)
        io = mapO[i]
        # TODO: hardcoded
        uP[io]       = Uq[end,end]
        betaP[io]    = betafun(equation,uP[io])
        rhologP[io]  = log(uP[io][1])
        betalogP[io] = log(betaP[io])
    end
end

function clear_flux_differencing_cache!(cache)
    @unpack QF1 = cache
    @. QF1 = zero(QF1)
end

function flux_differencing_volume!(cache,prealloc,param,discrete_data)
    @unpack equation = param
    @unpack QF1      = cache
    @unpack Srsh_nnz = discrete_data.ops

    dim = get_dim_type(equation)
    K  = get_num_elements(param)
    Nq = discrete_data.sizes.Nq
    Nh = size(QF1,1)
    Ui = zero(SVector{5,Float64})
    Uj = zero(SVector{5,Float64})
    @batch for k = 1:K
        for (i,j) in Srsh_nnz
            Ui = get_U_beta!(i,k,cache,prealloc,param.equation,dim)
            Uj = get_U_beta!(j,k,cache,prealloc,param.equation,dim)
            accumulate_QF1!(QF1,i,Ui,j,Uj,k,param,discrete_data,equation)
        end
    end
end

# Get (utilde, beta, log(rho), log(beta)) at index i,element k
function get_U_beta!(idx,k,cache,prealloc,equation,dim::Dim1)
    @unpack u_tilde             = prealloc
    @unpack beta,rholog,betalog = cache

    return SVector(u_tilde[idx,k][1],u_tilde[idx,k][2]/u_tilde[idx,k][1],beta[idx,k],rholog[idx,k],betalog[idx,k])
end

function get_U_beta!(idx,k,cache,prealloc,equation,dim::Dim2)
    @unpack u_tilde             = prealloc
    @unpack beta,rholog,betalog = cache

    return SVector(u_tilde[idx,k][1],u_tilde[idx,k][2]/u_tilde[idx,k][1],u_tilde[idx,k][3]/u_tilde[idx,k][1],beta[idx,k],rholog[idx,k],betalog[idx,k])
end

function accumulate_QF1!(QF1,i,Ui,j,Uj,k,param,discrete_data,equation)
    ϵ = param.global_constants.ZEROTOL
    dim = get_dim_type(equation)
    # TODO: assume Sxyh_db_ij = -Sxyh_db_ji
    #              Sxyh_db_ji = get_Sx(j,i,k,discrete_data,dim)
    Sxyh_db_ij = get_Sx(i,j,k,discrete_data,dim)
    fxy = fS_prim_log(equation,Ui,Uj)
    Sfxy_ij = Sxyh_db_ij .* fxy
    Sfxy_ji = -Sfxy_ij
    QF1[i,k] += Sfxy_ij
    QF1[j,k] += Sfxy_ji
end

function flux_differencing_surface!(cache,prealloc,param,discrete_data)
    @unpack equation = param

    K  = get_num_elements(param)
    @batch for k = 1:K
        accumulate_numerical_flux!(prealloc,cache,k,param,discrete_data,equation)
    end
end

function accumulate_numerical_flux!(prealloc,cache,k,param,discrete_data,equation)
    @unpack BF_H,u_tilde = prealloc
    @unpack uP,LFc       = cache
    
    # Boundary contributions (B F)1
    Nq  = size(prealloc.Uq,1)
    Nh  = size(u_tilde,1)
    Nfp = size(BF_H,1)
    dim = get_dim_type(equation)
    uf = @view u_tilde[Nq+1:Nh,:]
    for i = 1:Nfp
        fxy = evaluate_high_order_surface_flux(prealloc,cache,param,i,k,get_high_order_surface_flux(param.rhs_type))
        Bxy_i = get_Bx(i,k,discrete_data,dim)
        BF_H[i,k] = Bxy_i.*fxy
        # Apply LF dissipation
        lf = LFc[i,k]*(uP[i,k]-uf[i,k])
        apply_LF_dissipation_to_BF(BF_H,param,i,k,lf,dim)
    end
end

function evaluate_high_order_surface_flux(prealloc,cache,param,i,k,surface_flux_type::ChandrashekarOnProjectedVal)
    @unpack equation                                      = param
    @unpack u_tilde                                       = prealloc
    @unpack beta,rholog,betalog,uP,betaP,rhologP,betalogP = cache

    Nq = size(prealloc.Uq,1)
    Nh = size(u_tilde,1)
    Nd = get_dim(equation)
    uf       = @view u_tilde[Nq+1:Nh,:]
    betaf    = @view beta[Nq+1:Nh,:]
    rhologf  = @view rholog[Nq+1:Nh,:]
    betalogf = @view betalog[Nq+1:Nh,:]
    return fS_prim_log(equation,(uf[i,k][1],(uf[i,k][c]/uf[i,k][1] for c in 2:2+Nd-1)...,betaf[i,k],rhologf[i,k],betalogf[i,k]),
                                (uP[i,k][1],(uP[i,k][c]/uP[i,k][1] for c in 2:2+Nd-1)...,betaP[i,k],rhologP[i,k],betalogP[i,k]))
end

function evaluate_high_order_surface_flux(prealloc,cache,param,i,k,surface_flux_type::LaxFriedrichsOnProjectedVal)
    @unpack equation = param
    @unpack u_tilde  = prealloc
    @unpack uP       = cache

    Nq = size(prealloc.Uq,1)
    Nh = size(u_tilde,1)
    uf = @view u_tilde[Nq+1:Nh,:]
    fxyf = euler_fluxes(equation,uf[i,k])
    fxyP = euler_fluxes(equation,uP[i,k])

    return .5 .* (fxyf.+fxyP)
end

function project_flux_difference_to_quad_unlimited!(k,cache,prealloc,discrete_data,tid)
    @unpack BF_H                      = prealloc
    @unpack MinvVhTQF1,MinvVfTBF1,QF1 = cache
    @unpack MinvVhT,MinvVfT           = discrete_data.ops

    @views mul!(MinvVhTQF1[:,k],MinvVhT,QF1[:,k])
    @views mul!(MinvVfTBF1[:,k],MinvVfT,BF_H[:,k])
end

function project_flux_difference_to_quad!(cache,prealloc,param,entropyproj_limiter_type::NoEntropyProjectionLimiter,discrete_data,k,nstage,tid)
    project_flux_difference_to_quad_unlimited!(k,cache,prealloc,discrete_data,tid)
end

# TODO: hardcoded, only for gauss
function project_flux_difference_to_quad!(cache,prealloc,param,entropyproj_limiter_type::ScaledExtrapolation,discrete_data,k,nstage,tid)
    @unpack BF_H                                                 = prealloc
    @unpack MinvVhTQF1,MinvVfTBF1,QF1,VhT_new,MinvVhT_new,Vf_new = cache
    @unpack Nq,Nh                                                = discrete_data.sizes

    # TODO: only consider Gauss and LGL for now, so Vq = I
    #       and we assume VqT_new = I
    #       VqT_new = @views VhT_new[:,1:Nq]
    #       VqT_new .= transpose(discrete_data.ops.Vq)
    update_limited_extrapolation!(cache,prealloc,param,param.entropyproj_limiter_type,discrete_data,k,nstage,tid)
    Vf_new      = @views Vf_new[:,:,tid]
    VhT_new     = @views VhT_new[:,:,tid]
    MinvVhT_new = @views MinvVhT_new[:,:,tid]
    VfT_new = @views VhT_new[:,Nq+1:Nh]
    VfT_new .= transpose(Vf_new)
    @. MinvVhT_new = (1/discrete_data.ops.wq)*VhT_new
    MinvVfT_new = @views MinvVhT_new[:,Nq+1:Nh]
    @views mul!(MinvVhTQF1[:,k],MinvVhT_new,QF1[:,k])
    @views mul!(MinvVfTBF1[:,k],MinvVfT_new,BF_H[:,k])
end

function update_limited_extrapolation!(cache,prealloc,param,entropyproj_limiter_type::ElementwiseScaledExtrapolation,discrete_data,k,nstage,tid)
    @unpack Vf_new    = cache
    @unpack Vf,Vf_low = discrete_data.ops

    l_k = prealloc.θ_arr[k,nstage]
    @. @views Vf_new[:,:,tid] = l_k*Vf+(1.0-l_k)*Vf_low
end

function update_limited_extrapolation!(cache,prealloc,param,entropyproj_limiter_type::NodewiseScaledExtrapolation,discrete_data,k,nstage,tid)
    @unpack Vf_new    = cache
    @unpack Vf,Vf_low = discrete_data.ops

    for i = 1:discrete_data.sizes.Nfp
        l_k_i = prealloc.θ_local_arr[i,k,nstage]
        @views @. Vf_new[i,:,tid] = l_k_i*Vf[i,:]+(1-l_k_i)*Vf_low[i,:]
    end
end

# Return whether on element k, the extrapolation is limited
function is_Vf_limited(prealloc,k,nstage,entropyproj_limiter_type::NoEntropyProjectionLimiter)
    return false
end

function is_Vf_limited(prealloc,k,nstage,entropyproj_limiter_type::ElementwiseScaledExtrapolation)
    return prealloc.θ_arr[k,nstage] < 1.0
end

function is_Vf_limited(prealloc,k,nstage,entropyproj_limiter_type::NodewiseScaledExtrapolation)
    return minimum(view(prealloc.θ_local_arr,:,k,nstage)) < 1.0
end

# TODO: dispatch
function assemble_rhs!(cache,prealloc,param,discrete_data,nstage)
    @unpack entropyproj_limiter_type  = param
    @unpack QF1,MinvVhTQF1,MinvVfTBF1 = cache
    @unpack BF_H,rhsH,rhsxyH          = prealloc
    @unpack Jq                        = discrete_data.geom
    @unpack MinvVhT,MinvVfT,Vq        = discrete_data.ops
 
    K  = get_num_elements(param)
    # Assemble RHS
    @batch for k = 1:K
        tid = Threads.threadid()
        # If 1. LGL
        #    2. Gauss with no entropy proj limiter, adaptive filter
        #                  scaled extrapolation with l_k = 1 (elementwise)
        #                                            l_k_i = 1 for all i (nodewise)
        # Apply precomputed matrix 
        # otherwise, if on Gauss with scaled extrapolation and nonzero limiting param
        #            apply limited Vf
        if !is_Vf_limited(prealloc,k,nstage,entropyproj_limiter_type)
            project_flux_difference_to_quad_unlimited!(k,cache,prealloc,discrete_data,tid)
        else
            project_flux_difference_to_quad!(cache,prealloc,param,entropyproj_limiter_type,discrete_data,k,nstage,tid)
        end
        # TODO: assume collocation scheme, so Nq = Np
        for i = 1:size(rhsH,1)
            rhsxyH[i,k] = -(MinvVhTQF1[i,k]+MinvVfTBF1[i,k])/Jq[i,k]
            rhsH[i,k] = sum(rhsxyH[i,k])
        end
        # TODO: Assume Vq = I for both LGL and Gauss
        # @views mul!(rhsH[:,k],Vq,sum.(rhsxyH[:,k]))
    end
end

