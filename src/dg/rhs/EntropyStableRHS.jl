#########################
### RHS of modal ESDG ###
#########################
function rhs_modalESDG!(prealloc,rhs_cache,param,discrete_data_gauss,discrete_data_LGL,bcdata,nstage,timer,need_proj=true)
    @unpack entropyproj_limiter_type = param

    cache = get_high_order_cache(rhs_cache)
    @timeit timer "entropy projection" begin
    if (need_proj)
        entropy_projection!(prealloc,param,entropyproj_limiter_type,discrete_data_gauss,discrete_data_LGL,nstage,timer)
    end
    end

    @timeit timer "calculate primitive variables" begin
    calculate_primitive_variables!(cache,prealloc,param,bcdata)
    end
    @timeit timer "calculate interface dissipation coefficients" begin
    calculate_interface_dissipation_coeff!(cache,prealloc,param,bcdata,discrete_data_gauss,discrete_data_LGL)
    end
    @timeit timer "enforce boundary conditions" begin
    enforce_BC!(cache,prealloc,param,bcdata)
    end

    # Flux differencing
    @timeit timer "clear cache" begin
    clear_flux_differencing_cache!(cache)
    end
    @timeit timer "flux differencing volume kernel" begin
    flux_differencing_volume!(cache,prealloc,param,discrete_data_LGL,discrete_data_gauss)
    end
    @timeit timer "flux differencing surface kernel" begin
    flux_differencing_surface!(cache,prealloc,param,discrete_data_LGL,discrete_data_gauss)
    end

    # Assemble RHS
    @timeit timer "assemble rhs" begin
    assemble_rhs!(cache,prealloc,param,discrete_data_gauss,discrete_data_LGL,nstage)
    end
end

function calculate_primitive_variables!(cache,prealloc,param,bcdata)
    @unpack equation   = param
    @unpack mapP       = bcdata
    @unpack u_tilde    = prealloc
    @unpack beta,rholog,betalog,uP,betaP,rhologP,betalogP = cache
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

function calculate_interface_dissipation_coeff!(cache,prealloc,param,bcdata,discrete_data_gauss,discrete_data_LGL)
    @unpack lam,LFc        = cache
    @unpack u_tilde,LGLind = prealloc
    @unpack equation       = param
    @unpack mapP           = bcdata

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

function enforce_BC!(cache,prealloc,param,bcdata)
    @unpack Uq                            = prealloc
    @unpack equation                      = param
    @unpack mapP,mapI,mapO,inflowarr      = bcdata
    @unpack LFc,uP,betaP,rhologP,betalogP = cache

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

function clear_flux_differencing_cache!(cache)
    @unpack QF1 = cache
    @. QF1 = zero(QF1)
end

function flux_differencing_volume!(cache,prealloc,param,discrete_data_LGL,discrete_data_gauss)
    @unpack equation  = param
    @unpack Ui,Uj,QF1 = cache
    @unpack LGLind    = prealloc

    K  = get_num_elements(param)
    Nh = size(QF1,1)
    for k = 1:K
        discrete_data = LGLind[k] ? discrete_data_LGL : discrete_data_gauss
        for j = 1:Nh
            accumulate_U_beta!(Uj,j,k,cache,prealloc,param.equation)
            for i = j+1:Nh
                accumulate_U_beta!(Ui,i,k,cache,prealloc,param.equation)
                accumulate_QF1!(QF1,i,Ui,j,Uj,k,discrete_data,equation)
            end
        end
    end
end

# Accumulate U with (utilde, beta, log(rho), log(beta)) at index i,element k
function accumulate_U_beta!(U,idx,k,cache,prealloc,equation)
    @unpack u_tilde             = prealloc
    @unpack beta,rholog,betalog = cache

    Nd = get_dim(equation)
    U[1]     = u_tilde[idx,k][1]
    for c = 2:2+Nd-1
        U[c] = u_tilde[idx,k][c]/U[1]
    end
    U[end-2] = beta[idx,k]
    U[end-1] = rholog[idx,k]
    U[end]   = betalog[idx,k]
end

function accumulate_QF1!(QF1,i,Ui,j,Uj,k,discrete_data,equation)
    dim = get_dim_type(equation)
    fxy = fS_prim_log(equation,Ui,Uj)
    Sxyh_db_ij = get_Sx(i,j,k,discrete_data,dim)
    Sxyh_db_ji = get_Sx(j,i,k,discrete_data,dim)
    QF1[i,k] += Sxyh_db_ij .* fxy
    QF1[j,k] += Sxyh_db_ji .* fxy
end

function flux_differencing_surface!(cache,prealloc,param,discrete_data_LGL,discrete_data_gauss)
    @unpack LGLind   = prealloc
    @unpack equation = param

    K  = get_num_elements(param)
    for k  = 1:K
        discrete_data = (LGLind[k]) ? discrete_data_LGL : discrete_data_gauss
        accumulate_numerical_flux!(prealloc,cache,k,param,discrete_data,equation)
        apply_LF_dissipation!(prealloc,cache,param,discrete_data,k)
    end
end

function accumulate_numerical_flux!(prealloc,cache,k,param,discrete_data,equation)
    @unpack BF_H = prealloc
    
    # Boundary contributions (B F)1
    Nfp = size(BF_H,1)
    dim = get_dim_type(equation)
    for i = 1:Nfp
        fxy = evaluate_high_order_surface_flux(prealloc,cache,param,i,k,get_high_order_surface_flux(param.rhs_type))
        Bxy_i = get_Bx(i,k,discrete_data,dim)     # TODO: redundant calculation
        BF_H[i,k] = Bxy_i.*fxy
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

function apply_LF_dissipation!(prealloc,cache,param,discrete_data,k)
    @unpack LFc,uP       = cache
    @unpack BF_H,u_tilde = prealloc

    # LF dissipation
    dim = get_dim_type(param.equation)
    Nq = size(prealloc.Uq,1)
    Nh = size(u_tilde,1)
    Nfp = size(BF_H,1)
    uf = @view u_tilde[Nq+1:Nh,:]
    for i = 1:Nfp
        lf = LFc[i,k]*(uP[i,k]-uf[i,k])
        Bxy_i = get_Bx(i,k,discrete_data,dim)     # TODO: redundant calculation
        apply_LF_dissipation_to_BF(BF_H,param,i,k,lf,dim)
    end
end

function project_flux_difference_to_quad!(cache,prealloc,param,entropyproj_limiter_type::Union{AdaptiveFilter,NoEntropyProjectionLimiter},discrete_data_gauss,k,nstage)
    @unpack BF_H                      = prealloc
    @unpack MinvVhTQF1,MinvVfTBF1,QF1 = cache

    @views mul!(MinvVhTQF1[:,k],discrete_data_gauss.ops.MinvVhT,QF1[:,k])
    @views mul!(MinvVfTBF1[:,k],discrete_data_gauss.ops.MinvVfT,BF_H[:,k])
end

function project_flux_difference_to_quad!(cache,prealloc,param,entropyproj_limiter_type::ScaledExtrapolation,discrete_data_gauss,k,nstage)
    @unpack BF_H                                                 = prealloc
    @unpack MinvVhTQF1,MinvVfTBF1,QF1,VhT_new,MinvVhT_new,Vf_new = cache
    @unpack Nq,Nh                                                = discrete_data_gauss.sizes

    update_limited_extrapolation!(cache,prealloc,param,param.entropyproj_limiter_type,discrete_data_gauss,k,nstage)
    VqT_new = @views VhT_new[:,1:Nq]
    VfT_new = @views VhT_new[:,Nq+1:Nh]
    VqT_new .= transpose(discrete_data_gauss.ops.Vq)
    VfT_new .= transpose(Vf_new)
    @. MinvVhT_new = (1/discrete_data_gauss.ops.wq)*VhT_new
    MinvVfT_new = @views MinvVhT_new[:,Nq+1:Nh]
    @views mul!(MinvVhTQF1[:,k],MinvVhT_new,QF1[:,k])
    @views mul!(MinvVfTBF1[:,k],MinvVfT_new,BF_H[:,k])
end

function update_limited_extrapolation!(cache,prealloc,param,entropyproj_limiter_type::ElementwiseScaledExtrapolation,discrete_data_gauss,k,nstage)
    @unpack Vf_new    = cache
    @unpack Vf,Vf_low = discrete_data_gauss.ops

    l_k = prealloc.θ_arr[k,nstage]
    @. Vf_new = l_k*Vf+(1.0-l_k)*Vf_low
end

function update_limited_extrapolation!(cache,prealloc,param,entropyproj_limiter_type::NodewiseScaledExtrapolation,discrete_data_gauss,k,nstage)
    @unpack Vf_new    = cache
    @unpack Vf,Vf_low = discrete_data_gauss.ops

    l_k = prealloc.θ_arr[k,nstage]
    for i = 1:discrete_data_gauss.sizes.Nfp
        l_k_i = prealloc.θ_local_arr[i,k,nstage]
        @views @. Vf_new[i,:] = l_k_i*Vf[i,:]+(1-l_k_i)*Vf_low[i,:]
    end
end

# TODO: dispatch
function assemble_rhs!(cache,prealloc,param,discrete_data_gauss,discrete_data_LGL,nstage)
    @unpack QF1,MinvVhTQF1,MinvVfTBF1 = cache
    @unpack BF_H,LGLind,rhsH,rhsxyH   = prealloc
    @unpack Jq                        = discrete_data_gauss.geom
 
    K  = get_num_elements(param)
    # Assemble RHS
    for k = 1:K
        discrete_data = LGLind[k] ? discrete_data_LGL : discrete_data_gauss
        @unpack MinvVhT,MinvVfT,Vq = discrete_data.ops
        if LGLind[k]
            @views mul!(MinvVhTQF1[:,k],discrete_data.ops.MinvVhT,QF1[:,k])
            @views mul!(MinvVfTBF1[:,k],discrete_data.ops.MinvVfT,BF_H[:,k])
        else
            project_flux_difference_to_quad!(cache,prealloc,param,param.entropyproj_limiter_type,discrete_data,k,nstage)
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

