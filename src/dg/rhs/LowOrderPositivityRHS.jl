##########################################
### RHS of positivity preserving Gauss ###
##########################################
function rhs_pos_Gauss!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,t,dt,nstage,need_proj=true)
    @unpack entropyproj_limiter_type,equation    = param
    @unpack mapP,mapI,mapO,inflowarr             = bcdata
    @unpack Jq                                   = discrete_data_gauss.geom
    @unpack Nh,Nq,Nfp                            = discrete_data_gauss.sizes
    @unpack u_tilde,flux,wavespeed,alphaarr,rhsL = prealloc
    @unpack αarr,LGLind                          = prealloc

    if (need_proj)
        entropy_projection!(prealloc,param,entropyproj_limiter_type,discrete_data_gauss,discrete_data_LGL,nstage)
    end
    calculate_wavespeed_and_inviscid_flux!(prealloc,param,discrete_data_gauss,discrete_data_LGL)

    # Assemble RHS
    clear_low_order_rhs!(prealloc,param)
    accumulate_low_order_rhs_volume!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    accumulate_low_order_rhs_surface!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata)
    scale_low_order_rhs_by_mass!(prealloc,param,discrete_data_gauss,discrete_data_LGL)

    # Determine positivity CFL
    if (nstage == 1)
        dt = calculate_lambda_and_low_order_CFL!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,t)
    end

    return dt
end

function calculate_wavespeed_and_inviscid_flux!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    @unpack equation = param
    @unpack Uq,Uf,u_tilde,wavespeed,flux,LGLind = prealloc
    
    K  = get_num_elements(param)
    wavespeed .= 0.0
    for k = 1:K
        discrete_data = (LGLind[k]) ? discrete_data_LGL : discrete_data_gauss
        update_face_values!(prealloc,k,discrete_data,get_low_order_surface_flux(param.rhs_type))
        update_wavespeed_and_inviscid_flux!(prealloc,k,param,discrete_data)
    end
end

function update_face_values!(prealloc,k,discrete_data,surface_flux_type::LaxFriedrichsOnNodalVal)
    @unpack Uq,Uf = prealloc

    Nfp = size(Uf,1)
    for i = 1:Nfp
        # TODO: preallocate into Fmask
        iq = findfirst(x->x==1.0, view(discrete_data.ops.Vf_low,i,:))
        Uf[i,k] = Uq[iq,k]
    end
end

function update_face_values!(prealloc,k,discrete_data,surface_flux_type::LaxFriedrichsOnProjectedVal)
    @unpack u_tilde,Uf = prealloc

    Nfp = size(Uf,1)
    Nq = size(prealloc.Uq,1)
    for i = 1:Nfp
        Uf[i,k] = u_tilde[i+Nq,k]
    end
end

function update_wavespeed_and_inviscid_flux!(prealloc,k,param,discrete_data)
    @unpack equation = param
    @unpack LGLind,Uq,Uf,wavespeed,wavespeed_f,flux = prealloc

    Nq  = size(Uq,1)
    Nfp = size(Uf,1)
    dim = get_dim_type(equation)
    # Volume wavespeed and inviscid flux
    for i = 1:Nq
        u_i = Uq[i,k]
        for j = 1:Nq
            Sxy0J_ij,n_ij_norm = get_Sx0_with_n(i,j,k,discrete_data,dim)
            if n_ij_norm > param.global_constants.ZEROTOL
                n_ij = Sxy0J_ij./n_ij_norm
                wavespeed[i,j,k] = wavespeed_davis_estimate(equation,u_i,n_ij)
            end
        end
        flux[i,k] = euler_fluxes(equation,u_i)
    end

    # Surface wavespeed and inviscid flux
    for i = 1:Nfp
        u_i = Uf[i,k]
        Bxy_i,n_i_norm = get_Bx_with_n(i,k,discrete_data,dim)
        n_i = Bxy_i./n_i_norm
        wavespeed_f[i,k] = wavespeed_davis_estimate(equation,u_i,n_i)
        flux[i+Nq,k] = euler_fluxes(equation,u_i)
    end
end

function clear_low_order_rhs!(prealloc,param)
    @unpack rhsxyL,Q0F1,λarr,λBarr = prealloc

    K  = get_num_elements(param)
    Nc = get_num_components(param.equation)
    Nd = get_dim(param.equation)
    for k = 1:K
        for i = 1:size(rhsxyL,1)
            rhsxyL[i,k] = zero(SVector{Nd,SVector{Nc,Float64}})
            Q0F1[i,k]   = zero(SVector{Nd,SVector{Nc,Float64}})
        end
    end
    λarr  .= 0.0
    λBarr .= 0.0
end

function accumulate_low_order_rhs_volume!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    @unpack equation = param
    @unpack Uq,rhsxyL,flux,wavespeed,LGLind,λarr,Q0F1 = prealloc
    
    dim = get_dim_type(equation)
    K  = get_num_elements(param)
    Nq = size(Uq,1)
    for k = 1:K
        discrete_data = LGLind[k] ? discrete_data_LGL : discrete_data_gauss
        # Volume contributions
        for j = 1:Nq
            for i = j+1:Nq
                Sxy0J_ij,n_ij_norm = get_Sx0_with_n(i,j,k,discrete_data,dim)
                Sxy0J_ji,n_ji_norm = get_Sx0_with_n(j,i,k,discrete_data,dim)
                if n_ij_norm > param.global_constants.ZEROTOL
                    Fxyij = @. .5*(flux[i,k]+flux[j,k])
                    wavespeed_ij = max(wavespeed[i,j,k],wavespeed[j,i,k])
                    λarr[i,j,k] = n_ij_norm*wavespeed_ij
                    λarr[j,i,k] = n_ji_norm*wavespeed_ij
                    ΛD_ij = get_graph_viscosity(prealloc,param,i,j,k,Sxy0J_ij,dim)
                    Q0F1[i,k] += 2.0*Sxy0J_ij.*Fxyij - ΛD_ij
                    Q0F1[j,k] += 2.0*Sxy0J_ji.*Fxyij + ΛD_ij
                end
            end
        end
    end

    for k = 1:K
        for i = 1:Nq
            rhsxyL[i,k] -= Q0F1[i,k]
        end
    end
end

function accumulate_low_order_rhs_surface!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata)
    @unpack equation = param
    @unpack Uq,Uf,rhsL,rhsxyL,flux,flux_L,wavespeed_f,LGLind,u_tilde,λBarr = prealloc
    @unpack mapP,mapI,mapO,inflowarr = bcdata

    K  = get_num_elements(param)
    dim = get_dim_type(equation)
    Nd  = get_dim(equation)
    Nq  = size(Uq,1)
    Nh  = size(u_tilde,1)
    Nfp = size(bcdata.mapP,1)
    for k = 1:K
        # Surface contributions
        discrete_data = LGLind[k] ? discrete_data_LGL : discrete_data_gauss
        for i = 1:Nfp
            # TODO: refactor
            idx = i+Nfp*(k-1)
            iP = mod1(mapP[i,k],Nfp)
            kP = div(mapP[i,k]-1,Nfp)+1
            Iidx = findfirst(x->(x==idx), mapI)
            Oidx = findfirst(x->(x==idx), mapO)
            flux_xy_P = !isnothing(Iidx) ? euler_fluxes(equation,inflowarr[Iidx]) : flux[iP+Nq,kP]
            uP = !isnothing(Iidx) ? inflowarr[Iidx] : Uf[iP,kP]

            Bxy_i,n_i_norm = get_Bx_with_n(i,k,discrete_data,dim)
            λBarr[i,k] = .5*n_i_norm*max(wavespeed_f[i,k],wavespeed_f[iP,kP])
            λB = (!isnothing(Iidx) || !isnothing(Oidx)) ? 0.0 : λBarr[i,k]

            flux_L[i,k] = Bxy_i.*(.5 .*(flux[i+Nq,k].+flux_xy_P))
            
            lf = λB*(uP-Uf[i,k])
            apply_LF_dissipation_to_flux(flux_L,param,i,k,lf,get_dim_type(param.equation))

            iq = findfirst(x->x==1.0, view(discrete_data.ops.Vf_low,i,:))
            rhsxyL[iq,k] -= flux_L[i,k]
        end
    end
end

function scale_low_order_rhs_by_mass!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    @unpack rhsL,rhsxyL,LGLind = prealloc
    @unpack Jq = discrete_data_gauss.geom

    K  = get_num_elements(param)
    for k = 1:K
        discrete_data = LGLind[k] ? discrete_data_LGL : discrete_data_gauss
        # Divide by mass
        for i = 1:size(rhsxyL,1)
            wq_i = discrete_data.ops.wq[i]
            wJq_i    = Jq[i,k]*wq_i
            rhsxyL[i,k] = rhsxyL[i,k]/wJq_i
        end
    end
    @. rhsL = sum(rhsxyL)
end

function calculate_lambda_and_low_order_CFL!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,t)
    @unpack CFL,dt0,T = param.timestepping_param
    @unpack Jq        = discrete_data_gauss.geom
    @unpack αarr,LGLind,Uq,u_tilde,λarr,λBarr = prealloc

    K  = get_num_elements(param)
    Nq  = size(prealloc.Uq,1)
    Nh  = size(prealloc.u_tilde,1)
    Nfp = Nh-Nq
    dt = min(CFL*dt0,T-t)
    for k = 1:K
        discrete_data = LGLind[k] ? discrete_data_LGL : discrete_data_gauss
        accumulate_alpha!(prealloc,k,param,discrete_data)
        for i = 1:Nq
            lambda_i = get_lambda_i(i,k,prealloc,param,discrete_data,bcdata)
            wq_i  = discrete_data.ops.wq[i]
            wJq_i = Jq[i,k]*wq_i
            dt = min(dt, CFL*.5*wJq_i/lambda_i)
        end
    end
    return dt
end

function accumulate_alpha!(prealloc,k,param,discrete_data)
    @unpack αarr,Uq,u_tilde = prealloc

    Nq  = size(prealloc.Uq,1)
    Nh  = size(prealloc.u_tilde,1)
    Nfp = Nh-Nq
    utilde_f    = @view u_tilde[Nq+1:Nh,:]
    for i = 1:Nfp
        # TODO: preallocate into Fmask, refactor
        iq = findfirst(x->x==1.0, view(discrete_data.ops.Vf_low,i,:))
        αarr[i,k] = find_alpha(param,Uq[iq,k],utilde_f[i,k])
    end
end

function get_lambda_i(i,k,prealloc,param,discrete_data,bcdata)
    @unpack equation = param
    @unpack λarr,λBarr,wavespeed,LGLind = prealloc

    dim = get_dim_type(equation)
    Nq = size(prealloc.Uq,1)
    Nh = size(prealloc.u_tilde,1)
    Nfp = size(bcdata.mapP,1)

    lambda_i = 0.0
    # TODO: can only consider upper diagonal
    for j = 1:Nq
        lambda_i += λarr[i,j,k]
    end
    
    surface_flux_type = LGLind[k] ? LaxFriedrichsOnNodalVal() : get_low_order_surface_flux(param.rhs_type)
    for j in (first(idx) for idx in pairs(view(discrete_data.ops.Vf_low,:,i)) if last(idx) == 1.0)      # TODO: precompute
        _,n_j_norm = get_Bx_with_n(j,k,discrete_data,dim)    # TODO: redundant
        lambda_i += get_lambda_B_CFL(prealloc,j,n_j_norm,k,surface_flux_type)
    end

    return lambda_i
end

function get_lambda_B_CFL(prealloc,i,n_i_norm,k,surface_flux_type::LaxFriedrichsOnNodalVal)
    return prealloc.λBarr[i,k]
end

function get_lambda_B_CFL(prealloc,i,n_i_norm,k,surface_flux_type::LaxFriedrichsOnProjectedVal)
    @unpack λBarr,αarr,wavespeed_f = prealloc

    return αarr[i,k]*λBarr[i,k] + .5*n_i_norm*wavespeed_f[i,k]
end

###############
###  Utils  ###
###############

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
