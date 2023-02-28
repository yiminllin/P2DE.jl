##########################################
### RHS of positivity preserving Gauss ###
##########################################
function rhs_pos_Gauss!(prealloc,rhs_cache,param,discrete_data_gauss,discrete_data_LGL,bcdata,t,dt,nstage,timer,need_proj=true)
    @unpack entropyproj_limiter_type,equation    = param

    cache = get_low_order_cache(rhs_cache)
    @timeit timer "entropy projection" begin
    if (need_proj)
        entropy_projection!(prealloc,param,entropyproj_limiter_type,discrete_data_gauss,discrete_data_LGL,nstage,timer)
    end
    end
    @timeit timer "calculate wavespeed and inviscid flux" begin
    calculate_wavespeed_and_inviscid_flux!(cache,prealloc,param,discrete_data_gauss,discrete_data_LGL)
    end

    # Assemble RHS
    @timeit timer "clear cache" begin
    clear_low_order_rhs!(cache,prealloc,param)
    end
    @timeit timer "Low order volume kernel" begin
    accumulate_low_order_rhs_volume!(cache,prealloc,param,discrete_data_gauss,discrete_data_LGL)
    end
    @timeit timer "Low order surface kernel" begin
    accumulate_low_order_rhs_surface!(cache,prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata)
    end
    @timeit timer "Scale low order solution by mass" begin
    scale_low_order_rhs_by_mass!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    end

    @timeit timer "Calculate low order positivity CFL" begin
    # Determine positivity CFL
    if (nstage == 1)
        dt = calculate_lambda_and_low_order_CFL!(cache,prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,t)
    end
    end

    return dt
end

function calculate_wavespeed_and_inviscid_flux!(cache,prealloc,param,discrete_data_gauss,discrete_data_LGL)
    @unpack LGLind    = prealloc
    
    K  = get_num_elements(param)
    for k = 1:K
        discrete_data = (LGLind[k]) ? discrete_data_LGL : discrete_data_gauss
        update_face_values!(cache,prealloc,k,discrete_data,get_low_order_surface_flux(param.rhs_type))
        update_wavespeed_and_inviscid_flux!(cache,prealloc,k,param,discrete_data)
    end
end

function update_face_values!(cache,prealloc,k,discrete_data,surface_flux_type::LaxFriedrichsOnNodalVal)
    @unpack Uf   = cache
    @unpack Uq   = prealloc
    @unpack fq2q = discrete_data.ops

    Nfp = size(Uf,1)
    for i = 1:Nfp
        iq = fq2q[i]
        Uf[i,k] = Uq[iq,k]
    end
end

function update_face_values!(cache,prealloc,k,discrete_data,surface_flux_type::LaxFriedrichsOnProjectedVal)
    @unpack Uf      = cache
    @unpack u_tilde = prealloc

    Nfp = size(Uf,1)
    Nq = size(prealloc.Uq,1)
    for i = 1:Nfp
        Uf[i,k] = u_tilde[i+Nq,k]
    end
end

function update_wavespeed_and_inviscid_flux!(cache,prealloc,k,param,discrete_data)
    @unpack equation = param
    @unpack Uq       = prealloc
    @unpack Srs0_nnz = discrete_data.ops
    @unpack Uf,wavespeed_f,flux = cache

    Nq  = size(Uq,1)
    Nfp = size(Uf,1)
    dim = get_dim_type(equation)

    # Volume inviscid flux
    for i = 1:Nq
        u_i = Uq[i,k]
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

function clear_low_order_rhs!(cache,prealloc,param)
    @unpack rhsxyL          = prealloc
    @unpack Q0F1,λarr,λBarr = cache

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

function accumulate_low_order_rhs_volume!(cache,prealloc,param,discrete_data_gauss,discrete_data_LGL)
    @unpack equation         = param
    @unpack rhsxyL,LGLind,Uq = prealloc
    @unpack flux,λarr,Q0F1   = cache
    
    dim = get_dim_type(equation)
    K  = get_num_elements(param)
    Nq = size(Q0F1,1)
    for k = 1:K
        discrete_data = LGLind[k] ? discrete_data_LGL : discrete_data_gauss
        @unpack Srs0_nnz = discrete_data.ops
        # Volume contributions
        for (i,j) in Srs0_nnz
            u_i = Uq[i,k]
            u_j = Uq[j,k]
            Sxy0J_ij,n_ij_norm = get_Sx0_with_n(i,j,k,discrete_data,dim)
            Sxy0J_ji,n_ji_norm = get_Sx0_with_n(j,i,k,discrete_data,dim)
            n_ij = Sxy0J_ij./n_ij_norm
            n_ji = Sxy0J_ji./n_ji_norm
            Fxyij = @. .5*(flux[i,k]+flux[j,k])
            wavespeed_ij = max(wavespeed_davis_estimate(equation,u_i,n_ij),wavespeed_davis_estimate(equation,u_j,n_ji))
            λarr[i,j,k] = n_ij_norm*wavespeed_ij
            λarr[j,i,k] = n_ji_norm*wavespeed_ij
            ΛD_ij = get_graph_viscosity(cache,prealloc,param,i,j,k,Sxy0J_ij,dim)
            Q0F1[i,k] += 2.0*Sxy0J_ij.*Fxyij - ΛD_ij
            Q0F1[j,k] += 2.0*Sxy0J_ji.*Fxyij + ΛD_ij
        end
    end

    for k = 1:K
        for i = 1:Nq
            rhsxyL[i,k] -= Q0F1[i,k]
        end
    end
end

function accumulate_low_order_rhs_surface!(cache,prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata)
    @unpack equation = param
    @unpack Uf,flux,wavespeed_f,λBarr = cache
    @unpack rhsxyL,BF_L,LGLind        = prealloc
    @unpack mapP,mapI,mapO,inflowarr  = bcdata

    K  = get_num_elements(param)
    dim = get_dim_type(equation)
    Nd  = get_dim(equation)
    Nq  = size(prealloc.Uq,1)
    Nfp = size(bcdata.mapP,1)
    for k = 1:K
        # Surface contributions
        discrete_data = LGLind[k] ? discrete_data_LGL : discrete_data_gauss
        @unpack fq2q = discrete_data.ops
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

            fxy = .5 .*(flux[i+Nq,k].+flux_xy_P)
            BF_L[i,k] = Bxy_i.*fxy
            
            lf = λB*(uP-Uf[i,k])
            apply_LF_dissipation_to_BF(BF_L,param,i,k,lf,get_dim_type(param.equation))

            iq = fq2q[i]
            rhsxyL[iq,k] -= BF_L[i,k]
        end
    end
end

function scale_low_order_rhs_by_mass!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    @unpack Jq = discrete_data_gauss.geom
    @unpack rhsL,rhsxyL,LGLind = prealloc

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

function calculate_lambda_and_low_order_CFL!(cache,prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,t)
    @unpack CFL,dt0,T = param.timestepping_param
    @unpack Jq        = discrete_data_gauss.geom
    @unpack LGLind    = prealloc

    K  = get_num_elements(param)
    Nq  = size(prealloc.Uq,1)
    dt = min(CFL*dt0,T-t)
    for k = 1:K
        discrete_data = LGLind[k] ? discrete_data_LGL : discrete_data_gauss
        accumulate_alpha!(cache,prealloc,k,param,discrete_data)
        for i = 1:Nq
            lambda_i = get_lambda_i(i,k,cache,prealloc,param,discrete_data,bcdata)
            wq_i  = discrete_data.ops.wq[i]
            wJq_i = Jq[i,k]*wq_i
            dt = min(dt, CFL*.5*wJq_i/lambda_i)
        end
    end
    return dt
end

function accumulate_alpha!(cache,prealloc,k,param,discrete_data)
    @unpack αarr       = cache
    @unpack Uq,u_tilde = prealloc
    @unpack fq2q       = discrete_data.ops

    Nq  = size(Uq,1)
    Nfp = size(αarr,1)
    Nh  = Nq+Nfp
    utilde_f    = @view u_tilde[Nq+1:Nh,:]
    for i = 1:Nfp
        # TODO: preallocate into Fmask, refactor
        iq = fq2q[i]
        αarr[i,k] = find_alpha(param,Uq[iq,k],utilde_f[i,k])
    end
end

function get_lambda_i(i,k,cache,prealloc,param,discrete_data,bcdata)
    @unpack equation = param
    @unpack λarr     = cache
    @unpack LGLind   = prealloc
    @unpack q2fq     = discrete_data.ops

    dim = get_dim_type(equation)
    Nq = size(prealloc.Uq,1)
    Nfp = size(bcdata.mapP,1)

    lambda_i = 0.0
    # TODO: can only consider upper diagonal
    for j = 1:Nq
        lambda_i += λarr[i,j,k]
    end
    
    surface_flux_type = LGLind[k] ? LaxFriedrichsOnNodalVal() : get_low_order_surface_flux(param.rhs_type)
    for j in q2fq[i]
        _,n_j_norm = get_Bx_with_n(j,k,discrete_data,dim)    # TODO: redundant
        lambda_i += get_lambda_B_CFL(cache,j,n_j_norm,k,surface_flux_type)
    end

    return lambda_i
end

function get_lambda_B_CFL(cache,i,n_i_norm,k,surface_flux_type::LaxFriedrichsOnNodalVal)
    return cache.λBarr[i,k]
end

function get_lambda_B_CFL(cache,i,n_i_norm,k,surface_flux_type::LaxFriedrichsOnProjectedVal)
    @unpack λBarr,αarr,wavespeed_f = cache

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
