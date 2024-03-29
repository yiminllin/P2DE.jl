##########################################
### RHS of positivity preserving Gauss ###
##########################################
function rhs_pos_Gauss!(prealloc,rhs_cache,param,discrete_data,bcdata,t,dt,nstage,timer,need_proj=true)
    @unpack entropyproj_limiter_type,equation    = param

    cache = get_low_order_cache(rhs_cache)
    @timeit_debug timer "entropy projection" begin
    if (need_proj)
        entropy_projection!(prealloc,param,entropyproj_limiter_type,discrete_data,nstage,timer)
    end
    end
    @timeit_debug timer "calculate wavespeed and inviscid flux" begin
    calculate_wavespeed_and_inviscid_flux!(cache,prealloc,param,discrete_data)
    end
    @timeit_debug timer "enforce boundary condition" begin
    get_uP_and_enforce_BC!(cache,prealloc,param,bcdata,discrete_data)
    end

    # Assemble RHS
    @timeit_debug timer "clear cache" begin
    clear_low_order_rhs!(cache,prealloc,param)
    end
    @timeit_debug timer "Low order volume kernel" begin
    accumulate_low_order_rhs_volume!(cache,prealloc,param,discrete_data)
    end
    @timeit_debug timer "Low order surface kernel" begin
    accumulate_low_order_rhs_surface!(cache,prealloc,param,discrete_data,bcdata)
    end
    @timeit_debug timer "Scale low order solution by mass" begin
    scale_low_order_rhs_by_mass!(prealloc,param,discrete_data)
    end

    @timeit_debug timer "Calculate low order positivity CFL" begin
    # Determine positivity CFL
    if (nstage == 1)
        dt = calculate_lambda_and_low_order_CFL!(cache,prealloc,param,discrete_data,bcdata,t)
    end
    end

    # check_low_order_entropy_stability(cache,prealloc,param,discrete_data,get_dim_type(equation))

    return dt
end

function calculate_wavespeed_and_inviscid_flux!(cache,prealloc,param,discrete_data)
    K  = get_num_elements(param)
    @batch for k = 1:K
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
        flux[i,k] = fluxes(equation,u_i)
    end

    # Surface wavespeed and inviscid flux
    for i = 1:Nfp
        u_i = Uf[i,k]
        Bxy_i,n_i_norm = get_Bx_with_n(i,k,discrete_data,dim)
        n_i = Bxy_i./n_i_norm
        wavespeed_f[i,k] = wavespeed_estimate(equation,u_i,n_i)
        flux[i+Nq,k] = fluxes(equation,u_i)
    end
end

function get_uP_and_enforce_BC!(cache,prealloc,param,bcdata,discrete_data)
    @unpack Uf,uP               = cache
    @unpack Uq,u_tilde          = prealloc
    @unpack equation            = param
    @unpack mapP,mapI,mapO,Ival = bcdata
    @unpack fq2q                = discrete_data.ops

    K   = get_num_elements(param)
    Nfp = size(mapP,1)
    # Initialize uP
    @batch for k = 1:K
        for i = 1:Nfp
            iP = mod1(mapP[i,k],Nfp)
            kP = div(mapP[i,k]-1,Nfp)+1
            uP[i,k] = Uf[iP,kP]
        end
    end

    # TODO: we still apply LF dissipation on Dirichlet boundary condition now
    # Enforce inflow BC
    @batch for i = 1:size(mapI,1)
        ii = mapI[i]
        iP = mod1(mapP[ii],Nfp)
        kP = div(mapP[ii]-1,Nfp)+1
        uP[ii] = Ival[i]
    end
 
    Nq = size(Uq,1)
    Nh = size(u_tilde,1)
    uf = @view u_tilde[Nq+1:Nh,:]
    @batch for i = 1:size(mapO,1)
        io = mapO[i]
        iP = mod1(io,Nfp)
        kP = div(io-1,Nfp)+1
        iq = fq2q[iP]
        uP[io] = Uq[iq,kP]
    end
end

function clear_low_order_rhs!(cache,prealloc,param)
    @unpack rhsxyL          = prealloc
    @unpack Q0F1,λarr,λBarr = cache

    K  = get_num_elements(param)
    Nc = get_num_components(param.equation)
    Nd = get_dim(param.equation)
    @batch for k = 1:K
        for i = 1:size(rhsxyL,1)
            rhsxyL[i,k] = zero(rhsxyL[i,k])
            Q0F1[i,k]   = zero(Q0F1[i,k])
        end
    end
end

function accumulate_low_order_rhs_volume!(cache,prealloc,param,discrete_data)
    @unpack equation       = param
    @unpack rhsxyL,Uq      = prealloc
    @unpack flux,λarr,Q0F1 = cache
    @unpack Srs0_nnz       = discrete_data.ops
    
    dim = get_dim_type(equation)
    K  = get_num_elements(param)
    Nq = size(Q0F1,1)
    @batch for k = 1:K
        # Volume contributions
        for (i,j) in Srs0_nnz
            u_i = Uq[i,k]
            u_j = Uq[j,k]
            Fxyij = @. .5*(flux[i,k]+flux[j,k])
            # TODO: assume Sxy0J_ij = -Sxy0J_ji
            #              n_ij_norm = n_ji_norm
            #              Sxy0J_ji,n_ji_norm = get_Sx0_with_n(j,i,k,discrete_data,dim)
            Sxy0J_ij,n_ij_norm = get_Sx0_with_n(i,j,k,discrete_data,dim)
            n_ij = Sxy0J_ij./n_ij_norm
            n_ji = -n_ij
            wavespeed_ij = max(wavespeed_estimate(equation,u_i,n_ij),wavespeed_estimate(equation,u_j,n_ji))
            λarr[i,j,k] = n_ij_norm*wavespeed_ij
            λarr[j,i,k] = λarr[i,j,k]
            ΛD_ij = get_graph_viscosity(cache,prealloc,param,i,j,k,Sxy0J_ij,dim)
            SFxy_ΛD_ij = 2.0*Sxy0J_ij.*Fxyij - ΛD_ij
            SFxy_ΛD_ji = -SFxy_ΛD_ij
            Q0F1[i,k] += SFxy_ΛD_ij
            Q0F1[j,k] += SFxy_ΛD_ji
        end
    end

    @batch for k = 1:K
        for i = 1:Nq
            rhsxyL[i,k] -= Q0F1[i,k]
        end
    end
end

function accumulate_low_order_rhs_surface!(cache,prealloc,param,discrete_data,bcdata)
    @unpack equation = param
    @unpack rhsxyL,BF_L,fstar_L       = prealloc
    @unpack Uf,uP,flux,wavespeed_f,λBarr = cache
    @unpack mapP                      = bcdata
    @unpack fq2q                      = discrete_data.ops

    K  = get_num_elements(param)
    dim = get_dim_type(equation)
    Nd  = get_dim(equation)
    Nq  = size(prealloc.Uq,1)
    Nfp = size(bcdata.mapP,1)
    @batch for k = 1:K
        # Surface contributions
        for i = 1:Nfp
            # TODO: refactor
            idx = i+Nfp*(k-1)
            iP = mod1(mapP[i,k],Nfp)
            kP = div(mapP[i,k]-1,Nfp)+1

            Bxy_i,n_i_norm = get_Bx_with_n(i,k,discrete_data,dim)
            λBarr[i,k] = .5*n_i_norm*max(wavespeed_f[i,k],wavespeed_f[iP,kP])

            flux_xy_P = fluxes(equation,uP[i,k])
            fstar_L[i,k] = .5 .*(flux[i+Nq,k].+flux_xy_P)
            BF_L[i,k] = Bxy_i.*fstar_L[i,k]
            
            lf = λBarr[i,k]*(uP[i,k]-Uf[i,k])
            apply_LF_dissipation_to_BF(BF_L,param,i,k,lf,get_dim_type(param.equation))
            apply_LF_dissipation_to_fstar(fstar_L,param,i,k,Bxy_i,lf,get_dim_type(param.equation))

            iq = fq2q[i]
            rhsxyL[iq,k] -= BF_L[i,k]
        end
    end
end

function scale_low_order_rhs_by_mass!(prealloc,param,discrete_data)
    @unpack Jq = discrete_data.geom
    @unpack wq = discrete_data.ops
    @unpack rhsL,rhsxyL = prealloc

    K  = get_num_elements(param)
    @batch for k = 1:K
        # Divide by mass
        for i = 1:size(rhsxyL,1)
            wJq_i    = Jq[i,k]*wq[i]
            rhsxyL[i,k] = rhsxyL[i,k]/wJq_i
            rhsL[i,k] = sum(rhsxyL[i,k])
        end
    end
end

function calculate_lambda_and_low_order_CFL!(cache,prealloc,param,discrete_data,bcdata,t)
    @unpack CFL,dt0,T = param.timestepping_param
    @unpack Jq        = discrete_data.geom
    @unpack wq        = discrete_data.ops
    @unpack dtarr     = cache

    K  = get_num_elements(param)
    Nq  = size(prealloc.Uq,1)
    surface_flux_type = get_low_order_surface_flux(param.rhs_type)
    @. dtarr = min(CFL*dt0,T-t)
    @batch for k = 1:K
        tid = Threads.threadid()
        dt = dtarr[tid]
        accumulate_alpha!(cache,prealloc,k,param,discrete_data,surface_flux_type)
        for i = 1:Nq
            lambda_i = get_lambda_i(i,k,cache,prealloc,param,discrete_data,bcdata)
            wq_i  = wq[i]
            wJq_i = Jq[i,k]*wq_i
            dt = min(dt, CFL*.5*wJq_i/lambda_i)
        end
        dtarr[tid] = dt
    end
    return minimum(dtarr)
end

function accumulate_alpha!(cache,prealloc,k,param,discrete_data,surface_flux_type::LaxFriedrichsOnNodalVal)
    return nothing
end

function accumulate_alpha!(cache,prealloc,k,param,discrete_data,surface_flux_type::LaxFriedrichsOnProjectedVal)
    @unpack αarr       = cache
    @unpack Uq,u_tilde = prealloc
    @unpack fq2q       = discrete_data.ops
    @unpack equation   = param

    Nq  = size(Uq,1)
    Nfp = size(αarr,1)
    Nh  = Nq+Nfp
    utilde_f    = @view u_tilde[Nq+1:Nh,:]
    for i = 1:Nfp
        # TODO: preallocate into Fmask, refactor
        iq = fq2q[i]
        αarr[i,k] = find_alpha(equation,param,Uq[iq,k],utilde_f[i,k])
    end
end

function get_lambda_i(i,k,cache,prealloc,param,discrete_data,bcdata)
    @unpack equation = param
    @unpack λarr     = cache
    @unpack q2fq     = discrete_data.ops

    dim = get_dim_type(equation)
    Nq = size(prealloc.Uq,1)
    Nfp = size(bcdata.mapP,1)

    lambda_i = 0.0
    # TODO: can only consider upper diagonal
    for j = 1:Nq
        lambda_i += λarr[i,j,k]
    end
    
    surface_flux_type = get_low_order_surface_flux(param.rhs_type)
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
function find_alpha(equation::CompressibleIdealGas,param,ui,uitilde)
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

# TODO: hardcoded
function find_alpha(equation::KPP,param,ui,uitilde)
    return 1.0
end

function check_low_order_entropy_stability(cache,prealloc,param,discrete_data,dim::Dim2)
    @unpack equation          = param
    @unpack rhsxyL,fstar_L,vq = prealloc
    @unpack Uf                = cache
    @unpack Jq                = discrete_data.geom
    @unpack wq                = discrete_data.ops
    @unpack Nq,Nfp            = discrete_data.sizes
    
    K  = get_num_elements(param)
    Nd = get_dim(equation)
    dim = get_dim_type(equation)
    @batch for k = 1:K
        entropy_estimate = zero(SVector{Nd,Float64})   # vT rhs
        # TODO: hardcoded
        update_face_values!(cache,prealloc,k,discrete_data,LaxFriedrichsOnNodalVal())
        for i = 1:Nq
            m_i = wq[i]
            entropy_estimate += m_i*Jq[i,k]*SVector(sum(vq[i,k].*rhsxyL[i,k][1]), sum(vq[i,k].*rhsxyL[i,k][2]))
        end
        sum_Bpsi = zero(SVector{Nd,Float64})   # 1T B psi
        vTBfstar = zero(SVector{Nd,Float64})   # vT B f*
        for i = 1:Nfp
            Bxy_i = get_Bx(i,k,discrete_data,dim)
            sum_Bpsi += Bxy_i .* psi_ufun(equation,Uf[i,k])
            vi = v_ufun(equation,Uf[i,k])
            vTBfstar += Bxy_i .* SVector(sum(vi.*fstar_L[i,k][1]), sum(vi.*fstar_L[i,k][2]))
        end
        diff = entropy_estimate-sum_Bpsi+vTBfstar
        tol = 1e-12
        if diff[1] > tol || diff[2] > tol
            @show k,diff
        end
    end
end
