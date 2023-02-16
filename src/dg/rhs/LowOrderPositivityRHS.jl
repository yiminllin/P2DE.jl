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

    # Determine positivity CFL
    if (nstage == 1)
        dt = calculate_lambda_and_low_order_CFL!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,t)
    else
        calculate_lambda_and_low_order_CFL!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,t)
    end

    # Assemble RHS
    clear_low_order_rhs!(prealloc,param)
    # TODO: write routine in K loop?
    accumulate_low_order_rhs_volume!(prealloc,param,discrete_data_gauss,discrete_data_LGL,equation)
    accumulate_low_order_rhs_surface!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,equation)
    scale_low_order_rhs_by_mass!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    # check_bar_states!(dt,prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,equation)

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
        update_wavespeed_and_inviscid_flux!(prealloc,k,param,discrete_data,equation)
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

function update_wavespeed_and_inviscid_flux!(prealloc,k,param,discrete_data,equation::EquationType{Dim1})
    @unpack u_tilde,wavespeed,flux_x,Uq,Uf = prealloc
    
    Nq = size(Uq,1)
    for i = 1:size(u_tilde,1)
        ui = (i <= Nq) ? Uq[i,k] : Uf[i-Nq,k]
        wavespeed[i,k] = wavespeed_davis_estimate(equation,ui)
        flux_x[i,k]    = euler_fluxes(equation,ui)
    end
end

# TODO: refactor with high order flux calculation
function update_wavespeed_and_inviscid_flux!(prealloc,k,param,discrete_data,equation::EquationType{Dim2})
    @unpack LGLind,Uq,Uf,wavespeed,flux_x,flux_y,n_i = prealloc
    @unpack Br,Bs,Sr0,Ss0       = discrete_data.ops
    @unpack rxJh,ryJh,sxJh,syJh = discrete_data.geom

    Nq  = size(Uq,1)
    Nfp = size(Uf,1)
    Nh  = size(rxJh,1)
    rxJf,ryJf,sxJf,syJf = (x->view(x,Nq+1:Nh,k)).((rxJh,ryJh,sxJh,syJh))
    # TODO: very ugly way to store wavespeed...
    #       We store volume wavespeed at wavespeed[1:Nq,1:Nq],
    #                where wavespeed[i,j] = \beta(u_i, n_ij) = \beta(u_i, n_ji), since n_ij = -n_ji
    #       and store interface wavespeed at wavespeed[Nq+1,Nq+1] to wavespeed[Nh,Nh]
    #                where wavespeed[i,i] = \beta(u_f,i, n_i)
    # Volume wavespeed and inviscid flux
    for i = 1:Nq
        u_i = Uq[i,k]
        for j = 1:Nq
            Sx0J_ij = rxJh[i,k]*Sr0[i,j]+sxJh[i,k]*Ss0[i,j]
            Sy0J_ij = ryJh[i,k]*Sr0[i,j]+syJh[i,k]*Ss0[i,j]
            n_ij_norm = sqrt(Sx0J_ij^2+Sy0J_ij^2)
            if n_ij_norm > param.global_constants.ZEROTOL
                n_i[1] = Sx0J_ij/n_ij_norm
                n_i[2] = Sy0J_ij/n_ij_norm
                wavespeed[i,j,k] = wavespeed_davis_estimate(equation,u_i,n_i)
            end
        end
        flux_x[i,k],flux_y[i,k] = euler_fluxes(equation,u_i)
    end

    # Surface wavespeed and inviscid flux
    for i = 1:Nfp
        u_i = Uf[i,k]
        Bx_i = rxJf[i]*Br[i,i]+sxJf[i]*Bs[i,i]      # TODO: refactor geometric factor calculation
        By_i = ryJf[i]*Br[i,i]+syJf[i]*Bs[i,i]
        n_i_norm = sqrt(Bx_i^2+By_i^2)
        n_i[1] = Bx_i/n_i_norm
        n_i[2] = By_i/n_i_norm
        wavespeed[i+Nq,i+Nq,k] = wavespeed_davis_estimate(equation,u_i,n_i)
        flux_x[i+Nq,k],flux_y[i+Nq,k] = euler_fluxes(equation,u_i)
    end
end

function calculate_lambda_and_low_order_CFL!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,t)
    @unpack CFL,dt0,T = param.timestepping_param
    @unpack Jq        = discrete_data_gauss.geom
    @unpack αarr,LGLind,wavespeed,Uq,u_tilde,λarr,λBarr = prealloc

    λarr  .= 0.0
    λBarr .= 0.0
    K  = get_num_elements(param)
    Nq  = size(prealloc.Uq,1)
    Nh  = size(prealloc.u_tilde,1)
    Nfp = Nh-Nq
    dt = min(CFL*dt0,T-t)
    for k = 1:K
        discrete_data = LGLind[k] ? discrete_data_LGL : discrete_data_gauss
        accumulate_alpha!(prealloc,k,param,discrete_data)
        for i = 1:Nq
            lambda_i = get_lambda_i(i,k,prealloc,param,discrete_data,bcdata,param.equation)
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

function get_lambda_i(i,k,prealloc,param,discrete_data,bcdata,equation::EquationType{Dim1})
    @unpack wavespeed,LGLind = prealloc
    @unpack Sr0,wq = discrete_data.ops
    @unpack mapP   = bcdata
    @unpack Jq     = discrete_data.geom

    Nq = size(wq,1)
    lambda_i = 0.0
    wq_i  = wq[i]
    wJq_i = Jq[i,k]*wq_i
    for j = 1:Nq
        Sr0_ij = Sr0[i,j]
        if Sr0_ij != 0
            lambda_i += abs(Sr0_ij)*max(wavespeed[i,k], wavespeed[j,k])
        end
    end
    surface_flux_type = LGLind[k] ? LaxFriedrichsOnNodalVal() : get_low_order_surface_flux(param.rhs_type)  # TODO: hardcoding
    # TODO: hardcoded
    if i == 1
        iface = 1
        n_j_norm = get_i_norm(iface,k,prealloc,discrete_data,equation)
        lambda_i += get_lambda_B(prealloc,mapP,iface,n_j_norm,k,discrete_data,equation)
    end
    if i == Nq
        iface = 2
        n_j_norm = get_i_norm(iface,k,prealloc,discrete_data,equation)
        lambda_i += get_lambda_B(prealloc,mapP,iface,n_j_norm,k,discrete_data,equation)
    end

    return lambda_i
end

function get_lambda_i(i,k,prealloc,param,discrete_data,bcdata,equation::EquationType{Dim2})
    @unpack equation = param
    @unpack λarr,λBarr,wavespeed,LGLind = prealloc
    @unpack Sr0,Ss0,Br,Bs,wq = discrete_data.ops
    @unpack mapP = bcdata
    @unpack Jq,rxJh,ryJh,sxJh,syJh = discrete_data.geom

    Nq = size(wq,1)
    Nh = size(rxJh,1)
    Nfp = size(mapP,1)
    lambda_i = 0.0
    wq_i  = wq[i]
    wJq_i = Jq[i,k]*wq_i
    
    # TODO: can only consider upper diagonal
    for j = 1:Nq
        Sx0J_ij = rxJh[i,k]*Sr0[i,j]+sxJh[i,k]*Ss0[i,j]
        Sy0J_ij = ryJh[i,k]*Sr0[i,j]+syJh[i,k]*Ss0[i,j]
        n_ij_norm = sqrt(Sx0J_ij^2+Sy0J_ij^2)
        if n_ij_norm > param.global_constants.ZEROTOL
            λarr[i,j,k] = n_ij_norm*max(wavespeed[i,j,k],wavespeed[j,i,k]) 
            lambda_i += λarr[i,j,k]
        end
    end
    
    surface_flux_type = LGLind[k] ? LaxFriedrichsOnNodalVal() : get_low_order_surface_flux(param.rhs_type)
    face_idx_arr = findall(x->x==1.0, view(discrete_data.ops.Vf_low,:,i))
    for j in face_idx_arr
        n_j_norm = get_i_norm(j,k,prealloc,discrete_data,equation)
        λBarr[j,k] = get_lambda_B(prealloc,mapP,j,n_j_norm,k,discrete_data,equation)
        lambda_i += get_lambda_B_CFL(prealloc,j,n_j_norm,k,equation,surface_flux_type)
    end

    return lambda_i
end

# TODO: hardcoded
function get_i_norm(i,k,prealloc,discrete_data,equation::EquationType{Dim1})
    return 1.0
end

# TODO: precompute
function get_i_norm(i,k,prealloc,discrete_data,equation::EquationType{Dim2})
    @unpack rxJh,ryJh,sxJh,syJh = discrete_data.geom
    @unpack Br,Bs = discrete_data.ops
    
    Nq  = size(prealloc.Uq,1)
    Nh  = size(prealloc.u_tilde,1)
    rxJf,ryJf,sxJf,syJf = (x->view(x,Nq+1:Nh,k)).((rxJh,ryJh,sxJh,syJh))

    Bx_i = rxJf[i]*Br[i,i]+sxJf[i]*Bs[i,i]
    By_i = ryJf[i]*Br[i,i]+syJf[i]*Bs[i,i]

    return sqrt(Bx_i^2+By_i^2)
end

# TODO: refactor
function get_lambda_B(prealloc,mapP,i,n_i_norm,k,discrete_data,equation::EquationType{Dim1})
    @unpack wavespeed = prealloc
    
    Nq  = size(prealloc.Uq,1)
    Nh  = size(prealloc.u_tilde,1)
    Nfp = size(mapP,1)
    wavespeed_f = @view wavespeed[Nq+1:Nh,:]

    return .5*max(wavespeed_f[i,k],wavespeed_f[mapP[i+(k-1)*Nfp]])
end

# TODO: refactor
function get_lambda_B(prealloc,mapP,i,n_i_norm,k,discrete_data,equation::EquationType{Dim2})
    @unpack wavespeed = prealloc
    @unpack rxJh,ryJh,sxJh,syJh = discrete_data.geom
    @unpack Br,Bs = discrete_data.ops

    Nq  = size(prealloc.Uq,1)
    Nh  = size(prealloc.u_tilde,1)
    Nfp = size(mapP,1)
    wavespeed_f = @view wavespeed[Nq+1:Nh,Nq+1:Nh,:]
    
    iP = mod1(mapP[i,k],Nfp)
    kP = div(mapP[i,k]-1,Nfp)+1

    return .5*n_i_norm*max(wavespeed_f[i,i,k],wavespeed_f[iP,iP,kP])
end

function get_lambda_B_CFL(prealloc,i,n_i_norm,k,equation,surface_flux_type::LaxFriedrichsOnNodalVal)
    return prealloc.λBarr[i,k]
end

function get_lambda_B_CFL(prealloc,i,n_i_norm,k,equation::EquationType{Dim1},surface_flux_type::LaxFriedrichsOnProjectedVal)
    @unpack λBarr,αarr,wavespeed = prealloc
    
    Nq  = size(prealloc.Uq,1)
    Nh  = size(prealloc.u_tilde,1)
    wavespeed_f = @view wavespeed[Nq+1:Nh,:]

    return αarr[i,k]*λBarr[i,k] + .5*wavespeed_f[i,k]
end

function get_lambda_B_CFL(prealloc,i,n_i_norm,k,equation::EquationType{Dim2},surface_flux_type::LaxFriedrichsOnProjectedVal)
    @unpack λBarr,αarr,wavespeed = prealloc

    Nq  = size(prealloc.Uq,1)
    Nh  = size(prealloc.u_tilde,1)
    wavespeed_f = @view wavespeed[Nq+1:Nh,Nq+1:Nh,:]

    return αarr[i,k]*λBarr[i,k] + .5*n_i_norm*wavespeed_f[i,i,k]
end

function clear_low_order_rhs!(prealloc,param)
    K  = get_num_elements(param)
    Nc = get_num_components(param.equation)
    rhsL = prealloc.rhsL
    # Assemble RHS
    for k = 1:K
        for i = 1:size(rhsL,1)
            rhsL[i,k] = zero(SVector{Nc,Float64})
        end
    end
end

function accumulate_low_order_rhs_volume!(prealloc,param,discrete_data_gauss,discrete_data_LGL,equation::EquationType{Dim1})
    @unpack Uq,rhsL,flux_x,wavespeed,LGLind = prealloc
    
    K  = get_num_elements(param)
    Nq = size(Uq,1)
    for k = 1:K
        # Volume contributions
        for j = 1:Nq
            for i = j+1:Nq
                Sr0_ij = LGLind[k] ? discrete_data_LGL.ops.Sr0[i,j] : discrete_data_gauss.ops.Sr0[i,j]
                if Sr0_ij != 0
                    Fij      = .5*(flux_x[i,k]+flux_x[j,k])
                    lambdaij = abs(Sr0_ij)*max(wavespeed[i,k], wavespeed[j,k])
                    LFij     = (2.0*Sr0_ij*Fij) - lambdaij*(Uq[j,k]-Uq[i,k])
                    rhsL[i,k] -= LFij
                    rhsL[j,k] += LFij
                end
            end
        end
    end
end

function accumulate_low_order_rhs_volume!(prealloc,param,discrete_data_gauss,discrete_data_LGL,equation::EquationType{Dim2})
    @unpack Uq,rhsL,flux_x,flux_y,wavespeed,LGLind,λarr = prealloc
    
    K  = get_num_elements(param)
    Nq = size(Uq,1)
    for k = 1:K
        discrete_data = LGLind[k] ? discrete_data_LGL : discrete_data_gauss
        @unpack rxJh,ryJh,sxJh,syJh = discrete_data.geom
        @unpack Sr0,Ss0 = discrete_data.ops
        # Volume contributions
        for j = 1:Nq
            for i = j+1:Nq
                Sx0J_ij = rxJh[i,k]*Sr0[i,j]+sxJh[i,k]*Ss0[i,j]
                Sy0J_ij = ryJh[i,k]*Sr0[i,j]+syJh[i,k]*Ss0[i,j]
                n_ij_norm = sqrt(Sx0J_ij^2+Sy0J_ij^2)
                if n_ij_norm > param.global_constants.ZEROTOL
                    Fxij = .5*(flux_x[i,k]+flux_x[j,k])
                    Fyij = .5*(flux_y[i,k]+flux_y[j,k])
                    lambdaij = λarr[i,j,k]
                    LFij = 2.0*(Sx0J_ij*Fxij+Sy0J_ij*Fyij) - lambdaij*(Uq[j,k]-Uq[i,k])
                    rhsL[i,k] -= LFij
                    rhsL[j,k] += LFij
                end
            end
        end
    end
end

function accumulate_low_order_rhs_surface!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,equation::EquationType{Dim1})
    @unpack equation = param
    @unpack Uq,Uf,rhsL,flux_x,flux_L,wavespeed,LGLind,u_tilde = prealloc
    @unpack mapP,mapI,mapO,inflowarr                     = bcdata

    K  = get_num_elements(param)
    Nq  = size(Uq,1)
    Nh  = size(u_tilde,1)
    Nfp = size(bcdata.mapP,1)
    flux_f      = @view flux_x[Nq+1:Nh,:]
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

# TODO: refactor
function accumulate_low_order_rhs_surface!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,equation::EquationType{Dim2})
    @unpack equation = param
    @unpack Uq,Uf,rhsL,flux_x,flux_y,flux_L,wavespeed,LGLind,u_tilde,λBarr = prealloc
    @unpack mapP,mapI,mapO,inflowarr = bcdata

    K  = get_num_elements(param)
    Nq  = size(Uq,1)
    Nh  = size(u_tilde,1)
    Nfp = size(bcdata.mapP,1)
    flux_x_f      = @view flux_x[Nq+1:Nh,:]
    flux_y_f      = @view flux_y[Nq+1:Nh,:]
    wavespeed_f = @view wavespeed[Nq+1:Nh,Nq+1:Nh,:]
    for k = 1:K
        # Surface contributions
        discrete_data = LGLind[k] ? discrete_data_LGL : discrete_data_gauss
        @unpack rxJh,ryJh,sxJh,syJh = discrete_data.geom
        @unpack Br,Bs = discrete_data.ops
        rxJf,ryJf,sxJf,syJf = (x->view(x,Nq+1:Nh,k)).((rxJh,ryJh,sxJh,syJh))
        for i = 1:Nfp
            # TODO: refactor
            idx = i+Nfp*(k-1)
            iP = mod1(mapP[i,k],Nfp)
            kP = div(mapP[i,k]-1,Nfp)+1
            Iidx = findfirst(x->(x==idx), mapI)
            Oidx = findfirst(x->(x==idx), mapO)
            flux_x_P,flux_y_P = !isnothing(Iidx) ? euler_fluxes(equation,inflowarr[Iidx]) : flux_x_f[iP,kP],flux_y_f[iP,kP]
            uP = !isnothing(Iidx) ? inflowarr[Iidx] : Uf[iP,kP]
            λB = (!isnothing(Iidx) || !isnothing(Oidx)) ? 0.0 : λBarr[i,k]

            Bx_i = rxJf[i]*Br[i,i]+sxJf[i]*Bs[i,i]
            By_i = ryJf[i]*Br[i,i]+syJf[i]*Bs[i,i]
            n_i_norm = .5*sqrt(Bx_i^2+By_i^2)
            flux_L[i,k] = .5*(Bx_i*(flux_x_f[idx]+flux_x_P)+By_i*(flux_y_f[idx]+flux_y_P))-λB*(uP-Uf[idx])

            iq = findfirst(x->x==1.0, view(discrete_data.ops.Vf_low,i,:))
            rhsL[iq,k] -= flux_L[i,k]
        end
    end
end

function scale_low_order_rhs_by_mass!(prealloc,param,discrete_data_gauss,discrete_data_LGL)
    @unpack rhsL,LGLind = prealloc
    @unpack Jq = discrete_data_gauss.geom

    K  = get_num_elements(param)
    for k = 1:K
        discrete_data = LGLind[k] ? discrete_data_LGL : discrete_data_gauss
        # Divide by mass
        for i = 1:size(rhsL,1)
            wq_i = discrete_data.ops.wq[i]
            wJq_i    = Jq[i,k]*wq_i
            rhsL[i,k] = rhsL[i,k]/wJq_i
        end
    end
end

# TODO: only works in 1D
function check_bar_states!(dt,prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,equation::EquationType{Dim1})
    @unpack Uq,rhsL,flux_x,flux_L,wavespeed,LGLind,u_tilde = prealloc
    @unpack Jq = discrete_data_gauss.geom
    @unpack mapP = bcdata

    K  = get_num_elements(param)
    Nq  = size(Uq,1)
    Nh  = size(u_tilde,1)
    Nfp = size(bcdata.mapP,1)
    flux_f      = @view flux_x[Nq+1:Nh,:]
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
                    rhsL_i -= (Sr0_ij*(flux_x[j,k]+flux_x[i,k])-lambda_ij*(Uq[j,k]-Uq[i,k]))/wJq_i
                end
            end
            # boundary 
            if i == 1
                lambda_ij = .5*max(wavespeed[1,k],wavespeed[Nq,mod1(k-1,K)])
                lambda_i += lambda_ij
                rhsL_i -= (-.5*(flux_x[1,k]+flux_x[Nq,mod1(k-1,K)])-lambda_ij*(Uq[Nq,mod1(k-1,K)]-Uq[1,k]))/wJq_i
            end
            if i == Nq
                lambda_ij = .5*max(wavespeed[Nq,k],wavespeed[1,mod1(k+1,K)])
                lambda_i += lambda_ij
                rhsL_i -= (.5*(flux_x[Nq,k]+flux_x[1,mod1(k+1,K)])-lambda_ij*(Uq[1,mod1(k+1,K)]-Uq[Nq,k]))/wJq_i
            end

            # Using bar states
            for j = 1:Nq
                Sr0_ij = discrete_data_gauss.ops.Sr0[i,j]
                if (Sr0_ij != 0)
                    lambda_ij = abs(Sr0_ij)*max(wavespeed[i,k], wavespeed[j,k])
                    ubar_ij = .5*(Uq[i,k]+Uq[j,k])-.5*Sr0_ij/lambda_ij*(flux_x[j,k]-flux_x[i,k])
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
                ubar_ij = .5*(Uq[1,k]+Uq[Nq,mod1(k-1,K)])-.5*Br0_i/lambda_ij*(flux_x[Nq,mod1(k-1,K)]-flux_x[1,k])
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
                ubar_ij = .5*(Uq[Nq,k]+Uq[1,mod1(k+1,K)])-.5*Br0_i/lambda_ij*(flux_x[1,mod1(k+1,K)]-flux_x[Nq,k])
                ubar_ij_L = .5*Uq[Nq,k]+.5*Br0_i/lambda_ij*flux_x[Nq,k]
                ubar_L    = 2*lambda_ij*Uq[Nq,k] + flux_x[Nq,k]
                ubar_ij_R = .5*Uq[1,mod1(k+1,K)]-.5*Br0_i/lambda_ij*flux_x[1,mod1(k+1,K)]
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