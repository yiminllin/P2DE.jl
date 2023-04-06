###############################
### Subcell limiter methods ###
###############################
function initialize_bounds!(cache,prealloc,equation::CompressibleIdealGas,bound_type::PositivityBound,param,discrete_data,bcdata,t,nstage,dim)
    cache.lbound_s_modified .= 0.0
end

function initialize_bounds!(cache,prealloc,equation::KPP,bound_type,param,discrete_data,bcdata,t,nstage,dim)
    # Do nothing
end

# TODO: only precompute s_modified now, unnecessary to precompute bound for
#       density and internal energy?
function initialize_bounds!(cache,prealloc,equation::CompressibleIdealGas,bound_type::Union{PositivityAndMinEntropyBound,PositivityAndRelaxedMinEntropyBound},param,discrete_data,bcdata,t,nstage,dim)
    initialize_s_modified!(cache,prealloc,param,t,nstage)
    initialize_lower_bound!(cache,prealloc,param,discrete_data,bcdata,nstage,dim)
end

function initialize_s_modified!(cache,prealloc,param,t,nstage)
    @unpack equation   = param
    @unpack t0         = param.timestepping_param
    @unpack s_modified = cache
    @unpack s_modified_min = cache
    @unpack Uq         = prealloc

    N1D = param.N+1
    K  = get_num_elements(param)
    Nq = size(Uq,1)
    # Preallocate s_modified at nodes
    @batch for k = 1:K
        for i = 1:Nq
            s_modified[i,k] = s_modified_ufun(equation,Uq[i,k])
        end
    end
    # If at the first time step, initialize minimum s_modified of the initial condition
    if t == t0 && nstage == 1
        s_modified_min[1] = minimum(s_modified)
    end
end

function initialize_lower_bound!(cache,prealloc,param,discrete_data,bcdata,nstage,dim::Dim1)
    @unpack equation   = param
    @unpack Uq         = prealloc
    @unpack mapP       = bcdata
    @unpack q2fq,fq2q  = discrete_data.ops
    @unpack s_modified,s_modified_min,lbound_s_modified,smooth_factor = cache

    N1D = param.N+1
    K  = get_num_elements(param)
    Nq = size(Uq,1)
    Nfp = size(mapP,1)
    @batch for k = 1:K
        epsk = smooth_factor[k,nstage]
        for i = 1:N1D
            stencil = get_low_order_stencil(i,k,N1D,Nfp,discrete_data,bcdata,dim)
            lbound_s_modified[i,k] = s_modified[i,k]
            for s in stencil
                lbound_s_modified[i,k]  = min(lbound_s_modified[i,k],s_modified[s...])
            end
            # (70) in https://arxiv.org/pdf/2004.08503.pdf
            lbound_s_modified[i,k] = epsk*lbound_s_modified[i,k]+(1-epsk)*s_modified_min[1]
        end
    end
end

function initialize_lower_bound!(cache,prealloc,param,discrete_data,bcdata,nstage,dim::Dim2)
    @unpack equation   = param
    @unpack Uq         = prealloc
    @unpack mapP       = bcdata
    @unpack q2fq,fq2q  = discrete_data.ops
    @unpack s_modified,s_modified_min,lbound_s_modified,smooth_factor = cache

    N1D = param.N+1
    K  = get_num_elements(param)
    Nq = size(Uq,1)
    Nfp = size(mapP,1)
    s_modified = reshape(s_modified,N1D,N1D,K)
    @batch for k = 1:K
        lbound_s_modified_k = reshape(view(lbound_s_modified,:,k),N1D,N1D)
        epsk = smooth_factor[k,nstage]
        for j = 1:N1D
            for i = 1:N1D
                stencil = get_low_order_stencil((i,j),k,N1D,Nfp,discrete_data,bcdata,dim)
                lbound_s_modified_k[i,j] = s_modified[i,j,k]
                for s in stencil
                    lbound_s_modified_k[i,j]  = min(lbound_s_modified_k[i,j],s_modified[s...])
                end
                # (70) in https://arxiv.org/pdf/2004.08503.pdf
                lbound_s_modified_k[i,j] = epsk*lbound_s_modified_k[i,j]+(1-epsk)*s_modified_min[1]
            end
        end
    end
end

# TODO: documentation... from the ipad note
function accumulate_f_bar!(cache,prealloc,param,discrete_data,dim::Dim1)
    @unpack f_bar_H,f_bar_L     = cache
    @unpack rhsL,rhsH,BF_H,BF_L = prealloc
    @unpack wq = discrete_data.ops
    @unpack Jq = discrete_data.geom

    K  = get_num_elements(param)
    Nq = size(prealloc.Uq,1)
    # TODO: f_bar_H, f_bar_L could be combine into a single cache?
    @batch for k = 1:K
        f_bar_H[1][1,k] = BF_H[1,k][1]
        f_bar_L[1][1,k] = BF_L[1,k][1]
        for i = 2:Nq+1
            f_bar_H[1][i,k] = f_bar_H[1][i-1,k]+Jq[i-1,k]*wq[i-1]*rhsH[i-1,k]
            f_bar_L[1][i,k] = f_bar_L[1][i-1,k]+Jq[i-1,k]*wq[i-1]*rhsL[i-1,k]
        end
    end
end

# TODO: use views instead of index flattening
function accumulate_f_bar!(cache,prealloc,param,discrete_data,dim::Dim2)
    @unpack f_bar_H,f_bar_L                   = cache
    @unpack rhsL,rhsH,rhsxyH,rhsxyL,BF_H,BF_L = prealloc
    @unpack wq = discrete_data.ops
    @unpack Jq = discrete_data.geom
    
    K   = get_num_elements(param)
    Nq  = size(prealloc.Uq,1)
    N1D = param.N+1    # TODO: hardcoded
    N1Dp1 = N1D+1
    @batch for k = 1:K
        # TODO: hardcoding views
        fx_bar_H_k = reshape(view(f_bar_H[1],:,k),N1Dp1,N1D)
        fx_bar_L_k = reshape(view(f_bar_L[1],:,k),N1Dp1,N1D)
        fy_bar_H_k = reshape(view(f_bar_H[2],:,k),N1D,N1Dp1)
        fy_bar_L_k = reshape(view(f_bar_L[2],:,k),N1D,N1Dp1)

        rhsxyH_k = reshape(view(rhsxyH,:,k),N1D,N1D)
        rhsxyL_k = reshape(view(rhsxyL,:,k),N1D,N1D)

        wq_k = reshape(view(wq,:),N1D,N1D)
        Jq_k = reshape(view(Jq,:,k),N1D,N1D)

        # For each stride along x direction
        for sj = 1:N1D
            iface = sj
            fx_bar_H_k[1,sj] = BF_H[iface,k][1]
            fx_bar_L_k[1,sj] = BF_L[iface,k][1]
            for si = 2:N1Dp1
                fx_bar_H_k[si,sj] = fx_bar_H_k[si-1,sj] + wq_k[si-1,sj]*Jq_k[si-1,sj]*rhsxyH_k[si-1,sj][1]
                fx_bar_L_k[si,sj] = fx_bar_L_k[si-1,sj] + wq_k[si-1,sj]*Jq_k[si-1,sj]*rhsxyL_k[si-1,sj][1]
            end
        end

        # For each stride along y direction
        for si = 1:N1D
            iface = si+2*N1D
            fy_bar_H_k[si,1] = BF_H[iface,k][2]
            fy_bar_L_k[si,1] = BF_L[iface,k][2]
            for sj = 2:N1Dp1
                fy_bar_H_k[si,sj] = fy_bar_H_k[si,sj-1] + wq_k[si,sj-1]*Jq_k[si,sj-1]*rhsxyH_k[si,sj-1][2]
                fy_bar_L_k[si,sj] = fy_bar_L_k[si,sj-1] + wq_k[si,sj-1]*Jq_k[si,sj-1]*rhsxyL_k[si,sj-1][2]
            end
        end
    end
end

function subcell_bound_limiter!(limiter_cache,shockcapture_cache,prealloc,equation::CompressibleIdealGas,param,discrete_data,bcdata,dt,nstage,dim::Dim1)
    @unpack uL_k,f_bar_H,f_bar_L = limiter_cache
    @unpack lbound_s_modified    = limiter_cache
    @unpack Uq,L_local_arr,rhsL  = prealloc
    @unpack blending_factor      = shockcapture_cache
    @unpack wq = discrete_data.ops
    @unpack Jq = discrete_data.geom
    @unpack rhs_limiter_type = param
    bound_type = get_bound_type(param)
    
    K  = get_num_elements(param)
    Nq = size(Uq,1)
    ζ = param.limiting_param.ζ
    Lrho(uL_i)  = ζ*uL_i[1]
    Lrhoe(uL_i) = ζ*rhoe_ufun(param.equation,uL_i)
    @views @. L_local_arr[:,:,:,nstage] = 1.0
    # Calculate limiting parameter
    @batch for k = 1:K
        tid = Threads.threadid()

        @views @. uL_k[:,tid] = Uq[:,k] + dt*rhsL[:,k]
        Urho  = Inf
        Urhoe = Inf
        # TODO: ugly...
        for i = 1:Nq
            wJq_i = (wq[i]*Jq[i,k])
            Lphi_i = lbound_s_modified[i,k]
            bound = (Lrho(uL_k[i,tid]),Lrhoe(uL_k[i,tid]),Lphi_i,Urho,Urhoe)
            L_local_arr[i,1,k,nstage] = min(L_local_arr[i,1,k,nstage], get_limiting_param(rhs_limiter_type,bound_type,param,uL_k[i,tid],-2*dt*(f_bar_H[1][i,k]-f_bar_L[1][i,k])/wJq_i,bound))
        end
        for i = 2:Nq+1
            wJq_im1 = (wq[i-1]*Jq[i-1,k])
            Lphi_i = lbound_s_modified[i-1,k]
            bound = (Lrho(uL_k[i-1,tid]),Lrhoe(uL_k[i-1,tid]),Lphi_i,Urho,Urhoe)
            L_local_arr[i,1,k,nstage] = min(L_local_arr[i,1,k,nstage], get_limiting_param(rhs_limiter_type,bound_type,param,uL_k[i-1,tid],2*dt*(f_bar_H[1][i,k]-f_bar_L[1][i,k])/wJq_im1,bound))
        end

        # Apply shock capturing
        l_shock = blending_factor[k,nstage]
        @. L_local_arr[:,1,k,nstage] = min(L_local_arr[:,1,k,nstage], l_shock)
    end
end

function subcell_bound_limiter!(limiter_cache,shockcapture_cache,equation::CompressibleIdealGas,prealloc,param,discrete_data,bcdata,dt,nstage,dim::Dim2)
    @unpack uL_k,f_bar_H,f_bar_L = limiter_cache
    @unpack lbound_s_modified    = limiter_cache
    @unpack Uq,rhsL,L_local_arr  = prealloc
    @unpack blending_factor      = shockcapture_cache
    @unpack mapP = bcdata
    @unpack wq = discrete_data.ops
    @unpack Jq = discrete_data.geom
    @unpack rhs_limiter_type = param
    bound_type = get_bound_type(param)

    K  = get_num_elements(param)
    Nq = size(Uq,1)
    N1D = param.N+1
    N1Dp1 = N1D+1
    ζ = param.limiting_param.ζ
    Lrho(uL_i)  = ζ*uL_i[1]
    Lrhoe(uL_i) = ζ*rhoe_ufun(param.equation,uL_i)
    # TODO: why these two lines result in allocations?
    # Lx_local = reshape(view(L_local_arr,:,1,:,nstage),N1Dp1,N1D,K)
    # Ly_local = reshape(view(L_local_arr,:,2,:,nstage),N1D,N1Dp1,K)

    @views @. L_local_arr[:,:,:,nstage] = 1.0

    @batch for k = 1:K
        tid = Threads.threadid()

        @views @. uL_k[:,tid] = Uq[:,k] + dt*rhsL[:,k]
        Urho  = Inf
        Urhoe = Inf

        # TODO: hardcoding views
        fx_bar_H_k = reshape(view(f_bar_H[1],:,k),N1Dp1,N1D)
        fx_bar_L_k = reshape(view(f_bar_L[1],:,k),N1Dp1,N1D)
        fy_bar_H_k = reshape(view(f_bar_H[2],:,k),N1D,N1Dp1)
        fy_bar_L_k = reshape(view(f_bar_L[2],:,k),N1D,N1Dp1)

        Lx_local_k = reshape(view(L_local_arr,:,1,k,nstage),N1Dp1,N1D)
        Ly_local_k = reshape(view(L_local_arr,:,2,k,nstage),N1D,N1Dp1)

        lbound_s_modified_k = reshape(view(lbound_s_modified,:,k),N1D,N1D)

        u_L_k = reshape(view(uL_k,:,tid),N1D,N1D)

        wq_k = reshape(view(wq,:),N1D,N1D)
        Jq_k = reshape(view(Jq,:,k),N1D,N1D)

        # For each stride along x direction
        for sj = 1:N1D
            # For each left subcell face
            for si = 1:N1D
                # index of quad node right to subcell face
                iq = si
                jq = sj
                wJq_i = wq_k[iq,jq]*Jq_k[iq,jq]
                uL_k_i = u_L_k[iq,jq]
                Lphi_ij = lbound_s_modified_k[iq,jq]
                bound = (Lrho(uL_k_i),Lrhoe(uL_k_i),Lphi_ij,Urho,Urhoe)
                Lx_local_k[si,sj] = min(Lx_local_k[si,sj], get_limiting_param(rhs_limiter_type,bound_type,param,uL_k_i,-4*dt*(fx_bar_H_k[si,sj]-fx_bar_L_k[si,sj])/wJq_i,bound))
            end
            # For each right subcell face
            for si = 2:N1Dp1
                # index of quad node left to subcell face
                iq = si-1
                jq = sj
                wJq_i = wq_k[iq,jq]*Jq_k[iq,jq]
                uL_k_i = u_L_k[iq,jq]
                Lphi_ij = lbound_s_modified_k[iq,jq]
                bound = (Lrho(uL_k_i),Lrhoe(uL_k_i),Lphi_ij,Urho,Urhoe)
                Lx_local_k[si,sj] = min(Lx_local_k[si,sj], get_limiting_param(rhs_limiter_type,bound_type,param,uL_k_i,4*dt*(fx_bar_H_k[si,sj]-fx_bar_L_k[si,sj])/wJq_i,bound))
            end
        end

        # For each stride along y direction
        for si = 1:N1D
            # For each bottom subcell face
            for sj = 1:N1D
                # index of quad node top to subcell face
                iq = si
                jq = sj
                wJq_i = wq_k[iq,jq]*Jq_k[iq,jq]
                uL_k_i = u_L_k[iq,jq]
                Lphi_ij = lbound_s_modified_k[iq,jq]
                bound = (Lrho(uL_k_i),Lrhoe(uL_k_i),Lphi_ij,Urho,Urhoe)
                Ly_local_k[si,sj] = min(Ly_local_k[si,sj], get_limiting_param(rhs_limiter_type,bound_type,param,uL_k_i,-4*dt*(fy_bar_H_k[si,sj]-fy_bar_L_k[si,sj])/wJq_i,bound))
            end
            # For each top subcell face
            for sj = 2:N1Dp1
                # index of quad node beneath the subcell face
                iq = si
                jq = sj-1
                wJq_i = wq_k[iq,jq]*Jq_k[iq,jq]
                uL_k_i = u_L_k[iq,jq]
                Lphi_ij = lbound_s_modified_k[iq,jq]
                bound = (Lrho(uL_k_i),Lrhoe(uL_k_i),Lphi_ij,Urho,Urhoe)
                Ly_local_k[si,sj] = min(Ly_local_k[si,sj], get_limiting_param(rhs_limiter_type,bound_type,param,uL_k_i,4*dt*(fy_bar_H_k[si,sj]-fy_bar_L_k[si,sj])/wJq_i,bound))
            end
        end

        # Apply shock capturing
        l_shock = blending_factor[k,nstage]
        @. Lx_local_k = min(Lx_local_k, l_shock)
        @. Ly_local_k = min(Ly_local_k, l_shock)
    end
end

function subcell_bound_limiter!(limiter_cache,shockcapture_cache,prealloc,equation::KPP,param,discrete_data,bcdata,dt,nstage,dim)
    @unpack L_local_arr     = prealloc
    @unpack blending_factor = shockcapture_cache

    K  = get_num_elements(param)

    @views @. L_local_arr[:,:,:,nstage] = 1.0
    @batch for k = 1:K
        tid = Threads.threadid()

        L_local_k = view(L_local_arr,:,:,k,nstage)

        # Apply shock capturing
        l_shock = blending_factor[k,nstage]
        @. L_local_k = min(L_local_k, l_shock)
    end
end

function symmetrize_limiting_parameters!(prealloc,param,bcdata,nstage,dim::Dim1)
    @unpack L_local_arr = prealloc 
    K  = get_num_elements(param)

    # Symmetrize limiting parameter TODO: hardcoded, should use mapP
    @batch for k = 1:K
        l = min(L_local_arr[1,1,k,nstage], L_local_arr[end,1,mod1(k-1,K),nstage])
        L_local_arr[1,1,k,nstage] = l
        L_local_arr[end,1,mod1(k-1,K),nstage] = l
    end
end

function symmetrize_limiting_parameters!(prealloc,param,bcdata,nstage,dim::Dim2)
    @unpack L_local_arr = prealloc 

    # TODO: refactor
    K  = get_num_elements(param)
    N1D = param.N+1
    N1Dp1 = N1D+1
    Lx_local = view(L_local_arr,:,1,:,nstage)
    Ly_local = view(L_local_arr,:,2,:,nstage)

    @batch for k = 1:K
        # Symmetrize limiting parameters
        # For each stride in x direction
        for sj = 1:N1D
            # For each subcell index on boundary
            for si = 1:N1D:N1Dp1
                siP,sjP,kP = get_subcell_index_P_x(si,sj,k,N1Dp1,bcdata)
                idx  = si+(sj-1)*N1Dp1
                idxP = siP+(sjP-1)*N1Dp1 
                l = min(Lx_local[idx,k],Lx_local[idxP,kP])
                Lx_local[idx,k]   = l
                Lx_local[idxP,kP] = l
            end
        end

        # For each stride in y direction
        for si = 1:N1D
            # For each subcell index on boundary
            for sj = 1:N1D:N1Dp1
                siP,sjP,kP = get_subcell_index_P_y(si,sj,k,N1Dp1,bcdata)
                idx  = si+(sj-1)*N1D
                idxP = siP+(sjP-1)*N1D 
                l = min(Ly_local[idx,k],Ly_local[idxP,kP])
                Ly_local[idx,k]   = l
                Ly_local[idxP,kP] = l
            end
        end
    end
end

# TODO: not necessary
function accumulate_f_bar_limited!(cache,prealloc,param,nstage,dim::Dim1)
    @unpack f_bar_H,f_bar_L,f_bar_lim = cache
    @unpack L_local_arr               = prealloc
    
    K  = get_num_elements(param)
    Nq = size(prealloc.Uq,1)
    # TODO: f_bar_H, f_bar_L could be combine into a single cache? df_bar?
    @batch for k = 1:K
        for i = 1:Nq+1
            f_bar_lim[1][i,k] = L_local_arr[i,1,k,nstage]*f_bar_H[1][i,k] + (1-L_local_arr[i,1,k,nstage])*f_bar_L[1][i,k]
        end
    end
end

# TODO: not necessary
function accumulate_f_bar_limited!(cache,prealloc,param,nstage,dim::Dim2)
    @unpack f_bar_H,f_bar_L,f_bar_lim = cache
    @unpack L_local_arr               = prealloc

    K  = get_num_elements(param)
    Nq = size(prealloc.Uq,1)
    N1D = param.N+1    # TODO: hardcoded
    N1Dp1 = N1D+1
    @batch for k = 1:K
        # TODO: hardcoding views
        fx_bar_H_k = reshape(view(f_bar_H[1],:,k),N1Dp1,N1D)
        fx_bar_L_k = reshape(view(f_bar_L[1],:,k),N1Dp1,N1D)
        fy_bar_H_k = reshape(view(f_bar_H[2],:,k),N1D,N1Dp1)
        fy_bar_L_k = reshape(view(f_bar_L[2],:,k),N1D,N1Dp1)
        fx_bar_lim_k = reshape(view(f_bar_lim[1],:,k),N1Dp1,N1D)
        fy_bar_lim_k = reshape(view(f_bar_lim[2],:,k),N1D,N1Dp1)

        Lx_local_k = reshape(view(L_local_arr,:,1,k,nstage),N1Dp1,N1D)
        Ly_local_k = reshape(view(L_local_arr,:,2,k,nstage),N1D,N1Dp1)

         # For each stride along x direction
        for j = 1:N1D
            for i = 1:N1Dp1
                l = Lx_local_k[i,j]
                fx_bar_lim_k[i,j] = l*fx_bar_H_k[i,j] + (1-l)*fx_bar_L_k[i,j]
            end
        end

        # For each stride along y direction
        for i = 1:N1D
            for j = 1:N1Dp1
                l = Ly_local_k[i,j]
                fy_bar_lim_k[i,j] = l*fy_bar_H_k[i,j] + (1-l)*fy_bar_L_k[i,j]
            end
        end       
    end
end

function apply_subcell_limiter!(prealloc,cache,param,discrete_data,dim::Dim1)
    @unpack rhsU      = prealloc
    @unpack f_bar_lim = cache
    @unpack wq = discrete_data.ops
    @unpack Jq = discrete_data.geom
    
    K  = get_num_elements(param)
    Nq = size(prealloc.Uq,1)
    # Update step
    @batch for k = 1:K
        for i = 1:Nq
            wJq_i     = (wq[i]*Jq[i,k])
            rhsU[i,k] = (f_bar_lim[1][i+1,k]-f_bar_lim[1][i,k])/wJq_i
        end
    end
end

function apply_subcell_limiter!(prealloc,cache,param,discrete_data,dim::Dim2)
    @unpack f_bar_lim   = cache
    @unpack rhsU,rhsxyU = prealloc
    @unpack wq = discrete_data.ops
    @unpack Jq = discrete_data.geom

    K  = get_num_elements(param)
    N1D = param.N+1    # TODO: hardcoded
    N1Dp1 = N1D+1
    Nq = size(prealloc.Uq,1)

    # Update step
    @batch for k = 1:K
        # TODO: hardcoding views
        fx_bar_lim_k = reshape(view(f_bar_lim[1],:,k),N1Dp1,N1D)
        fy_bar_lim_k = reshape(view(f_bar_lim[2],:,k),N1D,N1Dp1)
        
        rhsxyU_k = reshape(view(rhsxyU,:,k),N1D,N1D)

        wq_k = reshape(view(wq,:),N1D,N1D)
        Jq_k = reshape(view(Jq,:,k),N1D,N1D)


        for j = 1:N1D
            for i = 1:N1D
                wJq_ij = wq_k[i,j]*Jq_k[i,j]
                rhsxyU_k[i,j] = SVector(fx_bar_lim_k[i+1,j]-fx_bar_lim_k[i,j],
                                        fy_bar_lim_k[i,j+1]-fy_bar_lim_k[i,j])/wJq_ij
            end
        end

        @. @views rhsU[:,k] = sum(rhsxyU[:,k])
    end
end

#########################
### Smoothness factor ###
#########################
# (69) in https://arxiv.org/pdf/2004.08503.pdf
function update_smoothness_factor!(bound_type::PositivityBound,cache,prealloc,param,nstage)
    # Use global minimum bound by default
    @views @. cache.smooth_factor[:,nstage] = 0.0
end

function update_smoothness_factor!(bound_type::PositivityAndMinEntropyBound,cache,prealloc,param,nstage)
    # Use global minimum bound by default
    @views @. cache.smooth_factor[:,nstage] = 1.0
end

function update_smoothness_factor!(bound_type::PositivityAndRelaxedMinEntropyBound,cache,prealloc,param,nstage)
    @unpack N                = param
    @unpack smooth_factor    = cache
    @unpack smooth_indicator = prealloc

    K = get_num_elements(param)
    kappa = 1.0
    s0    = log(10,N^-4)
    @batch for k = 1:K
        sk = log(10,smooth_indicator[k])
        if sk < s0-kappa
            smooth_factor[k,nstage] = 0.0
        elseif sk > s0+kappa
            smooth_factor[k,nstage] = 1.0
        else
            smooth_factor[k,nstage] = .5-.5*sin(pi*(sk-s0)/(2*kappa))
        end
    end
end