###############################
### Subcell limiter methods ###
###############################
function initialize_bounds!(cache,prealloc,equation::CompressibleIdealGas,bound_type::Union{PositivityBound,PositivityAndCellEntropyBound,PositivityAndRelaxedCellEntropyBound},param,discrete_data,bcdata,t,nstage,dim)
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

function subcell_bound_limiter!(limiter_cache,shockcapture_cache,prealloc,equation::CompressibleIdealGas,param,discrete_data,bcdata,dt,nstage,dim::Dim2)
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

function enforce_ES_subcell!(cache,prealloc,param,discrete_data,bcdata,nstage,bound_type::Union{PositivityBound,PositivityAndMinEntropyBound,PositivityAndRelaxedMinEntropyBound},dim)
    # Do nothing
end

function enforce_ES_subcell!(cache,prealloc,param,discrete_data,bcdata,nstage,bound_type::Union{PositivityAndCellEntropyBound,PositivityAndRelaxedCellEntropyBound},dim)
    initialize_ES_subcell_limiting!(cache,prealloc,param,discrete_data,bcdata,nstage,dim)
    enforce_ES_subcell_volume!(cache,prealloc,param,discrete_data,bcdata,nstage,dim)
    enforce_ES_subcell_interface!(cache,prealloc,param,discrete_data,bcdata,nstage,param.approximation_basis_type,dim)
end

function initialize_ES_subcell_limiting!(cache,prealloc,param,discrete_data,bcdata,nstage,dim::Dim1)
    @unpack equation = param
    @unpack Uq,vq    = prealloc
    @unpack fq2q     = discrete_data.ops
    @unpack Nfp,Nq   = discrete_data.sizes
    @unpack vf,psif,dvdf,f_bar_H,f_bar_L,sum_Bpsi,sum_dvfbarL = cache

    K     = get_num_elements(param)
    @batch for k = 1:K
        # TODO: redundant
        for i = 1:Nq
            vq[i,k] = v_ufun(equation,Uq[i,k])
        end
        sum_Bpsi[k] = zero(sum_Bpsi[k])
        for i = 1:Nfp
            iq = fq2q[i]
            uf = Uq[iq,k]
            vf[i,k]   = v_ufun(equation,uf)
            psif[i,k] = psi_ufun(equation,uf)
            Bxy_i        = get_Bx(i,k,discrete_data,dim)
            sum_Bpsi[k] += Bxy_i.*psif[i,k]
        end

        # TODO: hardcoding views
        dvdf_k    = view(dvdf[1],:,k)
        vq_k      = view(vq,:,k)
        f_bar_H_k = view(f_bar_H[1],:,k)
        f_bar_L_k = view(f_bar_L[1],:,k)
        
        sum_dvfbarL_k = 0.0
        for si = 2:param.N+1
            fxL = f_bar_L_k[si]
            fxH = f_bar_H_k[si]
            dfx = fxH-fxL
            dv  = vq_k[si-1]-vq_k[si]
            dvdf_k[si-1]   = sum(dv.*dfx)
            sum_dvfbarL_k += sum(dv.*fxL)
        end
        sum_dvfbarL[k] = SVector(sum_dvfbarL_k, )
    end
end

function initialize_ES_subcell_limiting!(cache,prealloc,param,discrete_data,bcdata,nstage,dim::Dim2)
    @unpack equation = param
    @unpack Uq,vq    = prealloc
    @unpack fq2q     = discrete_data.ops
    @unpack Nfp,Nq   = discrete_data.sizes
    @unpack vf,psif,dvdf,f_bar_H,f_bar_L,sum_Bpsi,sum_dvfbarL = cache

    K     = get_num_elements(param)
    N1D   = param.N+1
    N1Dm1 = N1D-1
    N1Dp1 = N1D+1
    @batch for k = 1:K
        # TODO: redundant
        for i = 1:Nq
            vq[i,k] = v_ufun(equation,Uq[i,k])
        end
        sum_Bpsi[k] = zero(sum_Bpsi[k])
        for i = 1:Nfp
            iq = fq2q[i]
            uf = Uq[iq,k]
            vf[i,k]   = v_ufun(equation,uf)
            psif[i,k] = psi_ufun(equation,uf)
            Bxy_i        = get_Bx(i,k,discrete_data,dim)
            sum_Bpsi[k] += Bxy_i.*psif[i,k]
        end

        # TODO: hardcoding views
        dvdfx_k    = reshape(view(dvdf[1],:,k),N1Dm1,N1D)
        dvdfy_k    = reshape(view(dvdf[2],:,k),N1D,N1Dm1)
        vq_k       = reshape(view(vq,:,k),N1D,N1D)
        fx_bar_H_k = reshape(view(f_bar_H[1],:,k),N1Dp1,N1D)
        fx_bar_L_k = reshape(view(f_bar_L[1],:,k),N1Dp1,N1D)
        fy_bar_H_k = reshape(view(f_bar_H[2],:,k),N1D,N1Dp1)
        fy_bar_L_k = reshape(view(f_bar_L[2],:,k),N1D,N1Dp1)
        
        sum_dvfxbarL = 0.0
        for sj = 1:N1D
            for si = 2:N1D
                fxL = fx_bar_L_k[si,sj]
                fxH = fx_bar_H_k[si,sj]
                dfx = fxH-fxL
                dv  = vq_k[si-1,sj]-vq_k[si,sj]
                dvdfx_k[si-1,sj] = sum(dv.*dfx)
                sum_dvfxbarL    += sum(dv.*fxL)
            end
        end
        sum_dvfybarL = 0.0
        for si = 1:N1D
            for sj = 2:N1D
                fyL = fy_bar_L_k[si,sj]
                fyH = fy_bar_H_k[si,sj]
                dfy = fyH-fyL
                dv  = vq_k[si,sj-1]-vq_k[si,sj]
                dvdfy_k[si,sj-1] = sum(dv.*dfy)
                sum_dvfybarL    += sum(dv.*fyL)
            end
        end
        sum_dvfbarL[k] = SVector(sum_dvfxbarL, sum_dvfybarL)
    end
end

function enforce_ES_subcell_volume!(cache,prealloc,param,discrete_data,bcdata,nstage,dim::Dim1)
    @unpack L_local_arr               = prealloc
    @unpack dvdf,sum_Bpsi,sum_dvfbarL = cache
    @unpack dvdf_order,smooth_factor  = cache
    @unpack Nq                        = discrete_data.sizes
    bound_type = get_bound_type(param)

    K   = get_num_elements(param)
    @batch for k = 1:K
        tid = Threads.threadid()

        # TODO: hardcoding views
        L_local_k    = view(L_local_arr,:,1,k,nstage)
        dvdf_k       = view(dvdf[1],:,k)
        dvdf_order_k = view(dvdf_order,:,tid)

        epsk = smooth_factor[k,nstage]

        # TODO: refactor
        # Check if current positive limiting factor already satisfies entropy bound
        sum_dvdf_k_poslim = 0.0
        for si = 2:param.N+1
            li = L_local_k[si]
            sum_dvdf_k_poslim += li*dvdf_k[si-1]
        end
        rhs = get_rhs_es(bound_type,sum_Bpsi[k][1],sum_dvfbarL[k][1],epsk)
        entropy_estimate_poslim = sum_dvdf_k_poslim - rhs
        
        tol = max(0.0, sum_dvfbarL[k][1]-sum_Bpsi[k][1])
        need_es_limiting = entropy_estimate_poslim > tol

        # if need_es_limiting
        # @show k,need_es_limiting
        # end
        # Enforce entropy stability on subcell volume faces
        if need_es_limiting
            # Sort dvdf_k
            for i = 1:Nq-1
                dvdf_order_k[i] = (dvdf_k[i],i)
            end
            sort!(dvdf_order_k,alg=QuickSort,rev=true)
            curr_idx = 1
            lhs = sum_dvdf_k_poslim
            # Greedy update
            # TODO: refactor
            while lhs > rhs+tol && curr_idx <= Nq-1
                idx = dvdf_order_k[curr_idx][2]
                si  = idx+1
                if dvdf_k[idx] < param.global_constants.ZEROTOL
                    break
                end
                lhs = lhs - L_local_k[si]*dvdf_k[si-1]
                curr_idx += 1
            end
            # Update limiting factors
            for i = 1:curr_idx-1
                idx = dvdf_order_k[i][2]
                si  = idx+1
                l_new = (i==curr_idx-1 ? max((rhs+tol-lhs)/dvdf_k[si-1], 0.0) : 0.0)
                L_local_k[si] = min(L_local_k[si], l_new)
            end
        end
    end
end

function enforce_ES_subcell_volume!(cache,prealloc,param,discrete_data,bcdata,nstage,dim::Dim2)
    @unpack L_local_arr               = prealloc
    @unpack dvdf,sum_Bpsi,sum_dvfbarL = cache
    @unpack dvdf_order,smooth_factor  = cache
    @unpack Nq                        = discrete_data.sizes
    bound_type = get_bound_type(param)

    K  = get_num_elements(param)
    N1D = param.N+1
    N1Dp1 = N1D+1
    N1Dm1 = N1D-1
    @batch for k = 1:K
        tid = Threads.threadid()

        # TODO: hardcoding views
        Lx_local_k = reshape(view(L_local_arr,:,1,k,nstage),N1Dp1,N1D)
        Ly_local_k = reshape(view(L_local_arr,:,2,k,nstage),N1D,N1Dp1)
        dvdfx_k    = reshape(view(dvdf[1],:,k),N1Dm1,N1D)
        dvdfy_k    = reshape(view(dvdf[2],:,k),N1D,N1Dm1)
        dvdfx_k_vec  = view(dvdf[1],:,k)
        dvdfy_k_vec  = view(dvdf[2],:,k)
        dvdf_order_k = view(dvdf_order,:,tid)

        epsk = smooth_factor[k,nstage]

        # TODO: refactor
        # Check if current positive limiting factor already satisfies entropy bound
        sum_dvdfx_k_poslim = 0.0
        for sj = 1:N1D
            for si = 2:N1D
                lij = Lx_local_k[si,sj]
                sum_dvdfx_k_poslim += lij*dvdfx_k[si-1,sj]
            end
        end
        rhsx = get_rhs_es(bound_type,sum_Bpsi[k][1],sum_dvfbarL[k][1],epsk)
        entropy_estimate_poslim_x = sum_dvdfx_k_poslim - rhsx
        
        sum_dvdfy_k_poslim = 0.0
        for si = 1:N1D
            for sj = 2:N1D
                lij = Ly_local_k[si,sj]
                sum_dvdfy_k_poslim += lij*dvdfy_k[si,sj-1]
            end
        end
        rhsy = get_rhs_es(bound_type,sum_Bpsi[k][2],sum_dvfbarL[k][2],epsk)
        entropy_estimate_poslim_y = sum_dvdfy_k_poslim - rhsy

        tolx = max(0.0, sum_dvfbarL[k][1]-sum_Bpsi[k][1])
        toly = max(0.0, sum_dvfbarL[k][2]-sum_Bpsi[k][2])
        need_es_limiting_x = entropy_estimate_poslim_x > tolx
        need_es_limiting_y = entropy_estimate_poslim_y > toly

        # Enforce entropy stability on subcell volume faces
        if need_es_limiting_x
            # Sort dvdfx_k
            # TODO: this results in allocation...
            # sortperm!(dvdf_order_k,dvdfx_k_vec,rev=true)
            for i = 1:Nq-N1D
                dvdf_order_k[i] = (dvdfx_k_vec[i],i)
            end
            sort!(dvdf_order_k,alg=QuickSort,rev=true)
            curr_idx = 1
            lhs = sum_dvdfx_k_poslim
            # Greedy update
            # TODO: refactor
            while lhs > rhsx+tolx && curr_idx <= Nq-N1D 
                idx = dvdf_order_k[curr_idx][2]
                si  = mod1(idx,N1Dm1)+1
                sj  = div(idx-1,N1Dm1)+1
                if dvdfx_k[si-1,sj] < param.global_constants.ZEROTOL
                    break
                end
                lhs = lhs - Lx_local_k[si,sj]*dvdfx_k[si-1,sj]
                curr_idx += 1
            end
            # Update limiting factors
            for i = 1:curr_idx-1
                idx = dvdf_order_k[i][2]
                si  = mod1(idx,N1Dm1)+1
                sj  = div(idx-1,N1Dm1)+1
                l_new = (i==curr_idx-1 ? max((rhsx+tolx-lhs)/dvdfx_k[si-1,sj], 0.0) : 0.0)
                Lx_local_k[si,sj] = min(Lx_local_k[si,sj], l_new)
            end
        end

        if need_es_limiting_y
            # Sort dvdfy_k
            # TODO: this results in allocation...
            # sortperm!(dvdf_order_k,dvdfy_k_vec,rev=true)
            for i = 1:Nq-N1D
                dvdf_order_k[i] = (dvdfy_k_vec[i],i)
            end
            sort!(dvdf_order_k,alg=QuickSort,rev=true)
            curr_idx = 1
            lhs = sum_dvdfy_k_poslim
            # Greedy update
            # TODO: refactor
            while lhs > rhsy+toly && curr_idx <= Nq-N1D
                idx = dvdf_order_k[curr_idx][2]
                si  = mod1(idx,N1D)
                sj  = div(idx-1,N1D)+2
                if dvdfy_k[si-1,sj] < param.global_constants.ZEROTOL
                    break
                end
                lhs = lhs - Ly_local_k[si,sj]*dvdfy_k[si,sj-1]
                curr_idx += 1
            end
            # Update limiting factors
            for i = 1:curr_idx-1
                idx = dvdf_order_k[i][2]
                si  = mod1(idx,N1D)
                sj  = div(idx-1,N1D)+2
                l_new = (i==curr_idx-1 ? max((rhsy+toly-lhs)/dvdfy_k[si,sj-1],0.0) : 0.0)
                Ly_local_k[si,sj] = min(Ly_local_k[si,sj], l_new)
            end
        end
    end
end

function get_rhs_es(bound_type::PositivityAndCellEntropyBound,sum_Bpsi_k,sum_dvfbarL_k,epsk)
    return sum_Bpsi_k - sum_dvfbarL_k
end

function get_rhs_es(bound_type::PositivityAndRelaxedCellEntropyBound,sum_Bpsi_k,sum_dvfbarL_k,epsk)
    beta = bound_type.beta
    return (1-beta*epsk)*(sum_Bpsi_k - sum_dvfbarL_k)
end

function enforce_ES_subcell_interface!(cache,prealloc,param,discrete_data,bcdata,nstage,basis_type::LobattoCollocation,dim)
    # Do nothing for Lobatto, since interface flux coincide
end

function enforce_ES_subcell_interface!(cache,prealloc,param,discrete_data,bcdata,nstage,basis_type::GaussCollocation,dim::Dim2)
    @unpack fstar_H,fstar_L,L_local_arr = prealloc
    @unpack vf,psif                     = cache
    @unpack mapP                        = bcdata

    Lx_local = view(L_local_arr,:,1,:,nstage)
    Ly_local = view(L_local_arr,:,2,:,nstage)

    K  = get_num_elements(param)
    N1D = param.N+1
    N1Dp1 = N1D+1
    @batch for k = 1:K
        # Enforce entropy stability on subcell interfaces
        # For each stride in x direction
        for sj = 1:N1D
            # For each subcell index on boundary
            # TODO: calculation of limiting param, redundant across subcell faces
            for si = 1:N1D:N1Dp1
                siP,sjP,kP = get_subcell_index_P_x(si,sj,k,N1Dp1,bcdata)
                idx        = si+(sj-1)*N1Dp1
                idxP       = siP+(sjP-1)*N1Dp1 
                ifq        = subcell_face_idx_to_quad_face_index_x(si,sj,k,N1D)
                fxstar_H_i = fstar_H[ifq,k][1]
                fxstar_L_i = fstar_L[ifq,k][1]
                dv    = vf[ifq,k]-vf[mapP[ifq,k]]
                dpsix = psif[ifq,k][1]-psif[mapP[ifq,k]][1]
                dvfxH = sum(dv.*fxstar_H_i)
                dvfxL = sum(dv.*fxstar_L_i)
                solve_l_es_interface!(Lx_local,idx,k,idxP,kP,dvfxH,dvfxL,dpsix)
            end
        end

        # For each stride in y direction
        for si = 1:N1D
            # For each subcell index on boundary
            # TODO: calculation of limiting param, redundant across subcell faces
            for sj = 1:N1D:N1Dp1
                siP,sjP,kP = get_subcell_index_P_y(si,sj,k,N1Dp1,bcdata)
                idx        = si+(sj-1)*N1D
                idxP       = siP+(sjP-1)*N1D 
                ifq        = subcell_face_idx_to_quad_face_index_y(si,sj,k,N1D)
                fystar_H_i = fstar_H[ifq,k][2]
                fystar_L_i = fstar_L[ifq,k][2]
                dv    = vf[ifq,k]-vf[mapP[ifq,k]]
                dpsiy = psif[ifq,k][2]-psif[mapP[ifq,k]][2]
                dvfyH = sum(dv.*fystar_H_i)
                dvfyL = sum(dv.*fystar_L_i)
                solve_l_es_interface!(Ly_local,idx,k,idxP,kP,dvfyH,dvfyL,dpsiy)
            end
        end
    end
end

function check_limited_flux_satisfies_entropy_stability(l,dvfH,dvfL,dpsi)
    return l*dvfH+(1-l)*dvfL <= dpsi
end

# Solve entropy stable limiting parameter l_es on element k, idx
#                                             and element kP, idxP
function solve_l_es_interface!(L_local,idx,k,idxP,kP,dvfH,dvfL,dpsi)
    l = min(L_local[idx,k],L_local[idxP,kP])
    f(l_i) = check_limited_flux_satisfies_entropy_stability(l_i,dvfH,dvfL,dpsi)
    les = bisection(f,0.0,l)
    L_local[idx,k] = les
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

function check_subcell_entropy_stability(cache,prealloc,param,discrete_data,dim::Dim2)
    @unpack equation                  = param
    @unpack rhsxyU,rhsxyH,rhsxyL      = prealloc
    @unpack Uq,vq,u_tilde,v_tilde     = prealloc
    @unpack fstar_H,fstar_L           = prealloc
    @unpack f_bar_H,f_bar_L,f_bar_lim = cache
    @unpack dfH_vol,dfL_vol,df_vol    = cache
    @unpack dfH_surf,dfL_surf,df_surf = cache
    @unpack fq2q,wq = discrete_data.ops
    @unpack Jq      = discrete_data.geom
    @unpack Nq,Nfp  = discrete_data.sizes

    K  = get_num_elements(param)
    N1D = param.N+1    # TODO: hardcoded
    N1Dp1 = N1D+1
    Nd = get_dim(equation)
    dim = get_dim_type(equation)
    
    # Accumulate volume and surface subcell part
    @batch for k = 1:K
        # TODO: hardcoding views
        rhsxyU_k = reshape(view(rhsxyU,:,k),N1D,N1D)
        rhsxyH_k = reshape(view(rhsxyH,:,k),N1D,N1D)
        rhsxyL_k = reshape(view(rhsxyL,:,k),N1D,N1D)
        wq_k     = reshape(view(wq,:),N1D,N1D)
        Jq_k     = reshape(view(Jq,:,k),N1D,N1D)

        fx_bar_H_k   = reshape(view(f_bar_H[1],:,k),N1Dp1,N1D)
        fx_bar_L_k   = reshape(view(f_bar_L[1],:,k),N1Dp1,N1D)
        fy_bar_H_k   = reshape(view(f_bar_H[2],:,k),N1D,N1Dp1)
        fy_bar_L_k   = reshape(view(f_bar_L[2],:,k),N1D,N1Dp1)
        fx_bar_lim_k = reshape(view(f_bar_lim[1],:,k),N1Dp1,N1D)
        fy_bar_lim_k = reshape(view(f_bar_lim[2],:,k),N1D,N1Dp1)
        
        dfxH_vol = reshape(view(dfH_vol[1],:,k),N1D,N1D)
        dfyH_vol = reshape(view(dfH_vol[2],:,k),N1D,N1D)
        dfxL_vol = reshape(view(dfL_vol[1],:,k),N1D,N1D)
        dfyL_vol = reshape(view(dfL_vol[2],:,k),N1D,N1D)
        dfx_vol  = reshape(view(df_vol[1],:,k),N1D,N1D)
        dfy_vol  = reshape(view(df_vol[2],:,k),N1D,N1D)

        dfxH_surf = reshape(view(dfH_surf[1],:,k),N1D,N1D)
        dfyH_surf = reshape(view(dfH_surf[2],:,k),N1D,N1D)
        dfxL_surf = reshape(view(dfL_surf[1],:,k),N1D,N1D)
        dfyL_surf = reshape(view(dfL_surf[2],:,k),N1D,N1D)
        dfx_surf  = reshape(view(df_surf[1],:,k),N1D,N1D)
        dfy_surf  = reshape(view(df_surf[2],:,k),N1D,N1D)

        for j = 1:N1D
            dfx_vol[1,j]   =  fx_bar_lim_k[2,j]
            dfxH_vol[1,j]  =  fx_bar_H_k[2,j]
            dfxL_vol[1,j]  =  fx_bar_L_k[2,j]
            dfx_surf[1,j]  = -fx_bar_lim_k[1,j]
            dfxH_surf[1,j] = -fx_bar_H_k[1,j]
            dfxL_surf[1,j] = -fx_bar_L_k[1,j]
            for i = 2:N1D-1
                dfx_vol[i,j]  = fx_bar_lim_k[i+1,j]-fx_bar_lim_k[i,j]
                dfxH_vol[i,j] = fx_bar_H_k[i+1,j]-fx_bar_H_k[i,j]
                dfxL_vol[i,j] = fx_bar_L_k[i+1,j]-fx_bar_L_k[i,j]
                # surf contribution is zero
            end
            dfx_vol[N1D,j]   = -fx_bar_lim_k[N1D,j]
            dfxH_vol[N1D,j]  = -fx_bar_H_k[N1D,j]
            dfxL_vol[N1D,j]  = -fx_bar_L_k[N1D,j]
            dfx_surf[N1D,j]  =  fx_bar_lim_k[N1D+1,j]
            dfxH_surf[N1D,j] =  fx_bar_H_k[N1D+1,j]
            dfxL_surf[N1D,j] =  fx_bar_L_k[N1D+1,j]
        end

        for i = 1:N1D
            dfy_vol[i,1]   =  fy_bar_lim_k[i,2]
            dfyH_vol[i,1]  =  fy_bar_H_k[i,2]
            dfyL_vol[i,1]  =  fy_bar_L_k[i,2]
            dfy_surf[i,1]  = -fy_bar_lim_k[i,1]
            dfyH_surf[i,1] = -fy_bar_H_k[i,1]
            dfyL_surf[i,1] = -fy_bar_L_k[i,1]
            for j = 2:N1D-1
                dfy_vol[i,j]  = fy_bar_lim_k[i,j+1]-fy_bar_lim_k[i,j]
                dfyH_vol[i,j] = fy_bar_H_k[i,j+1]-fy_bar_H_k[i,j]
                dfyL_vol[i,j] = fy_bar_L_k[i,j+1]-fy_bar_L_k[i,j]
            end
            dfy_vol[i,N1D]   = -fy_bar_lim_k[i,N1D]
            dfyH_vol[i,N1D]  = -fy_bar_H_k[i,N1D]
            dfyL_vol[i,N1D]  = -fy_bar_L_k[i,N1D]
            dfy_surf[i,N1D]  =  fy_bar_lim_k[i,N1D+1]
            dfyH_surf[i,N1D] =  fy_bar_H_k[i,N1D+1]
            dfyL_surf[i,N1D] =  fy_bar_L_k[i,N1D+1]
        end

        # Check dfxy_vol + dfxy_surf = M rhsxy
        for j = 1:N1D
            for i = 1:N1D
                wJq_i = Jq_k[i,j]*wq_k[i,j]
                diffx   = wJq_i*rhsxyU_k[i,j][1]-dfx_vol[i,j]-dfx_surf[i,j]
                diffx_H = wJq_i*rhsxyH_k[i,j][1]-dfxH_vol[i,j]-dfxH_surf[i,j]
                diffx_L = wJq_i*rhsxyL_k[i,j][1]-dfxL_vol[i,j]-dfxL_surf[i,j]
                diffy   = wJq_i*rhsxyU_k[i,j][2]-dfy_vol[i,j]-dfy_surf[i,j]
                diffy_H = wJq_i*rhsxyH_k[i,j][2]-dfyH_vol[i,j]-dfyH_surf[i,j]
                diffy_L = wJq_i*rhsxyL_k[i,j][2]-dfyL_vol[i,j]-dfyL_surf[i,j]
                tol = 1e-12
                if (norm(diffx) > tol || norm(diffx_H) > tol || norm(diffx_L) > tol ||
                    norm(diffy) > tol || norm(diffy_H) > tol || norm(diffy_L) > tol)
                    @show k,i,j,diffx,diffx_H,diffx_L,diffy,diffy_H,diffy_L
                end
            end
        end

        # Calculate low, high, limited solution entropy estimate
        entropy_estimate_vol_L  = zero(SVector{Nd,Float64})   # vT \Delta_vol f^L
        entropy_estimate_surf_L = zero(SVector{Nd,Float64})   # vT \Delta_surf f^L
        entropy_estimate_L      = zero(SVector{Nd,Float64})   # vT \Delta f^L
        entropy_estimate_vol_H  = zero(SVector{Nd,Float64})   # vT \Delta_vol f^H
        entropy_estimate_surf_H = zero(SVector{Nd,Float64})   # vT \Delta_surf f^H
        entropy_estimate_H      = zero(SVector{Nd,Float64})   # vT \Delta f^H
        entropy_estimate_vol    = zero(SVector{Nd,Float64})   # vT \Delta_vol f
        entropy_estimate_surf   = zero(SVector{Nd,Float64})   # vT \Delta_surf f
        entropy_estimate        = zero(SVector{Nd,Float64})   # vT \Delta f
        for i = 1:Nq
            wJq_i = wq[i]*Jq[i,k]
            entropy_estimate_vol_L  += SVector(sum(vq[i,k].*dfL_vol[1][i,k]),sum(vq[i,k].*dfL_vol[2][i,k]))
            entropy_estimate_surf_L += SVector(sum(vq[i,k].*dfL_surf[1][i,k]),sum(vq[i,k].*dfL_surf[2][i,k]))
            entropy_estimate_L      += wJq_i*SVector(sum(vq[i,k].*rhsxyL[i,k][1]),sum(vq[i,k].*rhsxyL[i,k][2]))
            entropy_estimate_vol_H  += SVector(sum(vq[i,k].*dfH_vol[1][i,k]),sum(vq[i,k].*dfH_vol[2][i,k]))
            entropy_estimate_surf_H += SVector(sum(vq[i,k].*dfH_surf[1][i,k]),sum(vq[i,k].*dfH_surf[2][i,k]))
            entropy_estimate_H      += wJq_i*SVector(sum(vq[i,k].*rhsxyH[i,k][1]),sum(vq[i,k].*rhsxyH[i,k][2]))
            entropy_estimate_vol    += SVector(sum(vq[i,k].*df_vol[1][i,k]),sum(vq[i,k].*df_vol[2][i,k]))
            entropy_estimate_surf   += SVector(sum(vq[i,k].*df_surf[1][i,k]),sum(vq[i,k].*df_surf[2][i,k]))
            entropy_estimate        += wJq_i*SVector(sum(vq[i,k].*rhsxyU[i,k][1]),sum(vq[i,k].*rhsxyU[i,k][2]))
        end

        # Calculate theory entropy estimates
        sum_Bpsi      = zero(SVector{Nd,Float64})   # 1T B psi
        sum_Bpsitilde = zero(SVector{Nd,Float64})   # 1T B psi_tilde
        vftildeBfH    = zero(SVector{Nd,Float64})   # vf_tilde^T B fH
        vfBfH         = zero(SVector{Nd,Float64})   # vf^T B fH
        vfBfL         = zero(SVector{Nd,Float64})   # vf^T B fL
        for i = 1:Nfp
            iq = fq2q[i]
            uf = Uq[iq,k]
            vf = v_ufun(equation,uf)
            Bxy_i = get_Bx(i,k,discrete_data,dim)
            sum_Bpsi      += Bxy_i .* psi_ufun(equation,uf)
            sum_Bpsitilde += Bxy_i .* psi_ufun(equation,u_tilde[Nq+i,k])
            vftildeBfH    += Bxy_i .* SVector(sum(v_tilde[Nq+i,k].*fstar_H[i,k][1]), sum(v_tilde[Nq+i,k].*fstar_H[i,k][2]))
            vfBfH         += Bxy_i .* SVector(sum(vf.*fstar_H[i,k][1]), sum(vf.*fstar_H[i,k][2]))
            vfBfL         += Bxy_i .* SVector(sum(vf.*fstar_L[i,k][1]), sum(vf.*fstar_L[i,k][2]))
        end

        diff_vol_L  = entropy_estimate_vol_L-sum_Bpsi
        diff_surf_L = entropy_estimate_surf_L+vfBfL
        diff_L      = entropy_estimate_L-sum_Bpsi+vfBfL
        diff_vol_H  = entropy_estimate_vol_H-sum_Bpsitilde+(vftildeBfH-vfBfH)
        diff_surf_H = entropy_estimate_surf_H+vfBfH
        diff_H      = entropy_estimate_H-sum_Bpsitilde+vftildeBfH
        diff_vol    = entropy_estimate_vol-sum_Bpsi
        diff_surf   = entropy_estimate_surf+vfBfL
        diff        = entropy_estimate-sum_Bpsi+vfBfL
        tol = 1e-10
        if diff_vol_L[1] > tol       || diff_vol_L[2] > tol ||
           abs(diff_surf_L[1]) > tol || abs(diff_surf_L[2]) > tol ||
           diff_L[1] > tol           || diff_L[2] > tol ||
           diff_vol[1] > tol         || diff_vol[2] > tol ||
           abs(diff_surf[1]) > tol   || abs(diff_surf[2]) > tol ||
           diff[1] > tol             || diff[2] > tol
            println("Violates entropy at element $k, $diff_L, $diff")
        end
    end
end

#########################
### Smoothness factor ###
#########################
# (69) in https://arxiv.org/pdf/2004.08503.pdf
function update_smoothness_factor!(bound_type::Union{PositivityBound,PositivityAndCellEntropyBound},cache,prealloc,param,nstage)
    # Use global minimum bound by default
    @views @. cache.smooth_factor[:,nstage] = 0.0
end

function update_smoothness_factor!(bound_type::PositivityAndMinEntropyBound,cache,prealloc,param,nstage)
    # Use global minimum bound by default
    @views @. cache.smooth_factor[:,nstage] = 1.0
end

function update_smoothness_factor!(bound_type::Union{PositivityAndRelaxedMinEntropyBound,PositivityAndRelaxedCellEntropyBound},cache,prealloc,param,nstage)
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