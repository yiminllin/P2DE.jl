# TODO: refactor
function compute_K_theta(equation::CompressibleNavierStokesIdealGas{Dim1},K,theta)
    # TODO: ugly syntax...
    return SVector{1}(K*theta[1], )
end

function compute_K_theta(equation::CompressibleNavierStokesIdealGas{Dim2},K,theta)
    Kxx,Kxy,Kyx,Kyy = K
    return SVector(Kxx*theta[1]+Kxy*theta[2], Kyx*theta[1]+Kyy*theta[2])
end

function compute_sigma(equation::CompressibleNavierStokesIdealGas{Dim},prealloc,param,discrete_data,bcdata,approximation_basis_type::GaussCollocation) where {Dim}
    # TODO: not supported
end

function compute_sigma(equation::CompressibleEulerIdealGas{Dim},prealloc,param,discrete_data,bcdata,approximation_basis_type) where {Dim}
    Nq = size(prealloc.Uq,1)
    K = get_num_elements(param)

    # Initialize viscous variables theta and sigma to zero
    @batch for k = 1:K
        for i = 1:Nq
            sigma[i,k] = zero(sigma[i,k])
        end
        for i = 1:Nfp
            sigmaP[i,k] = zero(sigmaP[i,k])
        end
    end
end

function compute_sigma(equation::CompressibleNavierStokesIdealGas{Dim},prealloc,param,discrete_data,bcdata,approximation_basis_type) where {Dim}
    @unpack equation = param
    @unpack Uq,v_tilde,theta,sigma,sigma_tilde,vP,sigmaP = prealloc
    @unpack Srsh_nnz,Srs0_nnz,fq2q,wq = discrete_data.ops
    @unpack mapP,mapI,mapO,mapWslip,mapWnoslip,Ival,nxy = bcdata
    @unpack Jq = discrete_data.geom

    Nq = size(Uq,1)
    Nh = size(v_tilde,1)
    Nfp = size(mapP,1)
    K = get_num_elements(param)
    dim = get_dim_type(equation)

    VU = @view v_tilde[1:Nq,:]
    vf = @view v_tilde[Nq+1:Nh,:]
    # Initialize viscous variables theta and sigma to zero
    @batch for k = 1:K
        for i = 1:Nq
            theta[i,k] = zero(theta[i,k])
            sigma[i,k] = zero(sigma[i,k])
        end
    end
    
    # Accumulate vP
    @batch for k = 1:K
        for i = 1:Nfp
            iP = mod1(mapP[i,k],Nfp)
            kP = div(mapP[i,k]-1,Nfp)+1
            vP[i,k] = vf[iP,kP]
        end
    end

    # Enforce boundary condition via vP
    # Enforce inflow BC
    @batch for i = 1:size(mapI,1)
        ii = mapI[i]
        vP[ii] = v_ufun(equation,Ival[i])
    end

    # Enforce outflow BC
    @batch for i = 1:size(mapO,1)
        io = mapO[i]
        iP = mod1(io,Nfp)
        kP = div(io-1,Nfp)+1
        iq = fq2q[iP]
        vP[io] = v_ufun(equation,Uq[iq,kP])
    end

    # Enforce wall boundary conditions
    # TODO: only consider wall BC for 2D CNS now
    @batch for j = 1:size(mapWslip,1)
        iw = mapWslip[j]
        i = mod1(iw,Nfp)
        k = div(iw-1,Nfp)+1
        iq = fq2q[i]
        vi = VU[iq,k]
        ni = (nxy[1][i,k], nxy[2][i,k])
        # v2,v3: v^+ = v - 2 v_n n
        vP[iw] = (vi[1], vi[2] - 2*ni[1]^2*vi[2], vi[3] - ni[2]^2*vi[3], vi[4])
    end

    @batch for j = 1:size(mapWnoslip,1)
        iw = mapWnoslip[j]
        i = mod1(iw,Nfp)
        k = div(iw-1,Nfp)+1
        iq = fq2q[i]
        vi = VU[iq,k]
        vP[iw] = (vi[1], -vi[2], -vi[3], vi[4])
    end

    # Accumulate volume parts of theta: (Q-Q')v
    @batch for k = 1:K
        for (i,j) in Srsh_nnz
            # TODO: hardcoded to not use hybridized operator for LGL
            if (i <= Nq && j <= Nq)
                VUi = VU[i,k]
                VUj = VU[j,k]
                Sxyh_ij = get_Sx(i,j,k,discrete_data,dim)/2
                # theta[i,k] += Sxyh_ij .* VUj
                # theta[j,k] -= Sxyh_ij .* VUi
                # theta[i,k] = map(x->x + Sxyh_ij .* VUj, theta[i,k])
                # theta[j,k] = map(x->x - Sxyh_ij .* VUi, theta[j,k])
                theta[i,k] = theta[i,k] .+ map(x->x .* VUj, Sxyh_ij)
                theta[j,k] = theta[j,k] .- map(x->x .* VUi, Sxyh_ij)
            end
        end
    end

    # Accumulate surface parts of theta: E'Bv^+
    @batch for k = 1:K
        for i = 1:Nfp
            Bxy_i = get_Bx(i,k,discrete_data,dim)
            iq = fq2q[i]
            vpi = vP[i,k]
            # theta[iq,k] += .5*SVector(Bxy_i[1]*vP[i,k], Bxy_i[2]*vP[i,k])
            theta[iq,k] += map(x->.5*x*vpi, Bxy_i)
        end
    end

    # Apply inverse mass matrix to theta
    @batch for k = 1:K
        for i = 1:Nq
            wJq_i = Jq[i,k]*wq[i]
            theta[i,k] = theta[i,k]/wJq_i
        end
    end

    # Construct sigma = K theta
    @batch for k = 1:K
        for i = 1:Nq
            Kvisc = get_Kvisc(equation,VU[i,k])
            sigma[i,k] = compute_K_theta(equation,Kvisc,theta[i,k])
        end
    end

    # Accumulate sigmaP
    @batch for k = 1:K
        for i = 1:Nfp
            iP = mod1(mapP[i,k],Nfp)
            kP = div(mapP[i,k]-1,Nfp)+1
            sigmaP[i,k] = sigma[fq2q[iP],kP]
        end
    end

    # Enforce boundary condition via sigmaP
    # Enforce inflow BC
    @batch for j = 1:size(mapI,1)
        ii = mapI[j]
        i = mod1(ii,Nfp)
        k = div(ii-1,Nfp)+1
        sigmaP[ii] = sigma[fq2q[i],k]
    end

    # Enforce outflow BC
    @batch for j = 1:size(mapO,1)
        io = mapO[j]
        i = mod1(io,Nfp)
        k = div(io-1,Nfp)+1
        sigmaP[io] = sigma[fq2q[i],k]
    end

    # Enforce wall boundary conditions
    # TODO: only consider wall BC for CNS now
    @batch for j = 1:size(mapWslip,1)
        iw = mapWslip[j]
        i = mod1(iw,Nfp)
        k = div(iw-1,Nfp)+1
        iq = fq2q[i]
        sigmai = sigma[iq,k]
        sigmaxi, sigmayi = sigmai
        ni = (nxy[1][i,k], nxy[2][i,k])
        # sigma_2, sigma_3: sigma^+ = 2 n sigma_n - sigma
        sigmax_n = sigmaxi[2]*ni[1] + sigmaxi[3]*ni[2]
        sigmay_n = sigmayi[2]*ni[1] + sigmayi[3]*ni[2]
        sigmaP[iw] = ((sigmaxi[1], 2*ni[1]*sigmax_n-sigmaxi[2], 2*ni[2]*sigmax_n-sigmaxi[3], -sigmaxi[4]),
                      (sigmayi[1], 2*ni[1]*sigmay_n-sigmayi[2], 2*ni[2]*sigmay_n-sigmayi[3], -sigmayi[4]))
    end

    @batch for j = 1:size(mapWnoslip,1)
        iw = mapWnoslip[j]
        i = mod1(iw,Nfp)
        k = div(iw-1,Nfp)+1
        iq = fq2q[i]
        sigmai = sigma[iq,k]
        sigmaxi, sigmayi = sigmai
        sigmaP[iw] = ((sigmaxi[1], sigmaxi[2], sigmaxi[3], -sigmaxi[4]),
                      (sigmayi[1], sigmayi[2], sigmayi[3], -sigmayi[4]))
    end


    # Define sigma_tilde
    # Accumulate sigmaP
    @batch for k = 1:K
        for i = 1:Nq
            sigma_tilde[i,k] = sigma[i,k]
        end
        for i = 1:Nfp
            sigma_tilde[Nq+i,k] = sigma[fq2q[i],k]
        end
    end
end