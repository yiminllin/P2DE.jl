function check_positivity(U, param, discrete_data)
    (; equation) = param
    (; K, Nq) = discrete_data.sizes

    rhomin = Inf
    rhoemin = Inf
    rhomax = -Inf
    rhoemax = -Inf
    for k = 1:K
        for i = 1:Nq
            is_positive, rho, p = check_positivity_node(U[i, k], param)
            if !is_positive
                @show k, i, "negative", U[i, k], rho, p
            end
            rhoe = rhoe_ufun(equation, U[i, k])
            rhomin = min(rhomin, rho)
            rhomax = max(rhomax, rho)
            rhoemin = min(rhoemin, rhoe)
            rhoemax = max(rhoemax, rhoe)
        end
    end
    # @show rhomin,rhomax,rhoemin,rhoemax
end

function check_positivity_node(Ui, param)
    (; equation) = param
    is_positive = true
    rho = Ui[1]
    p = pfun(equation, Ui)
    if (rho < 0.0 || p < 0.0)
        display("!!!!! Negativity detected !!!!!")
        @show Ui
        is_positive = false
    end
    return is_positive, rho, p
end

function check_flux_diff_entropy_stability(cache, prealloc, param, discrete_data, dim::Dim2)
    (; equation) = param
    (; v_tilde, psi_tilde, vq) = prealloc
    (; fstar_H, rhsxyH) = prealloc
    (; MinvVhTQF1, MinvVfTBF1) = cache
    (; wq) = discrete_data.ops
    (; Jq) = discrete_data.geom
    (; K, Nd, Nq, Nfp) = discrete_data.sizes

    # Check entropy stability
    dim = dim_type(equation)
    @batch for k = 1:K
        entropy_estimate_vol = zero(SVector{Nd,Float64})   # vT rhs_vol
        entropy_estimate_surf = zero(SVector{Nd,Float64})   # vT rhs_surf
        entropy_estimate = zero(SVector{Nd,Float64})   # vT rhs
        for i = 1:Nq
            m_i = wq[i]
            entropy_estimate_vol += -m_i * SVector(sum(vq[i, k] .* MinvVhTQF1[i, k][1]), sum(vq[i, k] .* MinvVhTQF1[i, k][2]))
            entropy_estimate_surf += -m_i * SVector(sum(vq[i, k] .* MinvVfTBF1[i, k][1]), sum(vq[i, k] .* MinvVfTBF1[i, k][2]))
            entropy_estimate += m_i * Jq[i, k] * SVector(sum(vq[i, k] .* rhsxyH[i, k][1]), sum(vq[i, k] .* rhsxyH[i, k][2]))
        end
        sum_Bpsi = zero(SVector{Nd,Float64})   # 1T B psi
        vTBfstar = zero(SVector{Nd,Float64})   # vT B f*
        for i = 1:Nfp
            Bxy_i = Bx(i, k, discrete_data, dim)
            sum_Bpsi += Bxy_i .* psi_tilde[Nq+i, k]
            vTBfstar += Bxy_i .* SVector(sum(v_tilde[i+Nq, k] .* fstar_H[i, k][1]), sum(v_tilde[i+Nq, k] .* fstar_H[i, k][2]))
        end
        diff_vol = entropy_estimate_vol - sum_Bpsi
        diff_surf = entropy_estimate_surf + vTBfstar
        diff = entropy_estimate - sum_Bpsi + vTBfstar
        tol = 1e-12
        if max(abs(diff_vol[1]), abs(diff_vol[2]), abs(diff_surf[1]), abs(diff_surf[2]), abs(diff[1]), abs(diff[2])) > tol
            @show k, diff_vol, diff_surf, diff
        end
    end
end

function check_low_order_entropy_stability(cache, prealloc, param, discrete_data, dim::Dim2)
    (; equation) = param
    (; rhsxyL, fstar_L, vq) = prealloc
    (; Uf) = cache
    (; Jq) = discrete_data.geom
    (; wq) = discrete_data.ops
    (; K, Nq, Nfp) = discrete_data.sizes

    @batch for k = 1:K
        entropy_estimate = zero(SVector{dim(equation),Float64})   # vT rhs
        # TODO: hardcoded
        update_face_values!(cache, prealloc, k, discrete_data, LaxFriedrichsOnNodalVal())
        for i = 1:Nq
            m_i = wq[i]
            entropy_estimate += m_i * Jq[i, k] * SVector(sum(vq[i, k] .* rhsxyL[i, k][1]), sum(vq[i, k] .* rhsxyL[i, k][2]))
        end
        sum_Bpsi = zero(SVector{dim(equation),Float64})   # 1T B psi
        vTBfstar = zero(SVector{dim(equation),Float64})   # vT B f*
        for i = 1:Nfp
            Bxy_i = Bx(i, k, discrete_data, dim_type(equation))
            sum_Bpsi += Bxy_i .* psi_ufun(equation, Uf[i, k])
            vi = v_ufun(equation, Uf[i, k])
            vTBfstar += Bxy_i .* SVector(sum(vi .* fstar_L[i, k][1]), sum(vi .* fstar_L[i, k][2]))
        end
        diff = entropy_estimate - sum_Bpsi + vTBfstar
        tol = 1e-12
        if diff[1] > tol || diff[2] > tol
            @show k, diff
        end
    end
end
