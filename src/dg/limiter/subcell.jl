###############################
### Subcell limiter methods ###
###############################
function initialize_entropy_bounds!(equation::CompressibleIdealGas, bound_type::Union{PositivityBound,PositivityAndCellEntropyBound,PositivityAndRelaxedCellEntropyBound,TVDBound,TVDAndCellEntropyBound,TVDAndRelaxedCellEntropyBound}, state, solver, state_param, time_param)
    @. state.cache.limiter_cache.lbound_s_modified = 0.0
end

function initialize_entropy_bounds!(equation::KPP, bound_type, state, solver, state_param, time_param)
    # Do nothing
end

# TODO: only precompute s_modified now, unnecessary to precompute bound for
#       density and internal energy?
function initialize_entropy_bounds!(equation::CompressibleIdealGas, bound_type::Union{PositivityAndMinEntropyBound,PositivityAndRelaxedMinEntropyBound,TVDAndMinEntropyBound,TVDAndRelaxedMinEntropyBound}, state, solver, state_param, time_param)
    initialize_s_modified!(state, solver, time_param)
    initialize_lower_bound!(dim_type(solver), state, solver, state_param, time_param)
end

function initialize_s_modified!(state, solver, time_param)
    (; t0) = solver.param.timestepping_param
    (; s_modified, s_modified_min) = state.cache.limiter_cache
    (; Uq) = state.preallocation
    (; K, Nq) = solver.discrete_data.sizes

    # Preallocate s_modified at nodes
    @batch for k = 1:K
        for i = 1:Nq
            s_modified[i, k] = s_modified_ufun(equation(solver), Uq[i, k])
        end
    end
    # If at the first time step, initialize minimum s_modified of the initial condition
    if time_param.t == t0 && time_param.nstage == 1
        s_modified_min[1] = minimum(s_modified)
    end
end

function initialize_lower_bound!(dim::Dim1, state, solver, state_param, time_param)
    (; K, N1D) = solver.discrete_data.sizes
    (; s_modified, s_modified_min, lbound_s_modified, smooth_factor) = state.cache.limiter_cache

    @batch for k = 1:K
        epsk = smooth_factor[k, time_param.nstage]
        for i = 1:N1D
            stencil = low_order_stencil(dim, i, k, solver, state_param)
            lbound_s_modified[i, k] = s_modified[i, k]
            for s in stencil
                lbound_s_modified[i, k] = min(lbound_s_modified[i, k], s_modified[s...])
            end
            # (70) in https://arxiv.org/pdf/2004.08503.pdf
            lbound_s_modified[i, k] = epsk * lbound_s_modified[i, k] + (1 - epsk) * s_modified_min[1]
        end
    end
end

function initialize_lower_bound!(dim::Dim2, state, solver, state_param, time_param)
    (; K, N1D) = solver.discrete_data.sizes
    (; s_modified, s_modified_min, lbound_s_modified, smooth_factor) = state.cache.limiter_cache

    s_modified = reshape(s_modified, N1D, N1D, K)
    @batch for k = 1:K
        lbound_s_modified_k = reshape(view(lbound_s_modified, :, k), N1D, N1D)
        epsk = smooth_factor[k, time_param.nstage]
        for j = 1:N1D
            for i = 1:N1D
                stencil = low_order_stencil(dim, (i, j), k, solver, state_param)
                lbound_s_modified_k[i, j] = s_modified[i, j, k]
                for s in stencil
                    lbound_s_modified_k[i, j] = min(lbound_s_modified_k[i, j], s_modified[s...])
                end
                # (70) in https://arxiv.org/pdf/2004.08503.pdf
                lbound_s_modified_k[i, j] = epsk * lbound_s_modified_k[i, j] + (1 - epsk) * s_modified_min[1]
            end
        end
    end
end

function initialize_TVD_bounds!(dim, equation::KPP, bound_type, state, solver, state_param, time_param)
    # Do nothing
end

# TODO: refactor... can initialize lbound_rho and ubound_rho for positivity here
function initialize_TVD_bounds!(dim, equation::CompressibleIdealGas, bound_type::Union{PositivityBound,PositivityAndCellEntropyBound,PositivityAndRelaxedCellEntropyBound,PositivityAndMinEntropyBound,PositivityAndRelaxedMinEntropyBound}, state, solver, state_param, time_param)
    # Do nothing
end

function initialize_TVD_bounds!(dim::Dim1, equation::CompressibleIdealGas, bound_type::Union{TVDBound,TVDAndCellEntropyBound,TVDAndRelaxedCellEntropyBound,TVDAndMinEntropyBound,TVDAndRelaxedMinEntropyBound}, state, solver, state_param, time_param)
    (; Uq, rhsL) = state.preallocation
    (; K, N1D, Nq) = solver.discrete_data.sizes
    (; rhoL, lbound_rho, ubound_rho) = state.cache.limiter_cache

    # Accumulate low order update
    @batch for k = 1:K
        for i = 1:Nq
            rhoL[i, k] = Uq[i, k][1] + time_param.dt * rhsL[i, k][1]
        end
    end

    # Compute TVD bounds
    @batch for k = 1:K
        for i = 1:N1D
            stencil = low_order_stencil(dim, i, k, solver, state_param)
            lbound_rho[i, k] = rhoL[i, k]
            ubound_rho[i, k] = rhoL[i, k]
            for s in stencil
                lbound_rho[i, k] = min(lbound_rho[i, k], rhoL[s...])
                ubound_rho[i, k] = max(ubound_rho[i, k], rhoL[s...])
            end
        end
    end
end

function initialize_TVD_bounds!(dim::Dim2, equation::CompressibleIdealGas, bound_type::Union{TVDBound,TVDAndCellEntropyBound,TVDAndRelaxedCellEntropyBound,TVDAndMinEntropyBound,TVDAndRelaxedMinEntropyBound}, state, solver, state_param, time_param)
    (; Uq, rhsL) = state.preallocation
    (; K, N1D, Nq) = solver.discrete_data.sizes
    (; rhoL, lbound_rho, ubound_rho) = state.cache.limiter_cache

    # Accumulate low order update
    @batch for k = 1:K
        for i = 1:Nq
            rhoL[i, k] = Uq[i, k][1] + time_param.dt * rhsL[i, k][1]
        end
    end

    # Compute TVD bounds
    rhoL = reshape(rhoL, N1D, N1D, K)
    @batch for k = 1:K
        lbound_rho_k = reshape(view(lbound_rho, :, k), N1D, N1D)
        ubound_rho_k = reshape(view(ubound_rho, :, k), N1D, N1D)
        for j = 1:N1D
            for i = 1:N1D
                stencil = low_order_stencil(dim, (i, j), k, solver, state_param)
                lbound_rho_k[i, j] = rhoL[i, j, k]
                ubound_rho_k[i, j] = rhoL[i, j, k]
                for s in stencil
                    lbound_rho_k[i, j] = min(lbound_rho_k[i, j], rhoL[s...])
                    ubound_rho_k[i, j] = max(ubound_rho_k[i, j], rhoL[s...])
                end
            end
        end
    end
end

# TODO: documentation... from the ipad note
function accumulate_f_bar!(dim::Dim1, state, solver)
    (; f_bar_H, f_bar_L) = state.cache.limiter_cache
    (; rhsL, rhsH, BF_H, BF_L) = state.preallocation
    (; wq) = solver.discrete_data.ops
    (; Jq) = solver.discrete_data.geom
    (; K, Nq) = solver.discrete_data.sizes

    # TODO: f_bar_H, f_bar_L could be combine into a single cache?
    @batch for k = 1:K
        f_bar_H[1][1, k] = BF_H[1, k][1]
        f_bar_L[1][1, k] = BF_L[1, k][1]
        for i = 2:Nq+1
            f_bar_H[1][i, k] = f_bar_H[1][i-1, k] + Jq[i-1, k] * wq[i-1] * rhsH[i-1, k]
            f_bar_L[1][i, k] = f_bar_L[1][i-1, k] + Jq[i-1, k] * wq[i-1] * rhsL[i-1, k]
        end
    end
end

# TODO: use views instead of index flattening
function accumulate_f_bar!(dim::Dim2, state, solver)
    (; f_bar_H, f_bar_L) = state.cache.limiter_cache
    (; rhsxyH, rhsxyL, BF_H, BF_L) = state.preallocation
    (; wq) = solver.discrete_data.ops
    (; Jq) = solver.discrete_data.geom
    (; K, N1D) = solver.discrete_data.sizes

    N1Dp1 = N1D + 1
    @batch for k = 1:K
        # TODO: hardcoding views
        fx_bar_H_k = reshape(view(f_bar_H[1], :, k), N1Dp1, N1D)
        fx_bar_L_k = reshape(view(f_bar_L[1], :, k), N1Dp1, N1D)
        fy_bar_H_k = reshape(view(f_bar_H[2], :, k), N1D, N1Dp1)
        fy_bar_L_k = reshape(view(f_bar_L[2], :, k), N1D, N1Dp1)

        rhsxyH_k = reshape(view(rhsxyH, :, k), N1D, N1D)
        rhsxyL_k = reshape(view(rhsxyL, :, k), N1D, N1D)

        wq_k = reshape(view(wq, :), N1D, N1D)
        Jq_k = reshape(view(Jq, :, k), N1D, N1D)

        # For each stride along x direction
        for sj = 1:N1D
            iface = sj
            fx_bar_H_k[1, sj] = BF_H[iface, k][1]
            fx_bar_L_k[1, sj] = BF_L[iface, k][1]
            for si = 2:N1Dp1
                fx_bar_H_k[si, sj] = fx_bar_H_k[si-1, sj] + wq_k[si-1, sj] * Jq_k[si-1, sj] * rhsxyH_k[si-1, sj][1]
                fx_bar_L_k[si, sj] = fx_bar_L_k[si-1, sj] + wq_k[si-1, sj] * Jq_k[si-1, sj] * rhsxyL_k[si-1, sj][1]
            end
        end

        # For each stride along y direction
        for si = 1:N1D
            iface = si + 2 * N1D
            fy_bar_H_k[si, 1] = BF_H[iface, k][2]
            fy_bar_L_k[si, 1] = BF_L[iface, k][2]
            for sj = 2:N1Dp1
                fy_bar_H_k[si, sj] = fy_bar_H_k[si, sj-1] + wq_k[si, sj-1] * Jq_k[si, sj-1] * rhsxyH_k[si, sj-1][2]
                fy_bar_L_k[si, sj] = fy_bar_L_k[si, sj-1] + wq_k[si, sj-1] * Jq_k[si, sj-1] * rhsxyL_k[si, sj-1][2]
            end
        end
    end
end

function subcell_bound_limiter!(dim::Dim1, equation::CompressibleIdealGas, state, solver, state_param, time_param)
    (; Uq, L_local_arr, rhsL) = state.preallocation
    (; uL_k, f_bar_H, f_bar_L, lbound_s_modified) = state.cache.limiter_cache
    (; blending_factor) = state.cache.shockcapture_cache
    (; wq) = solver.discrete_data.ops
    (; Jq) = solver.discrete_data.geom
    (; K, Nq) = solver.discrete_data.sizes
    (; dt, nstage) = time_param

    ζ = solver.param.limiting_param.ζ
    Lrhoe(uL_i) = ζ * rhoe_ufun(equation, uL_i)
    @views @. L_local_arr[:, :, :, time_param.nstage] = 1.0
    # Calculate limiting parameter
    @batch for k = 1:K
        tid = Threads.threadid()

        @views @. uL_k[:, tid] = Uq[:, k] + dt * rhsL[:, k]
        Urhoe = Inf
        # TODO: ugly...
        for i = 1:Nq
            wJq_i = (wq[i] * Jq[i, k])
            Lphi_i = lbound_s_modified[i, k]
            Lrho_i, Urho_i = rho_bound(dim_type(solver), bound_type(solver), state, solver, i, k, tid)
            bound = (Lrho_i, Lrhoe(uL_k[i, tid]), Lphi_i, Urho_i, Urhoe)
            L_local_arr[i, 1, k, nstage] = min(L_local_arr[i, 1, k, nstage], limiting_param(limiter(solver), bound_type(solver), solver, uL_k[i, tid], -2 * dt * (f_bar_H[1][i, k] - f_bar_L[1][i, k]) / wJq_i, bound))
        end
        for i = 2:Nq+1
            wJq_im1 = (wq[i-1] * Jq[i-1, k])
            Lphi_i = lbound_s_modified[i-1, k]
            Lrho_i, Urho_i = rho_bound(dim_type(solver), bound_type(solver), state, solver, i - 1, k, tid)
            bound = (Lrho_i, Lrhoe(uL_k[i-1, tid]), Lphi_i, Urho_i, Urhoe)
            L_local_arr[i, 1, k, nstage] = min(L_local_arr[i, 1, k, nstage], limiting_param(limiter(solver), bound_type(solver), solver, uL_k[i-1, tid], 2 * dt * (f_bar_H[1][i, k] - f_bar_L[1][i, k]) / wJq_im1, bound))
        end

        # Apply shock capturing
        l_shock = blending_factor[k, nstage]
        @. L_local_arr[:, 1, k, nstage] = min(L_local_arr[:, 1, k, nstage], l_shock)
    end
end

function subcell_bound_limiter!(dim::Dim2, equation::CompressibleIdealGas, state, solver, state_param, time_param)
    (; uL_k, f_bar_H, f_bar_L, lbound_s_modified) = state.cache.limiter_cache
    (; Uq, rhsL, L_local_arr) = state.preallocation
    (; blending_factor) = state.cache.shockcapture_cache
    (; wq) = solver.discrete_data.ops
    (; Jq) = solver.discrete_data.geom
    (; K, N1D) = solver.discrete_data.sizes
    (; dt, nstage) = time_param

    N1Dp1 = N1D + 1
    ζ = solver.param.limiting_param.ζ
    Lrhoe(uL_i) = ζ * rhoe_ufun(equation, uL_i)
    # TODO: why these two lines result in allocations?
    # Lx_local = reshape(view(L_local_arr,:,1,:,nstage),N1Dp1,N1D,K)
    # Ly_local = reshape(view(L_local_arr,:,2,:,nstage),N1D,N1Dp1,K)

    @views @. L_local_arr[:, :, :, nstage] = 1.0

    @batch for k = 1:K
        tid = Threads.threadid()

        @views @. uL_k[:, tid] = Uq[:, k] + dt * rhsL[:, k]
        Urhoe = Inf

        # TODO: hardcoding views
        fx_bar_H_k = reshape(view(f_bar_H[1], :, k), N1Dp1, N1D)
        fx_bar_L_k = reshape(view(f_bar_L[1], :, k), N1Dp1, N1D)
        fy_bar_H_k = reshape(view(f_bar_H[2], :, k), N1D, N1Dp1)
        fy_bar_L_k = reshape(view(f_bar_L[2], :, k), N1D, N1Dp1)

        Lx_local_k = reshape(view(L_local_arr, :, 1, k, nstage), N1Dp1, N1D)
        Ly_local_k = reshape(view(L_local_arr, :, 2, k, nstage), N1D, N1Dp1)

        lbound_s_modified_k = reshape(view(lbound_s_modified, :, k), N1D, N1D)

        u_L_k = reshape(view(uL_k, :, tid), N1D, N1D)

        wq_k = reshape(view(wq, :), N1D, N1D)
        Jq_k = reshape(view(Jq, :, k), N1D, N1D)

        # For each stride along x direction
        for sj = 1:N1D
            # For each left subcell face
            for si = 1:N1D
                # index of quad node right to subcell face
                iq = si
                jq = sj
                wJq_i = wq_k[iq, jq] * Jq_k[iq, jq]
                uL_k_i = u_L_k[iq, jq]
                Lphi_ij = lbound_s_modified_k[iq, jq]
                Lrho_i, Urho_i = rho_bound(dim_type(solver), bound_type(solver), state, solver, (iq, jq), k, tid)
                bound = (Lrho_i, Lrhoe(uL_k_i), Lphi_ij, Urho_i, Urhoe)
                Lx_local_k[si, sj] = min(Lx_local_k[si, sj], limiting_param(limiter(solver), bound_type(solver), solver, uL_k_i, -4 * dt * (fx_bar_H_k[si, sj] - fx_bar_L_k[si, sj]) / wJq_i, bound))
            end
            # For each right subcell face
            for si = 2:N1Dp1
                # index of quad node left to subcell face
                iq = si - 1
                jq = sj
                wJq_i = wq_k[iq, jq] * Jq_k[iq, jq]
                uL_k_i = u_L_k[iq, jq]
                Lphi_ij = lbound_s_modified_k[iq, jq]
                Lrho_i, Urho_i = rho_bound(dim_type(solver), bound_type(solver), state, solver, (iq, jq), k, tid)
                bound = (Lrho_i, Lrhoe(uL_k_i), Lphi_ij, Urho_i, Urhoe)
                Lx_local_k[si, sj] = min(Lx_local_k[si, sj], limiting_param(limiter(solver), bound_type(solver), solver, uL_k_i, 4 * dt * (fx_bar_H_k[si, sj] - fx_bar_L_k[si, sj]) / wJq_i, bound))
            end
        end

        # For each stride along y direction
        for si = 1:N1D
            # For each bottom subcell face
            for sj = 1:N1D
                # index of quad node top to subcell face
                iq = si
                jq = sj
                wJq_i = wq_k[iq, jq] * Jq_k[iq, jq]
                uL_k_i = u_L_k[iq, jq]
                Lphi_ij = lbound_s_modified_k[iq, jq]
                Lrho_i, Urho_i = rho_bound(dim_type(solver), bound_type(solver), state, solver, (iq, jq), k, tid)
                bound = (Lrho_i, Lrhoe(uL_k_i), Lphi_ij, Urho_i, Urhoe)
                Ly_local_k[si, sj] = min(Ly_local_k[si, sj], limiting_param(limiter(solver), bound_type(solver), solver, uL_k_i, -4 * dt * (fy_bar_H_k[si, sj] - fy_bar_L_k[si, sj]) / wJq_i, bound))
            end
            # For each top subcell face
            for sj = 2:N1Dp1
                # index of quad node beneath the subcell face
                iq = si
                jq = sj - 1
                wJq_i = wq_k[iq, jq] * Jq_k[iq, jq]
                uL_k_i = u_L_k[iq, jq]
                Lphi_ij = lbound_s_modified_k[iq, jq]
                Lrho_i, Urho_i = rho_bound(dim_type(solver), bound_type(solver), state, solver, (iq, jq), k, tid)
                bound = (Lrho_i, Lrhoe(uL_k_i), Lphi_ij, Urho_i, Urhoe)
                Ly_local_k[si, sj] = min(Ly_local_k[si, sj], limiting_param(limiter(solver), bound_type(solver), solver, uL_k_i, 4 * dt * (fy_bar_H_k[si, sj] - fy_bar_L_k[si, sj]) / wJq_i, bound))
            end
        end

        # Apply shock capturing
        l_shock = blending_factor[k, nstage]
        @. Lx_local_k = min(Lx_local_k, l_shock)
        @. Ly_local_k = min(Ly_local_k, l_shock)
    end
end

# TODO: refactor
function rho_bound(dim::Dim1, bound_type::Union{TVDBound,TVDAndCellEntropyBound,TVDAndRelaxedCellEntropyBound,TVDAndMinEntropyBound,TVDAndRelaxedMinEntropyBound}, state, solver, i, k, tid)
    (; lbound_rho, ubound_rho) = state.cache.limiter_cache
    Lrho = lbound_rho[i, k]
    Urho = ubound_rho[i, k]
    return (Lrho, Urho)
end

function rho_bound(dim::Dim2, bound_type::Union{TVDBound,TVDAndCellEntropyBound,TVDAndRelaxedCellEntropyBound,TVDAndMinEntropyBound,TVDAndRelaxedMinEntropyBound}, state, solver, i, k, tid)
    (; lbound_rho, ubound_rho) = state.cache.limiter_cache
    iq, jq = i
    N1D = solver.param.N + 1
    lbound_rho_k = reshape(view(lbound_rho, :, k), N1D, N1D)
    ubound_rho_k = reshape(view(ubound_rho, :, k), N1D, N1D)
    Lrho = lbound_rho_k[iq, jq]
    Urho = ubound_rho_k[iq, jq]
    return (Lrho, Urho)
end

function rho_bound(dim::Dim1, bound_type::Union{PositivityBound,PositivityAndCellEntropyBound,PositivityAndRelaxedCellEntropyBound,PositivityAndMinEntropyBound,PositivityAndRelaxedMinEntropyBound}, state, solver, i, k, tid)
    (; uL_k) = state.cache.limiter_cache
    (; ζ) = solver.param.limiting_param
    Lrho = ζ * uL_k[i, tid][1]
    Urho = Inf
    return (Lrho, Urho)
end

function rho_bound(dim::Dim2, bound_type::Union{PositivityBound,PositivityAndCellEntropyBound,PositivityAndRelaxedCellEntropyBound,PositivityAndMinEntropyBound,PositivityAndRelaxedMinEntropyBound}, state, solver, i, k, tid)
    (; uL_k) = state.cache.limiter_cache
    (; ζ) = solver.param.limiting_param
    iq, jq = i
    N1D = solver.param.N + 1
    u_L_k = reshape(view(uL_k, :, tid), N1D, N1D)
    Lrho = ζ * u_L_k[iq, jq][1]
    Urho = Inf
    return (Lrho, Urho)
end

function subcell_bound_limiter!(dim, equation::KPP, state, solver, state_param, time_param)
    (; L_local_arr) = state.preallocation
    (; blending_factor) = state.cache.shockcapture_cache
    (; K) = solver.discrete_data.sizes
    (; nstage) = time_param

    @views @. L_local_arr[:, :, :, nstage] = 1.0
    @batch for k = 1:K
        L_local_k = view(L_local_arr, :, :, k, nstage)

        # Apply shock capturing
        l_shock = blending_factor[k, nstage]
        @. L_local_k = min(L_local_k, l_shock)
    end
end

function symmetrize_limiting_parameters!(dim::Dim1, state, solver, state_param, time_param)
    (; L_local_arr) = state.preallocation
    (; K) = solver.discrete_data.sizes
    (; nstage) = time_param

    # Symmetrize limiting parameter TODO: hardcoded, should use mapP
    @batch for k = 1:K
        l = min(L_local_arr[1, 1, k, nstage], L_local_arr[end, 1, mod1(k - 1, K), nstage])
        L_local_arr[1, 1, k, nstage] = l
        L_local_arr[end, 1, mod1(k - 1, K), nstage] = l
    end
end

function symmetrize_limiting_parameters!(dim::Dim2, state, solver, state_param, time_param)
    (; L_local_arr) = state.preallocation
    (; K, N1D) = solver.discrete_data.sizes
    (; nstage) = time_param

    # TODO: refactor
    N1Dp1 = N1D + 1
    Lx_local = view(L_local_arr, :, 1, :, nstage)
    Ly_local = view(L_local_arr, :, 2, :, nstage)

    @batch for k = 1:K
        # Symmetrize limiting parameters
        # For each stride in x direction
        for sj = 1:N1D
            # For each subcell index on boundary
            for si = 1:N1D:N1Dp1
                siP, sjP, kP = subcell_index_P_x(si, sj, k, solver, state_param)
                idx = si + (sj - 1) * N1Dp1
                idxP = siP + (sjP - 1) * N1Dp1
                l = min(Lx_local[idx, k], Lx_local[idxP, kP])
                Lx_local[idx, k] = l
                Lx_local[idxP, kP] = l
            end
        end

        # For each stride in y direction
        for si = 1:N1D
            # For each subcell index on boundary
            for sj = 1:N1D:N1Dp1
                siP, sjP, kP = subcell_index_P_y(si, sj, k, solver, state_param)
                idx = si + (sj - 1) * N1D
                idxP = siP + (sjP - 1) * N1D
                l = min(Ly_local[idx, k], Ly_local[idxP, kP])
                Ly_local[idx, k] = l
                Ly_local[idxP, kP] = l
            end
        end
    end
end

function enforce_ES_subcell!(bound_type::Union{PositivityBound,PositivityAndMinEntropyBound,PositivityAndRelaxedMinEntropyBound,TVDBound,TVDAndMinEntropyBound,TVDAndRelaxedMinEntropyBound}, state, solver, state_param, time_param)
    # Do nothing
end

function enforce_ES_subcell!(bound_type::Union{PositivityAndCellEntropyBound,PositivityAndRelaxedCellEntropyBound,TVDAndCellEntropyBound,TVDAndRelaxedCellEntropyBound}, state, solver, state_param, time_param)
    initialize_ES_subcell_limiting!(dim_type(solver), state, solver)
    enforce_ES_subcell_volume!(dim_type(solver), state, solver, time_param)
    enforce_ES_subcell_interface!(dim_type(solver), solver.param.approximation_basis_type, state, solver, state_param, time_param)
end

function initialize_ES_subcell_limiting!(dim::Dim1, state, solver)
    (; Uq, vq) = state.preallocation
    (; fq2q) = solver.discrete_data.ops
    (; K, Nfp, Nq) = solver.discrete_data.sizes
    (; vf, psif, dvdf, f_bar_H, f_bar_L, sum_Bpsi, sum_dvfbarL) = state.cache.limiter_cache

    @batch for k = 1:K
        # TODO: redundant
        for i = 1:Nq
            vq[i, k] = v_ufun(equation(solver), Uq[i, k])
        end
        sum_Bpsi[k] = zero(sum_Bpsi[k])
        for i = 1:Nfp
            iq = fq2q[i]
            uf = Uq[iq, k]
            vf[i, k] = v_ufun(equation(solver), uf)
            psif[i, k] = psi_ufun(equation(solver), uf)
            Bxy_i = Bx(dim, i, k, solver)
            sum_Bpsi[k] += @. Bxy_i * psif[i, k]
        end

        # TODO: hardcoding views
        dvdf_k = view(dvdf[1], :, k)
        vq_k = view(vq, :, k)
        f_bar_H_k = view(f_bar_H[1], :, k)
        f_bar_L_k = view(f_bar_L[1], :, k)

        sum_dvfbarL_k = 0.0
        for si = 2:solver.param.N+1
            fxL = f_bar_L_k[si]
            fxH = f_bar_H_k[si]
            dfx = fxH - fxL
            dv = vq_k[si-1] - vq_k[si]
            dvdf_k[si-1] = sum(@. dv * dfx)
            sum_dvfbarL_k += sum(@. dv * fxL)
        end
        sum_dvfbarL[k] = SVector(sum_dvfbarL_k,)
    end
end

function initialize_ES_subcell_limiting!(dim::Dim2, state, solver)
    (; Uq, vq) = state.preallocation
    (; fq2q) = solver.discrete_data.ops
    (; K, N1D, Nfp, Nq) = solver.discrete_data.sizes
    (; vf, psif, dvdf, f_bar_H, f_bar_L, sum_Bpsi, sum_dvfbarL) = state.cache.limiter_cache

    N1Dm1 = N1D - 1
    N1Dp1 = N1D + 1
    @batch for k = 1:K
        # TODO: redundant
        for i = 1:Nq
            vq[i, k] = v_ufun(equation(solver), Uq[i, k])
        end
        sum_Bpsi[k] = zero(sum_Bpsi[k])
        for i = 1:Nfp
            iq = fq2q[i]
            uf = Uq[iq, k]
            vf[i, k] = v_ufun(equation(solver), uf)
            psif[i, k] = psi_ufun(equation(solver), uf)
            Bxy_i = Bx(dim, i, k, solver)
            sum_Bpsi[k] += @. Bxy_i * psif[i, k]
        end

        # TODO: hardcoding views
        dvdfx_k = reshape(view(dvdf[1], :, k), N1Dm1, N1D)
        dvdfy_k = reshape(view(dvdf[2], :, k), N1D, N1Dm1)
        vq_k = reshape(view(vq, :, k), N1D, N1D)
        fx_bar_H_k = reshape(view(f_bar_H[1], :, k), N1Dp1, N1D)
        fx_bar_L_k = reshape(view(f_bar_L[1], :, k), N1Dp1, N1D)
        fy_bar_H_k = reshape(view(f_bar_H[2], :, k), N1D, N1Dp1)
        fy_bar_L_k = reshape(view(f_bar_L[2], :, k), N1D, N1Dp1)

        sum_dvfxbarL = 0.0
        for sj = 1:N1D
            for si = 2:N1D
                fxL = fx_bar_L_k[si, sj]
                fxH = fx_bar_H_k[si, sj]
                dfx = fxH - fxL
                dv = vq_k[si-1, sj] - vq_k[si, sj]
                dvdfx_k[si-1, sj] = sum(@. dv * dfx)
                sum_dvfxbarL += sum(@. dv * fxL)
            end
        end
        sum_dvfybarL = 0.0
        for si = 1:N1D
            for sj = 2:N1D
                fyL = fy_bar_L_k[si, sj]
                fyH = fy_bar_H_k[si, sj]
                dfy = fyH - fyL
                dv = vq_k[si, sj-1] - vq_k[si, sj]
                dvdfy_k[si, sj-1] = sum(@. dv * dfy)
                sum_dvfybarL += sum(@. dv * fyL)
            end
        end
        sum_dvfbarL[k] = SVector(sum_dvfxbarL, sum_dvfybarL)
    end
end


function enforce_ES_subcell_volume!(dim::Dim1, state, solver, time_param)
    (; L_local_arr) = state.preallocation
    (; dvdf, sum_Bpsi, sum_dvfbarL, dvdf_order, smooth_factor) = state.cache.limiter_cache
    (; K, Nq) = solver.discrete_data.sizes
    (; nstage) = time_param

    @batch for k = 1:K
        tid = Threads.threadid()

        # TODO: hardcoding views
        L_local_k = view(L_local_arr, :, 1, k, nstage)
        dvdf_k = view(dvdf[1], :, k)
        dvdf_order_k = view(dvdf_order, :, tid)

        epsk = smooth_factor[k, nstage]

        # TODO: refactor
        # Check if current positive limiting factor already satisfies entropy bound
        sum_dvdf_k_poslim = 0.0
        for si = 2:solver.param.N+1
            li = L_local_k[si]
            sum_dvdf_k_poslim += li * dvdf_k[si-1]
        end
        rhs = rhs_es(bound_type(solver), sum_Bpsi[k][1], sum_dvfbarL[k][1], epsk)
        entropy_estimate_poslim = sum_dvdf_k_poslim - rhs

        tol = max(0.0, sum_dvfbarL[k][1] - sum_Bpsi[k][1])
        need_es_limiting = entropy_estimate_poslim > tol

        # if need_es_limiting
        # @show k,need_es_limiting
        # end
        # Enforce entropy stability on subcell volume faces
        if need_es_limiting
            # Sort dvdf_k
            for i = 1:Nq-1
                dvdf_order_k[i] = (dvdf_k[i], i)
            end
            sort!(dvdf_order_k, alg=QuickSort, rev=true)
            curr_idx = 1
            lhs = sum_dvdf_k_poslim
            # Greedy update
            # TODO: refactor
            while lhs > rhs + tol && curr_idx <= Nq - 1
                idx = dvdf_order_k[curr_idx][2]
                si = idx + 1
                if dvdf_k[idx] < solver.param.global_constants.ZEROTOL
                    break
                end
                lhs = lhs - L_local_k[si] * dvdf_k[si-1]
                curr_idx += 1
            end
            # Update limiting factors
            for i = 1:curr_idx-1
                idx = dvdf_order_k[i][2]
                si = idx + 1
                l_new = (i == curr_idx - 1 ? max((rhs + tol - lhs) / dvdf_k[si-1], 0.0) : 0.0)
                L_local_k[si] = min(L_local_k[si], l_new)
            end
        end
    end
end

function enforce_ES_subcell_volume!(dim::Dim2, state, solver, time_param)
    (; L_local_arr) = state.preallocation
    (; dvdf, sum_Bpsi, sum_dvfbarL, dvdf_order, smooth_factor) = state.cache.limiter_cache
    (; K, N1D, Nq) = solver.discrete_data.sizes
    (; nstage) = time_param

    N1Dp1 = N1D + 1
    N1Dm1 = N1D - 1
    @batch for k = 1:K
        tid = Threads.threadid()

        # TODO: hardcoding views
        Lx_local_k = reshape(view(L_local_arr, :, 1, k, nstage), N1Dp1, N1D)
        Ly_local_k = reshape(view(L_local_arr, :, 2, k, nstage), N1D, N1Dp1)
        dvdfx_k = reshape(view(dvdf[1], :, k), N1Dm1, N1D)
        dvdfy_k = reshape(view(dvdf[2], :, k), N1D, N1Dm1)
        dvdfx_k_vec = view(dvdf[1], :, k)
        dvdfy_k_vec = view(dvdf[2], :, k)
        dvdf_order_k = view(dvdf_order, :, tid)

        epsk = smooth_factor[k, nstage]

        # TODO: refactor
        # Check if current positive limiting factor already satisfies entropy bound
        sum_dvdfx_k_poslim = 0.0
        for sj = 1:N1D
            for si = 2:N1D
                lij = Lx_local_k[si, sj]
                sum_dvdfx_k_poslim += lij * dvdfx_k[si-1, sj]
            end
        end
        rhsx = rhs_es(bound_type(solver), sum_Bpsi[k][1], sum_dvfbarL[k][1], epsk)
        entropy_estimate_poslim_x = sum_dvdfx_k_poslim - rhsx

        sum_dvdfy_k_poslim = 0.0
        for si = 1:N1D
            for sj = 2:N1D
                lij = Ly_local_k[si, sj]
                sum_dvdfy_k_poslim += lij * dvdfy_k[si, sj-1]
            end
        end
        rhsy = rhs_es(bound_type(solver), sum_Bpsi[k][2], sum_dvfbarL[k][2], epsk)
        entropy_estimate_poslim_y = sum_dvdfy_k_poslim - rhsy

        tolx = max(0.0, sum_dvfbarL[k][1] - sum_Bpsi[k][1])
        toly = max(0.0, sum_dvfbarL[k][2] - sum_Bpsi[k][2])
        need_es_limiting_x = entropy_estimate_poslim_x > tolx
        need_es_limiting_y = entropy_estimate_poslim_y > toly

        # Enforce entropy stability on subcell volume faces
        if need_es_limiting_x
            # Sort dvdfx_k
            # TODO: this results in allocation...
            # sortperm!(dvdf_order_k,dvdfx_k_vec,rev=true)
            for i = 1:Nq-N1D
                dvdf_order_k[i] = (dvdfx_k_vec[i], i)
            end
            sort!(dvdf_order_k, alg=QuickSort, rev=true)
            curr_idx = 1
            lhs = sum_dvdfx_k_poslim
            # Greedy update
            # TODO: refactor
            while lhs > rhsx + tolx && curr_idx <= Nq - N1D
                idx = dvdf_order_k[curr_idx][2]
                si = mod1(idx, N1Dm1) + 1
                sj = div(idx - 1, N1Dm1) + 1
                if dvdfx_k[si-1, sj] < solver.param.global_constants.ZEROTOL
                    break
                end
                lhs = lhs - Lx_local_k[si, sj] * dvdfx_k[si-1, sj]
                curr_idx += 1
            end
            # Update limiting factors
            for i = 1:curr_idx-1
                idx = dvdf_order_k[i][2]
                si = mod1(idx, N1Dm1) + 1
                sj = div(idx - 1, N1Dm1) + 1
                l_new = (i == curr_idx - 1 ? max((rhsx + tolx - lhs) / dvdfx_k[si-1, sj], 0.0) : 0.0)
                Lx_local_k[si, sj] = min(Lx_local_k[si, sj], l_new)
            end
        end

        if need_es_limiting_y
            # Sort dvdfy_k
            # TODO: this results in allocation...
            # sortperm!(dvdf_order_k,dvdfy_k_vec,rev=true)
            for i = 1:Nq-N1D
                dvdf_order_k[i] = (dvdfy_k_vec[i], i)
            end
            sort!(dvdf_order_k, alg=QuickSort, rev=true)
            curr_idx = 1
            lhs = sum_dvdfy_k_poslim
            # Greedy update
            # TODO: refactor
            while lhs > rhsy + toly && curr_idx <= Nq - N1D
                idx = dvdf_order_k[curr_idx][2]
                si = mod1(idx, N1D)
                sj = div(idx - 1, N1D) + 2
                if dvdfy_k[si, sj-1] < solver.param.global_constants.ZEROTOL
                    break
                end
                lhs = lhs - Ly_local_k[si, sj] * dvdfy_k[si, sj-1]
                curr_idx += 1
            end
            # Update limiting factors
            for i = 1:curr_idx-1
                idx = dvdf_order_k[i][2]
                si = mod1(idx, N1D)
                sj = div(idx - 1, N1D) + 2
                l_new = (i == curr_idx - 1 ? max((rhsy + toly - lhs) / dvdfy_k[si, sj-1], 0.0) : 0.0)
                Ly_local_k[si, sj] = min(Ly_local_k[si, sj], l_new)
            end
        end
    end
end

function rhs_es(bound_type::Union{PositivityAndCellEntropyBound,TVDAndCellEntropyBound}, sum_Bpsi_k, sum_dvfbarL_k, epsk)
    return sum_Bpsi_k - sum_dvfbarL_k
end

function rhs_es(bound_type::Union{PositivityAndRelaxedCellEntropyBound,TVDAndRelaxedCellEntropyBound}, sum_Bpsi_k, sum_dvfbarL_k, epsk)
    beta = bound_type.beta
    return (1 - beta * epsk) * (sum_Bpsi_k - sum_dvfbarL_k)
end

function enforce_ES_subcell_interface!(dim, basis_type::LobattoCollocation, state, solver, state_param, time_param)
    # Do nothing for Lobatto, since interface flux coincide
end

function enforce_ES_subcell_interface!(dim::Dim2, basis_type::GaussCollocation, state, solver, state_param, time_param)
    (; fstar_H, fstar_L, L_local_arr) = state.preallocation
    (; vf, psif) = state.cache.limiter_cache
    (; mapP) = state_param.bcdata
    (; K, N1D) = solver.discrete_data.sizes
    (; nstage) = time_param

    Lx_local = view(L_local_arr, :, 1, :, nstage)
    Ly_local = view(L_local_arr, :, 2, :, nstage)

    N1Dp1 = N1D + 1
    @batch for k = 1:K
        # Enforce entropy stability on subcell interfaces
        # For each stride in x direction
        for sj = 1:N1D
            # For each subcell index on boundary
            # TODO: calculation of limiting param, redundant across subcell faces
            for si = 1:N1D:N1Dp1
                siP, sjP, kP = subcell_index_P_x(si, sj, k, solver, state_param)
                idx = si + (sj - 1) * N1Dp1
                idxP = siP + (sjP - 1) * N1Dp1
                ifq = subcell_face_idx_to_quad_face_index_x(si, sj, solver)
                fxstar_H_i = fstar_H[ifq, k][1]
                fxstar_L_i = fstar_L[ifq, k][1]
                dv = vf[ifq, k] - vf[mapP[ifq, k]]
                dpsix = psif[ifq, k][1] - psif[mapP[ifq, k]][1]
                dvfxH = sum(@. dv * fxstar_H_i)
                dvfxL = sum(@. dv * fxstar_L_i)
                solve_l_es_interface!(Lx_local, idx, k, idxP, kP, dvfxH, dvfxL, dpsix)
            end
        end

        # For each stride in y direction
        for si = 1:N1D
            # For each subcell index on boundary
            # TODO: calculation of limiting param, redundant across subcell faces
            for sj = 1:N1D:N1Dp1
                siP, sjP, kP = subcell_index_P_y(si, sj, k, solver, state_param)
                idx = si + (sj - 1) * N1D
                idxP = siP + (sjP - 1) * N1D
                ifq = subcell_face_idx_to_quad_face_index_y(si, sj, solver)
                fystar_H_i = fstar_H[ifq, k][2]
                fystar_L_i = fstar_L[ifq, k][2]
                dv = vf[ifq, k] - vf[mapP[ifq, k]]
                dpsiy = psif[ifq, k][2] - psif[mapP[ifq, k]][2]
                dvfyH = sum(@. dv * fystar_H_i)
                dvfyL = sum(@. dv * fystar_L_i)
                solve_l_es_interface!(Ly_local, idx, k, idxP, kP, dvfyH, dvfyL, dpsiy)
            end
        end
    end
end

function check_limited_flux_satisfies_entropy_stability(l, dvfH, dvfL, dpsi)
    return l * dvfH + (1 - l) * dvfL <= dpsi
end

# Solve entropy stable limiting parameter l_es on element k, idx
#                                             and element kP, idxP
function solve_l_es_interface!(L_local, idx, k, idxP, kP, dvfH, dvfL, dpsi)
    l = min(L_local[idx, k], L_local[idxP, kP])
    f(l_i) = check_limited_flux_satisfies_entropy_stability(l_i, dvfH, dvfL, dpsi)
    les = bisection(f, 0.0, l)
    L_local[idx, k] = les
end

# TODO: not necessary
function accumulate_f_bar_limited!(dim::Dim1, state, solver, time_param)
    (; f_bar_H, f_bar_L, f_bar_lim) = state.cache.limiter_cache
    (; L_local_arr) = state.preallocation
    (; K, Nq) = solver.discrete_data.sizes
    (; nstage) = time_param

    # TODO: f_bar_H, f_bar_L could be combine into a single cache? df_bar?
    @batch for k = 1:K
        for i = 1:Nq+1
            f_bar_lim[1][i, k] = L_local_arr[i, 1, k, nstage] * f_bar_H[1][i, k] + (1 - L_local_arr[i, 1, k, nstage]) * f_bar_L[1][i, k]
        end
    end
end

# TODO: not necessary
function accumulate_f_bar_limited!(dim::Dim2, state, solver, time_param)
    (; f_bar_H, f_bar_L, f_bar_lim) = state.cache.limiter_cache
    (; L_local_arr) = state.preallocation
    (; K, N1D) = solver.discrete_data.sizes
    (; nstage) = time_param

    N1Dp1 = N1D + 1
    @batch for k = 1:K
        # TODO: hardcoding views
        fx_bar_H_k = reshape(view(f_bar_H[1], :, k), N1Dp1, N1D)
        fx_bar_L_k = reshape(view(f_bar_L[1], :, k), N1Dp1, N1D)
        fy_bar_H_k = reshape(view(f_bar_H[2], :, k), N1D, N1Dp1)
        fy_bar_L_k = reshape(view(f_bar_L[2], :, k), N1D, N1Dp1)
        fx_bar_lim_k = reshape(view(f_bar_lim[1], :, k), N1Dp1, N1D)
        fy_bar_lim_k = reshape(view(f_bar_lim[2], :, k), N1D, N1Dp1)

        Lx_local_k = reshape(view(L_local_arr, :, 1, k, nstage), N1Dp1, N1D)
        Ly_local_k = reshape(view(L_local_arr, :, 2, k, nstage), N1D, N1Dp1)

        # For each stride along x direction
        for j = 1:N1D
            for i = 1:N1Dp1
                l = Lx_local_k[i, j]
                fx_bar_lim_k[i, j] = l * fx_bar_H_k[i, j] + (1 - l) * fx_bar_L_k[i, j]
            end
        end

        # For each stride along y direction
        for i = 1:N1D
            for j = 1:N1Dp1
                l = Ly_local_k[i, j]
                fy_bar_lim_k[i, j] = l * fy_bar_H_k[i, j] + (1 - l) * fy_bar_L_k[i, j]
            end
        end
    end
end

function apply_subcell_limiter!(dim::Dim1, state, solver)
    (; rhsU) = state.preallocation
    (; f_bar_lim) = state.cache.limiter_cache
    (; wq) = solver.discrete_data.ops
    (; Jq) = solver.discrete_data.geom
    (; K, Nq) = solver.discrete_data.sizes

    # Update step
    @batch for k = 1:K
        for i = 1:Nq
            wJq_i = (wq[i] * Jq[i, k])
            rhsU[i, k] = (f_bar_lim[1][i+1, k] - f_bar_lim[1][i, k]) / wJq_i
        end
    end
end

function apply_subcell_limiter!(dim::Dim2, state, solver)
    (; f_bar_lim) = state.cache.limiter_cache
    (; rhsU, rhsxyU) = state.preallocation
    (; wq) = solver.discrete_data.ops
    (; Jq) = solver.discrete_data.geom
    (; K, N1D) = solver.discrete_data.sizes

    N1Dp1 = N1D + 1
    # Update step
    @batch for k = 1:K
        # TODO: hardcoding views
        fx_bar_lim_k = reshape(view(f_bar_lim[1], :, k), N1Dp1, N1D)
        fy_bar_lim_k = reshape(view(f_bar_lim[2], :, k), N1D, N1Dp1)

        rhsxyU_k = reshape(view(rhsxyU, :, k), N1D, N1D)

        wq_k = reshape(view(wq, :), N1D, N1D)
        Jq_k = reshape(view(Jq, :, k), N1D, N1D)


        for j = 1:N1D
            for i = 1:N1D
                wJq_ij = wq_k[i, j] * Jq_k[i, j]
                rhsxyU_k[i, j] = SVector(fx_bar_lim_k[i+1, j] - fx_bar_lim_k[i, j],
                    fy_bar_lim_k[i, j+1] - fy_bar_lim_k[i, j]) / wJq_ij
            end
        end

        @. @views rhsU[:, k] = sum(rhsxyU[:, k])
    end
end

# (69) in https://arxiv.org/pdf/2004.08503.pdf
function update_smoothness_factor!(bound_type::Union{PositivityBound,PositivityAndCellEntropyBound,TVDBound,TVDAndCellEntropyBound}, state, solver, time_param)
    # Use global minimum bound by default
    @views @. state.cache.limiter_cache.smooth_factor[:, time_param.nstage] = 0.0
end

function update_smoothness_factor!(bound_type::Union{PositivityAndMinEntropyBound,TVDAndMinEntropyBound}, state, solver, time_param)
    # Use global minimum bound by default
    @views @. state.cache.limiter_cache.smooth_factor[:, time_param.nstage] = 1.0
end

function update_smoothness_factor!(bound_type::Union{PositivityAndRelaxedMinEntropyBound,PositivityAndRelaxedCellEntropyBound,TVDAndRelaxedCellEntropyBound,TVDAndRelaxedMinEntropyBound}, state, solver, time_param)
    (; N) = solver.param
    (; smooth_factor) = state.cache.limiter_cache
    (; smooth_indicator) = state.preallocation
    (; K) = solver.discrete_data.sizes
    (; nstage) = time_param

    kappa = 1.0
    s0 = log(10, N^-4)
    @batch for k = 1:K
        sk = log(10, smooth_indicator[k])
        if sk < s0 - kappa
            smooth_factor[k, nstage] = 0.0
        elseif sk > s0 + kappa
            smooth_factor[k, nstage] = 1.0
        else
            smooth_factor[k, nstage] = 0.5 - 0.5 * sin(pi * (sk - s0) / (2 * kappa))
        end
    end
end
