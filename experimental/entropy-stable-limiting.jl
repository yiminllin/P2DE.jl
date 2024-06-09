##################
# In DGFilter.jl #
##################

function compute_entropyproj_limiting_param_ES!(param, discrete_data, prealloc, cache, bcdata, approx_basis_type::LobattoCollocation, nstage)
    clear_entropyproj_limiting_parameter_cache!(prealloc, param.entropyproj_limiter_type, nstage)
end

# TODO: put into entropy projection to avoid an extra projection step
function compute_entropyproj_limiting_param_ES!(param, discrete_data, prealloc, cache, bcdata, approx_basis_type::GaussCollocation, nstage)
    (; θ_local_arr) = prealloc
    (; equation) = param
    (; mapP) = bcdata
    (; POSTOL) = param.global_constants
    N1D = param.N + 1
    K = num_elements(param)
    Nfp = discrete_data.sizes.Nfp
    # TODO: possible redundant calculation, only used for calculation of bounds on the fly
    calc_face_values_ES!(prealloc, cache, param, equation, discrete_data, bcdata)

    θ_local_arr_stage = view(θ_local_arr, :, :, nstage)
    Nfp = discrete_data.sizes.Nfp
    # # Symmetrize limiting factor first
    # @batch for k = 1:K
    #     for i = 1:Nfp
    #         idxP = mapP[i,k]
    #         θ_i = min(θ_local_arr_stage[i,k],θ_local_arr_stage[idxP])
    #         θ_local_arr_stage[i,k] = θ_i
    #         θ_local_arr_stage[idxP] = θ_i
    #     end
    # end

    # Enforce dv B_k ftilde >= dv B_k f
    # tmparr = zeros(1001,1001)
    @batch for k = 1:K
        tmpx = 0.0
        for i = 1:Nfp
            θpos_i = θ_local_arr_stage[i, k]
            iP = mod1(mapP[i, k], Nfp)
            kP = div(mapP[i, k] - 1, Nfp) + 1
            θPpos_i = θ_local_arr_stage[iP, kP]
            # if θpos_i < 1.0
            #     @show i,k,θpos_i,θPpos_i,calculate_dvBdfx_i_k(i,k,θpos_i,θPpos_i,prealloc,param,discrete_data)
            # end

            # if k == 45 || k == 44 || k == 46
            if i <= 2 * N1D
                dvBdfx = calculate_dvBdfx_i_k(i, k, θpos_i, θPpos_i, prealloc, param, discrete_data)
                tmpx += dvBdfx
                # @show k,dvBdfx
            end
            # end

            # a = collect(0.0:0.001:θpos_i)
            # b = collect(0.0:0.001:θPpos_i)
            # if i == 4 && k == 45
            #     @show θpos_i,θPpos_i
            #     for m = 1:length(a)
            #         for n = 1:length(b)
            #             theta = a[m]
            #             thetaP = b[n]
            #             # @show size(tmparr),m,n,theta,thetaP
            #             tmparr[m,n] = calculate_dvBdfx_i_k(i,k,theta,thetaP,prealloc,param,discrete_data)
            #         end
            #     end
            # end

            # if i <= 2*N1D
            #     fx(θ_i) = (calculate_dvBdfx_i_k(i,k,θ_i,prealloc,param,discrete_data) >= 0.0)
            #     θ = bisection(fx,0.0,θpos_i)
            #     if (θpos_i < 1.0 && θ < θpos_i)
            #         @show i,k,θ,θpos_i
            #     end
            #     θ_local_arr_stage[i,k]   = θ
            # else
            #     fy(θ_i) = (calculate_dvBdfy_i_k(i,k,θ_i,prealloc,param,discrete_data) >= 0.0)
            #     θ = bisection(fy,0.0,θpos_i)
            #     θ_local_arr_stage[i,k]   = θ
            # end
        end
        if tmpx < -1e-8
            @show k, tmpx
        end
        # if k == 45 || k == 44 || k == 46
        #     @show k,tmpx
        # end
    end


    # x = collect(0.0:0.001:1.0)
    # y = hcat([x for _ = 1:1001]...)
    # x = collect(y')
    # # plot(x[:],y[:],tmparr[:])
    # @show maximum(tmparr),minimum(tmparr)
    # scatter(x[:],y[:],tmparr[:],camera=(45,45),zcolor=tmparr[:],markersize=2,markerstrokewidth=0)
    # savefig("./outputs/a.png")
    # scatter(x[:],y[:],tmparr[:],camera=(0,90),zcolor=tmparr[:],markersize=2,markerstrokewidth=0)
    # savefig("./outputs/b.png")
    # tmparr = [t <= 0 ? -10 : 10 for t in tmparr]
    # scatter(x[:],y[:],tmparr[:],camera=(45,45),zcolor=tmparr[:],markersize=2,markerstrokewidth=0)
    # savefig("./outputs/c.png")

    # # Symmetrize limiting factor again
    # @batch for k = 1:K
    #     for i = 1:Nfp
    #         idxP = mapP[i,k]
    #         θ_i = min(θ_local_arr_stage[i,k],θ_local_arr_stage[idxP])
    #         θ_local_arr_stage[i,k] = θ_i
    #         θ_local_arr_stage[idxP] = θ_i
    #     end
    # end

    # # Enforce entropy stability
    # @batch for k = 1:K
    #     tid = Threads.threadid()
    #     Garr  = zeros(SVector{2,Float64},Nfp)
    #     Pidx  = zeros(Nfp)  # 0 if pos, 1 if negative
    #     sum_pos_x  = 0.0
    #     sum_pos_y  = 0.0
    #     for i = 1:Nfp
    #         iP = mod1(mapP[i,k],Nfp)
    #         kP = div(mapP[i,k]-1,Nfp)+1
    #         θpos_i = θ_local_arr_stage[i,k] 
    #         G_i_k  = calculate_G_i_k(cache,prealloc,i,k,θpos_i,param,discrete_data,nstage)
    #         Garr[i]  = G_i_k
    #         # TODO: hardcoded, skip one direction since G_k will be 0.0 in a direction
    #         if i <= 2*N1D
    #             if G_i_k[1] < 0.0
    #                 Pidx[i] = 1.0
    #             else
    #                 sum_pos_x  += Garr[i][1]
    #             end
    #         else
    #             if G_i_k[2] < 0.0
    #                 Pidx[i] = 1.0
    #             else
    #                 sum_pos_y  += Garr[i][2]
    #             end
    #         end
    #     end

    #     @show k
    #     @show sum(Garr[1:2*N1D]),sum(GParr[1:2*N1D])
    #     # display(Garr[1:2*N1D])
    #     # display(Pidx[1:2*N1D])
    #     # num_negative_x = sum(view(Pidx,1:2*N1D))
    #     # num_negative_y = sum(view(Pidx,2*N1D+1:Nfp))
    #     # # Lbound_x  = num_negative_x == 0 ? 0.0 : -1/num_negative_x*sum_pos_x
    #     # # LboundP_x = num_negative_x == 0 ? 0.0 : -1/num_negative_x*sum_posP_x
    #     # # Lbound_y  = num_negative_y == 0 ? 0.0 : -1/num_negative_y*sum_pos_y
    #     # # LboundP_y = num_negative_y == 0 ? 0.0 : -1/num_negative_y*sum_posP_y
    #     # Lbound_x  = num_negative_x == 0 ? 0.0 : -1/num_negative_x*sum_pos_x
    #     # LboundP_x = num_negative_x == 0 ? 0.0 : -1/num_negative_x*sum_posP_x
    #     # Lbound_y  = num_negative_y == 0 ? 0.0 : -1/num_negative_y*sum_pos_y
    #     # LboundP_y = num_negative_y == 0 ? 0.0 : -1/num_negative_y*sum_posP_y
    #     # # Calculate new factor on nodes with negative contribution
    #     # for i = 1:Nfp
    #     #     iP = mod1(mapP[i,k],Nfp)
    #     #     kP = div(mapP[i,k]-1,Nfp)+1
    #     #     θpos_i = θ_local_arr_stage[i,k] 
    #     #     if i <= 2*N1D
    #     #         if Pidx[i] == 1.0
    #     #             # negative x direction
    #     #             fx(θ_i) = (calculate_Gx_i_k(cache,prealloc,i,k,θ_i,param,discrete_data,nstage) >= Lbound_x-1e-8 && calculate_Gx_i_k(cache,prealloc,iP,kP,θ_i,param,discrete_data,nstage) >= LboundP_x-1e-8)
    #     #             θ = bisection(fx,0.0,θpos_i)
    #     #             # if (θ < θpos_i)
    #     #             #     @show i,k,θ,θpos_i,num_negative_x,sum_pos_x,sum(Garr[1:2*N1D]),sum(GParr[1:2*N1D])
    #     #             #     display(Garr[1:2*N1D])
    #     #             # end
    #     #             θ_local_arr_stage[i,k]   = θ
    #     #             θ_local_arr_stage[iP,kP] = θ
    #     #         end
    #     #     else
    #     #         if Pidx[i] == 1.0
    #     #             # negative y direction
    #     #             fy(θ_i) = (calculate_Gy_i_k(cache,prealloc,i,k,θ_i,param,discrete_data,nstage) >= Lbound_y-1e-8 && calculate_Gy_i_k(cache,prealloc,iP,kP,θ_i,param,discrete_data,nstage) >= LboundP_y-1e-8)
    #     #             θ = bisection(fy,0.0,θpos_i)
    #     #             θ_local_arr_stage[i,k]   = θ
    #     #             θ_local_arr_stage[iP,kP] = θ
    #     #         end
    #     #     end
    #     # end
    # end

    # # Enforce entropy stability
    # @batch for k = 1:K
    #     tid = Threads.threadid()
    #     Garr  = zeros(SVector{2,Float64},Nfp)
    #     GParr = zeros(SVector{2,Float64},Nfp)
    #     Pidx  = zeros(Nfp)  # 0 if pos, 1 if negative
    #     sum_pos_x  = 0.0
    #     sum_pos_y  = 0.0
    #     sum_posP_x = 0.0
    #     sum_posP_y = 0.0
    #     for i = 1:Nfp
    #         iP = mod1(mapP[i,k],Nfp)
    #         kP = div(mapP[i,k]-1,Nfp)+1
    #         θpos_i = θ_local_arr_stage[i,k] 
    #         G_i_k  = calculate_G_i_k(cache,prealloc,i,k,θpos_i,param,discrete_data,nstage)
    #         GP_i_k = calculate_G_i_k(cache,prealloc,iP,kP,θpos_i,param,discrete_data,nstage)
    #         Garr[i]  = G_i_k
    #         GParr[i] = GP_i_k
    #         # TODO: hardcoded, skip one direction since G_k will be 0.0 in a direction
    #         if i <= 2*N1D
    #             if min(G_i_k[1],GP_i_k[1]) < 0.0
    #                 Pidx[i] = 1.0
    #             else
    #                 sum_pos_x  += Garr[i][1]
    #                 sum_posP_x += GParr[i][1]
    #             end
    #         else
    #             if min(G_i_k[2],GP_i_k[2]) < 0.0
    #                 Pidx[i] = 1.0
    #             else
    #                 sum_pos_y  += Garr[i][2]
    #                 sum_posP_y += GParr[i][2]
    #             end
    #         end
    #     end

    #     @show k
    #     @show sum(Garr[1:2*N1D]),sum(GParr[1:2*N1D])
    #     display(Garr[1:2*N1D])
    #     display(Pidx[1:2*N1D])
    #     num_negative_x = sum(view(Pidx,1:2*N1D))
    #     num_negative_y = sum(view(Pidx,2*N1D+1:Nfp))
    #     # Lbound_x  = num_negative_x == 0 ? 0.0 : -1/num_negative_x*sum_pos_x
    #     # LboundP_x = num_negative_x == 0 ? 0.0 : -1/num_negative_x*sum_posP_x
    #     # Lbound_y  = num_negative_y == 0 ? 0.0 : -1/num_negative_y*sum_pos_y
    #     # LboundP_y = num_negative_y == 0 ? 0.0 : -1/num_negative_y*sum_posP_y
    #     Lbound_x  = num_negative_x == 0 ? 0.0 : -1/num_negative_x*sum_pos_x
    #     LboundP_x = num_negative_x == 0 ? 0.0 : -1/num_negative_x*sum_posP_x
    #     Lbound_y  = num_negative_y == 0 ? 0.0 : -1/num_negative_y*sum_pos_y
    #     LboundP_y = num_negative_y == 0 ? 0.0 : -1/num_negative_y*sum_posP_y
    #     # Calculate new factor on nodes with negative contribution
    #     for i = 1:Nfp
    #         iP = mod1(mapP[i,k],Nfp)
    #         kP = div(mapP[i,k]-1,Nfp)+1
    #         θpos_i = θ_local_arr_stage[i,k] 
    #         if i <= 2*N1D
    #             if Pidx[i] == 1.0
    #                 # negative x direction
    #                 fx(θ_i) = (calculate_Gx_i_k(cache,prealloc,i,k,θ_i,param,discrete_data,nstage) >= Lbound_x-1e-8 && calculate_Gx_i_k(cache,prealloc,iP,kP,θ_i,param,discrete_data,nstage) >= LboundP_x-1e-8)
    #                 θ = bisection(fx,0.0,θpos_i)
    #                 # if (θ < θpos_i)
    #                 #     @show i,k,θ,θpos_i,num_negative_x,sum_pos_x,sum(Garr[1:2*N1D]),sum(GParr[1:2*N1D])
    #                 #     display(Garr[1:2*N1D])
    #                 # end
    #                 θ_local_arr_stage[i,k]   = θ
    #                 θ_local_arr_stage[iP,kP] = θ
    #             end
    #         else
    #             if Pidx[i] == 1.0
    #                 # negative y direction
    #                 fy(θ_i) = (calculate_Gy_i_k(cache,prealloc,i,k,θ_i,param,discrete_data,nstage) >= Lbound_y-1e-8 && calculate_Gy_i_k(cache,prealloc,iP,kP,θ_i,param,discrete_data,nstage) >= LboundP_y-1e-8)
    #                 θ = bisection(fy,0.0,θpos_i)
    #                 θ_local_arr_stage[i,k]   = θ
    #                 θ_local_arr_stage[iP,kP] = θ
    #             end
    #         end
    #     end
    # end
end

function calculate_dvB_i_k(i, k, θ_i, prealloc, param, discrete_data)
    (; VUfH, VUfL) = prealloc
    (; equation) = param
    dim = dim_type(equation)
    Bxy_i, n_i_norm = Bx_with_n(i, k, discrete_data, dim)
    vf_tilde_i_k = θ_i * VUfH[i, k] + (1.0 - θ_i) * VUfL[i, k]
    vf_i_k = VUfL[i, k]
    dv = vf_tilde_i_k - vf_i_k
    return SVector(Bxy_i[1] * dv, Bxy_i[2] * dv)
end

function calculate_dvBdfx_i_k(i, k, θ_i, θP_i, prealloc, param, discrete_data)
    return calculate_dvBdf_i_k(i, k, θ_i, θP_i, prealloc, param, discrete_data)[1]
end

function calculate_dvBdfy_i_k(i, k, θ_i, θP_i, prealloc, param, discrete_data)
    return calculate_dvBdf_i_k(i, k, θ_i, θP_i, prealloc, param, discrete_data)[2]
end

function calculate_dvBdf_i_k(i, k, θ_i, θP_i, prealloc, param, discrete_data)
    (; VUfH, VUfL, VUPH, VUPL, fstar_L) = prealloc
    (; equation) = param
    dim = dim_type(equation)
    vf_tilde_i_k = θ_i * VUfH[i, k] + (1.0 - θ_i) * VUfL[i, k]
    vP_tilde_i_k = θP_i * VUPH[i, k] + (1.0 - θP_i) * VUPL[i, k]
    uf_tilde_i_k = u_vfun(equation, vf_tilde_i_k)
    uP_tilde_i_k = u_vfun(equation, vP_tilde_i_k)
    flux_f = fstar_L[i, k]
    flux_tilde_f = calculate_numerical_flux(uf_tilde_i_k, uP_tilde_i_k, i, k, prealloc, param, discrete_data, dim)
    df = flux_tilde_f - flux_f
    dvB = calculate_dvB_i_k(i, k, θ_i, prealloc, param, discrete_data)
    return SVector(sum(dvB[1] .* df[1]), sum(dvB[2] .* df[2]))
end

function calculate_Gx_i_k(i, k, θ_i, prealloc, param, discrete_data)
    return calculate_G_i_k(i, k, θ_i, prealloc, param, discrete_data)[1]
end

function calculate_Gy_i_k(i, k, θ_i, prealloc, param, discrete_data)
    return calculate_G_i_k(i, k, θ_i, prealloc, param, discrete_data)[2]
end

function calculate_G_i_k(i, k, θ_i, prealloc, param, discrete_data)
    (; VUfH, VUfL, psif, fstar_L) = prealloc
    (; equation) = param
    dim = dim_type(equation)
    Bxy_i = Bx(i, k, discrete_data, dim)

    vf_tilde_i_k = θ_i * VUfH[i, k] + (1.0 - θ_i) * VUfL[i, k]
    uf_tilde_i_k = u_vfun(equation, vf_tilde_i_k)
    psif_tilde_i_k = psi_ufun(uf_tilde_i_k)
    flux_f = fstar_L[i, k]
    dpsi = psif[i, k] - psif_tilde_i_k
    dvB = calculate_dvB_i_k(i, k, θ_i, prealloc, param, discrete_data)

    return SVector(Bxy_i[1] * dpsi[1] - sum(dvB[1] .* flux_f[1]), Bxy_i[2] * dpsi[2] - sum(dvB[2] .* flux_f[2]))
end

# TODO: refactor
# TODO: redundant with limiter
function calc_face_values_ES!(prealloc, cache, param, equation, discrete_data, bcdata)
    (; equation) = param
    (; Uq, vq) = prealloc
    (; UfL, UPL, VUfH, VUfL, VUPL, VUPH, psif, psiP, fstar_L) = prealloc
    (; Vf, Vf_low) = discrete_data.ops
    (; mapP) = bcdata

    K = num_elements(param)
    Nfp = discrete_data.sizes.Nfp
    dim = dim_type(equation)
    # Initialize entropy variable and entropy potential
    @batch for k = 1:K
        mul!(view(UfL, :, k), Vf_low, view(Uq, :, k))
        mul!(view(VUfL, :, k), Vf_low, view(vq, :, k))
        mul!(view(VUfH, :, k), Vf, view(vq, :, k))
        for i = 1:Nfp
            psif[i, k] = psi_ufun(equation, UfL[i, k])
        end
    end

    @batch for k = 1:K
        for i = 1:Nfp
            idxP = mapP[i, k]
            psiP[i, k] = psif[idxP]
            UPL[i, k] = UfL[idxP]
            VUPL[i, k] = VUfL[idxP]
            VUPH[i, k] = VUfH[idxP]
        end
    end

    # TODO: fstar_L duplicates with Low order positivity RHS
    @batch for k = 1:K
        for i = 1:Nfp
            uf_i_k = UfL[i, k]
            uP_i_k = UPL[i, k]
            fstar_L[i, k] = calculate_numerical_flux(uf_i_k, uP_i_k, i, k, prealloc, param, discrete_data, dim)
        end
    end
end

# TODO: refactor
function calculate_numerical_flux(uf, uP, i, k, prealloc, param, discrete_data, dim)
    (; equation, N) = param
    (; UfL, UPL) = prealloc
    N1D = N + 1
    Bxy_i, n_i_norm = Bx_with_n(i, k, discrete_data, dim)
    n_i = Bxy_i ./ n_i_norm

    flux = 0.5 .* (fluxes(equation, uf) .+ fluxes(equation, uP))
    LFc = 0.5 * n_i_norm * max(wavespeed_estimate(equation, uf, n_i), wavespeed_estimate(equation, uP, n_i)) * (uP - uf)
    if i <= 2 * N1D
        flux = SVector(flux[1] - LFc / Bxy_i[1], flux[2])
    else
        flux = SVector(flux[1], flux[2] - LFc / Bxy_i[2])
    end
    return flux
end

###################
# In DGLimiter.jl #
###################
function apply_rhs_limiter!(prealloc, param, discrete_data, bcdata, caches, t, dt, nstage, rhs_limiter_type::SubcellLimiter, timer)
    (; limiter_cache, shockcapture_cache) = caches
    dim = dim_type(param.equation)
    bound_type = bound_type(param)
    @timeit_debug timer "Initialize smoothness indicator" begin
        initialize_smoothness_indicator!(shockcapture_type(param), bound_type, prealloc, param, discrete_data, nstage)
    end
    @timeit_debug timer "calculate blending factor" begin
        update_blending_factor!(shockcapture_type(param), shockcapture_cache, prealloc, param, discrete_data, nstage)
    end
    @timeit_debug timer "calculate smoothness factor" begin
        update_smoothness_factor!(bound_type, limiter_cache, prealloc, param, nstage)
    end
    @timeit_debug timer "Precompute bounds" begin
        initialize_bounds!(limiter_cache, prealloc, bound_type, param, discrete_data, bcdata, t, nstage, dim)
    end
    @timeit_debug timer "Precompute entropy variables" begin
        initialize_entropy_vars!(limiter_cache, prealloc, bound_type, param, discrete_data, bcdata, t, nstage, dim)
    end
    @timeit_debug timer "Accumulate low and high order subcell fluxes" begin
        accumulate_f_bar!(limiter_cache, prealloc, param, discrete_data, dim)
    end
    @timeit_debug timer "Find subcell limiting parameters" begin
        subcell_bound_limiter!(limiter_cache, shockcapture_cache, prealloc, param, discrete_data, bcdata, dt, nstage, dim)
    end
    @timeit_debug timer "Symmetrize subcell limiting parameters" begin
        symmetrize_limiting_parameters!(prealloc, param, bcdata, nstage, dim)
    end
    @timeit_debug timer "Precompute variables for entropy stability enforcement" begin
        initialize_entropy_stable_limiting!(limiter_cache, prealloc, param, discrete_data, bcdata, nstage, dim)
    end
    @timeit_debug timer "Enforce ES on numerical flux" begin
        enforce_ES_subcell_flux!(limiter_cache, prealloc, param, discrete_data, bcdata, nstage, dim)
    end
    @timeit_debug timer "Accumulate limited subcell fluxes" begin
        accumulate_f_bar_limited!(limiter_cache, prealloc, param, nstage, dim)
    end
    @timeit_debug timer "Apply subcell limiter, accumulate limited rhs" begin
        apply_subcell_limiter!(prealloc, limiter_cache, param, discrete_data, dim)
    end
end




########################
# In DGLimiterUtils.jl #
########################
# TODO: refactor
function subcell_face_idx_to_quad_face_index_x(si, sj, k, N1D)
    iface = 0
    if (si == 1)
        iface = sj
    elseif (si == N1D + 1)
        iface = sj + N1D
    end

    return iface
end

# TODO: refactor
function subcell_face_idx_to_quad_face_index_y(si, sj, k, N1D)
    iface = 0
    if (sj == 1)
        iface = si + 2 * N1D
    elseif (sj == N1D + 1)
        iface = si + 3 * N1D
    end

    return iface
end





########################
# In SubcellLimiter.jl #
########################

function initialize_entropy_vars!(cache, prealloc, bound_type, param, discrete_data, bcdata, t, nstage, dim)
    (; equation) = param
    (; t0) = param.timestepping_param
    (; Uq, vq, psiq) = prealloc
    (; UfL, VUfL, VUPL, psif, psiP) = prealloc
    (; Vf_low) = discrete_data.ops
    (; mapP) = bcdata
    (; f_bar_H, f_bar_L) = cache
    (; sum_Bpsi) = cache

    N1D = param.N + 1
    N1Dp1 = N1D + 1
    N1Dm1 = N1D - 1
    K = num_elements(param)
    Nq = size(Uq, 1)
    # Preallocate vq,phiq at nodes
    @batch for k = 1:K
        for i = 1:Nq
            vq[i, k] = v_ufun(equation, Uq[i, k])    # TODO: maybe redundant
            psiq[i, k] = psi_ufun(equation, Uq[i, k])
        end
    end

    # Initialize entropy variable and entropy potential
    Nfp = discrete_data.sizes.Nfp
    @batch for k = 1:K
        mul!(view(UfL, :, k), Vf_low, view(Uq, :, k))
        mul!(view(VUfL, :, k), Vf_low, view(vq, :, k))
        sum_Bpsi[k] = zero(sum_Bpsi[k])
        for i = 1:Nfp
            Bxy_i = Bx(i, k, discrete_data, dim)
            psif[i, k] = psi_ufun(equation, UfL[i, k])
            sum_Bpsi[k] += Bxy_i .* psif[i, k]
        end
    end

    @batch for k = 1:K
        for i = 1:Nfp
            idxP = mapP[i, k]
            psiP[i, k] = psif[idxP]
            VUPL[i, k] = VUfL[idxP]
        end
    end
end



function initialize_entropy_stable_limiting!(cache, prealloc, param, discrete_data, bcdata, nstage, dim)
    (; equation) = param
    (; vq) = prealloc
    (; f_bar_H, f_bar_L, sum_dvfbarL, sum_dvfbarH, dvdf) = cache

    # TODO: refactor
    K = num_elements(param)
    N1D = param.N + 1
    N1Dp1 = N1D + 1
    N1Dm1 = N1D - 1

    # TODO: refactor
    # Calculate sum_dvfbarL,dvdf for enforcing entropy stability
    @batch for k = 1:K
        # TODO: hardcoding views
        fx_bar_H_k = reshape(view(f_bar_H[1], :, k), N1Dp1, N1D)
        fx_bar_L_k = reshape(view(f_bar_L[1], :, k), N1Dp1, N1D)
        fy_bar_H_k = reshape(view(f_bar_H[2], :, k), N1D, N1Dp1)
        fy_bar_L_k = reshape(view(f_bar_L[2], :, k), N1D, N1Dp1)

        dvdfx_k = reshape(view(dvdf[1], :, k), N1Dm1, N1D)
        dvdfy_k = reshape(view(dvdf[2], :, k), N1D, N1Dm1)

        vq_k = reshape(view(vq, :, k), N1D, N1D)

        sum_dvfxbarL = 0.0
        sum_dvfxbarH = 0.0
        for sj = 1:N1D
            # For each subcell index in interior
            for si = 2:N1D
                dv = vq_k[si-1, sj] - vq_k[si, sj]
                fxL = fx_bar_L_k[si, sj]
                fxH = fx_bar_H_k[si, sj]
                dfx = fxH - fxL
                dvdfx_k[si-1, sj] = sum(dv .* dfx)
                sum_dvfxbarL += sum(dv .* fxL)
                sum_dvfxbarH += sum(dv .* fxH)
            end
        end

        sum_dvfybarL = 0.0
        sum_dvfybarH = 0.0
        for si = 1:N1D
            # For each subcell index in interior
            for sj = 2:N1D
                dv = vq_k[si, sj-1] - vq_k[si, sj]
                fyL = fy_bar_L_k[si, sj]
                fyH = fy_bar_H_k[si, sj]
                dfy = fyH - fyL
                dvdfy_k[si, sj-1] = sum(dv .* dfy)
                sum_dvfybarL += sum(dv .* fyL)
                sum_dvfybarH += sum(dv .* fyH)
            end
        end

        sum_dvfbarL[k] = SVector(sum_dvfxbarL, sum_dvfybarL)
        sum_dvfbarH[k] = SVector(sum_dvfxbarH, sum_dvfybarH)
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
    L_local[idxP, kP] = les
end

# TODO: refactor x,y direction
function enforce_ES_subcell_flux!(cache, prealloc, param, discrete_data, bcdata, nstage, dim::Dim2)
    (; equation) = param
    (; Uq, vq, psiq) = prealloc
    (; UfL, VUfL, VUPL, psif, psiP) = prealloc
    (; L_local_arr, fstar_L, fstar_H) = prealloc
    (; f_bar_H, f_bar_L) = cache
    (; sum_Bpsi, sum_dvfbarL, sum_dvfbarH, dvdf) = cache
    (; Vf_low) = discrete_data.ops
    (; mapP) = bcdata
    (; LPmodels) = prealloc

    # TODO: refactor
    K = num_elements(param)
    Nc = discrete_data.sizes.Nc
    Nfp = discrete_data.sizes.Nfp
    N1D = param.N + 1
    N1Dp1 = N1D + 1
    N1Dm1 = N1D - 1
    Lx_local = view(L_local_arr, :, 1, :, nstage)
    Ly_local = view(L_local_arr, :, 2, :, nstage)

    total_x = zeros(Threads.nthreads())
    total_y = zeros(Threads.nthreads())
    @batch for k = 1:K
        tid = Threads.threadid()
        model_x = LPmodels[1][tid]
        model_y = LPmodels[2][tid]

        # TODO: hardcoding views
        fx_bar_H_k = reshape(view(f_bar_H[1], :, k), N1Dp1, N1D)
        fx_bar_L_k = reshape(view(f_bar_L[1], :, k), N1Dp1, N1D)
        fy_bar_H_k = reshape(view(f_bar_H[2], :, k), N1D, N1Dp1)
        fy_bar_L_k = reshape(view(f_bar_L[2], :, k), N1D, N1Dp1)

        dvdfx_k = reshape(view(dvdf[1], :, k), N1Dm1, N1D)
        dvdfy_k = reshape(view(dvdf[2], :, k), N1D, N1Dm1)

        Lx_local_k = reshape(view(L_local_arr, :, 1, k, nstage), N1Dp1, N1D)
        Ly_local_k = reshape(view(L_local_arr, :, 2, k, nstage), N1D, N1Dp1)

        vq_k = reshape(view(vq, :, k), N1D, N1D)
        psiq_k = reshape(view(psiq, :, k), N1D, N1D)

        # TODO: refactor
        # Check if current positive limiting factor already satisfies entropy bound
        sum_dvdfx_k_poslim = 0.0
        for sj = 1:N1D
            for si = 2:N1D
                lij = Lx_local_k[si, sj]
                sum_dvdfx_k_poslim += lij * dvdfx_k[si-1, sj]
            end
        end
        entropy_estimate_poslim_x = sum_dvdfx_k_poslim + sum_dvfbarL[k][1] - sum_Bpsi[k][1]
        entropy_estimate_L_x = sum_dvfbarL[k][1] - sum_Bpsi[k][1]
        entropy_estimate_H_x = sum_dvfbarH[k][1] - sum_Bpsi[k][1]

        sum_dvdfy_k_poslim = 0.0
        for si = 1:N1D
            for sj = 2:N1D
                lij = Ly_local_k[si, sj]
                sum_dvdfy_k_poslim += lij * dvdfy_k[si, sj-1]
            end
        end
        entropy_estimate_poslim_y = sum_dvdfy_k_poslim + sum_dvfbarL[k][2] - sum_Bpsi[k][2]

        # need_es_limiting_x = entropy_estimate_poslim_x > 0.0
        # need_es_limiting_y = entropy_estimate_poslim_y > 0.0
        need_es_limiting_x = entropy_estimate_poslim_x > 0.0
        need_es_limiting_y = entropy_estimate_poslim_y > 0.0
        # @show k,need_es_limiting_x,need_es_limiting_y

        # if (need_es_limiting_x)
        #     @show k,entropy_estimate_poslim_x,sum(Lx_local_k)
        #     total_x[tid] += 1
        # end

        # if (need_es_limiting_y)
        #     @show k,entropy_estimate_poslim_y,sum(Ly_local_k)
        #     total_y[tid] += 1
        # end

        # Enforce entropy stability on subcell volume faces
        if need_es_limiting_x
            # Modify the entropy stability bound and
            # Modify the limiting factor bound (only need to modify the upper bounds):
            for sj = 1:N1D
                for si = 2:N1D
                    set_normalized_coefficient(model_x[:con_es], model_x[:lx][si-1, sj], dvdfx_k[si-1, sj])
                    set_normalized_rhs(model_x[:con_ubound][si-1, sj], Lx_local_k[si, sj])
                end
            end
            set_normalized_rhs(model_x[:con_es], sum_Bpsi[k][1] - sum_dvfbarL[k][1])

            # @show k
            # if k == 59
            #     for sj = 1:N1D
            #         for si = 2:N1D
            #             @show si,sj,-dvdfx_k[si-1,sj]
            #         end
            #     end
            #     @show ss,-entropy_estimate_H_x
            # print(model_x)
            # end

            # Optimize
            optimize!(model_x)

            # Update interior subcell limiting factors
            for sj = 1:N1D
                for si = 2:N1D
                    Lx_local_k[si, sj] = value(model_x[:lx][si-1, sj])
                end
            end
        end

        if need_es_limiting_y
            for si = 1:N1D
                for sj = 2:N1D
                    set_normalized_coefficient(model_y[:con_es], model_y[:ly][si, sj-1], dvdfy_k[si, sj-1])
                    set_normalized_rhs(model_y[:con_ubound][si, sj-1], Ly_local_k[si, sj])
                end
            end
            set_normalized_rhs(model_y[:con_es], sum_Bpsi[k][2] - sum_dvfbarL[k][2])

            # Optimize
            optimize!(model_y)

            # Update interior subcell limiting factors
            for si = 1:N1D
                for sj = 2:N1D
                    Ly_local_k[si, sj] = value(model_y[:ly][si, sj-1])
                end
            end
        end

        # sum_dvdfx_k_poslim = 0.0
        # for sj = 1:N1D
        #     for si = 2:N1D
        #         lij = Lx_local_k[si,sj]
        #         sum_dvdfx_k_poslim += lij*dvdfx_k[si-1,sj]
        #     end
        # end
        # entropy_estimate_poslim_x = sum_dvdfx_k_poslim + sum_dvfbarL[k][1] - sum_Bpsi[k][1]
        # if need_es_limiting_x
        #     @show "after limiting the estimate is",k, entropy_estimate_poslim_x,sum(Lx_local_k)
        #         # display(Lx_local_k)
        # end

        # sum_dvdfy_k_poslim = 0.0
        # for si = 1:N1D
        #     for sj = 2:N1D
        #         lij = Ly_local_k[si,sj]
        #         sum_dvdfy_k_poslim += lij*dvdfy_k[si,sj-1]
        #     end
        # end
        # entropy_estimate_poslim_y = sum_dvdfy_k_poslim + sum_dvfbarL[k][2] - sum_Bpsi[k][2]
        # if need_es_limiting_y
        # @show "after limiting the estimate is",k, entropy_estimate_poslim_y,sum(Ly_local_k)
        # end

        # Enforce entropy stability on subcell interfaces
        # For each stride in x direction
        for sj = 1:N1D
            # For each subcell index on boundary
            # TODO: calculation of limiting param, redundant across subcell faces
            for si = 1:N1D:N1Dp1
                siP, sjP, kP = subcell_index_P_x(si, sj, k, N1Dp1, bcdata)
                idx = si + (sj - 1) * N1Dp1
                idxP = siP + (sjP - 1) * N1Dp1
                ifq = subcell_face_idx_to_quad_face_index_x(si, sj, k, N1D)
                fxstar_H_i = fstar_H[ifq, k][1]
                fxstar_L_i = fstar_L[ifq, k][1]
                dv = VUfL[ifq, k] - VUPL[mapP[ifq, k]]
                dpsix = psif[ifq, k][1] - psiP[mapP[ifq, k]][1]
                dvfxH = sum(dv .* fxstar_H_i)
                dvfxL = sum(dv .* fxstar_L_i)
                solve_l_es_interface!(Lx_local, idx, k, idxP, kP, dvfxH, dvfxL, dpsix)
            end
        end

        # For each stride in y direction
        for si = 1:N1D
            # For each subcell index on boundary
            # TODO: calculation of limiting param, redundant across subcell faces
            for sj = 1:N1D:N1Dp1
                siP, sjP, kP = subcell_index_P_y(si, sj, k, N1Dp1, bcdata)
                idx = si + (sj - 1) * N1D
                idxP = siP + (sjP - 1) * N1D
                ifq = subcell_face_idx_to_quad_face_index_y(si, sj, k, N1D)
                fystar_H_i = fstar_H[ifq, k][2]
                fystar_L_i = fstar_L[ifq, k][2]
                dv = VUfL[ifq, k] - VUPL[mapP[ifq, k]]
                dpsiy = psif[ifq, k][2] - psiP[mapP[ifq, k]][2]
                dvfyH = sum(dv .* fystar_H_i)
                dvfyL = sum(dv .* fystar_L_i)
                solve_l_es_interface!(Ly_local, idx, k, idxP, kP, dvfyH, dvfyL, dpsiy)
            end
        end
    end
    # @show sum(total_x),sum(total_y)
end

function check_subcell_entropy_stability(cache, prealloc, param, discrete_data, dim::Dim2)
    (; equation) = param
    (; rhsxyU, rhsxyH, rhsxyL) = prealloc
    (; Uq, vq, u_tilde, v_tilde) = prealloc
    (; fstar_H, fstar_L) = prealloc
    (; f_bar_H, f_bar_L, f_bar_lim) = cache
    (; dfH_vol, dfL_vol, df_vol) = cache
    (; dfH_surf, dfL_surf, df_surf) = cache
    (; fq2q, wq) = discrete_data.ops
    (; Jq) = discrete_data.geom
    (; K, N1D, Nd, Nq, Nfp) = discrete_data.sizes

    N1Dp1 = N1D + 1
    # Accumulate volume and surface subcell part
    @batch for k = 1:K
        # TODO: hardcoding views
        rhsxyU_k = reshape(view(rhsxyU, :, k), N1D, N1D)
        rhsxyH_k = reshape(view(rhsxyH, :, k), N1D, N1D)
        rhsxyL_k = reshape(view(rhsxyL, :, k), N1D, N1D)
        wq_k = reshape(view(wq, :), N1D, N1D)
        Jq_k = reshape(view(Jq, :, k), N1D, N1D)

        fx_bar_H_k = reshape(view(f_bar_H[1], :, k), N1Dp1, N1D)
        fx_bar_L_k = reshape(view(f_bar_L[1], :, k), N1Dp1, N1D)
        fy_bar_H_k = reshape(view(f_bar_H[2], :, k), N1D, N1Dp1)
        fy_bar_L_k = reshape(view(f_bar_L[2], :, k), N1D, N1Dp1)
        fx_bar_lim_k = reshape(view(f_bar_lim[1], :, k), N1Dp1, N1D)
        fy_bar_lim_k = reshape(view(f_bar_lim[2], :, k), N1D, N1Dp1)

        dfxH_vol = reshape(view(dfH_vol[1], :, k), N1D, N1D)
        dfyH_vol = reshape(view(dfH_vol[2], :, k), N1D, N1D)
        dfxL_vol = reshape(view(dfL_vol[1], :, k), N1D, N1D)
        dfyL_vol = reshape(view(dfL_vol[2], :, k), N1D, N1D)
        dfx_vol = reshape(view(df_vol[1], :, k), N1D, N1D)
        dfy_vol = reshape(view(df_vol[2], :, k), N1D, N1D)

        dfxH_surf = reshape(view(dfH_surf[1], :, k), N1D, N1D)
        dfyH_surf = reshape(view(dfH_surf[2], :, k), N1D, N1D)
        dfxL_surf = reshape(view(dfL_surf[1], :, k), N1D, N1D)
        dfyL_surf = reshape(view(dfL_surf[2], :, k), N1D, N1D)
        dfx_surf = reshape(view(df_surf[1], :, k), N1D, N1D)
        dfy_surf = reshape(view(df_surf[2], :, k), N1D, N1D)

        for j = 1:N1D
            dfx_vol[1, j] = fx_bar_lim_k[2, j]
            dfxH_vol[1, j] = fx_bar_H_k[2, j]
            dfxL_vol[1, j] = fx_bar_L_k[2, j]
            dfx_surf[1, j] = -fx_bar_lim_k[1, j]
            dfxH_surf[1, j] = -fx_bar_H_k[1, j]
            dfxL_surf[1, j] = -fx_bar_L_k[1, j]
            for i = 2:N1D-1
                dfx_vol[i, j] = fx_bar_lim_k[i+1, j] - fx_bar_lim_k[i, j]
                dfxH_vol[i, j] = fx_bar_H_k[i+1, j] - fx_bar_H_k[i, j]
                dfxL_vol[i, j] = fx_bar_L_k[i+1, j] - fx_bar_L_k[i, j]
                # surf contribution is zero
            end
            dfx_vol[N1D, j] = -fx_bar_lim_k[N1D, j]
            dfxH_vol[N1D, j] = -fx_bar_H_k[N1D, j]
            dfxL_vol[N1D, j] = -fx_bar_L_k[N1D, j]
            dfx_surf[N1D, j] = fx_bar_lim_k[N1D+1, j]
            dfxH_surf[N1D, j] = fx_bar_H_k[N1D+1, j]
            dfxL_surf[N1D, j] = fx_bar_L_k[N1D+1, j]
        end

        for i = 1:N1D
            dfy_vol[i, 1] = fy_bar_lim_k[i, 2]
            dfyH_vol[i, 1] = fy_bar_H_k[i, 2]
            dfyL_vol[i, 1] = fy_bar_L_k[i, 2]
            dfy_surf[i, 1] = -fy_bar_lim_k[i, 1]
            dfyH_surf[i, 1] = -fy_bar_H_k[i, 1]
            dfyL_surf[i, 1] = -fy_bar_L_k[i, 1]
            for j = 2:N1D-1
                dfy_vol[i, j] = fy_bar_lim_k[i, j+1] - fy_bar_lim_k[i, j]
                dfyH_vol[i, j] = fy_bar_H_k[i, j+1] - fy_bar_H_k[i, j]
                dfyL_vol[i, j] = fy_bar_L_k[i, j+1] - fy_bar_L_k[i, j]
            end
            dfy_vol[i, N1D] = -fy_bar_lim_k[i, N1D]
            dfyH_vol[i, N1D] = -fy_bar_H_k[i, N1D]
            dfyL_vol[i, N1D] = -fy_bar_L_k[i, N1D]
            dfy_surf[i, N1D] = fy_bar_lim_k[i, N1D+1]
            dfyH_surf[i, N1D] = fy_bar_H_k[i, N1D+1]
            dfyL_surf[i, N1D] = fy_bar_L_k[i, N1D+1]
        end

        # Check dfxy_vol + dfxy_surf = M rhsxy
        for j = 1:N1D
            for i = 1:N1D
                wJq_i = Jq_k[i, j] * wq_k[i, j]
                diffx = wJq_i * rhsxyU_k[i, j][1] - dfx_vol[i, j] - dfx_surf[i, j]
                diffx_H = wJq_i * rhsxyH_k[i, j][1] - dfxH_vol[i, j] - dfxH_surf[i, j]
                diffx_L = wJq_i * rhsxyL_k[i, j][1] - dfxL_vol[i, j] - dfxL_surf[i, j]
                diffy = wJq_i * rhsxyU_k[i, j][2] - dfy_vol[i, j] - dfy_surf[i, j]
                diffy_H = wJq_i * rhsxyH_k[i, j][2] - dfyH_vol[i, j] - dfyH_surf[i, j]
                diffy_L = wJq_i * rhsxyL_k[i, j][2] - dfyL_vol[i, j] - dfyL_surf[i, j]
                tol = 1e-12
                if (norm(diffx) > tol || norm(diffx_H) > tol || norm(diffx_L) > tol ||
                    norm(diffy) > tol || norm(diffy_H) > tol || norm(diffy_L) > tol)
                    @show k, i, j, diffx, diffx_H, diffx_L, diffy, diffy_H, diffy_L
                end
            end
        end

        # Calculate low, high, limited solution entropy estimate
        entropy_estimate_vol_L = zero(SVector{Nd,Float64})   # vT \Delta_vol f^L
        entropy_estimate_surf_L = zero(SVector{Nd,Float64})   # vT \Delta_surf f^L
        entropy_estimate_L = zero(SVector{Nd,Float64})   # vT \Delta f^L
        entropy_estimate_vol_H = zero(SVector{Nd,Float64})   # vT \Delta_vol f^H
        entropy_estimate_surf_H = zero(SVector{Nd,Float64})   # vT \Delta_surf f^H
        entropy_estimate_H = zero(SVector{Nd,Float64})   # vT \Delta f^H
        entropy_estimate_vol = zero(SVector{Nd,Float64})   # vT \Delta_vol f
        entropy_estimate_surf = zero(SVector{Nd,Float64})   # vT \Delta_surf f
        entropy_estimate = zero(SVector{Nd,Float64})   # vT \Delta f
        for i = 1:Nq
            wJq_i = wq[i] * Jq[i, k]
            entropy_estimate_vol_L += SVector(sum(vq[i, k] .* dfL_vol[1][i, k]), sum(vq[i, k] .* dfL_vol[2][i, k]))
            entropy_estimate_surf_L += SVector(sum(vq[i, k] .* dfL_surf[1][i, k]), sum(vq[i, k] .* dfL_surf[2][i, k]))
            entropy_estimate_L += wJq_i * SVector(sum(vq[i, k] .* rhsxyL[i, k][1]), sum(vq[i, k] .* rhsxyL[i, k][2]))
            entropy_estimate_vol_H += SVector(sum(vq[i, k] .* dfH_vol[1][i, k]), sum(vq[i, k] .* dfH_vol[2][i, k]))
            entropy_estimate_surf_H += SVector(sum(vq[i, k] .* dfH_surf[1][i, k]), sum(vq[i, k] .* dfH_surf[2][i, k]))
            entropy_estimate_H += wJq_i * SVector(sum(vq[i, k] .* rhsxyH[i, k][1]), sum(vq[i, k] .* rhsxyH[i, k][2]))
            entropy_estimate_vol += SVector(sum(vq[i, k] .* df_vol[1][i, k]), sum(vq[i, k] .* df_vol[2][i, k]))
            entropy_estimate_surf += SVector(sum(vq[i, k] .* df_surf[1][i, k]), sum(vq[i, k] .* df_surf[2][i, k]))
            entropy_estimate += wJq_i * SVector(sum(vq[i, k] .* rhsxyU[i, k][1]), sum(vq[i, k] .* rhsxyU[i, k][2]))
        end

        # Calculate theory entropy estimates
        sum_Bpsi = zero(SVector{Nd,Float64})   # 1T B psi
        sum_Bpsitilde = zero(SVector{Nd,Float64})   # 1T B psi_tilde
        vftildeBfH = zero(SVector{Nd,Float64})   # vf_tilde^T B fH
        vfBfH = zero(SVector{Nd,Float64})   # vf^T B fH
        vfBfL = zero(SVector{Nd,Float64})   # vf^T B fL
        for i = 1:Nfp
            iq = fq2q[i]
            uf = Uq[iq, k]
            vf = v_ufun(equation, uf)
            Bxy_i = Bx(i, k, discrete_data, dim_type(equation))
            sum_Bpsi += Bxy_i .* psi_ufun(equation, uf)
            sum_Bpsitilde += Bxy_i .* psi_ufun(equation, u_tilde[Nq+i, k])
            vftildeBfH += Bxy_i .* SVector(sum(v_tilde[Nq+i, k] .* fstar_H[i, k][1]), sum(v_tilde[Nq+i, k] .* fstar_H[i, k][2]))
            vfBfH += Bxy_i .* SVector(sum(vf .* fstar_H[i, k][1]), sum(vf .* fstar_H[i, k][2]))
            vfBfL += Bxy_i .* SVector(sum(vf .* fstar_L[i, k][1]), sum(vf .* fstar_L[i, k][2]))
        end

        diff_vol_L = entropy_estimate_vol_L - sum_Bpsi
        diff_surf_L = entropy_estimate_surf_L + vfBfL
        diff_L = entropy_estimate_L - sum_Bpsi + vfBfL
        diff_vol_H = entropy_estimate_vol_H - sum_Bpsitilde + (vftildeBfH - vfBfH)
        diff_surf_H = entropy_estimate_surf_H + vfBfH
        diff_H = entropy_estimate_H - sum_Bpsitilde + vftildeBfH
        diff_vol = entropy_estimate_vol - sum_Bpsi
        diff_surf = entropy_estimate_surf + vfBfL
        diff = entropy_estimate - sum_Bpsi + vfBfL
        tol = 1e-10
        if diff_vol_L[1] > tol || diff_vol_L[2] > tol ||
           abs(diff_surf_L[1]) > tol || abs(diff_surf_L[2]) > tol ||
           diff_L[1] > tol || diff_L[2] > tol ||
           diff_vol[1] > tol || diff_vol[2] > tol ||
           abs(diff_surf[1]) > tol || abs(diff_surf[2]) > tol ||
           diff[1] > tol || diff[2] > tol
            println("Violates entropy at element $k, $diff_L, $diff")
        end
    end
end


######################
# In DGInitialize.jl #
######################
function initialize_preallocations(param, md, sizes)
    (; Np, Nh, Nq, Nfp, Nc, Ns) = sizes

    K = num_elements(param)
    Nd = dim(param.equation)
    N1D = Nd == 1 ? 1 : param.N + 1      # TODO: hardcoded

    Uq = zeros(SVector{Nc,Float64}, Nq, K)
    vq = zeros(SVector{Nc,Float64}, Nq, K)
    psiq = zeros(SVector{Nd,Float64}, Nq, K)
    u_tilde = zeros(SVector{Nc,Float64}, Nh, K)
    v_tilde = zeros(SVector{Nc,Float64}, Nh, K)
    UfL = zeros(SVector{Nc,Float64}, Nfp, K)
    UPL = zeros(SVector{Nc,Float64}, Nfp, K)
    VUfL = zeros(SVector{Nc,Float64}, Nfp, K)
    VUfH = zeros(SVector{Nc,Float64}, Nfp, K)
    VUPL = zeros(SVector{Nc,Float64}, Nfp, K)
    VUPH = zeros(SVector{Nc,Float64}, Nfp, K)
    psif = zeros(SVector{Nd,Float64}, Nfp, K)
    psiP = zeros(SVector{Nd,Float64}, Nfp, K)
    rhsH = zeros(SVector{Nc,Float64}, Nq, K)
    rhsL = zeros(SVector{Nc,Float64}, Nq, K)
    rhsU = zeros(SVector{Nc,Float64}, Nq, K)
    rhsxyH = zeros(SVector{Nd,SVector{Nc,Float64}}, Nq, K)
    rhsxyL = zeros(SVector{Nd,SVector{Nc,Float64}}, Nq, K)
    rhsxyU = zeros(SVector{Nd,SVector{Nc,Float64}}, Nq, K)
    BF_H = zeros(SVector{Nd,SVector{Nc,Float64}}, Nfp, K)
    BF_L = zeros(SVector{Nd,SVector{Nc,Float64}}, Nfp, K)
    fstar_H = zeros(SVector{Nd,SVector{Nc,Float64}}, Nfp, K)
    fstar_L = zeros(SVector{Nd,SVector{Nc,Float64}}, Nfp, K)
    Larr = zeros(Float64, K, Ns)
    L_local_arr = zeros(Float64, Nq + N1D, Nd, K, Ns)
    θ_arr = zeros(Float64, K, Ns)                # TODO: rename F, eta to theta
    θ_local_arr = zeros(Float64, Nfp, K, Ns)
    resW = zeros(SVector{Nc,Float64}, Nq, K)
    resZ = zeros(SVector{Nc,Float64}, Nq, K)
    indicator = zeros(Float64, Nq, K)
    indicator_modal = zeros(Float64, Np, K)
    smooth_indicator = zeros(Float64, K)
    LPmodels = initialize_LP_models(N1D)

    prealloc = Preallocation{Nc,Nd}(Uq, vq, psiq, u_tilde, v_tilde, UfL, UPL, VUfL, VUfH, VUPL, VUPH, psif, psiP,
        rhsH, rhsL, rhsU, rhsxyH, rhsxyL, rhsxyU, BF_H, BF_L, fstar_H, fstar_L,
        Larr, L_local_arr, θ_arr, θ_local_arr,
        resW, resZ,
        indicator, indicator_modal, smooth_indicator)
    return prealloc
end

function initialize_LP_models(N1D)
    N1Dm1 = N1D - 1

    modelx = Model(HiGHS.Optimizer; add_bridges=false)
    set_silent(modelx)
    # set_string_names_on_creation(modelx, false)
    @variable(modelx, lx[1:N1Dm1, 1:N1D])
    # Entropy stability constraint
    @constraint(modelx, con_es, sum(lx) <= 0.0)
    # Lower bound of limiting factors
    @constraint(modelx, con_lbound, lx .>= 0.0)
    # Upper bound (positivity) of limiting factors
    @constraint(modelx, con_ubound, lx .<= 1.0)
    # Objective
    @objective(modelx, Max, sum(lx))
    # @objective(modelx, Min, sum(lx.^2))

    modely = Model(HiGHS.Optimizer; add_bridges=false)
    set_silent(modely)
    # set_string_names_on_creation(modely, false)
    @variable(modely, ly[1:N1D, 1:N1Dm1])
    # Entropy stability constraint
    @constraint(modely, con_es, sum(ly) <= 0.0)
    # Lower bound of limiting factors
    @constraint(modely, con_lbound, ly .>= 0.0)
    # Upper bound (positivity) of limiting factors
    @constraint(modely, con_ubound, ly .<= 1.0)
    # Objective
    @objective(modely, Max, sum(ly))
    # @objective(modely, Min, sum(ly.^2))

    LPmodels = ([copy(modelx) for _ in 1:Threads.nthreads()],
        [copy(modely) for _ in 1:Threads.nthreads()])

    for i = 1:2
        for t = 1:1:Threads.nthreads()
            set_optimizer(LPmodels[i][t], HiGHS.Optimizer; add_bridges=false)
            set_silent(LPmodels[i][t])
        end
    end

    return LPmodels
end


############
# types.jl #
############

struct SubcellLimiterCache{DIM,Nc} <: LimiterCache{DIM,Nc}
    uL_k::Array{SVector{Nc,Float64},2}
    P_k::Array{SVector{Nc,Float64},2}
    f_bar_H::NTuple{DIM,Array{SVector{Nc,Float64},2}}
    f_bar_L::NTuple{DIM,Array{SVector{Nc,Float64},2}}
    f_bar_lim::NTuple{DIM,Array{SVector{Nc,Float64},2}}       # TODO: unnecessary
    s_modified::Array{Float64,2}
    var_s_modified::Array{Float64,2}
    lbound_s_modified::Array{Float64,2}
    # TODO: use array so the value could be mutated... not a clean solution
    s_modified_min::Array{Float64,1}              # Global s_modified minimum
    smooth_factor::Array{Float64,2}
    sum_Bpsi::Array{SVector{DIM,Float64},1}   # 1T B_k psi_k
    sum_dvfbarL::Array{SVector{DIM,Float64},1}   # 1T (Dv)^T fbar_H
    sum_dvfbarH::Array{SVector{DIM,Float64},1}   # 1T (Dv)^T fbar_H
    dvdf::NTuple{DIM,Array{Float64,2}}    # (Dv)^T (fbar_H-fbar_L)
end

SubcellLimiterCache{DIM,Nc}(; K=0, Nq=0, N1D=0, Ns=Ns, Nthread=1, s_modified_min=0) where {DIM,Nc} =
    SubcellLimiterCache{DIM,Nc}(zeros(SVector{Nc,Float64}, Nq, Nthread),
        zeros(SVector{Nc,Float64}, Nq, Nthread),
        tuple([zeros(SVector{Nc,Float64}, Nq + N1D, K) for _ in 1:DIM]...),
        tuple([zeros(SVector{Nc,Float64}, Nq + N1D, K) for _ in 1:DIM]...),
        tuple([zeros(SVector{Nc,Float64}, Nq + N1D, K) for _ in 1:DIM]...),
        zeros(Float64, Nq, K),
        zeros(Float64, Nq, K),
        zeros(Float64, Nq, K),
        zeros(Float64, 1),
        zeros(Float64, K, Ns),
        zeros(SVector{DIM,Float64}, K),
        zeros(SVector{DIM,Float64}, K),
        zeros(SVector{DIM,Float64}, K),
        tuple([zeros(Float64, Nq - N1D, K) for _ in 1:DIM]...))

struct Preallocation{Nc,DIM}
    Uq::Array{SVector{Nc,Float64},2}
    vq::Array{SVector{Nc,Float64},2}       # entropy variables at quad points
    psiq::Array{SVector{DIM,Float64},2}
    u_tilde::Array{SVector{Nc,Float64},2}       # entropy projected conservative variables
    v_tilde::Array{SVector{Nc,Float64},2}       # projected entropy variables
    UfL::Array{SVector{Nc,Float64},2}     # TODO: Redundant
    UPL::Array{SVector{Nc,Float64},2}     # TODO: Redundant
    VUfL::Array{SVector{Nc,Float64},2}     # TODO: redundant, refactor
    VUfH::Array{SVector{Nc,Float64},2}     # TODO: redundant, refactor
    VUPL::Array{SVector{Nc,Float64},2}
    VUPH::Array{SVector{Nc,Float64},2}
    psif::Array{SVector{DIM,Float64},2}
    psiP::Array{SVector{DIM,Float64},2}
    rhsH::Array{SVector{Nc,Float64},2}
    rhsL::Array{SVector{Nc,Float64},2}
    rhsU::Array{SVector{Nc,Float64},2}
    rhsxyH::Array{SVector{DIM,SVector{Nc,Float64}},2}
    rhsxyL::Array{SVector{DIM,SVector{Nc,Float64}},2}
    rhsxyU::Array{SVector{DIM,SVector{Nc,Float64}},2}
    BF_H::Array{SVector{DIM,SVector{Nc,Float64}},2}
    BF_L::Array{SVector{DIM,SVector{Nc,Float64}},2}
    fstar_H::Array{SVector{DIM,SVector{Nc,Float64}},2}
    fstar_L::Array{SVector{DIM,SVector{Nc,Float64}},2}
    Larr::Array{Float64,2}
    L_local_arr::Array{Float64,4}
    θ_arr::Array{Float64,2}
    θ_local_arr::Array{Float64,3}
    resW::Array{SVector{Nc,Float64},2}
    resZ::Array{SVector{Nc,Float64},2}
    indicator::Array{Float64,2}
    indicator_modal::Array{Float64,2}
    smooth_indicator::Array{Float64,1}     # modal energyN/total_energy
    LPmodels::NTuple{DIM,Array{Model,1}}
end



