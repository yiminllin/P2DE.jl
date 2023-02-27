using Revise
using UnPack
using StaticArrays
using DataFrames

using P2DE

using JLD2

function exact_sol(eqn,x,t)
    rhoL = 1.0
    rhoR = 1e-3
    pL   = (eqn.γ-1)*1e-1
    pR   = (eqn.γ-1)*1e-10
    if (t == 0)
        if x < 0.33
            return rhoL,0.0,pL
        else
            return rhoR,0.0,pR
        end
    end
    xi = (x-0.33)/t
    rhoLstar = 5.4079335349316249*1e-2
    rhoRstar = 3.9999980604299963*1e-3
    vstar    = 0.62183867139173454
    pstar    = 0.51557792765096996*1e-3
    lambda1  = 0.49578489518897934
    lambda3  = 0.82911836253346982
    if xi <= -1/3
        return rhoL, 0.0, pL
    elseif xi <= lambda1
        return (0.75-0.75*xi)^3, 0.75*(1/3+xi), 1/15*(0.75-0.75*xi)^5
    elseif xi <= vstar
        return rhoLstar, vstar, pstar
    elseif xi <= lambda3
        return rhoRstar, vstar, pstar
    else
        return rhoR, 0.0, pR
    end
end

function initial_boundary_conditions(param,md)
    @unpack K,equation = param
    @unpack mapP       = md

    mapI = [1]
    mapO = [2*K]
 
    rho       = 1.0
    u         = 0.0
    p         = (get_γ(equation)-1)*1e-1
    inflowarr = [primitive_to_conservative(equation, (rho,u,p))]

    bcdata = BCData(mapP,mapI,mapO,inflowarr)

    return bcdata
end

function initial_condition(param,x)
    # TODO: use getter
    t0 = param.timestepping_param.t0
    return primitive_to_conservative(param.equation,SVector{3,Float64}(exact_sol(param.equation,x,t0)))
end

jld_path = "outputs/jld2/leblanc/leblanc.jld2"

γ = 5/3
CFL = 0.8
param = Param(N=2, K=200, xL=0.0, xR=1.0,
              global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
              timestepping_param=TimesteppingParameter(T=2/3, CFL=CFL, dt0=1e-3, t0=0.01),
              limiting_param=LimitingParameter(ζ=0.1, η=1.0),
              postprocessing_param=PostprocessingParameter(output_interval=1000),
              equation=CompressibleEulerIdealGas{Dim1}(γ),
              rhs_type=ESLimitedLowOrderPos(low_order_surface_flux_type=LaxFriedrichsOnNodalVal(),
                                            high_order_surface_flux_type=LaxFriedrichsOnProjectedVal()),
              approximation_basis_type=GaussCollocation(),
              entropyproj_limiter_type=NodewiseScaledExtrapolation(),
              positivity_limiter_type=SubcellLimiter())

T = param.timestepping_param.T
N = param.N
K = param.K
equation = param.equation

rd_gauss,md_gauss,discrete_data_gauss,rd_LGL,md_LGL,discrete_data_LGL,transfer_ops,bcdata,prealloc,rhs_data,limiter_cache,entropyproj_limiter_cache = initialize_DG(param,initial_condition,initial_boundary_conditions)

data_hist = SSP33!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,rhs_data,limiter_cache,entropyproj_limiter_cache)

err_data = calculate_error(prealloc.Uq,param,discrete_data_gauss,discrete_data_LGL,md_gauss,md_LGL,prealloc,exact_sol)

plot_path     = "outputs/figures/leblanc/N=$N,K=$K,CFL=$CFL,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.positivity_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η).png"
plotzoom_path = "outputs/figures/leblanc/N=$N,K=$K,CFL=$CFL,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.positivity_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η),zoom.png"
gif_path      = "outputs/figures/leblanc/N=$N,K=$K,CFL=$CFL,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.positivity_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η),zoom.gif"

plot_component(param,discrete_data_gauss,md_gauss,md_LGL,prealloc,
               [u[1] for u in prealloc.Uq],1,K,0,1.2,
               plot_path,
               true,md_gauss.xq,[exact_sol(equation,xi,T)[1] for xi in md_gauss.xq],1,K)
plot_component(param,discrete_data_gauss,md_gauss,md_LGL,prealloc,
               [u[1] for u in prealloc.Uq],Int64(round(0.7*K)),Int64(round(0.9*K)),0,0.11,
               plotzoom_path,
               true,md_gauss.xq,[exact_sol(equation,xi,T)[1] for xi in md_gauss.xq],Int64(round(0.7*K)),Int64(round(0.9*K)))

plot_rho_animation(md_gauss,md_LGL,param,prealloc,data_hist,data_hist.Fhist,0,1.2,
                   gif_path)

df = DataFrame([name => [] for name in (fieldnames(Param)..., fieldnames(ErrorData)...,:data_history)])
write_to_jld2(param,data_hist,err_data,df,jld_path)

df = load(jld_path,"data")
visualize_error_data(df)