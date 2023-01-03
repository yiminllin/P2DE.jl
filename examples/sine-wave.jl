using Revise
using UnPack
using StaticArrays
using DataFrames

using SimplePDE

function exact_sol(eqn,x,t)
    return 1e-6 + 1.0 + sin(2*pi*(x-t)), 1.0, 1.0
    # return 2.0 + sin(2*pi*(x-t)), 1.0, 1.0
end

function initial_boundary_conditions(param,md)
    @unpack K    = param
    @unpack mapP = md

    # Make periodic
    mapP[1]   = mapP[end] 
    mapP[end] = 1

    mapI = []
    mapO = []
    inflowarr = []

    bcdata = BCData(mapP,mapI,mapO,inflowarr)

    return bcdata
end

function initial_condition(param,x)
    # TODO: use getter
    t0 = param.timestepping_param.t0
    return primitive_to_conservative(param.equation,SVector{3,Float64}(exact_sol(param.equation,x,t0)))
end

γ = 1.4
param = Param(N=3, K=80, XL=0.0, XR=1.0,
              global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
              timestepping_param=TimesteppingParameter(T=0.1, CFL=0.75, dt0=1e-4, t0=0.0),
              limiting_param=LimitingParameter(ζ=0.1, η=1.0),
              postprocessing_param=PostprocessingParameter(output_interval=1000),
              equation=CompressibleEulerIdealGas{Dim1}(γ),
              rhs_type=ESLimitedLowOrderPos(low_order_surface_flux_type=LaxFriedrichsOnNodalVal(),
                                            high_order_surface_flux_type=LaxFriedrichsOnProjectedVal()),
              approximation_basis_type=GaussCollocation(),
              entropyproj_limiter_type=ExponentialFilter(),
              positivity_limiter_type=SubcellLimiter())

T = param.timestepping_param.T
N = param.N
K = param.K
equation = param.equation

rd_gauss,md_gauss,discrete_data_gauss,rd_LGL,md_LGL,discrete_data_LGL,transfer_ops,bcdata,prealloc = initialize_DG(param,initial_condition,initial_boundary_conditions)

data_hist = SSP33!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc)

err_data = calculate_error(prealloc.Uq,param,discrete_data_gauss,discrete_data_LGL,md_gauss,md_LGL,prealloc,exact_sol)

plot_component(param,discrete_data_gauss,md_gauss,md_LGL,prealloc,
               [u[1] for u in prealloc.Uq],1,K,0,3,"outputs/figures/sine-wave/N=$N,K=$K,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.positivity_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η).png",
               true,md_gauss.xq,[exact_sol(equation,xi,T)[1] for xi in md_gauss.xq],1,K)
plot_component(param,discrete_data_gauss,md_gauss,md_LGL,prealloc,
               [u[1] for u in prealloc.Uq],Int64(round(K*3/4)),K,0,0.02,"outputs/figures/sine-wave/N=$N,K=$K,rhs=$(param.rhs_type),vproj_limiter=$(param.entropyproj_limiter_type),pos=$(param.positivity_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η),zoom.png",
               true,md_gauss.xq,[exact_sol(equation,xi,T)[1] for xi in md_gauss.xq],Int64(round(K*3/4)),K)

plot_rho_animation(md_gauss,md_LGL,param,prealloc,data_hist,data_hist.Fhist,0,3,
                   "outputs/figures/sine-wave/N=$N,K=$K,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.positivity_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η).gif")

df = DataFrame(param = Param[], data_hist = DataHistory[], err_data = ErrorData[])
write_to_jld2(param,data_hist,err_data,df,"outputs/jld2/sine-wave/sine-wave.jld2")