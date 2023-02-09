using Revise
using UnPack
using StaticArrays
using DataFrames
using JLD2

using P2DE

function exact_sol(eqn,x,t)
    return 1e-6 + 1.0 + sin(2*pi*(x-t)), 1.0, 1.0
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

# TODO: refactor convergence
jld_path = "outputs/jld2/sine-wave/sine-wave.jld2"

for limiter_type in [(NoEntropyProjectionLimiter()    ,LaxFriedrichsOnNodalVal()    ,ZhangShuLimiter(),LobattoCollocation());
                     (NoEntropyProjectionLimiter()    ,LaxFriedrichsOnNodalVal()    ,SubcellLimiter() ,LobattoCollocation());
                     (ElementwiseScaledExtrapolation(),LaxFriedrichsOnNodalVal()    ,SubcellLimiter() ,GaussCollocation());
                     (NodewiseScaledExtrapolation()   ,LaxFriedrichsOnNodalVal()    ,SubcellLimiter() ,GaussCollocation());
                     (ElementwiseScaledExtrapolation(),LaxFriedrichsOnProjectedVal(),ZhangShuLimiter(),GaussCollocation());
                     (NodewiseScaledExtrapolation()   ,LaxFriedrichsOnProjectedVal(),ZhangShuLimiter(),GaussCollocation());]
for N in [2;5]
for K in [10;20;40;80]

entropyproj_type,low_order_flux_type,pos_lim_type,discretization_type = limiter_type
γ = 1.4
param = Param(N=N, K=K, XL=0.0, XR=1.0,
              global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
              timestepping_param=TimesteppingParameter(T=0.1, CFL=0.75, dt0=1e-4, t0=0.0),
              limiting_param=LimitingParameter(ζ=0.1, η=0.5),
              postprocessing_param=PostprocessingParameter(output_interval=1000),
              equation=CompressibleEulerIdealGas{Dim1}(γ),
              rhs_type=ESLimitedLowOrderPos(low_order_surface_flux_type=low_order_flux_type,
                                            high_order_surface_flux_type=LaxFriedrichsOnProjectedVal()),
              approximation_basis_type=discretization_type,
              entropyproj_limiter_type=entropyproj_type,
              positivity_limiter_type=pos_lim_type)

T = param.timestepping_param.T
N = param.N
K = param.K
equation = param.equation

rd_gauss,md_gauss,discrete_data_gauss,rd_LGL,md_LGL,discrete_data_LGL,transfer_ops,bcdata,prealloc = initialize_DG(param,initial_condition,initial_boundary_conditions)

data_hist = SSP33!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc)

err_data = calculate_error(prealloc.Uq,param,discrete_data_gauss,discrete_data_LGL,md_gauss,md_LGL,prealloc,exact_sol)

# TODO: refactor
plot_path     = "outputs/figures/sine-wave/N=$N,K=$K,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.positivity_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η).png"
plotzoom_path = "outputs/figures/sine-wave/N=$N,K=$K,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.positivity_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η),zoom.png"
gif_path      = "outputs/figures/sine-wave/N=$N,K=$K,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.positivity_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η).gif"

plot_component(param,discrete_data_gauss,md_gauss,md_LGL,prealloc,
               [u[1] for u in prealloc.Uq],1,K,0,3,plot_path,
               true,md_gauss.xq,[exact_sol(equation,xi,T)[1] for xi in md_gauss.xq],1,K)
plot_component(param,discrete_data_gauss,md_gauss,md_LGL,prealloc,
               [u[1] for u in prealloc.Uq],Int64(round(K*3/4)),K,0,0.02,plotzoom_path,
               true,md_gauss.xq,[exact_sol(equation,xi,T)[1] for xi in md_gauss.xq],Int64(round(K*3/4)),K)

plot_rho_animation(md_gauss,md_LGL,param,prealloc,data_hist,data_hist.Fhist,0,3,
                   gif_path)

df = DataFrame([name => [] for name in (fieldnames(Param)..., fieldnames(ErrorData)...,:data_history)])
write_to_jld2(param,data_hist,err_data,df,jld_path)

end
end
end

df = load(jld_path,"data")
visualize_error_data(df)