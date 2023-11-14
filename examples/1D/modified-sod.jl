using Revise
using StaticArrays
using DataFrames
using MAT
using StartUpDG

using P2DE

function exact_sol(eqn,x,t)
    # TODO: not exact solution
    rho = (x < 0.3) ? 1.0  : 0.125
    u   = (x < 0.3) ? 0.75 : 0.0
    p   = (x < 0.3) ? 1.0  : 0.1
    return rho,u,p
end

function initial_boundary_conditions(param,md)
    @unpack K,equation = param
    @unpack mapP       = md

    # Shu-osher
    mapI = [1]
    mapO = [2*K]
 
    rho       = 1.0
    u         = 0.75
    p         = 1.0
    inflowarr = [primitive_to_conservative(equation, (rho,u,p))]

    bcdata = BCData(mapP,mapI,mapO,inflowarr)

    return bcdata
end

function initial_condition(param,x)
    # TODO: use getter
    t0 = param.timestepping_param.t0
    return primitive_to_conservative(param.equation,SVector{3,Float64}(exact_sol(param.equation,x,t0)))
end

Uhist = []
limiter_types = [(EntropyStable(surface_flux_type=LaxFriedrichsOnProjectedVal()), NoRHSLimiter());
                (StdDGLimitedLowOrderPos(low_order_surface_flux_type=LaxFriedrichsOnNodalVal(),high_order_surface_flux_type=LaxFriedrichsOnProjectedVal()), SubcellLimiter(bound_type=PositivityAndCellEntropyBound(),shockcapture_type=NoShockCapture()));
                (StdDGLimitedLowOrderPos(low_order_surface_flux_type=LaxFriedrichsOnNodalVal(),high_order_surface_flux_type=LaxFriedrichsOnProjectedVal()), SubcellLimiter(bound_type=PositivityAndMinEntropyBound(),shockcapture_type=NoShockCapture()));
                (StdDGLimitedLowOrderPos(low_order_surface_flux_type=LaxFriedrichsOnNodalVal(),high_order_surface_flux_type=LaxFriedrichsOnProjectedVal()), SubcellLimiter(bound_type=PositivityBound(),shockcapture_type=NoShockCapture()))]

K = 100
xplot = zeros(Float64,11,K)

for limiter_type in limiter_types
rhs_type, rhs_limiter_type = limiter_type

γ = 1.4
# param = Param(N=3, K=100, xL=0.0, xR=1.0,
#               global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
#               timestepping_param=TimesteppingParameter(T=0.2, CFL=0.5, dt0=1e-4, t0=0.0),
#               limiting_param=LimitingParameter(ζ=0.5, η=1.0),
#               postprocessing_param=PostprocessingParameter(output_interval=1000),
#               equation=CompressibleEulerIdealGas{Dim1}(γ),
#               rhs_type=StdDGLimitedLowOrderPos(low_order_surface_flux_type=LaxFriedrichsOnNodalVal(),
#                                                high_order_surface_flux_type=LaxFriedrichsOnProjectedVal()),
#               approximation_basis_type=LobattoCollocation(),
#               entropyproj_limiter_type=NoEntropyProjectionLimiter(),
#               rhs_limiter_type=SubcellLimiter(bound_type=PositivityAndCellEntropyBound(),
#                                               shockcapture_type=NoShockCapture()))
global K
param = Param(N=3, K=K, xL=0.0, xR=1.0,
              global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
              timestepping_param=TimesteppingParameter(T=0.2, CFL=0.5, dt0=1e-4, t0=0.0),
              limiting_param=LimitingParameter(ζ=0.5, η=1.0),
              postprocessing_param=PostprocessingParameter(output_interval=1000),
              equation=CompressibleEulerIdealGas{Dim1}(γ),
              rhs_type=rhs_type,
              approximation_basis_type=LobattoCollocation(),
              entropyproj_limiter_type=NoEntropyProjectionLimiter(),
              rhs_limiter_type=rhs_limiter_type)

T = param.timestepping_param.T
N = param.N
K = param.K
equation = param.equation

rd,md,discrete_data,bcdata,prealloc,caches = initialize_DG(param,initial_condition,initial_boundary_conditions)

data_hist = SSP33!(param,discrete_data,bcdata,prealloc,caches)

err_data = calculate_error(prealloc.Uq,param,discrete_data,md,prealloc,exact_sol)

# plot_path     = "/data/yl184/outputs/figures/modified-sod/N=$N,K=$K,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),lim=$(param.rhs_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η).png"
# gif_path      = "/data/yl184/outputs/figures/modified-sod/N=$N,K=$K,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),lim=$(param.rhs_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η),zoom.png"

# plot_component(param,discrete_data,md,prealloc,
#                [u[1] for u in prealloc.Uq],1,K,0,1.5,
#                plot_path)

@unpack Vp = rd
push!(Uhist, Vp*prealloc.Uq)

for k = 1:K
    xplot[:,k] = Vp*md.xq[:,k]
end
end

using Plots
# gr(x_lim=[0.0,1.0],ylim=[0.0,1.5],label=false,legend=false)
gr(x_lim=[0.0,1.0],ylim=[0.0,1.5],legendfontsize=12,tickfontsize=12)

# add_nan(x) = vcat(x, fill(NaN,1,size(x,2)))
# add_end_to_x(x) = vcat(x, x[end,:]')
add_nan(x) = vcat(fill(NaN,1,size(x,2)), x[2:end,:])
# add_nan(x) = vcat(x[1:end-1,:], fill(NaN,1,size(x,2)))
add_end_to_x(x) = x

plot(add_end_to_x(xplot)[:],add_nan([u[1] for u in Uhist[4]])[:],lw=2,label="DGSEM")
plot!(add_end_to_x(xplot)[:],add_nan([u[1] for u in Uhist[1]])[:],lw=2,ls=:dash,label="ESDG")
plot!(add_end_to_x(xplot)[:],add_nan([u[1] for u in Uhist[2]])[:],lw=2,ls=:dot,label="DGSEM + Cell entropy stability")
plot!(add_end_to_x(xplot)[:],add_nan([u[1] for u in Uhist[3]])[:],lw=2,ls=:dashdot,label="DGSEM + Minimum entropy principle")

plot_path = "./outputs/modified-sod-K=$(K).png"
savefig(plot_path)

# xl = div(K,5)*11
# xr = 2*div(K,5)*11
xl = div(K,5)*12
xr = 2*div(K,5)*12

gr(ylim=[0.4,1.3],legendfontsize=12,tickfontsize=12)
plot(add_end_to_x(xplot)[xl:xr],add_nan([u[1] for u in Uhist[4]])[xl:xr],lw=3,label="DGSEM")
plot!(add_end_to_x(xplot)[xl:xr],add_nan([u[1] for u in Uhist[1]])[xl:xr],lw=3,label="ESDG")
plot!(add_end_to_x(xplot)[xl:xr],add_nan([u[1] for u in Uhist[2]])[xl:xr],lw=3,label="DGSEM + Cell entropy stability")
plot!(add_end_to_x(xplot)[xl:xr],add_nan([u[1] for u in Uhist[3]])[xl:xr],lw=3,label="DGSEM + Minimum entropy principle")

plot_path = "./outputs/modified-sod-zoom-K=$(K).png"
savefig(plot_path)
