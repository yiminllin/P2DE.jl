using Revise
using UnPack
using StaticArrays
using DataFrames
using JLD2
using StartUpDG

using P2DE

function exact_sol(eqn,x,y,t)
    # return 1e-6 + 1.0 + sin(2*pi*(x-t)), 1.0, 1.0
    # return 2.0 + sin(2*pi*(x-t)), 1.0, 0.0, 1.0
    return 2.0 + sin(2*pi*(y-t)), 0.0, 1.0, 1.0
end

function initial_boundary_conditions(param,md)
    @unpack K,xf,yf,mapM,mapP,mapB = md

    Nc = get_num_components(param.equation)
    # Make periodic
    md = make_periodic(md)
    @unpack mapP = md

    mapI = []
    mapO = []
    inflowarr = []

    bcdata = BCData{Nc}(mapP,mapI,mapO,inflowarr)

    return bcdata
end

function initial_condition(param,x,y)
    # TODO: use getter
    t0 = param.timestepping_param.t0
    return primitive_to_conservative(param.equation,SVector(exact_sol(param.equation,x,y,t0)))
end

jld_path = "outputs/jld2/2D/sine-wave/sine-wave.jld2"

γ = 1.4
param = Param(N=3, K=(20,20), xL=(0.0,0.0), xR=(1.0,1.0),
              global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
              timestepping_param=TimesteppingParameter(T=0.1, CFL=0.5, dt0=1e-4, t0=0.0),
              limiting_param=LimitingParameter(ζ=0.1, η=1.0),
              postprocessing_param=PostprocessingParameter(output_interval=1000),
              equation=CompressibleEulerIdealGas{Dim2}(γ),
              rhs_type=EntropyStable(surface_flux_type=LaxFriedrichsOnProjectedVal()),
              approximation_basis_type=GaussCollocation(),
              entropyproj_limiter_type=NoEntropyProjectionLimiter(),
              positivity_limiter_type=NoPositivityLimiter())

T = param.timestepping_param.T
N = param.N
K = param.K
equation = param.equation

rd_gauss,md_gauss,discrete_data_gauss,rd_LGL,md_LGL,discrete_data_LGL,transfer_ops,bcdata,prealloc = initialize_DG(param,initial_condition,initial_boundary_conditions)

data_hist = SSP33!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc)

err_data = calculate_error(prealloc.Uq,param,discrete_data_gauss,discrete_data_LGL,md_gauss,md_LGL,prealloc,exact_sol)

using Plots
gr(aspect_ratio=1, legend=false,
   markerstrokewidth=0, markersize=2,xlim=[0,1],ylim=[0,1])
x = md_LGL.xq[:]
y = md_LGL.yq[:]
rho = [x[1] for x in prealloc.Uq][:]
scatter(x,y,rho,zcolor=rho,camera=(0,90))
savefig("~/Desktop/test.png")

# # TODO: refactor
# plot_path     = "outputs/figures/2D/sine-wave/N=$N,K=$K,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.positivity_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η).png"
# plotzoom_path = "outputs/figures/2D/sine-wave/N=$N,K=$K,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.positivity_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η),zoom.png"
# gif_path      = "outputs/figures/2D/sine-wave/N=$N,K=$K,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.positivity_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η).gif"

# plot_component(param,discrete_data_gauss,md_gauss,md_LGL,prealloc,
#                [u[1] for u in prealloc.Uq],1,K,0,3,plot_path,
#                true,md_gauss.xq,[exact_sol(equation,xi,T)[1] for xi in md_gauss.xq],1,K)
# plot_component(param,discrete_data_gauss,md_gauss,md_LGL,prealloc,
#                [u[1] for u in prealloc.Uq],Int64(round(K*3/4)),K,0,0.02,plotzoom_path,
#                true,md_gauss.xq,[exact_sol(equation,xi,T)[1] for xi in md_gauss.xq],Int64(round(K*3/4)),K)

# plot_rho_animation(md_gauss,md_LGL,param,prealloc,data_hist,data_hist.Fhist,0,3,
#                    gif_path)

# df = DataFrame([name => [] for name in (fieldnames(Param)..., fieldnames(ErrorData)...,:data_history)])
# write_to_jld2(param,data_hist,err_data,df,jld_path)

# df = load(jld_path,"data")
# visualize_error_data(df)