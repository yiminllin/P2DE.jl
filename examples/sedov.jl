using Revise
using UnPack
using StaticArrays
using DataFrames
using JLD2

using P2DE

function exact_sol(eqn,x,t)
    rho = 1.0
    u   = 0.0
    p   = (eqn.γ-1)*1e-12
    return rho, u, p
end

function initial_boundary_conditions(param,md)
    @unpack K    = param
    @unpack mapP = md

    Nc = get_num_components(param.equation)
    # Make periodic
    mapP[1]   = mapP[end] 
    mapP[end] = 1

    mapI = []
    mapO = []
    inflowarr = []

    bcdata = BCData{Nc}(mapP,mapI,mapO,inflowarr)

    return bcdata
end

function initial_condition(param,x)
    # TODO: use getter
    γ = param.equation.γ
    t0 = param.timestepping_param.t0
    rho = 1.0
    u   = 0.0
    # TODO: hardcoded
    dx  = (param.XR-param.XL)/param.K
    dr  = 1*dx
    p   = abs(x) < dr ? (γ-1)*1.0/pi/(dr) : (γ-1)*1e-5
    return primitive_to_conservative(param.equation,SVector{3,Float64}(rho,u,p))
end

jld_path = "outputs/jld2/sedov/sedov.jld2"

γ = 5/3
param = Param(N=2, K=100, xL=-1.5, xR=1.5,
              global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
              timestepping_param=TimesteppingParameter(T=1.0, CFL=0.75, dt0=1e-4, t0=0.0),
              limiting_param=LimitingParameter(ζ=0.1, η=2.0),
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

rd_gauss,md_gauss,discrete_data_gauss,rd_LGL,md_LGL,discrete_data_LGL,transfer_ops,bcdata,prealloc = initialize_DG(param,initial_condition,initial_boundary_conditions)

data_hist = SSP33!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc)

err_data = calculate_error(prealloc.Uq,param,discrete_data_gauss,discrete_data_LGL,md_gauss,md_LGL,prealloc,exact_sol)

# # TODO: refactor
plot_rho_path = "outputs/figures/sedov/N=$N,K=$K,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.positivity_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η)-rho.png"
plot_E_path   = "outputs/figures/sedov/N=$N,K=$K,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.positivity_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η)-E.png"
gif_path      = "outputs/figures/sedov/N=$N,K=$K,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.positivity_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η).gif"

plot_component(param,discrete_data_gauss,md_gauss,md_LGL,prealloc,
               [u[1] for u in prealloc.Uq],1,K,0,8,plot_rho_path,
               true,md_gauss.xq,[exact_sol(equation,xi,T)[1] for xi in md_gauss.xq],1,K)
plot_component(param,discrete_data_gauss,md_gauss,md_LGL,prealloc,
               [u[3] for u in prealloc.Uq],1,K,0,3,plot_E_path,
               true,md_gauss.xq,[exact_sol(equation,xi,T)[3] for xi in md_gauss.xq],1,K)
plot_rho_animation(md_gauss,md_LGL,param,prealloc,data_hist,data_hist.Fhist,0,8,
                   gif_path)

# df = DataFrame([name => [] for name in (fieldnames(Param)..., fieldnames(ErrorData)...,:data_history)])
# write_to_jld2(param,data_hist,err_data,df,jld_path)

# df = load(jld_path,"data")
# visualize_error_data(df)