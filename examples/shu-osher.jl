using Revise
using UnPack
using StaticArrays
using DataFrames
using MAT

using SimplePDE

function exact_sol(eqn,x,t)
    # TODO: not exact solution
    rho = (x < -4.0) ? 3.857143 : 1+.2*sin(5*x)
    u   = (x < -4.0) ? 2.629369 : 0.0
    p   = (x < -4.0) ? 10.3333 : 1.0
    return rho,u,p
end

function initial_boundary_conditions(param,md)
    @unpack K,equation = param
    @unpack mapP       = md

    # Shu-osher
    mapI = [1]
    mapO = [2*K]
 
    rho       = 3.857143
    u         = 2.629369
    p         = 10.3333
    inflowarr = [primitive_to_conservative(equation, (rho,u,p))]

    bcdata = BCData(mapP,mapI,mapO,inflowarr)

    return bcdata
end

function initial_condition(param,x)
    # TODO: use getter
    t0 = param.timestepping_param.t0
    return primitive_to_conservative(param.equation,SVector{3,Float64}(exact_sol(param.equation,x,t0)))
end

γ = 1.4
param = Param(N=3, K=128, XL=-5.0, XR=5.0,
              global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
              timestepping_param=TimesteppingParameter(T=1.8, CFL=0.8, dt0=1e-4, t0=0.0),
              limiting_param=LimitingParameter(ζ=0.1, η=1.0),
              postprocessing_param=PostprocessingParameter(output_interval=1000),
              equation=CompressibleEulerIdealGas{Dim1}(γ),
              rhs_type=ESLimitedLowOrderPos(),
              approximation_basis_type=GaussCollocation(),
              entropyproj_limiter_type=ExponentialFilter(),
              positivity_limiter_type=ZhangShuLimiter())

T = param.timestepping_param.T
N = param.N
K = param.K
equation = param.equation

rd_gauss,md_gauss,discrete_data_gauss,rd_LGL,md_LGL,discrete_data_LGL,transfer_ops,bcdata,prealloc = initialize_DG(param,initial_condition,initial_boundary_conditions)

data_hist = SSP33!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc)

err_data = calculate_error(prealloc.Uq,param,discrete_data_gauss,discrete_data_LGL,md_gauss,md_LGL,prealloc,exact_sol)

weno_sol = matread("data/weno5_shuosher.mat")
plot_component(param,discrete_data_gauss,md_gauss,md_LGL,prealloc,
               [u[1] for u in prealloc.Uq],1,K,0,6,
               "outputs/figures/shu-osher/N=$N,K=$K,rhstype=$(param.rhs_type),entropyproj_limiter_type=$(param.entropyproj_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η).png",
               true,weno_sol["x"],weno_sol["rho"],1,size(weno_sol["x"],2))

plot_rho_animation(md_gauss,md_LGL,param,prealloc,data_hist,data_hist.Fhist,0,6,
                   "outputs/figures/shu-osher/N=$N,K=$K,rhstype=$(param.rhs_type),entropyproj_limiter_type=$(param.entropyproj_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η).gif")

df = DataFrame(param = Param[], data_hist = DataHistory[], err_data = ErrorData[])
write_to_jld2(param,data_hist,err_data,df,"outputs/jld2/shu-osher/shu-osher.jld2")