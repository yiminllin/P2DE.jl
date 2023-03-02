using Revise
using UnPack
using StaticArrays
using DataFrames
using JLD2
using StartUpDG

using P2DE

function exact_sol(eqn,x,y,t)
    γ = get_γ(eqn)
    x0 = 9.0
    y0 = 5.0
    beta = 8.5
    r2 = @. (x-x0-t)^2 + (y-y0)^2

    u   = @. 1 - beta*exp(1-r2)*(y-y0)/(2*pi)
    v   = @. beta*exp(1-r2)*(x-x0-t)/(2*pi)
    rho = @. 1 - (1/(8*γ*pi^2))*(γ-1)/2*(beta*exp(1-r2))^2
    rho = @. rho^(1/(γ-1))
    p   = @. rho^γ

    return (rho, u, v, p)
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

γ = 1.4
param = Param(N=3, K=(40,20), xL=(0.0,0.0), xR=(20.0,10.0),
              global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
              timestepping_param=TimesteppingParameter(T=1.0, CFL=0.5, dt0=1e-3, t0=0.0),
              limiting_param=LimitingParameter(ζ=0.1, η=1.0),
              postprocessing_param=PostprocessingParameter(output_interval=100),
              equation=CompressibleEulerIdealGas{Dim2}(γ),
              rhs_type=ESLimitedLowOrderPos(low_order_surface_flux_type=LaxFriedrichsOnNodalVal(),
                                            high_order_surface_flux_type=LaxFriedrichsOnProjectedVal()),
              approximation_basis_type=GaussCollocation(),
              entropyproj_limiter_type=NodewiseScaledExtrapolation(),
              positivity_limiter_type=SubcellLimiter())

T = param.timestepping_param.T
N = param.N
K = param.K
equation = param.equation

rd,md,discrete_data,bcdata,prealloc,caches = initialize_DG(param,initial_condition,initial_boundary_conditions)

data_hist = SSP33!(param,discrete_data,bcdata,prealloc,caches)

err_data = calculate_error(prealloc.Uq,param,discrete_data,md,prealloc,exact_sol)

construct_vtk_file!(caches.postprocessing_cache,param,data_hist,"./outputs/isentropic-vortex","isentropic-vortex")