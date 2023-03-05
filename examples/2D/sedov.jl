using Revise
using UnPack
using StaticArrays
using DataFrames
using JLD2
using StartUpDG

using P2DE

function exact_sol(eqn,x,y,t)
    γ = get_γ(eqn)
    K1D = 80                # TODO: hardcoded
    r_ini = 4*3/K1D
    r = sqrt(x^2+y^2)
    if (r < r_ini)
        p = (γ-1)*1/pi/r_ini/r_ini
    else
        p = 1e-5
    end
    return 1.0, 0.0, 0.0, p
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
param = Param(N=3, K=(80,80), xL=(-1.5,-1.5), xR=(1.5,1.5),
              global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
              timestepping_param=TimesteppingParameter(T=1.0, CFL=0.5, dt0=1e-3, t0=0.0),
              limiting_param=LimitingParameter(ζ=0.1, η=0.1),
              postprocessing_param=PostprocessingParameter(output_interval=100),
              equation=CompressibleEulerIdealGas{Dim2}(γ),
              rhs_type=ESLimitedLowOrderPos(low_order_surface_flux_type=LaxFriedrichsOnProjectedVal(),
                                            high_order_surface_flux_type=LaxFriedrichsOnProjectedVal()),
              approximation_basis_type=GaussCollocation(),
              entropyproj_limiter_type=NodewiseScaledExtrapolation(),
              positivity_limiter_type=ZhangShuLimiter())

T = param.timestepping_param.T
N = param.N
K = param.K
equation = param.equation

rd,md,discrete_data,bcdata,prealloc,caches = initialize_DG(param,initial_condition,initial_boundary_conditions)

data_hist = SSP33!(param,discrete_data,bcdata,prealloc,caches)

err_data = calculate_error(prealloc.Uq,param,discrete_data,md,prealloc,exact_sol)

construct_vtk_file!(caches.postprocessing_cache,param,data_hist,"/data/yl184/outputs/figures/sedov","sedov")