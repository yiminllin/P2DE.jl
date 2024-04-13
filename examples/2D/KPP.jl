using Revise
using StaticArrays
using DataFrames
using JLD2
using StartUpDG

using P2DE

function exact_sol(eqn, x, y, t)
    r2 = x^2 + y^2
    if r2 <= 1
        return SVector(7 * pi / 2,)
    else
        return SVector(pi / 4,)
    end
end

function initial_boundary_conditions(param, md)
    @unpack K, xf, yf, mapM, mapP, mapB = md

    Nc = get_num_components(param.equation)
    # Make periodic
    md = make_periodic(md)
    @unpack mapP = md

    mapI = []
    mapO = []
    inflowarr = []

    bcdata = BCData{Nc}(mapP, mapI, mapO, inflowarr)

    return bcdata
end

function initial_condition(param, x, y)
    # TODO: use getter
    t0 = param.timestepping_param.t0
    return SVector(exact_sol(param.equation, x, y, t0))
end

γ = 1.4
param = Param(N=3, K=(64, 64), xL=(-2.0, -2.0), xR=(2.0, 2.0),
    global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
    timestepping_param=TimesteppingParameter(T=1.0, CFL=0.5, dt0=1e-3, t0=0.0),
    limiting_param=LimitingParameter(ζ=0.1, η=1.0),
    postprocessing_param=PostprocessingParameter(output_interval=100),
    equation=KPP{Dim2}(),
    rhs_type=ESLimitedLowOrderPos(low_order_surface_flux_type=LaxFriedrichsOnProjectedVal(),
        high_order_surface_flux_type=LaxFriedrichsOnProjectedVal()),
    approximation_basis_type=LobattoCollocation(),
    entropyproj_limiter_type=NoEntropyProjectionLimiter(),
    rhs_limiter_type=SubcellLimiter(shockcapture_type=HennemannShockCapture()))

T = param.timestepping_param.T
N = param.N
K = param.K
equation = param.equation

rd, md, discrete_data, bcdata, prealloc, caches = initialize_DG(param, initial_condition, initial_boundary_conditions)

data_hist = SSP33!(param, discrete_data, bcdata, prealloc, caches)

err_data = calculate_error(prealloc.Uq, param, discrete_data, md, prealloc, exact_sol)

construct_vtk_file!(caches.postprocessing_cache, param, data_hist, "/data/yl184/outputs/figures/KPP", "KPP")
