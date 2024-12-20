using Revise
using StaticArrays
using DataFrames
using MAT

using P2DE

function exact_sol(eqn, x, t)
    # TODO: not exact solution
    rho = (x < -4.0) ? 3.857143 : 1 + 0.2 * sin(5 * x)
    u = (x < -4.0) ? 2.629369 : 0.0
    p = (x < -4.0) ? 10.3333 : 1.0
    return rho, u, p
end

function initial_boundary_conditions(param, md)
    (; K, equation) = param
    (; mapP) = md

    # Shu-osher
    mapI = [1]
    mapO = [2 * K]

    rho = 3.857143
    u = 2.629369
    p = 10.3333
    inflowarr = [primitive_to_conservative(equation, (rho, u, p))]

    bcdata = BCData(mapP, mapI, mapO, inflowarr)

    return bcdata
end

function initial_condition(param, x)
    # TODO: use getter
    t0 = param.timestepping_param.t0
    return primitive_to_conservative(param.equation, SVector{3,Float64}(exact_sol(param.equation, x, t0)))
end

jld_path = "/data/yl184/outputs/jld2/shu-soher.jld2"

gamma = 1.4
param = Param(N=3, K=64, xL=-5.0, xR=5.0,
    global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
    timestepping_param=TimesteppingParameter(T=1.8, CFL=0.5, dt0=1e-4, t0=0.0),
    limiting_param=LimitingParameter(zeta=0.1, eta=1.0),
    postprocessing_param=PostprocessingParameter(output_interval=1000),
    equation=CompressibleEulerIdealGas{Dim1}(gamma),
    rhs=ESLimitedLowOrderPos(low_order_surface_flux=LaxFriedrichsOnNodalVal(),
        high_order_surface_flux=LaxFriedrichsOnProjectedVal()),
    approximation_basis=GaussCollocation(),
    entropyproj_limiter=NodewiseScaledExtrapolation(),
    rhs_limiter=SubcellLimiter(bound=PositivityBound(),
        shockcapture=NoShockCapture()))

T = param.timestepping_param.T
N = param.N
K = param.K
equation = param.equation

rd, md, discrete_data, bcdata, prealloc, caches = initialize_DG(param, initial_condition, initial_boundary_conditions)

data_hist = SSP33!(param, discrete_data, bcdata, prealloc, caches)

err_data = calculate_error(prealloc.Uq, param, discrete_data, md, prealloc, exact_sol)

plot_path = "/data/yl184/outputs/figures/shu-osher/N=$N,K=$K,rhs=$(param.rhs),vproj=$(param.entropyproj_limiter),lim=$(param.rhs_limiter),ZETA=$(param.limiting_param.zeta),ETA=$(param.limiting_param.eta).png"
gif_path = "/data/yl184/outputs/figures/shu-osher/N=$N,K=$K,rhs=$(param.rhs),vproj=$(param.entropyproj_limiter),lim=$(param.rhs_limiter),ZETA=$(param.limiting_param.zeta),ETA=$(param.limiting_param.eta),zoom.png"

weno_sol = matread("data/weno5_shuosher.mat")
plot_component(param, discrete_data, md, prealloc,
    [u[1] for u in prealloc.Uq], 1, K, 0, 6,
    plot_path,
    true, weno_sol["x"], weno_sol["rho"], 1, size(weno_sol["x"], 2))

plot_rho_animation(md, param, prealloc, discrete_data, data_hist, data_hist.thetahist, 0, 6,
    gif_path)

df = DataFrame([name => [] for name in (fieldnames(Param)..., fieldnames(ErrorData)..., :data_history)])
write_to_jld2(param, data_hist, err_data, df, jld_path)
