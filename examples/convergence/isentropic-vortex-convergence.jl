using Revise
using StaticArrays
using DataFrames
using JLD2
using StartUpDG

using P2DE

function exact_sol(eqn, x, y, t)
    γ = get_γ(eqn)
    x0 = 4.5
    y0 = 5.0
    beta = 8.5
    r2 = @. (x - x0 - t)^2 + (y - y0)^2

    u = @. 1 - beta * exp(1 - r2) * (y - y0) / (2 * pi)
    v = @. beta * exp(1 - r2) * (x - x0 - t) / (2 * pi)
    rho = @. 1 - (1 / (8 * γ * pi^2)) * (γ - 1) / 2 * (beta * exp(1 - r2))^2
    rho = @. rho^(1 / (γ - 1))
    p = @. rho^γ

    return (rho, u, v, p)
end

function initial_boundary_conditions(param, md)
    (; K, xf, yf, mapM, mapP, mapB) = md

    Nc = get_num_components(param.equation)
    # Make periodic
    md = make_periodic(md)
    (; mapP) = md

    mapI = []
    Ival = []
    mapO = []

    bcdata = BCData{Nc}(mapP, mapI, mapO, Ival)
    return bcdata
end

function initial_condition(param, x, y)
    # TODO: use getter
    t0 = param.timestepping_param.t0
    return primitive_to_conservative(param.equation, SVector(exact_sol(param.equation, x, y, t0)))
end

# TODO: refactor convergence
jld_path = "/data/yl184/outputs/jld2/isentropic-vortex.jld2"

for limiter_type in [(NoEntropyProjectionLimiter(), LaxFriedrichsOnNodalVal(), ZhangShuLimiter(), LobattoCollocation());
    (NoEntropyProjectionLimiter(), LaxFriedrichsOnNodalVal(), SubcellLimiter(bound_type=PositivityBound()), LobattoCollocation());
    (NoEntropyProjectionLimiter(), LaxFriedrichsOnNodalVal(), SubcellLimiter(bound_type=PositivityAndMinEntropyBound()), LobattoCollocation());
    (NodewiseScaledExtrapolation(), LaxFriedrichsOnNodalVal(), SubcellLimiter(bound_type=PositivityBound()), GaussCollocation());
    (NodewiseScaledExtrapolation(), LaxFriedrichsOnNodalVal(), SubcellLimiter(bound_type=PositivityAndMinEntropyBound()), GaussCollocation());
    (NodewiseScaledExtrapolation(), LaxFriedrichsOnProjectedVal(), ZhangShuLimiter(), GaussCollocation())]
    for N in [1; 2; 3; 4]
        for K in [(5, 5), (10, 10), (20, 20), (40, 40), (80, 80)]


            entropyproj_type, low_order_flux_type, rhs_lim_type, discretization_type = limiter_type
            γ = 1.4
            param = Param(N=N, K=K, xL=(0.0, 0.0), xR=(10.0, 10.0),
                global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
                timestepping_param=TimesteppingParameter(T=1.0, CFL=0.75, dt0=1e-2, t0=0.0),
                limiting_param=LimitingParameter(ζ=0.1, η=0.5),
                postprocessing_param=PostprocessingParameter(output_interval=100),
                equation=CompressibleEulerIdealGas{Dim2}(γ),
                rhs_type=ESLimitedLowOrderPos(low_order_surface_flux_type=low_order_flux_type,
                    high_order_surface_flux_type=LaxFriedrichsOnProjectedVal()),
                approximation_basis_type=discretization_type,
                entropyproj_limiter_type=entropyproj_type,
                rhs_limiter_type=rhs_lim_type)

            T = param.timestepping_param.T
            N = param.N
            K = param.K
            equation = param.equation

            rd, md, discrete_data, bcdata, prealloc, caches = initialize_DG(param, initial_condition, initial_boundary_conditions)

            data_hist = SSP33!(param, discrete_data, bcdata, prealloc, caches)

            err_data = calculate_error(prealloc.Uq, param, discrete_data, md, prealloc, exact_sol)

            df = DataFrame([name => [] for name in (fieldnames(Param)..., fieldnames(ErrorData)..., :data_history)])
            write_to_jld2(param, data_hist, err_data, df, jld_path)

        end
    end
end

df = load(jld_path, "data")
visualize_error_data(df)
