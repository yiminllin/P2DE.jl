using StaticArrays
using StartUpDG

using P2DE

function exact_sol(eqn, x, y, t)
    gamma = get_gamma(eqn)
    x0 = 4.5
    y0 = 5.0
    beta = 8.5
    r2 = @. (x - x0 - t)^2 + (y - y0)^2

    u = @. 1 - beta * exp(1 - r2) * (y - y0) / (2 * pi)
    v = @. beta * exp(1 - r2) * (x - x0 - t) / (2 * pi)
    rho = @. 1 - (1 / (8 * gamma * pi^2)) * (gamma - 1) / 2 * (beta * exp(1 - r2))^2
    rho = @. rho^(1 / (gamma - 1))
    p = @. rho^gamma

    return (rho, u, v, p)
end

function initial_boundary_conditions(param, md)
    (; K, xf, yf, mapM, mapP, mapB) = md

    Nc = num_components(param.equation)
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

for limiter_type in [
    (NoEntropyProjectionLimiter(), LaxFriedrichsOnNodalVal(), ZhangShuLimiter(), LobattoCollocation());
    (NoEntropyProjectionLimiter(), LaxFriedrichsOnNodalVal(), SubcellLimiter(bound_type=PositivityBound()), LobattoCollocation());
    (NoEntropyProjectionLimiter(), LaxFriedrichsOnNodalVal(), SubcellLimiter(bound_type=PositivityBound(), shockcapture_type=HennemannShockCapture()), LobattoCollocation());
    (NoEntropyProjectionLimiter(), LaxFriedrichsOnNodalVal(), SubcellLimiter(bound_type=PositivityAndMinEntropyBound()), LobattoCollocation());
    (NoEntropyProjectionLimiter(), LaxFriedrichsOnNodalVal(), SubcellLimiter(bound_type=PositivityAndRelaxedMinEntropyBound()), LobattoCollocation());
    (NoEntropyProjectionLimiter(), LaxFriedrichsOnNodalVal(), SubcellLimiter(bound_type=PositivityAndCellEntropyBound()), LobattoCollocation());
    (NoEntropyProjectionLimiter(), LaxFriedrichsOnNodalVal(), SubcellLimiter(bound_type=PositivityAndRelaxedCellEntropyBound(beta=0.5)), LobattoCollocation())
]
    for N in [1; 2; 3; 4]
        for K in [(5, 5)]
            entropyproj_type, low_order_flux_type, rhs_lim_type, discretization_type = limiter_type
            gamma = 1.4
            param = Param(N=N, K=K, xL=(0.0, 0.0), xR=(10.0, 10.0),
                global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
                timestepping_param=TimesteppingParameter(T=2e-2, CFL=1.0, dt0=1e-2, t0=0.0),
                limiting_param=LimitingParameter(zeta=0.1, eta=0.5),
                postprocessing_param=PostprocessingParameter(output_interval=10000),
                equation=CompressibleEulerIdealGas{Dim2}(gamma),
                rhs=ESLimitedLowOrderPos(low_order_surface_flux_type=low_order_flux_type,
                    high_order_surface_flux_type=LaxFriedrichsOnProjectedVal()),
                approximation_basis_type=discretization_type,
                entropyproj_limiter_type=entropyproj_type,
                rhs_limiter_type=rhs_lim_type)

            T = param.timestepping_param.T
            N = param.N
            K = param.K
            equation = param.equation

            solver, state, state_param = initialize_DG(param, initial_condition, initial_boundary_conditions)

            data_hist = SSP33!(state, solver, state_param)

            err_data = calculate_error(state, solver, exact_sol)
        end
    end
end

