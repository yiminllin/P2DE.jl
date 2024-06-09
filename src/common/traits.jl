function equation(solver::Solver)
    return solver.param.equation
end

function limiter(solver::Solver)
    return solver.param.rhs_limiter_type
end

function rhs(solver::Solver)
    return solver.param.rhs_type
end

function basis(solver::Solver)
    return solver.param.approximation_basis_type
end

function shockcapture_type(solver::Solver)
    return shockcapture_type(limiter(solver))
end

function entropyproj_limiter(solver::Solver)
    return solver.param.entropyproj_limiter_type
end

function low_order_surface_flux_type(rhs_type::LowOrderPositivity)
    return rhs_type.surface_flux_type
end

function low_order_surface_flux_type(rhs_type::LimitedDG)
    return rhs_type.low_order_surface_flux_type
end

function low_order_surface_flux_type(solver::Solver)
    return low_order_surface_flux_type(rhs(solver))
end

function high_order_surface_flux_type(rhs_type::FluxDiffRHS)
    return rhs_type.surface_flux_type
end

function high_order_surface_flux_type(rhs_type::LimitedDG)
    return rhs_type.high_order_surface_flux_type
end

function high_order_surface_flux_type(solver::Solver)
    return high_order_surface_flux_type(rhs(solver))
end

function high_order_volume_flux_type(rhs_type::FluxDiffRHS)
    return rhs_type.volume_flux_type
end

function high_order_volume_flux_type(rhs_type::LimitedDG)
    return rhs_type.high_order_volume_flux_type
end

function high_order_volume_flux_type(solver::Solver)
    return high_order_volume_flux_type(rhs(solver))
end

function bound_type(limiter::ZhangShuLimiter)
    return PositivityBound()
end

function bound_type(limiter::SubcellLimiter)
    return limiter.bound_type
end

function bound_type(solver::Solver)
    return bound_type(limiter(solver))
end

function shockcapture_type(limiter::NoRHSLimiter)
    return NoShockCapture()
end

function shockcapture_type(limiter::Union{ZhangShuLimiter,SubcellLimiter})
    return limiter.shockcapture_type
end

function low_order_cache(rhs_cache::LowOrderPositivityCache)
    return rhs_cache
end

function low_order_cache(rhs_cache::LimitedDGCache)
    return rhs_cache.cacheL
end

function high_order_cache(rhs_cache::FluxDiffCache)
    return rhs_cache
end

function high_order_cache(rhs_cache::LimitedDGCache)
    return rhs_cache.cacheH
end

function rhs_cache(state::State)
    return state.cache.rhs_cache
end

function high_order_cache(state::State)
    return high_order_cache(rhs_cache(state))
end

function low_order_cache(state::State)
    return low_order_cache(rhs_cache(state))
end

function get_γ(equation::CompressibleIdealGas)
    return equation.γ
end

function get_γ(solver::Solver)
    return get_γ(equation(solver))
end

function to_equation_1D(equation::CompressibleIdealGas{Dim2})
    return CompressibleEulerIdealGas{Dim1}(get_γ(equation))
end

function dim(equation::EquationType{Dim1})
    return 1
end

function dim(equation::EquationType{Dim2})
    return 2
end

function dim(solver::Solver)
    return dim(equation(solver))
end

function dim_type(equation::EquationType{Dim1})
    return Dim1()
end

function dim_type(equation::EquationType{Dim2})
    return Dim2()
end

function dim_type(solver::Solver)
    return dim_type(equation(solver))
end

function num_components(equation::CompressibleFlow{Dim1})
    return 3
end

function num_components(equation::CompressibleFlow{Dim2})
    return 4
end

function num_components(equation::KPP{Dim2})
    return 1
end

function num_components(solver::Solver)
    return num_components(equation(solver))
end

# TODO: refactor
function num_elements(param)
    return num_elements(param, param.equation)
end

function num_elements(param, equation::EquationType{Dim1})
    return param.K
end

# TODO: hardcoded for uniform mesh
function num_elements(param, equation::EquationType{Dim2})
    return param.K[1] * param.K[2]
end

function num_elements(solver::Solver)
    return num_elements(solver.param)
end

