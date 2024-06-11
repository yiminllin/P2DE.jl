function equation(solver::Solver)
    return solver.param.equation
end

function limiter(solver::Solver)
    return solver.param.rhs_limiter
end

function rhs(solver::Solver)
    return solver.param.rhs
end

function basis(solver::Solver)
    return solver.param.approximation_basis
end

function shockcapture(solver::Solver)
    return shockcapture(limiter(solver))
end

function entropyproj_limiter(solver::Solver)
    return solver.param.entropyproj_limiter
end

function low_order_surface_flux(rhs::LowOrderPositivity)
    return rhs.surface_flux
end

function low_order_surface_flux(rhs::LimitedDG)
    return rhs.low_order_surface_flux
end

function low_order_surface_flux(solver::Solver)
    return low_order_surface_flux(rhs(solver))
end

function high_order_surface_flux(rhs::FluxDiffRHS)
    return rhs.surface_flux
end

function high_order_surface_flux(rhs::LimitedDG)
    return rhs.high_order_surface_flux
end

function high_order_surface_flux(solver::Solver)
    return high_order_surface_flux(rhs(solver))
end

function high_order_volume_flux(rhs::FluxDiffRHS)
    return rhs.volume_flux
end

function high_order_volume_flux(rhs::LimitedDG)
    return rhs.high_order_volume_flux
end

function high_order_volume_flux(solver::Solver)
    return high_order_volume_flux(rhs(solver))
end

function bound(limiter::ZhangShuLimiter)
    return PositivityBound()
end

function bound(limiter::SubcellLimiter)
    return limiter.bound
end

function bound(solver::Solver)
    return bound(limiter(solver))
end

function shockcapture(limiter::NoRHSLimiter)
    return NoShockCapture()
end

function shockcapture(limiter::Union{ZhangShuLimiter,SubcellLimiter})
    return limiter.shockcapture
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

function get_gamma(equation::CompressibleIdealGas)
    return equation.gamma
end

function get_gamma(solver::Solver)
    return get_gamma(equation(solver))
end

function to_equation_1D(equation::CompressibleIdealGas{Dim2})
    return CompressibleEulerIdealGas{Dim1}(get_gamma(equation))
end

function Ndim(equation::EquationType{Dim1})
    return 1
end

function Ndim(equation::EquationType{Dim2})
    return 2
end

function Ndim(solver::Solver)
    return dim(equation(solver))
end

function dim(equation::EquationType{Dim1})
    return Dim1()
end

function dim(equation::EquationType{Dim2})
    return Dim2()
end

function dim(solver::Solver)
    return dim(equation(solver))
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

