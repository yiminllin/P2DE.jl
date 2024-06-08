function low_order_surface_flux_type(rhs_type::LowOrderPositivity)
    return rhs_type.surface_flux_type
end

function low_order_surface_flux_type(rhs_type::LimitedDG)
    return rhs_type.low_order_surface_flux_type
end

function high_order_surface_flux_type(rhs_type::FluxDiffRHS)
    return rhs_type.surface_flux_type
end

function high_order_surface_flux_type(rhs_type::LimitedDG)
    return rhs_type.high_order_surface_flux_type
end

function high_order_volume_flux_type(rhs_type::FluxDiffRHS)
    return rhs_type.volume_flux_type
end

function high_order_volume_flux_type(rhs_type::LimitedDG)
    return rhs_type.high_order_volume_flux_type
end

function bound_type(limiter::ZhangShuLimiter)
    return PositivityBound()
end

function bound_type(limiter::SubcellLimiter)
    return limiter.bound_type
end

function shockcapture_type(limiter::NoRHSLimiter)
    return NoShockCapture()
end

function shockcapture_type(limiter::Union{ZhangShuLimiter,SubcellLimiter})
    return limiter.shockcapture_type
end

function get_γ(equation::CompressibleIdealGas)
    return equation.γ
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

function dim_type(equation::EquationType{Dim1})
    return Dim1()
end

function dim_type(equation::EquationType{Dim2})
    return Dim2()
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

function bound_type(param::Param)
    return bound_type(param.rhs_limiter_type)
end

function shockcapture_type(param::Param)
    return shockcapture_type(param.rhs_limiter_type)
end

