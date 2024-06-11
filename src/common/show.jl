function Base.show(io::IO, ::MIME"text/plain", param::Param)
    @nospecialize param
    println(io, "N=$(param.N),K=$(param.K),eta=$(param.limiting_param.eta),zeta=$(param.limiting_param.zeta)")
    println(io, "basis                      type: ", param.approximation_basis_type)
    println(io, "rhs                        type: ", param.rhs)
    println(io, "entropy projection limiter type: ", param.entropyproj_limiter_type)
    println(io, "rhs limiter                type: ", param.rhs_limiter_type)
end

function Base.show(io::IO, rhs::ESLimitedLowOrderPos)
    print(io, "ESLimitedLowOrderPos(FBL=", low_order_surface_flux_type(rhs), ",FBH=", high_order_surface_flux_type(rhs), ")")
end

function Base.show(io::IO, rhs::StdDGLimitedLowOrderPos)
    print(io, "StdDGLimitedLowOrderPos(FBL=", low_order_surface_flux_type(rhs), ")")
end

function Base.show(io::IO, rhs::LowOrderPositivity)
    print(io, "LowOrderPositivity(FBL=", low_order_surface_flux_type(rhs), ")")
end

function Base.show(io::IO, rhs::EntropyStable)
    print(io, "EntropyStable(FBL=", high_order_surface_flux_type(rhs), ")")
end

function Base.show(io::IO, rhs::StandardDG)
    print(io, "StandardDG()")
end

function Base.show(io::IO, flux_type::LaxFriedrichsOnNodalVal)
    print(io, "LFNodal")
end

function Base.show(io::IO, flux_type::LaxFriedrichsOnProjectedVal)
    print(io, "LFProjected")
end

function Base.show(io::IO, flux_type::ChandrashekarOnProjectedVal)
    print(io, "ECProjected")
end

function Base.show(io::IO, entropyproj_limiter_type::NoEntropyProjectionLimiter)
    print(io, "None")
end

function Base.show(io::IO, entropyproj_limiter_type::NodewiseScaledExtrapolation)
    print(io, "NodeOpBlend")
end

function Base.show(io::IO, bound_type::PositivityBound)
    print(io, "PosBound")
end

function Base.show(io::IO, bound_type::PositivityAndMinEntropyBound)
    print(io, "PosMinEntropyBound")
end

function Base.show(io::IO, bound_type::PositivityAndRelaxedMinEntropyBound)
    print(io, "PosRelaxMinEntropyBound")
end

function Base.show(io::IO, bound_type::PositivityAndCellEntropyBound)
    print(io, "PosCellEntropyBound")
end

function Base.show(io::IO, bound_type::PositivityAndRelaxedCellEntropyBound)
    print(io, "PosRelaxCellEntropyBound(beta=", bound_type.beta, ")")
end

function Base.show(io::IO, bound_type::TVDBound)
    print(io, "TVDBound")
end

function Base.show(io::IO, bound_type::TVDAndMinEntropyBound)
    print(io, "TVDMinEntropyBound")
end

function Base.show(io::IO, bound_type::TVDAndRelaxedMinEntropyBound)
    print(io, "TVDRelaxMinEntropyBound")
end

function Base.show(io::IO, bound_type::TVDAndCellEntropyBound)
    print(io, "TVDCellEntropyBound")
end

function Base.show(io::IO, bound_type::TVDAndRelaxedCellEntropyBound)
    print(io, "TVDRelaxCellEntropyBound(beta=", bound_type.beta, ")")
end

function Base.show(io::IO, shockcapture_type::NoShockCapture)
    print(io, "None")
end

function Base.show(io::IO, shockcapture_type::HennemannShockCapture)
    print(io, "Modal")
end

function Base.show(io::IO, limiter_type::SubcellLimiter)
    print(io, "Subcell(bound=", bound_type(limiter_type), ",shockcapture=", shockcapture_type(limiter_type), ")")
end

function Base.show(io::IO, limiter_type::ZhangShuLimiter)
    print(io, "ZhangShu(bound=", bound_type(limiter_type), ",shockcapture=", shockcapture_type(limiter_type), ")")
end


