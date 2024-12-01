function Base.show(io::IO, ::MIME"text/plain", param::Param)
    @nospecialize param
    println(io, "N=$(param.N),K=$(param.K),eta=$(param.limiting_param.eta),zeta=$(param.limiting_param.zeta)")
    println(io, "basis                      type: ", param.approximation_basis)
    println(io, "rhs                        type: ", param.rhs)
    println(io, "entropy projection limiter type: ", param.entropyproj_limiter)
    println(io, "rhs limiter                type: ", param.rhs_limiter)
end

function Base.show(io::IO, rhs::ESLimitedLowOrderPos)
    print(io, "ESLimitedLowOrderPos(FBL=", low_order_surface_flux(rhs), ",FBH=", high_order_surface_flux(rhs), ")")
end

function Base.show(io::IO, rhs::StdDGLimitedLowOrderPos)
    print(io, "StdDGLimitedLowOrderPos(FBL=", low_order_surface_flux(rhs), ")")
end

function Base.show(io::IO, rhs::LowOrderPositivity)
    print(io, "LowOrderPositivity(FBL=", low_order_surface_flux(rhs), ")")
end

function Base.show(io::IO, rhs::EntropyStable)
    print(io, "EntropyStable(FBL=", high_order_surface_flux(rhs), ")")
end

function Base.show(io::IO, rhs::StandardDG)
    print(io, "StandardDG()")
end

function Base.show(io::IO, flux::LaxFriedrichsOnNodalVal)
    print(io, "LFNodal")
end

function Base.show(io::IO, flux::LaxFriedrichsOnProjectedVal)
    print(io, "LFProjected")
end

function Base.show(io::IO, flux::ChandrashekarOnProjectedVal)
    print(io, "ECProjected")
end

function Base.show(io::IO, entropyproj_limiter::NoEntropyProjectionLimiter)
    print(io, "None")
end

function Base.show(io::IO, entropyproj_limiter::NodewiseScaledExtrapolation)
    print(io, "NodeOpBlend")
end

function Base.show(io::IO, bound::PositivityBound)
    print(io, "PosBound")
end

function Base.show(io::IO, bound::PositivityAndMinEntropyBound)
    print(io, "PosMinEntropyBound")
end

function Base.show(io::IO, bound::PositivityAndRelaxedMinEntropyBound)
    print(io, "PosRelaxMinEntropyBound")
end

function Base.show(io::IO, bound::PositivityAndCellEntropyBound)
    print(io, "PosCellEntropyBound")
end

function Base.show(io::IO, bound::PositivityAndRelaxedCellEntropyBound)
    print(io, "PosRelaxCellEntropyBound(beta=", bound.beta, ")")
end

function Base.show(io::IO, bound::TVDBound)
    print(io, "TVDBound")
end

function Base.show(io::IO, bound::TVDAndMinEntropyBound)
    print(io, "TVDMinEntropyBound")
end

function Base.show(io::IO, bound::TVDAndRelaxedMinEntropyBound)
    print(io, "TVDRelaxMinEntropyBound")
end

function Base.show(io::IO, bound::TVDAndCellEntropyBound)
    print(io, "TVDCellEntropyBound")
end

function Base.show(io::IO, bound::TVDAndRelaxedCellEntropyBound)
    print(io, "TVDRelaxCellEntropyBound(beta=", bound.beta, ")")
end

function Base.show(io::IO, shockcapture::NoShockCapture)
    print(io, "None")
end

function Base.show(io::IO, shockcapture::HennemannShockCapture)
    print(io, "Modal")
end

function Base.show(io::IO, limiter::SubcellLimiter)
    print(io, "Subcell(bound=", bound(limiter), ",shockcapture=", shockcapture(limiter), ")")
end

function Base.show(io::IO, limiter::ZhangShuLimiter)
    print(io, "ZhangShu(bound=", bound(limiter), ",shockcapture=", shockcapture(limiter), ")")
end


