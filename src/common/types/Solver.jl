abstract type RHS end
Base.@kwdef struct LowOrderPositivity{SURFACEFLUXTYPE} <: RHS
    surface_flux::SURFACEFLUXTYPE
end
struct FluxDiffRHS{VOLUMEFLUXTYPE,SURFACEFLUXTYPE} <: RHS
    volume_flux::VOLUMEFLUXTYPE
    surface_flux::SURFACEFLUXTYPE
end
struct LimitedDG{LOWSURFACEFLUXTYPE,HIGHSURFACEFLUXTYPE,HIGHVOLUMEFLUXTYPE} <: RHS
    low_order_surface_flux::LOWSURFACEFLUXTYPE
    high_order_surface_flux::HIGHSURFACEFLUXTYPE
    high_order_volume_flux::HIGHVOLUMEFLUXTYPE
end

abstract type VolumeFluxType end
struct ChandrashekarFlux <: VolumeFluxType end
struct CentralFlux <: VolumeFluxType end

abstract type SurfaceFluxType end
struct ChandrashekarOnProjectedVal <: SurfaceFluxType end
struct LaxFriedrichsOnNodalVal <: SurfaceFluxType end
struct LaxFriedrichsOnProjectedVal <: SurfaceFluxType end

const EntropyStable{SURFACEFLUXTYPE} = FluxDiffRHS{ChandrashekarFlux,SURFACEFLUXTYPE}
const StandardDG = FluxDiffRHS{CentralFlux,LaxFriedrichsOnProjectedVal}
const ESLimitedLowOrderPos{LOWSURFACEFLUXTYPE,HIGHSURFACEFLUXTYPE} = LimitedDG{LOWSURFACEFLUXTYPE,HIGHSURFACEFLUXTYPE,ChandrashekarFlux}
const StdDGLimitedLowOrderPos{LOWSURFACEFLUXTYPE} = LimitedDG{LOWSURFACEFLUXTYPE,LaxFriedrichsOnProjectedVal,CentralFlux}

EntropyStable(; surface_flux=LaxFriedrichsOnProjectedVal()) =
    FluxDiffRHS(ChandrashekarFlux(), surface_flux)
ESLimitedLowOrderPos(; low_order_surface_flux=LaxFriedrichsOnNodalVal(),
    high_order_surface_flux=LaxFriedrichsOnProjectedVal()) =
    LimitedDG(low_order_surface_flux, high_order_surface_flux, ChandrashekarFlux())
StdDGLimitedLowOrderPos(; low_order_surface_flux=LaxFriedrichsOnNodalVal(),
    high_order_surface_flux=LaxFriedrichsOnProjectedVal()) =
    LimitedDG(low_order_surface_flux, high_order_surface_flux, CentralFlux())

abstract type EntropyProjectionLimiterType end
abstract type ScaledExtrapolation <: EntropyProjectionLimiterType end
struct NoEntropyProjectionLimiter <: EntropyProjectionLimiterType end
struct NodewiseScaledExtrapolation <: ScaledExtrapolation end

abstract type RHSLimiterType end
struct NoRHSLimiter <: RHSLimiterType end

# TODO: It should depend on Equation type... Hardcode for CompressibleIdealGas for now
abstract type LimiterBoundType end
struct PositivityBound <: LimiterBoundType end
struct PositivityAndMinEntropyBound <: LimiterBoundType end
struct PositivityAndRelaxedMinEntropyBound <: LimiterBoundType end
struct PositivityAndCellEntropyBound <: LimiterBoundType end
Base.@kwdef struct PositivityAndRelaxedCellEntropyBound <: LimiterBoundType
    beta::Float64
end

# TODO: refactor...
struct TVDBound <: LimiterBoundType end
struct TVDAndMinEntropyBound <: LimiterBoundType end
struct TVDAndRelaxedMinEntropyBound <: LimiterBoundType end
struct TVDAndCellEntropyBound <: LimiterBoundType end
Base.@kwdef struct TVDAndRelaxedCellEntropyBound <: LimiterBoundType
    beta::Float64
end

abstract type ShockCaptureType end
struct NoShockCapture <: ShockCaptureType end
# Equation (41) on https://www.sciencedirect.com/science/article/pii/S0021999120307099
struct HennemannShockCapture <: ShockCaptureType
    a::Float64
    c::Float64
end

# Equation (42) on https://www.sciencedirect.com/science/article/pii/S0021999120307099
HennemannShockCapture(; a=0.5, c=1.8) = HennemannShockCapture(a, c)

struct ZhangShuLimiter{SHOCKCAPTURETYPE<:ShockCaptureType} <: RHSLimiterType
    shockcapture::SHOCKCAPTURETYPE
end

struct SubcellLimiter{BOUNDTYPE<:LimiterBoundType,SHOCKCAPTURETYPE<:ShockCaptureType} <: RHSLimiterType
    bound::BOUNDTYPE
    shockcapture::SHOCKCAPTURETYPE
end

ZhangShuLimiter(; shockcapture=NoShockCapture()) = ZhangShuLimiter(shockcapture)
SubcellLimiter(; bound=PositivityBound(),
    shockcapture=NoShockCapture()) = SubcellLimiter(bound, shockcapture)

abstract type ApproxBasisType end
struct GaussCollocation <: ApproxBasisType end
struct LobattoCollocation <: ApproxBasisType end

abstract type QuadratureType end
struct GaussQuadrature <: QuadratureType end
struct LobattoQuadrature <: QuadratureType end

abstract type Dimension end
struct Dim1 <: Dimension end
struct Dim2 <: Dimension end

abstract type EquationType{DIM<:Dimension} end
abstract type CompressibleFlow{DIM} <: EquationType{DIM} end
abstract type CompressibleParam end
Base.@kwdef struct CompressibleEulerParam <: CompressibleParam
    # TODO: fill in
end
Base.@kwdef struct CompressibleNavierStokesParam <: CompressibleParam
    # TODO: fill in
end
Base.@kwdef struct CompressibleIdealGas{DIM,PARAM<:CompressibleParam} <: CompressibleFlow{DIM}
    param::PARAM
    gamma::Float64
end

const CompressibleEulerIdealGas{DIM} = CompressibleIdealGas{DIM,CompressibleEulerParam}
const CompressibleNavierStokesIdealGas{DIM} = CompressibleIdealGas{DIM,CompressibleNavierStokesParam}

CompressibleEulerIdealGas{DIM}(gamma) where {DIM} = CompressibleIdealGas{DIM,CompressibleEulerParam}(gamma=gamma, param=CompressibleEulerParam())
CompressibleNavierStokesIdealGas{DIM}(gamma) where {DIM} = CompressibleIdealGas{DIM,CompressibleNavierStokesParam}(gamma=gamma, param=CompressibleNavierStokesParam())

struct KPP{DIM} <: EquationType{DIM} end

Base.@kwdef struct GlobalConstant
    POSTOL::Float64    # Tolerance for positivity
    ZEROTOL::Float64    # Tolerance for zero
end

Base.@kwdef struct TimesteppingParameter
    T::Float64         # End time
    CFL::Float64         # CFL number
    dt0::Float64         # initial time step size
    t0::Float64         # initial time
end

Base.@kwdef struct PostprocessingParameter
    output_interval::Int64      # interval of saving snapshots 
end

Base.@kwdef struct PostprocessingCache{Nc}
    xp::Array{Float64,2}
    yp::Array{Float64,2}
    Up::Array{SVector{Nc,Float64},2}
end

# TODO: put parameters into limiter
Base.@kwdef struct LimitingParameter
    zeta::Float64           # Positiivty relaxation parameter ρ,ρe >= eta min
    eta::Float64           # Bound relaxation parameter ρ,ρe ∈ [(1-zeta)min, (1+zeta)max]
end

Base.@kwdef struct Param{KTYPE,XL,XR,EQUATIONTYPE,APPROXBASISTYPE,RHS,ENTROPYPROJECTIONLIMITERTYPE,RHSLIMITERTYPE}
    N::Int64
    K::KTYPE      # Number of elements in 1D.
    # In 2D, it is a tuple (Kx,Ky), number of elements along
    # each direction

    xL::XL        # In 1D, the domain is [xL,xR]
    xR::XR        # In 2D, the domain is [xL[1],xR[1]] x [xL[2],xR[2]]

    global_constants::GlobalConstant
    timestepping_param::TimesteppingParameter
    limiting_param::LimitingParameter
    postprocessing_param::PostprocessingParameter

    equation::EQUATIONTYPE
    approximation_basis::APPROXBASISTYPE
    rhs::RHS
    entropyproj_limiter::ENTROPYPROJECTIONLIMITERTYPE
    rhs_limiter::RHSLIMITERTYPE
end

# TODO: define iterator to loop instead of size
struct SizeData
    K::Int64
    N1D::Int64
    Nd::Int64
    Nc::Int64
    Np::Int64
    Nq::Int64
    Nfp::Int64
    Nh::Int64
    Ns::Int64    # number of stages in time integrator
end

struct GeomData{NGEO}
    J::Array{Float64,2}
    Jq::Array{Float64,2}
    GJh::NTuple{NGEO,Array{Float64,2}}   # rxJh in 1D
    # rxJh, sxJh, ryJh, syJh in 2D
end

struct Operators{DIM}
    Srsh_db::NTuple{DIM,Array{Float64,2}}
    Srs0::NTuple{DIM,SparseMatrixCSC{Float64,Int64}}
    Srsh_nnz::Array{Tuple{Int64,Int64},1}
    Srs0_nnz::Array{Tuple{Int64,Int64},1}
    Brs::NTuple{DIM,Array{Float64,2}}    # TODO: to sparse matrix
    Vh::Array{Float64,2}
    MinvVhT::Array{Float64,2}
    VDM_inv::Array{Float64,2}
    VDMinvPq::Array{Float64,2}
    VqVDM::Array{Float64,2}
    VhPq::Array{Float64,2}
    Vq::Array{Float64,2}
    Vf::Array{Float64,2}
    Vf_low::Array{Float64,2}
    Pq::Array{Float64,2}
    MinvVfT::Array{Float64,2}
    wq::Array{Float64,1}
    q2fq::Array{Array{Int64,1},1}
    fq2q::Array{Int64,1}
end

struct DiscretizationData{DIM,NGEO}
    sizes::SizeData
    geom::GeomData{NGEO}
    ops::Operators{DIM}
end

Base.@kwdef struct Solver{PARAM,REFELEMDATA,MESHDATA,DISCRETIZATIONDATA}
    param::PARAM
    rd::REFELEMDATA
    md::MESHDATA
    discrete_data::DISCRETIZATIONDATA
end

