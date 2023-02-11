# TODO: better display of types
abstract type RHSType end
Base.@kwdef struct LowOrderPositivity{SURFACEFLUXTYPE}   <: RHSType
    surface_flux_type::SURFACEFLUXTYPE
end
Base.@kwdef struct EntropyStable{SURFACEFLUXTYPE}        <: RHSType
    surface_flux_type::SURFACEFLUXTYPE
end
Base.@kwdef struct ESLimitedLowOrderPos{LOWSURFACEFLUXTYPE,HIGHSURFACEFLUXTYPE} <: RHSType
    low_order_surface_flux_type ::LOWSURFACEFLUXTYPE
    high_order_surface_flux_type::HIGHSURFACEFLUXTYPE
end

function get_low_order_surface_flux(rhs_type::LowOrderPositivity)
    return rhs_type.surface_flux_type
end

function get_low_order_surface_flux(rhs_type::ESLimitedLowOrderPos)
    return rhs_type.low_order_surface_flux_type
end

function get_high_order_surface_flux(rhs_type::EntropyStable)
    return rhs_type.surface_flux_type
end

function get_high_order_surface_flux(rhs_type::ESLimitedLowOrderPos)
    return rhs_type.high_order_surface_flux_type
end

abstract type SurfaceFluxType end
struct ChandrashekarOnProjectedVal <: SurfaceFluxType end
struct LaxFriedrichsOnNodalVal     <: SurfaceFluxType end
struct LaxFriedrichsOnProjectedVal <: SurfaceFluxType end

abstract type EntropyProjectionLimiterType end
abstract type AdaptiveFilter          <: EntropyProjectionLimiterType end
abstract type ScaledExtrapolation     <: EntropyProjectionLimiterType end
struct NoEntropyProjectionLimiter     <: EntropyProjectionLimiterType end
struct ExponentialFilter              <: AdaptiveFilter end
struct ZhangShuFilter                 <: AdaptiveFilter end
struct ElementwiseScaledExtrapolation <: ScaledExtrapolation end
struct NodewiseScaledExtrapolation    <: ScaledExtrapolation end

abstract type PositivityLimiterType end
struct NoPositivityLimiter <: PositivityLimiterType end
struct ZhangShuLimiter     <: PositivityLimiterType end
struct SubcellLimiter      <: PositivityLimiterType end

abstract type ApproxBasisType end
struct GaussCollocation   <: ApproxBasisType end
struct LobattoCollocation <: ApproxBasisType end
struct HybridGaussLGL     <: ApproxBasisType end

abstract type QuadratureType end
struct GaussQuadrature   <: QuadratureType end
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
    γ    ::Float64
end

const CompressibleEulerIdealGas{DIM}        = CompressibleIdealGas{DIM,CompressibleEulerParam}
const CompressibleNavierStokesIdealGas{DIM} = CompressibleIdealGas{DIM,CompressibleNavierStokesParam}

CompressibleEulerIdealGas{DIM}(γ) where DIM        = CompressibleIdealGas{DIM,CompressibleEulerParam}(γ=γ,param=CompressibleEulerParam())
CompressibleNavierStokesIdealGas{DIM}(γ) where DIM = CompressibleIdealGas{DIM,CompressibleNavierStokesParam}(γ=γ,param=CompressibleNavierStokesParam())

function get_γ(equation::CompressibleIdealGas)
    return equation.γ
end

function get_dim(equation::EquationType{Dim1})
    return 1
end

function get_dim(equation::EquationType{Dim2})
    return 2
end

function get_num_components(equation::EquationType{Dim1})
    return 3
end

function get_num_components(equation::EquationType{Dim2})
    return 4
end

# TODO: iterator on Gauss elements (for k = 1:K  if (!LGLind[k])) end end)

Base.@kwdef struct GlobalConstant
    POSTOL ::Float64    # Tolerance for positivity
    ZEROTOL::Float64    # Tolerance for zero
end

Base.@kwdef struct TimesteppingParameter
    T  ::Float64         # End time
    CFL::Float64         # CFL number
    dt0::Float64         # initial time step size
    t0 ::Float64         # initial time
end

Base.@kwdef struct PostprocessingParameter
    output_interval::Int64      # interval of saving snapshots 
end

# TODO: put parameters into limiter
Base.@kwdef struct LimitingParameter
    ζ::Float64           # Positiivty relaxation parameter ρ,ρe >= η min
    η::Float64           # Bound relaxation parameter ρ,ρe ∈ [(1-ζ)min, (1+ζ)max]
end

Base.@kwdef struct Param{KTYPE,XL,XR,EQUATIONTYPE,APPROXBASISTYPE,RHSTYPE,ENTROPYPROJECTIONLIMITERTYPE,POSITIVITYLIMITERTYPE}
    N::Int64
    K::KTYPE      # Number of elements in 1D.
                  # In 2D, it is a tuple (Kx,Ky), number of elements along
                  # each direction

    xL::XL        # In 1D, the domain is [xL,xR]
    xR::XR        # In 2D, the domain is [xL[1],xR[1]] x [xL[2],xR[2]]

    global_constants    ::GlobalConstant
    timestepping_param  ::TimesteppingParameter
    limiting_param      ::LimitingParameter
    postprocessing_param::PostprocessingParameter

    equation                ::EQUATIONTYPE
    approximation_basis_type::APPROXBASISTYPE
    rhs_type                ::RHSTYPE
    entropyproj_limiter_type::ENTROPYPROJECTIONLIMITERTYPE
    positivity_limiter_type ::POSITIVITYLIMITERTYPE
end

# TODO: tuple
mutable struct BCData
    mapP::Array{Int64,2}
    mapI::Array{Int64,1}
    mapO::Array{Int64,1}  # List of global indices with inflow and outflow 
                          # boundary conditions
    inflowarr::Array{SVector{3,Float64},1}
end

# TODO: tuple
mutable struct GeomData
    J   ::Array{Float64,2}
    Jq  ::Array{Float64,2}
    rxJh::Array{Float64,2}
end

mutable struct Operators{DIM}
    Srsh_db    ::NTuple{DIM,Array{Float64,2}}
    Srs0       ::NTuple{DIM,SparseMatrixCSC{Float64,Int64}}
    Brs        ::NTuple{DIM,Array{Float64,2}}    # TODO: to sparse matrix
    Vh         ::Array{Float64,2}
    MinvVhT    ::Array{Float64,2}
    VDM_inv    ::Array{Float64,2}
    VDMinvPq   ::Array{Float64,2}
    VqVDM      ::Array{Float64,2}
    VhPq       ::Array{Float64,2}
    Vq         ::Array{Float64,2}
    Vf         ::Union{Array{Float64,2},SparseMatrixCSC{Float64,Int64}}   # TODO: hardcoded...
    Vf_low     ::SparseMatrixCSC{Float64,Int64}
    Pq         ::Array{Float64,2}
    LIFT       ::Union{Array{Float64,2},SparseMatrixCSC{Float64,Int64}}
    wq         ::Array{Float64,1}
end

mutable struct TransferOperators
    T_g2l::Array{Float64,2}
    T_l2g::Array{Float64,2}
end

# TODO: define iterator to loop instead of size
mutable struct SizeData
    Nc ::Int64
    Np ::Int64
    Nq ::Int64
    Nfp::Int64
    Nh ::Int64
    Ns ::Int64    # number of stages in time integrator
end

mutable struct DiscretizationData
    sizes::SizeData
    geom ::GeomData
    ops  ::Operators
end

mutable struct Preallocation{Nc}
    Uq     ::Array{SVector{Nc,Float64},2}
    vq     ::Array{SVector{Nc,Float64},2}       # entropy variables at quad points
    v_tilde::Array{SVector{Nc,Float64},2}       # projected entropy variables
    u_tilde::Array{SVector{Nc,Float64},2}       # entropy projected conservative variables
    beta   ::Array{Float64,2}
    rholog ::Array{Float64,2}
    betalog::Array{Float64,2}
    lam    ::Array{Float64,2}
    LFc    ::Array{Float64,2}
    rhsH   ::Array{SVector{Nc,Float64},2}
    Ui     ::Array{Float64,1}
    Uj     ::Array{Float64,1}
    QF1    ::Array{SVector{Nc,Float64},2}
    BF1    ::Array{SVector{Nc,Float64},2}
    uP      ::Array{SVector{Nc,Float64},2}
    betaP   ::Array{Float64,2}
    rhologP ::Array{Float64,2}
    betalogP::Array{Float64,2}
    flux     ::Array{SVector{Nc,Float64},2}
    flux_H   ::Array{SVector{Nc,Float64},2}
    flux_L   ::Array{SVector{Nc,Float64},2}
    wavespeed::Array{Float64,2}
    alphaarr ::Array{Float64,2}
    rhsL     ::Array{SVector{Nc,Float64},2}
    Larr     ::Array{Float64,2}
    L_local_arr::Array{Float64,3}
    rhsU     ::Array{SVector{Nc,Float64},2}
    v3tilde  ::Array{Float64,1}
    rhotilde ::Array{Float64,1}
    rhoetilde::Array{Float64,1}
    vq_k     ::Array{SVector{Nc,Float64},1}
    v_tilde_k::Array{SVector{Nc,Float64},1}
    u_tilde_k::Array{SVector{Nc,Float64},1}
    U_modal  ::Array{SVector{Nc,Float64},2}
    U_k      ::Array{SVector{Nc,Float64},1}
    Uq_k     ::Array{SVector{Nc,Float64},1}
    spatial  ::Array{SVector{Nc,Float64},2}
    boundary ::Array{SVector{Nc,Float64},2}
    resW     ::Array{SVector{Nc,Float64},2}
    resZ     ::Array{SVector{Nc,Float64},2}
    Farr     ::Array{Float64,2}
    θ_local_arr::Array{Float64,3}
    αarr     ::Array{Float64,2}
    LGLind   ::BitArray
    L_G2L_arr::Array{Float64,2}
    L_L2G_arr::Array{Float64,2}
    L_Vf_arr ::Array{Float64,2}
    VhPq_new ::Array{Float64,2}
    Vf_new   ::Array{Float64,2}
    VhT_new    ::Array{Float64,2}
    MinvVhT_new::Array{Float64,2}
    uL_k     ::Array{SVector{Nc,Float64},1}
    P_k      ::Array{SVector{Nc,Float64},1}
    f_bar_H  ::Array{SVector{Nc,Float64},2}
    f_bar_L  ::Array{SVector{Nc,Float64},2}
    f_bar_lim::Array{SVector{Nc,Float64},2}
    Uf       ::Array{SVector{Nc,Float64},2}
    VUf      ::Array{SVector{Nc,Float64},2}
    rhoef    ::Array{Float64,2}
end

mutable struct DataHistory 
    Uhist     ::Vector{Array{SVector{3,Float64},2}}
    Lhist     ::Vector{Array{Float64,2}}
    Fhist     ::Vector{Array{Float64,2}}
    alphahist ::Vector{Array{Float64,2}}
    thist     ::Vector{Float64}
    dthist    ::Vector{Float64}
    LGLindhist::Vector{BitArray}
    L_L2G_hist::Vector{Array{Float64,2}}
    L_G2L_hist::Vector{Array{Float64,2}}
    L_Vf_hist ::Vector{Array{Float64,2}}
end

mutable struct ErrorData
    L1err  ::Float64
    L2err  ::Float64
    Linferr::Float64
end

#####################
# Readable printing #
#####################
function Base.show(io::IO,::MIME"text/plain",param::Param)
    @nospecialize param
    println(io,"N=$(param.N),K=$(param.K),η=$(param.limiting_param.η),ζ=$(param.limiting_param.ζ)")
    println(io,"basis                      type: ", param.approximation_basis_type)
    println(io,"rhs                        type: ", param.rhs_type)
    println(io,"entropy projection limiter type: ", param.entropyproj_limiter_type)
    println(io,"positivity limiter         type: ", param.positivity_limiter_type)
end

function Base.show(io::IO,rhs_type::ESLimitedLowOrderPos)
    text = print(io,"ESLimitedLowOrderPos(FBL=",get_low_order_surface_flux(rhs_type),",FBH=",get_high_order_surface_flux(rhs_type))
end

function Base.show(io::IO,rhs_type::LowOrderPositivity)
    text = print(io,"LowOrderPositivity(FBL=",get_low_order_surface_flux(rhs_type))
end

function Base.show(io::IO,rhs_type::EntropyStable)
    text = print(io,"EntropyStable(FBL=",get_high_order_surface_flux(rhs_type))
end

###################
# Unpack routines #
###################
function Base.getproperty(ops::Operators{DIM}, s::Symbol) where {DIM}
    if s == :Srh_db
        return getfield(ops,:Srsh_db)[1]
    elseif s == :Ssh_db
        return getfield(ops,:Srsh_db)[2]
    elseif s == :Sr0
        return getfield(ops,:Srs0)[1]
    elseif s == :Ss0
        return getfield(ops,:Srs0)[2]
    elseif s == :Br
        return getfield(ops,:Brs)[1]
    elseif s == :Bs
        return getfield(ops,:Brs)[2]
    else
        return getfield(ops,s)
    end
end