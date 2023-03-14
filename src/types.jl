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

abstract type Cache{DIM,Nc} end
struct LowOrderPositivityCache{DIM,Nc} <: Cache{DIM,Nc}
    flux       ::Array{SVector{DIM,SVector{Nc,Float64}},2}
    Q0F1       ::Array{SVector{DIM,SVector{Nc,Float64}},2}
    wavespeed_f::Array{Float64,2}
    alphaarr   ::Array{Float64,2}
    Uf         ::Array{SVector{Nc,Float64},2}   # TODO: Redundant with limiters cache
    uP         ::Array{SVector{Nc,Float64},2}
    λarr       ::Array{Float64,3}
    λBarr      ::Array{Float64,2}
    αarr       ::Array{Float64,2}
    dtarr      ::Array{Float64,1}
end

LowOrderPositivityCache{DIM,Nc}(; K=0,Np=0,Nq=0,Nh=0,Nfp=0,Nthread=1) where {DIM,Nc} =
    LowOrderPositivityCache(zeros(SVector{DIM,SVector{Nc,Float64}},Nh,K),
                            zeros(SVector{DIM,SVector{Nc,Float64}},Nq,K),
                            zeros(Float64,Nfp,K),
                            zeros(Float64,Nfp,K),
                            zeros(SVector{Nc,Float64},Nfp,K),
                            zeros(SVector{Nc,Float64},Nfp,K),
                            zeros(Float64,Nq,Nq,K),
                            zeros(Float64,Nfp,K),
                            zeros(Float64,Nfp,K),
                            zeros(Float64,Nthread))

struct EntropyStableCache{DIM,Nc} <: Cache{DIM,Nc}
    beta       ::Array{Float64,2}
    rholog     ::Array{Float64,2}
    betalog    ::Array{Float64,2}
    uP         ::Array{SVector{Nc,Float64},2}
    betaP      ::Array{Float64,2}
    rhologP    ::Array{Float64,2}
    betalogP   ::Array{Float64,2}
    lam        ::Array{Float64,2}
    LFc        ::Array{Float64,2}
    Vf_new     ::Array{Float64,3}
    VhT_new    ::Array{Float64,3}
    MinvVhT_new::Array{Float64,3}
    QF1        ::Array{SVector{DIM,SVector{Nc,Float64}},2}
    MinvVhTQF1 ::Array{SVector{DIM,SVector{Nc,Float64}},2}
    MinvVfTBF1 ::Array{SVector{DIM,SVector{Nc,Float64}},2}
end

EntropyStableCache{DIM,Nc}(; K=0,Np=0,Nq=0,Nh=0,Nfp=0,Nthread=1) where {DIM,Nc} =
    EntropyStableCache(zeros(Float64,Nh,K),
                       zeros(Float64,Nh,K),
                       zeros(Float64,Nh,K),
                       zeros(SVector{Nc,Float64},Nfp,K),
                       zeros(Float64,Nfp,K),
                       zeros(Float64,Nfp,K),
                       zeros(Float64,Nfp,K),
                       zeros(Float64,Nfp,K),
                       zeros(Float64,Nfp,K),
                       zeros(Float64,Nfp,Nq,Nthread),
                       reshape(hcat([[diagm(ones(Nq)) zeros(Nq,Nfp)] for _ = 1:Nthread]...),Nq,Nq+Nfp,Nthread),
                       zeros(Float64,Np,Nh,Nthread),
                       zeros(SVector{DIM,SVector{Nc,Float64}},Nh,K),
                       zeros(SVector{DIM,SVector{Nc,Float64}},Np,K),
                       zeros(SVector{DIM,SVector{Nc,Float64}},Np,K))

Base.@kwdef struct ESLimitedLowOrderPosCache{CACHEHTYPE,CACHELTYPE}
    cacheH  ::CACHEHTYPE
    cacheL  ::CACHELTYPE
end

# TODO: pass in SizeData
function get_rhs_cache(rhs_type::LowOrderPositivity,param,sizes)
    @unpack Np,Nh,Nq,Nfp,Nc,Ns = sizes
    K  = get_num_elements(param)
    Nd = get_dim(param.equation)

    return LowOrderPositivityCache{Nd,Nc}(K=K,Np=Np,Nq=Nq,Nh=Nh,Nfp=Nfp,Nthread=Threads.nthreads())
end

function get_rhs_cache(rhs_type::EntropyStable,param,sizes)
    @unpack Np,Nh,Nq,Nfp,Nc,Ns = sizes
    K  = get_num_elements(param)
    Nd = get_dim(param.equation)

    return EntropyStableCache{Nd,Nc}(K=K,Np=Np,Nq=Nq,Nh=Nh,Nfp=Nfp,Nthread=Threads.nthreads())
end

function get_rhs_cache(rhs_type::ESLimitedLowOrderPos,param,sizes)
    @unpack Np,Nh,Nq,Nfp,Nc,Ns = sizes
    K  = get_num_elements(param)
    Nd = get_dim(param.equation)

    cacheH = EntropyStableCache{Nd,Nc}(K=K,Np=Np,Nq=Nq,Nh=Nh,Nfp=Nfp,Nthread=Threads.nthreads())
    cacheL = LowOrderPositivityCache{Nd,Nc}(K=K,Np=Np,Nq=Nq,Nh=Nh,Nfp=Nfp,Nthread=Threads.nthreads())
    return ESLimitedLowOrderPosCache(cacheH = cacheH, cacheL = cacheL)
end

function get_low_order_cache(rhs_cache::LowOrderPositivityCache)
    return rhs_cache
end

function get_low_order_cache(rhs_cache::ESLimitedLowOrderPosCache)
    return rhs_cache.cacheL
end

function get_high_order_cache(rhs_cache::EntropyStableCache)
    return rhs_cache
end

function get_high_order_cache(rhs_cache::ESLimitedLowOrderPosCache)
    return rhs_cache.cacheH
end

abstract type EntropyProjectionLimiterType end
abstract type ScaledExtrapolation     <: EntropyProjectionLimiterType end
struct NoEntropyProjectionLimiter     <: EntropyProjectionLimiterType end
struct ElementwiseScaledExtrapolation <: ScaledExtrapolation end
struct NodewiseScaledExtrapolation    <: ScaledExtrapolation end

# TODO: rename, shouldn't called positivity limiter. RHS limiter instead?
abstract type PositivityLimiterType end
struct NoPositivityLimiter <: PositivityLimiterType end
struct ZhangShuLimiter     <: PositivityLimiterType end

# TODO: It should depend on Equation type... Hardcode for CompressibleIdealGas for now
abstract type LimiterBoundType end
struct PositivityBound              <: LimiterBoundType end
struct PositivityAndMinEntropyBound <: LimiterBoundType end

Base.@kwdef struct SubcellLimiter{BOUNDTYPE<:LimiterBoundType} <: PositivityLimiterType
    bound_type::BOUNDTYPE
end

function get_bound_type(limiter::ZhangShuLimiter)
    return PositivityBound()
end

function get_bound_type(limiter::SubcellLimiter)
    return limiter.bound_type
end

abstract type ApproxBasisType end
struct GaussCollocation   <: ApproxBasisType end
struct LobattoCollocation <: ApproxBasisType end

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

function get_equation_1D(equation::CompressibleIdealGas{Dim2})
    return CompressibleEulerIdealGas{Dim1}(get_γ(equation))
end

function get_dim(equation::EquationType{Dim1})
    return 1
end

function get_dim(equation::EquationType{Dim2})
    return 2
end

function get_dim_type(equation::EquationType{Dim1})
    return Dim1()
end

function get_dim_type(equation::EquationType{Dim2})
    return Dim2()
end

function get_num_components(equation::CompressibleFlow{Dim1})
    return 3
end

function get_num_components(equation::CompressibleFlow{Dim2})
    return 4
end

struct KPP{DIM} <: EquationType{DIM} end

function get_num_components(equation::KPP{Dim2})
    return 1
end

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

Base.@kwdef struct PostprocessingCache{Nc}
    xp::Array{Float64,2}
    yp::Array{Float64,2}
    Up::Array{SVector{Nc,Float64},2}
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

# TODO: refactor
function get_num_elements(param)
    return get_num_elements(param,param.equation)
end

function get_num_elements(param,equation::EquationType{Dim1})
    return param.K
end

# TODO: hardcoded for uniform mesh
function get_num_elements(param,equation::EquationType{Dim2})
    return param.K[1]*param.K[2]
end

struct BCData{Nc}
    mapP::Array{Int64,2}
    mapI::Array{Int64,1}
    mapO::Array{Int64,1}  # List of global indices with inflow and outflow
                          # (do nothing) boundary conditions
    Ival::Array{SVector{Nc,Float64},1}
end

struct GeomData{NGEO}
    J   ::Array{Float64,2}
    Jq  ::Array{Float64,2}
    GJh::NTuple{NGEO,Array{Float64,2}}   # rxJh in 1D
                                         # rxJh, sxJh, ryJh, syJh in 2D
end

struct Operators{DIM}
    Srsh_db    ::NTuple{DIM,Array{Float64,2}}
    Srs0       ::NTuple{DIM,SparseMatrixCSC{Float64,Int64}}
    Srsh_nnz   ::Array{Tuple{Int64,Int64},1}
    Srs0_nnz   ::Array{Tuple{Int64,Int64},1}
    Brs        ::NTuple{DIM,Array{Float64,2}}    # TODO: to sparse matrix
    Vh         ::Array{Float64,2}
    MinvVhT    ::Array{Float64,2}
    VDM_inv    ::Array{Float64,2}
    VDMinvPq   ::Array{Float64,2}
    VqVDM      ::Array{Float64,2}
    VhPq       ::Array{Float64,2}
    Vq         ::Array{Float64,2}
    Vf         ::Array{Float64,2}
    Vf_low     ::Array{Float64,2}
    Pq         ::Array{Float64,2}
    MinvVfT    ::Array{Float64,2}
    wq         ::Array{Float64,1}
    q2fq       ::Array{Array{Int64,1},1}
    fq2q       ::Array{Int64,1}
end

struct TransferOperators
    T_g2l::Array{Float64,2}
    T_l2g::Array{Float64,2}
end

# TODO: define iterator to loop instead of size
struct SizeData
    Nc ::Int64
    Np ::Int64
    Nq ::Int64
    Nfp::Int64
    Nh ::Int64
    Ns ::Int64    # number of stages in time integrator
end

struct DiscretizationData{DIM,NGEO}
    sizes::SizeData
    geom ::GeomData{NGEO}
    ops  ::Operators{DIM}
end

struct Preallocation{Nc,DIM}
    Uq         ::Array{SVector{Nc,Float64},2}
    vq         ::Array{SVector{Nc,Float64},2}       # entropy variables at quad points
    u_tilde    ::Array{SVector{Nc,Float64},2}       # entropy projected conservative variables
    v_tilde    ::Array{SVector{Nc,Float64},2}       # projected entropy variables
    rhsH       ::Array{SVector{Nc,Float64},2}
    rhsL       ::Array{SVector{Nc,Float64},2}
    rhsU       ::Array{SVector{Nc,Float64},2}
    rhsxyH     ::Array{SVector{DIM,SVector{Nc,Float64}},2}
    rhsxyL     ::Array{SVector{DIM,SVector{Nc,Float64}},2}
    rhsxyU     ::Array{SVector{DIM,SVector{Nc,Float64}},2}
    BF_H       ::Array{SVector{DIM,SVector{Nc,Float64}},2}
    BF_L       ::Array{SVector{DIM,SVector{Nc,Float64}},2}
    Larr       ::Array{Float64,2}
    L_local_arr::Array{Float64,4}
    θ_arr      ::Array{Float64,2}
    θ_local_arr::Array{Float64,3}
    resW       ::Array{SVector{Nc,Float64},2}
    resZ       ::Array{SVector{Nc,Float64},2}
end

abstract type LimiterCache{DIM,Nc} <: Cache{DIM,Nc} end
struct NoPositivityLimiterCache{DIM,Nc} <: LimiterCache{DIM,Nc} end
struct ZhangShuLimiterCache{DIM,Nc} <: LimiterCache{DIM,Nc}
    uL_k     ::Array{SVector{Nc,Float64},2}
    P_k      ::Array{SVector{Nc,Float64},2}
end

ZhangShuLimiterCache{DIM,Nc}(; Nq=0,Nthread=1) where {DIM,Nc} =
    ZhangShuLimiterCache{DIM,Nc}(zeros(SVector{Nc,Float64},Nq,Nthread),
                                 zeros(SVector{Nc,Float64},Nq,Nthread))

struct SubcellLimiterCache{DIM,Nc} <: LimiterCache{DIM,Nc}
    uL_k     ::Array{SVector{Nc,Float64},2}
    P_k      ::Array{SVector{Nc,Float64},2}
    f_bar_H  ::NTuple{DIM,Array{SVector{Nc,Float64},2}}
    f_bar_L  ::NTuple{DIM,Array{SVector{Nc,Float64},2}}
    f_bar_lim::NTuple{DIM,Array{SVector{Nc,Float64},2}}       # TODO: unnecessary
    s_modified       ::Array{Float64,2}
    var_s_modified   ::Array{Float64,2}
    lbound_s_modified::Array{Float64,2}
end

SubcellLimiterCache{DIM,Nc}(; K=0,Nq=0,N1D=0,Nthread=1) where {DIM,Nc} =
    SubcellLimiterCache{DIM,Nc}(zeros(SVector{Nc,Float64},Nq,Nthread),
                                zeros(SVector{Nc,Float64},Nq,Nthread),
                                tuple([zeros(SVector{Nc,Float64},Nq+N1D,K) for _ in 1:DIM]...),
                                tuple([zeros(SVector{Nc,Float64},Nq+N1D,K) for _ in 1:DIM]...),
                                tuple([zeros(SVector{Nc,Float64},Nq+N1D,K) for _ in 1:DIM]...),
                                zeros(Float64,Nq,K),
                                zeros(Float64,Nq,K),
                                zeros(Float64,Nq,K))

# TODO: hardcoded for Compressible Euler
abstract type EntropyProjLimiterCache{DIM,Nc} <: Cache{DIM,Nc} end
struct NoEntropyProjectionLimiterCache{DIM,Nc} <: EntropyProjLimiterCache{DIM,Nc} end
struct EntropyProjectionLimiterCache{DIM,Nc} <: EntropyProjLimiterCache{DIM,Nc}
    vq_k     ::Array{SVector{Nc,Float64},2}
    v_tilde_k::Array{SVector{Nc,Float64},2}   # TODO: refactor with v_tilde, u_tilde
    u_tilde_k::Array{SVector{Nc,Float64},2}
    v3tilde  ::Array{Float64,2}
    rhotilde ::Array{Float64,2}
    rhoetilde::Array{Float64,2}
    Uf       ::Array{SVector{Nc,Float64},2}
    VUf      ::Array{SVector{Nc,Float64},2}
    rhoef    ::Array{Float64,2}
end

EntropyProjectionLimiterCache{DIM,Nc}(; K=0,Np=0,Nq=0,Nh=0,Nfp=0,Nthread=1) where {DIM,Nc} =
    EntropyProjectionLimiterCache{DIM,Nc}(zeros(SVector{Nc,Float64},Nq,Nthread),
                                          zeros(SVector{Nc,Float64},Nh,Nthread),
                                          zeros(SVector{Nc,Float64},Nh,Nthread),
                                          zeros(Float64,Nh,Nthread),
                                          zeros(Float64,Nh,Nthread),
                                          zeros(Float64,Nh,Nthread),
                                          zeros(SVector{Nc,Float64},Nfp,K),
                                          zeros(SVector{Nc,Float64},Nfp,K),
                                          zeros(Float64,Nfp,K))

function get_limiter_cache(limiter_type::NoPositivityLimiter,param,sizes)
    @unpack Np,Nh,Nq,Nfp,Nc,Ns = sizes
    Nd = get_dim(param.equation)

    return NoPositivityLimiterCache{Nd,Nc}()
end

function get_limiter_cache(limiter_type::ZhangShuLimiter,param,sizes)
    @unpack Np,Nh,Nq,Nfp,Nc,Ns = sizes
    K  = get_num_elements(param)
    Nd = get_dim(param.equation)

    return ZhangShuLimiterCache{Nd,Nc}(Nq=Nq,Nthread=Threads.nthreads())
end

function get_limiter_cache(limiter_type::SubcellLimiter,param,sizes)
    @unpack Np,Nh,Nq,Nfp,Nc,Ns = sizes
    K  = get_num_elements(param)
    Nd = get_dim(param.equation)
    N1D = Nd == 1 ? 1 : param.N+1      # TODO: hardcoded

    return SubcellLimiterCache{Nd,Nc}(K=K,Nq=Nq,N1D=N1D,Nthread=Threads.nthreads())
end

function get_entropyproj_limiter_cache(entropyproj_limiter_type::NoEntropyProjectionLimiter,param,sizes)
    @unpack Np,Nh,Nq,Nfp,Nc,Ns = sizes
    Nd = get_dim(param.equation)

    return NoEntropyProjectionLimiterCache{Nd,Nc}()
end

function get_entropyproj_limiter_cache(entropyproj_limiter_type::ScaledExtrapolation,param,sizes)
    @unpack Np,Nh,Nq,Nfp,Nc,Ns = sizes
    K  = get_num_elements(param)
    Nd = get_dim(param.equation)

    return EntropyProjectionLimiterCache{Nd,Nc}(K=K,Np=Np,Nq=Nq,Nh=Nh,Nfp=Nfp,Nthread=Threads.nthreads())
end


struct Caches{RHSCACHE,LIMITERCACHE,ENTROPYPROJCACHE,POSTPROCESSCACHE}
    rhs_cache                ::RHSCACHE
    limiter_cache            ::LIMITERCACHE
    entropyproj_limiter_cache::ENTROPYPROJCACHE
    postprocessing_cache     ::POSTPROCESSCACHE
end

struct DataHistory{Nc}
    Uhist     ::Vector{Array{SVector{Nc,Float64},2}}
    Lhist     ::Vector{Array{Float64,2}}
    θhist     ::Vector{Array{Float64,2}}
    thist     ::Vector{Float64}
    dthist    ::Vector{Float64}
end

struct ErrorData
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
    text = print(io,"ESLimitedLowOrderPos(FBL=",get_low_order_surface_flux(rhs_type),",FBH=",get_high_order_surface_flux(rhs_type),")")
end

function Base.show(io::IO,rhs_type::LowOrderPositivity)
    text = print(io,"LowOrderPositivity(FBL=",get_low_order_surface_flux(rhs_type),")")
end

function Base.show(io::IO,rhs_type::EntropyStable)
    text = print(io,"EntropyStable(FBL=",get_high_order_surface_flux(rhs_type),")")
end

function Base.show(io::IO,flux_type::LaxFriedrichsOnNodalVal)
    text = print(io,"LFNodal")
end

function Base.show(io::IO,flux_type::LaxFriedrichsOnProjectedVal)
    text = print(io,"LFProjected")
end

function Base.show(io::IO,flux_type::ChandrashekarOnProjectedVal)
    text = print(io,"ECProjected")
end

function Base.show(io::IO,entropyproj_limiter_type::NoEntropyProjectionLimiter)
    text = print(io,"None")
end

function Base.show(io::IO,entropyproj_limiter_type::NodewiseScaledExtrapolation)
    text = print(io,"NodeOpBlend")
end

function Base.show(io::IO,entropyproj_limiter_type::ElementwiseScaledExtrapolation)
    text = print(io,"ElemOpBlend")
end

function Base.show(io::IO,bound_type::PositivityBound)
    text = print(io,"PosBound")
end

function Base.show(io::IO,bound_type::PositivityAndMinEntropyBound)
    text = print(io,"PosMinEntropyBound")
end

function Base.show(io::IO,limiter_type::SubcellLimiter)
    text = print(io,"Subcell(",get_bound_type(limiter_type),")")
end

function Base.show(io::IO,limiter_type::ZhangShuLimiter)
    text = print(io,"ZhangShu(",get_bound_type(limiter_type),")")
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

function Base.getproperty(geom::GeomData{NGEO}, s::Symbol) where {NGEO}
    if s == :rxJh
        return getfield(geom,:GJh)[1]
    elseif s == :sxJh
        return getfield(geom,:GJh)[2]
    elseif s == :ryJh
        return getfield(geom,:GJh)[3]
    elseif s == :syJh
        return getfield(geom,:GJh)[4]
    else
        return getfield(geom,s)
    end
end
