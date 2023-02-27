# N = 3
# Nq = 16
# N1D = N+1
# N1Dp1 = N1D+1
# for j = 1:N1D
#     iface = 1+(j-1)*N1Dp1                   # fbar index of leftmost subcell face
#     ifacequad = j                           # Index of the surface quadrature node correspond to the leftmost subcell face
#     @show iface,ifacequad
#     # For each subcell face right of leftmost face
#     for i = 2:N1Dp1
#         iface = i+(j-1)*N1Dp1               # fbar index of current subcell face
#         iquad = i-1+(j-1)*N1D               # Index of the quadrature node left to the current subcell face
#         @show iface,iquad
#     end
# end

# # For each stride along y direction
# for i = 1:N1D
#     iface = 1+(i-1)*N1Dp1                   # fbar index of bottommost subcell face
#     ifacequad = i+2*N1D                     # Index of the surface quadrature node correspond to the leftmost subcell face
#     @show iface,ifacequad
#     for j = 2:N1Dp1
#         iface = j+(i-1)*N1Dp1               # fbar index of current subcell face
#         iquad = i+(j-2)*N1D               # Index of the quadrature node beneath the current subcell face
#         @show iface,iquad
#     end
# end

# K = 100
# A = ones(Nq,K)
# @. A[:,1] = 1:16

# using BenchmarkTools

# function test(A)
#     for i = 1:10000
#         tmp = reshape(view(A,:,1),4,4)
#     end
# end

# # @btime test($A)

# A = zeros(SVector{3,Float64},10,100)
# B = zeros(SVector{3,Float64},100,100)
# T = ones(100,10)
# function test(B,T,A)
#     for j = 1:100
#         @views mul!(B[:,j],T,A[:,j])
#     end
# end
# # B = TA
# @btime mul!($B,$T,$A)
# @btime test($B,$T,$A)

using StaticArrays

abstract type Cache{DIM,Nc} end
struct LowOrderPositivityCache{DIM,Nc} <: Cache{DIM,Nc}
    flux       ::Array{SVector{DIM,SVector{Nc,Float64}},2}
    Q0F1       ::Array{SVector{DIM,SVector{Nc,Float64}},2}
    wavespeed  ::Array{Float64,3}
    wavespeed_f::Array{Float64,2}
    alphaarr   ::Array{Float64,2}
    Uf         ::Array{SVector{Nc,Float64},2}    # TODO: Redundant with limiters cache
    λarr       ::Array{Float64,3}
    λBarr      ::Array{Float64,2}
end

LowOrderPositivityCache{DIM,Nc}(; K=0,Np=0,Nq=0,Nh=0,Nfp=0) where {DIM,Nc} =
    LowOrderPositivityCache(zeros(SVector{DIM,SVector{Nc,Float64}},Nh,K),
                            zeros(SVector{DIM,SVector{Nc,Float64}},Nq,K),
                            zeros(Float64,Nq,Nq,K),
                            zeros(Float64,Nfp,K),
                            zeros(Float64,Nfp,K),
                            zeros(SVector{Nc,Float64},Nfp,K),
                            zeros(Float64,Nq,Nq,K),
                            zeros(Float64,Nfp,K))

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
    Ui         ::Array{Float64,1}
    Uj         ::Array{Float64,1}
    Vf_new     ::Array{Float64,2}
    VhT_new    ::Array{Float64,2}
    MinvVhT_new::Array{Float64,2}
    QF1        ::Array{SVector{DIM,SVector{Nc,Float64}},2}
    MinvVhTQF1 ::Array{SVector{DIM,SVector{Nc,Float64}},2}     # TODO: inconsistent with Q0F1
    MinvVfTBF1 ::Array{SVector{DIM,SVector{Nc,Float64}},2}
end

EntropyStableCache{DIM,Nc}(; K=0,Np=0,Nq=0,Nh=0,Nfp=0) where {DIM,Nc} =
    EntropyStableCache(zeros(Float64,Nh,K),
                       zeros(Float64,Nh,K),
                       zeros(Float64,Nh,K),
                       zeros(SVector{Nc,Float64},Nfp,K),
                       zeros(Float64,Nfp,K),
                       zeros(Float64,Nfp,K),
                       zeros(Float64,Nfp,K),
                       zeros(Float64,Nfp,K),
                       zeros(Float64,Nfp,K),
                       zeros(Float64,Nc+2),
                       zeros(Float64,Nc+2),
                       zeros(Float64,Nfp,Nq),
                       zeros(Float64,Np,Nh),
                       zeros(Float64,Np,Nh),
                       zeros(SVector{DIM,SVector{Nc,Float64}},Nh,K),
                       zeros(SVector{DIM,SVector{Nc,Float64}},Np,K),
                       zeros(SVector{DIM,SVector{Nc,Float64}},Np,K))

abstract type SurfaceFluxType end
struct ChandrashekarOnProjectedVal <: SurfaceFluxType end
struct LaxFriedrichsOnNodalVal     <: SurfaceFluxType end
struct LaxFriedrichsOnProjectedVal <: SurfaceFluxType end

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

abstract type RHSData end
Base.@kwdef struct RHSMonoData{RHSTYPE,CACHETYPE} <: RHSData
    rhs_type::RHSTYPE
    cache   ::CACHETYPE
end
Base.@kwdef struct RHSLimitData{RHSTYPE,CACHEHTYPE,CACHELTYPE} <: RHSData
    rhs_type::RHSTYPE
    cacheH  ::CACHEHTYPE
    cacheL  ::CACHELTYPE
end

const LowOrderPositivityData   = RHSMonoData{LowOrderPositivity,LowOrderPositivityCache}
const EntropyStableData        = RHSMonoData{EntropyStable     ,EntropyStableCache}
const ESLimitedLowOrderPosData = RHSLimitData{ESLimitedLowOrderPos,EntropyStableCache,LowOrderPositivityCache}

function get_low_order_cache(rhs_data::LowOrderPositivityData)
    return rhs_data.cache
end

function get_low_order_cache(rhs_data::ESLimitedLowOrderPosData)
    return rhs_data.cacheL
end

function get_high_order_cache(rhs_data::EntropyStableCache)
    return rhs_data.cache
end

function get_high_order_cache(rhs_data::ESLimitedLowOrderPosData)
    return rhs_data.cacheH
end

Nd = 1
Nc = 3
K = 100
Np = 16
Nq = 16
Nh = 32
Nfp = 16
rhsL_type = LowOrderPositivity(LaxFriedrichsOnNodalVal())
rhsH_type = LowOrderPositivity(LaxFriedrichsOnProjectedVal())
rhs_type  = ESLimitedLowOrderPos(LaxFriedrichsOnProjectedVal(),LaxFriedrichsOnProjectedVal())
cacheL    = LowOrderPositivityCache{Nd,Nc}()
cacheH    = EntropyStableCache{Nd,Nc}()


rhsL_data = get_rhs_data(rhsL_type,Nd,Nc,K,Np,Nq,Nh,Nfp)
rhsH_data = get_rhs_data(rhsH_type,Nd,Nc,K,Np,Nq,Nh,Nfp)
rhs_data  = get_rhs_data(rhs_type,Nd,Nc,K,Np,Nq,Nh,Nfp)