struct Preallocation{Nc,DIM}
    Uq::Array{SVector{Nc,Float64},2}
    vq::Array{SVector{Nc,Float64},2}       # entropy variables at quad points
    u_tilde::Array{SVector{Nc,Float64},2}       # entropy projected conservative variables
    v_tilde::Array{SVector{Nc,Float64},2}       # projected entropy variables
    psi_tilde::Array{SVector{DIM,Float64},2}
    rhsH::Array{SVector{Nc,Float64},2}
    rhsL::Array{SVector{Nc,Float64},2}
    rhsU::Array{SVector{Nc,Float64},2}
    rhsxyH::Array{SVector{DIM,SVector{Nc,Float64}},2}
    rhsxyL::Array{SVector{DIM,SVector{Nc,Float64}},2}
    rhsxyU::Array{SVector{DIM,SVector{Nc,Float64}},2}
    BF_H::Array{SVector{DIM,SVector{Nc,Float64}},2}
    BF_L::Array{SVector{DIM,SVector{Nc,Float64}},2}
    fstar_H::Array{SVector{DIM,SVector{Nc,Float64}},2}
    fstar_L::Array{SVector{DIM,SVector{Nc,Float64}},2}
    L::Array{Float64,2}
    L_local::Array{Float64,4}
    theta::Array{Float64,2}
    theta_local::Array{Float64,3}
    resW::Array{SVector{Nc,Float64},2}
    resZ::Array{SVector{Nc,Float64},2}
    indicator::Array{Float64,2}
    indicator_modal::Array{Float64,2}
    smooth_indicator::Array{Float64,1}     # modal energyN/total_energy
end

abstract type Cache{DIM,Nc} end
struct LowOrderPositivityCache{DIM,Nc} <: Cache{DIM,Nc}
    flux::Array{SVector{DIM,SVector{Nc,Float64}},2}
    Q0F1::Array{SVector{DIM,SVector{Nc,Float64}},2}
    wavespeed_f::Array{Float64,2}
    Uf::Array{SVector{Nc,Float64},2}   # TODO: Redundant with limiters cache
    uP::Array{SVector{Nc,Float64},2}
    lambda::Array{Float64,3}
    lambdaB::Array{Float64,2}
    alpha::Array{Float64,2}
    dtarr::Array{Float64,1}
end

LowOrderPositivityCache{DIM,Nc}(; K=0, Np=0, Nq=0, Nh=0, Nfp=0, Nthread=1) where {DIM,Nc} =
    LowOrderPositivityCache(zeros(SVector{DIM,SVector{Nc,Float64}}, Nh, K),
        zeros(SVector{DIM,SVector{Nc,Float64}}, Nq, K),
        zeros(Float64, Nfp, K),
        zeros(SVector{Nc,Float64}, Nfp, K),
        zeros(SVector{Nc,Float64}, Nfp, K),
        zeros(Float64, Nq, Nq, K),
        zeros(Float64, Nfp, K),
        zeros(Float64, Nfp, K),
        zeros(Float64, Nthread))

struct FluxDiffCache{DIM,Nc} <: Cache{DIM,Nc}
    beta::Array{Float64,2}
    rholog::Array{Float64,2}
    betalog::Array{Float64,2}
    uP::Array{SVector{Nc,Float64},2}
    betaP::Array{Float64,2}
    rhologP::Array{Float64,2}
    betalogP::Array{Float64,2}
    lam::Array{Float64,2}
    LFc::Array{Float64,2}
    Vf_new::Array{Float64,3}
    VhT_new::Array{Float64,3}
    MinvVhT_new::Array{Float64,3}
    QF1::Array{SVector{DIM,SVector{Nc,Float64}},2}
    MinvVhTQF1::Array{SVector{DIM,SVector{Nc,Float64}},2}
    MinvVfTBF1::Array{SVector{DIM,SVector{Nc,Float64}},2}
end

FluxDiffCache{DIM,Nc}(; K=0, Np=0, Nq=0, Nh=0, Nfp=0, Nthread=1) where {DIM,Nc} =
    FluxDiffCache(zeros(Float64, Nh, K),
        zeros(Float64, Nh, K),
        zeros(Float64, Nh, K),
        zeros(SVector{Nc,Float64}, Nfp, K),
        zeros(Float64, Nfp, K),
        zeros(Float64, Nfp, K),
        zeros(Float64, Nfp, K),
        zeros(Float64, Nfp, K),
        zeros(Float64, Nfp, K),
        zeros(Float64, Nfp, Nq, Nthread),
        reshape(hcat([[diagm(ones(Nq)) zeros(Nq, Nfp)] for _ = 1:Nthread]...), Nq, Nq + Nfp, Nthread),
        zeros(Float64, Np, Nh, Nthread),
        zeros(SVector{DIM,SVector{Nc,Float64}}, Nh, K),
        zeros(SVector{DIM,SVector{Nc,Float64}}, Np, K),
        zeros(SVector{DIM,SVector{Nc,Float64}}, Np, K))

Base.@kwdef struct LimitedDGCache{CACHEHTYPE,CACHELTYPE}
    cacheH::CACHEHTYPE
    cacheL::CACHELTYPE
end

# TODO: pass in SizeData
function rhs_cache(rhs::LowOrderPositivity, param, sizes)
    (; K, Nd, Np, Nh, Nq, Nfp, Nc) = sizes

    return LowOrderPositivityCache{Nd,Nc}(K=K, Np=Np, Nq=Nq, Nh=Nh, Nfp=Nfp, Nthread=Threads.nthreads())
end

function rhs_cache(rhs::FluxDiffRHS, param, sizes)
    (; K, Nd, Np, Nh, Nq, Nfp, Nc) = sizes

    return FluxDiffCache{Nd,Nc}(K=K, Np=Np, Nq=Nq, Nh=Nh, Nfp=Nfp, Nthread=Threads.nthreads())
end

function rhs_cache(rhs::LimitedDG, param, sizes)
    (; K, Nd, Np, Nh, Nq, Nfp, Nc) = sizes

    cacheH = FluxDiffCache{Nd,Nc}(K=K, Np=Np, Nq=Nq, Nh=Nh, Nfp=Nfp, Nthread=Threads.nthreads())
    cacheL = LowOrderPositivityCache{Nd,Nc}(K=K, Np=Np, Nq=Nq, Nh=Nh, Nfp=Nfp, Nthread=Threads.nthreads())
    return LimitedDGCache(cacheH=cacheH, cacheL=cacheL)
end

struct ShockCaptureCache{DIM,Nc} <: Cache{DIM,Nc}
    blending_factor::Array{Float64,2}
end

ShockCaptureCache{DIM,Nc}(; K=0, Ns=0) where {DIM,Nc} =
    ShockCaptureCache{DIM,Nc}(zeros(K, Ns))

abstract type LimiterCache{DIM,Nc} <: Cache{DIM,Nc} end
struct NoRHSLimiterCache{DIM,Nc} <: LimiterCache{DIM,Nc} end
struct ZhangShuLimiterCache{DIM,Nc} <: LimiterCache{DIM,Nc}
    uL_k::Array{SVector{Nc,Float64},2}
    P_k::Array{SVector{Nc,Float64},2}
end

ZhangShuLimiterCache{DIM,Nc}(; Nq=0, Nthread=1) where {DIM,Nc} =
    ZhangShuLimiterCache{DIM,Nc}(zeros(SVector{Nc,Float64}, Nq, Nthread),
        zeros(SVector{Nc,Float64}, Nq, Nthread))

struct SubcellLimiterCache{DIM,Nc} <: LimiterCache{DIM,Nc}
    rhoL::Array{Float64,2}                      # Low order density update
    vf::Array{SVector{Nc,Float64},2}          # Low order v and psi at interface
    psif::Array{SVector{DIM,Float64},2}
    uL_k::Array{SVector{Nc,Float64},2}
    P_k::Array{SVector{Nc,Float64},2}
    f_bar_H::NTuple{DIM,Array{SVector{Nc,Float64},2}}
    f_bar_L::NTuple{DIM,Array{SVector{Nc,Float64},2}}
    f_bar_lim::NTuple{DIM,Array{SVector{Nc,Float64},2}}       # TODO: unnecessary
    dfH_vol::NTuple{DIM,Array{SVector{Nc,Float64},2}}       # \Delta^vol fbarH
    dfL_vol::NTuple{DIM,Array{SVector{Nc,Float64},2}}
    df_vol::NTuple{DIM,Array{SVector{Nc,Float64},2}}
    dfH_surf::NTuple{DIM,Array{SVector{Nc,Float64},2}}       # \Delta^surf fbarH
    dfL_surf::NTuple{DIM,Array{SVector{Nc,Float64},2}}
    df_surf::NTuple{DIM,Array{SVector{Nc,Float64},2}}
    lbound_rho::Array{Float64,2}              # TVD bound
    ubound_rho::Array{Float64,2}
    s_modified::Array{Float64,2}
    var_s_modified::Array{Float64,2}
    lbound_s_modified::Array{Float64,2}
    # TODO: use array so the value could be mutated... not a clean solution
    s_modified_min::Array{Float64,1}              # Global s_modified minimum
    smooth_factor::Array{Float64,2}
    sum_Bpsi::Array{SVector{DIM,Float64},1}   # 1T B_k psi_k
    sum_dvfbarL::Array{SVector{DIM,Float64},1}   # 1T (Dv)^T fbar_H
    dvdf::NTuple{DIM,Array{Float64,2}}    # (Dv)^T (fbar_H-fbar_L)
    dvdf_order::Array{Tuple{Float64,Int64},2}
end

SubcellLimiterCache{DIM,Nc}(; K=0, Nq=0, Nfp=0, N1D=0, Ns=Ns, Nthread=1) where {DIM,Nc} =
    SubcellLimiterCache{DIM,Nc}(zeros(Float64, Nq, K),
        zeros(SVector{Nc,Float64}, Nfp, K),
        zeros(SVector{DIM,Float64}, Nfp, K),
        zeros(SVector{Nc,Float64}, Nq, Nthread),
        zeros(SVector{Nc,Float64}, Nq, Nthread),
        tuple([zeros(SVector{Nc,Float64}, Nq + N1D, K) for _ in 1:DIM]...),
        tuple([zeros(SVector{Nc,Float64}, Nq + N1D, K) for _ in 1:DIM]...),
        tuple([zeros(SVector{Nc,Float64}, Nq + N1D, K) for _ in 1:DIM]...),
        tuple([zeros(SVector{Nc,Float64}, Nq, K) for _ in 1:DIM]...),
        tuple([zeros(SVector{Nc,Float64}, Nq, K) for _ in 1:DIM]...),
        tuple([zeros(SVector{Nc,Float64}, Nq, K) for _ in 1:DIM]...),
        tuple([zeros(SVector{Nc,Float64}, Nq, K) for _ in 1:DIM]...),
        tuple([zeros(SVector{Nc,Float64}, Nq, K) for _ in 1:DIM]...),
        tuple([zeros(SVector{Nc,Float64}, Nq, K) for _ in 1:DIM]...),
        zeros(Float64, Nq, K),
        zeros(Float64, Nq, K),
        zeros(Float64, Nq, K),
        zeros(Float64, Nq, K),
        zeros(Float64, Nq, K),
        zeros(Float64, 1),
        zeros(Float64, K, Ns),
        zeros(SVector{DIM,Float64}, K),
        zeros(SVector{DIM,Float64}, K),
        tuple([zeros(Float64, Nq - N1D, K) for _ in 1:DIM]...),
        [(0.0, 0) for _ = 1:Nq-N1D, _ = 1:Nthread])

# TODO: hardcoded for Compressible Euler
abstract type EntropyProjLimiterCache{DIM,Nc} <: Cache{DIM,Nc} end
struct NoEntropyProjectionLimiterCache{DIM,Nc} <: EntropyProjLimiterCache{DIM,Nc} end
struct EntropyProjectionLimiterCache{DIM,Nc} <: EntropyProjLimiterCache{DIM,Nc}
    vq_k::Array{SVector{Nc,Float64},2}   # TODO: unnecessary? Clean up
    v_tilde_k::Array{SVector{Nc,Float64},2}   # TODO: refactor with v_tilde, u_tilde
    u_tilde_k::Array{SVector{Nc,Float64},2}
    v3tilde::Array{Float64,2}
    rhotilde::Array{Float64,2}
    rhoetilde::Array{Float64,2}
    Uf::Array{SVector{Nc,Float64},2}
    VUf::Array{SVector{Nc,Float64},2}
    rhoef::Array{Float64,2}
end

EntropyProjectionLimiterCache{DIM,Nc}(; K=0, Nq=0, Nh=0, Nfp=0, Nthread=1) where {DIM,Nc} =
    EntropyProjectionLimiterCache{DIM,Nc}(zeros(SVector{Nc,Float64}, Nq, Nthread),
        zeros(SVector{Nc,Float64}, Nh, Nthread),
        zeros(SVector{Nc,Float64}, Nh, Nthread),
        zeros(Float64, Nh, Nthread),
        zeros(Float64, Nh, Nthread),
        zeros(Float64, Nh, Nthread),
        zeros(SVector{Nc,Float64}, Nfp, K),
        zeros(SVector{Nc,Float64}, Nfp, K),
        zeros(Float64, Nfp, K))

function shockcapture_cache(shockcapture, param, sizes)
    (; Nc, Ns, Nd, K) = sizes

    return ShockCaptureCache{Nd,Nc}(K=K, Ns=Ns)
end

function limiter_cache(limiter::NoRHSLimiter, param, sizes)
    (; Nd, Nc) = sizes

    return NoRHSLimiterCache{Nd,Nc}()
end

function limiter_cache(limiter::ZhangShuLimiter, param, sizes)
    (; Nq, Nc, Nd) = sizes

    return ZhangShuLimiterCache{Nd,Nc}(Nq=Nq, Nthread=Threads.nthreads())
end

function limiter_cache(limiter::SubcellLimiter, param, sizes)
    (; K, Nd, N1D, Nq, Nfp, Nc, Ns) = sizes

    return SubcellLimiterCache{Nd,Nc}(K=K, Nq=Nq, Nfp=Nfp, N1D=N1D, Ns=Ns, Nthread=Threads.nthreads())
end

function entropyproj_limiter_cache(entropyproj_limiter::NoEntropyProjectionLimiter, param, sizes)
    (; Nc, Nd) = sizes

    return NoEntropyProjectionLimiterCache{Nd,Nc}()
end

function entropyproj_limiter_cache(entropyproj_limiter::ScaledExtrapolation, param, sizes)
    (; K, Nd, Np, Nh, Nq, Nfp, Nc) = sizes

    return EntropyProjectionLimiterCache{Nd,Nc}(K=K, Np=Np, Nq=Nq, Nh=Nh, Nfp=Nfp, Nthread=Threads.nthreads())
end


struct Caches{RHSCACHE,LIMITERCACHE,SHOCKCAPTURECACHE,ENTROPYPROJCACHE,POSTPROCESSCACHE}
    rhs_cache::RHSCACHE
    limiter_cache::LIMITERCACHE
    shockcapture_cache::SHOCKCAPTURECACHE
    entropyproj_limiter_cache::ENTROPYPROJCACHE
    postprocessing_cache::POSTPROCESSCACHE
end

Base.@kwdef struct State{PREALLOCATION,CACHES}
    preallocation::PREALLOCATION
    cache::CACHES
end


