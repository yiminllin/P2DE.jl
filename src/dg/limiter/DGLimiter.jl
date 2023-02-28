include("./DGLimiterUtils.jl")
include("./ZhangShuLimiter.jl")
include("./SubcellLimiter.jl")

############################################
### Appy positivity limiter to limit RHS ###
############################################
function apply_positivity_limiter!(prealloc,param,discrete_data,bcdata,cache,dt,nstage,positivity_limiter_type::ZhangShuLimiter,timer)
    @unpack Uq,rhsL,rhsH,rhsU = prealloc
    @unpack uL_k,P_k          = cache
    
    K  = get_num_elements(param)
    ζ = param.limiting_param.ζ
    for k = 1:K
        @views @. uL_k = Uq[:,k] + dt*rhsL[:,k]
        @views @. P_k  = dt*(rhsH[:,k]-rhsL[:,k])
        Lrho(uL_i)  = ζ*uL_i[1]
        Lrhoe(uL_i) = ζ*rhoe_ufun(param.equation,uL_i)
        Urho  = Inf
        Urhoe = Inf
        zhang_shu_bound_limiter!(prealloc.Larr,param,uL_k,P_k,k,Lrho,Lrhoe,Urho,Urhoe,nstage)
        l = prealloc.Larr[k,nstage]
        @views @. rhsU[:,k] = (1-l)*rhsL[:,k] + l*(rhsH[:,k])
    end
end

function apply_positivity_limiter!(prealloc,param,discrete_data,bcdata,cache,dt,nstage,positivity_limiter_type::SubcellLimiter,timer)
    dim = get_dim_type(param.equation)
    @timeit_debug timer "Accumulate low and high order subcell fluxes" begin
    accumulate_f_bar!(cache,prealloc,param,discrete_data,dim)
    end
    @timeit_debug timer "Find subcell limiting parameters" begin
    subcell_bound_limiter!(cache,prealloc,param,discrete_data,bcdata,dt,nstage,dim)
    end
    @timeit_debug timer "Accumulate limited subcell fluxes" begin
    accumulate_f_bar_limited!(cache,prealloc,param,nstage,dim)
    end
    @timeit_debug timer "Apply subcell limiter, accumulate limited rhs" begin
    apply_subcell_limiter!(prealloc,cache,param,discrete_data,dim)
    end
end
