include("./DGLimiterUtils.jl")
include("./ZhangShuLimiter.jl")
include("./SubcellLimiter.jl")

########################
### Appy RHS limiter ###
########################
function apply_rhs_limiter!(prealloc,param,discrete_data,bcdata,caches,dt,nstage,rhs_limiter_type::ZhangShuLimiter,timer)
    @unpack limiter_cache,shockcapture_cache = caches
    shockcapture_type = get_shockcapture_type(rhs_limiter_type)
    bound_type = get_bound_type(rhs_limiter_type)
    @timeit_debug timer "Initialize smoothness indicator" begin
    initialize_smoothness_indicator!(shockcapture_type,bound_type,prealloc,param,discrete_data,nstage)
    end
    @timeit_debug timer "calculate blending factor" begin
    update_blending_factor!(shockcapture_type,shockcapture_cache,prealloc,param,discrete_data,nstage)
    end
    @timeit_debug timer "Apply Zhang-Shu limiter" begin
    apply_zhang_shu_limiter!(prealloc,limiter_cache,shockcapture_cache,param,dt,nstage)
    end
end

function apply_rhs_limiter!(prealloc,param,discrete_data,bcdata,caches,dt,nstage,rhs_limiter_type::SubcellLimiter,timer)
    @unpack limiter_cache,shockcapture_cache = cache
    dim = get_dim_type(param.equation)
    bound_type = get_bound_type(param.rhs_limiter_type)
    @timeit_debug timer "Precompute bounds" begin
    initialize_bounds!(limiter_cache,prealloc,bound_type,param,discrete_data,bcdata,dim)
    end
    @timeit_debug timer "Accumulate low and high order subcell fluxes" begin
    accumulate_f_bar!(limiter_cache,prealloc,param,discrete_data,dim)
    end
    @timeit_debug timer "Find subcell limiting parameters" begin
    subcell_bound_limiter!(limiter_cache,prealloc,param,discrete_data,bcdata,dt,nstage,dim)
    end
    @timeit_debug timer "Symmetrize subcell limiting parameters" begin
    symmetrize_limiting_parameters!(prealloc,param,bcdata,nstage,dim)
    end
    @timeit_debug timer "Accumulate limited subcell fluxes" begin
    accumulate_f_bar_limited!(limiter_cache,prealloc,param,nstage,dim)
    end
    @timeit_debug timer "Apply subcell limiter, accumulate limited rhs" begin
    apply_subcell_limiter!(prealloc,limiter_cache,param,discrete_data,dim)
    end
end
