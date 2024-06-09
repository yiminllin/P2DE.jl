include("./limiter_utils.jl")
include("./zhangshu.jl")
include("./subcell.jl")

########################
### Appy RHS limiter ###
########################
function apply_rhs_limiter!(rhs_limiter_type::ZhangShuLimiter, state, solver, state_param, time_param)
    @timeit_debug time_param.timer "Initialize smoothness indicator" begin
        initialize_smoothness_indicator!(shockcapture_type(solver), bound_type(solver), state, solver)
    end
    @timeit_debug time_param.timer "calculate blending factor" begin
        update_blending_factor!(shockcapture_type(solver), state, solver, time_param)
    end
    @timeit_debug time_param.timer "Apply Zhang-Shu limiter" begin
        apply_zhang_shu_limiter!(state, solver, time_param)
    end
end

function apply_rhs_limiter!(rhs_limiter_type::SubcellLimiter, state, solver, state_param, time_param)
    @timeit_debug time_param.timer "Initialize smoothness indicator" begin
        initialize_smoothness_indicator!(shockcapture_type(solver), bound_type(solver), state, solver)
    end
    @timeit_debug time_param.timer "calculate blending factor" begin
        update_blending_factor!(shockcapture_type(solver), state, solver, time_param)
    end
    @timeit_debug time_param.timer "calculate smoothness factor" begin
        update_smoothness_factor!(bound_type(solver), state, solver, time_param)
    end
    @timeit_debug time_param.timer "Precompute bounds on modified s" begin
        initialize_entropy_bounds!(equation(solver), bound_type(solver), state, solver, state_param, time_param)
    end
    @timeit_debug time_param.timer "Precompute TVD bounds" begin
        initialize_TVD_bounds!(dim_type(solver), equation(solver), bound_type(solver), state, solver, state_param, time_param)
    end
    @timeit_debug time_param.timer "Accumulate low and high order subcell fluxes" begin
        accumulate_f_bar!(dim_type(solver), state, solver)
    end
    @timeit_debug time_param.timer "Find subcell limiting parameters" begin
        subcell_bound_limiter!(dim_type(solver), equation(solver), state, solver, state_param, time_param)
    end
    @timeit_debug time_param.timer "Find subcell limiting parameters for entropy stability" begin
        enforce_ES_subcell!(bound_type(solver), state, solver, state_param, time_param)
    end
    @timeit_debug time_param.timer "Symmetrize subcell limiting parameters" begin
        symmetrize_limiting_parameters!(dim_type(solver), state, solver, state_param, time_param)
    end
    @timeit_debug time_param.timer "Accumulate limited subcell fluxes" begin
        accumulate_f_bar_limited!(dim_type(solver), state, solver, time_param)
    end
    @timeit_debug time_param.timer "Apply subcell limiter, accumulate limited rhs" begin
        apply_subcell_limiter!(dim_type(solver), state, solver)
    end

    # check_subcell_entropy_stability(limiter_cache,prealloc,param,discrete_data,dim)
end
