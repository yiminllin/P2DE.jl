using Revise
using UnPack
using StaticArrays
using DataFrames
using JLD2
using StartUpDG

using P2DE

using TimerOutputs
TimerOutputs.enable_debug_timings(P2DE)

function exact_sol(eqn,x,y,t)
    rho = 0.5
    u   = 0.0
    v   = 0.0
    p   = 0.4127

    return (rho, u, v, p)
end

function initial_boundary_conditions(param,md)
    @unpack xL,xR,equation         = param
    @unpack K,xf,yf,mapM,mapP,mapB = md
    @unpack ZEROTOL                = param.global_constants

    Nc = get_num_components(equation)
    K  = get_num_elements(param)
    Nfp = size(mapP,1)
    # Make periodic
    md = make_periodic(md)
    @unpack mapP = md

    mapI = Float64[]
    mapO = Float64[]
    Ival = SVector{Nc,Float64}[]

    for k = 1:K 
        for i = 1:Nfp
            idx = i+Nfp*(k-1)
            # Left boundary
            if abs(xf[i,k]-xL[1]) < ZEROTOL
                val = zero(SVector{Nc,Float64})
                # Inflow
                if abs(yf[i,k]) < 0.05+ZEROTOL
                    val = primitive_to_conservative(equation, (5.0,800.0,0.0,0.4127))
                # Left boundary, but w/o inflow
                else
                    val = primitive_to_conservative(equation, (0.5,0.0,0.0,0.4127))
                end
                push!(mapI,idx)
                push!(Ival,val)
            # Right boundary
            elseif abs(xf[i,k]-xR[1]) < ZEROTOL
                # outflow
                push!(mapO,idx)
            end
        end
    end

    bcdata = BCData{Nc}(mapP,mapI,mapO,Ival)

    return bcdata
end

function initial_condition(param,x,y)
    # TODO: use getter
    t0 = param.timestepping_param.t0
    return primitive_to_conservative(param.equation,SVector(exact_sol(param.equation,x,y,t0)))
end

γ = 5/3
param = Param(N=3, K=(100,100), xL=(-0.5,-0.5), xR=(0.5,0.5),
              global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
              timestepping_param=TimesteppingParameter(T=1e-3, CFL=0.5, dt0=1e-6, t0=0.0),
              limiting_param=LimitingParameter(ζ=0.5, η=0.1),
              postprocessing_param=PostprocessingParameter(output_interval=200),
              equation=CompressibleEulerIdealGas{Dim2}(γ),
              rhs_type=ESLimitedLowOrderPos(low_order_surface_flux_type=LaxFriedrichsOnNodalVal(),
                                            high_order_surface_flux_type=LaxFriedrichsOnProjectedVal()),
              approximation_basis_type=GaussCollocation(),
              entropyproj_limiter_type=NodewiseScaledExtrapolation(),
              rhs_limiter_type=SubcellLimiter(bound_type=PositivityAndMinEntropyBound(),
                                              shockcapture_type=NoShockCapture()))

T = param.timestepping_param.T
N = param.N
K = param.K
equation = param.equation

rd,md,discrete_data,bcdata,prealloc,caches = initialize_DG(param,initial_condition,initial_boundary_conditions)

data_hist = SSP33!(param,discrete_data,bcdata,prealloc,caches)

err_data = calculate_error(prealloc.Uq,param,discrete_data,md,prealloc,exact_sol)

construct_vtk_file!(caches.postprocessing_cache,param,data_hist,"/data/yl184/outputs/figures/astro-jet","astro-jet")