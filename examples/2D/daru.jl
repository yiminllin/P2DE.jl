using Revise
using StaticArrays
using DataFrames
using StartUpDG

using P2DE

using JLD2

const γ = 1.4
const mu = 0.001
const Re = 1/mu
const lambda = 2/3*mu
const Pr = 0.73
const cp = γ/(γ-1)
const cv = 1/(γ-1)
const kappa = mu*cp/Pr

function exact_sol(eqn,x,y,t)
    if (x <= 0.5)
        rho0 = 120.0
        u0   = 0.0
        v0   = 0.0
        p0   = 120.0/γ
    else
        rho0 = 1.2
        u0   = 0.0
        v0   = 0.0
        p0   = 1.2/γ
    end

    return rho0, u0, v0, p0
end

function initial_boundary_conditions(param,md)
    @unpack K,N,equation = param
    @unpack mapP,xyzf,nxyz = md
    xf = xyzf[1]
    yf = xyzf[2]
    Nfp = size(xf,1)

    mapI = Int64[]
    mapO = Int64[]
    mapWslip = Int64[]
    mapWnoslip = Int64[]
    inflowarr = SVector{4,Float64}[]

    for k in 1:get_num_elements(param)
        for i in 1:Nfp
            idx = i+Nfp*(k-1)
            if (abs(xf[i,k]) <= 1e-14 && i <= 2*(N+1))
                push!(mapWnoslip,idx)
            elseif (abs(xf[i,k]-1.0) <= 1e-14 && i <= 2*(N+1))
                push!(mapWnoslip,idx)
            elseif (abs(yf[i,k]) <= 1e-12 && i > 2*(N+1))
                push!(mapWnoslip,idx)
            elseif (abs(yf[i,k]-0.5) <= 1e-12 && i > 2*(N+1))
                push!(mapWslip,idx)
            end
        end
    end

    bcdata = BCData{2,4}(mapP,mapI,mapO,mapWslip,mapWnoslip,inflowarr,nxyz)

    return bcdata
end

function initial_condition(param,x,y)
    # TODO: use getter
    t0 = param.timestepping_param.t0
    return primitive_to_conservative(param.equation,SVector{4,Float64}(exact_sol(param.equation,x,y,t0)))
end

CFL = 0.5
# K1D need to be multiple of 2
param = Param(N=3, K=(300,150), xL=(0.0,0.0), xR=(1.0,0.5),
              global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
              timestepping_param=TimesteppingParameter(T=1.0, CFL=CFL, dt0=1e-3, t0=0.0),
              limiting_param=LimitingParameter(ζ=0.1, η=0.1),
              postprocessing_param=PostprocessingParameter(output_interval=100),
              equation=CompressibleNavierStokesIdealGas{Dim2}(γ,Re,mu,lambda,Pr,cp,cv,kappa),
              rhs_type=StdDGLimitedLowOrderPos(high_order_surface_flux_type=LaxFriedrichsOnProjectedVal(),
                                               low_order_surface_flux_type=LaxFriedrichsOnNodalVal()),
              approximation_basis_type=LobattoCollocation(),
              entropyproj_limiter_type=NoEntropyProjectionLimiter(),
              rhs_limiter_type=SubcellLimiter(bound_type=PositivityAndCellEntropyBound()))

T = param.timestepping_param.T
N = param.N
K = param.K
equation = param.equation

rd,md,discrete_data,bcdata,prealloc,caches = initialize_DG(param,initial_condition,initial_boundary_conditions)

data_hist = SSP33!(param,discrete_data,bcdata,prealloc,caches)

@show "hello"
construct_vtk_file!(caches.postprocessing_cache,md,rd,param,data_hist,"/data/yl184/outputs/figures/daru","daru")
