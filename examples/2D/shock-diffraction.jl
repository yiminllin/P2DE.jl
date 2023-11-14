using Revise
using StaticArrays
using DataFrames
using StartUpDG

using P2DE

using JLD2

const Ma = 5.09
const γ = 1.4
const mu = 0.001
const Re = 1/mu
const lambda = 2/3*mu
const Pr = 3/4
const cp = γ/(γ-1)
const cv = 1/(γ-1)
const kappa = mu*cp/Pr

const rhoR = 1.4
const pR = 1.0
const rhoL = (γ+1)*Ma^2*rhoR/((γ-1)*Ma^2+2)
const pL = pR*(2*γ*Ma^2-(γ-1))/(γ+1)

function exact_sol(eqn,x,y,t)
    if (x <= 0.5)
        rho0 = rhoL
        u0   = 0.0
        v0   = 0.0
        p0   = pL
    else
        rho0 = rhoR
        u0   = 0.0
        v0   = 0.0
        p0   = pR
    end

    if (x <= 0.5 && y <= 6)
        rho0 = rhoR
        u0   = 0.0
        v0   = 0.0
        p0   = pR
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
            if (abs(xf[i,k]) <= 1e-14 && yf[i,k] >= 6.0 && i <= 2*(N+1))
                push!(mapI,idx)
                push!(inflowarr,primitive_to_conservative(equation,(rhoL,0.0,0.0,pL)))
            elseif (abs(xf[i,k]) <= 1e-14 && yf[i,k] < 6.0 && i <= 2*(N+1))
                push!(mapWslip,idx)
            elseif (abs(xf[i,k]-13.0) <= 1e-12 && i <= 2*(N+1))
                push!(mapO,idx)
            elseif (abs(yf[i,k]-11.0) <= 1e-10 && i > 2*(N+1))
                push!(mapWslip,idx)
                # push!(mapO,idx)
            elseif (abs(yf[i,k]-0.0) <= 1e-12 && i > 2*(N+1))
                push!(mapWslip,idx)
            elseif (abs(xf[i,k]-1.0) <= 1e-12 && yf[i,k] <= 6.0 && yf[i,k] >= 0.0 && i <= 2*(N+1)) 
                push!(mapWslip,idx)
            elseif (abs(yf[i,k]-6.0) <= 1e-12 && xf[i,k] < 1.0-1e-12 && xf[i,k] >= 0.0 && i > 2*(N+1)) 
                push!(mapWslip,idx)
            elseif (abs(xf[i,k]-0.0) <= 1e-12 && yf[i,k] <= 6.0 && yf[i,k] >= 0.0 && i <= 2*(N+1)) 
                push!(mapWslip,idx)
            elseif (abs(yf[i,k]-0.0) <= 1e-12 && xf[i,k] < 1.0-1e-14 && xf[i,k] >= 0.0 && i > 2*(N+1)) 
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
param = Param(N=3, K=(130,110), xL=(0.0,0.0), xR=(13.0,11.0),
              global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
              timestepping_param=TimesteppingParameter(T=4.0, CFL=CFL, dt0=1e-3, t0=0.0),
              limiting_param=LimitingParameter(ζ=0.5, η=0.5),
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

construct_vtk_file!(caches.postprocessing_cache,param,data_hist,"/data/yl184/outputs/figures/shock-diffraction","shock-diffraction")
