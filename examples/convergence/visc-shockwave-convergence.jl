using Revise
using StaticArrays
using DataFrames
using StartUpDG

using P2DE

using JLD2

# Becker viscous shocktube
const γ = 1.4
const M_0 = 20.0
const mu = 0.001
const Re = 1/mu
const lambda = 2/3*mu
const Pr = 3/4
const cp = γ/(γ-1)
const cv = 1/(γ-1)
const kappa = mu*cp/Pr

const v_inf = 0.2
const rho_0 = 1.0
const v_0 = 1.0
const m_0 = rho_0*v_0
const v_1 = (γ-1+2/M_0^2)/(γ+1)
const v_01 = sqrt(v_0*v_1)

const uL = v_0+v_inf
const uR = v_1+v_inf
const rhoL = m_0/v_0
const rhoR = m_0/v_1
const eL = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-v_0^2)
const eR = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-v_1^2)
const pL = (γ-1)*rhoL*eL
const pR = (γ-1)*rhoR*eR
const EL = pL/(γ-1)+0.5*rhoL*uL^2
const ER = pR/(γ-1)+0.5*rhoR*uR^2


function bisection_solve_velocity(x,max_iter,tol)
    v_L = v_1
    v_R = v_0
    num_iter = 0

    L_k = kappa/m_0/cv
    f(v) = -x+2*L_k/(γ+1)*(v_0/(v_0-v_1)*log((v_0-v)/(v_0-v_01))-v_1/(v_0-v_1)*log((v-v_1)/(v_01-v_1)))

    v_new = (v_L+v_R)/2
    while num_iter < max_iter
        v_new = (v_L+v_R)/2

        if abs(f(v_new)) < tol
            return v_new
        elseif sign(f(v_L)) == sign(f(v_new))
            v_L = v_new
        else
            v_R = v_new
        end
        num_iter += 1
    end

    return v_new
end

const max_iter = 10000
const tol = 1e-14

function exact_sol(eqn,x,t)
    u   = bisection_solve_velocity(x-v_inf*t,max_iter,tol)
    rho = m_0/u
    e   = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-u^2)
    return rho, v_inf+u, pfun(eqn,SVector(rho, rho*(v_inf+u), rho*(e+1/2*(v_inf+u)^2)))
end

function initial_boundary_conditions(param,md)
    @unpack K,equation = param
    @unpack mapP       = md

    mapI = [1]
    mapO = [2*K]
 
    inflowarr = [primitive_to_conservative(equation, (rhoL,uL,pL))]

    bcdata = BCData(mapP,mapI,mapO,inflowarr)

    return bcdata
end

function initial_condition(param,x)
    # TODO: use getter
    t0 = param.timestepping_param.t0
    return primitive_to_conservative(param.equation,SVector{3,Float64}(exact_sol(param.equation,x,t0)))
end

# TODO: refactor convergence
jld_path = "/data/yl184/outputs/jld2/visc-shockwave.jld2"

# for N = [2;3]
# for K = [50;100;200;400;800;1600]
for limiter_type in [ZhangShuLimiter();SubcellLimiter(bound_type=PositivityAndCellEntropyBound());SubcellLimiter(bound_type=PositivityBound())]
for N = [3]
for K = [50;100;200;400;800;1600]

CFL = 0.5
param = Param(N=N, K=K, xL=-1.0, xR=1.5,
              global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
              timestepping_param=TimesteppingParameter(T=1.0, CFL=CFL, dt0=1e-3, t0=0.0),
              limiting_param=LimitingParameter(ζ=0.1, η=0.1),
              postprocessing_param=PostprocessingParameter(output_interval=1000),
              equation=CompressibleNavierStokesIdealGas{Dim1}(γ,Re,mu,lambda,Pr,cp,cv,kappa),
              rhs_type=StdDGLimitedLowOrderPos(high_order_surface_flux_type=LaxFriedrichsOnProjectedVal(),
                                               low_order_surface_flux_type=LaxFriedrichsOnNodalVal()),
              approximation_basis_type=LobattoCollocation(),
              entropyproj_limiter_type=NoEntropyProjectionLimiter(),
              rhs_limiter_type=limiter_type)

T = param.timestepping_param.T
N = param.N
K = param.K
equation = param.equation

rd,md,discrete_data,bcdata,prealloc,caches = initialize_DG(param,initial_condition,initial_boundary_conditions)

data_hist = SSP33!(param,discrete_data,bcdata,prealloc,caches)

err_data = calculate_error(prealloc.Uq,param,discrete_data,md,prealloc,exact_sol)

plot_path     = "/home/yl184/P2DE.jl/outputs/figures/visc-shockwave/N=$N,K=$K,CFL=$CFL,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.rhs_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η).png"
plotzoom_path = "/home/yl184/P2DE.jl/outputs/figures/visc-shockwave/N=$N,K=$K,CFL=$CFL,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.rhs_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η),zoom.png"
gif_path      = "/home/yl184/P2DE.jl/outputs/figures/visc-shockwave/N=$N,K=$K,CFL=$CFL,rhs=$(param.rhs_type),vproj=$(param.entropyproj_limiter_type),pos=$(param.rhs_limiter_type),ZETA=$(param.limiting_param.ζ),ETA=$(param.limiting_param.η),zoom.gif"

plot_component(param,discrete_data,md,prealloc,
               [u[1] for u in prealloc.Uq],1,K,0,6.0,
               plot_path,
               true,md.xq,[exact_sol(equation,xi,T)[1] for xi in md.xq],1,K)
plot_component(param,discrete_data,md,prealloc,
               [u[1] for u in prealloc.Uq],Int64(round(0.7*K)),Int64(round(0.9*K)),0,6.0,
               plotzoom_path,
               true,md.xq,[exact_sol(equation,xi,T)[1] for xi in md.xq],Int64(round(0.7*K)),Int64(round(0.9*K)))

plot_rho_animation(md,param,prealloc,data_hist,data_hist.θhist,0,6.0,
                   gif_path)

df = DataFrame([name => [] for name in (fieldnames(Param)..., fieldnames(ErrorData)...,:data_history)])
write_to_jld2(param,data_hist,err_data,df,jld_path)

end
end
end

df = load(jld_path,"data")
visualize_error_data(df)



# using Revise
# using StaticArrays
# using DataFrames
# using StartUpDG

# using P2DE

# using JLD2

# # Becker viscous shocktube
# const γ = 1.4
# const M_0 = 20.0
# const mu = 0.001
# const Re = 1/mu
# const lambda = 2/3*mu
# const Pr = 3/4
# const cp = γ/(γ-1)
# const cv = 1/(γ-1)
# const kappa = mu*cp/Pr

# const v_inf = 0.2
# const rho_0 = 1.0
# const v_0 = 1.0
# const m_0 = rho_0*v_0
# const v_1 = (γ-1+2/M_0^2)/(γ+1)
# const v_01 = sqrt(v_0*v_1)

# # const uL = v_0+v_inf
# # const uR = v_1+v_inf
# # const vL = 0.0
# # const vR = 0.0
# const vL = v_0+v_inf
# const vR = v_1+v_inf
# const uL = 0.0
# const uR = 0.0

# const rhoL = m_0/v_0
# const rhoR = m_0/v_1
# const eL = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-v_0^2)
# const eR = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-v_1^2)
# const pL = (γ-1)*rhoL*eL
# const pR = (γ-1)*rhoR*eR
# const EL = pL/(γ-1)+0.5*rhoL*uL^2
# const ER = pR/(γ-1)+0.5*rhoR*uR^2


# function bisection_solve_velocity(x,max_iter,tol)
#     v_L = v_1
#     v_R = v_0
#     num_iter = 0

#     L_k = kappa/m_0/cv
#     f(v) = -x+2*L_k/(γ+1)*(v_0/(v_0-v_1)*log((v_0-v)/(v_0-v_01))-v_1/(v_0-v_1)*log((v-v_1)/(v_01-v_1)))

#     v_new = (v_L+v_R)/2
#     while num_iter < max_iter
#         v_new = (v_L+v_R)/2

#         if abs(f(v_new)) < tol
#             return v_new
#         elseif sign(f(v_L)) == sign(f(v_new))
#             v_L = v_new
#         else
#             v_R = v_new
#         end
#         num_iter += 1
#     end

#     return v_new
# end

# const max_iter = 10000
# const tol = 1e-14

# function exact_sol(eqn,x,y,t)
#     # u   = bisection_solve_velocity(x-v_inf*t,max_iter,tol)
#     # rho = m_0/u
#     # e   = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-u^2)
#     # return rho, v_inf+u, 0.0, pfun(eqn,SVector(rho, rho*(v_inf+u), 0.0, rho*(e+1/2*(v_inf+u)^2)))
#     u   = bisection_solve_velocity(y-v_inf*t,max_iter,tol)
#     rho = m_0/u
#     e   = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-u^2)
#     return rho, 0.0, v_inf+u, pfun(eqn,SVector(rho, 0.0, rho*(v_inf+u), rho*(e+1/2*(v_inf+u)^2)))
# end

# function initial_boundary_conditions(param,md)
#     @unpack K,equation = param
#     @unpack mapP,xyzf  = md
#     xf = xyzf[1]
#     yf = xyzf[2]
#     Nfp = size(xf,1)

#     mapI = Int64[]
#     mapO = Int64[]
#     inflowarr = SVector{4,Float64}[]

#     inflow_val = primitive_to_conservative(equation, (rhoL,uL,vL,pL))
#     for k in 1:get_num_elements(param)
#         for i in 1:Nfp
#             idx = i+Nfp*(k-1)
#             # if (abs(xf[i,k]+1.0) <= 1e-14)
#             if (abs(yf[i,k]+1.0) <= 1e-14)
#                 push!(mapI,idx)
#                 push!(inflowarr, inflow_val)
#             # elseif (abs(xf[i,k]-1.5) <= 1e-14)
#             elseif (abs(yf[i,k]-1.5) <= 1e-14)
#                 push!(mapO,idx)
#             end
#         end
#     end

#     bcdata = BCData(mapP,mapI,mapO,inflowarr)

#     return bcdata
# end

# function initial_condition(param,x,y)
#     # TODO: use getter
#     t0 = param.timestepping_param.t0
#     return primitive_to_conservative(param.equation,SVector{4,Float64}(exact_sol(param.equation,x,y,t0)))
# end

# # TODO: refactor convergence
# jld_path = "/data/yl184/outputs/jld2/visc-shockwave.jld2"

# for N = [3]
# # for K in [(50,1),(100,1),(200,1),(400,1),(800,1),(1600,1)]
# for K in [(1,50),(1,100),(1,200),(1,400),(1,800),(1,1600)]
# # for K in [(1,50)]

# CFL = 0.5
# # param = Param(N=N, K=K, xL=(-1.0,-0.25), xR=(1.5,0.25),
# param = Param(N=N, K=K, xL=(-0.25,-1.0), xR=(0.25,1.5),
#               global_constants=GlobalConstant(POSTOL=1e-14, ZEROTOL=5e-16),
#               timestepping_param=TimesteppingParameter(T=1.0, CFL=CFL, dt0=1e-3, t0=0.0),
#               limiting_param=LimitingParameter(ζ=0.1, η=0.1),
#               postprocessing_param=PostprocessingParameter(output_interval=1000),
#               equation=CompressibleNavierStokesIdealGas{Dim2}(γ,Re,mu,lambda,Pr,cp,cv,kappa),
#               rhs_type=ESLimitedLowOrderPos(high_order_surface_flux_type=LaxFriedrichsOnProjectedVal(),
#                                             low_order_surface_flux_type=LaxFriedrichsOnNodalVal()),
#             #   rhs_type=LowOrderPositivity(surface_flux_type=LaxFriedrichsOnNodalVal()),
#               approximation_basis_type=LobattoCollocation(),
#               entropyproj_limiter_type=NoEntropyProjectionLimiter(),
#               rhs_limiter_type=SubcellLimiter(bound_type=PositivityAndCellEntropyBound()))
#             #   rhs_limiter_type=ZhangShuLimiter())
#             #   rhs_limiter_type=NoRHSLimiter())

# T = param.timestepping_param.T
# N = param.N
# K = param.K
# equation = param.equation

# rd,md,discrete_data,bcdata,prealloc,caches = initialize_DG(param,initial_condition,initial_boundary_conditions)

# data_hist = SSP33!(param,discrete_data,bcdata,prealloc,caches)

# err_data = calculate_error(prealloc.Uq,param,discrete_data,md,prealloc,exact_sol)

# df = DataFrame([name => [] for name in (fieldnames(Param)..., fieldnames(ErrorData)...,:data_history)])
# write_to_jld2(param,data_hist,err_data,df,jld_path)

# end
# end

# df = load(jld_path,"data")
# visualize_error_data(df)