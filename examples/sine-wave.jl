using Revise
using UnPack
using EntropyStableEuler
using StaticArrays
using DataFrames

using SimplePDE

function exact_sol(eqn,x,t)
    return 1e-6 + 1.0 + sin(2*pi*(x-t)), 1.0, 1.0
end

function initial_boundary_conditions(param,md)
    @unpack K,EQN = param
    @unpack mapP  = md

    # Make periodic
    mapP[1]   = mapP[end] 
    mapP[end] = 1

    mapI = []
    mapO = []
    inflowarr = []

    bcdata = BCData(mapP,mapI,mapO,inflowarr)

    return bcdata
end

function initial_condition(param,x)
    return primitive_to_conservative(param.EQN,SVector{3,Float64}(exact_sol(param.EQN,x,param.t0)))
end

MODIFYVF = false
RHSTYPE = 0.5
FILTERTYPE = 2.0
ZETA = 0.1
ETA  = 1.0

POSTOL = 1e-14
ZEROTOL = 5e-16
N = 3
K = 80

T = 0.1
EQN = EntropyStableEuler.Euler{1}();
DISCRETIZATION_TYPE = 2
t0  = 0.0
CFL = 0.75
dt0 = 1e-4
OUTPUT_INTERVAL = 1000

XL = 0.0
XR = 1.0

param = Param(POSTOL,ZEROTOL,ZETA,ETA,N,K,XL,XR,T,CFL,dt0,t0,OUTPUT_INTERVAL,EQN,DISCRETIZATION_TYPE,FILTERTYPE,RHSTYPE,MODIFYVF)
U,rd_gauss,md_gauss,discrete_data_gauss,rd_LGL,md_LGL,discrete_data_LGL,transfer_ops,bcdata,prealloc = initialize_DG(param,initial_condition,initial_boundary_conditions)

data_hist = SSP33!(U,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc)

err_data = calculate_error(U,param,discrete_data_gauss,discrete_data_LGL,md_gauss,md_LGL,prealloc,exact_sol)

plot_component(param,discrete_data_gauss,md_gauss,md_LGL,prealloc,
               [u[1] for u in U],1,K,0,3,"outputs/figures/sine-wave/N=$N,K=$K,RHSTYPE=$RHSTYPE,FILTERTYPE=$FILTERTYPE,ZETA=$ZETA,ETA=$ETA.png",
               true,md_gauss.xq,[exact_sol(EQN,xi,T)[1] for xi in md_gauss.xq],1,K)
plot_component(param,discrete_data_gauss,md_gauss,md_LGL,prealloc,
               [u[1] for u in U],Int64(round(K*3/4)),K,0,0.02,"outputs/figures/sine-wave/N=$N,K=$K,RHSTYPE=$RHSTYPE,FILTERTYPE=$FILTERTYPE,ZETA=$ZETA,ETA=$ETA,zoom.png",
               true,md_gauss.xq,[exact_sol(EQN,xi,T)[1] for xi in md_gauss.xq],Int64(round(K*3/4)),K)

plot_rho_animation(md_gauss,md_LGL,param,prealloc,data_hist,data_hist.Fhist,0,3,
                   "outputs/figures/sine-wave/N=$N,K=$K,RHSTYPE=$RHSTYPE,FILTERTYPE=$FILTERTYPE,ZETA=$ZETA,ETA=$ETA.gif")

df = DataFrame(param = Param[], data_hist = DataHistory[], err_data = ErrorData[])
write_to_jld2(param,data_hist,err_data,df,"outputs/jld2/sine-wave/sine-wave.jld2")