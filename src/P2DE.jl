module P2DE

###########################
# Load necessary packages #
###########################

# Packages language related
using Revise
using UnPack

# Performance related
using Polyester

# Packages math related
using LinearAlgebra
using StaticArrays
using SparseArrays
using StartUpDG
using NodesAndModes

# Packages for benchmark
using BenchmarkTools
using TimerOutputs

# Packages for post-processing
using Plots
using DataFrames
using JLD2
using MAT
using PrettyTables
using WriteVTK


########################
# Include source files #
########################
include("./types.jl")
include("./math/nonlinear_solvers.jl")
include("./math/CompressibleNavierStokes.jl")
include("./math/KPP.jl")
include("./dg/DGInitialize.jl")
include("./dg/DGPostProcessing.jl")
include("./dg/DGFilter.jl")
include("./dg/limiter/DGLimiter.jl")
include("./dg/limiter/DGShockCapture.jl")
include("./dg/rhs/DGRHS.jl")
include("./dg/DGUtils.jl")
include("./timestepping/TimeStepping.jl")

################
# Export types #
################
export Param,BCData,DataHistory,ErrorData
export GlobalConstant,TimesteppingParameter,PostprocessingParameter,
       LimitingParameter
export LowOrderPositivity,EntropyStable,ESLimitedLowOrderPos
export ChandrashekarOnProjectedVal,LaxFriedrichsOnNodalVal,LaxFriedrichsOnProjectedVal
export NoEntropyProjectionLimiter,NodewiseScaledExtrapolation
export NoRHSLimiter,ZhangShuLimiter,SubcellLimiter
export PositivityBound,PositivityAndMinEntropyBound,PositivityAndRelaxedMinEntropyBound
export NoShockCapture,HennemannShockCapture
export GaussCollocation,LobattoCollocation
export Dim1,Dim2,Dim3
export CompressibleEulerIdealGas,CompressibleNavierStokesIdealGas,KPP

##################
# Export methods #
##################
export get_γ
export initialize_DG
export primitive_to_conservative
export SSP33!
export calculate_error,plot_component,plot_rho_animation,write_to_jld2
export visualize_error_data
export construct_vtk_file!
export check_positivity,check_conservation
export get_num_elements,get_num_components

#####################
# TODO: for testing #
#####################
export rhs!,rhs_pos_Gauss!,rhs_modalESDG!,compute_entropyproj_limiting_param!

end
