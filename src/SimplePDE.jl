module SimplePDE

###########################
# Load necessary packages #
###########################

# Packages language related
using Revise
using UnPack

# Packages math related
using LinearAlgebra
using StaticArrays
using SparseArrays
using StartUpDG
using NodesAndModes

# Packages for benchmark
using BenchmarkTools

# Packages for post-processing
using Plots
using DataFrames
using JLD2
using MAT


########################
# Include source files #
########################
include("./types.jl")
include("./math/nonlinear_solvers.jl")
include("./math/CompressibleNavierStokes.jl")
include("./dg/DGInitialize.jl")
include("./dg/DGPostProcessing.jl")
include("./dg/DGFilter.jl")
include("./dg/DGLimiter.jl")
include("./dg/DGRHS.jl")
include("./timestepping/TimeStepping.jl")

################
# Export types #
################
export Param,BCData,DataHistory,ErrorData
export GlobalConstant,TimesteppingParameter,PostprocessingParameter,
       LimitingParameter
export LowOrderPositivity,EntropyStable,ESLimitedLowOrderPos
export NoEntropyProjectionLimiter,ExponentialFilter,ZhangShuFilter,
       ElementwiseScaledExtrapolation,NodewiseScaledExtrapolation
export NoPositivityLimiter,ZhangShuLimiter,SubcellLimiter
export GaussCollocation,LobattoCollocation,HybridGaussLGL
export Dim1,Dim2,Dim3
export CompressibleEulerIdealGas,CompressibleNavierStokesIdealGas

##################
# Export methods #
##################
export initialize_DG
export primitive_to_conservative
export SSP33!
export calculate_error,plot_component,plot_rho_animation,write_to_jld2

end
