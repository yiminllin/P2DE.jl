module P2DE

###########################
# Load necessary packages #
###########################

# Packages language related
using Revise

# Performance related
using Polyester

# Packages math related
using LinearAlgebra
using StaticArrays
using SparseArrays
using StartUpDG
using NodesAndModes

# Packages for optimization
using JuMP
using HiGHS

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
include("./common/types/Solver.jl")
include("./common/types/State.jl")
include("./common/types/StateParam.jl")
include("./common/types/TimeParam.jl")
include("./common/types/PostprocessData.jl")
include("./common/traits.jl")
include("./common/show.jl")
include("./common/getproperty.jl")
include("./math/nonlinear_solvers.jl")
include("./math/compressible_Navier_Stokes.jl")
include("./math/KPP.jl")
include("./dg/init.jl")
include("./dg/postprocess.jl")
include("./dg/filter.jl")
include("./dg/limiter/limiter.jl")
include("./dg/limiter/shock_capture.jl")
include("./dg/rhs/rhs.jl")
include("./dg/utils.jl")
include("./timestepping/SSPRK33.jl")

################
# Export types #
################
export Param, BCData, DataHistory, ErrorData
export GlobalConstant, TimesteppingParameter, PostprocessingParameter, LimitingParameter
export LowOrderPositivity, EntropyStable, StandardDG, ESLimitedLowOrderPos, StdDGLimitedLowOrderPos
export ChandrashekarOnProjectedVal, LaxFriedrichsOnNodalVal, LaxFriedrichsOnProjectedVal
export NoEntropyProjectionLimiter, NodewiseScaledExtrapolation
export NoRHSLimiter, ZhangShuLimiter, SubcellLimiter
export PositivityBound, PositivityAndMinEntropyBound, PositivityAndRelaxedMinEntropyBound
export PositivityAndCellEntropyBound, PositivityAndRelaxedCellEntropyBound
export TVDBound, TVDAndMinEntropyBound, TVDAndRelaxedMinEntropyBound
export TVDAndCellEntropyBound, TVDAndRelaxedCellEntropyBound
export NoShockCapture, HennemannShockCapture
export GaussCollocation, LobattoCollocation
export Dim1, Dim2
export CompressibleEulerIdealGas, CompressibleNavierStokesIdealGas, KPP

##################
# Export methods #
##################
export get_γ
export initialize_DG
export primitive_to_conservative
export SSP33!
export calculate_error, plot_component, plot_rho_animation, write_to_jld2
export visualize_error_data
export construct_vtk_file!
export check_positivity, check_conservation
export num_elements, num_components

#####################
# TODO: for testing #
#####################
export rhs!, rhs_pos_Gauss!, rhs_fluxdiff!, compute_entropyproj_limiting_param!

end
