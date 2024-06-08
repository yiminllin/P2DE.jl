struct BCData{Nc}
    mapP::Array{Int64,2}
    mapI::Array{Int64,1}
    mapO::Array{Int64,1}  # List of global indices with inflow and outflow
    # (do nothing) boundary conditions
    Ival::Array{SVector{Nc,Float64},1}
end

