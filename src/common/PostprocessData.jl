struct DataHistory{Nc}
    Uhist::Vector{Array{SVector{Nc,Float64},2}}
    Lhist::Vector{Array{Float64,2}}
    θhist::Vector{Array{Float64,2}}
    thist::Vector{Float64}
    dthist::Vector{Float64}
end

struct ErrorData
    L1err::Float64
    L2err::Float64
    Linferr::Float64
end

