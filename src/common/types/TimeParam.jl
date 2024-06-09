Base.@kwdef struct TimeParam
    t::Float64
    dt::Float64
    nstage::Int64
    timer::TimerOutput
end

