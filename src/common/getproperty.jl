function Base.getproperty(ops::Operators{DIM}, s::Symbol) where {DIM}
    if s == :Srh_db
        return getfield(ops, :Srsh_db)[1]
    elseif s == :Ssh_db
        return getfield(ops, :Srsh_db)[2]
    elseif s == :Sr0
        return getfield(ops, :Srs0)[1]
    elseif s == :Ss0
        return getfield(ops, :Srs0)[2]
    elseif s == :Br
        return getfield(ops, :Brs)[1]
    elseif s == :Bs
        return getfield(ops, :Brs)[2]
    else
        return getfield(ops, s)
    end
end

function Base.getproperty(geom::GeomData{NGEO}, s::Symbol) where {NGEO}
    if s == :rxJh
        return getfield(geom, :GJh)[1]
    elseif s == :sxJh
        return getfield(geom, :GJh)[2]
    elseif s == :ryJh
        return getfield(geom, :GJh)[3]
    elseif s == :syJh
        return getfield(geom, :GJh)[4]
    else
        return getfield(geom, s)
    end
end

