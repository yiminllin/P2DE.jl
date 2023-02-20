function reference_to_physical(U,GJ,dim::Dim1)
    Ur  = U[1]
    rxJ = GJ[1]
    return SVector(rxJ*Ur, )
end

function reference_to_physical(U,GJ,dim::Dim2)
    Ur,Us           = U
    rxJ,sxJ,ryJ,syJ = GJ
    return SVector(rxJ*Ur+sxJ*Us, ryJ*Ur+syJ*Us)
end

function get_Bx(i,k,discrete_data,dim::Dim1)
    @unpack Br   = discrete_data.ops
    @unpack rxJh = discrete_data.geom

    iface = i+discrete_data.sizes.Nq
    return reference_to_physical((Br[i,i],), (rxJh[iface,k],), dim)
end

function get_Bx(i,k,discrete_data,dim::Dim2)
    @unpack Br,Bs               = discrete_data.ops
    @unpack rxJh,ryJh,sxJh,syJh = discrete_data.geom

    iface = i+discrete_data.sizes.Nq
    return reference_to_physical((Br[i,i],Bs[i,i]), (rxJh[iface,k],sxJh[iface,k],ryJh[iface,k],syJh[iface,k]), dim)
end

function get_Bx_with_n(i,k,discrete_data,dim::Dim1)
    Bx_i = get_Bx(i,k,discrete_data,dim)
    return Bx_i,abs(Bx_i[1])
end

function get_Bx_with_n(i,k,discrete_data,dim::Dim2)
    Bx_i,By_i = get_Bx(i,k,discrete_data,dim)
    return SVector(Bx_i,By_i),norm((Bx_i,By_i))
end

function get_Sx(i,j,k,discrete_data,dim::Dim1)
    @unpack Srh_db = discrete_data.ops
    @unpack rxJh   = discrete_data.geom

    return reference_to_physical((Srh_db[i,j],), (rxJh[i,k],), dim)
end

function get_Sx(i,j,k,discrete_data,dim::Dim2)
    @unpack Srh_db,Ssh_db       = discrete_data.ops
    @unpack rxJh,ryJh,sxJh,syJh = discrete_data.geom

    return reference_to_physical((Srh_db[i,j],Ssh_db[i,j]), (rxJh[i,k],sxJh[i,k],ryJh[i,k],syJh[i,k]), dim)
end

function get_Sx0(i,j,k,discrete_data,dim::Dim1)
    @unpack Sr0  = discrete_data.ops
    @unpack rxJh = discrete_data.geom

    return reference_to_physical((Sr0[i,j],), (rxJh[i,k],), dim)
end

function get_Sx0(i,j,k,discrete_data,dim::Dim2)
    @unpack Sr0,Ss0             = discrete_data.ops
    @unpack rxJh,ryJh,sxJh,syJh = discrete_data.geom

    return reference_to_physical((Sr0[i,j],Ss0[i,j]), (rxJh[i,k],sxJh[i,k],ryJh[i,k],syJh[i,k]), dim)
end

function get_Sx0_with_n(i,j,k,discrete_data,dim::Dim1)
    Sx0_ij = get_Sx0(i,j,k,discrete_data,dim)
    return (Sx0_ij),abs(Sx0_ij[1])
end

function get_Sx0_with_n(i,j,k,discrete_data,dim::Dim2)
    Sx0_ij,Sy0_ij = get_Sx0(i,j,k,discrete_data,dim)
    return SVector(Sx0_ij,Sy0_ij),norm((Sx0_ij,Sy0_ij))
end

# TODO: hardcoded
function apply_LF_dissipation_to_flux(flux_f,param,i,k,lf,dim::Dim1)
    flux_f[i,k] = SVector{1}(flux_f[i,k][1]-lf)
end

function apply_LF_dissipation_to_flux(flux_f,param,i,k,lf,dim::Dim2)
    N1D = param.N+1
    if i <= 2*N1D
        flux_f[i,k] = SVector(flux_f[i,k][1]-lf,flux_f[i,k][2])
    else
        flux_f[i,k] = SVector(flux_f[i,k][1],flux_f[i,k][2]-lf)
    end
end
