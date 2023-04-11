# Find l s.t. rho(UL + lP)  ∈ [Lrho, Urho]
#             rhoe(UL + lP) ∈ [Lrhoe, Urhoe]
# TODO: refactor
function get_limiting_param(rhs_limiter_type::ZhangShuLimiter,bound_type,param,UL,P,bound)
    Lrho,Lrhoe,Urho,Urhoe = bound
    l = min(1.0, get_limiting_param_bound_rho_rhoe(param,UL,P,Lrho,Lrhoe,Urho,Urhoe))
    return l
end

function get_limiting_param(rhs_limiter_type::SubcellLimiter,bound_type::Union{PositivityBound,PositivityAndCellEntropyBound},param,UL,P,bound)
    Lrho,Lrhoe,Lphi,Urho,Urhoe = bound
    l = get_limiting_param_bound_rho_rhoe(param,UL,P,Lrho,Lrhoe,Urho,Urhoe)
    return l
end

# Find l s.t. rho(UL + lP)  ∈ [Lrho, Urho]
#             rhoe(UL + lP) ∈ [Lrhoe, Urhoe]
#             phi(UL + lP)  ∈ [Lphi, inf),    phi = rhoe rho^{-\gamma}, modified specific entropy
function get_limiting_param(rhs_limiter_type::SubcellLimiter,bound_type::Union{PositivityAndMinEntropyBound,PositivityAndRelaxedMinEntropyBound},param,UL,P,bound)
    Lrho,Lrhoe,Lphi,Urho,Urhoe = bound
    lpos = get_limiting_param_bound_rho_rhoe(param,UL,P,Lrho,Lrhoe,Urho,Urhoe)
    l = get_limiting_param_bound_phi(param,UL,P,Lphi,lpos)   # TODO: assume for l \in [0,lpos] gives positive quantities
    return l
end

function get_limiting_param_bound_rho_rhoe(param,U,P,Lrho,Lrhoe,Urho,Urhoe)
    l = 1.0
    # Limit density, lower bound
    if U[1] + P[1] < Lrho
        l = max((Lrho-U[1])/P[1],0.0)
    end
    # Limit density, upper bound
    if U[1] + P[1] > Urho
        l = min(l, max((Urho-U[1])/P[1],0.0))
    end

    l = min(l,rhoe_quadratic_solve(param,U,P,Lrhoe),
              rhoe_quadratic_solve(param,U,P,Urhoe))
    return l
end

function get_limiting_param_bound_phi(param,U,P,Lphi,lpos)
    @unpack equation = param
    @unpack POSTOL   = param.global_constants

    # TODO: refactor
    f(l) = s_modified_ufun(equation,U+l*P) >= Lphi - POSTOL
    l = bisection(f,0,lpos)
    return l
end

function rhoe_quadratic_solve(param,UL,P,Lrhoe)
    @unpack ZEROTOL = param.global_constants

    dim = get_dim_type(param.equation)
    if Lrhoe == Inf
        return 1.0
    end

    # limiting internal energy (via quadratic function) lower bound
    a,b,c = get_rhoe_quadratic_coefficients(UL,P,Lrhoe,dim)

    l_eps_ij = 1.0
    if b^2-4*a*c >= 0
        r1 = (-b+sqrt(b^2-4*a*c))/(2*a)
        r2 = (-b-sqrt(b^2-4*a*c))/(2*a)
        if r1 > ZEROTOL && r2 > ZEROTOL
            l_eps_ij = min(r1,r2)
        elseif r1 > ZEROTOL && r2 < -ZEROTOL
            l_eps_ij = r1
        elseif r2 > ZEROTOL && r1 < -ZEROTOL
            l_eps_ij = r2
        end
    end

    return l_eps_ij
end

function get_rhoe_quadratic_coefficients(U,P,Lrhoe,dim::Dim1)
    a = P[1]*P[3]-1.0/2.0*P[2]^2
    b = U[3]*P[1]+U[1]*P[3]-U[2]*P[2]-P[1]*Lrhoe
    c = U[3]*U[1]-1.0/2.0*U[2]^2-U[1]*Lrhoe
    return a,b,c
end

function get_rhoe_quadratic_coefficients(U,P,Lrhoe,dim::Dim2)
    a = P[1]*P[4]-1.0/2.0*(P[2]^2+P[3]^2)
    b = U[4]*P[1]+U[1]*P[4]-U[2]*P[2]-U[3]*P[3]-P[1]*Lrhoe
    c = U[4]*U[1]-1.0/2.0*(U[2]^2+U[3]^2)-U[1]*Lrhoe
    return a,b,c
end


#############################
### Subcell limiter Utils ###
#############################
# TODO: refactor
function subcell_face_idx_to_quad_face_index_x(si,sj,k,N1D)
    iface = 0
    if (si == 1)
        iface = sj
    elseif (si == N1D+1)
        iface = sj+N1D
    end

    return iface
end

# TODO: refactor
function subcell_face_idx_to_quad_face_index_y(si,sj,k,N1D)
    iface = 0
    if (sj == 1)
        iface = si+2*N1D
    elseif (sj == N1D+1)
        iface = si+3*N1D
    end

    return iface
end

# TODO: refactor
function get_subcell_index_P_x(si,sj,k,N1Dp1,bcdata)
    @unpack mapP = bcdata
    
    N1D = N1Dp1-1
    Nfp = 4*N1D
    
    # map subcell index to face quad index
    iface = 0
    if (si == 1)
        iface = sj
    elseif (si == N1Dp1)
        iface = sj+N1D
    end

    ifaceP = mapP[iface,k]
    iP = mod1(ifaceP,Nfp)
    kP = div(ifaceP-1,Nfp)+1

    # map face quad index to subcell index
    sjP = mod1(iP,N1D)
    if div(iP-1,N1D) == 0
        siP = 1
    else
        siP = N1Dp1
    end

    return siP,sjP,kP
end

# TODO: refactor
function get_subcell_index_P_y(si,sj,k,N1Dp1,bcdata)
    @unpack mapP = bcdata

    N1D = N1Dp1-1
    Nfp = 4*N1D

    iface = 0
    if (sj == 1)
        iface = si+2*N1D
    elseif (sj == N1Dp1)
        iface = si+3*N1D
    end

    ifaceP = mapP[iface,k]
    iP = mod1(ifaceP,Nfp)
    kP = div(ifaceP-1,Nfp)+1

    # map face quad index to subcell index
    siP = mod1(iP,N1D)
    if div(iP-1,N1D) == 2
        sjP = 1
    else
        sjP = N1Dp1
    end

    return siP,sjP,kP
end

# TODO: refactor
function quad_index_to_quad_index_P(idx,k,N1D,Nfp,q2fq,fq2q,mapP,dim::Dim1)
    i = idx
    iface = q2fq[i][1]
    iP    = mod1(mapP[iface,k],Nfp)
    iP    = fq2q[iP]
    kP    = div(mapP[iface,k]-1,Nfp)+1
    return iP,kP
end

# TODO: hardcoding... Assume q2fq come in with order: first vertical face nodes
#                     then horizontal face nodes
function quad_index_to_quad_index_P(idx,k,N1D,Nfp,q2fq,direction,fq2q,mapP,dim::Dim2)
    i = idx
    iface = length(q2fq[i]) == 1 ? q2fq[i][1] : q2fq[i][direction]
    iP    = mod1(mapP[iface,k],Nfp)
    iP    = fq2q[iP]
    kP    = div(mapP[iface,k]-1,Nfp)+1
    jP    = div(iP-1,N1D)+1
    iP    = mod1(iP,N1D)
    return iP,jP,kP
end

function get_low_order_stencil(idx,k,N1D,Nfp,discrete_data,bcdata,dim::Dim1)
    @unpack q2fq,fq2q = discrete_data.ops
    @unpack mapP      = bcdata
    i = idx
    sl = i-1 >= 1   ? (i-1,k) : quad_index_to_quad_index_P(i,k,N1D,Nfp,q2fq,fq2q,mapP,dim)
    sr = i+1 <= N1D ? (i+1,k) : quad_index_to_quad_index_P(i,k,N1D,Nfp,q2fq,fq2q,mapP,dim)
    return (sl,sr)
end

function get_low_order_stencil(idx,k,N1D,Nfp,discrete_data,bcdata,dim::Dim2)
    @unpack q2fq,fq2q = discrete_data.ops
    @unpack mapP      = bcdata
    i,j = idx
    idxq = i+(j-1)*N1D 
    sl = i-1 >= 1   ? (i-1,j,k) : quad_index_to_quad_index_P(idxq,k,N1D,Nfp,q2fq,1,fq2q,mapP,dim)
    sr = i+1 <= N1D ? (i+1,j,k) : quad_index_to_quad_index_P(idxq,k,N1D,Nfp,q2fq,1,fq2q,mapP,dim)
    sb = j-1 >= 1   ? (i,j-1,k) : quad_index_to_quad_index_P(idxq,k,N1D,Nfp,q2fq,2,fq2q,mapP,dim)
    st = j+1 <= N1D ? (i,j+1,k) : quad_index_to_quad_index_P(idxq,k,N1D,Nfp,q2fq,2,fq2q,mapP,dim)
    return (sl,sr,sb,st)
end
