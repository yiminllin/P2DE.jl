# TODO: change file names
# TODO: preallocation to cache
function initialize_preallocations(param,md,sizes)
    @unpack Np,Nh,Nq,Nfp,Nc,Ns = sizes

    K  = get_num_elements(param)
    Nd = get_dim(param.equation)
    Uq      = zeros(SVector{Nc,Float64},Nq,K)
    vq      = zeros(SVector{Nc,Float64},Nq,K)
    v_tilde = zeros(SVector{Nc,Float64},Nh,K)
    u_tilde = zeros(SVector{Nc,Float64},Nh,K)
    beta    = zeros(Float64,Nh,K)
    rholog  = zeros(Float64,Nh,K)
    betalog = zeros(Float64,Nh,K)
    lam     = zeros(Float64,Nfp,K)
    LFc     = zeros(Float64,Nfp,K)
    rhsH    = zeros(SVector{Nc,Float64},Nq,K)
    Ui      = zeros(Float64,Nc+2)
    Uj      = zeros(Float64,Nc+2)
    QF1     = zeros(SVector{Nd,SVector{Nc,Float64}},Nh,K)
    Q0F1    = zeros(SVector{Nd,SVector{Nc,Float64}},Nq,K)
    BF1     = zeros(SVector{Nc,Float64},Nfp,K)
    uP       = zeros(SVector{Nc,Float64},Nfp,K)
    betaP    = zeros(Float64,Nfp,K)
    rhologP  = zeros(Float64,Nfp,K)
    betalogP = zeros(Float64,Nfp,K)
    flux       = zeros(SVector{Nd,SVector{Nc,Float64}},Nh,K)
    flux_H     = zeros(SVector{Nd,SVector{Nc,Float64}},Nfp,K)
    flux_L     = zeros(SVector{Nd,SVector{Nc,Float64}},Nfp,K)
    wavespeed   = zeros(Float64,Nq,Nq,K)
    wavespeed_f = zeros(Float64,Nfp,K)
    alphaarr   = zeros(Float64,Nfp,K)
    rhsL    = zeros(SVector{Nc,Float64},Nq,K)
    Larr    = zeros(Float64,K,Ns)
    L_local_arr = zeros(Float64,Nq+1,K,Ns)
    rhsU    = zeros(SVector{Nc,Float64},Nq,K)
    v3tilde   = zeros(Float64,Nh)
    rhotilde  = zeros(Float64,Nh)
    rhoetilde = zeros(Float64,Nh)
    vq_k      = zeros(SVector{Nc,Float64},Nq)
    v_tilde_k = zeros(SVector{Nc,Float64},Nh)      # TODO: refactor with v_tilde, u_tilde
    u_tilde_k = zeros(SVector{Nc,Float64},Nh)
    U_modal   = zeros(SVector{Nc,Float64},Np,K)
    U_k       = zeros(SVector{Nc,Float64},Np)
    Uq_k      = zeros(SVector{Nc,Float64},Nq)
    spatial   = zeros(SVector{Nd,SVector{Nc,Float64}},Np,K)
    boundary  = zeros(SVector{Nc,Float64},Np,K)
    resW      = zeros(SVector{Nc,Float64},Nq,K)
    resZ      = zeros(SVector{Nc,Float64},Nq,K)
    Farr      = zeros(Float64,K,Ns)            # TODO: rename F, eta to theta
    θ_local_arr = zeros(Float64,Nfp,K,Ns)
    αarr      = zeros(Float64,Nfp,K)
    resW      = zeros(SVector{Nc,Float64},Nq,K)
    resZ      = zeros(SVector{Nc,Float64},Nq,K)
    LGLind    = falses(K)                       # TODO: array of BasisType, singleton type
    L_G2L_arr = ones(Float64,K,Ns)
    L_L2G_arr = ones(Float64,K,Ns)
    L_Vf_arr  = ones(Float64,K,Ns)              # TODO: Refactor with Farr
    VhPq_new  = zeros(Float64,Nh,Nq)            # TODO: unnecessary?
    VhPq_new[1:Nq,1:Nq] = diagm(ones(Nq))
    Vf_new      = zeros(Float64,Nfp,Nq)
    VhT_new     = zeros(Float64,Np,Nh)
    MinvVhT_new = zeros(Float64,Np,Nh)
    uL_k      = zeros(SVector{Nc,Float64},Nq)
    P_k       = zeros(SVector{Nc,Float64},Nq)
    N1D = Nd == 1 ? 1 : param.N+1
    f_bar_H   = tuple([zeros(SVector{Nc,Float64},Nq+N1D,K) for _ in 1:Nd]...)
    f_bar_L   = tuple([zeros(SVector{Nc,Float64},Nq+N1D,K) for _ in 1:Nd]...)
    f_bar_lim = tuple([zeros(SVector{Nc,Float64},Nq+N1D,K) for _ in 1:Nd]...)  # TODO: unnecessary
    Uf        = zeros(SVector{Nc,Float64},Nfp,K)
    VUf       = zeros(SVector{Nc,Float64},Nfp,K)
    rhoef     = zeros(Float64,Nfp,K)
    λarr      = zeros(Float64,Nq,Nq,K)
    λBarr     = zeros(Float64,Nfp,K)

    prealloc = Preallocation{Nc,Nd}(Uq,vq,v_tilde,u_tilde,beta,rholog,betalog,lam,LFc,rhsH,Ui,Uj,QF1,Q0F1,BF1,uP,betaP,rhologP,betalogP,flux,flux_H,flux_L,wavespeed,wavespeed_f,alphaarr,rhsL,Larr,L_local_arr,rhsU,v3tilde,rhotilde,rhoetilde,vq_k,v_tilde_k,u_tilde_k,U_modal,U_k,Uq_k,spatial,boundary,resW,resZ,Farr,θ_local_arr,αarr,LGLind,L_G2L_arr,L_L2G_arr,L_Vf_arr,VhPq_new,Vf_new,VhT_new,MinvVhT_new,uL_k,P_k,f_bar_H,f_bar_L,f_bar_lim,Uf,VUf,rhoef,λarr,λBarr)
    return prealloc
end

function initialize_DG(param,initial_condition,initial_boundary_conditions)
    rd_gauss,md_gauss,discrete_data_gauss,rd_LGL,md_LGL,discrete_data_LGL,transfer_ops = initialize_data(param)

    @unpack sizes = discrete_data_gauss 
    bcdata = initial_boundary_conditions(param,md_gauss)
    prealloc = initialize_preallocations(param,md_gauss,sizes)

    init_U!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,md_gauss,md_LGL,prealloc,initial_condition)

    return rd_gauss,md_gauss,discrete_data_gauss,rd_LGL,md_LGL,discrete_data_LGL,transfer_ops,bcdata,prealloc
end

function initialize_data(param)
    @unpack N,equation = param

    rd_gauss,rd_LGL              = initialize_reference_data(N,equation)
    md_gauss,discrete_data_gauss = initialize_operators(param,rd_gauss,GaussQuadrature())
    md_LGL  ,discrete_data_LGL   = initialize_operators(param,rd_LGL,LobattoQuadrature())
    transfer_ops                 = initialize_transfer_operators(rd_gauss.element_type,param,rd_gauss,rd_LGL)

    return rd_gauss,md_gauss,discrete_data_gauss,rd_LGL,md_LGL,discrete_data_LGL,transfer_ops
end

function initialize_reference_data(N,equation::EquationType{Dim1})
    element_type = Line()
    rd_gauss = construct_gauss_reference_data(RefElemData(element_type,N,quad_rule_vol=gauss_quad(0,0,N)))
    rd_LGL   = RefElemData(element_type,N,quad_rule_vol=gauss_lobatto_quad(0,0,N))

    return rd_gauss,rd_LGL
end

function initialize_reference_data(N,equation::EquationType{Dim2})
    element_type = Quad()

    # create degree N tensor product Gauss quadrature rule
    r1D,w1D = gauss_quad(0, 0, N)
    sq,rq = vec.(StartUpDG.meshgrid(r1D))
    ws,wr = vec.(StartUpDG.meshgrid(w1D))
    wq = @. wr*ws

    rd_gauss = construct_gauss_reference_data(RefElemData(element_type,N,quad_rule_vol=(rq,sq,wq),quad_rule_face=(r1D,w1D)))
    rd_LGL   = RefElemData(element_type,SBP(),N)

    return rd_gauss,rd_LGL
end

# Construct Gauss collocation reference element data from the reference element
# data with Lagrangian basis on LGL node and Gauss quadrature
function construct_gauss_reference_data(rd)
    gauss_to_lobatto = rd.Pq
    lobatto_to_gauss = inv(gauss_to_lobatto)
    # Set Gauss RefElemData to be defined on Lagrangian basis on Gauss nodes
    # instead of LGL nodes
    # TODO: fv, Fmask, should not be used
    # TODO: sparsify
    rst  = rd.rstq 
    V1   = lobatto_to_gauss*rd.V1
    VDM  = lobatto_to_gauss*rd.VDM
    Vp   = rd.Vp*gauss_to_lobatto
    Vq   = I         # rd.Vq*gauss_to_lobatto
    Vf   = rd.Vf*gauss_to_lobatto
    M    = gauss_to_lobatto'*rd.M*gauss_to_lobatto
    Pq   = I         # lobatto_to_gauss*rd.Pq
    Dr   = lobatto_to_gauss*rd.Dr*gauss_to_lobatto
    Drst = (x->lobatto_to_gauss*x*gauss_to_lobatto).(rd.Drst)
    LIFT = rd.M\(rd.Vf'*diagm(rd.wf))

    return RefElemData(rd.element_type, Polynomial, rd.N, rd.fv, V1,
                       rst, VDM, rd.Fmask,
                       rd.Nplot, rd.rstp, Vp,
                       rd.rstq, rd.wq, Vq,
                       rd.rstf, rd.wf, Vf, rd.nrstJ,
                       M, Pq, Drst, LIFT)
end

# TODO: specialize for 1D
function initialize_operators(param,rd,quad_type)
    @unpack N = param
    ZEROTOL = param.global_constants.ZEROTOL

    # TODO: Assume uniform mesh
    # Construct mesh
    md = initialize_uniform_mesh_data(param,rd,rd.element_type)
    @unpack x,xq,rxJ,nxJ,J,mapP = md
    @unpack r,rq,wq,wf,M,Pq,Vq,Vf,LIFT,nrstJ,Drst,VDM = rd
    Vf = Matrix(Vf)

    # Construct geometric factors
    Jq,GJh = get_geometric_factors(param,rd,md,rd.element_type)

    # Construct hybridized SBP operators
    Vh   = [Vq;Vf]
    Qrs = (A->Matrix(droptol!(sparse(Pq'*M*A*Pq),ZEROTOL))).(Drst)
    Ef = Vf*Pq
    Brs = (A->Matrix(droptol!(sparse(diagm(wf.*A)),ZEROTOL))).(nrstJ)
    Qrsh = (QB->Matrix(droptol!(sparse(.5*[QB[1]-QB[1]' Ef'*QB[2];
                                          -QB[2]*Ef     QB[2]     ]),ZEROTOL))).(zip(Qrs,Brs))
    Srsh = (A->.5*(A-A')).(Qrsh)
    Srsh_db = Tuple((A->2*A).(Srsh))

    # Define sizes
    Ns  = 3   # TODO: define get_num_stage() for RK time stepper
    Nc  = get_num_components(param.equation)
    Np  = size(VDM,2)
    Nq  = length(wq)
    Nfp = size(Vf,1) 
    Nh  = Nq+Nfp

    # Define operators
    Vq,Pq = (A->typeof(A)<:UniformScaling ? diagm(ones(Nq)) : A).((Vq,Pq))  # TODO: bad runtime type check...
    M,Vq,Vh,Pq = (A->Matrix(droptol!(sparse(A),ZEROTOL))).((M,Vq,Vh,Pq)) 
    MinvVhT  = M\transpose(Vh)
    MinvVfT  = M\transpose(Vf)
    VDMinvPq = VDM\Pq
    VqVDM    = Vq*VDM
    VhPq     = Vh*Pq

    # Low order operators
    Srs0,Vf_low = get_low_order_operators(param,rd,rd.element_type,quad_type)
    Vf_low = Matrix(Vf_low)  # TODO: for type stability...
    # TODO: for LGL, there is value close to 1.0 but not exactly 1.0, hardcode for now...
    for i = 1:Nfp
        for j = 1:Nq
            abs(Vf_low[i,j]-1.0) < param.global_constants.ZEROTOL ? Vf_low[i,j] = 1.0 : nothing
        end
    end

    sizes = SizeData(Nc,Np,Nq,Nfp,Nh,Ns)
    geom  = GeomData(J,Jq,GJh)
    ops   = Operators(Srsh_db,Srs0,Brs,Vh,MinvVhT,inv(VDM),VDMinvPq,VqVDM,VhPq,Vq,Vf,Vf_low,Pq,MinvVfT,wq)
    discrete_data = DiscretizationData(sizes,geom,ops)

    return md,discrete_data
end

function initialize_uniform_mesh_data(param,rd,element_type::Line)
    @unpack xL,xR,K = param

    VXYZ,EToV = uniform_mesh(rd.element_type,K)
    VX = VXYZ[1]
    @. VX = (VX+1.0)/2.0*(xR-xL)+xL
    md = MeshData(VX,EToV,rd)

    return md
end

function initialize_uniform_mesh_data(param,rd,element_type::Quad)
    @unpack xL,xR,K = param

    VXYZ,EToV = uniform_mesh(rd.element_type,K[1],K[2])
    VX = VXYZ[1]
    VY = VXYZ[2]
    @. VX = (VX+1.0)/2.0*(xR[1]-xL[1]).+xL[1]
    @. VY = (VY+1.0)/2.0*(xR[2]-xL[2]).+xL[2]
    md = MeshData((VX,VY),EToV,rd)

    return md
end

function get_geometric_factors(param,rd,md,element_type::Line)
    @unpack Vq,Vf = rd
    @unpack J,rxJ = md

    Vh   = [Vq;Vf]
    Jq   = Vq*J
    rxJh = Vh*rxJ

    return Jq,(rxJh,)
end

function get_geometric_factors(param,rd,md,element_type::Quad)
    @unpack Vq,Vf = rd
    @unpack J,rxJ,sxJ,ryJ,syJ = md

    Vh = [Vq;Vf]
    Jq = Vq*J
    rxJh,sxJh,ryJh,syJh = (x->Vh*x).((rxJ,sxJ,ryJ,syJ))

    return Jq,(rxJh,sxJh,ryJh,syJh)
end

function construct_low_order_operators_1D(param)
    N = param.N
    ZEROTOL = param.global_constants.ZEROTOL
    Nq = N+1

    # low order operator
    d  = zeros(Float64,Nq)
    dl = -0.5*ones(Float64,Nq-1) 
    du =  0.5*ones(Float64,Nq-1) 
    d[1]   = -0.5
    d[end] =  0.5
    Qr0 = droptol!(sparse(Tridiagonal(dl,d,du)),ZEROTOL)
    Sr0 = droptol!(sparse(.5*(Qr0-Qr0')),ZEROTOL)

    Vf_low      = spzeros(2,Nq)
    Vf_low[1]   = 1.0
    Vf_low[end] = 1.0

    return Qr0,Sr0,Vf_low
end

function get_low_order_operators(param,rd,element_type::Line,quad_type)
    _,Sr0,Vf_low = construct_low_order_operators_1D(param) 
    return (Sr0,),Vf_low
end

function get_low_order_operators(param,rd,element_type::Quad,quad_type::GaussQuadrature)
    _,w1D = gauss_quad(0,0,param.N)
    return get_low_order_operators(param,rd,element_type,quad_type,w1D)
end

function get_low_order_operators(param,rd,element_type::Quad,quad_type::LobattoQuadrature)
    _,w1D = gauss_lobatto_quad(0,0,param.N)
    return get_low_order_operators(param,rd,element_type,quad_type,w1D)
end

function get_low_order_operators(param,rd,element_type::Quad,quad_type,w1D)
    ZEROTOL = param.global_constants.ZEROTOL
    M1D = diagm(w1D)
    Q01D,_,_ = construct_low_order_operators_1D(param)
    Qr0 = droptol!(sparse(kron(M1D,Q01D)),ZEROTOL)
    Qs0 = droptol!(sparse(kron(Q01D,M1D)),ZEROTOL)
    Sr0 = droptol!(sparse(.5*(Qr0-Qr0')),ZEROTOL)
    Ss0 = droptol!(sparse(.5*(Qs0-Qs0')),ZEROTOL)
    Vf_low = get_low_order_extrapolation(param,rd,element_type,quad_type)
    return (Sr0,Ss0),Vf_low
end

function get_low_order_extrapolation(param,rd,element_type::Quad,quad_type::GaussQuadrature)
    N    = param.N
    Nq1D = N+1
    Nq   = (N+1)*(N+1)
    Nfp  = 4*(N+1)
    Is = collect(1:Nfp)
    Js = collect([1:Nq1D:((Nq1D-1)*Nq1D+1);
                  Nq1D:Nq1D:Nq1D*Nq1D;
                  1:Nq1D;
                  ((Nq1D-1)*Nq1D+1):Nq1D*Nq1D])
    Vs = ones(Nfp)
    return sparse(Is,Js,Vs,Nfp,Nq)
end

function get_low_order_extrapolation(param,rd,element_type::Quad,quad_type::LobattoQuadrature)
    return droptol!(sparse(rd.Vf),param.global_constants.ZEROTOL)
end

function initialize_transfer_operators(elem_type,param,rd_gauss,rd_LGL)
    @unpack N = param

    T_g2l = vandermonde(elem_type,N,rd_LGL.rst...)/vandermonde(elem_type,N,rd_gauss.rst...)
    T_l2g = vandermonde(elem_type,N,rd_gauss.rst...)/vandermonde(elem_type,N,rd_LGL.rst...)
    transfer_ops = TransferOperators(T_g2l,T_l2g)

    return transfer_ops
end

function init_U!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,md_gauss,md_LGL,prealloc,initial_condition)
    @unpack Nq = discrete_data_gauss.sizes

    K  = get_num_elements(param)
    update_indicator!(prealloc,param.approximation_basis_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,true)
    for k = 1:K
        for i = 1:Nq
            set_initial_condition!(prealloc,i,k,param,initial_condition,md_gauss,md_LGL,param.equation)
        end
    end
end

function set_initial_condition!(prealloc,i,k,param,initial_condition,md_gauss,md_LGL,equation::EquationType{Dim1})
    xqi = (prealloc.LGLind[k]) ? md_LGL.xq[i,k] : md_gauss.xq[i,k]
    prealloc.Uq[i,k] = initial_condition(param,xqi)
end

function set_initial_condition!(prealloc,i,k,param,initial_condition,md_gauss,md_LGL,equation::EquationType{Dim2})
    xqi = (prealloc.LGLind[k]) ? md_LGL.xq[i,k] : md_gauss.xq[i,k]
    yqi = (prealloc.LGLind[k]) ? md_LGL.yq[i,k] : md_gauss.yq[i,k]
    prealloc.Uq[i,k] = initial_condition(param,xqi,yqi)
end