# TODO: change file names
# TODO: preallocation to cache
function initialize_preallocations(param,sizes)
    @unpack K                  = param
    @unpack Np,Nh,Nq,Nfp,Nc,Ns = sizes

    Uq      = zeros(SVector{3,Float64},Nq,K)
    vq      = zeros(SVector{3,Float64},Nq,K)
    v_tilde = zeros(SVector{3,Float64},Nh,K)
    u_tilde = zeros(SVector{3,Float64},Nh,K)
    beta    = zeros(Float64,Nh,K)
    rholog  = zeros(Float64,Nh,K)
    betalog = zeros(Float64,Nh,K)
    lam     = zeros(Float64,Nfp,K)
    LFc     = zeros(Float64,Nfp,K)
    rhsH    = zeros(SVector{3,Float64},Nq,K)
    Ui      = zeros(Float64,Nc+2)
    Uj      = zeros(Float64,Nc+2)
    QF1     = zeros(SVector{3,Float64},Nh,K)
    BF1     = zeros(SVector{3,Float64},Nfp,K)
    uP       = zeros(SVector{3,Float64},Nfp,K)
    betaP    = zeros(Float64,Nfp,K)
    rhologP  = zeros(Float64,Nfp,K)
    betalogP = zeros(Float64,Nfp,K)
    flux       = zeros(SVector{3,Float64},Nh,K)
    wavespeed  = zeros(Float64,Nh,K)
    alphaarr   = zeros(Float64,Nfp,K)
    rhsL    = zeros(SVector{3,Float64},Nq,K)
    Larr    = zeros(Float64,K,Ns)
    rhsU    = zeros(SVector{3,Float64},Nq,K)
    v3tilde   = zeros(Float64,Nh)
    rhotilde  = zeros(Float64,Nh)
    rhoetilde = zeros(Float64,Nh)
    vq_k      = zeros(SVector{3,Float64},Nq)
    v_tilde_k = zeros(SVector{3,Float64},Nh)      # TODO: refactor with v_tilde, u_tilde
    u_tilde_k = zeros(SVector{3,Float64},Nh)
    U_modal   = zeros(SVector{3,Float64},Np,K)
    U_k       = zeros(SVector{3,Float64},Np)
    Uq_k      = zeros(SVector{3,Float64},Nq)
    spatial   = zeros(SVector{3,Float64},Np,K)
    boundary  = zeros(SVector{3,Float64},Np,K)
    resW      = zeros(SVector{3,Float64},Nq,K)
    resZ      = zeros(SVector{3,Float64},Nq,K)
    Farr      = zeros(Float64,K,Ns)            # TODO: rename F, eta to theta
    αarr      = zeros(Float64,Nfp,K)
    resW      = zeros(SVector{3,Float64},Nq,K)
    resZ      = zeros(SVector{3,Float64},Nq,K)
    LGLind    = falses(K)                       # TODO: array of BasisType, singleton type
    L_G2L_arr = ones(Float64,K,Ns)
    L_L2G_arr = ones(Float64,K,Ns)
    L_Vf_arr  = ones(Float64,K,Ns)              # TODO: Refactor with Farr
    VhPq_new  = zeros(Float64,Nh,Nq)
    VhPq_new[1:Nq,1:Nq] = diagm(ones(Nq))
    Vf_new      = zeros(Float64,Nfp,Nq)
    VhT_new     = zeros(Float64,Np,Nh)
    MinvVhT_new = zeros(Float64,Np,Nh)
    uL_k      = zeros(SVector{3,Float64},Nq)
    P_k       = zeros(SVector{3,Float64},Nq)

    prealloc = Preallocation(Uq,vq,v_tilde,u_tilde,beta,rholog,betalog,lam,LFc,rhsH,Ui,Uj,QF1,BF1,uP,betaP,rhologP,betalogP,flux,wavespeed,alphaarr,rhsL,Larr,rhsU,v3tilde,rhotilde,rhoetilde,vq_k,v_tilde_k,u_tilde_k,U_modal,U_k,Uq_k,spatial,boundary,resW,resZ,Farr,αarr,LGLind,L_G2L_arr,L_L2G_arr,L_Vf_arr,VhPq_new,Vf_new,VhT_new,MinvVhT_new,uL_k,P_k)
    return prealloc
end

function initialize_Gauss_rd(param)
    # TODO: refactor
    @unpack N = param

    # Set up reference element and the mesh
    # This set up the basis at Gauss node
    rd = RefElemData(Line(),N)
    @unpack fv = rd

    elem = Line()
    Nplot = 10
    # Construct matrices on reference elements
    r,w = gauss_quad(0,0,N)
    Fmask = [1 N+1]
    VDM = vandermonde(elem, N, r)
    Dr = grad_vandermonde(elem, N, r)/VDM

    V1 = vandermonde(elem, 1, r)/vandermonde(elem, 1, [-1; 1])

    rq, wq = gauss_quad(0,0,N)
    Vq = vandermonde(elem, N, rq) / VDM
    M = Vq' * diagm(wq) * Vq
    Pq = M \ (Vq' * diagm(wq))

    rf  = [-1.0; 1.0]
    nrJ = [-1.0; 1.0]
    wf  = [1.0; 1.0]
    Vf = vandermonde(elem, N, rf) / VDM
    LIFT = M \ (Vf') # lift matrix

    # plotting nodes
    rp = equi_nodes(elem, Nplot)
    Vp = vandermonde(elem, N, rp) / VDM

    rd = RefElemData(Line(), Polynomial, N, fv, V1,
                     tuple(r), VDM, vec(Fmask),
                     Nplot, tuple(rp), Vp,
                     tuple(rq), wq, Vq,
                     tuple(rf), wf, Vf, tuple(nrJ),
                     M, Pq, tuple(Dr), LIFT)

    return rd
end

function initialize_LGL_rd(param)
    @unpack N = param

    rd = RefElemData(Line(),N,quad_rule_vol=gauss_lobatto_quad(0,0,N))
    return rd
end

# TODO: specialize for 1D
function initialize_operators(param,rd)
    @unpack N,K,XL,XR = param
    ZEROTOL = param.global_constants.ZEROTOL

    # VX,EToV = uniform_mesh(Line(),K)
    # VX = VX[1]
    # @. VX = (VX + 1.0)/2.0
    # TODO: refactor
    VX = LinRange(XL,XR,K+1)
    EToV = transpose(reshape(sort([1:K; 2:K+1]),2,K))
    md = MeshData(VX,EToV,rd)

    @unpack x,xq,rxJ,nxJ,J,mapP = md
    @unpack r,rq,wq,wf,M,Pq,Vq,Vf,LIFT,nrJ,Dr,VDM = rd

    # Construct hybridized SBP operators
    Qr = Pq'*M*Dr*Pq
    Ef = Vf*Pq
    Br = diagm(wf.*nrJ)
    Qrh = .5*[Qr-Qr' Ef'*Br;
            -Br*Ef  Br]
    Qrh_skew = .5*(Qrh-Qrh')
    Qrh_skew = Matrix(droptol!(sparse(Qrh_skew),ZEROTOL))
    Qrh_skew_db = 2*Qrh_skew

    # TODO: hardcoded
    Ns  = 3
    Nc  = 3
    Np  = N+1
    Nq  = length(wq)
    Nfp = size(Vf,1) 
    Nh  = Nq+Nfp

    # low order operator
    d  = zeros(Float64,Nq)
    dl = -0.5*ones(Float64,Nq-1) 
    du =  0.5*ones(Float64,Nq-1) 
    d[1]   = -0.5
    d[end] =  0.5
    Qr0 = droptol!(sparse(Tridiagonal(dl,d,du)),ZEROTOL)
    Sr0 = droptol!(sparse(.5*(Qr0-Qr0')),ZEROTOL)

    Jq       = Vq*J
    Vh       = [Vq;Vf]
    rxJh     = Vh*rxJ
    MinvVhT  = M\transpose(Vh)
    VDMinvPq = VDM\Pq
    VqVDM    = Vq*VDM
    VhPq     = Vh*Pq

    Vf_low      = zero.(Vf)
    Vf_low[1]   = 1.0
    Vf_low[end] = 1.0

    Ef_low  = Vf_low*Pq
    Qrh_low = Matrix(droptol!(.5*[Qr0-Qr0'   Ef_low'*Br;
                                 -Br*Ef_low  Br         ],ZEROTOL))
    Qrh_skew_low    = .5*(Qrh_low-Qrh_low')
    Qrh_skew_low    = Matrix(droptol!(sparse(Qrh_skew_low),ZEROTOL))
    Qrh_skew_low_db = 2.0*Qrh_skew_low

    sizes = SizeData(Nc,Np,Nq,Nfp,Nh,Ns)
    geom  = GeomData(J,Jq,rxJh)
    ops   = Operators(Qrh,Qrh_skew_db,Qrh_skew_low_db,Sr0,Br,Vh,MinvVhT,inv(VDM),VDMinvPq,VqVDM,VhPq,Vq,Vf,Vf_low,Pq,LIFT,wq)
    discrete_data = DiscretizationData(sizes,geom,ops)
    return md,discrete_data
end

function init_U!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,md_gauss,md_LGL,prealloc,initial_condition)
    @unpack K  = param
    @unpack Nq = discrete_data_gauss.sizes

    for k = 1:K
        for i = 1:Nq
            xqi = (prealloc.LGLind[k]) ? md_LGL.xq[i,k] : md_gauss.xq[i,k]
            # TODO: pass initial_condition differently
            prealloc.Uq[i,k] = initial_condition(param,xqi)
        end
    end
    update_indicator!(prealloc,param.approximation_basis_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,true)
end

function initialize_data(param)
    @unpack N = param

    rd_gauss = initialize_Gauss_rd(param)
    rd_LGL   = initialize_LGL_rd(param)
    md_gauss,discrete_data_gauss = initialize_operators(param,rd_gauss)
    md_LGL,discrete_data_LGL = initialize_operators(param,rd_LGL)
    T_g2l = vandermonde(Line(),N,rd_LGL.r)/vandermonde(Line(),N,rd_gauss.r)
    T_l2g = vandermonde(Line(),N,rd_gauss.r)/vandermonde(Line(),N,rd_LGL.r)
    transfer_ops = TransferOperators(T_g2l,T_l2g)

    return rd_gauss,md_gauss,discrete_data_gauss,rd_LGL,md_LGL,discrete_data_LGL,transfer_ops
end

function initialize_DG(param,initial_condition,initial_boundary_conditions)
    rd_gauss,md_gauss,discrete_data_gauss,rd_LGL,md_LGL,discrete_data_LGL,transfer_ops = initialize_data(param)

    @unpack sizes,geom,ops = discrete_data_gauss 
    bcdata = initial_boundary_conditions(param,md_gauss)
    prealloc = initialize_preallocations(param,sizes)

    init_U!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,md_gauss,md_LGL,prealloc,initial_condition)

    return rd_gauss,md_gauss,discrete_data_gauss,rd_LGL,md_LGL,discrete_data_LGL,transfer_ops,bcdata,prealloc
end