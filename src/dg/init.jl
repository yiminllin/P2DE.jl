function initialize_preallocations(param, md, sizes)
    (; K, Nd, N1D, Np, Nh, Nq, Nfp, Nc, Ns) = sizes

    Uq = zeros(SVector{Nc,Float64}, Nq, K)
    vq = zeros(SVector{Nc,Float64}, Nq, K)
    u_tilde = zeros(SVector{Nc,Float64}, Nh, K)
    v_tilde = zeros(SVector{Nc,Float64}, Nh, K)
    psi_tilde = zeros(SVector{Nd,Float64}, Nh, K)
    rhsH = zeros(SVector{Nc,Float64}, Nq, K)
    rhsL = zeros(SVector{Nc,Float64}, Nq, K)
    rhsU = zeros(SVector{Nc,Float64}, Nq, K)
    rhsxyH = zeros(SVector{Nd,SVector{Nc,Float64}}, Nq, K)
    rhsxyL = zeros(SVector{Nd,SVector{Nc,Float64}}, Nq, K)
    rhsxyU = zeros(SVector{Nd,SVector{Nc,Float64}}, Nq, K)
    BF_H = zeros(SVector{Nd,SVector{Nc,Float64}}, Nfp, K)
    BF_L = zeros(SVector{Nd,SVector{Nc,Float64}}, Nfp, K)
    fstar_H = zeros(SVector{Nd,SVector{Nc,Float64}}, Nfp, K)
    fstar_L = zeros(SVector{Nd,SVector{Nc,Float64}}, Nfp, K)
    L = zeros(Float64, K, Ns)
    L_local = zeros(Float64, Nq + N1D, Nd, K, Ns)
    theta = zeros(Float64, K, Ns)
    theta_local = zeros(Float64, Nfp, K, Ns)
    resW = zeros(SVector{Nc,Float64}, Nq, K)
    resZ = zeros(SVector{Nc,Float64}, Nq, K)
    indicator = zeros(Float64, Nq, K)
    indicator_modal = zeros(Float64, Np, K)
    smooth_indicator = zeros(Float64, K)

    prealloc = Preallocation{Nc,Nd}(Uq, vq, u_tilde, v_tilde, psi_tilde,
        rhsH, rhsL, rhsU, rhsxyH, rhsxyL, rhsxyU, BF_H, BF_L, fstar_H, fstar_L,
        L, L_local, theta, theta_local,
        resW, resZ,
        indicator, indicator_modal, smooth_indicator)
    return prealloc
end

function initialize_cache(param, md, sizes)
    (; rhs, rhs_limiter, entropyproj_limiter) = param

    return Caches(rhs_cache(rhs, param, sizes), limiter_cache(rhs_limiter, param, sizes), shockcapture_cache(shockcapture(rhs_limiter), param, sizes), entropyproj_limiter_cache(entropyproj_limiter, param, sizes), postprocessing_cache(param, md, sizes, dim(param.equation)))
end

function initialize_DG(param, initial_condition, initial_boundary_conditions)
    # To run plot on server
    # https://stackoverflow.com/questions/64371677/plot-window-showing-up-when-run-from-terminal-in-plots-jl/64374367#64374367
    ENV["GKSwstype"] = "nul"

    rd, md, discrete_data = initialize_data(param)

    (; sizes) = discrete_data
    bcdata = initial_boundary_conditions(param, md)
    prealloc = initialize_preallocations(param, md, sizes)
    caches = initialize_cache(param, md, sizes)

    init_U!(param, discrete_data, md, prealloc, initial_condition)

    return Solver(rd=rd, md=md, discrete_data=discrete_data, param=param), State(preallocation=prealloc, cache=caches), StateParam(bcdata=bcdata)
end

function initialize_data(param)
    return initialize_reference_data(param, param.equation, param.approximation_basis)
end

function initialize_reference_data(param, equation::EquationType{Dim1}, approx_basis::GaussCollocation)
    (; N) = param
    rd_gauss = construct_gauss_reference_data(RefElemData(Line(), N, quad_rule_vol=gauss_quad(0, 0, N)))
    md_gauss, discrete_data_gauss = initialize_operators(param, rd_gauss, GaussQuadrature())

    return rd_gauss, md_gauss, discrete_data_gauss
end

function initialize_reference_data(param, equation::EquationType{Dim1}, approx_basis::LobattoCollocation)
    (; N) = param
    rd_LGL = RefElemData(Line(), N, quad_rule_vol=gauss_lobatto_quad(0, 0, N))
    md_LGL, discrete_data_LGL = initialize_operators(param, rd_LGL, LobattoQuadrature())

    return rd_LGL, md_LGL, discrete_data_LGL
end

function initialize_reference_data(param, equation::EquationType{Dim2}, approx_basis::GaussCollocation)
    (; N) = param

    # create degree N tensor product Gauss quadrature rule
    r1D, w1D = gauss_quad(0, 0, N)
    sq, rq = vec.(StartUpDG.meshgrid(r1D))
    ws, wr = vec.(StartUpDG.meshgrid(w1D))
    wq = @. wr * ws

    rd_gauss = construct_gauss_reference_data(RefElemData(Quad(), N, quad_rule_vol=(rq, sq, wq), quad_rule_face=(r1D, w1D)))
    md_gauss, discrete_data_gauss = initialize_operators(param, rd_gauss, GaussQuadrature())

    return rd_gauss, md_gauss, discrete_data_gauss
end

function initialize_reference_data(param, equation::EquationType{Dim2}, approx_basis::LobattoCollocation)
    (; N) = param
    rd_LGL = RefElemData(Quad(), SBP(), N)
    md_LGL, discrete_data_LGL = initialize_operators(param, rd_LGL, LobattoQuadrature())

    return rd_LGL, md_LGL, discrete_data_LGL
end

# Construct Gauss collocation reference element data from the reference element
# data with Lagrangian basis on LGL node and Gauss quadrature
function construct_gauss_reference_data(rd)
    gauss_to_lobatto = rd.Pq
    lobatto_to_gauss = inv(gauss_to_lobatto)
    # Set Gauss RefElemData to be defined on Lagrangian basis on Gauss nodes
    # instead of LGL nodes
    rst = rd.rstq
    V1 = lobatto_to_gauss * rd.V1
    VDM = lobatto_to_gauss * rd.VDM
    Vp = rd.Vp * gauss_to_lobatto
    Vq = I         # rd.Vq*gauss_to_lobatto
    Vf = rd.Vf * gauss_to_lobatto
    M = gauss_to_lobatto' * rd.M * gauss_to_lobatto
    Pq = I         # lobatto_to_gauss*rd.Pq
    Drst = (x -> lobatto_to_gauss * x * gauss_to_lobatto).(rd.Drst)
    LIFT = rd.M \ (rd.Vf' * diagm(rd.wf))

    return RefElemData(rd.element_type, Polynomial, rd.N, rd.fv, V1,
        rst, VDM, rd.Fmask,
        rd.rstp, Vp,
        rd.rstq, rd.wq, Vq,
        rd.rstf, rd.wf, Vf, rd.nrstJ,
        M, Pq, Drst, LIFT)
end

function initialize_operators(param, rd, quad)
    (; N) = param
    ZEROTOL = param.global_constants.ZEROTOL

    # TODO: Assume uniform mesh
    # Construct mesh
    md = initialize_uniform_mesh_data(param, rd, rd.element_type)
    (; x, J) = md
    (; wq, wf, M, Pq, Vq, Vf, nrstJ, Drst, VDM) = rd
    Vf = Matrix(Vf)

    # Construct geometric factors
    Jq, GJh = geometric_factors(param, rd, md, rd.element_type)

    # Construct hybridized SBP operators
    Vh = [Vq; Vf]
    Qrs = (A -> Matrix(droptol!(sparse(Pq' * M * A * Pq), ZEROTOL))).(Drst)
    Ef = Vf * Pq
    Brs = (A -> Matrix(droptol!(sparse(diagm(wf .* A)), ZEROTOL))).(nrstJ)
    Qrsh = (QB -> Matrix(droptol!(sparse(0.5 * [QB[1]-QB[1]' Ef'*QB[2];
            -QB[2]*Ef QB[2]]), ZEROTOL))).(zip(Qrs, Brs))
    Srsh = (A -> 0.5 * (A - A')).(Qrsh)
    Srsh_db = Tuple((A -> 2 * A).(Srsh))

    # Define sizes
    K = num_elements(param)
    N1D = N + 1
    Nd = Ndim(param.equation)
    Ns = 3   # TODO: define num_stage() for RK time stepper
    Nc = num_components(param.equation)
    Np = size(VDM, 2)
    Nq = length(wq)
    Nfp = size(Vf, 1)
    Nh = Nq + Nfp

    # Define operators
    Vq, Pq = (A -> typeof(A) <: UniformScaling ? diagm(ones(Nq)) : A).((Vq, Pq))  # TODO: bad runtime type check...
    M, Vq, Vh, Pq = (A -> Matrix(droptol!(sparse(A), ZEROTOL))).((M, Vq, Vh, Pq))
    MinvVhT = M \ transpose(Vh)
    MinvVfT = M \ transpose(Vf)
    VDMinvPq = VDM \ Pq
    VqVDM = Vq * VDM
    VhPq = Vh * Pq

    # Low order operators
    Srs0, Vf_low = low_order_operators(param, rd, rd.element_type, quad)
    Vf_low = Matrix(Vf_low)  # TODO: for type stability...
    # TODO: for LGL, there is value close to 1.0 but not exactly 1.0, hardcode for now...
    for i = 1:Nfp
        for j = 1:Nq
            abs(Vf_low[i, j] - 1.0) < param.global_constants.ZEROTOL ? Vf_low[i, j] = 1.0 : nothing
        end
    end

    # Construct nonzero entries in Srsh and Srs0
    Srsh_nnz = Vector{Tuple{Int64,Int64}}()
    for j = 1:Nh
        for i = j+1:Nh
            if sum([abs(s[i, j]) for s in Srsh_db]) != 0
                push!(Srsh_nnz, (i, j))
            end
        end
    end

    Srs0_nnz = Vector{Tuple{Int64,Int64}}()
    for j = 1:Nq
        for i = j+1:Nq
            if sum([abs(s[i, j]) for s in Srs0]) != 0
                push!(Srs0_nnz, (i, j))
            end
        end
    end

    # Face quadrature and volume quadrature mapping
    # q2fq       ::Array{Array{Float64,1},1}
    # fq2q       ::Array{Float64,1}
    fq2q = zeros(Int64, Nfp)
    for i = 1:Nfp
        iq = findfirst(x -> x == 1.0, view(Vf_low, i, :))
        fq2q[i] = iq
    end

    q2fq = [Array{Int64,1}() for _ = 1:Nq]
    for i = 1:Nq
        for iface in (first(idx) for idx in pairs(view(Vf_low, :, i)) if last(idx) == 1.0)
            push!(q2fq[i], iface)
        end
    end

    sizes = SizeData(K, N1D, Nd, Nc, Np, Nq, Nfp, Nh, Ns)
    geom = GeomData(J, Jq, GJh)
    ops = Operators(Srsh_db, Srs0, Srsh_nnz, Srs0_nnz, Brs, Vh, MinvVhT, inv(VDM), VDMinvPq, VqVDM, VhPq, Vq, Vf, Vf_low, Pq, MinvVfT, wq, q2fq, fq2q)
    discrete_data = Discretization(sizes, geom, ops)

    return md, discrete_data
end

function initialize_uniform_mesh_data(param, rd, element::Line)
    (; xL, xR, K) = param

    VXYZ, EToV = uniform_mesh(rd.element_type, K)
    VX = VXYZ[1]
    @. VX = (VX + 1.0) / 2.0 * (xR - xL) + xL
    md = MeshData(VX, EToV, rd)

    return md
end

function initialize_uniform_mesh_data(param, rd, element::Quad)
    (; xL, xR, K) = param

    VXYZ, EToV = uniform_mesh(rd.element_type, K[1], K[2])
    VX = VXYZ[1]
    VY = VXYZ[2]
    @. VX = (VX + 1.0) / 2.0 * (xR[1] - xL[1]) .+ xL[1]
    @. VY = (VY + 1.0) / 2.0 * (xR[2] - xL[2]) .+ xL[2]
    md = MeshData((VX, VY), EToV, rd)

    return md
end

function geometric_factors(param, rd, md, element::Line)
    (; Vq, Vf) = rd
    (; J, rxJ) = md

    Vh = [Vq; Vf]
    Jq = Vq * J
    rxJh = Vh * rxJ

    return Jq, (rxJh,)
end

function geometric_factors(param, rd, md, element::Quad)
    (; Vq, Vf) = rd
    (; J, rxJ, sxJ, ryJ, syJ) = md

    Vh = [Vq; Vf]
    Jq = Vq * J
    rxJh, sxJh, ryJh, syJh = (x -> Vh * x).((rxJ, sxJ, ryJ, syJ))

    return Jq, (rxJh, sxJh, ryJh, syJh)
end

function construct_low_order_operators_1D(param)
    N = param.N
    ZEROTOL = param.global_constants.ZEROTOL
    Nq = N + 1

    # low order operator
    d = zeros(Float64, Nq)
    dl = -0.5 * ones(Float64, Nq - 1)
    du = 0.5 * ones(Float64, Nq - 1)
    d[1] = -0.5
    d[end] = 0.5
    Qr0 = droptol!(sparse(Tridiagonal(dl, d, du)), ZEROTOL)
    Sr0 = droptol!(sparse(0.5 * (Qr0 - Qr0')), ZEROTOL)

    Vf_low = spzeros(2, Nq)
    Vf_low[1] = 1.0
    Vf_low[end] = 1.0

    return Qr0, Sr0, Vf_low
end

function low_order_operators(param, rd, element::Line, quad)
    _, Sr0, Vf_low = construct_low_order_operators_1D(param)
    return (Sr0,), Vf_low
end

function low_order_operators(param, rd, element::Quad, quad::GaussQuadrature)
    _, w1D = gauss_quad(0, 0, param.N)
    return low_order_operators(param, rd, element, quad, w1D)
end

function low_order_operators(param, rd, element::Quad, quad::LobattoQuadrature)
    _, w1D = gauss_lobatto_quad(0, 0, param.N)
    return low_order_operators(param, rd, element, quad, w1D)
end

function low_order_operators(param, rd, element::Quad, quad, w1D)
    ZEROTOL = param.global_constants.ZEROTOL
    M1D = diagm(w1D)
    Q01D, _, _ = construct_low_order_operators_1D(param)
    Qr0 = droptol!(sparse(kron(M1D, Q01D)), ZEROTOL)
    Qs0 = droptol!(sparse(kron(Q01D, M1D)), ZEROTOL)
    Sr0 = droptol!(sparse(0.5 * (Qr0 - Qr0')), ZEROTOL)
    Ss0 = droptol!(sparse(0.5 * (Qs0 - Qs0')), ZEROTOL)
    Vf_low = low_order_extrapolation(param, rd, element, quad)
    return (Sr0, Ss0), Vf_low
end

function low_order_extrapolation(param, rd, element::Quad, quad::GaussQuadrature)
    N = param.N
    Nq1D = N + 1
    Nq = (N + 1) * (N + 1)
    Nfp = 4 * (N + 1)
    Is = collect(1:Nfp)
    Js = collect([1:Nq1D:((Nq1D-1)*Nq1D+1);
        Nq1D:Nq1D:Nq1D*Nq1D;
        1:Nq1D;
        ((Nq1D-1)*Nq1D+1):Nq1D*Nq1D])
    Vs = ones(Nfp)
    return sparse(Is, Js, Vs, Nfp, Nq)
end

function low_order_extrapolation(param, rd, element::Quad, quad::LobattoQuadrature)
    return droptol!(sparse(rd.Vf), param.global_constants.ZEROTOL)
end

function init_U!(param, discrete_data, md, prealloc, initial_condition)
    (; K, Nq) = discrete_data.sizes

    for k = 1:K
        for i = 1:Nq
            set_initial_condition!(prealloc, i, k, param, initial_condition, md, param.equation)
        end
    end
end

function set_initial_condition!(prealloc, i, k, param, initial_condition, md, equation::EquationType{Dim1})
    xqi = md.xq[i, k]
    prealloc.Uq[i, k] = initial_condition(param, xqi)
end

function set_initial_condition!(prealloc, i, k, param, initial_condition, md, equation::EquationType{Dim2})
    xqi = md.xq[i, k]
    yqi = md.yq[i, k]
    prealloc.Uq[i, k] = initial_condition(param, xqi, yqi)
end
