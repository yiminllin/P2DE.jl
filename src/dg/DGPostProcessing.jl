function calculate_error(U,param,discrete_data,md,prealloc,exact_sol)
    @unpack equation,N = param
    @unpack Nq,Nc      = discrete_data.sizes
    
    K  = get_num_elements(param)
    T = param.timestepping_param.T

    L1err     = zero(SVector{Nc,Float64})
    L2err     = zero(SVector{Nc,Float64})
    Linferr   = zero(SVector{Nc,Float64})
    L1exact   = zero(SVector{Nc,Float64})
    L2exact   = zero(SVector{Nc,Float64})
    Linfexact = zero(SVector{Nc,Float64})
    for k = 1:K
        U_k = @views U[:,k]
        for i = 1:Nq
            exact_U_k_i = get_exact_solution(prealloc,i,k,T,md,equation,exact_sol)
            wJq_i       = discrete_data.ops.wq[i]*discrete_data.geom.Jq[i]
            L1err       = L1err + wJq_i*abs.(exact_U_k_i-U_k[i])
            L2err       = L2err + wJq_i*abs.(exact_U_k_i-U_k[i]).^2
            Linferr     = max.(Linferr, abs.(exact_U_k_i-U_k[i]))
            L1exact     = L1exact + wJq_i*abs.(exact_U_k_i)
            L2exact     = L2exact + wJq_i*abs.(exact_U_k_i).^2
            Linfexact   = max.(Linfexact, abs.(exact_U_k_i))
        end
    end

    L1_err   = 0.0
    L2_err   = 0.0
    Linf_err = 0.0
    for i = 1:Nc
        if Linfexact[i] > 1e-14
            L1_err   += L1err[i]/L1exact[i]
            L2_err   += sqrt(L2err[i])/sqrt(L2exact[i])
            Linf_err += Linferr[i]/Linfexact[i]
        end
    end
    err = ErrorData(L1_err,L2_err,Linf_err)

    println("N = $N, K = $K")
    println("L1 error is $L1_err")
    println("L2 error is $L2_err")
    println("Linf error is $Linf_err")

    return err
end

function get_exact_solution(prealloc,i,k,T,md,equation::CompressibleIdealGas{Dim1},exact_sol)
    @unpack xq = md
    xq_i = xq[i,k]
    return primitive_to_conservative(equation,exact_sol(equation,xq_i,T))
end

function get_exact_solution(prealloc,i,k,T,md,equation::CompressibleIdealGas{Dim2},exact_sol)
    @unpack xq,yq = md
    xq_i = xq[i,k]
    yq_i = yq[i,k]
    return primitive_to_conservative(equation,exact_sol(equation,xq_i,yq_i,T))
end

# TODO: hardcoded
function get_exact_solution(prealloc,i,k,T,md,equation::KPP{Dim2},exact_sol)
    @unpack xq,yq = md
    xq_i = xq[i,k]
    yq_i = yq[i,k]
    return exact_sol(equation,xq_i,yq_i,T)
end

function plot_component(param,discrete_data,md,prealloc,
                        rhoq,kL,kR,PlotL,PlotU,output_filename,
                        plot_exact=false,xexact=nothing,rhoexact=nothing,kLexact=kL,kRexact=kR)
    @unpack xL,xR = param
    @unpack Nq    = discrete_data.sizes

    gr(x_lim=[xL,xR],ylim=[PlotL,PlotU],label=false,legend=false)
    xplot        = zeros(Float64,Nq,kR-kL+1)
    for k = kL:kR
        xplot[:,k-kL+1] = md.xq[:,k]
    end
    rhoplot      = rhoq[:,kL:kR]
    plot(xplot[:],rhoplot[:])
    if (plot_exact)
        xexactplot   = xexact[:,kLexact:kRexact]
        rhoexactplot = rhoexact[:,kLexact:kRexact]
        plot!(xexactplot[:],rhoexactplot[:])
    end
    savefig(output_filename)
end

function plot_rho_animation(md,param,prealloc,data_hist,limiting_hist,PlotL,PlotU,output_filename)
    @unpack xL,xR = param
    @unpack Uhist = data_hist

    K  = get_num_elements(param)
    gr(x_lim=[xL,xR],ylim=[PlotL,PlotU],label=false,legend=false)
    anim = Animation()
    normalization_factor = reduce(max, map(x->maximum(x), limiting_hist))
    xplot = zeros(Float64,size(md.xq,1),K)
    for k = 1:K
        xplot[:,k] = md.xq[:,k]
    end
    for i = 1:length(Uhist)
        rho = [u[1] for u in Uhist[i]]
        plot(xplot[:],rho[:])

        # plot filter bars
        Bl = xL
        Br = xR
        barL = (limiting_hist[i][:,1]+limiting_hist[i][:,2]+limiting_hist[i][:,3])/3
        # TODO: assume uniform mesh
        ptL = Bl+(Br-Bl)/K/2
        ptR = Br-(Br-Bl)/K/2
        hplot = (Br-Bl)/K
        for k = 1:K
            plot_limiting_bar!(param.entropyproj_limiter_type,[ptL+(k-1)*hplot;ptL+k*hplot],barL,normalization_factor,k)
        end
        frame(anim)
    end

    gif(anim, output_filename, fps=Int64(floor(length(Uhist)/2)))
end

function plot_limiting_bar!(entropyproj_limiter_type::NoEntropyProjectionLimiter,x,barL,normalization_factor,k)
    # Do nothing
end

function plot_limiting_bar!(entropyproj_limiter_type::ScaledExtrapolation,x,barL,normalization_factor,k)
    plot!(x,(1-barL[k])*ones(2),st=:bar,alpha=0.2)
end

function write_to_jld2(param,data_hist,err_data,df,output_filename)
    try
        df = load(output_filename,"data")
    catch
    end

    params = [getfield(param,n) for n in fieldnames(Param)]
    errs   = [getfield(err_data,n) for n in fieldnames(ErrorData)]
    push!(df,(params...,errs...,data_hist))
    save(output_filename,"data",df)
    visualize_error_data(df)
end

function visualize_error_data(df)
    df[!,:num_steps] = [length(dthist) for dthist in [datahist.dthist for datahist in df[!,:data_history]]]
    pretty_table(df[:, [:N,:K,:timestepping_param,:limiting_param,:approximation_basis_type,:rhs_type,:entropyproj_limiter_type,:positivity_limiter_type,:L1err,:L2err,:Linferr,:num_steps]])
end

# TODO: hardcoded
function get_postprocessing_cache(param,md,dim::Dim1)
    @unpack K = param
    @unpack N = param
    @unpack xq = md
    N1D = N+1
    Nc = get_num_components(param.equation)

    Up = zeros(SVector{Nc,Float64},N1D,K)

    return PostprocessingCache(xp=xq,yp=xq,Up=Up)
end

# TODO: only works for rectangular 2D quad mesh
function get_postprocessing_cache(param,md,dim::Dim2)
    @unpack K = param
    @unpack N = param
    @unpack xq,yq = md
    N1D = N+1
    Kx,Ky = K
    K  = get_num_elements(param)
    Nc = get_num_components(param.equation)

    Up = zeros(SVector{Nc,Float64},N1D*Kx,N1D*Ky)
    xp = zeros(Float64,N1D*Kx,N1D*Ky)
    yp = zeros(Float64,N1D*Kx,N1D*Ky)

    for k = 1:K
        ik = mod1(k,Kx)
        jk = div(k-1,Kx)+1
        xq_k = reshape(view(xq,:,k),N1D,N1D)
        yq_k = reshape(view(yq,:,k),N1D,N1D)
        irange = (ik-1)*N1D+1:ik*N1D
        jrange = (jk-1)*N1D+1:jk*N1D
        @views @. xp[irange,jrange] = xq_k
        @views @. yp[irange,jrange] = yq_k
    end

    return PostprocessingCache(xp=xp,yp=yp,Up=Up)
end

function construct_vtk_file!(cache,param,data_hist,output_path,filename)
    @unpack Uhist,thist = data_hist
    @unpack xp,yp,Up    = cache
    @unpack N,K         = param
    N1D = N+1
    Kx,Ky = K
    K = get_num_elements(param)

    pvd = paraview_collection("$(output_path)/$(filename)_N=$(N)_K=$(K).pvd")
    for i in 1:length(Uhist)
        t = thist[i]
        U = Uhist[i]
        vtk_grid("$(output_path)/$(filename)_N=$(N)_K=$(K)_t=$(t)",xp,yp) do vtk
            for k = 1:K
                ik = mod1(k,Kx)
                jk = div(k-1,Kx)+1
                irange = (ik-1)*N1D+1:ik*N1D
                jrange = (jk-1)*N1D+1:jk*N1D
                @views Up[irange,jrange] = reshape(U[:,k],N1D,N1D)
            end    
            set_vtk_field!(vtk,Up,param.equation)

            pvd[t] = vtk
        end
    end

    vtk_save(pvd)
end

function set_vtk_field!(vtk,Up,equation::CompressibleIdealGas)
    vtk["rho"]  = [u[1] for u in Up]
    vtk["rhou"] = [u[2] for u in Up]
    vtk["rhov"] = [u[3] for u in Up]
    vtk["E"]    = [u[4] for u in Up]
end

function set_vtk_field!(vtk,Up,equation::KPP)
    vtk["u"]  = [u[1] for u in Up]
end
