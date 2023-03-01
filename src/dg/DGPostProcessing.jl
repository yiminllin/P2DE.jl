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

function get_exact_solution(prealloc,i,k,T,md,equation::EquationType{Dim1},exact_sol)
    @unpack xq = md
    xq_i = xq[i,k]
    return primitive_to_conservative(equation,exact_sol(equation,xq_i,T))
end

function get_exact_solution(prealloc,i,k,T,md,equation::EquationType{Dim2},exact_sol)
    @unpack xq,yq = md
    xq_i = xq[i,k]
    yq_i = yq[i,k]
    return primitive_to_conservative(equation,exact_sol(equation,xq_i,yq_i,T))
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

function plot_limiting_bar!(entropyproj_limiter_type::ExponentialFilter,x,barL,normalization_factor,k)
    plot!(x,barL[k]*ones(2)/normalization_factor,st=:bar,alpha=0.2)
end

function plot_limiting_bar!(entropyproj_limiter_type::Union{ZhangShuFilter,ScaledExtrapolation},x,barL,normalization_factor,k)
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