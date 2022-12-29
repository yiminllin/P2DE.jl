function calculate_error(U,param,discrete_data_gauss,discrete_data_LGL,md_gauss,md_LGL,prealloc,exact_sol)
    @unpack equation,N,K = param
    @unpack Nq,Nc        = discrete_data_gauss.sizes
    T = param.timestepping_param.T

    L1err     = zero(SVector{3,Float64})
    L2err     = zero(SVector{3,Float64})
    Linferr   = zero(SVector{3,Float64})
    L1exact   = zero(SVector{3,Float64})
    L2exact   = zero(SVector{3,Float64})
    Linfexact = zero(SVector{3,Float64})
    for k = 1:K
        U_k = @views U[:,k]
        for i = 1:Nq
            xq_i        = (prealloc.LGLind[k]) ? md_LGL.xq[i,k] : md_gauss.xq[i,k]
            wJq_i       = (prealloc.LGLind[k]) ? discrete_data_LGL.ops.wq[i]*discrete_data_LGL.geom.Jq[i] : discrete_data_gauss.ops.wq[i]*discrete_data_gauss.geom.Jq[i]
            exact_U_k_i = primitive_to_conservative(equation,exact_sol(equation,xq_i,T))
            L1err       = L1err + wJq_i*abs.(exact_U_k_i-U_k[i])
            L2err       = L2err + wJq_i*abs.(exact_U_k_i-U_k[i]).^2
            Linferr     = max.(Linferr, abs.(exact_U_k_i-U_k[i]))
            L1exact     = L1exact + wJq_i*abs.(exact_U_k_i)
            L2exact     = L2exact + wJq_i*abs.(exact_U_k_i).^2
            Linfexact   = max.(Linfexact, abs.(exact_U_k_i))
        end
    end

    L1err   = sum(L1err./L1exact)
    L2err   = sum(sqrt.(L2err)./sqrt.(L2exact))
    Linferr = sum(Linferr./Linfexact)
    err = ErrorData(L1err,L2err,Linferr)

    println("N = $N, K = $K")
    println("L1 error is $L1err")
    println("L2 error is $L2err")
    println("Linf error is $Linferr")

    return err
end

function plot_component(param,discrete_data_gauss,md_gauss,md_LGL,prealloc,
                        rhoq,kL,kR,PlotL,PlotU,output_filename,
                        plot_exact=false,xexact=nothing,rhoexact=nothing,kLexact=kL,kRexact=kR)
    @unpack XL,XR = param
    @unpack Nq    = discrete_data_gauss.sizes

    gr(x_lim=[XL,XR],ylim=[PlotL,PlotU],label=false,legend=false)
    xplot        = zeros(Float64,Nq,kR-kL+1)
    for k = kL:kR
        xplot[:,k-kL+1] = (prealloc.LGLind[k]) ? md_LGL.xq[:,k] : md_gauss.xq[:,k]
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

function plot_rho_animation(md_gauss,md_LGL,param,prealloc,data_hist,limiting_hist,PlotL,PlotU,output_filename)
    @unpack XL,XR,K = param
    @unpack Uhist   = data_hist

    gr(x_lim=[XL,XR],ylim=[PlotL,PlotU],label=false,legend=false)
    anim = Animation()
    normalization_factor = reduce(max, map(x->maximum(x), limiting_hist))
    xplot = zeros(Float64,size(md_gauss.xq,1),K)
    for k = 1:K
        xplot[:,k] = (prealloc.LGLind[k]) ? md_LGL.xq[:,k] : md_gauss.xq[:,k]
    end
    for i = 1:length(Uhist)
        rho = [u[1] for u in Uhist[i]]
        plot(xplot[:],rho[:])

        # plot filter bars
        Bl = XL
        Br = XR
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

    push!(df,(param,data_hist,err_data))
    @show output_filename
    save(output_filename,"data",df)
end