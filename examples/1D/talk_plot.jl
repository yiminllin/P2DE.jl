using StartUpDG
using Plots
using LaTeXStrings

linewidth=6

gr(size=(600,600),aspect_ratio=1,axis=nothing,ticks=nothing,label=false,legend=false,markerstrokewidth=0)

N  = 10
rq1D,wq1D = gauss_lobatto_quad(0,0,N)
rd = RefElemData(Line(),N,quad_rule_vol=gauss_lobatto_quad(0,0,N))
@unpack rq,Vp = rd

Nplot = 50
rp = equi_nodes(Line(),Nplot)
Vp = vandermonde(Line(),N,rp) / vandermonde(Line(),N,nodes(Line(),N))

function shock(x)
    if (x <= 0.0)
        return 10.0
    else
        return 0.5
    end
end

function loworder(x)
    if (x <= -0.7)
        return 10.0
    elseif (x >= 0.8)
        return 0.5
    else
        return -tanh(x*2)/2*10 + 5
    end
end

x = [rq1D.-2 rq1D rq1D.+2]
xp = Vp*x
plot(x,zeros(size(x)),lw=3,seriescolor=:black,label=nothing)
endpts = [-3; -1; 1; 3]
for p in endpts
    plot!([p;p],[-0.1;0.1],lw=3,seriescolor=:black,label=nothing)
end
# plot!(x,shock.(x))
yhigh = Vp*shock.(x)/5
ylow = Vp*loworder.(x)/5
plot!(xp[:],yhigh[:],lw=linewidth,seriescolor=:royalblue1,label="High order solution")
plot!(xp[:],ylow[:],lw=linewidth,seriescolor=:goldenrod1,label="Low order solution")

l = 0.5
plot!(xp[:],l*yhigh[:]+(1-l)ylow[:],lw=linewidth,ls=:dash,seriescolor=:gray70,label="Limited solution")

xi = [-2:-1]
yi = [-0.8;-0.8]
plot!(xi,yi,lw=linewidth,seriescolor=:royalblue1)
plot!(xi,yi.-0.5,lw=linewidth,seriescolor=:goldenrod1)
plot!(xi,yi.-1,lw=linewidth,ls=:dash,seriescolor=:gray70)
annotate!(0.8,-0.8,("High order solution (ESDG)",14))
annotate!(0.35,-1.3,("Low order solution",14))
annotate!(0.22,-1.8,("Limited solution",14))

savefig("outputs/figures/paper-plots/Zhang-Shu.png")


gr(size=(600,600),aspect_ratio=1,axis=nothing,ticks=nothing,label=false,legend=false,markerstrokewidth=0)

x = [rq1D.-2 rq1D rq1D.+2]
xp = Vp*x
plot(x,zeros(size(x)),lw=3,seriescolor=:black,label=nothing)
scatter!(rq[:],zeros(size(rq[:])),markersize=4,markercolor=:darkorange1,markerstrokecolor=:black,markerstrokewidth=2,markershape=:rect)
endpts = [-3; -1; 1; 3]
for p in endpts
    plot!([p;p],[-0.1;0.1],lw=3,seriescolor=:black,label=nothing)
end
# plot!(x,shock.(x))
yhigh = Vp*shock.(x)/5
ylow = Vp*loworder.(x)/5
plot!(xp[:],yhigh[:],lw=linewidth,seriescolor=:royalblue1,label="High order solution")
plot!(xp[:],ylow[:],lw=linewidth,seriescolor=:goldenrod1,label="Low order solution")

l = 0.5
plot!(xp[:],l*yhigh[:]+(1-l)ylow[:],lw=linewidth,ls=:dash,seriescolor=:gray70,label="Limited solution")

xi = [-2:-1]
yi = [-0.8;-0.8]
plot!(xi,yi,lw=linewidth,seriescolor=:royalblue1)
plot!(xi,yi.-0.5,lw=linewidth,seriescolor=:goldenrod1)
plot!(xi,yi.-1,lw=linewidth,ls=:dash,seriescolor=:gray70)
annotate!(0.8,-0.8,("High order solution (ESDG)",14))
annotate!(0.35,-1.3,("Low order solution",14))
annotate!(0.99,-1.8,("Elementwise Limited solution",14))


for rqi in rq
    if rqi > 0.1
        annotate!(rqi, -0.3, text("↑",:center,12+10,:black,:stroke))
    else
        annotate!(rqi, 2.5, text("↓",:center,12+10,:black,:stroke))
    end
end

savefig("outputs/figures/paper-plots/Zhang-Shu-2.png")



gr(size=(600,600),aspect_ratio=1,axis=nothing,ticks=nothing,label=false,legend=false,markerstrokewidth=0)


x = [rq1D.-2 rq1D rq1D.+2]
xp = Vp*x
plot(x,zeros(size(x)),lw=3,seriescolor=:black,label=nothing)
scatter!(rq[:],zeros(size(rq[:])),markersize=4,markercolor=:darkorange1,markerstrokecolor=:black,markerstrokewidth=2,markershape=:rect)
endpts = [-3; -1; 1; 3]
for p in endpts
    plot!([p;p],[-0.1;0.1],lw=3,seriescolor=:black,label=nothing)
end
# plot!(x,shock.(x))
yhigh = Vp*shock.(x)/5
ylow = Vp*loworder.(x)/5
plot!(xp[:],yhigh[:],lw=linewidth,seriescolor=:royalblue1,label="High order solution")
plot!(xp[:],ylow[:],lw=linewidth,seriescolor=:goldenrod1,label="Low order solution")

ylim = copy(yhigh)
for i = 10:30
    ylim[i,2] = yhigh[i,2] + (1-i/30)*(ylow[i,2] - yhigh[i,2])
end
for i = 30:51
    ylim[i,2] = yhigh[i,2] + ((i-30)/10)*(ylow[i,2] - yhigh[i,2])
end

plot!(xp[:],ylim[:],lw=linewidth,ls=:dash,seriescolor=:gray70,label="Limited solution")

xi = [-2:-1]
yi = [-0.8;-0.8]
plot!(xi,yi,lw=linewidth,seriescolor=:royalblue1)
plot!(xi,yi.-0.5,lw=linewidth,seriescolor=:goldenrod1)
plot!(xi,yi.-1,lw=linewidth,ls=:dash,seriescolor=:gray70)
annotate!(0.9,-0.8,("High order solution (DGSEM)",14))
annotate!(0.35,-1.3,("Low order solution",14))
annotate!(0.67,-1.8,("Subcell Limited solution",14))


for rqi in rq
    if rqi > 0.6
        annotate!(rqi, -0.4, text("↑",:center,9+5,:black,:stroke))
    elseif rqi > 0.2
        annotate!(rqi, -0.4, text("↑",:center,15+10,:black,:stroke))
    elseif rqi < -0.1
        annotate!(rqi, 2.5, text("↓",:center,7+4,:black,:stroke))
    else
        annotate!(rqi, 2.5, text("↓",:center,11+8,:black,:stroke))
    end
end

savefig("outputs/figures/paper-plots/Subcell.png")





