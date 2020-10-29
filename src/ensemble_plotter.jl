include("../src/eki.jl")
include("../src/transforms.jl")
using Plots
using Plots.PlotMeasures
using LinearAlgebra
using JLD2
using Statistics

using .EKI


######################################################################33
function main()

    #variables
    homedir=split(pwd(),"/utils")[1]*"/"
    outdir=homedir*"output/"

    disc="T21"
    tdisc="T21"
    res="l"
    numlats=32
#    exp_id="designs_"*disc*"_inflate_samples"
    exp_id="designs_"*disc*"_singlegamma_seed300"
    truth_id="_phys"
    ekidir=outdir*"eki_"*tdisc*"truth"*truth_id*"_"*res*exp_id*"/"
    

    #Plotting parameters
    min_eki_it=1
    max_eki_it=10
    true_params_raw=[0.7,7200]
    true_params=[transform_rh(true_params_raw[1]),transform_t(true_params_raw[2])]
    #load the ensemble parameters
    @load ekidir*"eki.jld2" ekiobj

    u=ekiobj.u[min_eki_it:max_eki_it]#it x [enssize x param]
    u_flat=cat(u...,dims=1)#[(itxens) x param]
    ens_size = ekiobj.J
    # Transformations
    u_raw = zeros(size(u_flat))
    
    u_raw[:,1] = inverse_transform_rh(u_flat[:,1])
    u_raw[:,2] = inverse_transform_t(u_flat[:,2])
    
    # Units
    #Convert timescale units from seconds to hours 
    u_raw[:,2] /= 3600.0
    true_params_raw[2] /= 3600.0

    #plot - transformed
    gr(size=(500,500))
#    Plots.scalefontsizes(1.25)

    circ=Shape(Plots.partialcircle(0, 2π))

    plot(u_flat[(min_eki_it - 1) * ens_size + 1 : min_eki_it * ens_size, 1],
         u_flat[(min_eki_it - 1) * ens_size + 1 : min_eki_it * ens_size, 2],
         seriestype=:scatter,
         markershape=circ,
         markercolor=:grey,
         markersize=5,
         msw=0,
         legend=false,
         grid=false,
         xlims=[-3.0,3.0],
         ylims=[7.5, 13],
         annotations=(-2.725,12.725,"(b)"),
         dpi=300,
         framestyle=:box,
         left_margin=50px,
         top_margin=25px,
         bottom_margin=50px)
    
    for i = min_eki_it+1 : max_eki_it
        if i<max_eki_it
            plot!(u_flat[(i - 1) * ens_size + 1 : i * ens_size , 1],
                  u_flat[(i - 1) * ens_size + 1 : i * ens_size , 2],
                  seriestype=:scatter,
                  markershape=circ,
                  markercolor=:grey,
                  markeralpha=0.3*(max_eki_it-i)/max_eki_it,
                  msw=0,
                  markersize=5 )
       else
            plot!(u_flat[(i - 1) * ens_size + 1 : i * ens_size , 1],
                  u_flat[(i - 1) * ens_size + 1 : i * ens_size , 2],
                  seriestype=:scatter,
                  markershape=circ,
                  markercolor=:lightpink,
                  msw=0,
                  markersize=5)
       end
        
    end
    
    vline!([true_params[1]],color=:blue,linealpha=0.5,linestyle=:dash)
    hline!([true_params[2]],color=:blue,linealpha=0.5,linestyle=:dash)

    xlabel!("logit-relative humidity parameter")
    ylabel!("log-timescale parameter (seconds)")
#    title!("Parameter space, transformed coordinates")
    savefig(outdir*"ensemble_transformed.pdf")


    #plot - untransformed
    
    plot(u_raw[(min_eki_it - 1) * ens_size + 1 : min_eki_it * ens_size, 1],
         u_raw[(min_eki_it - 1) * ens_size + 1 : min_eki_it * ens_size, 2],
         seriestype=:scatter,
         markershape=circ,
         markercolor=:grey,
         msw=0,
         markersize=5,
         legend=false,
         grid=false,
         annotations=(0.05,62,"(a)"),
         xlims=[0,1],
         ylims=[0,65],
         dpi=300,
         framestyle=:box,
         left_margin=50px,
         top_margin=25px,
         bottom_margin=50px)
    
    for i = min_eki_it+1 : max_eki_it
        if i<max_eki_it
            plot!(u_raw[(i - 1) * ens_size + 1 : i * ens_size , 1],
                  u_raw[(i - 1) * ens_size + 1 : i * ens_size , 2],
                  seriestype=:scatter,
                  markershape=circ,
                  markercolor=:grey,
                  markeralpha=0.3*(max_eki_it-i)/max_eki_it,
                  msw=0,
                  markersize=5 )
       else
            plot!(u_raw[(i - 1) * ens_size + 1 : i * ens_size , 1],
                  u_raw[(i - 1) * ens_size + 1 : i * ens_size , 2],
                  seriestype=:scatter,
                  markershape=circ,
                  markercolor=:lightpink,
                  msw=0,
                  markersize=5)
       end
        
    end
    vline!([true_params_raw[1]],color=:blue,linealpha=0.5,linestyle=:dash)
    hline!([true_params_raw[2]],color=:blue,linealpha=0.5,linestyle=:dash)

    xlabel!("relative humidity parameter")
    ylabel!("timescale parameter (hours)")
    savefig(outdir*"ensemble_untransformed.pdf")

    #zoomed in
    zoom_rhum_bd = [0.6,0.8]
    zoom_tau_bd = [1.0,4.0]
    zoom_rhum_in_bd = (u_raw[:,1] .> zoom_rhum_bd[1]) .& (u_raw[:,1] .< zoom_rhum_bd[2])
    zoom_tau_in_bd = (u_raw[:,2] .> zoom_tau_bd[1]) .& (u_raw[:,2] .< zoom_tau_bd[2])
    onetoN = 1:ens_size*(max_eki_it - min_eki_it+1)
    idx_in_bd = onetoN[zoom_rhum_in_bd .& zoom_tau_in_bd]
    
    u_raw_in_bd = u_raw[idx_in_bd,:]

    it_in_bd = (idx_in_bd .>= (min_eki_it - 1) * ens_size + 1 ) .& (idx_in_bd .<= min_eki_it * ens_size)
    plot(u_raw_in_bd[it_in_bd, 1],
         u_raw_in_bd[it_in_bd, 2],
         seriestype=:scatter,
         markershape=circ,
         markercolor=:grey,
         msw=0,
         markersize=5,
         legend=false,
         grid=false,
         annotations=(0.05,62,"(a)"),
         xlims=zoom_rhum_bd,
         ylims=zoom_tau_bd,
         dpi=300,
         framestyle=:box,
         left_margin=50px,
         top_margin=25px,
         bottom_margin=50px)
    
    for i = min_eki_it+1 : max_eki_it
        it_in_bd = (idx_in_bd .>= (i - 1) * ens_size + 1 ) .& (idx_in_bd .<= i * ens_size)
        if i<max_eki_it
            plot!(u_raw_in_bd[it_in_bd, 1],
                  u_raw_in_bd[it_in_bd, 2],
                  seriestype=:scatter,
                  markershape=circ,
                  markercolor=:grey,
                  markeralpha=0.3*(max_eki_it-i)/max_eki_it,
                  msw=0,
                  markersize=5 )
       else
            plot!(u_raw_in_bd[it_in_bd, 1],
                  u_raw_in_bd[it_in_bd, 2],
                  seriestype=:scatter,
                  markershape=circ,
                  markercolor=:lightpink,
                  msw=0,
                  markersize=5)
       end
        
    end
    vline!([true_params_raw[1]],color=:blue,linealpha=0.5,linestyle=:dash)
    hline!([true_params_raw[2]],color=:blue,linealpha=0.5,linestyle=:dash)

    xlabel!("relative humidity parameter")
    ylabel!("timescale parameter (hours)")
    savefig(outdir*"ensemble_zoom_untransformed.pdf")





end

main()
