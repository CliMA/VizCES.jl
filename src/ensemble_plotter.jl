include("../src/ekp.jl") # for ekpobj
include("../src/transforms.jl") # for transform_prior_to_real, transform_real_to_prior
include("../src/units.jl") # for apply_units_transform

using Plots
using Plots.PlotMeasures
using LinearAlgebra
using JLD2
using Statistics

using .EKP

"""
Plotting script for the ensemble on a 2D parameter space, It plots:
- iterations of EKP in parameter space
- the location of the true parameters in parameter space
- first iteration in dark grey, intermediate iterations in light grey, final iteration in pink

3 panels        'transformed' (transformed parameters coordinates) 
              'untransformed' (real parameter coordinates) 
         'zoom_untransformed' (zoom-in around true parameters of 'untransformed')

requires:
--------
- EKP object
- min and max iteration of EKP
- true parameter coordinates (atm just given as numbers should be stored somewhere)
- transformations and inverse transformations real->prior space (if required)
- units transform to add units (if required)
- limits for x/y coordinates etc. and other plot preferences

"""

######################################################################
function main()

    #object
    @load "ekp.jld2" ekpobj

    #number of EKP iterations
    min_ekp_it = 1
    max_ekp_it = 10

    #true parameters
    true_params_raw = [0,0]   

    #obtain parameters
    u = ekpobj.u[min_ekp_it:max_ekp_it]#it x [enssize x param]
    u_flat = cat(u...,dims=1)#[(itxens) x param]
    ens_size = ekpobj.J
    
    # Transformations
    u_raw = transform_prior_to_real(u_flat)
    true_params = transform_real_to_prior(true_params)
    
    # Specific units
    u_raw = apply_units_transform(u_raw)
    true_params_raw = apply_units_transform(true_params_raw)

    #plot - transformed
    gr(size=(500,500))

    circ=Shape(Plots.partialcircle(0, 2π))

    plot(u_flat[(min_ekp_it - 1) * ens_size + 1 : min_ekp_it * ens_size, 1],
         u_flat[(min_ekp_it - 1) * ens_size + 1 : min_ekp_it * ens_size, 2],
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
    
    for i = min_ekp_it+1 : max_ekp_it
        if i<max_ekp_it
            plot!(u_flat[(i - 1) * ens_size + 1 : i * ens_size , 1],
                  u_flat[(i - 1) * ens_size + 1 : i * ens_size , 2],
                  seriestype=:scatter,
                  markershape=circ,
                  markercolor=:grey,
                  markeralpha=0.3*(max_ekp_it-i)/max_ekp_it,
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
    savefig(outdir*"ensemble_transformed.pdf")


    #plot - untransformed    
    plot(u_raw[(min_ekp_it - 1) * ens_size + 1 : min_ekp_it * ens_size, 1],
         u_raw[(min_ekp_it - 1) * ens_size + 1 : min_ekp_it * ens_size, 2],
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
    
    for i = min_ekp_it+1 : max_ekp_it
        if i<max_ekp_it
            plot!(u_raw[(i - 1) * ens_size + 1 : i * ens_size , 1],
                  u_raw[(i - 1) * ens_size + 1 : i * ens_size , 2],
                  seriestype=:scatter,
                  markershape=circ,
                  markercolor=:grey,
                  markeralpha=0.3*(max_ekp_it-i)/max_ekp_it,
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

    #zoomed in - removes points outside of zoomed in domain area
    zoom_rhum_bd = [0.6,0.8]
    zoom_tau_bd = [1.0,4.0]
    zoom_rhum_in_bd = (u_raw[:,1] .> zoom_rhum_bd[1]) .& (u_raw[:,1] .< zoom_rhum_bd[2])
    zoom_tau_in_bd = (u_raw[:,2] .> zoom_tau_bd[1]) .& (u_raw[:,2] .< zoom_tau_bd[2])
    onetoN = 1:ens_size*(max_ekp_it - min_ekp_it+1)
    idx_in_bd = onetoN[zoom_rhum_in_bd .& zoom_tau_in_bd]
    
    u_raw_in_bd = u_raw[idx_in_bd,:]

    it_in_bd = (idx_in_bd .>= (min_ekp_it - 1) * ens_size + 1 ) .& (idx_in_bd .<= min_ekp_it * ens_size)
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
    
    for i = min_ekp_it+1 : max_ekp_it
        it_in_bd = (idx_in_bd .>= (i - 1) * ens_size + 1 ) .& (idx_in_bd .<= i * ens_size)
        if i<max_ekp_it
            plot!(u_raw_in_bd[it_in_bd, 1],
                  u_raw_in_bd[it_in_bd, 2],
                  seriestype=:scatter,
                  markershape=circ,
                  markercolor=:grey,
                  markeralpha=0.3*(max_ekp_it-i)/max_ekp_it,
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
