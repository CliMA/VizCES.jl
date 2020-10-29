include("../src/mcmc.jl")
include("../src/truth.jl")
include("../src/eki.jl")
include("../src/transforms.jl")

using Plots
using Plots.PlotMeasures
using LinearAlgebra
using ColorSchemes
using .MCMC
using .EKI
using Statistics
using Distributions
using JLD2
using Random
using StatsPlots, KernelDensity


function main()

    Random.seed!(100)

    disc="T21"
    res="l"
    tdisc="T21"
    truth_id="_phys" #"_meanparam"
    exp_id=res*"designs_"*disc*"_ysample_seed100"
#    exp_id=res*"designs_"*disc*"_param_grid_1600_from_value"
    eki_opt_flag = true
    #create an Array of the MCMC stored
    homedir=split(pwd(),"/test")[1]*"/"
    outdir=homedir*"output/"
    datdir=outdir*"eki_"*tdisc*"truth"*truth_id*"_"*exp_id*"/"

    hours=3600.0
    true_params_raw=[0.7,7200/hours]
    true_params = [transform_rh(true_params_raw[1]),transform_t(true_params_raw[2]*hours) ]
    
    #load the ensemble optimized params
    if eki_opt_flag
        @load datdir*"eki.jld2" ekiobj
    
        U=ekiobj.u[6]#typically the last iteration of EKI we use
        eki_params=mean(U,dims=1)
        eki_params_raw = [inverse_transform_rh(eki_params[1]),inverse_transform_t(eki_params[2])/hours]
    end

    #load MCMC results get prior
    mcmcfile=datdir*"mcmc_1-32.jld2"

    sample_min=10_001
    
    @load mcmcfile mcmc
    posterior_samples=mcmc.posterior[sample_min:end,:]
    println("plot samples", sample_min, " to ", size(mcmc.posterior,1)) 
    
       
    #plots
    gr(size=(500,500))

    rhbd = [-1.25,-0.5]
    taubd = [8.0,10.0]

    rhbd_raw = reverse([inverse_transform_rh(rhbd[1]), inverse_transform_rh(rhbd[2])])
    taubd_raw = [inverse_transform_t(taubd[1]), inverse_transform_t(taubd[2])] ./ hours
    
    #Kernel density estimator
    postd=kde(posterior_samples, boundary=((rhbd[1],rhbd[2]),(taubd[1],taubd[2])) )

    #plot in transformed coordinates
    plot(postd,
         seriestype=:contour,
         seriescolor=cgrad(:tempo),
         colorbar=:right,
         fill=true,
         dpi=300,
         legend=false,
         xlims=rhbd,
         ylims=taubd,
         framestyle=:box,
         left_margin=50px,
         bottom_margin=50px)
    
    plot!([true_params[1]],[true_params[2]], markercolor=:blue, markershape=:circle,msw=0,markersize=5)#need the extra [ ] 
    if eki_opt_flag
        plot!([eki_params[1]],[eki_params[2]], markercolor=:red, markershape=:cross,markersize=8)#need the extra [ ] 
    end
    xlabel!("logit-relative humidity parameter")
    ylabel!("log-timescale parameter (seconds)")
  
    savefig(outdir*"posterior_density_transformed.pdf")

    #untransform
    posterior_samples_raw = zeros(size(posterior_samples))
    posterior_samples_raw[:,1] = inverse_transform_rh.(posterior_samples[:,1])
    posterior_samples_raw[:,2] = inverse_transform_t.(posterior_samples[:,2]) ./ hours
    
    postd_raw = kde(posterior_samples_raw, boundary = ((rhbd_raw[1],rhbd_raw[2]),(taubd_raw[1],taubd_raw[2])) )

    plot(postd_raw,
         seriestype=:contour,
         seriescolor=cgrad(:tempo),
         colorbar=:right,
         fill=true,
         dpi=300,
         legend=false,
         xlims=rhbd_raw,
         ylims=taubd_raw,
         framestyle=:box,
         left_margin=50px,
         bottom_margin=50px)

    plot!([true_params_raw[1]],[true_params_raw[2]], markercolor=:blue, markershape=:circle,msw=0,markersize=5)#need the extra [ ] 
    if eki_opt_flag 
        plot!([eki_params_raw[1]],[eki_params_raw[2]], markercolor=:red, markershape=:cross,markersize=8)#need the extra [ ] 
    end
    xlabel!("relative humidity parameter")
    ylabel!("timescale parameter (hours)")

    savefig(outdir*"posterior_density_untransformed.pdf")

end

main()
