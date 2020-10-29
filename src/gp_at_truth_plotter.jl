include("../src/truth.jl")

using Plots
using Plots.PlotMeasures
using LinearAlgebra
using JLD2
using Statistics

function main()

    homedir = split(pwd(),"/utils")[1]*"/"
    outdir = homedir*"output/"
    disc = "T21"
    exp_id = "ldesigns_"*disc*"_ysample_seed300"
    truth_id="_phys"

    gpdir=outdir*"eki_T21truth"*truth_id*"_"*exp_id*"/"
    truthdir=gpdir
    #load data objects
    @load gpdir*"gpmean_at_true.jld2" y_pred
    @load gpdir*"gpcov_at_true.jld2" y_predcov    
    @load truthdir*"truth.jld2" truthobj #uninflated truth run which has been stored

    lats = truthobj.lats
    nlats = length(lats)

    rhum_idx = 1:nlats
    precip_idx = nlats + 1 : 2 * nlats
    ext_idx = 2 * nlats + 1 : 3 * nlats
    
    #prepare the "GCM" (with no inflation)
    g_t = hcat(truthobj.sample...)
    truth_mean = truthobj.mean
    # we could use quantiles here but we know already the truth is normally distributed
    # A = cov(truthobj.sample)
    # B = perform_truth_inflation(truthobj.data_names, truth_mean, A, 0.2, 0.2, 0.2)
    # println(sum(sum(B - truthobj.cov)))
    # exit()
    
    #add the inflation of the truth:
    obs_truth_cov = 0.5*truthobj.cov
    gcm_truth_cov = 0.5*truthobj.cov
    truth_sd = sqrt.(diag(gcm_truth_cov)) # this is the cov the gp learns
    
    #prepare the GP
    gp_mean = y_pred
    gp_var = diag(reshape(y_predcov,(96,96)))
    
    gp_sd = sqrt.(gp_var)
    println(size(gp_sd))
    #backend
    gr(size=(350,350))
    Plots.scalefontsizes(1.25)
    

    #relative humidity  
    plot(lats,
         gp_mean[rhum_idx],
         ribbon=(2*gp_sd[rhum_idx],2*gp_sd[rhum_idx]),
         xlims=(-90,90),
         ylims=(0.3,1.0),
         grid=false,
         legend=false,
         framestyle=:box,
         dpi=300,
         color="orange",
         left_margin=50px,
         bottom_margin=50px)

    plot!(lats,
         truth_mean[rhum_idx],
         yerror=(2*truth_sd[rhum_idx], 2*truth_sd[rhum_idx]),
         marker=:circ,
          markersize=1.5,
          markerstrokewidth=0.5,
         color="blue")
    xlabel!("Latitude")
    ylabel!("Relative Humidity")
    savefig(outdir*"rhumplot_gp_and_gcm.pdf")
    

    #daily precipitation  
    plot(lats,
         gp_mean[precip_idx],
         ribbon=(2*gp_sd[precip_idx],2*gp_sd[precip_idx]),
         xlims=(-90,90),
         ylims=(0.0,20.0),
         grid=false,
         legend=false,
         framestyle=:box,
         dpi=300,
         color="orange",
         left_margin=50px,
         bottom_margin=50px)

    plot!(lats,
         truth_mean[precip_idx],
         yerror=(2*truth_sd[precip_idx], 2*truth_sd[precip_idx]),
         marker=:circ,
          markersize=1.5,
          markerstrokewidth=0.5,
         color="blue")
    xlabel!("Latitude")
    ylabel!("Precipitation [mm/day]")
    savefig(outdir*"precipplot_gp_and_gcm.pdf")
    
    #extreme precipitation  
    plot(lats,
         gp_mean[ext_idx],
         ribbon=(2*gp_sd[ext_idx],2*gp_sd[ext_idx]),
         xlims=(-90,90),
         ylims=(0.0,0.3),
         grid=false,
         label="GP",
         legend=:topright,
         framestyle=:box,
         dpi=300,
         color="orange",
         left_margin=50px,
         bottom_margin=50px)

    plot!(lats,
          truth_mean[ext_idx],
          yerror=(2*truth_sd[ext_idx], 2*truth_sd[ext_idx]),
          label="GCM",
          marker=:circ,
          markersize=1.5,
          markerstrokewidth=0.5,
          color="blue")
    xlabel!("Latitude")
    ylabel!("Extreme events")
    savefig(outdir*"extplot_gp_and_gcm.pdf")






end


main()
