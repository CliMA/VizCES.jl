include("../src/mcmc.jl")
include("../src/truth.jl")
include("../src/eki.jl")
using Plots
using Plots.PlotMeasures
using LinearAlgebra
using .MCMC
using .EKI
using Statistics
using Distributions
using JLD2
using Random
using StatsPlots, KernelDensity
    #backend
    pyplot()


function main()

    Random.seed!(100)

    disc="T21"
    res="l"
    tdisc="T21"
    truth_id="_phys" #"_meanparam"
#    exp_id=res*"designs_"*disc*"_inflate_0pt2_0pt5"
     exp_id=res*"designs_"*disc*"_no_eki_forcing_sampledmcmc"
             #create an Array of the MCMC stored
    homedir=split(pwd(),"/test")[1]*"/"
    outdir=homedir*"output/"
    datdir=outdir*"eki_"*tdisc*"truth"*truth_id*"_"*exp_id*"/"

    #load truth for true params
    if truth_id == "_phys"
        trueparams=[log(1.0/0.7-1.0),log(7200.0)]
    elseif truth_id == "_meanparam" 
        trueparams=[log(1.0/0.5-1.0),10.173595774232203]
    end
    #load the ensemble optimized params
    
    @load datdir*"eki.jld2" ekiobj
    
    U=ekiobj.u[6]#typically the last iteration of EKI we use
    #U=cat(U...,dims=1)
    ekiparams=mean(U,dims=1)
    

    #load MCMC results get prior
    mcmcfiles=filter!(x->occursin("mcmc",x),readdir(datdir))
    sort!(mcmcfiles, by = x->parse(Int64,split(split(x,"_")[2],"-")[1]))
    utils=zeros(length(mcmcfiles),2)

    idmin=zeros(Int64,length(mcmcfiles))
    idmax=zeros(Int64,length(mcmcfiles))
    for k=1:length(idmin) 
        idmin[k]=parse(Int64,split(split(mcmcfiles[k],"_")[2], "-")[1])#to get x mcmc_x-y.png
        idmax[k]=parse(Int64,split(split(mcmcfiles[k],"-")[2], ".")[1])#to get y mcmc_x-y.png
    end

    #do first MCMC separately as 
    #sample_min=0
    #sample_min=mcmc.burnin
    sample_min=10_001
    
    @load datdir*mcmcfiles[1] mcmc
    posterior_samples=mcmc.posterior[sample_min:end,:]
    println("plot samples for ", length(mcmcfiles), " design runs, using samples ", sample_min, " to ", size(mcmc.posterior,1)) 
    #posterior_samples=get_posterior(mcmc)#samples of posterior      
    prior=mcmc.prior #stores prior meta information
    println(mcmc.prior)
    prior_samples=zeros(size(posterior_samples))#post=samples x params   
    for (idx,pri) in enumerate(prior)
        if pri["distribution"] == "uniform"
            dist=Uniform(pri["min"],pri["max"])
        elseif pri["distribution"] == "normal"
            dist=Normal(pri["mean"],pri["sd"])
        else
            println("distribution not implemented, see src/mcmc.jl for distributions")
            sys.exit()
        end
        prior_samples[:,idx]=rand(dist,size(prior_samples,1))#get distrubtion at current parameter values
    end

    #KD for prior
    priord=kde(prior_samples)  

    for i=1:length(mcmcfiles)
        @load datdir*mcmcfiles[i] mcmc
        #posterior_samples=get_posterior(mcmc)#samples of posterior      
        posterior_samples=mcmc.posterior#samples of posterior      
        
        #plot with KD
        postd=kde(posterior_samples)
        
        contour(priord,
                seriescolor=:grays,
                label="prior",
                dpi=300,
                alpha=0.2,
                fill=true,
                legend=false,
                xlims=(-2.0,-0.0),
                ylims=(8.0,10.0))
#                xlims=(-1.0,-0.5),
#                ylims=(8.5,9.5))

        contour!(postd,
                 seriescolor=:reds,
                 label="posterior",
                 #             dpi=300,
                 #             legend=false,
                 alpha=0.5,
                 fill=true,
                 #              xlims=(0.5,0.8),
                 #              ylims=(7.0,11.0)
                 )

        plot!([trueparams[1]],[trueparams[2]], markercolor=:blue, markershape=:+,markersize=5)#need the extra [ ] 

        plot!([ekiparams[1]],[ekiparams[2]], markercolor=:red, markershape=:+,markersize=10)#need the extra [ ] 

        #use just plots.jl
        #p=plot(prior_samples,seriestype=:contour,seriescolor=:grays)
        #plot!(p,posterior_samples,seriestype=:contour, seriescolor=:reds)

        savefig(outdir*"pr2po_"*disc*"_"*string(idmin[i])*"-"*string(idmax[i])*".png")
    end
end

main()
