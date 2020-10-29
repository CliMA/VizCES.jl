include("../src/eki.jl")

using .EKI
using Plots
using Plots.PlotMeasures
using LinearAlgebra
using JLD2
using Statistics
#using LaTeXStrings

using .EKI


function get_marginal_sd(param_sample)
    # n_samples, n_params = size(param_sample)
    param_var=var(param_sample,dims=1)
    param_sd=sqrt.(param_var)
    
    return param_sd
    
end
function compute_error_new(self::EKIObj,it)
    meang=dropdims(mean(self.g[it],dims=1),dims=1)
    diff = self.g_t - meang
    err = diff' * inv(self.cov)* diff #as the mean of g has a reduced mean due to sample size
    return err
end

######################################################################33
function main()

    #variables
    homedir=split(pwd(),"/utils")[1]*"/"
    outdir=homedir*"output/"

    disc="T21"
    tdisc="T21"
    res="l"
    numlats=32
    exp_id="designs_"*disc*"_ysample_seed300"
    truth_id="_phys"
    ekidir=outdir*"eki_"*tdisc*"truth"*truth_id*"_"*res*exp_id*"/"
    

    #Plotting parameters
    min_eki_it=1
    max_eki_it=10
    #load the ensemble parameters
    @load ekidir*"eki.jld2" ekiobj

    u=ekiobj.u[min_eki_it:max_eki_it]#it x [enssize x param]
    ens_size = ekiobj.J
    
    #get sd
    u_sd = get_marginal_sd.(u) #u_sd[i] = [RH_sd,tau_sd] 
    u_sd = cat(u_sd...,dims=1) #make into matrix with 2 columns
    
    #get error
    #g_residual = ekiobj.error
    g_residual = map(x->compute_error_new(ekiobj,x), min_eki_it:max_eki_it)

    iterations = 0:size(u_sd)[1]-1 #start at 0 for "ICs"    
    #plot - SD of parameters
    gr(size=(500,500))
    Plots.scalefontsizes(1.25)

#    circ=Shape(Plots.partialcircle(0, 2π))

    plot(iterations, 
         u_sd[:,1], 
         label="\\theta_{RH}",
         seriestype=:scatter, 
         markershape=:circle,
         markercolor=:blue,
         markersize=8,
         msw=0,
         dpi=300,
         legend=:topright,
         framestyle=:box,
         grid=false,
         left_margin=50px,
         bottom_margin=50px)

    plot!(iterations,
          u_sd[:,2], 
          seriestype=:scatter, 
          markershape=:cross,
          markercolor=:orange,
          markersize=6,
          msw=0,
          label="\\theta_\\tau")

    xlabel!("Iteration")
    ylabel!("Ensemble Standard Deviation")
    savefig(outdir*"sd_eki.pdf")

    #plot - Residual
    
    plot(iterations, 
         g_residual, 
#         yaxis=:log10,
#         ylims=(1,100),
         line = :scatter,
         markershape=:circle,
         markercolor=:grey,
         markersize=8,
         msw=0,
         dpi=300,
         legend=false,
         grid=false,
         framestyle=:box,
         left_margin=75px,
         bottom_margin=50px)
    xlabel!("Iteration")
    ylabel!("Residual")
    savefig(outdir*"residual_eki.pdf")


end

main()
