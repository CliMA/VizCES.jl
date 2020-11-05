include("../src/eki.jl")

using .EKI
using Plots
using Plots.PlotMeasures
using LinearAlgebra
using JLD2
using Statistics
#using LaTeXStrings

using .EKI



# utility functions
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

"""
Plotting script for the ensemble, It plots the following traced over iterations of EKP:
- The residual of mean difference to the truth data weighted by the covariance (as stored in the EKP)
- The standard deviations of each parameter in the ensemble

I assume 2 parameters for now, but easily extendable to more

requires:
---------
- EKP object
- the min and make of EKP iterations to plot over

"""

######################################################################
function main()

    #object
    @load ekpdir*"ekp.jld2" ekpobj


    #plot from min to max iteration of ekp
    min_ekp_it=1
    max_ekp_it=10
    
    u=ekpobj.u[min_ekp_it:max_ekp_it]#it x [enssize x param]
    
    #get sd
    u_sd = get_marginal_sd.(u) #u_sd[i] = [RH_sd,tau_sd] 
    u_sd = cat(u_sd...,dims=1) #make into matrix with num_param columns
    
    #get error
    g_residual = map(x->compute_error_new(ekpobj,x), min_ekp_it:max_ekp_it)

    iterations = 0:size(u_sd)[1]-1 #start at 0 for "ICs"    
    #plot - SD of parameters
    gr(size=(500,500))
    Plots.scalefontsizes(1.25)

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
    savefig(outdir*"sd_ekp.pdf")

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
    savefig(outdir*"residual_ekp.pdf")


end

main()
