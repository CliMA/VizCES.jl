include("../src/mcmc.jl")
include("../src/truth.jl")
using Plots
using Plots.PlotMeasures
using LinearAlgebra #required to load MCMC object
using .MCMC
using JLD2

#backend
#pyplot()
"""
Plotting script of the trace for the Markov Chain Monte Carlo (MCMC),

Requires:
--------
- MCMC object

produces one plot over the MCMC iterations per parameter in the MCMC learning.

"""

function main()
    #create an Array of the MCMC stored
    homedir=split(pwd(),"/test")[1]*"/"
    outdir=homedir*"output/"
    datdir=outdir*"exp_name"
    mcmcfile = datdir*"MCMC.jld2"

   
    @load mcmcfile mcmc
    posterior=mcmc.posterior #without burnin
    
    for i = 1:size(posterior)[2]
        plot(posterior[:,i],     
             legend=false,
             dpi=300,
             left_margin=50px,
             bottom_margin=50px)
        xlabel!("iteration")
        ylabel!("param_"*string(i) )
        savefig(outdir*"mcmctrace_param_"*string(i)*".pdf")
    end
        
end

