include("../src/mcmc.jl")
include("../src/truth.jl")
using Plots
using Plots.PlotMeasures
using LinearAlgebra
using .MCMC
using Statistics
using JLD2

#backend
#pyplot()

#create an Array of the MCMC stored
homedir=split(pwd(),"/test")[1]*"/"
outdir=homedir*"output/"
datdir=outdir*"eki_T85truth_ldesigns_T21_sigN/"


#load MCMC results and calc utils
mcmcfiles=filter!(x->occursin("mcmc",x),readdir(datdir))
sort!(mcmcfiles, by = x->parse(Int64,split(split(x,"_")[2],"-")[1]))
utils=zeros(length(mcmcfiles),2)

idmin=zeros(Int64,length(mcmcfiles))
idmax=zeros(Int64,length(mcmcfiles))
for k=1:length(idmin) 
    idmin[k]=parse(Int64,split(split(mcmcfiles[k],"_")[2], "-")[1])#to get x mcmc_x-y.png
    idmax[k]=parse(Int64,split(split(mcmcfiles[k],"-")[2], ".")[1])#to get y mcmc_x-y.png
end



for i=1:length(mcmcfiles)
    @load datdir*mcmcfiles[i] mcmc
    #posterior=get_posterior(mcmc)      
    posterior=mcmc.posterior #without burnin

    plot(posterior[:,1],     
     legend=false,
     dpi=300,
     left_margin=50px,
     bottom_margin=50px)
    xlabel!("iteration")
    ylabel!("param1")
    savefig(outdir*"mcmctrace_param1-"*string(idmin[i])*"-"*string(idmax[i])*".png")
    

    plot(posterior[:,2],     
     legend=false,
     dpi=300,
     left_margin=50px,
     bottom_margin=50px)
    xlabel!("iteration")
    ylabel!("param2")
    
    savefig(outdir*"mcmctrace_param2-"*string(idmin[i])*"-"*string(idmax[i])*".png")


end

