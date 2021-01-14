include("../src/truth.jl")

using Plots
using Plots.PlotMeasures
using LinearAlgebra
using JLD2
using Statistics


"""
Plotting script for GP at the true parameters compared with evaluations of the model at the true parameters 
(for use in perfect model setting where one knows the true parameters)

produces a plot for each data type in the truth 
- with the truth mean and 95% confidence interval, 
- with gp predicted mean and 95% interval 


Requires:
--------
- gp prediction of mean at the true parameters
- gp prediction of covariance at the true parameters
- truthobject (processed data from true parameters)
- the data types (currently assumed equal sized)
"""

function main()

    #load data objects
    @load gpdir*"gpmean_at_true.jld2" y_pred
    @load gpdir*"gpcov_at_true.jld2" y_predcov    
    @load truthdir*"truth.jld2" truthobj #uninflated truth run which has been stored

    #simple assumption on data 
    data_types = 3
    plot_idx = reshape(1:size(truthobj.mean), (data_types,size(truthobj.mean)/data_types))

    #prepare the "GCM" (with no inflation)
    truth_mean = truthobj.mean 
    training_cov = cov(truthobj.sample) # this is the cov the gp learns
    truth_sd = sqrt.(diag(training_cov)) 
    
    #prepare the GP
    gp_mean = y_pred
    gp_var = diag(reshape(y_predcov,(size(truthobj.mean),size(truthobj.mean))))
    gp_sd = sqrt.(gp_var)
   
    #backend
    gr(size=(350,350))
    Plots.scalefontsizes(1.25)
    
    for dt in 1:data_types
        xcoords = plot_idx[dt]
        plot(xcoords,
             gp_mean[xcoords],
             ribbon=(2*gp_sd[xcoords],2*gp_sd[xcoords]),
             grid=false,
             legend=false,
             framestyle=:box,
             dpi=300,
             color="orange",
             left_margin=50px,
             bottom_margin=50px)

        plot!(xcoords,
              truth_mean[xcoords],
              yerror=(2*truth_sd[xcoords], 2*truth_sd[xcoords]),
              marker=:circ,
              markersize=1.5,
              markerstrokewidth=0.5,
              color="blue")
        savefig("gp_and_gcm_"*string(dt)*".pdf")
    end

end


main()
