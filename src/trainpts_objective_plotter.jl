include("../src/ekp.jl") # for the ekpobj
include("../src/truth.jl") #for the covariance structure
include("../src/transforms.jl") # for transform_prior_to_real, transform_real_to_prior
include("../src/units.jl") # for apply_units_transform

using Plots
using Plots.PlotMeasures
using ColorSchemes
using Distributions
using Random
using LinearAlgebra
using JLD2
using Statistics

using .EKI

"""
Plotting script for the Parameter Ensemble and the GP,

produces two plots,

Requires:
--------
- EKP object
- parameter grid defining the bounds of the plotting region
- gp prediction of mean on the parameter grid
- gp prediction of covariance on the parameter grid
- transformations and inverse transformations real->prior space (if required)
- units transform to add units (if required)
- truthobject (the covariances)
"""

######################################################################33
function main()

    @load "ekp.jld2" ekpobj
    @load "truth.jld2" truthobj
    
    @load "mean_mesh.jld2" mean_mesh
    @load "var_mesh.jld2" var_mesh    
    @load "param1vals.jld2" param1vals    
    @load "param2vals.jld2" param1vals    

    
    #Plotting parameters
    min_eki_it = 1
    max_eki_it = 6
   
    #contour plot levels
    n_levels = 30
   
    yt = ekiobj.g_t #the inflated sample used in eki
    # the truth
    @load truthdir*"truth.jld2" truthobj
    gcm_truth_cov = cov(truthobj.sample)
    obs_truth_cov = truthobj.cov - gcm_truth_cov
   
    u = ekiobj.u[min_eki_it:max_eki_it]#it x [enssize x param]
    u_flat = cat(u...,dims=1)#[(itxens) x param]
    ens_size = ekiobj.J

    # meshes, see emulate_sample.jl, "emulator_uncertainty_graphs(...)"
    # GP mean and sd in svd - transformed coordinates
    # mesh = [outputdim x N x N ]
    # We calculate the objective function by 
    # Obj(i,j) = 0.5sum_{k=1}^outputdim [ sd(i,j)^{-2} (y(k) - GP(k,i,j))^2 ] 
    # note we do not account for the additional terms from prior
    
    # transform y into SVD coordinates first.
    SVD = svd(gcm_truth_cov) #svd.U * svd.S * svd.Vt (can also get Vt)
    Dinv = Diagonal(1.0 ./ sqrt.(SVD.S)) #diagonal matrix of 1/eigenvalues
    y_transform = Dinv*SVD.Vt*yt
    
    objective = zeros((length(param1vals),length(param2vals)))
    for i = 1:length(param1vals)
        for j = 1:length(param2vals)
            diff = mean_mesh[:,i,j] - y_transform
            data_var = Diagonal(1.0 ./ var_mesh[:,i,j] )
            gp_fidelity = 0.5*sum(log.(var_mesh[:,i,j] ))      
            objective[i,j] = 0.5*diff'*data_var*diff + gp_fidelity
        end
    end
    minobj = minimum(objective)
    maxobj = maximum(objective)
    println("minimum of objective", minobj)
    println("maximum of objective", maxobj)
    param1_bd = [minimum(param1vals), maximum(param1vals)]
    param2_bd = [minimum(param2vals), maximum(param2vals)]
    
    #plot - transformed
    gr(size = (500,500))
    objlevels = collect(minobj:(maxobj - minobj) / (n_levels - 1):maxobj)
    
    #Plot the objective function
    plot(param1vals,
         param2vals,
         objective',
         seriestype=:contour,
         levels=objlevels,
         seriescolor=cgrad(:tempo), #white = small obj., dark = big obj.
         fill=true,
         legend=false,
         grid=false,
         xlims=param1_bd,
         ylims=param2_bd,
         dpi=300,
         colorbar=:right,
         clims=[minobj,maxobj],
         framestyle=:box,
         top_margin=25px,
         left_margin=50px,
         bottom_margin=50px)
    
    #plot the EKI points over the top
    circ = Shape(Plots.partialcircle(0, 2π))
    
    #get EKP points in domain
    param1_in_bd = (u_flat[:,1] .> param1_bd[1]) .& (u_flat[:,1].<param1_bd[2])
    param2_in_bd = (u_flat[:,2] .> param2_bd[1]) .& (u_flat[:,2].<param2_bd[2])
    u_in_bd = u_flat[(param1_in_bd) .& (param2_in_bd),:]
    
    plot!(u_in_bd[:,1],
          u_in_bd[:,2],
          seriestype=:scatter,
          markershape=circ,
          markercolor=:black,
          markersize=5,
          msw=0)
        
    xlabel!("transformed param1")
    ylabel!("transformed param2")
    savefig(outdir*"trainpts_over_objective_transformed.pdf")
    

    #untransformed 'raw' coordinates
    param1vals_raw,param2vals_raw = transform_prior_to_real([param1vals,param2vals])
    param1vals_raw,param2vals_raw = apply_units_transform([param1vals_raw,param2vals_raw])

    u_raw_in_bd = transform_prior_to_real(u_in_bd)
    u_raw_in_bd = apply_units_transform(u_in_bd)
    
    param1_bd_raw = [minimum(param1vals_raw), maximum(param1vals_raw)]
    param2_bd_raw = [minimum(param2vals_raw), maximum(param2vals_raw)]
      
    #plot in untransformed coordinates
    
    #Plot the objective function
    
    plot(param1vals_raw,
         param2vals_raw,
         objective_raw', #may need transpose here
         #seriestype=:heatmap,
         seriestype=:contour,
         levels=objlevels,
         seriescolor=cgrad(:tempo), #white = small obj., dark = big obj.
         fill=true,
         legend=false,
         grid=false,
         xlims=param1_bd_raw,
         ylims=param2_bd_raw,
         dpi=300,
         colorbar=:right,
         clims=[minobj,maxobj],
         framestyle=:box,
         top_margin=25px,
         left_margin=50px,
         bottom_margin=50px)
    
    plot!(u_raw_in_bd[:,1],
          u_raw_in_bd[:,2],
          seriestype=:scatter,
          markershape=circ,
          markercolor=:black,
          markersize=5,
          msw=0)

    xlabel!("param1")
    ylabel!("param2")
    savefig("trainpts_over_objective_untransformed.pdf")

end

main()
