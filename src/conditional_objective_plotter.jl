include("../src/ekp.jl")
include("../src/truth.jl")
include("../src/transforms.jl")
include("../src/units.jl") # for apply_units_transform

using Plots
using Plots.PlotMeasures
using ColorSchemes
using LinearAlgebra
using JLD2
using Statistics
using Distributions
using Random #(for seed)
using .EKP


######################################################################33
"""
Plots various objective functions along one parameter conditioned the other is
at its truth value.

each panel contains 2 objectives
- Noisy GCM objective \$\\| y - \\mathcal{G}_T(x)\\|_\\Gamma\$
- GP objective: \$\\|\\tilde y - \\tilde \\mathcal{G}_{GP}(x)\\|_{\\Gamma_{GP}(x)}\$
"""

function main()

    #load objects
    @load "ekp.jld2" ekpobj
    @load "truth.jld2" truthobj
    # gp runs at fixed param2
    @load "mean_mesh_fixed_param2.jld2" mean_mesh
    mean_mesh_fixed_param2 = mean_mesh
    @load "var_mesh_fixed_param2.jld2" var_mesh    
    var_mesh_fixed_param2 = var_mesh
    # gp runs at fixed param1
    @load "mean_mesh_fixed_param1.jld2" mean_mesh
    mean_mesh_fixed_param1= mean_mesh
    @load "var_mesh_fixed_param1.jld2" var_mesh    
    var_mesh_fixed_param1 = var_mesh
    
    # the outputs from the gcm at fixed param1 and param2 
    @load fixed_param2_dir*"ekp.jld2" ekpobj
    gcm_fixed_param2 = ekpobj.g[1]
    @load fixed_param1_dir*"ekp.jld2" ekpobj
    gcm_fixed_param1 = ekpobj.g[1]
    
    true_params_raw = [0,0]
    
    # mean, sample or sample_mean
    
    # load the truth (with inflated cov)
    @load truthdir*"truth.jld2" truthobj
    gcm_truth_cov = cov(truthobj.sample)
    obs_truth_cov = truthobj.cov - gcm_truth_cov
    inv_truth_cov = inv(gcm_truth_cov+obs_truth_cov)
    
    #sample of truth in ekp
    yt = ekpobj.g_t 

    #the param values:
    gcm_params_fixed_param2 = collect(0.6 : (0.8-0.6)/99.0 : 0.8)
    gcm_params_fixed_param1 = collect(3600.0 : (21600.0 - 3600.0)/99.0 : 21600.0)
    @load gpdir*"param1vals_fixed_param2.jld2" param1vals    
    @load gpdir*"param2vals_fixed_param1.jld2" param2vals  
        
    # meshes, see emulate_sample.jl, "emulator_uncertainty_graphs(...)"
    # GP mean and sd in svd - transformed coordinates
    # mesh = [outputdim x N x N ]
    # We calculate the objective function by 
    # Obj(i,j) = 0.5sum_{k=1}^outputdim [ sd(i,j)^{-2} (y(k) - GP(k,i,j))^2 ] 
    # note we do not account for the additional terms from prior+normalization constant
    
    # transform y into SVD coordinates first.
    #SVD=svd(truthobj.cov)#svd.U * svd.S * svd.Vt (can also get Vt)
    SVD = svd(gcm_truth_cov)
    Dinv = Diagonal(1.0 ./ sqrt.(SVD.S)) #diagonal matrix of 1/eigenvalues
    D = Diagonal(sqrt.(SVD.S))
    y_transform = Dinv*SVD.Vt*yt
    
    #create the EKP-GP objective for fixed param2
    gp_objective_fixed_param2 = zeros(size(param1vals))
    for i =1:length(param1vals)
        diff = mean_mesh_fixed_param2[:,i] - y_transform
        data_cov = Diagonal(var_mesh_fixed_param2[:,i]) + inflation_cov_transform #Gamma + Delta
        gp_fidelity = 0.5*log(det(data_cov)) 
        gp_objective_fixed_param2[i] = 0.5*diff'*inv(data_cov)*diff + gp_fidelity        
    end
    #create the EKP-GP objective for fixed param1
    gp_objective_fixed_param1 = zeros(size(param2vals))
    for i =1:length(param2vals)
        diff = mean_mesh_fixed_param1[:,i] - y_transform
        data_cov = Diagonal(var_mesh_fixed_param1[:,i]) + inflation_cov_transform #Gamma + Delta
        gp_fidelity = 0.5*log(det(data_cov)) 
        gp_objective_fixed_param1[i] = 0.5*diff'*inv(data_cov)*diff + gp_fidelity
    end
    
 
    #create the GCM objective for fixed param2
    gcm_objective_fixed_param2 = zeros(size(gcm_params_fixed_param2))
    for i =1:length(gcm_params_fixed_param2)
        # in untransformed coordinates
        diff = gcm_fixed_param2[i,:] - yt 
        gcm_objective_fixed_param2[i] = 0.5*diff'*inv_truth_cov*diff
    end
    
    
    #create the GCM objective for fixed param2
    gcm_objective_fixed_param1 = zeros(size(gcm_params_fixed_param1))
    for i =1:length(gcm_params_fixed_param1)
        # in untransformed coordinates
        diff = gcm_fixed_param1[i,:] - yt
        gcm_objective_fixed_param1[i] = 0.5*diff'*inv_truth_cov*diff 
    end
    
    
    param1vals_raw,param2vals_raw = transform_prior_to_real([param1vals,param2vals])
    param1vals_raw,param2vals_raw = apply_units_transform([param1vals_raw,param2vals_raw])

    param1_in_bd = (gcm_params_fixed_param2 .> minimum(param1vals_raw)) .& (gcm_params_fixed_param2 .< maximum(param1vals_raw))
    param2_in_bd = (gcm_params_fixed_param1 .> minimum(param2vals_raw)) .& (gcm_params_fixed_param1 .< maximum(param2vals_raw)) 

    gcm_params_fixed_param2_in_bd = gcm_params_fixed_param2[param1_in_bd] 
    gcm_params_fixed_param1_in_bd = gcm_params_fixed_param1[param2_in_bd]
    gcm_objective_fixed_param2_in_bd = gcm_objective_fixed_param2[param1_in_bd]
    gcm_objective_fixed_param1_in_bd = gcm_objective_fixed_param1[param2_in_bd]

    param1vals_raw,param2vals_raw = apply_units_transform([param1vals_raw,param2vals_raw])
    gcm_params_fixed_param2_in_bd,gcm_params_fixed_param1_in_bd = apply_units_transform([gcm_params_fixed_param2_in_bd,gcm_params_fixed_param1_in_bd])

    #prints min and max of objectives
    gp_minobj_fixed_param2 = minimum(gp_objective_fixed_param2)
    gp_maxobj_fixed_param2 = maximum(gp_objective_fixed_param2)
    println("minimum of fixed param2 gp objective", gp_minobj_fixed_param2)
    println("maximum of fixed param2 gp objective", gp_maxobj_fixed_param2)
    
    gp_minobj_fixed_param1 = minimum(gp_objective_fixed_param1)
    gp_maxobj_fixed_param1 = maximum(gp_objective_fixed_param1)
    println("minimum of fixed param1 gp objective", gp_minobj_fixed_param1)
    println("maximum of fixed param1 gp objective", gp_maxobj_fixed_param1)

    gcm_minobj_fixed_param2 = minimum(gcm_objective_fixed_param2_in_bd)
    gcm_maxobj_fixed_param2 = maximum(gcm_objective_fixed_param2_in_bd)
    println("minimum of fixed param2 gcm objective", gcm_minobj_fixed_param2)
    println("maximum of fixed param2 gcm objective", gcm_maxobj_fixed_param2)

    gcm_minobj_fixed_param1 = minimum(gcm_objective_fixed_param1_in_bd)
    gcm_maxobj_fixed_param1 = maximum(gcm_objective_fixed_param1_in_bd)
    println("minimum of fixed param1 gcm objective", gcm_minobj_fixed_param1)
    println("maximum of fixed param1 gcm objective", gcm_maxobj_fixed_param1)

    
    gr(size=(500,500))
    circ=Shape(Plots.partialcircle(0, 2π))

    #plot fixed_param2 plot
    plot(param1vals_raw,
         gp_objective_fixed_param2, 
         color=:orange,
         linewidth=4,
         dpi=300,
         framestyle=:box,
         grid=false,
         legend=false,
         left_margin=50px,
         bottom_margin=50px)

    plot!(gcm_params_fixed_param2_in_bd,
          gcm_objective_fixed_param2_in_bd, 
          seriestype=:scatter, 
          markershape=:circ,
          markercolor=:grey,
          markersize=6,
          msw=0)

    vline!([true_params_raw[1]],color=:blue,linealpha=0.5,linestyle=:dash)

    xlabel!("param1")
    ylabel!("objective")
    savefig(outdir*"objective_fixed_param2.pdf")

    #plot fixed_param1 plot
    plot(param2vals_raw,
         gp_objective_fixed_param1, 
         label="GP",
         color=:orange,
         linewidth=4,
             dpi=300,
         framestyle=:box,
         grid=false,
         left_margin=50px,
         bottom_margin=50px)

    plot!(gcm_params_fixed_param1_in_bd,
          gcm_objective_fixed_param1_in_bd, 
          seriestype=:scatter, 
          label="GCM",
          markershape=:circle,
          markercolor=:grey,
          markersize=6,
          msw=0)

    vline!([true_params_raw[2]/hour],color=:blue,linealpha=0.5,linestyle=:dash,label="")
        
    xlabel!("param2")
    ylabel!("objective")
    savefig(outdir*"objective_fixed_param1.pdf")
        
end

main()

