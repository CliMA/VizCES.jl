include("../src/eki.jl")
include("../src/truth.jl")
include("../src/transforms.jl")

using Plots
using Plots.PlotMeasures
using ColorSchemes
using LinearAlgebra
using JLD2
using Statistics
using Distributions
using Random #(for seed)
using .EKI


######################################################################33
"""
Plots various objective functions along one parameter conditioned the other is
at its truth value.

each panel contains 3 objectives
- Noisy GCM objective \$\\| y - \\mathcal{G}_T(x)\\|_\\Gamma\$
- Two GP objectives: \$\\|\\tilde y - \\tilde \\mathcal{G}_{GP}(x)\\|_{\\Gamma_{GP}(x)}\$
The EKI-GP is trained on EKI points, the GoldStandard GP is trained on a grid
"""

function main()

    #variables
    homedir=split(pwd(),"/utils")[1]*"/"
    outdir=homedir*"output/"

    disc="T21"
    tdisc="T21"
    res="l"
    numlats=32
    truth_id="_phys"
    
    #for gcm runs on the different conditional parameter sets
    fixed_tau_id = res*"designs_"*disc*"_fixed_tau_from_value"
    fixed_rh_id = res*"designs_"*disc*"_fixed_rh_from_value"   
    fixed_tau_dir=outdir*"eki_"*tdisc*"truth"*truth_id*"_"*fixed_tau_id*"/"
    fixed_rh_dir=outdir*"eki_"*tdisc*"truth"*truth_id*"_"*fixed_rh_id*"/"

    #for eki-gp: gpdir. For gs-gp: goldgpdir
    exp_id="designs_"*disc*"_ysample_seed300"
    goldgp_id = "designs_"*disc*"_param_grid_1600_from_value_no_eki_forcing"   
    gpdir=outdir*"eki_"*tdisc*"truth"*truth_id*"_"*res*exp_id*"/"
    ekidir=gpdir
    goldgpdir=outdir*"eki_"*tdisc*"truth"*truth_id*"_"*res*goldgp_id*"/"
    truthdir=gpdir
    
    true_params_raw = [0.7, 7200]
    
    # mean, sample or sample_mean
    gp_type = "mean"
    #if option sample_mean chose:
    sample_size = 1000

    # load the truth (with inflated cov)
    @load truthdir*"truth.jld2" truthobj
    gcm_truth_cov = 0.5*truthobj.cov
    obs_truth_cov = 0.5*truthobj.cov
    inv_truth_cov = inv(gcm_truth_cov+obs_truth_cov)
    truth_var_transformed = zeros(length(truthobj.mean)) #ones(length(truthobj.mean))

    @load ekidir*"eki.jld2" ekiobj
    yt = ekiobj.g_t #inflated truth used in eki        

    #the param values:
    gcm_params_fixed_tau = collect(0.6 : (0.8-0.6)/99.0 : 0.8)
    gcm_params_fixed_rh = collect(3600.0 : (21600.0 - 3600.0)/99.0 : 21600.0)
    @load gpdir*"rhumvals_fixed_tau.jld2" rhumvals    
    @load gpdir*"logtauvals_fixed_rh.jld2" logtauvals  
    
    # the outputs from the eki-gp at fixed tau = 7200
    @load gpdir*"mean_mesh_fixed_tau.jld2" mean_mesh
    mean_mesh_fixed_tau = mean_mesh
    @load gpdir*"var_mesh_fixed_tau.jld2" var_mesh    
    var_mesh_fixed_tau = var_mesh
    # the outputs from the gs-gp at fixed tau = 7200
    @load goldgpdir*"mean_mesh_fixed_tau.jld2" mean_mesh
    gold_mean_mesh_fixed_tau = mean_mesh
    @load goldgpdir*"var_mesh_fixed_tau.jld2" var_mesh    
    gold_var_mesh_fixed_tau = var_mesh
    # the outputs from the gcm at fixed tau = 7200
    @load fixed_tau_dir*"eki.jld2" ekiobj
    gcm_fixed_tau = ekiobj.g[1]
  
    # the outputs from the eki-gp at fixed rh = 0.7
    @load gpdir*"mean_mesh_fixed_rh.jld2" mean_mesh
    mean_mesh_fixed_rh = mean_mesh
    @load gpdir*"var_mesh_fixed_rh.jld2" var_mesh    
    var_mesh_fixed_rh = var_mesh
    # the outputs from the gs-gp at fixed rh = 0.7
    @load goldgpdir*"mean_mesh_fixed_rh.jld2" mean_mesh
    gold_mean_mesh_fixed_rh = mean_mesh
    @load goldgpdir*"var_mesh_fixed_rh.jld2" var_mesh    
    gold_var_mesh_fixed_rh = var_mesh
    # the outputs from the gcm at fixed rh = 0.7
    @load fixed_rh_dir*"eki.jld2" ekiobj
    gcm_fixed_rh = ekiobj.g[1]
    
    println("inflating the gcm runs with the correct level of noise")
    Random.seed!(991199)
    @load gpdir*"inflation.jld2" inflation_only_cov #the inflation from file
    
    n_samples,size_data = size(gcm_fixed_tau)
    noise_samples = rand(MvNormal(zeros(size_data),Diagonal(diag(inflation_only_cov))), 2*n_samples) #noise_samples: size data x samples 
    gcm_fixed_tau += noise_samples[:,1:n_samples]'
    gcm_fixed_rh += noise_samples[:,n_samples+1:2*n_samples]'
    
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
    
    #create the EKI-GP objective for fixed tau
    gp_objective_fixed_tau = zeros(size(rhumvals))
    for i =1:length(rhumvals)
        if gp_type == "sample"
            GP_sample = rand(MvNormal(mean_mesh_fixed_tau[:,i],Diagonal(var_mesh_fixed_tau[:,i])), 1)
            diff =(GP_sample - y_transform)[:]
            inv_data_var = Diagonal(1.0 ./ (truth_var_transformed + var_mesh_fixed_tau[:,i]) )       
            gp_fidelity = 0.5*sum(log.(var_mesh_fixed_tau[:,i] ))      
            gp_objective_fixed_tau[i] = 0.5*diff'*inv_data_var*diff + gp_fidelity
        
        elseif gp_type == "sample_mean"
            GP_sample = rand(MvNormal(mean_mesh_fixed_tau[:,i],Diagonal(var_mesh_fixed_tau[:,i])), sample_size)
            GP_sample = reshape(GP_sample,(length(mean_mesh_fixed_tau[:,i]),sample_size))
            diff =(GP_sample .- y_transform)
            inv_data_var = Diagonal(1.0 ./ (truth_var_transformed +var_mesh_fixed_tau[:,i]) )       
            gp_fidelity = 0.5*sum(log.(var_mesh_fixed_tau[:,i] ))
            misfit = 0
            for j = 1:sample_size
                misfit +=  0.5 * diff[:,j]'*inv_data_var*diff[:,j]
            end
            gp_objective_fixed_tau[i] = 1/sample_size*misfit + gp_fidelity
                                                      
        elseif gp_type == "mean"
            #here the variance only is given by the observational noise
            # In this setup, this MUST be similar structure to the GP noise
            diff = mean_mesh_fixed_tau[:,i] - y_transform
            inv_data_var = Diagonal(1.0 ./  var_mesh_fixed_tau[:,i])
            gp_fidelity = 0.5*sum(log.( var_mesh_fixed_tau[:,i])) 
            gp_objective_fixed_tau[i] = 0.5*diff'*inv_data_var*diff + gp_fidelity
        end
      
    end
    #create the EKI-GP objective for fixed rh
    gp_objective_fixed_rh = zeros(size(logtauvals))
    for i =1:length(logtauvals)
        if gp_type == "sample"
            GP_sample = rand(MvNormal(mean_mesh_fixed_rh[:,i],Diagonal(var_mesh_fixed_rh[:,i])), 1)
            diff =(GP_sample - y_transform)[:]
            inv_data_var = Diagonal(1.0 ./ (truth_var_transformed + var_mesh_fixed_rh[:,i]) )
            gp_fidelity = 0.5*sum(log.(var_mesh_fixed_rh[:,i] ))
            gp_objective_fixed_rh[i] = 0.5*diff'*inv_data_var*diff + gp_fidelity

        elseif gp_type == "sample_mean"
            GP_sample = rand(MvNormal(mean_mesh_fixed_rh[:,i],Diagonal(var_mesh_fixed_rh[:,i])), sample_size)
            GP_sample = reshape(GP_sample,(length(mean_mesh_fixed_rh[:,i]),sample_size))
            diff =(GP_sample .- y_transform)
            inv_data_var = Diagonal(1.0 ./ (truth_var_transformed + var_mesh_fixed_rh[:,i] ))       
            misfit = 0
            for j = 1:sample_size
                misfit +=  0.5 * diff[:,j]'*inv_data_var*diff[:,j]
            end

            gp_fidelity = 0.5*sum(log.(var_mesh_fixed_rh[:,i] ))      
            gp_objective_fixed_rh[i] = 1/sample_size*sum(misfit) + gp_fidelity
     
        elseif gp_type == "mean"            
            diff = mean_mesh_fixed_rh[:,i] - y_transform
            inv_data_var = Diagonal(1.0 ./ (var_mesh_fixed_rh[:,i] ))
            gp_fidelity = 0.5*sum(log.(var_mesh_fixed_rh[:,i] ))
            gp_objective_fixed_rh[i] = 0.5*diff'*inv_data_var*diff + gp_fidelity
            
        end
    end
    
 
    # #create the GS-GP objective for fixed tau
    # gold_gp_objective_fixed_tau = zeros(size(rhumvals))
    # for i =1:length(rhumvals)
    #     if gp_type == "sample"
    #         GP_sample = rand(MvNormal(gold_mean_mesh_fixed_tau[:,i],Diagonal(gold_var_mesh_fixed_tau[:,i])), 1)
    #         diff =(GP_sample - y_transform)[:]
    #         inv_data_var = Diagonal(1.0 ./ gold_var_mesh_fixed_tau[:,i] )       
    #         gp_fidelity = 0.5*sum(log.(gold_var_mesh_fixed_tau[:,i] ))      
    #         gold_gp_objective_fixed_tau[i] = 0.5*diff'*inv_data_var*diff + gp_fidelity
        
    #     elseif gp_type == "sample_mean"
    #         GP_sample = rand(MvNormal(gold_mean_mesh_fixed_tau[:,i],Diagonal(gold_var_mesh_fixed_tau[:,i])), sample_size)
    #         GP_sample = reshape(GP_sample,(length(gold_mean_mesh_fixed_tau[:,i]),sample_size))
    #         diff =(GP_sample .- y_transform)
    #         inv_data_var = Diagonal(1.0 ./ gold_var_mesh_fixed_tau[:,i] )       
    #         misfit = 0
    #         for j = 1:sample_size
    #             misfit +=  0.5 * diff[:,j]'*inv_data_var*diff[:,j]
    #         end

    #         gp_fidelity = 0.5*sum(log.(gold_var_mesh_fixed_tau[:,i] ))      
    #         gold_gp_objective_fixed_tau[i] = 1/sample_size*sum(misfit) + gp_fidelity
                                                      
    #     elseif gp_type == "mean"
    #         diff = gold_mean_mesh_fixed_tau[:,i] - y_transform
    #         var_mesh = gold_var_mesh_fixed_tau[:,i] .- var_shift
    #         inv_data_var = Diagonal(1.0 ./ var_mesh)
    #         gp_fidelity = 0.5*sum(log.(var_mesh)) 
    #         gold_gp_objective_fixed_tau[i] = 0.5*diff'*inv_data_var*diff + gp_fidelity
            
    #     end
   
    # end
    # gold_gp_minobj_fixed_tau = minimum(gold_gp_objective_fixed_tau)
    # gold_gp_maxobj_fixed_tau = maximum(gold_gp_objective_fixed_tau)
    # println("minimum of objective", gold_gp_minobj_fixed_tau)
    # println("maximum of objective", gold_gp_maxobj_fixed_tau)
    
    # #create the GS-GP objective for fixed rh
    # gold_gp_objective_fixed_rh = zeros(size(logtauvals))
    # for i =1:length(logtauvals)
 
    #    if gp_type == "sample"
    #         GP_sample = rand(MvNormal(gold_mean_mesh_fixed_rh[:,i],Diagonal(gold_var_mesh_fixed_rh[:,i])), 1)
    #         diff =(GP_sample - y_transform)[:]
    #         inv_data_var = Diagonal(1.0 ./ gold_var_mesh_fixed_rh[:,i] )       
    #         gp_fidelity = 0.5*sum(log.(gold_var_mesh_fixed_rh[:,i] ))      
    #         gold_gp_objective_fixed_rh[i] = 0.5*diff'*inv_data_var*diff + gp_fidelity
        
    #     elseif gp_type == "sample_mean"
    #         GP_sample = rand(MvNormal(gold_mean_mesh_fixed_rh[:,i],Diagonal(gold_var_mesh_fixed_rh[:,i])), sample_size)
    #         GP_sample = reshape(GP_sample,(length(gold_mean_mesh_fixed_rh[:,i]),sample_size))
    #         diff =(GP_sample .- y_transform)
    #         inv_data_var = Diagonal(1.0 ./ gold_var_mesh_fixed_rh[:,i] )       
    #         misfit = 0
    #         for j = 1:sample_size
    #             misfit +=  0.5 * diff[:,j]'*inv_data_var*diff[:,j]
    #         end

    #         gp_fidelity = 0.5*sum(log.(gold_var_mesh_fixed_rh[:,i] ))      
    #         gold_gp_objective_fixed_rh[i] = 1/sample_size*sum(misfit) + gp_fidelity
                                                      
    #     elseif gp_type == "mean"
    #         diff = gold_mean_mesh_fixed_rh[:,i] - y_transform
    #         var_mesh = gold_var_mesh_fixed_rh[:,i] .- var_shift
    #         inv_data_var = Diagonal(1.0 ./ var_mesh)
    #         gp_fidelity = 0.5*sum(log.(var_mesh)) 
    #         gold_gp_objective_fixed_rh[i] = 0.5*diff'*inv_data_var*diff + gp_fidelity
            
    #     end
    # end
    
    # gold_gp_minobj_fixed_rh = minimum(gold_gp_objective_fixed_rh)
    # gold_gp_maxobj_fixed_rh = maximum(gold_gp_objective_fixed_rh)
    # println("minimum of objective", gold_gp_minobj_fixed_rh)
    # println("maximum of objective", gold_gp_maxobj_fixed_rh)

 
    #create the GCM objective for fixed tau
    gcm_objective_fixed_tau = zeros(size(gcm_params_fixed_tau))
    for i =1:length(gcm_params_fixed_tau)
        #variance given by the sum of internal variability (gcm) and observation noise (y)
        #inv_var = 1.0/2.0
        #diff = Dinv*SVD.Vt*gcm_fixed_tau[i,:] - y_transform
        #gcm_objective_fixed_tau[i] = inv_var*0.5*diff'*diff
        # in untransformed coordinates below
         diff = gcm_fixed_tau[i,:] - yt 
         gcm_objective_fixed_tau[i] = 0.5*diff'*inv_truth_cov*diff
    end
    
    
    #create the GCM objective for fixed tau
    gcm_objective_fixed_rh = zeros(size(gcm_params_fixed_rh))
    for i =1:length(gcm_params_fixed_rh)
        #inv_var = 1.0/2.0
        #diff = Dinv*SVD.Vt*gcm_fixed_rh[i,:] - y_transform
        #gcm_objective_fixed_rh[i] = inv_var*0.5*diff'*diff 
        # in untransformed coordinates below
        diff = gcm_fixed_rh[i,:] - yt
        gcm_objective_fixed_rh[i] = 0.5*diff'*inv_truth_cov*diff 
    end
    
    
    #units
    hour = 3600.0
    #plots

    rh_in_bd = (gcm_params_fixed_tau .> minimum(inverse_transform_rh.(rhumvals))) .& (gcm_params_fixed_tau .< maximum(inverse_transform_rh.(rhumvals))) 
    tau_in_bd = (gcm_params_fixed_rh .> minimum(inverse_transform_t.(logtauvals))) .& (gcm_params_fixed_rh .< maximum(inverse_transform_t.(logtauvals))) 

    gcm_params_fixed_tau_in_bd = gcm_params_fixed_tau[rh_in_bd] 
    gcm_params_fixed_rh_in_bd = gcm_params_fixed_rh[tau_in_bd]
    gcm_objective_fixed_tau_in_bd = gcm_objective_fixed_tau[rh_in_bd]
    gcm_objective_fixed_rh_in_bd = gcm_objective_fixed_rh[tau_in_bd]


    #prints min and max of objectives
    gp_minobj_fixed_tau = minimum(gp_objective_fixed_tau)
    gp_maxobj_fixed_tau = maximum(gp_objective_fixed_tau)
    println("minimum of fixed tau gp objective", gp_minobj_fixed_tau)
    println("maximum of fixed tau gp objective", gp_maxobj_fixed_tau)
    
    gp_minobj_fixed_rh = minimum(gp_objective_fixed_rh)
    gp_maxobj_fixed_rh = maximum(gp_objective_fixed_rh)
    println("minimum of fixed rh gp objective", gp_minobj_fixed_rh)
    println("maximum of fixed rh gp objective", gp_maxobj_fixed_rh)

    gcm_minobj_fixed_tau = minimum(gcm_objective_fixed_tau_in_bd)
    gcm_maxobj_fixed_tau = maximum(gcm_objective_fixed_tau_in_bd)
    println("minimum of fixed tau gcm objective", gcm_minobj_fixed_tau)
    println("maximum of fixed tau gcm objective", gcm_maxobj_fixed_tau)

    gcm_minobj_fixed_rh = minimum(gcm_objective_fixed_rh_in_bd)
    gcm_maxobj_fixed_rh = maximum(gcm_objective_fixed_rh_in_bd)
    println("minimum of fixed rh gcm objective", gcm_minobj_fixed_rh)
    println("maximum of fixed rh gcm objective", gcm_maxobj_fixed_rh)

    
    gr(size=(500,500))
    #    Plots.scalefontsizes(1.25)
    
    circ=Shape(Plots.partialcircle(0, 2π))

    #plot fixed_tau plot - logscale
    # plot(inverse_transform_rh.(rhumvals),
    #      gp_objective_fixed_tau, 
    #      yaxis=:log10,
    #      color=:orange,
    #      linewidth=4,
    #      dpi=300,
    #      framestyle=:box,
    #      grid=false,
    #      legend=false,
    #      left_margin=50px,
    #      bottom_margin=50px)
    
    # plot!(inverse_transform_rh.(rhumvals),
    #      gold_gp_objective_fixed_tau, 
    #      yaxis=:log10,
    #      color=:darkred,
    #      linewidth=4)

    # plot!(gcm_params_fixed_tau_in_bd,
    #       gcm_objective_fixed_tau_in_bd, 
    #       seriestype=:scatter, 
    #       markershape=:circ,
    #       markercolor=:grey,
    #       markersize=6,
    #       msw=0)

    # vline!([true_params_raw[1]],color=:blue,linealpha=0.5,linestyle=:dash)

    # xlabel!("\\theta_{RH}")
    # ylabel!("objective")
    # savefig(outdir*"objective_fixed_tau_logscale.pdf")

    #plot fixed_rh plot - logscale
    # plot(inverse_transform_t.(logtauvals)/hour,
    #      gp_objective_fixed_rh, 
    #      yaxis=:log10,
    #      label="GP (EKI)",
    #      color=:orange,
    #      linewidth=4,
    #          dpi=300,
    #      framestyle=:box,
    #      grid=false,
    #      left_margin=50px,
    #      bottom_margin=50px)

    # plot!(inverse_transform_t.(logtauvals)/hour,
    #      gold_gp_objective_fixed_rh, 
    #      label="GP (Gold)",
    #      yaxis=:log10,
    #      color=:darkred,
    #      linewidth=4)

    # plot!(gcm_params_fixed_rh_in_bd/hour,
    #       gcm_objective_fixed_rh_in_bd, 
    #       seriestype=:scatter, 
    #       label="GCM",
    #       markershape=:circle,
    #       markercolor=:grey,
    #       markersize=6,
    #       msw=0)

    # vline!([true_params_raw[2]/hour],color=:blue,linealpha=0.5,linestyle=:dash,label="")
        
    # xlabel!("\\theta_{\\tau} (hours)")
    # ylabel!("objective")
    # savefig(outdir*"objective_fixed_rh_logscale.pdf")

    #plot fixed_tau plot
    plot(inverse_transform_rh.(rhumvals),
         gp_objective_fixed_tau, 
         color=:orange,
         linewidth=4,
         dpi=300,
         framestyle=:box,
         grid=false,
         legend=false,
         left_margin=50px,
         bottom_margin=50px)
   
    # plot!(inverse_transform_rh.(rhumvals),
    #      gold_gp_objective_fixed_tau, 
    #      color=:darkred,
    #      linewidth=4)
    
    plot!(gcm_params_fixed_tau_in_bd,
          gcm_objective_fixed_tau_in_bd, 
          seriestype=:scatter, 
          markershape=:circ,
          markercolor=:grey,
          markersize=6,
          msw=0)

    vline!([true_params_raw[1]],color=:blue,linealpha=0.5,linestyle=:dash)

    xlabel!("\\theta_{RH}")
    ylabel!("objective")
    savefig(outdir*"objective_fixed_tau.pdf")

    #plot fixed_rh plot
    plot(inverse_transform_t.(logtauvals)/hour,
         gp_objective_fixed_rh, 
         label="GP (EKI)",
         color=:orange,
         linewidth=4,
             dpi=300,
         framestyle=:box,
         grid=false,
         left_margin=50px,
         bottom_margin=50px)

    # plot!(inverse_transform_t.(logtauvals)/hour,
    #      gold_gp_objective_fixed_rh, 
    #      label="GP (Gold)",
    #      color=:darkred,
    #      linewidth=4)

    plot!(gcm_params_fixed_rh_in_bd/hour,
          gcm_objective_fixed_rh_in_bd, 
          seriestype=:scatter, 
          label="GCM",
          markershape=:circle,
          markercolor=:grey,
          markersize=6,
          msw=0)

    vline!([true_params_raw[2]/hour],color=:blue,linealpha=0.5,linestyle=:dash,label="")
        
    xlabel!("\\theta_{\\tau} (hours)")
    ylabel!("objective")
    savefig(outdir*"objective_fixed_rh.pdf")
        
end

main()

