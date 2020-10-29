include("../src/eki.jl")
include("../src/truth.jl")
include("../src/transforms.jl")

using Plots
using Plots.PlotMeasures
using ColorSchemes
using Distributions
using Random
using LinearAlgebra
using JLD2
using Statistics

using .EKI


######################################################################33
function main()

    #variables
    homedir = split(pwd(),"/utils")[1]*"/"
    outdir = homedir*"output/"

    disc = "T21"
    tdisc = "T21"
    res = "l"
    numlats = 32
#    exp_id = "designs_"*disc*"_param_grid_1600_from_value"
    exp_id = "designs_"*disc*"_ysample_seed300"
    truth_id = "_phys"
    ekidir = outdir*"eki_"*tdisc*"truth"*truth_id*"_"*res*exp_id*"/"
    gpdir = ekidir
    truthdir = ekidir #outdir*"eki_"*tdisc*"truth"*truth_id*"_"*res*"designs_"*disc*"_ysample_seed300/"

    #Plotting parameters
    min_eki_it = 1
    max_eki_it = 6
    true_params_raw = [0.7,7200]
    true_params = [transform_rh(true_params_raw[1]),transform_t(true_params_raw[2])]
   
    #load the ensemble parameters
    @load ekidir*"eki.jld2" ekiobj
    yt = ekiobj.g_t #the inflated sample used in eki
    # the truth
    @load truthdir*"truth.jld2" truthobj
    gcm_truth_cov = 0.5*truthobj.cov
    obs_truth_cov = 0.5*truthobj.cov

    #the outputs from the gp
    @load gpdir*"mean_mesh.jld2" mean_mesh
    @load gpdir*"var_mesh.jld2" var_mesh    
    @load gpdir*"rhumvals.jld2" rhumvals    
    @load gpdir*"logtauvals.jld2" logtauvals  

    taubound_raw = [2000,3600*3.75]
    taubound = transform_t.(taubound_raw)
     
    tau_in_bd = (logtauvals .<= taubound[2]) .& (logtauvals .>= taubound[1]) 
    logtauvals = logtauvals[tau_in_bd]
    mean_mesh = mean_mesh[:,:,tau_in_bd]
    var_mesh = var_mesh[:,:,tau_in_bd]
    
    u = ekiobj.u[min_eki_it:max_eki_it]#it x [enssize x param]
    u_flat = cat(u...,dims=1)#[(itxens) x param]
    ens_size = ekiobj.J
    # u_mesh = collect(Iterators.product(rhumvals,logtauvals)) #makes a 'meshgrid' of rhum and log vals.
    #gp type mean, sample_mean
    gp_type = "mean"
    sample_size = 1000 # if sample_mean chosen
    truth_var_transformed = zeros(length(truthobj.mean))
    #we use long term mean as truth

    #@load gpdir*"inflation.jld2" inflation_only_cov
    #noise_sample = rand(MvNormal(zeros(length(truthobj.mean)),Diagonal(diag(inflation_only_cov))), 1) #noise_samples: size data x samples 
     #Random.seed!(100)
    #Random.seed!(200)       
    #Random.seed!(100)
    #sample_ind=randperm!(collect(1:length(truthobj.sample)))[1]
    #yt = truthobj.sample[sample_ind] + noise_sample[:,1]

    # meshes, see emulate_sample.jl, "emulator_uncertainty_graphs(...)"
    # GP mean and sd in svd - transformed coordinates
    # mesh = [outputdim x N x N ]
    # We calculate the objective function by 
    # Obj(i,j) = 0.5sum_{k=1}^outputdim [ sd(i,j)^{-2} (y(k) - GP(k,i,j))^2 ] 
    # note we do not account for the additional terms from prior+normalization constant
    
    # transform y into SVD coordinates first.
    SVD = svd(gcm_truth_cov)#svd.U * svd.S * svd.Vt (can also get Vt)
    Dinv = Diagonal(1.0 ./ sqrt.(SVD.S)) #diagonal matrix of 1/eigenvalues
    y_transform = Dinv*SVD.Vt*yt
    
    objective = zeros((length(rhumvals),length(logtauvals)))
    for i =1:length(rhumvals)
        for j = 1:length(logtauvals)
            if gp_type == "mean"
                diff= mean_mesh[:,i,j] - y_transform
                data_var=Diagonal(1.0 ./ var_mesh[:,i,j] )
                gp_fidelity = 0.5*sum(log.(var_mesh[:,i,j] ))      
                objective[i,j]= 0.5*diff'*data_var*diff + gp_fidelity

            elseif gp_type == "sample_mean"
                GP_sample = rand(MvNormal(mean_mesh[:,i,j],Diagonal(var_mesh[:,i,j])), sample_size)
                GP_sample = reshape(GP_sample,(length(mean_mesh[:,i,j]),sample_size))
                diff =(GP_sample .- y_transform)
                inv_data_var = Diagonal(1.0 ./ (truth_var_transformed+ var_mesh[:,i,j] ))       
                misfit = 0
                for k = 1:sample_size
                    misfit +=  0.5 * diff[:,k]'*inv_data_var*diff[:,k]
                end
                gp_fidelity = 0.5*sum(log.(var_mesh[:,i,j] ))      
                objective[i,j] = 1/sample_size*sum(misfit) + gp_fidelity

            end
        end
    end
    minobj = minimum(objective)
    maxobj = maximum(objective)
    println("minimum of objective", minobj)
    println("maximum of objective", maxobj)
    rhum_bd = [minimum(rhumvals), maximum(rhumvals)]
    logtau_bd = [minimum(logtauvals), maximum(logtauvals)]
    
    # 
    
    #plot - transformed
    gr(size = (500,500))
    n_levels = 30
    
    objlevels = collect(minobj:(maxobj - minobj) / (n_levels - 1):maxobj)
    
    #Plot the objective function
    plot(rhumvals,
         logtauvals,
         objective', #may need transpose here
         #seriestype=:heatmap,
         seriestype=:contour,
         levels=objlevels,
         seriescolor=cgrad(:tempo), #white = small obj., dark = big obj.
         fill=true,
         legend=false,
         grid=false,
         xlims=rhum_bd,
         ylims=logtau_bd,
         dpi=300,
         colorbar=:right,
         clims=[minobj,maxobj],
         framestyle=:box,
         top_margin=25px,
         left_margin=50px,
         bottom_margin=50px)
    
    #plot the EKI points over the top
    circ = Shape(Plots.partialcircle(0, 2π))
    
    #get points in domain
     rhum_in_bd = (u_flat[:,1] .> rhum_bd[1]) .& (u_flat[:,1].<rhum_bd[2])
     logtau_in_bd = (u_flat[:,2] .> logtau_bd[1]) .& (u_flat[:,2].<logtau_bd[2])
     u_in_bd = u_flat[(rhum_in_bd) .& (logtau_in_bd),:]
    
    plot!(u_in_bd[:,1],
          u_in_bd[:,2],
          seriestype=:scatter,
          markershape=circ,
          markercolor=:black,
          markersize=5,
          msw=0)
    
    #        vline!([true_params[1]],color=:blue,linealpha=0.5,linestyle=:dash)
    #        hline!([true_params[2]],color=:blue,linealpha=0.5,linestyle=:dash)
    
    xlabel!("logit-relative humidity parameter")
    ylabel!("log-timescale parameter (seconds)")
    savefig(outdir*"trainpts_over_objective_transformed.pdf")
    

    #untransform
    hours = 3600.0
    #Note the rhumvals flips the order low -> high transformed becomes high -> low!
    #need to flip later on for contour
    rhumvals_raw = reverse(inverse_transform_rh.(rhumvals))
    objective_raw = reverse(objective, dims=1)
    
    logtauvals_raw = inverse_transform_t.(logtauvals) ./ hours
    rhum_bd_raw = [minimum(rhumvals_raw), maximum(rhumvals_raw)]
    logtau_bd_raw = [minimum(logtauvals_raw), maximum(logtauvals_raw)]
    u_raw_in_bd = zeros(size(u_in_bd))
    u_raw_in_bd[:,1] = inverse_transform_rh.(u_in_bd[:,1])
    u_raw_in_bd[:,2] = inverse_transform_t.(u_in_bd[:,2]) ./ hours
      
    #plot in untransformed coordinates
    
    #Plot the objective function
    
    plot(rhumvals_raw,
         logtauvals_raw,
         objective_raw', #may need transpose here
         #seriestype=:heatmap,
         seriestype=:contour,
         levels=objlevels,
         seriescolor=cgrad(:tempo), #white = small obj., dark = big obj.
         fill=true,
         legend=false,
         grid=false,
         xlims=rhum_bd_raw,
         ylims=logtau_bd_raw,
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

    xlabel!("relative humidity parameter")
    ylabel!("timescale parameter (hours)")
    savefig(outdir*"trainpts_over_objective_untransformed.pdf")

end

main()
