include("../src/mcmc.jl")
include("../src/truth.jl")
include("../src/eki.jl")
include("../src/transforms.jl")

using Plots
using Cubature
using Plots.PlotMeasures
using LinearAlgebra
using ColorSchemes
using .MCMC
using .EKI
using Statistics
using Distributions
using JLD2
using Random
using StatsPlots, KernelDensity
using Interpolations


function main()

    #provide contours? (NB exclude min and max value of posterior - these will be included)
    #if true, then put in the contour values,
    # if false then code tries to find contours which bound the desired percentiles
    contour_provided=true
    if contour_provided
        # pre-calculated contours for paper
        input_contours=[ 0.010931395363755262,0.47360176939380966, 1.2698317730628141] #100
        #input_contours=[0.008546721811537945, 0.3055192376001232, 0.8163661671511474] #200
        #input_contours=[0.01271488771979764, 0.5600734817558085, 1.4829691339258493] #300
        #input_contours=[0.009910002269882732, 0.5146163328374784, 1.5181388274911354] #400
    else
        integral_target = [0.5,0.75,0.99]
    end
    Random.seed!(100)

    disc="T21"
    res="l"
    tdisc="T21"
    truth_id="_phys" #"_meanparam"
#    exp_id=res*"designs_"*disc*"_inflateyonly_400"
    exp_id=res*"designs_"*disc*"_param_grid_1600_inflateyonly_take2_100_from_value"
    eki_opt_flag = false
    label_flag = false

    #create an Array of the MCMC stored
    homedir=split(pwd(),"/test")[1]*"/"
    outdir=homedir*"output/"
    datdir=outdir*"eki_"*tdisc*"truth"*truth_id*"_"*exp_id*"/"

    hours=3600.0
    true_params_raw=[0.7,7200/hours]
    true_params = [transform_rh(true_params_raw[1]),transform_t(true_params_raw[2]*hours) ]
    
    #load the ensemble optimized params
    if eki_opt_flag
        @load datdir*"eki.jld2" ekiobj
    
        U=ekiobj.u[6]#typically the last iteration of EKI we use
        eki_params=mean(U,dims=1)
        eki_params_raw = [inverse_transform_rh(eki_params[1]),inverse_transform_t(eki_params[2])/hours]
    end

    #load MCMC results get prior
    mcmcfile=datdir*"mcmc_1-32.jld2"

    sample_min=10_001
    
    @load mcmcfile mcmc
    posterior_samples=mcmc.posterior[sample_min:end,:]
    println("plotter will plot samples from ", sample_min, " to ", size(mcmc.posterior,1)) 
    
       
    #plots
    gr(size=(500,500))
    Plots.scalefontsizes(1.5)

    #Box to plot grid
    rhbd = [-1.8,-0.3]
    taubd = [6,10]

    #small box to find contours
#    rhbd = [-1.3, -0.5]
#    taubd = [7.5, 10.0]

    xmin = [rhbd[1],taubd[1]]
    xmax = [rhbd[2],taubd[2]]
     
    #Kernel density estimator
    postd=kde(posterior_samples, boundary=((rhbd[1],rhbd[2]),(taubd[1],taubd[2])) )
    interp_postd = InterpKDE(postd)
    min_cval = 0.0
    max_cval = maximum(maximum(postd.density))
    println("contour upper bound: ", max_cval)
    
    if contour_provided
        contour_values = input_contours
    else 
        #  
        # if the code hangs then either due to small tolerances (where hcubature will hang, or
        # due to domain being too large, so initial h-grid misses the posterior distribution
        # this is likely to happen when max(post-c,0) for c large and the distribution becomes more spiked  
        
        function posterior_mass(x,pd) 
            return pdf(pd,x...)
        end

        target_tol = 1e-3
        int_tol = target_tol/10
        max_evals=Int64(0) #default
        (tot_integral, err) = hcubature(x -> posterior_mass(x,interp_postd), xmin, xmax, abstol=int_tol, maxevals=max_evals) #should be 1 but may be less
        println("full density value, error:(", tot_integral, " ", err,")")
        #contour finder
        println("Finding contours:")
        flush(stdout)
        contour_values = zeros(size(integral_target))
        for (i,target) in enumerate(integral_target)
            count = 0
            cval = max_cval / 2
            
            cval_tmpmin = min_cval
            cval_tmpmax = max_cval
            integral_tmpmin = 0
            integral_tmpmax = 1
            #calc integral of midpoint firstly
            (integral,err) = hcubature(x-> max(posterior_mass(x,interp_postd) - cval, 0), xmin, xmax, abstol=int_tol, maxevals=max_evals)
            integral = integral / tot_integral
            
            while abs(integral - target)>target_tol           
                # update the window
                if integral > target
                    cval_tmpmin = cval # increase the tmp minimum to the value
                    integral_tmpmax = integral # record the new max integral
                elseif integral < target
                    cval_tmpmax = cval # decrease the tmp maximum to the value
                    integral_tmpmin = integral #record the new min integral
                end
                
                # update linearly
                stepsize = (target - integral_tmpmin) / (integral_tmpmax - integral_tmpmin)
                cval = cval_tmpmin + (1-stepsize)*(cval_tmpmax - cval_tmpmin)       
                
                #recalculate integral
                (integral, err) = hcubature(x -> max(posterior_mass(x,interp_postd) - cval,0), xmin, xmax, abstol=int_tol, maxevals=max_evals)
                integral = integral /tot_integral
                count = count+1
                println("Target ", target, ", current integral, error (", integral, " ", err,") after ", count, " iterations. Contour value = ", cval)
                flush(stdout)
                @assert count < 100
            end
            
            println("Target ", target, " achieved with integral ", integral, " after ", count, " iterations. Contour value = ", cval)
            flush(stdout)
            contour_values[i] = cval            
          
        end
    end
        
    rhbd_raw = reverse([inverse_transform_rh(rhbd[1]), inverse_transform_rh(rhbd[2])])
    taubd_raw = [inverse_transform_t(taubd[1]), inverse_transform_t(taubd[2])] ./ hours
    
    c_values = sort([contour_values... , max_cval, 0.0])
   
    #plot in transformed coordinates
    plot(postd,
         seriestype=:contour,
         levels=c_values,
         seriescolor=cgrad(:tempo),
         #colorbar=:right,
         fill=true,
         dpi=300,
         legend=false,
         grid=false,
         xlims=rhbd,
         ylims=taubd,
         framestyle=:box,
         left_margin=50px,
         bottom_margin=50px)
    
    plot!([true_params[1]],[true_params[2]], markercolor=:blue, markershape=:circle,msw=0,markersize=5)#need the extra [ ] 
    if eki_opt_flag
        plot!([eki_params[1]],[eki_params[2]], markercolor=:red, markershape=:cross,markersize=8)#need the extra [ ] 
    end
    if label_flag
        xlabel!("Logit-relative humidity")
        ylabel!("Log-timescale (seconds)")
    end
    savefig(outdir*"posterior_density_transformed.pdf")

    #untransform
    posterior_samples_raw = zeros(size(posterior_samples))
    posterior_samples_raw[:,1] = inverse_transform_rh.(posterior_samples[:,1])
    posterior_samples_raw[:,2] = inverse_transform_t.(posterior_samples[:,2]) ./ hours
    
    postd_raw = kde(posterior_samples_raw, boundary = ((rhbd_raw[1],rhbd_raw[2]),(taubd_raw[1],taubd_raw[2])) )
    max_cval_raw = maximum(maximum(postd_raw.density))
    c_values_raw = sort([contour_values... , max_cval_raw, 0.0])
    plot(postd_raw,
         seriestype=:contour,
         levels=c_values_raw,
         seriescolor=cgrad(:tempo),
         #colorbar=:right,
         fill=true,
         dpi=300,
         grid=false,
         legend=false,
         xlims=rhbd_raw,
         ylims=taubd_raw,
         framestyle=:box,
         left_margin=50px,
         bottom_margin=50px)

    plot!([true_params_raw[1]],[true_params_raw[2]], markercolor=:blue, markershape=:circle,msw=0,markersize=5)#need the extra [ ] 
    if eki_opt_flag 
        plot!([eki_params_raw[1]],[eki_params_raw[2]], markercolor=:red, markershape=:cross,markersize=8)#need the extra [ ] 
    end
    if label_flag
    xlabel!("Relative humidity")
    ylabel!("Timescale (hours)")
    end
    savefig(outdir*"posterior_density_untransformed.pdf")

end

main()
