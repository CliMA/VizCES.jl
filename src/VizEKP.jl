module VizEKP

using Plots
using JLD
using LinearAlgebra
using Statistics
using BoundingSphere

export ekp_sphere_evol
export plot_outputs
export plot_ekp_params
export plot_error_evolution

"""
    function ekp_sphere_evol(ekp_u::Array{Array{Float64,2},1})

Returns the time evolution of the BoundingSphere parameters (center, radius)
for the EK particle ensemble.
"""
function ekp_sphere_evol(ekp_u::Array{Array{Float64,2},1})
    #N-dimensional measures
    ekp_center = zeros((length(ekp_u), length(ekp_u[1][1,:])))
    ekp_radius = zeros(length(ekp_u))
    ekp_center_jump = zeros(length(ekp_u))
    #Iterate through EKP stages
    for i in 1:length(ekp_u)
        ekp_center[i,:], ekp_radius[i] = boundingsphere(
            mapslices(x->[x], ekp_u[i], dims=2)[:])
        if i > 1
            ekp_center_jump[i] = sqrt( (ekp_center[i,:]-ekp_center[i-1,:])'*(
                ekp_center[i,:]-ekp_center[i-1,:]) )
        end
    end
    return ekp_radius, ekp_center_jump
end

"""
    function plot_ekp_params(ekp_u::Array{Array{Float64,2},1}; 
                         exp_transform::Bool=false,
                         param_names::Union{Vector{String}, Nothing}=nothing)

Plots the evolution of EK parameters with time. The ensemble is represented by 
the mean, min and max value of each parameter at each timestep. 

If exp_transform, the parameter values are exponentiated before plotting.
"""
function plot_ekp_params(
    ekp_u::Array{Array{Float64,2},1}; 
    exp_transform::Bool=false,
    param_names::Union{Vector{String}, Nothing}=nothing)
    
    if exp_transform
        ekp_u_ = deepcopy( map(x->exp.(x), ekp_u) )
    else
        ekp_u_ = ekp_u
    end
    #One-dimensional measures
    ekp_mean = zeros( (length(ekp_u_), length(ekp_u_[1][1,:])) )
    ekp_min =  zero(ekp_mean)
    ekp_max =  zero(ekp_mean)
    #Iterate through EKP stages
    for i in 1:length(ekp_u_)
        ekp_mean[i,:] = mean(ekp_u_[i], dims=1)
        ekp_min[i,:] = minimum(ekp_u_[i], dims=1)
        ekp_max[i,:] = maximum(ekp_u_[i], dims=1)
    end

    for j in 1:length(ekp_u_[1][1,:])
        plot(ekp_mean[:,j], ribbon=(
            -ekp_min[:,j].+ekp_mean[:,j], ekp_max[:,j].-ekp_mean[:,j]),
            label=string("parameter ", j), lw=2)
        xlabel!("EKP iteration")
        if !isnothing(param_names)
            ylabel!(param_names[j])
        end
        xlabel!("EKP iteration")
        savefig(string("evol_param_",j,".png"))
    end
    return
end

function plot_ekp_params(
    ekp_u::Array{Array{Array{Float64,2},1},1}; 
    exp_transform::Bool=false,
    param_names::Union{Vector{String}, Nothing}=nothing)
    
    ekp_u_ = hcat(ekp_u...)
    if exp_transform
        ekp_u_ = deepcopy( map(x->exp.(x), ekp_u_) )
    end

    #One-dimensional measures
    ekp_mean = zeros( (length(ekp_u_[:,1]), length(ekp_u_[1,1][1,:])) )
    ekp_min =  zero(ekp_mean)
    ekp_max =  zero(ekp_mean)
    for j in 1:length(ekp_u_[1,1][1,:]) # For each parameter
        plot()
        for k in 1:length(ekp_u_[1,:]) # For each simulation
            #Iterate through EKP stages
            for i in 1:length(ekp_u_[:,1]) #For each iteration
                ekp_mean[i,:] = mean(ekp_u_[i,k], dims=1)
                ekp_min[i,:] = minimum(ekp_u_[i,k], dims=1)
                ekp_max[i,:] = maximum(ekp_u_[i,k], dims=1)
            end

            plot!(ekp_mean[:,j], ribbon=(
                -ekp_min[:,j].+ekp_mean[:,j], ekp_max[:,j].-ekp_mean[:,j]),
                label=string("parameter ", j), lw=2)
        end
        if !isnothing(param_names)
            ylabel!(param_names[j])
        end
        xlabel!("EKP iteration")
        savefig(string("evol_param_",j,".png"))
    end
    return
end

"""
    function plot_outputs(ekp_g::Array{Array{Float64,2},1},
                      les_g::Array{Float64,1},
                      num_var::Int64,
                      num_heights::Int64)

Plots the mean output from the last EK ensemble. It is assumed that all 
outputs are vertical profiles with the same number of vertical levels.
"""
function plot_outputs(ekp_g::Array{Array{Float64,2},1},
                      les_g::Array{Float64,1},
                      num_var::Int64,
                      num_heights::Int64)
    for sim in 1:length(num_var)
        h_r = range(0, 1, length=Integer(num_heights[sim]))
        for k in 1:num_var[sim]
            inf_lim = Integer(
                num_heights[sim]*(k-1) + (num_heights[1:sim-1]'*num_var[1:sim-1]))+1
            sup_lim = Integer( inf_lim + num_heights[sim] - 1)
            y_model_mean = mean(ekp_g[end][:,inf_lim:sup_lim],dims=1)[1,:]

            plot(les_g[inf_lim:sup_lim] , h_r, label="LES", lw=2)
            plot!(y_model_mean, h_r, label="SCM", lw=2, ls=:dash)
            ylabel!("Normalized height")
            savefig(string("sim_",sim,"_output_variable",k,".png"))
        end
    end
    return
end

"""
    function plot_error_evolution(ekp_err::Union{Vector{Float64}, Array{Vector{Float64},1} };
                              ekp_std_scale::Union{Float64, Array{Float64,1}}=1.0 )

Plots the observation covariance-weighted EK error for each iteration. If Arrays are provided, 
the function returns a single plot with the errors from all EK processes provided.  

ekp_std_scale defines a proportionality constant for the errors from each EK process.

"""
function plot_error_evolution(ekp_err::Union{Vector{Float64}, Array{Vector{Float64},1} };
                              ekp_std_scale::Union{Float64, Array{Float64,1}}=1.0,
                              newplot::Bool=true)

    if ekp_err isa Array{Vector{Float64},1}
        # Wrapper for kwarg broadcasting
        wrapper(ekp_err, ekp_std_scale) = plot_error_evolution(
                                    ekp_err, ekp_std_scale=ekp_std_scale, newplot=false)
        plot()
        wrapper.(ekp_err, ekp_std_scale)
    else
        ekp_err_ = deepcopy(ekp_err.*(ekp_std_scale*ekp_std_scale))
        if newplot
            plot(ekp_err_, lw=2)
        else
            plot!(ekp_err_, lw=2)
        end
        xlabel!("EKP iteration")
        savefig("ekp_error.png")
    end
    return
end

end #module