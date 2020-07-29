module VizEKP

using Plots
using JLD
using LinearAlgebra
using Statistics
using BoundingSphere

export ekp_sphere_evol
export plot_outputs
export plot_ekp_params

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
    function plot_ekp_params(ekp_u::Array{Array{Float64,2},1})

Plots the evolution of EK parameters with time. The ensemble is represented by 
the mean, min and max value of each parameter at each timestep.
"""
function plot_ekp_params(ekp_u::Array{Array{Float64,2},1})
    #One-dimensional measures
    ekp_mean = zeros( (length(ekp_u), length(ekp_u[1][1,:])) )
    ekp_min =  zeros( (length(ekp_u), length(ekp_u[1][1,:])) )
    ekp_max =  zeros( (length(ekp_u), length(ekp_u[1][1,:])) )
    #Iterate through EKP stages
    for i in 1:length(ekp_u)
        ekp_mean[i,:] = mean(ekp_u[i], dims=1)
        ekp_min[i,:] = minimum(ekp_u[i], dims=1)
        ekp_max[i,:] = maximum(ekp_u[i], dims=1)
    end

    for j in 1:length(ekp_u[1][1,:])
        plot(ekp_mean[:,j], ribbon=(
            -ekp_min[:,j].+ekp_mean[:,j], ekp_max[:,j].-ekp_mean[:,j]),
            label=string("parameter ", j), lw=2)
        xlabel!("EKP iteration")
        savefig(string("evol_param_",j,".png"))
    end
    return
end

"""
    function plot_outputs(ekp_g, les_g, num_var, num_heights)

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
    function plot_outputs(ekp_g, les_g, num_var, num_heights)

Plots the mean output from the last EK ensemble. It is assumed that all 
outputs are vertical profiles with the same number of vertical levels.
"""
function plot_error_evolution(ekp_g::Array{Array{Float64,2},1},
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

end #module