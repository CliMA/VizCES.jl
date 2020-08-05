using JLD
using Plots

using VizCES.VizEKP

# σ = 1
filedir = "results_p9_n1.0_e55_i20_d632"
filedir2 = "results_p9_n1.0_e110_i20_d632"
filedir3 = "results_p9_n1.0_e200_i20_d632"
# σ = 10
filedir4 = "results_p9_n10.0_e55_i20_d632"
filedir5 = "results_p9_n10.0_e110_i20_d632"
# σ = .1
filedir6 = "results_p9_n0.1_e55_i20_d632"
filedir7 = "results_p9_n0.1_e110_i20_d632"

num_var = [5, 4, 2, 5]
num_heights = [30, 8, 75, 60]
# paramnames can be dumped from ekiobj
param_names = ["entrainment_factor", "detrainment_factor", "sorting_power", 
    "tke_ed_coeff", "tke_diss_coeff", "pressure_normalmode_adv_coeff", 
         "pressure_normalmode_buoy_coeff1", "pressure_normalmode_drag_coeff", "static_stab_coeff"]

# Load EKP object
ekp = load(string(filedir,"/eki.jld"))
ekp2 = load(string(filedir2,"/eki.jld"))
ekp3 = load(string(filedir3,"/eki.jld"))

ekp4 = load(string(filedir4,"/eki.jld"))
ekp5 = load(string(filedir5,"/eki.jld"))

ekp6 = load(string(filedir6,"/eki.jld"))
ekp7 = load(string(filedir7,"/eki.jld"))

# Bounding sphere
ekp_radius, ekp_center_jump = ekp_sphere_evol(ekp3["eki_u"])
plot(ekp_radius)
savefig("ekp_nradius.png")

# Param range evolution
min_shape = size(ekp3["eki_u"])[1]
ekiu_arr = [ekp["eki_u"][1:min_shape], 
        ekp2["eki_u"][1:min_shape], ekp3["eki_u"][1:min_shape]]

plot_ekp_params(ekiu_arr, exp_transform=true, param_names=param_names)

err_arr = [ekp["eki_err"], ekp2["eki_err"], ekp3["eki_err"]]
println(ekp7["eki_err"])
plot_error_evolution(err_arr, ekp_std_scale=[1.0, 1.0, 1.0], ylims=[1.0e2, 2.0e4])

plot_outputs(ekp3["eki_g"], ekp3["truth_mean"], num_var, num_heights)

