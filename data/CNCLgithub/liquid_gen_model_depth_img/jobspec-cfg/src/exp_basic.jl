using Reexport

include("./fluid_gen.jl")
@reexport using .FluidGen

include("./model/inference.jl")


num_particles = 5
total_masks = 3

specific_dir_list = String[]
if length(ARGS) > 0
    specific_dir_list = [joinpath(BASE_LIB_PATH, ARGS[1])]
end
init_state_dict, observations_dict = collect_observations(specific_dir_list, total_masks)
run(num_particles, init_state_dict, observations_dict, total_masks)

