using Reexport
using Random
using Dates
# Random.seed!(1234)
# rand(1)

include("./cloth_gen.jl")
@reexport using .ClothGen

include("./model/inference.jl")

num_particles = 20
total_masks = 3
init_frame_num = 1
@eval Constants TOTAL_INFER_ITERATIONS = 1

specific_dir_list = String[]
if length(ARGS) > 0
    specific_dir_list = [joinpath(BASE_LIB_PATH, ARGS[1])]
    if length(ARGS) > 1
        @eval Constants JOB_ID =  ARGS[2]
    end
    cur_dir = split(ARGS[1], "/")[end]
    @eval Constants CUR_SCENE_MASS_STIFF_COMB = $cur_dir
    prev_sim_dir = joinpath(FLEX_SIM_PATH, "t_"*CUR_SCENE_MASS_STIFF_COMB)
    (isdir(prev_sim_dir)) && (rm(prev_sim_dir, recursive=true))
end

println("===============================================================================")
println("total_mask => $(total_masks)")
println("depth_map_var => $(DEPTH_MAP_VAR)")
println("MASS_VAR_SMALL => $(MASS_VAR_SMALL)")
println("MASS_VAR_LARGE => $(MASS_VAR_LARGE)")
println("MASS_BERNOULLI => $(MASS_BERNOULLI)")
println("STIFF_VAR_SMALL => $(STIFF_VAR_SMALL)")
println("STIFF_VAR_LARGE => $(STIFF_VAR_LARGE)")
println("STIFF_BERNOULLI => $(STIFF_BERNOULLI)")
println("BALL_SCENARIO_START_INDEX => $(BALL_SCENARIO_START_INDEX)")
println(DEPTH_MAP_CONFIG)
println("===============================================================================")

init_state_dict, observations_dict, bb_map = load_observations(specific_dir_list, total_masks, init_frame_num)

run(num_particles, init_state_dict, observations_dict, total_masks, init_frame_num, bb_map)

simulation_dir = joinpath(BASE_PY_PATH, "experiments", "simulation", "out", "t_" * cur_dir)
if isdir(simulation_dir)
    rm(simulation_dir, recursive=true)
end