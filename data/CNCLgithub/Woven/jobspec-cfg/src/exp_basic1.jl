using Reexport
using Random
using Dates
# Random.seed!(1234)
# rand(1)

include("./cloth_gen.jl")
@reexport using .ClothGen

include("./model/inference.jl")

num_particles = 3
total_masks = 3
init_frame_num = 1
@eval Constants TOTAL_INFER_ITERATIONS = 1

# --------------------------------------------------------------------------- #
specific_dir_list = String[]
mass_prior = ""
stiff_prior = ""
# specific_dir_list = [joinpath(BASE_LIB_PATH, "1/wind_0.25_0.0078125")]
if length(ARGS) > 0
    specific_dir_list = [joinpath(BASE_LIB_PATH, ARGS[1])]
    if length(ARGS) > 1
        @eval Constants EXP_COND = ARGS[2]
        @eval Constants PRIOR_CLOTH = ARGS[3]
        mass_prior = ARGS[4]
        mass_prior = split(mass_prior,"_")
        mass_prior = map(x->parse(Float64,x),mass_prior)
        stiff_prior = ARGS[5]
        stiff_prior = split(stiff_prior,"_")
        stiff_prior = map(x->parse(Float64,x),stiff_prior)
        prior_w = ARGS[6]
        prior_w = split(prior_w,"_")
        prior_w = map(x->parse(Float64,x),prior_w)
        @eval Constants JOB_ID =  ARGS[7]
    end
    # [wb]: Remove previously simulated folders like simulation/t_wind_0.5_0.5
    cur_dir = split(ARGS[1], "/")[end]
    @eval Constants CUR_SCENE_MASS_STIFF_COMB = $cur_dir
    prev_sim_dir = joinpath(FLEX_SIM_PATH, "t_"*CUR_SCENE_MASS_STIFF_COMB)
    (isdir(prev_sim_dir)) && (rm(prev_sim_dir, recursive=true))
end
# --------------------------------------------------------------------------- #



println("===============================================================================")
println("EXP_COND => $(EXP_COND)")
println("PRIOR_CLOTH => $(PRIOR_CLOTH)")
println("mass_prior => $(mass_prior)")
println("stiff_prior => $(stiff_prior)")
println("prior_w => $(prior_w)")

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

## [wb]: Method 1 --> load raw .obj file and calculate dp
# init_state_dict, observations_dict, observations_bb_dict = collect_observations(specific_dir_list, total_masks, init_frame_num)
## [wb]: Method 2 --> load pre-caculated dp
init_state_dict, observations_dict, bb_map = load_observations(specific_dir_list, total_masks, init_frame_num)
# print(collect(keys(observations_dict)))

# [wb: for debug] Check bb_map keys: sort(collect(keys(bb_map)))
run(num_particles, init_state_dict, observations_dict, total_masks, init_frame_num, bb_map, mass_prior, stiff_prior, prior_w)
