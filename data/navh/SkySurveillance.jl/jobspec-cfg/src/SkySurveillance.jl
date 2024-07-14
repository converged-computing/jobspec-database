module SkySurveillance

using CUDA
using CommonRLInterface: AbstractEnv
using Dates: format, now
using Distributions: Normal, Sampleable, Uniform, pdf
using Flux
using Flux: glorot_uniform, mse
using IntervalSets
using JLD2
using POMDPTools.BeliefUpdaters: NothingUpdater, PreviousObservationUpdater
using POMDPTools.Simulators: Sim, run_parallel, stepthrough
using POMDPTools:
    Deterministic, HistoryRecorder, POMDPTools, RandomPolicy, RandomSolver, eachstep
using POMDPs:
    POMDP, POMDPs, Policy, Solver, Updater, discount, isterminal, reward, simulate, solve
using Plots: Plots, RGB, Shape, distinguishable_colors, mp4, pgfplotsx, plot, plot!, savefig
using Random: AbstractRNG, Xoshiro
using StaticArrays: SVector
using Statistics: mean, std, var
using TOML: parse, parsefile

export FlatPOMDP,
    MultiFilterUpdater,
    BeliefPOMDP,
    RandomSolver,
    SimpleGreedySolver,
    SequentialSolver,
    HighestVarianceSolver,
    SingleSweepSolver,
    Target,
    FlatState,
    SingleFilter, #just for betazero experiment 
    score_search,
    score_tracking

include("Flat_POMDP/types.jl")
include("Flat_POMDP/flat_pomdp.jl")
include("Flat_POMDP/belief_pomdp.jl")
include("Flat_POMDP/updater.jl")
include("Flat_POMDP/solver_random.jl")
include("Flat_POMDP/solver_sequential.jl") # Used in solver_single_sweep
include("Flat_POMDP/solver_single_sweep.jl") # Needs solver_sequential
include("Flat_POMDP/solver_highest_variance.jl")
include("Flat_POMDP/solver_simple_net.jl")
# include("Flat_POMDP/solver_particle_tree.jl")
# include("Flat_POMDP/solver_raw_network.jl")
include("Flat_POMDP/visualizations.jl")

end
