#= Julia code for launching jobs on the slurm cluster.

This code is expected to be run from an sbatch script after a module load julia command has been run.
It starts the remote processes with srun within an allocation.
If you get an error make sure to Pkg.checkout("CluterManagers").

=#
using ClusterManagers
using Distributed
using DrWatson
@quickactivate :RLExp

n_workers = parse(Int, ENV["SLURM_NTASKS"])
addprocs_slurm(n_workers; topology = :master_worker, exeflags=["--project=.", "--color=yes"])

@everywhere using DrWatson
@everywhere @quickactivate :RLExp
include("RLExp_DUQN_Atari.jl")
# include("RLExp_GDQN_Atari.jl")
# include("Dopamine_DQN_Atari.jl")
# include("RLExp_NDQN_Atari.jl")

games = ["breakout"]
# experiments = [E`RLExp_BDQN_Atari($(game))` for game in games]
experiments = [E`RLExp_DUQN_Atari(pong)`]
# experiments = [E`Dopamine_DQN_Atari(pong)`]

# pmap(run, experiments)
run(experiments[1])
