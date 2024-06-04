# Bash script in order to submit a job to the cluster
#!/bin/bash
#SBATCH --job-name=redistricting
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --nodes=2
#SBATCH --tasks-per-node=4

module load julia/1.6.1
# julia your_script.jl
julia mggg.jl