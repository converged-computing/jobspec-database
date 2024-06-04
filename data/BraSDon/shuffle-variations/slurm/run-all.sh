#!/bin/bash

# TODO: Test this entire file!
config_path="../run-configs/config.yaml"
time=$(yq eval '.time' $config_path)
nodes=$(yq eval '.nodes' $config_path)
gpus_per_node=$(yq eval '.gpus-per-node' $config_path)
ntasks=$(($nodes * $gpus_per_node))

SBATCH_SCRIPT="#!/bin/bash

#SBATCH --time=$time
#SBATCH --partition=accelerated
#SBATCH --nodes=$nodes
#SBATCH --ntasks=$ntasks
#SBATCH --gres=gpu:$gpus_per_node
#SBATCH --account='hk-project-madonna'


module load compiler/gnu/11
module load devel/cuda/11.8
module load mpi/openmpi/4.1

source /hkfs/work/workspace/scratch/tz6121-paper/paper/venv/bin/activate

cd ..

srun python -u src/main.py --config_path $config_path"

# For each case (pre_step_local, pre_step_noshuffle, baseline, etc.),
# adjust script by appending --case <case> to the end of the last line.
# And then run the script using sbatch.
echo "$SBATCH_SCRIPT --case baseline" > baseline.sh
echo "$SBATCH_SCRIPT --case pre_step_local" > pre_step_local.sh
echo "$SBATCH_SCRIPT --case pre_step_noshuffle" > pre_step_noshuffle.sh
echo "$SBATCH_SCRIPT --case asis_step_local" > asis_step_local.sh
echo "$SBATCH_SCRIPT --case asis_step_noshuffle" > asis_step_noshuffle.sh
echo "$SBATCH_SCRIPT --case asis_seq_local" > asis_seq_local.sh
echo "$SBATCH_SCRIPT --case asis_seq_noshuffle" > asis_seq_noshuffle.sh

sbatch baseline.sh
sbatch pre_step_local.sh
sbatch pre_step_noshuffle.sh
sbatch asis_step_local.sh
sbatch asis_step_noshuffle.sh
sbatch asis_seq_local.sh
sbatch asis_seq_noshuffle.sh

# Remove the scripts after running them.
rm baseline.sh
rm pre_step_local.sh
rm pre_step_noshuffle.sh
rm asis_step_local.sh
rm asis_step_noshuffle.sh
rm asis_seq_local.sh
rm asis_seq_noshuffle.sh