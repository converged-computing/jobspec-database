#!/bin/bash
#
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20000           # 20 GB RAM 
#SBATCH -t 360                # time (minutes)
#SBATCH -o /scratch/gpfs/jbreda/ephys/kilosort/W122/logs/output_%a_%j.out
#SBATCH -e /scratch/gpfs/jbreda/ephys/kilosort/W122/logs/error_%a_%j.err


# where the directorys containing .bin files are 
input_base_path="/scratch/gpfs/jbreda/ephys/kilosort/W122/preprocessed_W122_19523713" 

# where the Brody_Lab_Ephys repo is
repo_path="/scratch/gpfs/jbreda/ephys/kilosort/Brody_Lab_Ephys"

# where the config and channel map info are (inputs to main_kilosort fx)
config_path="/scratch/gpfs/jbreda/ephys/kilosort/Brody_Lab_Ephys/utils/cluster_kilosort"
 
# step 1: get list of all directories & array index

echo "Array Index: $SLURM_ARRAY_TASK_ID"

cd $input_base_path
bin_folders=`ls -d */`
bin_folders_arr=($bin_folders)
arr=$SLURM_ARRAY_TASK_ID

# step 2: pass a bin folder in using array task id to kilosort

cd $config_path

# load matlab
module purge
module load matlab/R2018b

# call main kilosort_wrapper the 500 = time in seconds where the sorting starts, skipping first chunck bc very noisy
	matlab -singleCompThread -nosplash -nodisplay -nodesktop -r "main_kilosort_forcluster_parallel_wrapper('${bin_folders_arr[${arr}]}','${input_base_path}','${config_path}','${repo_path}', 500);exit"

