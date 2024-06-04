#!/bin/bash
# 
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# TODO:
#   - Set name of the job below changing "Test" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch examle_submit.sh

# -= Resources =-
#
#SBATCH --job-name=Test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --constraint=tesla_t4|tesla_k80
#SBATCH --mem-per-cpu=5G
#SBATCH --time=24:00:00
#SBATCH --output=test-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=idemir18@ku.edu.tr
#SBATCH --account=ai
################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

## Load Python 3.6.3
echo "Activating Python 3.9.5"
module load python/3.9.5
module load anaconda/3.21.05
module load cuda/11.3
source activate sim_env
## Load GCC-7.2.1
echo "Activating GCC-7.2.1..."
module load gcc/7.2.1

echo ""
echo "======================================================================================"
env
echo "======================================================================================"
echo ""

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "Running Example Job...!"
echo "==============================================================================="

output_path="../../results/v1/masks.hdf5"
img_dir="../../GQA_ImageSet"
scene_graph="../../scenegraph_generation/results/generated_sg.json"

echo "detectron"
python generate_masks.py detectron2 --img-dir $img_dir --scene-graph $scene-graph  --output $output_path --use-gpu
sleep 60

echo "detic"
python generate_masks.py detic --img-dir $img_dir --scene-graph $scene_graph --output $output_path --use-gpu
sleep 60

echo "detic + custom vocabulary + attributes"
python generate_masks.py detic --img-dir $img_dir --scene-graph $scene_graph --output $output_path --custom-vocabulary --include-attributes --use-gpu

