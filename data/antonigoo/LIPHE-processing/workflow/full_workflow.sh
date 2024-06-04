#!/bin/bash
#SBATCH --job-name=full_workflow
#SBATCH --account=project_2008498
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30GB
#SBATCH --partition=small
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge
export PATH="/projappl/project_2003180/samantha/bin:$PATH"

BASE_PATH="/scratch/project_2008498/antongoo/fgi/snakemake/"
BASE_NAME="200406_100502_Sample_resample005_"

input_las="${BASE_PATH}${BASE_NAME}.laz"
output_las="${BASE_PATH}output/${BASE_NAME}/${BASE_NAME}_normalized.las"
python 01/01_add_parameters_and_normalize.py $input_las $output_las

input_las="${BASE_PATH}output/${BASE_NAME}/${BASE_NAME}_normalized.las"
output_las="${BASE_PATH}output/${BASE_NAME}/${BASE_NAME}_georef.las"
python 02/02_georeference.py $input_las $output_las

input_las="${BASE_PATH}output/${BASE_NAME}/${BASE_NAME}_georef.las"
output_dir="${BASE_PATH}output/${BASE_NAME}/single_trees/"
python 04/04_clipping_trees.py $input_las $output_dir

input_las_dir="${BASE_PATH}output/${BASE_NAME}/single_trees/"
output_dir="${BASE_PATH}output/${BASE_NAME}/single_trees_normalized_to_ground/"
python 05/05_normalize_to_ground.py $input_las_dir $output_dir

input_las_dir="${BASE_PATH}output/${BASE_NAME}/single_trees_normalized_to_ground/"
output_las_dir=/"${BASE_PATH}output/${BASE_NAME}/fine_segmentation/"
output_noise_dir="${BASE_PATH}output/${BASE_NAME}/fine_segmentation_noise/"
python 06/06_fine_segmentation.py $input_las_dir $output_las_dir $output_noise_dir
