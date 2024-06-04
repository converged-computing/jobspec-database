#!/bin/bash -l
#SBATCH --job-name=smplx-obj-write
#SBATCH --output=output_slurm/write_obj_log.txt
#SBATCH --error=output_slurm/write_obj_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20g
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=shdpm0
#SBATCH --partition=spgpu
#SBATCH --array=1-10

##### END preamble

my_job_header

conda activate soma3.7

module load clang/2022.1.2
module load gcc/10.3.0
module load gcc/13.2.0
module load intel/2022.1.2
module load boost/1.78.0
module load eigen tbb
module load blender
module list

# mkdir output_slurm

slurm_name=$SLURM_JOB_NAME
slurm_task_id=$SLURM_ARRAY_TASK_ID

cd transfer_model
python -u write_obj.py \
--model-folder ../models/ \
--motion-file /nfs/turbo/coe-shdpm/leyang/VEHS-7M/Mesh/SOMA_SMPLX_pkl/ \
--output-folder /nfs/turbo/coe-shdpm/leyang/VEHS-7M/Mesh/SMPLX_obj/ \
--model-type smplx \
--batch-moshpp \
--batch-id $slurm_task_id

