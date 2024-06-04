#!/bin/bash
#SBATCH --job-name=llm_math_debate_sft
#SBATCH --open-mode=append
#SBATCH --output=job_outputs/%x_%j.out
#SBATCH --error=job_outputs/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --mem=96G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/llm-math-debate.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "

source /ext3/env.sh

python -m llm_math_debate.training.sft
python -m llm_math_debate.training.sft_test
"
