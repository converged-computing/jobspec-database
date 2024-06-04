#!/bin/bash
#SBATCH --partition=learnfair
#SBATCH --job-name=evalopenclip
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:8
#SBATCH --open-mode=append
#SBATCH --exclusive
#SBATCH --output=/data/home/amroabbas/projects/open_clip/src/jobs/eval.%j_%A.out
#SBATCH --error=/data/home/amroabbas/projects/open_clip/src/jobs/eval.%j_%A.error
#SBATCH --time=25
#SBATCH --array=18-36:2

export MASTER_PORT=12802

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export PYTHONPATH="$PYTHONPATH:$PWD/src"

keep_complex_samples="False"
keep_hard="True"

source /data/home/amroabbas/projects/clipenv1/bin/activate

srun --cpu_bind=v --accel-bind=gn python -u training/main.py \
    --imagenet-val /datasets01/imagenet_full_size/061417/val \
    --model ViT-B-32 \
    --pretrained /checkpoint/amroabbas/datapruning/openclip-for-density-based-pruning/new_exps/random/amro_random_0.6_cont_Apr24/checkpoints/epoch_${SLURM_ARRAY_TASK_ID}.pt \
    --logs /checkpoint/amroabbas/datapruning/openclip-for-density-based-pruning/new_exps/eval/eval-random/amro_random_0.6 \
    --name eval_ep${SLURM_ARRAY_TASK_ID}



#   --pretrained /checkpoint/amroabbas/datapruning/openclip_multimodal_pruning/qp_pruning/text_only/text-only-openclip-Vit-b32-qp-pruning-clip-score-filtered-60%-laiondedup280m-16n-352bs-45ep/checkpoints/epoch_${SLURM_ARRAY_TASK_ID}.pt \
#   --logs /checkpoint/amroabbas/datapruning/openclip_multimodal_pruning/eval/eval-qp_pruning/text_only/text-only-openclip-Vit-b32-qp-pruning-clip-score-filtered-60%-laiondedup280m-16n-352bs-45ep/ \
    
