#!/bin/bash

#SBATCH --job-name=llmcompr

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=V.Moskvoretskii@skoltech.ru

#SBATCH --output=zh_logs/gptvq_log.txt
#SBATCH --time=0-05

#SBATCH --mem=32G

#SBATCH --nodes=1

#SBATCH -c 8

#SBATCH --gpus=1

srun singularity exec --bind /trinity/home/v.moskvoretskii/:/home -f --nv /trinity/home/v.moskvoretskii/images/gptvq.sif bash -c '
    cd /home;
    export HF_TOKEN=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;
    export SAVING_DIR=/home/cache/;
    export HF_HOME=/home/cache/;
    export TRANSFORMERS_CACHE=/home/cache/;
    export WANDB_API_KEY=2b740bffb4c588c7274a6e8cf4e39bd56344d492;
    export CUDA_LAUNCH_BLOCKING=1;
    cd /home/LLM_Compression/gptvq;
    nvidia-smi;
    pip list;
    python llama.py meta-llama/Llama-2-7b-hf wikitext2 --columns-per-group 128 --use-vq --kmeans-iters 100 --kmeans-init-method mahalanobis --hessian-weighted-lookups --include-m-step --wbits 2 --vq-dim 2 --groupsize 2048 --codebook-bitwidth 8 --quantize-per-codebook  --output-dir /home/cache/gptvq_llama7b/
'

