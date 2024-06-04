#!/bin/bash
#SBATCH -J amodalSeg						  # name of job
#SBATCH -p dgx								  # name of partition or queue
#SBATCH -o /nfs/hpc/share/browjost/detr_apple/logdirs/amodalSeg/amodalSeg-%a.out			  # name of output file for this submission script
#SBATCH -e /nfs/hpc/share/browjost/detr_apple/logdirs/amodalSeg/amodalSeg-%a.err				  # name of error file for this submission script
#SBATCH --gres=gpu:2
#SBATCH -t 0-18:00:00                # time limit for job (HH:MM:SS)

module load python3
source /nfs/hpc/share/browjost/detr_apple/venv_amodal/bin/activate

# run my job (e.g. matlab, python)
srun --export ALL python3 -m torch.distributed.launch --nproc_per_node=2 --use_env main2.py --masks --lr_drop 15 --coco_path /nfs/hpc/share/browjost/detr_apple/coco_apples/ --batch_size 2 --epochs 25 --output_dir /nfs/hpc/share/browjost/detr_apple/logdirs/amodalSeg --dataset_file coco_apples_amodal --frozen_weights /nfs/hpc/share/browjost/detr_apple/logdirs/amodalFT/checkpoint.pth
