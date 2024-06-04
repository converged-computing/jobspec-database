#!/bin/bash
#SBATCH -J modalSeg						  # name of job
#SBATCH -p dgx								  # name of partition or queue
#SBATCH -o /nfs/hpc/share/browjost/detr_apple/logdirs/modalSeg/modalSeg-%a.out			  # name of output file for this submission script
#SBATCH -e /nfs/hpc/share/browjost/detr_apple/logdirs/modalSeg/modalSeg-%a.err				  # name of error file for this submission script
#SBATCH -t 0-18:00:00                # time limit for job
#SBATCH --gres=gpu:2

module load python3
source /nfs/hpc/share/browjost/detr_apple/venv/bin/activate

# run my job (e.g. matlab, python)
srun --export ALL python3 -m torch.distributed.launch --nproc_per_node=2 --use_env main2.py --masks --lr_drop 15 --coco_path /nfs/hpc/share/browjost/detr_apple/coco_apples/ --batch_size 2 --epochs 25 --output_dir /nfs/hpc/share/browjost/detr_apple/logdirs/modalSeg --dataset_file coco_apples_modal --frozen_weights /nfs/hpc/share/browjost/detr_apple/logdirs/modalFT/checkpoint.pth
