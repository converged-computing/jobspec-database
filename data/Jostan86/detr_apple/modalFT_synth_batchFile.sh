#!/bin/bash
#SBATCH -J modalFineTune						  # name of job
#SBATCH -p dgx								  # name of partition or queue
#SBATCH -o /nfs/hpc/share/browjost/detr_apple/logdirs/modalFT_sy/modalFineTune-%a.out			  # name of output file for this submission script
#SBATCH -e /nfs/hpc/share/browjost/detr_apple/logdirs/modalFT_sy/modalFineTune-%a.err				  # name of error file for this submission script1
#SBATCH --gres=gpu:2

module load python3
source /nfs/hpc/share/browjost/detr_apple/venv/bin/activate

# run my job (e.g. matlab, python)
srun --export ALL python3 -m torch.distributed.launch --nproc_per_node=2 --use_env main2.py --coco_path /nfs/hpc/share/browjost/detr_apple/coco_apples_synth/ --batch_size 2 --resume /nfs/hpc/share/browjost/detr_apple/weights/detr-r50_no-class-head.pth --epochs 50 --output_dir /nfs/hpc/share/browjost/detr_apple/logdirs/modalFT_sy --dataset_file coco_apples_modal_synth
