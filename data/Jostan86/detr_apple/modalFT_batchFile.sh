#!/bin/bash
#SBATCH -J modalFT_r101						  # name of job
#SBATCH -p dgx								  # name of partition or queue
#SBATCH -o /nfs/hpc/share/browjost/detr_apple/logdirs/modalFT_r101dc5/modalFineTune-%a.out			  # name of output file for this submission script
#SBATCH -e /nfs/hpc/share/browjost/detr_apple/logdirs/modalFT_r101dc5/modalFineTune-%a.err				  # name of error file for this submission script1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dgx2-3

module load python3
source /nfs/hpc/share/browjost/detr_apple/venv/bin/activate

# run my job (e.g. matlab, python)
srun --export ALL python3 -m torch.distributed.launch --nproc_per_node=1 --use_env main2.py --coco_path /nfs/hpc/share/browjost/detr_apple/coco_apples/ --batch_size 2 --resume /nfs/hpc/share/browjost/detr_apple/weights/detr-r101-dc5_no-class-head.pth --epochs 50 --output_dir /nfs/hpc/share/browjost/detr_apple/logdirs/modalFT_r101dc5 --dataset_file coco_apples_modal --backbone resnet101 --dilation
