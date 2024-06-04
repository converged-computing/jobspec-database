#!/bin/bash

#SBATCH --job-name yolo_train
#SBATCH --error error_%j.txt
#SBATCH --output output_%j.txt
#SBATCH --mail-user pietro.girotto@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 23:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 70G
#SBATCH --gres=gpu:rtx:2

model_name=medium_lv_mhp_merged_b16
dataset_path=LV-MHP-v1-YOLO-merge

cd $WORKING_DIR/Code/YOLOv8_mhp
mkdir $model_name
cd $model_name
echo $PWD

echo "Sourcing the .bashrc file"
source /home/girottopie/.bashrc
echo "Sourced!"

echo "Donwloading yolo env..."
gdrivedownload 1Y5F4GiSOkCblF3zkXWj50OXsTKPzh478 /ext/yolo_env.sif
echo "YOLO env donwloaded!"

srun singularity exec --nv /ext/yolo_env.sif yolo segment train data=../$dataset_path/dataset.yaml model=yolov8m-seg.pt epochs=100 imgsz=640 batch=16 device=0,1

mv ../*.txt ./
