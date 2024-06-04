#!/bin/bash
#SBATCH --mem-per-cpu=128G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100:1

module load nvidia/11.1
module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate idenv

python3 ../analyze/analyze_resnet_101.py --augment JPEGImages --model pascal_voc
python3 ../analyze/analyze_resnet_101.py --augment horizontal_shift --model pascal_voc
python3 ../analyze/analyze_resnet_101.py --augment vertical_shift --model pascal_voc
python3 ../analyze/analyze_resnet_101.py --augment vertical_flip --model pascal_voc
python3 ../analyze/analyze_resnet_101.py --augment horizontal_flip --model pascal_voc
python3 ../analyze/analyze_resnet_101.py --augment channel_shift --model pascal_voc
python3 ../analyze/analyze_resnet_101.py --augment rotation --model pascal_voc

python3 ../analyze/analyze_resnet_101.py --augment image_2 --model kitti
python3 ../analyze/analyze_resnet_101.py --augment horizontal_shift --model kitti
python3 ../analyze/analyze_resnet_101.py --augment vertical_shift --model kitti
python3 ../analyze/analyze_resnet_101.py --augment vertical_flip --model kitti
python3 ../analyze/analyze_resnet_101.py --augment horizontal_flip --model kitti
python3 ../analyze/analyze_resnet_101.py --augment channel_shift --model kitti
python3 ../analyze/analyze_resnet_101.py --augment rotation --model kitti

python3 ../analyze/analyze_resnet_101.py --augment test2017 --model coco
python3 ../analyze/analyze_resnet_101.py --augment horizontal_shift --model coco
python3 ../analyze/analyze_resnet_101.py --augment vertical_shift --model coco
python3 ../analyze/analyze_resnet_101.py --augment vertical_flip --model coco
python3 ../analyze/analyze_resnet_101.py --augment horizontal_flip --model coco
python3 ../analyze/analyze_resnet_101.py --augment channel_shift --model coco
python3 ../analyze/analyze_resnet_101.py --augment rotation --model coco

conda deactivate
