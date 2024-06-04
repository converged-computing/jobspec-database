#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 4096 # 4GB solicitados.
#SBATCH -p mlow # or mlow Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written
python task1_1.py -m "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" -d "fasterRCNN_50FPN_detections" -v "/home/group05/m6_dataset/vdo.avi"
python task1_1.py -m "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" -d "fasterRCNN_X_101_detections" -v "/home/group05/m6_dataset/vdo.avi"
python task1_1.py -m "COCO-Detection/retinanet_R_101_FPN_3x.yaml" -d "retinanet_101_detections" -v "/home/group05/m6_dataset/vdo.avi"
python task1_1.py -m "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml" -d "maskRCNN_101_detections" -v "/home/group05/m6_dataset/vdo.avi"
python task1_1.py -m "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" -d "maskRCNN_50FPN_detections" -v "/home/group05/m6_dataset/vdo.avi"
