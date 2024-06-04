#!/bin/bash -l
#SBATCH -o /work/dlclarge1/dsengupt-lth_ws/nemo_logs/all_swa.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e /work/dlclarge1/dsengupt-lth_ws/nemo_logs/all_swa.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J Net2_Prune
#SBATCH -N 1
#SBATCH -t 19:59:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de

cd $(ws_find lth_ws)
# python3 -m venv lth_env
source lth_env/bin/activate
pip list
cd LTH_Master

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"



echo "Start IMP for Net2"
python3 -m src.vanilla_pytorch.shortcut_pruning --model Net2 --batch-size 60 --epochs 30 --lr 2e-4 --dataset cifar10 --name Net2_IMP_Run1

echo "Starting SWA for Net2"
python3 -m src.vanilla_pytorch.shortcut_pruning --model Net2 --batch-size 60 --epochs 30 --lr 2e-4 --dataset cifar10 --use-swa --name Net2SWARun1

echo "Starting 2nd SWA for Resnet"
python3 -m src.vanilla_pytorch.shortcut_pruning --model Resnets --batch-size 512 --epochs 30 --lr 0.01 --dataset cifar10 --use-swa --name ResNet_SWARun2


deactivate
