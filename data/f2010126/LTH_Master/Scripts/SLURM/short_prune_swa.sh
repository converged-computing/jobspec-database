#!/bin/bash -l
#SBATCH -o /work/dlclarge1/dsengupt-lth_ws/nemo_logs/shortSWA_res1.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e /work/dlclarge1/dsengupt-lth_ws/nemo_logs/shortSWA_res1.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J ResNet_ShortPrune1
#SBATCH -N 1
#SBATCH -t 19:59:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de

cd $(ws_find lth_ws)
# python3 -m venv lth_env
source lth_env/bin/activate
pip list
cd LTH_Master

#pip install numpy matplotlib torch tensorboard torchvision pandas
#pip install pytorch-model-summary pytorch-lightning torchsummary

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

python3 -m src.vanilla_pytorch.shortcut_pruning --model Resnets --batch-size 512 --epochs 30 --lr 0.01 --dataset cifar10 --use-swa --name ResNet_IMP_SWA1

deactivate
