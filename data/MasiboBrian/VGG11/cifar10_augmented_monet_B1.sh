#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}
export SCRATCH_DISK=/disk/scratch
export SCRATCH_HOME=${SCRATCH_DISK}/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

export dest_path_model=${SCRATCH_HOME}/Resnet

export dest_path_data=${SCRATCH_HOME}/Resnet/data

rm -r ${dest_path_data}/
rm -r ${dest_path_model}/

mkdir -p ${dest_path_data}

#####################################################################################################################################
echo "Moving all model files"

rsync --archive --update --compress /home/${STUDENT_ID}/Code/Cluster/cifar10_augmented_B1_monet/*.py ${dest_path_model}/

echo "done"

echo "moving data"

mkdir -p ${dest_path_data}/Monet/

rsync --archive --update --compress /home/${STUDENT_ID}/Code/Cluster/cifar10_augmented_B1_monet/data/cifar-10-batches-py ${dest_path_data}/
rsync --archive --update --compress /home/${STUDENT_ID}/Code/Cluster/cifar10_augmented_B1_monet/data/Monet/train_*_*_fake.png ${dest_path_data}/Monet/

echo "done. verifying."

ls ${dest_path_data}

echo "done."

#######################################################################################################################################
# Activate the relevant virtual environment:
echo "Activating conda environment"
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ${SCRATCH_HOME}

########################################################################################################################################

cd ${dest_path_model}

echo "running model"
python train_cifar10_augmented.py --num_epochs 300 --learning_rate 0.1 --weight_decay 0.0001

echo "Done."

#########################################################################################################################################
#Move data back to DFS

echo "Moving results back to DFS"

mkdir -p /home/${STUDENT_ID}/Code/Cluster/cifar10_augmented_B1_monet/Results/

rsync --archive --update --compress /disk/scratch/s2089883/Resnet/exp_1/result_outputs /home/${STUDENT_ID}/Code/Cluster/cifar10_augmented_B1_monet/Results/
rsync --archive --update --compress /disk/scratch/s2089883/Resnet/exp_1/saved_models/train_model_latest /home/${STUDENT_ID}/Code/Cluster/cifar10_augmented_B1_monet/Results/

echo "Done."


#clearing folders
rm -r ${dest_path_data}/
rm -r ${dest_path_model}/

#########################################################################################################################################
