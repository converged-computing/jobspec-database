#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=out
#SBATCH --error=err
#SBATCH --time=06:00:00
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
source ~/pyenv
ml purge
ml Stages/Devel-2019a
ml GCC/8.3.0
ml ParaStationMPI/5.4.4-1
ml CUDA/10.1.105
ml NCCL/2.4.6-1-CUDA-10.1.105
ml cuDNN/7.5.1.10-CUDA-10.1.105
export NCCL_DEBUG=INFO
export NCCL_IB_CUDA_SUPPORT=0
export NCCL_IB_DISABLE=1
#rm -fr  training_artifacts/imagenet-256
srun python train_alae.py --config-file configs/imagenet.yaml
