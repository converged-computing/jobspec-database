#!/bin/sh
#$ -S /bin/sh
# -- SBATCH -- options
#SBATCH --time=12:00:00
#SBATCH --job-name="mnist-wgan"
#SBATCH --output=mnist-wgan.out
#SBATCH --mail-user=anderscs@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --mem=12000
# For using GPU
#SBATCH --partition=EPICALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load goolfc/2017b
module load TensorFlow/1.7.0-Python-3.6.3
source ~/venv/bin/activate


uname -a

export PROJECT=CompressedDNN
if [[ ! -d ${PROJECT} ]]; then
 echo "Can't find $PROJECT."
 echo "Home is $HOME"
else
 cd ${PROJECT}
fi

# Set env
. ./get_env.sh


# Parameters
MODEL=cifar10_wgan
EPOCH=10000
ACTION=train_model
GPU=0
CONFIG=mnist-wgan.yaml

python do.py --config ${CONFIG} --gpu ${GPU}
