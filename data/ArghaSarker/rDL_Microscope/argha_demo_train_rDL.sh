#!/bin/bash
#SBATCH --time=10:00:00 # Run time
#SBATCH --nodes 2  # Number of reaquested nodes 
#SBATCH --ntasks-per-node=1
##SBATCH --mem 600GB
#SBATCH -c 54
#SBATCH -p gpu
#SBTACH --job-name A100-test-first-run
#SBATCH --error=A100_multi_error.o%j
#SBATCH --output=A100_multi_output.o%j
#SBATCH --requeue
#SBATCH --mail-user=asarker@uni-osnabrueck.de

#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
##SBATCH --mail-type=ALL
#SBATCH --signal=SIGTERM@90
echo "running in shell: " "$SHELL"
export NCCL_SOCKET_IFNAME=lo

## Please add any modules you want to load here, as an example we have commented out the modules
## that you may need such as cuda, cudnn, miniconda3, uncomment them if that is your use case 
## term handler the function is executed once the job gets the TERM signal


spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate thesis


srun bash ./src/demo_train_rDL_SIM_Model.sh
