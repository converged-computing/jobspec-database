#!/bin/bash
#SBATCH --mail-user=chenchal.subraveti@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=maxwell
#SBATCH --account=palmeri_gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=a_accre_train
#SBATCH --output=/scratch/subravcr/trainedImagenet/myModels/xferLearning/a_accre_xfer_train_%A.out

setpkgs -a matlab_r2016b
setpkgs -a gcc_compiler_4.9.3
setpkgs -a cuda7.5
setpkgs -a cudnn7.5-v5
setpkgs -a matlab_r2016b
echo "SLURM_JOBID: "$SLURM_JOBID

# test_flag[1=test, 0=real]
testMode=0

baseResultDir="/scratch/subravcr/trainedImagenet/myModels/xferLearning"
baseNetToUse="${baseResultDir}/net-epoch-20/base-net-epoch-20.mat"
echo "    testMode: "$testMode
echo "baseNetTouse: "$baseNetToUse

# bash a_accre_train.sh $SLURM_JOBID $testMode $baseNetToUse
bash a_accre_train.sh $SLURM_JOBID $testMode ${baseNetToUse}



