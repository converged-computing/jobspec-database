#!/bin/bash

# set the number of nodes
#SBATCH --time=24:00:00
#SBATCH --job-name=single_core
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --partition=htc
##SBATCH --mem=32768

# set number of GPUs
#SBATCH --gres=gpu:p100:4

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=guillefix@gmail.com

##SBATCH --array=0-2
##SBATCH --array=0-0

#Launching the commands within script.sh

#archs=(vgg19 vgg16 resnet50 resnet101 resnet152 resnetv2_50 resnetv2_101 resnetv2_152 resnext50 resnext101 densenet121 densenet169 densenet201 mobilenetv2 nasnet)
#
#echo ${archs[$SLURM_ARRAY_TASK_ID]}.sh
#rm ${archs[$SLURM_ARRAY_TASK_ID]}.sh
#echo '#!/bin/bash' > ${archs[$SLURM_ARRAY_TASK_ID]}.sh
#echo './meta_script_arch_sweep '${archs[$SLURM_ARRAY_TASK_ID]} >> ${archs[$SLURM_ARRAY_TASK_ID]}.sh
#chmod +x ${archs[$SLURM_ARRAY_TASK_ID]}.sh

#vars=(0.1 0.3 0.6 1.0 1.3 1.6 2.0 2.3 2.6 3.0)
#vars=(500 1000 5000 10000 20000 30000 40000)
#vars=(20000 30000 40000)
#vars=(none max avg)
#vars=(mnist cifar)
#vars=(mnist)
#vars=(EMNIST)

#net=vgg16
#net=cnn

#echo ${vars[$SLURM_ARRAY_TASK_ID]}.sh
##filename=scripts/${net}_${vars[$SLURM_ARRAY_TASK_ID]}.sh
#filename=scripts/${vars[$SLURM_ARRAY_TASK_ID]}.sh
#rm $filename
#echo '#!/bin/bash' > $filename
##echo './meta_script '${net}' '${vars[$SLURM_ARRAY_TASK_ID]} >> $filename
#echo './meta_script_msweep '${vars[$SLURM_ARRAY_TASK_ID]}' fc none 4' >> $filename
#chmod +x $filename
module load anaconda3/2019.03
module load gpu/cuda/10.0.130
module load gpu/cudnn/7.5.0__cuda-10.0
module load mpi
 
source activate $DATA/tensor-env
#./meta_script_msweep_arc ${vars[$SLURM_ARRAY_TASK_ID]} fc none 1
./meta_script_msweep_arc mnist fc none 1

#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c ./densenet201.sh
#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c ./meta_script
#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c ./meta_script_layer_sweep

#/jmain01/apps/docker/tensorflow-batch -v 19.09-py3 -c $filename

#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c $(echo ./meta_script $net vars[$SLURM_ARRAY_TASK_ID])
#/jmain01/apps/docker/tensorflow-batch -v 19.05-py2 -c $(echo ./meta_script $net vars[$SLURM_ARRAY_TASK_ID])
#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c $(echo ./meta_script_arch_sweep $net vars[$SLURM_ARRAY_TASK_ID])

#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c ./meta_script
#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c ./meta_script
