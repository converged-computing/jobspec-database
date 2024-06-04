#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1
# set max wallclock time
#SBATCH --time=24:00:00 # set name of job
#SBATCH -J msweep

#small for 1 gpu, big for 4 or 8
#SBATCH --partition=big

# set number of GPUs
#SBATCH --gres=gpu:8

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
#vars=(EMNIST)

#net=vgg16
#net=cnn

#echo ${vars[$SLURM_ARRAY_TASK_ID]}.sh
##filename=scripts/${net}_${vars[$SLURM_ARRAY_TASK_ID]}.sh
#filename=scripts/${vars[$SLURM_ARRAY_TASK_ID]}.sh
#rm $filename
#echo '#!/bin/bash' > $filename
##echo './meta_script '${net}' '${vars[$SLURM_ARRAY_TASK_ID]} >> $filename
##echo './meta_script_msweep_jade '${vars[$SLURM_ARRAY_TASK_ID]}' cnn none 8' >> $filename
##echo './meta_script_msweep_jade '${vars[$SLURM_ARRAY_TASK_ID]}' fc none 8' >> $filename
#echo './meta_script_msweep_jade2 '${vars[$SLURM_ARRAY_TASK_ID]}' resnet50 max 8' >> $filename
#chmod +x $filename

dataset=$1
network=$2
pool=$3
n=$4
m=$5

filename=${dataset}_${network}_${pool}.sh
rm $filename
echo '#!/bin/bash' > $filename
echo '/home_directory/nn-pacbayes/meta_script_msweep_jade2 '${dataset}' '${network}' '${pool}' '${n}' '${m} >> $filename
chmod +x $filename

#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c ./densenet201.sh
#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c ./meta_script
#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c ./meta_script_layer_sweep

/jmain01/apps/docker/tensorflow-batch -v 19.09-py3 -c $filename

#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c $(echo ./meta_script $net vars[$SLURM_ARRAY_TASK_ID])
#/jmain01/apps/docker/tensorflow-batch -v 19.05-py2 -c $(echo ./meta_script $net vars[$SLURM_ARRAY_TASK_ID])
#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c $(echo ./meta_script_arch_sweep $net vars[$SLURM_ARRAY_TASK_ID])

#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c ./meta_script
#/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c ./meta_script
