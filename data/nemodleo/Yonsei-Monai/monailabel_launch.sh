#!/bin/bash

#SBATCH --partition=bc-mig
#SBATCH --ntasks=1
#SBATCH --nodes=1             # Max is 1
#SBATCH --gres=gpu:1          # Max is event specific
#SBATCH --time=8:00:00        # Max is event specific
##SBATCH --cpus-per-gpu=4      # limit CPU cores
#SBATCH --cpus-per-task=4

#${MYEXE} -l 60 2>&1 | tee out.${SLURM_JOBID}

##removing the old port forwading
squeue -u $USER > no_jobs.txt
back=`wc -l no_jobs.txt`
nl=${back:0:1}
echo $nl
echo "--------"
if [ $nl -ne "2" ];then
        echo "Multiple job submission not allowed. Please kill the old job using scancel command"
        exit
else
        echo "Single Job command"
fi
#getting the port and dgx node name
SERVER="`hostname`"

SNL=$SLURM_NODELIST
SJG=$SLURM_JOB_GPUS
CVD=$CUDA_VISIBLE_DEVICES
#echo $CVD

IFS='/' read -ra IDs <<<  "$CVD"
A=${SNL:4:5}  # dgx node id
B=$SJG        # GPU instance
C=${IDs[1]}   # MIG GPU Instance
D=${IDs[2]}   # MIG Compute Instance
#echo $A $B $C $D
PORT_JU=$(( 8000 +  $B*100 + $C*10 +$D  ))
PORT_AA=$(( 9000 +  $B*100 + $C*10 +$D  ))
#echo $PORT

#PORT=`shuf -i 8000-9000 -n 1`
HASHNUMBER=$1

echo $SERVER
echo $PORT_JU  $PORT_AA
echo $HASHNUMBER


rm ~/port_forwarding_command
# Launch the Jupyter Notebook Server
set -x
mkdir -p /workspaces/$USER
cd /workspaces/$USER


# Launch the Jupyter Notebook Server


DIR="/workspaces/$USER/workspace-monailabel-labs"
if [ ! -d "$DIR" ]; then
   echo "Installing labs files in ${DIR}..."
   cp -rT /lustre/shared/bootcamps/monailabel/workspace-monailabel-labs /lustre/workspaces/$USER/workspace-monailabel-labs
   echo "dataset copy : done"
   singularity run --bind /lustre/workspaces /lustre/shared/bootcamps/monailabel/monai-x1.simg
	echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} -L localhost:9999:${SERVER}:${PORT_AA} ssh.axisapps.io -l ${HASHNUMBER}"> ~/port_forwarding_command
   singularity run --nv --bind /lustre/workspaces /lustre/shared/bootcamps/monailabel/monai-x1.simg jupyter lab --notebook-dir=/lustre/workspaces/$USER/workspace-monailabel-labs --port=$PORT_JU --ip=0.0.0.0 --no-browser --NotebookApp.token=""
else
  echo "Config files already there."
  echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} -L localhost:9999:${SERVER}:${PORT_AA} ssh.axisapps.io -l ${HASHNUMBER}"> ~/port_forwarding_command

  singularity run --nv --bind /lustre/workspaces /lustre/shared/bootcamps/monailabel/monai-x1.simg jupyter lab --notebook-dir=/lustre/workspaces/$USER/workspace-monailabel-labs --port=$PORT_JU --ip=0.0.0.0 --no-browser --NotebookApp.token=""
fi
