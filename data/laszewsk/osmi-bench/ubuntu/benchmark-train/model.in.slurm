#!/usr/bin/env bash

#SBATCH --job-name={identifier}
#SBATCH --output=osmi-{identifier}-%u-%j.out
#SBATCH --error=osmi-{identifier}-%u-%j.err
{slurm.sbatch}
#SBATCH --partition=batch
#SBATCH --account=gen150_smartsim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time={time.{experiment.model}}

PROGRESS() {
    echo "# ###############################################"
    echo "# cloudmesh status=$1 progress=$2 msg=$3 pid=$$"
    echo "# ###############################################"
}

VENV=ENV3-OSMI

USER_NAME=`whoami`
GROUP_NAME=`id -g`
USER_ID=`id -u`
GROUP_ID=`getent group | fgrep ${GROUP_NAME} | cut -d":" -f 3`


PROGRESS "running" "modules" 1

if echo "$hostname" | grep -q "crusher"; then
    echo "Module load on 'crusher'"

    source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh

    module load cray-python
    module load rocm
    module list
    WITH_CONTAINER="NONE"

elif echo "$hostname" | grep -q "summit"; then
    echo "Module load on 'summit'"
    
    module load open-ce/1.1.3-py38-0
    module load cuda/11.0.2
    WITH_CONTAINER="NONE"

else
    echo "Regular Linux machine. No module load"
    # WITH_CONTAINER="apptainer"
    # WITH_CONTAINER="docker"
    CONTAINER_DIR=../../../../images
fi


PROGRESS "running" "python" 1


source ../../../$VENV/bin/activate

which python
python --version
python ../../../test-tf.py


PROGRESS "running" "gpus" 1

if echo "$hostname" | grep -q "crusher"; then
    rocm-smi
else
    nvidia-smi
fi  

MODELS_DIR=./models

#export USER_SCRATCH=/scratch/$USER


RESULT_DIR=`pwd`

#if [ -z "$OSMI_TARGET" ]; then
#    echo "OSMI_TARGET is not set"
#    exit 1
#fi
#export CONTAINER_DIR=$OSMI_TARGET/images/

#module purge
#module load apptainer

MODEL={experiment.model}
CONTAINER=$CONTAINER_DIR/cloudmesh-tfs-23-10-nv.sif

echo "============================================================"
echo "PROJECT_ID: {identifier}"
echo "MODELS_DIR: $MODELS_DIR"
echo "MODEL: $MODEL"
echo "REPEAT: {experiment.repeat}"

PROGRESS "running" "training" 3

cd $MODELS_DIR

# time apptainer exec --nv $CONTAINER python train.py $MODEL
#time apptainer exec --nv $CONTAINER python train.py $MODEL
# time python train.py $MODEL 2>&1 | tee | tr -d '\033\b' > $RESULT_DIR/train.log

if [ "$WITH_CONTAINER" = "apptainer" ]; then
    time apptainer exec --nv $CONTAINER python train.py $MODEL 2>&1 | tr -d '\033\b' | tee $RESULT_DIR/train.log
elif [ "$WITH_CONTAINER" = "docker" ]; then
    #docker run --gpus all -v $(pwd):$(pwd) $CONTAINER python train.py $MODEL 2>&1 | tr -d '\033\b' | tee $RESULT_DIR/train.log

	docker run -it \
		-v /home/${USER}:/home/${USER} \
		-v ${PWD}:${PWD} \
		-u ${USER_ID}:${GROUP_NAME} \
		-w ${PWD} \
		--gpus all \
	    --volume="/etc/group:/etc/group:ro" \
    	--volume="/etc/passwd:/etc/passwd:ro" \
    	--volume="/etc/shadow:/etc/shadow:ro" \
		osmi.docker /bin/bash -c "cd ${PWD}; python train.py $MODEL"	

else
    time python train.py $MODEL 2>&1 | tr -d '\033\b' | tee $RESULT_DIR/train.log
fi

PROGRESS "completed" "done" 100

#tr -cd '\11\12\15\40-\176' < $OUTPUT > tmp-output
#mv tmp-output $OUTPUT
