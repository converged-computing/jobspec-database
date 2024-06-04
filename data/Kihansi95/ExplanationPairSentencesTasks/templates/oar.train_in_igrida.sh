#!/usr/bin/env bash
#OAR -n model_dataset_target
#OAR -l {host='igrida-abacus3.irisa.fr'}/nodes=1/gpu_device=1,walltime=48:00:00
#OAR -O /srv/tempdd/dunguyen/RUNS/OARLOG/%jobid%.out.log
#OAR -E /srv/tempdd/dunguyen/RUNS/OARLOG/%jobid%.err.log

#patch to be aware of "module" inside a job
. /etc/profile.d/modules.sh
module load spack/cuda/11.3.1
module load spack/cudnn/8.0.4.30-11.0-linux-x64
conda deactivate
source $VENV/eps/bin/activate
EXEC_FILE=src/lstm_attention.py
echo
echo =============== RUN ${OAR_JOB_ID} ===============
echo Run $EXEC_FILE at `date +"%T, %d-%m-%Y"`

python $EXEC_FILE -o $RUNDIR -b 512 -e 50 --vectors glove.840B.300d -m exp --name $OAR_JOB_NAME --data esnli --lambda_supervise 0.5 --n_lstm 1 --version run=2_lstm=1_lsup=0.5

echo Done