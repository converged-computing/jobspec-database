#!/bin/bash
#PBS -l nodes=1:ppn={cpus}
#PBS -l pmem={memory}
#PBS -l mem={memory}
#PBS -l vmem={memory}
#PBS -l pvmem={memory}
#PBS -l walltime={walltime}
#PBS -o /home/vbasu/scripts/simulation_scripts/dagfiles/22010/processing/step_0_inject_veto_muons/logs/step_0_inject_veto_muons_run_1_${PBS_JOBID}.out
#PBS -e /home/vbasu/scripts/simulation_scripts/dagfiles/22010/processing/step_0_inject_veto_muons/logs/step_0_inject_veto_muons_run_1_${PBS_JOBID}.err
#PBS -q long
#PBS -S /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/icetray-start
FINAL_OUT=/data/user/vbasu/simulation_scripts/00000-00999/Level0.0_nugen_IC86.2012_pass2.022010.000001.i3.bz2
KEEP_CRASHED_FILES=0


echo 'Starting job on Host: '$HOSTNAME
echo 'Loading py3-v4.1.0'
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh`
export PYTHONUSERBASE=/mnt/lfs7/user/mhuennefeld/DNN_reco/virtualenvs/tensorflow_gpu_py3-v4.1.0
echo 'Using PYTHONUSERBASE: '${PYTHONUSERBASE}

export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.7/site-packages:$PYTHONPATH

# export CUDA
export CUDA_HOME=/usr/local/cuda-10.0;
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64


echo $FINAL_OUT
if [ -z ${PBS_JOBID} ] && [ -z ${_CONDOR_SCRATCH_DIR} ]
then
    echo 'Running Script w/o temporary scratch'
    /home/vbasu/scripts/simulation_scripts/steps/step_0_inject_veto_muons.py /home/vbasu/scripts/simulation_scripts/dagfiles/22010/processing/step_0_inject_veto_muons/veto_muons__NuGen_21217.yaml 1 --no-scratch
    ICETRAY_RC=$?
    echo 'IceTray finished with Exit Code: ' $ICETRAY_RC
    if [ $ICETRAY_RC -ne 0 ] && [ $KEEP_CRASHED_FILES -eq 0 ] ; then
        echo 'Deleting partially processed file! ' $FINAL_OUT
        rm $FINAL_OUT
    fi
else
    echo 'Running Script w/ temporary scratch'
    if [ -z ${_CONDOR_SCRATCH_DIR} ]
    then
        cd /scratch/${USER}
    else
        cd ${_CONDOR_SCRATCH_DIR}
    fi
    /home/vbasu/scripts/simulation_scripts/steps/step_0_inject_veto_muons.py /home/vbasu/scripts/simulation_scripts/dagfiles/22010/processing/step_0_inject_veto_muons/veto_muons__NuGen_21217.yaml 1 --scratch
    ICETRAY_RC=$?
    echo 'IceTray finished with Exit Code: ' $ICETRAY_RC
    if [ $ICETRAY_RC -eq 0 ] || [ $KEEP_CRASHED_FILES -eq 1 ]; then
        cp *.i3.bz2 /data/user/vbasu/simulation_scripts/00000-00999
    fi
    rm *.i3.bz2
fi
exit $ICETRAY_RC

