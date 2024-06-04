#!/bin/bash -e
#SBATCH --account=t3
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --time=23:00:00
#SBATCH --nodes=1

echo "------------------------------------------------------------"
echo "[`date`] Job started"
echo "------------------------------------------------------------"
DATE_START=`date +%s`

echo HOSTNAME: ${HOSTNAME}
echo HOME: ${HOME}
echo USER: ${USER}
echo X509_USER_PROXY: ${X509_USER_PROXY}
echo CMD-LINE ARGS: $@

if [[ "$2" != "test" ]]; then
  SLURM_ARRAY_TASK_ID=$2
else
  SLURM_ARRAY_TASK_ID=1
fi

if [ -z ${SLURM_ARRAY_TASK_ID} ]; then
  printf "%s\n" "Environment variable \"SLURM_ARRAY_TASK_ID\" is not defined. Job will be stopped." 1>&2
  exit 1
fi

# define SLURM_JOB_NAME and SLURM_ARRAY_JOB_ID, if they are not defined already (e.g. if script is executed locally)
[ ! -z ${SLURM_JOB_NAME} ] || SLURM_JOB_NAME=${HOSTNAME}
[ ! -z ${SLURM_ARRAY_JOB_ID} ] || SLURM_ARRAY_JOB_ID=local$(date +%y%m%d%H%M%S)

#SLURM_JOB_NAME=$1
echo SLURM_JOB_NAME: ${SLURM_JOB_NAME}
echo SLURM_JOB_ID: ${SLURM_JOB_ID}
echo SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}
echo SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}

USERDIR=$4
if [[ ${USERDIR} == /pnfs/* ]]; then
    (
      (! command -v scram &> /dev/null) || eval `scram unsetenv -sh`
      gfal-mkdir -p root://t3dcachedb.psi.ch:1094/$USERDIR
      gfal-mkdir -p root://t3dcachedb.psi.ch:1094/$USERDIR/logs
      gfal-mkdir -p root://t3dcachedb.psi.ch:1094/$USERDIR/merged
      sleep 5
    )
else
    mkdir -p $USERDIR
    mkdir -p $USERDIR/logs
    mkdir -p $USERDIR/merged
fi
echo OUTPUT_DIR: $USERDIR

# local /scratch dir to be used by the job
TMPDIR=/scratch/${USER}/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
echo TMPDIR: ${TMPDIR}
mkdir -p ${TMPDIR}
NUM_LUMIBLOCK=${SLURM_ARRAY_TASK_ID}
cd ${TMPDIR}

#ARG parsing
customise=$5
echo $customise
customise_commands=`echo $6 | sed "s;*;';g"`
echo $customise_commands
era=$7
conditions=$8
sample_type=$9
step=${10}
redirector=${11}
source /cvmfs/cms.cern.ch/cmsset_default.sh

echo
echo "--------------------------------------------------------------------------------"
echo "--------------------------------------------------------------------------------"
echo "                          Creating JOB ["$2"]"
echo

#export SCRAM_ARCH=el8_amd64_gcc10
cd ${TMPDIR}

CMSSW_VER="13_3_1_patch1"
scramv1 project CMSSW CMSSW_$CMSSW_VER
cd CMSSW_$CMSSW_VER
eval `scram runtime -sh`
cd src

git cms-addpkg PhysicsTools/NanoAOD
git cms-addpkg FWCore
git cms-merge-topic TizianoBevilacqua:devel-naod-v13

output="nAOD_"$2".root"

echo
echo "--------------------------------------------------------------------------------"
echo "                                JOB ["$2"] ready"
echo "                                  Compiling..."
echo

#scramv1 b
scram b -j 4
if [ ! -f ${step}_cfg.py ]; then
    if [ "${customise_commands}" == "skip" ]; then
        if [ "${customise}" == "skip" ]; then
            echo cmsDriver.py NANO -s NANO --python_filename ${step}_cfg.py --eventcontent $step --datatier $step --era $era --conditions $conditions  $sample_type --no_exec
            cmsDriver.py NANO -s NANO --python_filename ${step}_cfg.py --eventcontent $step --datatier $step --era $era --conditions $conditions  $sample_type --no_exec -n -1
        else
            echo cmsDriver.py NANO -s NANO --python_filename ${step}_cfg.py --eventcontent $step --datatier $step --era $era --conditions $conditions --customise \"$customise\"  $sample_type --no_exec
            cmsDriver.py NANO -s NANO --python_filename ${step}_cfg.py --eventcontent $step --datatier $step --era $era --conditions $conditions --customise "$customise"  $sample_type --no_exec -n -1
        fi
    elif [ "${customise}" == "skip" ]; then
        echo cmsDriver.py NANO -s NANO --python_filename ${step}_cfg.py --eventcontent $step --datatier $step --era $era --conditions $conditions --customise_commands \"$customise_commands\" $sample_type --no_exec
        cmsDriver.py NANO -s NANO --python_filename ${step}_cfg.py --eventcontent $step --datatier $step --era $era --conditions $conditions --customise_commands "$customise_commands" $sample_type --no_exec -n -1
    else
        echo cmsDriver.py NANO -s NANO --python_filename ${step}_cfg.py --eventcontent $step --datatier $step --era $era --conditions $conditions --customise \"$customise\" --customise_commands \"$customise_commands\" $sample_type --no_exec
        cmsDriver.py NANO -s NANO --python_filename ${step}_cfg.py --eventcontent $step --datatier $step --era $era --conditions $conditions --customise "$customise" --customise_commands "$customise_commands" $sample_type --no_exec -n -1
    fi

    cp $1 .      #copy the GoodLumi.json list to working directory
    cat $3 | sed "s;^;${redirector};" > tmp.sh
    cat tmp.sh
    chmod 755 tmp.sh  

    files=`cat tmp.sh | grep .root | awk '{print $NF}' | sed "s;^;file:;" | tr "\n" "," | sed "s:,:\", \":g" | sed 's/.\{3\}$//' | sed 's:^:\":' | sed 's:\":\":g'`
    sed -e "s;'file:NANO_PAT.root';${files};g" ${step}_cfg.py > tmp
    if [ ${sample_type} == "--data" ]; then
        if [[ "$USERDIR" =~ .*"2017".* ]]; then
            sed -e "s;\# Other statements;\# Other statements \\nimport FWCore.PythonUtilities.LumiList as LumiList\\nprocess.source.lumisToProcess = LumiList.LumiList(filename = 'Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.json').getVLuminosityBlockRange()\\n;g" tmp > ${step}_cfg.py
        elif [[ "$USERDIR" =~ .*"2018".* ]]; then
            echo "no good lumi selection, this is already implemented in HiggsDNA"
            mv tmp ${step}_cfg.py
        fi  
    else
        mv tmp ${step}_cfg.py
    fi

    chmod 755 ${step}_cfg.py
    cp ${step}_cfg.py /work/bevila_t
fi

echo
echo "--------------------------------------------------------------------------------"
echo "                                 Compiling ready"
echo "                               Starting JOB ["$2"]"
echo

echo cmsRun ${step}_cfg.py `cat ${step}_cfg.py | grep "input ="` `cat ${step}_cfg.py | grep "file:"`
cmsRun ${step}_cfg.py

echo
echo "--------------------------------------------------------------------------------"
echo "                                  JOB ["$2"] Finished"
echo "                              Writing output to pnfs(?)..."
echo

# Copy to pnfs
if [[ ${USERDIR} == /pnfs/* ]]; then
    xrdcp -f -N NANO_NANO.root root://t3dcachedb.psi.ch:1094//$USERDIR/$output
    echo "file copied..."
    xrdcp -f -N /work/${USER}/test/.slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${2}.out root://t3dcachedb.psi.ch:1094/$USERDIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${2}.out
    xrdcp -f -N /work/${USER}/test/.slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${2}.err root://t3dcachedb.psi.ch:1094/$USERDIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${2}.err
    echo "logs copied..."
    if [ ! -f ${USERDIR}/${step}_cfg.py ]; then
        xrdcp -f -N ${step}_cfg.py ${USERDIR}/${step}_cfg.py
    fi
else
    cp  NANOAOD_NANO.root $USERDIR/$output
    cp  /work/${USER}/test/.slurm/%x_%A_${2}.out $USERDIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${2}.out
    cp  /work/${USER}/test/.slurm/%x_%A_${2}.err $USERDIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${2}.err
fi

echo
echo "Output: "
ls -l $USERDIR/$output

cd ../../
rm -rf CMSSW CMSSW_$CMSSW_VER
cd .. 
rm -rf ${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo
echo "--------------------------------------------------------------------------------"
echo "                                 JOB ["$2"] DONE"
echo "--------------------------------------------------------------------------------"
echo "--------------------------------------------------------------------------------"
