#!/bin/bash
#BSUB -J tpxo_dayextr[1-180]%30                 # Name of the job array, example tpxo_dayextr[1-365]%30
#BSUB -o /work/oda/ag15419/job_scratch/tpxo_%J.out  # Appends std output to file %J.out.
#BSUB -e /work/oda/ag15419/job_scratch/tpxo_%J.err  # Appends std error to file %J.err.
#BSUB -cwd "/work/oda/ag15419/job_scratch/%J/"
#BSUB -q s_long
#BSUB -n 1                                      # Number of CPUs
#BSUB -R "rusage[mem=1G]"
#BSUB -P 0284 
#
# ACG Sep 2020
# Script for tpxo9 z tide extraction
# Ini file: tpxo_dayextr.ini 
#
#set -u
set -e
#set -x 
################### ENV SETTINGS ##############################
SRC_DIR="/users_home/oda/ag15419/tpxo2eas/"
echo "Job-Array element: ${LSB_JOBINDEX}"
################### PREPROC ###################################

# Source ini file
  INIFILE="${SRC_DIR}/tpxo_dayextr.ini"
  source $INIFILE
  echo "source $INIFILE ... Done!"

  module load ${EXTR_MODULE}  
  #module list

# Read and check infos (work dir, file names, archive dir, etc.)

  # Workdir check and subdir mk
  if [[ -d $EXTR_WORKDIR ]]; then
     cd $EXTR_WORKDIR
     #mkdir day_${LSB_JOBINDEX}
     #cd day_${LSB_JOBINDEX}
     #EXTR_SUBWORKDIR=${EXTR_WORKDIR}/day_${LSB_JOBINDEX}
     echo "WORKDIR: $(pwd)"
     
     # Clean workdir
     #echo "WARNING: I am going to remove all files in $EXTR_WORKDIR ..."
     #sleep 10
     #for TO_BE_RM in $( ls $EXTR_WORKDIR ); do
     ##    rm -vr $EXTR_WORKDIR/$TO_BE_RM
     #    echo $TO_BE_RM
     #done
     # Cp the exe file to the workdir
     if [[ ! -f ${EXE_OTPS} ]]; then 
        echo "I need to copy the exe file to the workdir.."
        cp ${SRC_DIR}/${EXE_OTPS} .
     fi

  else
     echo "ERROR: WORKDIR $EXTR_WORKDIR NOT FOUND!!"
     exit
  fi

  # Cp the Model_atlas file to the workdir
  if [[ ! -d DATA ]] && [[ ! -f DATA/Model_atlas ]]; then
        echo "I need to copy the DATA dir to the workdir.."
        cp -rv ${SRC_DIR}/DATA .
  else
        echo "ERROR: DATA/Model_atlas NOT FOUND in SRC_DIR.."
  fi



  # Set the date based on the index of the array
  DAY_IDX=$(( ${LSB_JOBINDEX} - 1 ))
  echo "The extraction starts on date: ${EXTR_STARTDATE}"
  DAY2EXTR=$(date -d "${EXTR_STARTDATE} ${DAY_IDX} day" +%Y%m%d)  
  echo "This job-array element extracts date: ${DAY2EXTR}"

  # Check mesh_mask file
  MESHMASK="${MESHMASK_PATH}/${MESHMASK_FILE}"
  if [[ -f ${MESHMASK} ]]; then
     echo "Lat and lon are taken from file: ${MESHMASK}"
  else
     echo "ERROR: mesh mask file NOT found..Why?"
     exit
  fi

  # Check out dir and built outfile name
  if [[ -d ${OUTNC_PATH} ]]; then
     OUTNC_NAME=$( echo "$OUTNC_TEMPLATE" | sed -e "s/%YYYYMMDD%/${DAY2EXTR}/g" )
     OUTFILE_NC="${OUTNC_PATH}/${OUTNC_NAME}"
     echo "The output.nc file is: ${OUTFILE_NC}"
  else
     echo "ERROR: OUT dir NOT found..Why?"
     exit
  fi

  # Run daily extraction..
  if [[ -f ${EXE_OTPS} ]]; then 
     echo "Run the script with the following args: "
     echo "./${EXE_OTPS} ${DAY2EXTR} ${MESHMASK} ${OUTFILE_NC}"
     time ./${EXE_OTPS} ${DAY2EXTR} ${MESHMASK} ${OUTFILE_NC}
  else
     echo "ERROR: Not found exe file ${EXE_OTPS}...Why?!"
     exit
  fi


###################### POSTPROC ###########################

# Output check

# Archive


