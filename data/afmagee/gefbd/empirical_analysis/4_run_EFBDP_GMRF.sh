#!/bin/bash

N_CORES=4
JOB_DIR="empirical_analysis/jobs_GMRF"
LOG_DIR="empirical_analysis/logs_GMRF"
exec=rb-mpi-coal

if [ ${JOB_DIR} != "" ]; then
  if [ ! -d ${JOB_DIR} ]; then
    mkdir ${JOB_DIR}
  else
    rm -f ${JOB_DIR}/*
  fi
fi

if [ ${LOG_DIR} != "" ]; then
  if [ ! -d ${LOG_DIR} ]; then
    mkdir ${LOG_DIR}
  else
    rm -f ${LOG_DIR}/*
  fi
fi


#for me_prior in "0.0" "0.5" "2.0";
for me_prior in "0.5";
do

#    for uncertainty in "none" "both";
    for uncertainty in "both";
    do

#        for ds in "Wilberg" "Stubbs";
        for ds in "Stubbs";
        do

            echo "#!/bin/bash
#SBATCH --job-name=${ds}_${uncertainty}_${me_prior}
#SBATCH --output=${ds}_${uncertainty}_${me_prior}.log
#SBATCH --error=${ds}_${uncertainty}_${me_prior}.err
#SBATCH --ntasks=${N_CORES}
#SBATCH --nodes=1
#SBATCH --mem=${N_CORES}G
#SBATCH --qos=high_prio
#SBATCH --time=7-00:00:00

module load gnu
module load boost
module load openmpi

# <path/to/rb> analysis_age_uncertainty.Rev --args <treefile> <TAXON_FILE> <BDP_prior> <hyperprior_file> <ME_hyperprior> <treatement_probability> <age_uncertainty> <NUM_REPS> <seed> <OUTPUT_DIR>
mpirun -np ${N_CORES} ${exec} src/analysis_age_uncertainty.Rev --args ${ds}.tre crocs_taxa_range_${ds}.tsv GMRFBDP ${ds}.priors.txt ${me_prior} 0.5 ${uncertainty} ${N_CORES} 1234 empirical_analysis/output_${ds}_${uncertainty} > ${LOG_DIR}/${ds}_${uncertainty}_${me_prior}.out
" > ${JOB_DIR}/${ds}_${uncertainty}_${me_prior}.sh
            sbatch ${JOB_DIR}/${ds}_${uncertainty}_${me_prior}.sh

        done

    done

done



echo "done ..."
