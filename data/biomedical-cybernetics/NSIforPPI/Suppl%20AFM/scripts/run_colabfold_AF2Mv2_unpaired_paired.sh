#!/bin/bash
#SBATCH --job-name ColabFold
##SBATCH --account=def-someuser
#SBATCH --time 24:00:00 ### (HH:MM:SS) the job will expire after this time, the maximum is 168:00:00
#SBATCH --gres=gpu:1 ### You need to request one GPU to be able to run AlphaFold properly
#SBATCH -N 1 ### number of nodes (1 node -> several CPUs)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 ### DO NOT INCREASE THIS AS ALPHAFOLD CANNOT TAKE ADVANTAGE OF MORE
#SBATCH --mem-per-cpu=10000
##SBATCH --threads-per-core=2
##SBATCH --hint=multithread
##SBATCH --mem=100G
#SBATCH -A p_networkgeometry
##SBATCH -e %j.err ### redirects stderr to this file
##SBATCH -o %j.out ### redirects standard output stdout to this file
#SBATCH -p alpha ### types of nodes on taurus: west, dandy, smp, gpu
##SBATCH --reservation=p_linkpredic_559
#SBATCH --mail-user ilyes.abdelhamid1@gmail.com ### email address to get updates about the jobs status
#SBATCH --mail-type ALL ### specify for what type of events you want to get a mail; valid options beside ALL are: BEGIN, END, FAIL, REQUEUE

#Run the command
module load modenv/hiera GCC/10.2.0 CUDA/11.3.1 OpenMPI/4.0.5

# ALPHAFOLD_DATA_PATH=/lustre/scratch2/ws/0/iabdelha-IA-AF-workspace/alphafold/data/databases
# ALPHAFOLD_MODELS=/lustre/scratch2/ws/0/iabdelha-IA-AF-workspace/alphafold/data/databases/params
# ALPHAFOLD_OUTPUT=/lustre/scratch2/ws/0/iabdelha-IA-AF-workspace/alphafold/alphafold_output
# ALPHAFOLD_DIR=/lustre/scratch2/ws/0/iabdelha-IA-AF-workspace/alphafold/data/alphafold_singularity

name='run_colabfold_AF2Mv2_unpaired_paired.sh'
maxsim=120
sleeptime=120
if [ "$#" -eq 0 ]
then
  sim=1
else
  sleep $sleeptime
  sim=$(($1+1))
fi
if [ $sim -lt $maxsim ]; then
  echo sbatch $name $sim
  sbatch $name $sim
fi

#MCR=/sw/global/applications/matlab/2017a/
#./run_run_colabfold_Yeast_AF2Mv1_unpaired_paired.sh $MCR

array_positive=($(ls /lustre/ssd/ws/iabdelha-IA-AF-SSD-workspace/alphafold/data/alphafold_singularity/FASTA_files/Yeast/Positive_set))
array_negative=($(ls /lustre/ssd/ws/iabdelha-IA-AF-SSD-workspace/alphafold/data/alphafold_singularity/FASTA_files/Yeast/Negative_set))
sorted_positive=($(printf '%s\n' "${array_positive[@]}" | sort -n))
sorted_negative=($(printf '%s\n' "${array_negative[@]}" | sort -n))
TEMPPATH=/lustre/ssd/ws/iabdelha-IA-AF-SSD-workspace/alphafold/data/alphafold_singularity/TEMP_AlphaFold2-multimer-v2_unpaired+paired
for (( i=0; i<${#sorted_negative[*]}; ++i));
do
  if [[ ! -f "$TEMPPATH/${sorted_negative[$i]}.TEMP" ]];
  then
    if [[ ${sorted_negative[$i]} ]];
    then
      PAIR_TO_TEST=${sorted_negative[$i]}
      SEQUENCE_FILE=/lustre/ssd/ws/iabdelha-IA-AF-SSD-workspace/alphafold/data/alphafold_singularity/FASTA_files/Yeast/Negative_set/
      COLABFOLD_OUTPUT=/lustre/ssd/ws/iabdelha-IA-AF-SSD-workspace/alphafold/alphafold_output/Yeast/Negative_set/AlphaFold2-multimer-v2/unpaired+paired/
      touch $TEMPPATH/${sorted_negative[$i]}.TEMP
      break
    fi
  elif [[ ! -f "$TEMPPATH/${sorted_positive[$i]}.TEMP" ]];
  then
    if [[ ${sorted_positive[$i]} ]];
    then
      PAIR_TO_TEST=${sorted_positive[$i]}
      SEQUENCE_FILE=/lustre/ssd/ws/iabdelha-IA-AF-SSD-workspace/alphafold/data/alphafold_singularity/FASTA_files/Yeast/Positive_set/
      COLABFOLD_OUTPUT=/lustre/ssd/ws/iabdelha-IA-AF-SSD-workspace/alphafold/alphafold_output/Yeast/Positive_set/AlphaFold2-multimer-v2/unpaired+paired/
      touch $TEMPPATH/${sorted_positive[$i]}.TEMP
      break
    else
      continue
    fi
  else
    continue
  fi
done

cd /lustre/ssd/ws/iabdelha-IA-AF-SSD-workspace/alphafold/data/colabfold_batch/bin
#cd /lustre/ssd/ws/iabdelha-IA-AF-SSD-workspace/alphafold/data
#bash install_colabbatch_linux.sh
export PATH="/lustre/ssd/ws/iabdelha-IA-AF-SSD-workspace/alphafold/data/colabfold_batch/bin:$PATH"
echo $SEQUENCE_FILE$PAIR_TO_TEST
colabfold_batch --num-recycle 3 --templates --pair-mode unpaired+paired --model-type AlphaFold2-multimer-v2 --rank intscore $SEQUENCE_FILE$PAIR_TO_TEST $COLABFOLD_OUTPUT$PAIR_TO_TEST
cd $COLABFOLD_OUTPUT$PAIR_TO_TEST
shopt -s extglob
rm -r !(*.pdb|*.png|config.json|timings.json|stats_*.json)
