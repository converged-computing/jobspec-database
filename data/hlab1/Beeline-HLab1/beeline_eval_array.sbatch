#!/bin/bash
#SBATCH --job-name=Beeline_L0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0:01:00
#SBATCH --output=slurm/slurm-%A_%a.out
#SBATCH --error=slurm/slurm-%A_%a.err

module purge

echo ${CONFIG_SPLIT_FILE}
echo ${CONFIG_SPLIT_SLURM_DIR}
echo ${METRIC}
mkdir -p ${CONFIG_SPLIT_SLURM_DIR}/${SLURM_ARRAY_JOB_ID}
cp ${CONFIG_SPLIT_FILE} ${CONFIG_SPLIT_SLURM_DIR}/${SLURM_ARRAY_JOB_ID}/ALL_SPLIT.txt
line_N=$( awk "NR==${SLURM_ARRAY_TASK_ID}" ${CONFIG_SPLIT_SLURM_DIR}/${SLURM_ARRAY_JOB_ID}/ALL_SPLIT.txt )  # NR means row-# in Awk
field_1=$( echo "$line_N" | cut -d "," -f 1 )  # grab comma-delim'd field #1
field_2=$( echo "$line_N" | cut -d "," -f 2 )  # grab comma-delim'd field #2
field_3=$( echo "$line_N" | cut -d "," -f 3 )  # grab comma-delim'd field #3

echo ${SLURM_ARRAY_TASK_ID} ${field_1} ${field_2} ${field_3}
singularity exec \
	    --overlay conda_greene/overlay-5GB-200K-beeline20211104.ext3:ro \
	    conda_greene/centos-8.2.2004.sif \
	    /bin/bash -c "source /ext3/env.sh; conda activate BEELINE; \
	    python BLEvaluator.py --config ${field_1} --dataset_names ${field_2} --algorithm_names ${field_3} --${METRIC}"

mv slurm/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out ${CONFIG_SPLIT_SLURM_DIR}/${SLURM_ARRAY_JOB_ID}
mv slurm/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err ${CONFIG_SPLIT_SLURM_DIR}/${SLURM_ARRAY_JOB_ID}

