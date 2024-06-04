#PBS -l walltime=24:00:00
#PBS -l mem=8gb
#PBS -N snake_workflow
#PBS -o logs/snake_workflow.o
#PBS -e logs/snake_workflow.e

cd ${PBS_O_WORKDIR}

snakemake_module="bbc/snakemake/snakemake-6.1.0"

module load $snakemake_module

snakemake \
-p \
--latency-wait 20 \
--snakefile 'Snakefile' \
--use-envmodules \
--jobs 50 \
--cluster "ssh ${PBS_O_LOGNAME}@submit 'module load $snakemake_module; cd ${PBS_O_WORKDIR}; qsub \
-q ${PBS_O_QUEUE} \
-V \
-l nodes=1:ppn={threads} \
-l mem={resources.mem_gb}gb \
-l walltime=48:00:00 \
-o {log.stdout} \
-e {log.stderr}'"

