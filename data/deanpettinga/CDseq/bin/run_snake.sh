#PBS -l walltime=100:00:00
#PBS -l mem=8gb
#PBS -m ae
#PBS -N CDseq
#PBS -o logs/workflow.o
#PBS -e logs/workflow.e

cd ${PBS_O_WORKDIR}
module load bbc/snakemake/snakemake-5.15.0

# save DAG job file with time stamp
TIME=$(date "+%Y-%m-%d_%H.%M.%S")
if [[ -f "logs/runs" ]]
then
	echo "logs/runs DOES exist."
else
	mkdir logs/runs/
	echo "creating directory logs/runs"
fi
snakemake --use-envmodules --cores 1 -n > logs/runs/workflow_${TIME}.txt
snakemake --dag --cores 1 | dot -Tpng > logs/runs/workflow_${TIME}.png

snakemake \
--use-envmodules \
--jobs 100 \
--cluster "qsub \
-q bbc \
-V \
-l nodes={resources.nodes}:ppn={resources.threads} \
-l mem={resources.mem_gb}gb \
-l walltime=100:00:00 \
-o logs/runs/ \
-e logs/runs/"
