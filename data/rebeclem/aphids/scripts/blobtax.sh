#!/bin/bash
#SBATCH -J blob_array
#SBATCH -o blob_%A_%a.out
#SBATCH -e blob_%A_%a.err
#SBATCH -N 1 -n 20
#SBATCH -t 2-00:00:00
#SBATCH -p atlas
#SBATCH -A aphid_phylogenomics
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rebeclem@gmail.com
#SBATCH --array=1-36

name1=$(sed -n "$SLURM_ARRAY_TASK_ID"p namelist.txt)

#--- Starrt timer
t1=$(date +"%s")

echo "Starting $name1"
module load miniconda
source activate discovarenv
# I used conda install 'blast>2.12'
#module load singularity/ce-3.11.0
#module load blast
blastparams="6 qseqid staxids bitscore std sscinames sskingdoms stitle"
#singularity exec /apps/singularity-3/blast/blast-2.9.0.sif
blastn -db /reference/data/NCBI/blast/2023-08-31/nt \
        -task megablast \
        -query ${name1}_primary.genome.scf.fasta \
        -outfmt "6 qseqid staxids bitscore std" \
        -culling_limit 5 \
        -evalue 1e-25 \
        -num_threads 16 \
        -out ${name1}_masurca.megablast_nt
#culling limit sets the number of hits returned per subject sequence


#---Complete job
t2=$(date +"%s")
diff=$(($t2-$t1))
echo "[---$SN---] ($(date)) $(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
echo "[---$SN---] ($(date)) $SN COMPLETE."
