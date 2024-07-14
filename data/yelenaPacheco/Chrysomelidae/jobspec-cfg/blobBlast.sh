#!/bin/bash
#SBATCH -J blob_blast
#SBATCH -o blob_%A_%a.out
#SBATCH -e blob_%A_%a.err
#SBATCH -N 1 -n 20
#SBATCH -t 2-00:00:00
#SBATCH -p short,scavenger,scavenger-gpu,scavenger-mem768,medium
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yelena.pacheco@usda.gov


#this uses megablast which matched longer sequences than blastn


#--- Starrt timer
t1=$(date +"%s")


module load blast+/2.13.0

blastparams="6 qseqid staxids bitscore std sscinames sskingdoms"


blastn -task megablast -query N_clydesmithi_hifiasm_contigs.fa -db /reference/data/NCBI/blast/2023-08-31/nt -outfmt "6 qseqid staxids bitscore std sscinames sskingdoms stitle" -culling_limit 5 -num_threads 16 -evalue 1e-25 -out N_clydesmithi.megablast_nt
#culling limit sets the number of hits returned per subject sequence

#---Complete job
t2=$(date +"%s")
diff=$(($t2-$t1))
echo "[---$SN---] ($(date)) $(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
echo "[---$SN---] ($(date)) $SN COMPLETE."




###This script was modified from R.Clement github, blobblast.sh