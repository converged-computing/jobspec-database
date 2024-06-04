#!/bin/bash -l
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --mail-user=araje002@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH -o ./history/IPRScan-%A.out
set -euv

# $((SLURM_MEM_PER_NODE/1000))'G'
# $SLURM_NTASKS
# $SLURM_ARRAY_TASK_ID

#this script will assign interpro results to the proteins predicted by Funannotate Predict. This is meant to be run as an array job with array ID corresponding to the suffix of the protein fasta file


module load interproscan

if [ ! -d "$IPROUT" ]; then
  mkdir ~/bigdata/Csativa/results/2_IPRScan
fi

interproscan.sh \
    -i ~/bigdata/Csativa/results/1_FunannotatePredict/predict_results/purple_kush.proteins.fa.$SLURM_ARRAY_TASK_ID \
    -appl CDD,COILS,Gene3D,HAMAP,MobiDBLite,Pfam,PIRSF,PRINTS,ProDom,PROSITEPATTERNS,PROSITEPROFILES,SFLD,SMART,SUPERFAMILY,TIGRFAM \
    -d ~/bigdata/Csativa/results/2_IPRScan \
    --goterms
