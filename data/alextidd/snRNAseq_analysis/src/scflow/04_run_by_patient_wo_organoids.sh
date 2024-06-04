#PBS -N run_by_patient_wo_organoids_scflow

#PBS -l select=1:ncpus=1:mem=20gb
#PBS -l walltime=48:00:00
#PBS -o ~/snRNAseq_analysis/log/scflow/run_by_patient_wo_organoids.out

cd ~/snRNAseq_analysis/

. ~/.bashrc
conda activate nfcore
module load gcc/8.2.0
NXF_OPTS='-Xms1g -Xmx4g'
  
# by patient wo organoids
for patient in {A..P} ; do
  nextflow run nf-core/scflow \
    -c config/scflow/oesophageal_10X.config \
    --manifest <( cat data/scflow/manifest.tsv | 
                  awk -v patient=$patient -F'.' \
                  '{ if ((NR==1) || ($1 == patient)) { print } }' |
                  grep -v organoid ) \
    --outdir output/scflow/by_patient_wo_organoids/$patient/ \
    -r dev \
    -profile imperial \
    -resume 
done