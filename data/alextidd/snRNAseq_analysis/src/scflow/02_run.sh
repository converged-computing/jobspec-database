#PBS -N run_scflow
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mem=20gb
#PBS -o ~/snRNAseq_analysis/log/scflow/run.out
#PBS -e ~/snRNAseq_analysis/log/scflow/run.err

# tmux new -s scflow

cd ~/snRNAseq_analysis/

. ~/.bashrc
conda activate nfcore
module load gcc/8.2.0
NXF_OPTS='-Xms1g -Xmx4g'
NXF_SINGULARITY_CACHEDIR='~/snRNAseq_analysis/singularity/'

nextflow run nf-core/scflow \
  -c config/scflow/oesophageal_10X.config \
  -r dev \
  -profile imperial \
  -resume \
  -w work/scflow/
