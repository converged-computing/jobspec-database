#!/bin/sh
#SBATCH --account=nn9036k --job-name=NF
#SBATCH --time=24:00:00
##SBATCH --partition=bigmem
#SBATCH --ntasks=2 --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=animesh.sharma@ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=nfSLURMLOG


WORKDIR=$PWD
cd ${WORKDIR}
export PATH=$PATH:$PWD
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

export http_proxy=proxy.saga:3128
export https_proxy=proxy.saga:3128
module load Miniconda3/22.11.1-1
#conda config --prepend channels bioconda
#conda config --show channels
#mamba install python=3.11 nf-core nextflow
#nf-core create
#cd nf-core-spritz/
#git checkout dev
#git remote add origin https://github.com/animesh/spritz_nf.git
#git push --all origin
#ln -s /cluster/projects/nn9036k/scripts/TK9/trimmomatic.1697725340.results/TK9_*P.fq.gz .
#vim samples.csv
conda activate nf-core
#nextflow  self-update
nextflow  -v
#nextflow run hello
 
nextflow main.nf --max_memory '160.GB' --max_cpus 20 -profile singularity --genome GRCh38 --input samples.csv --outdir rnaHG38sa
#rm -rf rnaHG38sa
#mv trimResHG38 rnaHG38sa
#rsync -Parv  login.nird-lmd.sigma2.no:PD/Animesh/TK/TK9R/
#mv TK9R/* .
#cd $HOME/scripts
#git checkout 65c53035fdd2c01c99c721a8a85be467e1e8bd4e scratch.slurm slurmTM.sh
#dos2unix scratch.slurm slurmTM.sh
#bash slurmTM.sh $PWD/TK9R
#cd $HOME/cluster/nfsp/nf-core-spritz
#ln -s $HOME/scripts/TK9R/trimmomatic.1701258726.results/*P.fq.gz 
#rm *R1.fastq.gz
#rename 'R.1P.fq' 'R1.fastq.gz' *
#rm *R2.fastq.gz
#rename 'R.2P.fq' 'R2.fastq.gz' *
#rename '.fastq.gz.gz' '.fastq.gz' *
#ls -1 *R1*gz > S1
#ls -1 *R2*gz > S2
#printf 'auto\n%.0s' {1..6} > S3
#echo "sample,fastq_1,fastq_2,strandedness" > samples.csv
#paste -d ','  S? >> samples.csv
#cat samples.csv
#rm nfSLURMLOG
#sbatch scratch.slurm
