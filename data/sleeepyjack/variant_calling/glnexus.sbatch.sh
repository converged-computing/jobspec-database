#!/bin/sh

# SLURM parameters (adjust according to your setup)
#SBATCH -J glnexus
#SBATCH -A pararch
#SBATCH -p nodelong
#SBATCH -t 100:00:00
#SBATCH --mem=470G
#SBATCH --exclusive
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=juenger@uni-mainz.de

# exit if anything goes wrong
set -e

# input/output (adjust according to your setup)
GVCFDIR=$1
OUTDIR=glnexus_output
JOBDIR=/localscratch/${SLURM_JOB_ID}

# add some information to the log file
echo "##################"
echo "#### GLNexus on MOGON"
echo "#### JOB ID:" ${SLURM_JOB_ID}
echo "#### NODE:" $(hostname)
echo "#### CORES:" $(nproc)
echo "#### INPUT DIRECTORY:" ${GVCFDIR}
echo "#### OUTPUT DIRECTORY:" ${OUTDIR}
echo "##################"


# load environment modules
module load tools/Singularity/3.5.2-Go-1.13.1
module load tools/parallel/20190822

# create a job-local tmp directory for the container (never use a shared /tmp)
mkdir -p ${JOBDIR}/tmp

# copy input files to local scratch
LOCAL_GVCFDIR=${JOBDIR}/gvcf
mkdir -p ${LOCAL_GVCFDIR}
find ${GVCFDIR}/ -type f -name "*.gvcf" -print0 | parallel -0 -j$(nproc) cp {} ${LOCAL_GVCFDIR}/
find ${LOCAL_GVCFDIR}/ -type f -name "*.gvcf" >> $JOBDIR/gvcf_list

echo "cd /tmp && LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so numactl --interleave=all glnexus_cli --config DeepVariantWGS --list $JOBDIR/gvcf_list | bcftools view - | bgzip -@ $(nproc) -c > $JOBDIR/output.cohort.vcf.gz" >> $JOBDIR/cmd.sh
cat $JOBDIR/cmd.sh
chmod +x $JOBDIR/cmd.sh

# execute GLnexus inside the singularity container
singularity exec \
    -B ${JOBDIR} \
    -B ${JOBDIR}/tmp:/tmp \
    environment.cpu.sif \
    $JOBDIR/cmd.sh



# copy results to the output directory
mkdir -p ${OUTDIR}
cp ${JOBDIR}/output.cohort.vcf.gz ${OUTDIR}/
