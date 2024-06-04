#!/bin/bash
#SBATCH --job-name=singIm
#SBATCH --mail-user=lptolik@gmail.com
#SBATCH --partition=short
#SBATCH --c 1
#SBATCH --mem=8G
#SBATCH --output=getImage_%j.out
#SBATCH --error=getImage_%j.err

datdir=$1
echo $datdir

wd=$PWD
# create a temporary directory for this job and save the name 
tempdir=$(mktemp -d /flash/GoryaninU/lptolik/makeTSV.XXXXXX)
echo $tempdir

# enter the temporary directory
cd $tempdir

module load singularity

singularity pull  --name alesssia-yampdocker.img docker://alesssia/yampdocker

ssh deigo "mkdir -p $datdir/work/singularity/"

# copy our result back to bucket. We need to use "scp"
# to copy the data back as bucket isn't writable directly. 
scp "alesssia-yampdocker.img" deigo:"$datdir/work/singularity/"

cd $wd
# Clean up by removing our temporary directory 
rm -r $tempdir

mkdir -p ./work/singularity/
cp alesssia-yampdocker.img ./work/singularity/alesssia-yampdocker.img
