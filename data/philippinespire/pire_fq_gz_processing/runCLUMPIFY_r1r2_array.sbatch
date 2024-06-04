#!/bin/bash -l

#SBATCH --job-name=clmp_r12
#SBATCH -o clmp_r1r2_-%A_%a.out
#SBATCH -p main
#SBATCH -c 20
#SBATCH --exclusive=user

enable_lmod
module load container_env pire_genome_assembly
#module load java/1.8
#module load bbmap/38.90
#module load parallel

export SINGULARITY_BIND=/home/e1garcia

#TEMPDIR=/scratch-lustre/cbird
#INDIR=/home/e1garcia/shotgun_PIRE/fq_4denovo_fp1
#OUTDIR=/home/e1garcia/shotgun_PIRE/fq_4denovo_fp1_clmp
FQPATTERN=${4}
TEMPDIR=${3}
INDIR=${1}
OUTDIR=${2}
THREADS=1   #clumpify uses a ton of ram, be conservative
GROUPS=auto   #controls how much ram is used, refer to manual
#GROUPS=1
RAMPERTHREAD=180g   #have had to set as high as 233g with groups=1
#RAMPERTHREAD=233g

#ulimit -u 40960
ulimit -n 40960 
#ulimit -s unlimited

ulimit -a

mkdir -p $OUTDIR

all_samples=$(ls $INDIR/$FQPATTERN | \
	sed -e 's/r1\.fq\.gz//' -e 's/.*\///g')
all_samples=($all_samples)

sample_name=${all_samples[${SLURM_ARRAY_TASK_ID}]}
echo ${sample_name}

#systemctl status  $PPID  
#systemctl status  $PPID  | head -1 | awk '{print $2}' | xargs systemctl show -p TasksMax
#cat /proc/sys/kernel/threads-max

crun clumpify.sh \
	in=$INDIR/${sample_name}r1.fq.gz \
	in2=$INDIR/${sample_name}r2.fq.gz \
	out=$OUTDIR/${sample_name}clmp.r1.fq.gz \
	out2=$OUTDIR/${sample_name}clmp.r2.fq.gz \
	groups=$GROUPS \
	overwrite=t \
	usetmpdir=t \
	tmpdir=$TEMPDIR \
	deletetemp=t \
	dedupe=t \
	addcount=t \
	subs=2 \
	containment=t \
	consensus=f \
	-Xmx$RAMPERTHREAD
