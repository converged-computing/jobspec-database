#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16 --mem 16gb
#SBATCH --output=logs/annotfunc.%a.log
#SBATCH --time=2-0:00:00
#SBATCH -p intel -J annotfunc

module load funannotate/1.7.3_sing
MEM=24G
export AUGUSTUS_CONFIG_PATH=$(realpath lib/augustus/3.3/config)

# Set some vars
export SINGULARITY_BINDPATH=/bigdata
export FUNANNOTATE_DB=/bigdata/stajichlab/shared/lib/funannotate_db
export SINGULARITYENV_PASACONF=/rhome/jstajich/pasa.CONFIG.template

CPUS=$SLURM_CPUS_ON_NODE
OUTDIR=annotate
INDIR=genomes
SAMPFILE=samples.csv
BUSCO=eurotiomycetes_odb9
species="Aspergillus fumigatus"
if [ -z $CPUS ]; then
 CPUS=1
fi

N=${SLURM_ARRAY_TASK_ID}

if [ -z $N ]; then
    N=$1
    if [ -z $N ]; then
        echo "need to provide a number by --array or cmdline"
        exit
    fi
fi
MAX=`wc -l $SAMPFILE | awk '{print $1}'`

if [ $N -gt $MAX ]; then
    echo "$N is too big, only $MAX lines in $SAMPFILE"
    exit
fi
IFS=,
cat $SAMPFILE | sed -n ${N}p | while read PREFIX LOCUS
do
	name=$PREFIX
	Strain=$PREFIX
	MOREFEATURE=""
	TEMPLATE=$(realpath lib/pangenome.sbt)
	if [ ! -f $TEMPLATE ]; then
		echo "NO TEMPLATE for $name"
		exit
	fi
	# need to add detect for antismash and then add that
	funannotate annotate --sbt $TEMPLATE --busco_db $BUSCO -i $OUTDIR/$name --species "$species" --strain "$Strain" --cpus $CPUS $MOREFEATURE $EXTRAANNOT
done
