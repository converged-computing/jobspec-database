#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16 --mem 16gb
#SBATCH --output=logs/annotfunc.%a.log
#SBATCH --time=2-0:00:00
#SBATCH -p intel -J annotfunc
module unload miniconda2
module load funannotate/1.8.1
module load phobius
CPUS=$SLURM_CPUS_ON_NODE
OUTDIR=annotate
SAMPFILE=strains.csv
BUSCO=eurotiomycetes_odb10
SEED_SPECIES=coccidioides_immitis
TEMPLATE=

if [ ! $CPUS ]; then
 CPUS=1
fi
N=${SLURM_ARRAY_TASK_ID}

if [ ! $N ]; then
    N=$1
    if [ ! $N ]; then
        echo "need to provide a number by --array or cmdline"
        exit
    fi
fi

MAX=$(wc -l $SAMPFILE | awk '{print $1}')
if [ $N -gt $MAX ]; then
    echo "$N is too big, only $MAX lines in $SAMPFILE"
    exit
fi
IFS=,
cat $SAMPFILE | sed -n ${N}p | while read BASE SPECIES STRAIN RNASEQSET LOCUS
do
	TEMPLATE=$(realpath lib/authors.sbt)

	if [ ! -d $OUTDIR/$BASE  ]; then
		echo "No annotation dir for ${BASE} ($OUTDIR/$BASE)"
		exit
 	fi
	# here we woudl conditionally make SBT file location from BASE info?

	MOREFEATURE=""
	#if [[ ! -z $TEMPLATE ]]; then
	#	 MOREFEATURE="--sbt $TEMPLATE"
	#fi
	# need to add detect for antismash and then add that
	echo "$OUTDIR/$BASE"
	ANTISMASHRESULT=$OUTDIR/$name/annotate_misc/antiSMASH.results.gbk
	if [[ ! -f $ANTISMASHRESULT && -d $OUTDIR/$BASE/antismash_local ]]; then
		ANTISMASH=$(ls $OUTDIR/$BASE/antismash_local/*.final.gbk | awk '{print $1}')
		rsync -a $ANTISMASH $ANTISMASHRESULT
	fi
	funannotate annotate --sbt $TEMPLATE --busco_db $BUSCO -i $OUTDIR/$BASE --species "$SPECIES" --strain "$STRAIN" --cpus $CPUS $MOREFEATURE $EXTRAANNOT
done
