#!/bin/bash
#SBATCH --ntasks 32 --nodes 1 --mem 96G -p intel 
#SBATCH --time 48:00:00 --out logs/iprscan.%a.%A.log

module unload miniconda2
module load miniconda3
module load funannotate/git-live
module load iprscan
CPU=1
if [ ! -z $SLURM_CPUS_ON_NODE ]; then
    CPU=$SLURM_CPUS_ON_NODE
fi

SAMPFILE=samples.csv
N=${SLURM_ARRAY_TASK_ID}

if [ ! $N ]; then
    N=$1
    if [ ! $N ]; then
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
sed -n ${N}p $SAMPFILE | while read SampleId Species Clade
do
	 if [ ! -d annotation_${SampleId} ]; then
		echo "No annotation dir for ${SampleId} did you run 00_setup.sh $N?"
		exit
 	fi
	pushd annotation_${SampleId}
	if [[ -f "config.txt" ]]; then
        	source config.txt
	else
        	echo "Need a config file"
        	exit
	fi

	mkdir -p $ODIR/annotate_misc
	XML=$ODIR/annotate_misc/iprscan.xml
	IPRPATH=$(which interproscan.sh)
	if [ ! -f $XML ]; then
	    funannotate iprscan -i funannot -o $XML -m local -c $CPU --iprscan_path $IPRPATH
	fi
	popd
done
