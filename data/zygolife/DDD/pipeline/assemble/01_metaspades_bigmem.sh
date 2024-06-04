#!/usr/bin/bash
#SBATCH -p intel,batch,highmem -N 1 -n 32 --mem 256gb --out logs/metaspades_bigmem.%a.log -J metaspades

module load spades/3.15.2

MEM=256
SAMPLES=samples_prefix.csv
INFOLDER=input
ASM=assembly
mkdir -p $ASM
CPU=$SLURM_CPUS_ON_NODE
if [ -z $CPU ]; then
  CPU=1
fi

N=${SLURM_ARRAY_TASK_ID}

if [ ! $N ]; then
    N=$1
    if [ ! $N ]; then
        echo "Need an array id or cmdline val for the job"
        exit
    fi
fi


#--meta
IFS=,
tail -n +2 $SAMPLES | sed -n ${N}p | while read SPECIES STRAIN JGILIBRARY BIOSAMPLE BIOPROJECT TAXONOMY_ID ORGANISM_NAME SRA_SAMPID SRA_RUNID LOCUSTAG TEMPLATE
do
    # Determine output directory
    STEM=$(echo -n $SPECIES | perl -p -e 's/\s+/_/g')
    OUTFOLDER=$ASM/${STEM}.spades
    echo -e "OUTPUT:\n\t${OUTFOLDER}"
    if [ -f $OUTFOLDER/scaffolds.fasta ]; then
	echo -e "\tSkipping -> already run"
	break
    fi
    # Run spades with either --meta or --plasmid
    if [ -d $OUTFOLDER ]; then
	echo "Restarting spades.py --meta -o $OUTFOLDER"
	time spades.py --threads $CPU -o $OUTFOLDER -m $MEM --restart-from last
    else
	echo "Running spades.py --meta --threads $CPU -m $MEM -1 ${INFOLDER}/${STEM}_R1.fq.gz -2 ${INFOLDER}/${STEM}_R2.fq.gz -o $OUTFOLDER"
	time spades.py --meta --threads $CPU -m $MEM --only-assembler \
	    -1 ${INFOLDER}/${STEM}_R1.fq.gz -2 ${INFOLDER}/${STEM}_R2.fq.gz \
		-o $OUTFOLDER
    fi
    # Clean up and compress
    if [ -f $OUTFOLDER/scaffolds.fasta ]; then
	echo "Cleaning..."
	rm -rf $OUTFOLDER/before_rr.fasta $OUTFOLDER/corrected $OUTFOLDER/K*
	rm -rf $OUTFOLDER/assembly_graph_after_simplification.gfa $OUTFOLDER/tmp
	if [ -f $OUTFOLDER/contigs.fasta ]; then
	    pigz $OUTFOLDER/contigs.fasta
	    pigz $OUTFOLDER/spades.log
	fi
    fi    
done
