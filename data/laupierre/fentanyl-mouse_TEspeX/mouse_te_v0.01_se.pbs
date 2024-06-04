#!/bin/bash
#PBS -l walltime=40:00:00
#PBS -l select=1:ncpus=12
#PBS -q workq
#PBS -N tespex

cd $PBS_O_WORKDIR


#### This is the TEspeX analysis for the mouse species - SE reads (v.0.0.1dev)

#### Variables
CONTAINER=/projects/ncrrbt_share_la/dev_pipe3
CPUS=12
LENGTH=50
ORIENTATION=reverse


## Insert BBMap filtering if necessary

#### TEspeX counting of TE families

array=("LINE" "SINE" "DNA" "LTR" "RC" "Other" "Retroposon")

for cls in ${array[@]}
do
   apptainer exec $CONTAINER/tespex.sif /bin/bash -c \
   "/home/famdb.py -i /home/Dfam_curatedonly.h5 families --include-class-in-name --class $cls -f fasta_name -ad 'Mus musculus' >> mmusculus.Dfam.fa"
done



find $(pwd) -maxdepth 1 -type f -path '*.gz' > reads.txt
gzip mmusculus.Dfam.fa

## sequence of mouse transcripts
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M32/gencode.vM32.lncRNA_transcripts.fa.gz
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M32/gencode.vM32.pc_transcripts.fa.gz
cdna=gencode.vM32.pc_transcripts.fa.gz
ncrna=gencode.vM32.lncRNA_transcripts.fa.gz



#### TEspeX for 50 nts SE reads in reverse orientation

# conda activate TEspeX_deps  ## no need to activate with sh, needed with bash

apptainer exec $CONTAINER/tespex.sif sh -c \
	"cd /home/TEspeX
	 tespex=$PWD

	 python3 TEspeX.py --num_threads $CPUS --TE $PBS_O_WORKDIR/mmusculus.Dfam.fa.gz \
	 --cdna $PBS_O_WORKDIR/$cdna \
	 --ncrna $PBS_O_WORKDIR/$ncrna \
	 --sample $PBS_O_WORKDIR/reads.txt --paired F --length $LENGTH --out $PBS_O_WORKDIR/tespex_results --strand $ORIENTATION"
   
   


