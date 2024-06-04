#!/bin/bash
#PBS -N SPAdes
#PBS -l select=1:ncpus=30:ompthreads=30:mem=150gb:scratch_local=30gb
#PBS -l walltime=48:00:00
#PBS -m ae
#PBS -j oe

cat $PBS_NODEFILE

#add modules
source /cvmfs/software.metacentrum.cz/modulefiles/5.1.0/loadmodules
module load spades

datadir='/storage/brno3-cerit/home/kika/UGA_decoding/peritromus/'
reads=$datadir'reads/'

#copy reads to scratch
cp $reads'pkah_trimmed_1.fq.gz' $reads'pkah_trimmed_2.fq.gz' $SCRATCHDIR


#compute on scratch
cd $SCRATCHDIR

fwd='pkah_trimmed_1.fq.gz'
rev='pkah_trimmed_2.fq.gz'

# #spades assembly
# spades.py -t $PBS_NUM_PPN \
# 	-1 $fwd -2 $rev \
# 	-o spades

# #metagenome assembly
# metaspades.py -t $PBS_NUM_PPN \
# 	-1 $fwd -2 $rev \
# 	-o metaspades

# #metagenome specifying k-mers
# metaspades.py --pe1-1 $pe1_1 --pe1-2 $pe1_2 -k 21,33,55,77,99,111  -t $PBS_NUM_PPN -o spades_kmers

# #using reference genome
# spades.py --pe1-1 $pe1_1 --pe1-2 $pe1_2 --trusted-contigs $cbom --careful -t $PBS_NUM_PPN -o spades_cbom_trusted

#single-cell specifying k-mers
spades.py --sc -1 $fwd -2 $rev -k 61,71,77,81,99,111  -t $PBS_NUM_PPN -o spades

# #single-cell using several libraries
# spades.py --sc --careful -t $PBS_NUM_PPN -o out \
# --pe1-m $pe1m --pe1-1 $pe11 --pe1-2 $pe12 --pe1-s $pe1u \
# --pe2-m $pe2m --pe2-1 $pe21 --pe2-2 $pe22 --pe2-s $pe2u \

#copy results back
rm *gz
cp -r * $datadir
