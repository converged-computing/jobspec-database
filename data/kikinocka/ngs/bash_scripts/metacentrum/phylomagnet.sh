#!/bin/bash
#PBS -N phylomagnet
#PBS -l select=1:ncpus=20:mem=20gb:scratch_local=20gb
#PBS -l walltime=02:00:00
#PBS -m ae
#PBS -j oe

cat $PBS_NODEFILE

data_dir='/storage/brno3-cerit/home/kika/oil_sands/metagenomes/P3S_1-02B_L001-ds.971c07c67a83443891de04bf749cee0b/'
eggnog_dir='/storage/brno3-cerit/home/kika/databases/'

#copy files to scratch
cp $data_dir'1-reads/P3S_all_trimmed_combined.fq.gz' $SCRATCHDIR
# cp $data_dir'1-reads/P3S_all_trimmed_2.fq.gz' $SCRATCHDIR
cp $eggnog_dir'eggnog_ogs.txt' $SCRATCHDIR


#compute on scratch
cd $SCRATCHDIR

reads='P3S_all_trimmed_combined.fq.gz'
eggnog='eggnog_ogs.txt'
lineage='Cryptophyceae,Intramacronucleata,family'
aligner='mafft-linsi'
tree='iqtree'

singularity shell -B /afs,/software,/packages,$SCRATCHDIR /software/phylomagnet/PhyloMagnet.sif <<END
module add phylomagnet-0.7

nextflow run /software/phylomagnet/0.7/main.nf \
	--reference_classes $eggnog\
	--fastq $reads \
	--lineage $lineage \
	--queries_dir 'queries_out'\
	--reference_dir 'refs_out'\
	--database 'ncbi'\
	--align_method $aligner\
	--phylo_method $tree \
	--cpus $PBS_NUM_PPN
END

#copy results to your folder
rm $reads $eggnog
cp -r * $data_dir
