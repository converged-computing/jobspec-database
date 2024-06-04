#PBS -l walltime=00:10:00
#PBS -l nodes=1:ppn=14
#PBS -N trainSNAP_rnd2
#PBS -A PAS1582

module unload xalt

cd /fs/scratch/PAS1582/`whoami`
#mkdir snap
cd  snap 
mkdir round2
cd round2
export ZOE=/fs/scratch/PAS1582/HCS7194_Files/Genome_Annotation/bin/snap/Zoe 
# export 'ALL' gene models from MAKER round 1 prediction
singularity exec ../../maker_version2.sif /usr/local/bin/maker/bin/maker2zff -n -d ../../Almond_pred1.maker.output/Almond_pred1_master_datastore_index.log

export PATH=$PATH:/fs/scratch/PAS1582/HCS7194_Files/Genome_Annotation/bin/snap/
# gather some stats and validate
fathom genome.ann genome.dna -gene-stats > gene-stats.log 2>&1
fathom genome.ann genome.25.dna -validate > validate.log 2>&1
# collect the training sequences and annotations, plus 1000 surrounding bp for training
fathom genome.ann genome.dna -categorize 1000 > categorize.log 2>&1
fathom uni.ann uni.dna -export 1000 -plus > uni-plus.log 2>&1
# create the training parameters
mkdir params
cd params
forge ../export.ann ../export.dna > ../forge.log 2>&1
cd ..
# assembly the HMM
hmm-assembler.pl genome params > Almond_BC_rnd2.zff.hmm
