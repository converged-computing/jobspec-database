#PBS -l nodes=1:ppn=4 -N mergebams.Yeast -j oe -o mergebams.Yeast
#PBS -d /shared/gen220/Yeast
module load java/1.7.0_17
module load picard

# in the config above we requested 16 procicessors on one node - (nodes=1:ppn=16) 

umask 0002 # let's make sure the umask is setup write so default permissions are easier for jason to see
BAMDIR=bam

# we want the comamnd line to be 
# INPUT=file1.bam INPUT=file2.bam ... OUTPUT=all_strains.combined.bam
# The following will make a single string INPUT=A.bam INPUT=b.bam ... for the files in the 'bam' dir
INPUTSTRING=`find $BAMDIR -name '*.bam' -exec echo -n "INPUT={} " \;`

java -jar $PICARD/MergeSamFiles.jar USE_THREADING=true MERGE_SEQUENCE_DICTIONARIES=true \
 ASSUME_SORTED=true $INPUTSTRING OUTPUT=All.combined.bam CREATE_INDEX=TRUE
