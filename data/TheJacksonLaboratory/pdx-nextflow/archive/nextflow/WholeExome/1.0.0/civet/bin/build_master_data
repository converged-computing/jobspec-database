#!/bin/bash
#PBS -q CLIA -l nodes=1:ppn=1,walltime=0:30:00

if [ "X$PIPELINE" == "X" ]; then
    PIPE=/opt/compsci/cga/${1}/exome/single_sample_exome.xml
    if [ ! -e $PIPE ] ; then
        echo You must specify a valid CGA version.
        exit 1
    fi
    # qsub with -v "PIPELINE=/path/to/pipeline.xml"
    echo -n "Building master data file in job:  "
    qsub -v "PIPELINE=${PIPE},VERSION=${1}" ${0}
    exit
fi
cd $PBS_O_WORKDIR

pwd
date

# Make sure this module list is up-to-date; check with
# cd cga # Get into a cga working directory
# (find . -name "*.xml" -exec grep module {} \+) | awk '{print $2;}' | sort | uniq

module load cga/${VERSION}
module load bwa/0.7.9a
module load GATK/3.1-1
module load java/1.7.0
module load NGSQCToolkit/2.3
module load R/3.0.2
module load samtools/0.1.19
module load somaticsniper/1.0.2
module load bedtools/2.17.0
module load pindel/0.2.5a3
module load perl/cga


module list

# Remove the existing master_file_list if one exists.  We want to create
# a new one, not update an existing one.
rm -f master_file_list

# The script requires dummy parameters to satisfy the XML.
create_file_validation_data ${PIPELINE} a b c d

# That command created master_file_list in the CWD.  Copy it to 
# the proper place.

VAL_FILE=/opt/compsci/cga/${VERSION}/exome/single_sample_exome_validation.data
CMD="cp master_file_list ${VAL_FILE}.new"
echo $CMD
$CMD

#
# Move it into place with an atomic operation

CMD="mv ${VAL_FILE}.new ${VAL_FILE}"
echo $CMD
$CMD
