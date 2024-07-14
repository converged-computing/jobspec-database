############################
#Set Defaults For Variables#
############################
r2threshold=0.7
ldwindowsize=1000000
min_neighbor_num=500
population=EUR
bedfilesorted=false

######################
#Read Input Arguments#
######################

ARGUMENTS=()

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --r2) r2threshold="$2"; shift 2 ;;
        --ldw) ldwindowsize="$2"; shift 2 ;;
        --min) min_neighbor_num=$2; shift 2 ;;
        --pop) population=$2; shift 2 ;;
        --*) echo "Unknown argument: $1"; exit 1 ;;
        *)  ARGUMENTS+=("$1"); shift ;;
    esac
done

if [[ ${#ARGUMENTS[@]} -ne 5 ]]; then
	echo -e "\nUsage: ConfigureGREGOR.sh bedfilelist snpdir refdir configdir outdir \n"
	echo -e "bedfilelist:\tA file with a list of file paths to bed files that will be processed.\n"
	echo -e "snpdir:\tDirectory containing files of either hg19 SNP positions or refSNP ids.\n"
	echo -e "refdir:\tDirectory containing GREGOR reference files.\n"
	echo -e "configdir:\tThe directory where the configuration files will be stored.\n"
	echo -e "outdir:\tThe output directory.\n"

	echo "Options: --r2     r2 threshold. (Default: 0.7)";
	echo "         --ldw LD window size. (Default: 1000000)";
        echo "         --min Mininum number of neighbors. (Default: 500)";
        echo "         --pop The reference population. (Default: EUR)";
	exit 0;
fi

bedfilelist=${ARGUMENTS[0]}
snpindexfiledir=${ARGUMENTS[1]}
REFDIR=${ARGUMENTS[2]}
configdir=${ARGUMENTS[3]}
outdir=${ARGUMENTS[4]}

####################
#WRITE CONFIG FILES#
####################

mkdir -p ${configdir}

allconfigfiles=${configdir}/configfiles.txt

rm -f ${allconfigfiles}
touch ${allconfigfiles}


allconfigouts=${configdir}/configouts.txt

rm -f ${allconfigouts}
touch ${allconfigouts}

find ${snpindexfiledir} -name "*.txt" | while read LINE
do
	snpindexfile=$(basename "${LINE}")
	
	#set up output directory
	curoutputdir=${outdir}${snpindexfile}
	mkdir ${curoutputdir}

	#write config

	outfile=${configdir}${snpindexfile}.config.txt
	echo "##############################################################################" > ${outfile}
	echo "# ATAC-SEQ ENRICHMENT CONFIGURATION FILE" >> ${outfile}
	echo "# This configuration file contains run-time configuration of" >> ${outfile}
	echo "# ATAC-SEQ ENRICHMENT" >> ${outfile}
	echo "###############################################################################" >> ${outfile}
	echo "## KEY ELEMENTS TO CONFIGURE : NEED TO MODIFY" >> ${outfile}
	echo "###############################################################################" >> ${outfile}
	echo "INDEX_SNP_FILE = ${LINE}  " >> ${outfile}
	echo "BED_FILE_INDEX = ${bedfilelist} " >> ${outfile}
	echo "REF_DIR = ${REFDIR}" >> ${outfile}
	echo "R2THRESHOLD = ${r2threshold} ## must be greater than 0.7" >> ${outfile}
	echo "LDWINDOWSIZE = ${ldwindowsize} ## must be less than 1MB; these two values define LD buddies" >> ${outfile}
	echo "OUT_DIR = ${curoutputdir} " >> ${outfile}
	echo "MIN_NEIGHBOR_NUM = ${min_neighbor_num} ## define the size of neighborhood" >> ${outfile}
	echo "BEDFILE_IS_SORTED = ${bedfilesorted}  ## false, if the bed files are not sorted" >> ${outfile}
	echo "POPULATION = ${population}  ## define the population, you can specify EUR, AFR, AMR or ASN" >> ${outfile}
	echo "TOPNBEDFILES = 4 " >> ${outfile}
	echo "JOBNUMBER = 1" >> ${outfile}
	echo "###############################################################################" >> ${outfile}
	echo "#BATCHTYPE = mosix ##  submit jobs on MOSIX" >> ${outfile}
	echo "#BATCHOPTS = -E/tmp -i -m2000 -j10,11,12,13,14,15,16,17,18,19,120,122,123,124,125 sh -c" >> ${outfile}
	echo "###############################################################################" >> ${outfile}
	echo "#BATCHTYPE = slurm   ##  submit jobs on SLURM" >> ${outfile}
	echo "#BATCHOPTS = --partition=main --time=0:30:0" >> ${outfile}
	echo "###############################################################################" >> ${outfile}
	echo "BATCHTYPE = local ##  run jobs on local machine" >> ${outfile}	
	
	echo ${outfile} >> ${allconfigfiles}
	echo ${snpindexfile} >> ${allconfigouts}
done
COUNT=$(find ${snpindexfiledir} -name "*.txt" | wc -l)

############################
#Write the GREGOR run shell#
############################
slurmfile=${configdir}/RunGREGOR.sh
rm -f ${slurmfile}
touch ${slurmfile}

echo \#!/bin/bash >> ${slurmfile}
echo \#SBATCH --nodes=1 >> ${slurmfile}
echo \#SBATCH --ntasks-per-node=1 >> ${slurmfile}
echo \#SBATCH --time=24:00:00 >> ${slurmfile}
echo \#SBATCH --mem-per-cpu=8G >> ${slurmfile}
echo \#SBATCH --job-name=GREGOR >> ${slurmfile}
echo \#SBATCH --array=1-${COUNT}%32 \#Run only 32 of these at one time >> ${slurmfile}

echo module load singularity >> ${slurmfile}
echo configfile=\$\(head -n \$SLURM_ARRAY_TASK_ID ${allconfigfiles} \| tail -1\) >> ${slurmfile}
echo sifpath=\$1 >> ${slurmfile}
echo singularity run \$sifpath RunGREGOR \$configfile >> ${slurmfile}


################################
#Write the GREGOR cleaner shell#
################################
slurmcleanfile=${configdir}/CleanGREGOR.sh
rm -f ${slurmcleanfile}
touch ${slurmcleanfile}

echo \#!/bin/bash >> ${slurmcleanfile}
echo \#SBATCH --nodes=1 >> ${slurmcleanfile}
echo \#SBATCH --ntasks-per-node=1 >> ${slurmcleanfile}
echo \#SBATCH --time=8:00:00 >> ${slurmcleanfile}
echo \#SBATCH --mem-per-cpu=8G >> ${slurmcleanfile}
echo \#SBATCH --job-name=GREGOR >> ${slurmcleanfile}
echo \#SBATCH --array=1-${COUNT}%32 \#Run only 32 of these at one time >> ${slurmcleanfile}

echo configout=\$\(head -n \$SLURM_ARRAY_TASK_ID ${allconfigouts} \| tail -1\) >> ${slurmcleanfile}
echo rm -r ${outdir}/\$configout >> ${slurmcleanfile}
