#!/bin/bash

# (c) Victor Soria-Carrasco
# victor.soria.carrasco@gmail.com
# Last modified: 23/04/2018 23:01:10
# Modified by Alexandra Jansen van Rensburg to run on Blue Crystal p3 (UoB)
# Last modified 23/10/2018

# Description:
# Call SNPs using a parallel approach using multiple nodes on BlueCrystal p3 (University of Bristol)
# It splits the reference in multiple regions and call SNPs for
# each one of them. 

# Changelog
# 1.0.1 - 05/07/2016
#	- Fixed minor bug tags input options
# 1.0.2 - 22/11/2016
#	- Fixed minor bug identifying SNP caller path
# 1.0.3 - 03/08/2017
#   - Added check for indexes of bam files and automatic
#     indexing if required
# 1.0.4 - 04/08/2017
#   - Updated samtools to version 1.5
# 1.0.5 - 23/08/2017
#   - Included flag for importing environment variables
# 1.0.6 - 23/08/2017
#   - Included loading of Genomics Software Repository
# 1.0.7 - 23/11/2017
#   - Included flag for calling invariants too
#   - Included flag for calling indels too
# 1.0.8 - 23/04/2018
#   - Updated samtools to version 1.8
# 1.1.0 - 23/10/2018
#   - Update for BlueCrystal compatibility
#	- change caller from call_SNVs.pl to call_SNVs_bluecrystal.pl
#	- Change SGE options to PBS options (for bluecrystal queue compatibility)
#	- Change Array ID to PBS compatible name: PBS_ARRAYID
# 	- Add section for loading necessary modules
#	- Add section to run script in local directory: Scratch on bluecrystal is the user directory
#	- Change chmod +x to chmod u+x for submission script
# 1.1.1 - 02/05/2019
#   - add -f for bcftools call to calculate individual GQ. Currently we're estimating GQ for the locus (i.e. likelihood that the locus has an alt allele), but not an individual's GQ (i.e. confidence in individual genotype) 
# 1.1.2 - 03/06/2019
# Modified path to caller
#

VERSION='1.1.0-2018.10.23'

#
#

#Define module versions to be loaded
MODULE1='apps/samtools-1.8'
MODULE2='apps/bcftools-1.8'
MODULE3='languages/perl-5.14.2'
MODULE4='languages/java-jdk-1.8.0-66'

#
#

CALLER='/newhome/aj18951/bristol-velocity/AJvR_VelocityPipeline/tools/call_SNVs_bluecrystal.pl'
SAMTOOLS='samtools'

# Default values for optional variables
REGIONS=''
CALLMODE='c'
VARONLY='1'
INDELS='0'
MINMQS=20
PVAR='0.05'
EMAIL=''
HRS=24
MEM=16
NJOBS=100
MAXNREGSPERJOB=50
JOBNAME='callSNVs'

# Show author
# -----------------------------------------------------------------------------
function author {
	echo
	echo "#########################################"
	echo "  $(basename $0)"
	echo "  version $VERSION"
	echo "  (c) Victor Soria-Carrasco"
	echo "  victor.soria.carrasco@gmail.com"
	echo "#########################################"
	echo
}
# -----------------------------------------------------------------------------

# Show usage
# -----------------------------------------------------------------------------
function usage {
	echo
	echo "Usage:"
	echo "  $(basename $0)"
	echo "      -i <input directory>  => Directory with BAM files (one file per sample/individual)"
	echo "      -r <fasta file>       => Reference sequence used for alignment"
	echo "      -o <output directory> => Output folder"
	echo "      -regs <regions file>  => Regions file (optional, use scaffolds/chromosomes grouped for up to $NJOBS jobs otherwise)"
	echo "      -c <c|m>              => bcftools caller; c=consensus, m=multi-allelic (optional, default=$CALLMODE)"
	echo "      -v <0|1>              => call only variants (optional, default=$VARONLY)"
	echo "      -d <0|1>              => call also indels (optional, default=$INDELS)"
	echo "      -s <quality score>    => minimum phred quality score for alignments to be used for calling (optional, default=$MINMQS)"
	echo "      -p <float 0-1>        => prob. of data under the hypothesis that locus is invariant (optional, default=$PVAR)"
	echo "      -t <allocated time>   => Allocated time (in hours) for the analysis (optional: default=$HRS)"
	echo "      -m <allocated memory> => Allocated memory (in gigabytes) for each analysis (optional: default=$MEM)"
	echo "      -q <queue>            => SGE queue: iceberg | popgenom | molecol (optional: default=iceberg)"
	echo "      -e <email>            => Notification email address (default=none)"
	echo "      -module1 <ModuleName> => Name of module to load"
	echo "      -module2 <ModuleName> => Name of module to load" 
	echo "      -module3 <ModuleName> => Name of module to load"
	echo "      -module4 <ModuleName> => Name of module to load"
	echo "      -job <name of job>    => Name of current job"
	echo "      -h                    => show this help"
	echo ""
	echo "  Example:"
	echo "      $(basename $0) -i input_dir -o output_dir -r reference -q popgenom"
	echo ""
	echo ""
	exit 0
}
# -----------------------------------------------------------------------------

author

# Get options from the command line
# -----------------------------------------------------------------------------
if [ "$#" -ge "6" ]; # min 6 args: 2 for -i <input directory>, 2 for -r <fasta file>, 2 for -o <output directory>
then 
	while [ $# -gt 0 ]; do
		case "$1" in
			-h|-help) usage
					  ;;
			-i)	shift
				INDIR=$(readlink -f $1)
				;;
			-r)	shift
				REFERENCE=$(readlink -f $1)
				;;
			-o)	shift
				OUTDIR=$(readlink -f $1)
				;;
			-regs)	shift
				REGIONS=$(readlink -f $1)
				;;
			-c)	shift
				CALLMODE=$1
				;;
			-v)	shift
				VARONLY=$1
				;;
			-d)	shift
				INDELS=$1
				;;
			-s)	shift
				MINMQS=$1
				;;
			-p)	shift
				PVAR=$1
				;;
			-t)	shift
				HRS=$1
				;;
			-m)	shift
				MEM=$1
				;;
			-q)	shift
				QUEUE=$1
				;;
			-module1)	shift
					MODULE1=$1
					;;
                        -module2)	shift
                                        MODULE2=$1
                                        ;;
                        -module3)	shift
                                        MODULE3=$1
                                        ;;
                        -module4)	shift
                                        MODULE4=$1
                                        ;;
			-job)	shift
				JOBNAME=$1
				;;
			-e)	shift
				EMAIL=$1
				;;	
			*)	echo 
				echo "ERROR - Invalid option: $1"
				echo
				usage
				;;
		esac
		shift
	done
else
	usage
fi
# -----------------------------------------------------------------------------

# Check $INDIR, $OUTDIR are $REFERENCE are defined
# -----------------------------------------------------------------------------
if [[ -z $INDIR || -z $OUTDIR || -z $REFERENCE ]]; then
	if [ -z $INDIR ]; then
		echo
		echo "ERROR: You must specify an input directory"
		echo
	fi
	if [ -z $OUTDIR ]; then
		echo
		echo "ERROR: You must specify an output directory"
		echo
	fi
	if [ -z $REFERENCE ]; then
		echo
		echo "ERROR: You must specify a reference file"
		echo
	fi
	usage
fi
# -----------------------------------------------------------------------------

# Check $INDIR and $REFERENCE exist
# -----------------------------------------------------------------------------
if [ ! -d $INDIR ]; then
	echo
	echo "ERROR: I can't find the input directory $INDIR"
	echo
	exit
fi

if [ ! -f $REFERENCE ]; then
	echo
	echo "ERROR: I can't find the reference file $REFERENCE"
	echo
	exit
fi
# -----------------------------------------------------------------------------

# Check $CALLMODE is OK

if [[ $CALLMODE != 'c' && $CALLMODE != 'm' ]];
then
	echo
	echo "ERROR: caller $CALLMODE not recognised (it must be 'c' or 'm')"
	echo
	exit
fi

# Check $VARONLY is OK

if [[ $VARONLY != '0' && $VARONLY != '1' ]];
then
	echo
	echo "ERROR: call only variants option $VARONLY not recognised (it must be '0' or '1')"
	echo
	exit
fi


# Get number of bam files in the input directory
N=$(find $INDIR -maxdepth 1 -name "*.bam" | wc -l | cut -f1 -d" ")

# Check there are bam files in the input directory
# -----------------------------------------------------------------------------
if [ "$N" -eq "0" ]; then
	echo
	echo "ERROR: No BAM files in $INDIR"
	echo
	exit
fi
# -----------------------------------------------------------------------------

# Index reference if it is not indexed
# -----------------------------------------------------------------------------
if [ ! -e $REFERENCE.fai ]; then
	echo
	echo "Building indexes for reference $REFERENCE..."
	echo "(this might take quite a bit...)"
	echo
	$SAMTOOLS faidx $REFERENCE >& $REFERENCE.faidx.log
	echo
	echo "Done. Log file saved in $REFERENCE.faidx.log"
	echo
	echo "----"
	echo
fi
# -----------------------------------------------------------------------------

# Index bam files if they are not indexed
# -----------------------------------------------------------------------------
for bam in $(find $INDIR -maxdepth 1 -name "*.bam");
do
	bai=${bam%.*}.bai
	if [[ ! -e $bam.bai && ! -e $bai ]];
	then
		echo "No index found for $bam, indexing..."
		$SAMTOOLS index $bam
	fi
done
echo
# -----------------------------------------------------------------------------

# Create output directory
# -----------------------------------------------------------------------------
if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi
# -----------------------------------------------------------------------------

# Get all regions from reference if no file is given
# -----------------------------------------------------------------------------
if [[ $REGIONS == ''  ||  ! -f $REGIONS ]];
then
	REFSIZE=$(cat $REFERENCE.fai | awk '{SUM+=$2}END{print SUM}')
	JOBSIZE=$(perl -e 'print int(('$REFSIZE'/'$NJOBS')+1)')
	REGIONS=($(cat $REFERENCE.fai | sort -V | awk '{print $1}'))
	REGSIZES=($(cat $REFERENCE.fai | sort -V | awk '{print $2}'))	
	# echo "Generating regions file (splitting genome in ~$NJOBS regions of ~$JOBSIZE bp)..."
	echo "Generating regions file "
	echo "  splitting genome in chunks:"
	echo "    max size per job:  $JOBSIZE bp"
	echo "    max no regions per job: $MAXNREGSPERJOB"
	echo 

	CUMSIZE=0
	NREGS=0
	echo -n > $OUTDIR/regions
	RC=1
	# for I in $REGIONS;
	for ((i=0; i<${#REGSIZES[*]}; i++));
	do
		# REGSIZE=$(cat $REFERENCE.fai | awk '{if ($1=="'$REG'") print $2}')
		REG=${REGIONS[$i]}
		REGSIZE=${REGSIZES[$i]}
		CUMSIZE=$(($CUMSIZE + $REGSIZE))

		if [[ $CUMSIZE -gt $JOBSIZE || $NREGS -gt $MAXNREGSPERJOB ]];
		then
			echo "$REG" >> $OUTDIR/regions
			# echo "Set of regions size: $CUMSIZE"
			CUMSIZE=0
			NREGS=0
			echo -ne "  number of jobs: $RC\r"
			RC=$(($RC + 1))
		else
			echo -n "$REG," >> $OUTDIR/regions
			NREGS=$((NREGS + 1))
		fi
	done
	
	if [[ $CUMSIZE -gt 0 ]];
	then
		echo -e "  number of jobs: $RC\r"
		echo >> $OUTDIR/regions
	fi
	
	perl -pi -e 's/,$//g' $OUTDIR/regions
	echo
	echo
else
	cp $REGIONS $OUTDIR/regions
fi

REGIONS=$OUTDIR/regions

# number of regions
N=$(cat $REGIONS | wc -l)
# -----------------------------------------------------------------------------

# Make submission file
# -----------------------------------------------------------------------------
SMSJOB="$OUTDIR/var_calling.$(date +%Y%m%d-%H%M%S).smsjob.sh"
LOG="$OUTDIR/var_calling.$(date +%Y%m%d-%H%M%S).smsjob.log"


# Initialize submission script
echo '#!/bin/bash' > $SMSJOB

# PBS OPTIONS
# -----------------------------------------
echo '#PBS -N '$JOBNAME'' >> $SMSJOB  
echo '#PBS -l nodes=1:ppn=1' >> $SMSJOB 
echo '#PBS -l mem='$MEM'gb' >> $SMSJOB 
echo '#PBS -l walltime='$HRS':00:00' >> $SMSJOB
echo '#PBS -j oe' >> $SMSJOB #concatenates error and output files (with prefix job1)
echo '#PBS -t 1-'$N >> $SMSJOB
echo '#PBS -o '$LOG >> $SMSJOB
# -----------------------------------------


echo '# Run in local directory' >> $SMSJOB
echo 'cd $PBS_O_WORKDIR' >> $SMSJOB
echo ''
echo ''


echo '# Load all necessary modules' >> $SMSJOB
echo 'module load '$MODULE1'' >> $SMSJOB
echo 'module load '$MODULE2'' >> $SMSJOB
echo 'module load '$MODULE3'' >> $SMSJOB
echo 'module load '$MODULE4'' >> $SMSJOB
echo ''
echo ''


VARTXT="variants"
if [[ $VARONLY == "0" ]];
then
	VARTXT="$VARTXT and invariants"
fi

cat >> $SMSJOB <<EOF

CALLER="$CALLER"

BAMDIR="$INDIR"
REFERENCE="$REFERENCE"
OUTDIR="$OUTDIR"
REGIONS=(\$(cat $REGIONS))

INDEX=\$PBS_ARRAYID

REG=\${REGIONS[\$INDEX]}

JOBID="job"\$(printf %04d \$INDEX)
OUTDIR="\$OUTDIR/\$JOBID"
IDLOG="$OUTDIR/\$JOBID.var_calling.log"

hostname > \$IDLOG
uname -a >> \$IDLOG
date >> \$IDLOG
echo "---------------------------------------------------" >> \$IDLOG
echo >> \$IDLOG
echo "Calling $VARTXT for region/s: \$REG..." >> \$IDLOG
echo >> \$IDLOG
echo "CMD: " >> \$IDLOG
echo "    \$CALLER \\\" >> \$IDLOG
echo "    -b \$BAMDIR \\\" >> \$IDLOG
echo "    -r \$REFERENCE \\\" >> \$IDLOG
echo "    -o \$OUTDIR \\\" >> \$IDLOG
echo "    -regs \$REG \\\" >> \$IDLOG
echo "    -n 1 \\\" >> \$IDLOG
echo "    -c $CALLMODE \\\" >> \$IDLOG
echo "    -v $VARONLY \\\" >> \$IDLOG
echo "    -d $INDELS \\\" >> \$IDLOG
echo "    -q $MINMQS \\\" >> \$IDLOG
echo "    -p $PVAR \\\" >> \$IDLOG
echo "    -quiet \\\" >> \$IDLOG
echo "    -clean" >> \$IDLOG
echo  >> \$IDLOG
echo "---------------------------------------------------" >> \$IDLOG
echo >> \$IDLOG

\$CALLER \\
-b \$BAMDIR \\
-r \$REFERENCE \\
-o \$OUTDIR \\
-regs \$REG \\
-n 1 \\
-c $CALLMODE \\
-v $VARONLY \\
-d $INDELS \\
-q $MINMQS \\
-p $PVAR \\
-f gc,gp \\
-quiet \\
-clean \\
>> \$IDLOG 2>&1


# Move and rename output
if [[ -e \$OUTDIR/variants.raw.bcf ]];
then
	mv \$OUTDIR/variants.raw.bcf \$OUTDIR/../\$JOBID.variants.raw.bcf
	mv \$OUTDIR/variants.raw.bcf.csi \$OUTDIR/../\$JOBID.variants.raw.bcf.csi
else
	if  [[ -e \$OUTDIR/sites.raw.bcf ]];
	then
		mv \$OUTDIR/sites.raw.bcf \$OUTDIR/../\$JOBID.sites.raw.bcf
		mv \$OUTDIR/sites.raw.bcf.csi \$OUTDIR/../\$JOBID.sites.raw.bcf.csi
	else
		echo "No variants called for this job" >> \$IDLOG
	fi
fi

REPORT=\$(ls \$OUTDIR/*report.txt)
OUTREPORT=\$(basename \$REPORT | perl -pe '\$_='\$JOBID'.".\$_";')
mv \$REPORT \$OUTDIR/../\$OUTREPORT

# Clean
rm -r \$OUTDIR

echo >> \$IDLOG
echo "---------------------------------------------------" >> \$IDLOG
echo >> \$IDLOG
date >> \$IDLOG
echo >> \$IDLOG

EOF

chmod u+x $SMSJOB

echo "Command to submit the job to Blue Crystal queue:"
echo
echo "qsub $SMSJOB"
echo
