#PBS -l nodes=1:ppn=1
#PBS -l mem=20gb
#PBS -l walltime=10:00:00
#PBS -e biorep_merged/results/peaks/logs/
#PBS -o biorep_merged/results/peaks/logs/
#PBS -N run_fithichip_peakcalling
#PBS -V

# example run:
# 1) qsub -t <index1>,<index2>,... workflow/scripts/peaks/run_fithichip_peakcalling.sh
# 2) qsub -t <index-range> workflow/scripts/peaks/run_fithichip_peakcalling.sh
# 3) qsub -t <index1>,<index2>,<index-range1> workflow/scripts/peaks/run_fithichip_peakcalling.sh
# 4) qsub -t <index-range1>,<index-range2>,... workflow/scripts/peaks/run_fithichip_peakcalling.sh
# 5) qsub -t <any combination of index + ranges> workflow/scripts/peaks/run_fithichip_peakcalling.sh

# print start time message
start_time=$(date "+%Y.%m.%d.%H.%M")
echo "Start time: $start_time"

# print start message
echo "Started: fithichip_peakcalling"

# run bash in strict mode
set -euo pipefail
IFS=$'\n\t'

# make sure to work starting from the github base directory for this script 
cd $PBS_O_WORKDIR
work_dir=$PBS_O_WORKDIR

# source tool paths
source workflow/source_paths.sh

# extract the sample information using the PBS ARRAYID
samplesheet="results/samplesheets/post-hicpro/mouse.biorep_merged.samplesheet.without_header.tsv"
sample_info=( $(cat $samplesheet | sed -n "${PBS_ARRAYID}p") )
sample_name="${sample_info[0]}"
org="${sample_info[1]}"

# printing sample information
echo
echo "Processing"
echo "----------"
echo "sample_name: $sample_name"
echo

# make the output directory
outdir="biorep_merged/results/peaks/fithichip/${sample_name}"
mkdir -p ${outdir}
#cat_outdir="results/peaks/fithichip/${sample_name}/cat_pairs/"
#mkdir -p $cat_outdir

ln -s -r -f biorep_merged/results/hicpro/${sample_name}/*.allValidPairs biorep_merged/results/hicpro/${sample_name}/rawdata_allValidPairs
#ln -s -r -f biorep_merged/results/hicpro/${sample_name}/*.DEPairs ${outdir}/
#ln -s -r -f biorep_merged/results/hicpro/${sample_name}/*.SCPairs ${outdir}/
#ln -s -r -f biorep_merged/results/hicpro/${sample_name}/*.REPairs ${outdir}/

# concatenate pairs files
# echo "# Concatenating pairs files"
# pairs_folder="biorep_merged/results/hicpro/${sample_name}"
# cd $pairs_folder

# if [ $(find -name "*.DEPairs" | wc -l) -ne 0 ]; then
#     cat *'.DEPairs' >> "${work_dir}/${cat_outdir}all_${sample_name}.bwt2pairs.DEPairs"
# fi

# if [ $(find -name "*.SCPairs" | wc -l) -ne 0 ]; then
#     cat *'.SCPairs' >> "${work_dir}/${cat_outdir}all_${sample_name}.bwt2pairs.SCPairs"
# fi

# if [ $(find -name "*.REPairs" | wc -l) -ne 0 ]; then
#     cat *'.REPairs' >> "${work_dir}/${cat_outdir}all_${sample_name}.bwt2pairs.REPairs"
# fi

# if [ $(find -name "*.allValidPairs" | wc -l) -ne 0 ]; then
#     cat *'.allValidPairs' >> "${work_dir}/${cat_outdir}rawdata_allValidPairs"
# fi

# cd $work_dir
# echo "# Concatenation done"
# echo

# reference genome
if [[ "$org" == "Homo_Sapiens" ]];
then
    refGenomeStr="hs"
elif [[ "$org" == "Mus_Musculus" ]];
then
    refGenomeStr="mm"
else
    echo "valid org not found"
    exit
fi
echo "Using genome: $refGenomeStr"

# get read length from samplesheet
file_samplesheet="results/samplesheets/post-hicpro/readlength.mouse.biorep_merged.samplesheet.without_header.tsv"
unset IFS
sample_info=( $(grep "${sample_name}" ${file_samplesheet}) )
# use the readlength of the first biological rep
#sample_info=( $(grep $(echo ${sample_name} | awk '{ print substr( $0, 1, length($0)-14 ) }') ${file_samplesheet}) )
ReadLengthR1=${sample_info[1]}
ReadLengthR2=${sample_info[2]}
echo "Using read length R1: $ReadLengthR1"
echo "Using read length R2: $ReadLengthR2"

# run fithichip peak calling
echo
echo "# running fithichip peak calling"
$fithichip_peakinferhichip -H biorep_merged/results/hicpro/${sample_name}/ -D $outdir -R $refGenomeStr -L $ReadLengthR1 -G $ReadLengthR2

# print end message
echo
echo "# Ended: fithichip peak calling"

# remove cat_pairs dir
# echo
# echo "# removing catpairs dir"
# echo
# rm -r $cat_outdir

# print end time message
end_time=$(date "+%Y.%m.%d.%H.%M")
echo "End time: $end_time"