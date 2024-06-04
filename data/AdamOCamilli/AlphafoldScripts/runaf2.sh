#!/bin/bash
#title           : runaf2.sh
#description     : This script submits one or more Alphafold batch jobs using a folder of proteins (.fasta or .fa files)
#author		 : Adam Camilli (a.o.camilli25@gmail.com)
#date            : 01/04/2022
#version         : 1.0    
#notes           : Configured to work only while already logged on to Tufts HPC cluster
#bash_version    : 4.2.46(2)-release (x86_64-redhat-linux-gnu)  
#==============================================================================

# Display correct usage if input is incorrect
function usage() {
  scriptName=$(basename $0);
  printf "usage: ./$scriptName [-h] [-o DIRECTORY] [-e DIRECTORY] [FILE/DIRECTORY]\n"
  printf "Run one or more Alphafold batch jobs for each protein sequence file in a directory. You can also pass in individual sequence files.\n\n"
  printf "  -h         display help\n"
  printf "  -e error   specify output directory \n"
  printf "             default format: alphafold_error_<current datetime>\n"
  printf "  -o output  specify error directory  (in current directory by default)\n"
  printf "             default format: alphafold_output_<current datetime>\n"
}

# One way to pass command-line arguments to job script
# From: https://stackoverflow.com/a/36303809  
function run_sbatch () {
sbatch <<EOT
#!/bin/sh
 
#SBATCH -p preempt  #if you don't have ccgpu access, use "preempt"
#SBATCH -n 8	# 8 cpu cores #SBATCH --mem=64g	#64GB of RAM
#SBATCH --mem=64g	#64GB of RAM
#SBATCH --time=2-0	#run 2 days, up to 7 days "7-00:00:00"
#SBATCH -o $1/output.%j # STDOUT
#SBATCH -e $1/error.%j  # STDOUT
#SBATCH -N 1
#SBATCH --gres=gpu:1	# number of GPUs, using v100 --gres=gpu:v100:1, using a100 --gres=gpu:a100:1
 
export alphafold_path=/cluster/tufts/cmdb295class/shared/alphafold/alphafold
module load cuda/11.0 cudnn/8.0.4-11.0 anaconda/2021.05
module list
nvidia-smi
 
source activate af2
 
#Make sure to specify the output_dir to a path that you have write permission
python3 /cluster/tufts/cmdb295class/shared/alphafold/alphafold/run_af2.py \
--output_dir=$2 \
--fasta_paths=$3
EOT
}

numArg=$#
errorDir=""
eFlag=""
outputDir=""
oFlag=""
proteinFiles=()

while getopts he:o: flag; do
  case "${flag}" in
    h) 
      usage
      exit 0;;
    e) 
       eFlag=1
       errorDir="$(realpath $OPTARG)"
       mkdir -p $errorDir;;
    o)
       oFlag=1
       outputDir="$(realpath $OPTARG)"
       mkdir -p $outputDir;;
   esac
done
shift $(($OPTIND - 1))

# Set default output and error directories
currentDateTime="$(date +%Y%m%d_%H%M%S)"
errorTitle="alphafold_error_$currentDateTime"
outputTitle="alphafold_output_$currentDateTime"

# Place in current directory by default
if  [[ -z $eFlag ]];
then
  mkdir $errorTitle
  errorDir="$(realpath ./$errorTitle)"
fi
if [[ -z $oFlag ]];
then
  mkdir $outputTitle
  outputDir="$(realpath ./$outputTitle)"
fi

# Run batch script for each protein sequence file 
if [[ -d $* ]]; 
then	
  shopt -s nullglob # don't match empty files
  for file in "$1"/*.{fasta,fa}; do
    proteinFiles+=("$(realpath $file)")
  done
  shopt -u nullglob
  for proteinFile in "${proteinFiles[@]}"; do
    proteinDir=$(basename $proteinFile)
    proteinErrorDir="$errorDir/${proteinDir%.*}"  
    mkdir -p $proteinErrorDir
    run_sbatch $proteinErrorDir $outputDir $proteinFile
  done
  exit 0
elif [[ -f $* ]];
then
  proteinDir=$(basename $*)
  proteinErrorDir="$errorDir/${proteinDir%.*}"  
  mkdir -p $proteinErrorDir
  run_sbatch $proteinErrorDir $outputDir $*
  exit 0
else
  echo "Invalid or nonexistent directory $*"
  usage
  exit 1
fi
