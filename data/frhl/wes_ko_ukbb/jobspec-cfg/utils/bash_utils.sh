#!/usr/bin/env bash

# set paths to executables
export PATH="${HOME}/htslib_1.11/bin:${PATH}"   # path to htslib 1.11





# utility functions
raise_error() {
  >&2 echo -e "Error: $1. Exiting." && exit 1
}

get_chr() {
  if [ $1 -eq 23 ]; then
    echo "X"
  elif [ $1 -eq 24 ]; then
    echo "Y"
  elif [ $1 -ge 1 ] && [ $1 -le 22 ]; then
    echo $1
  else 
    raise_error "chromosome number must be between 1 and 24"
  fi
}

get_ukbb_free_disk() {
  echo "$( df -h  "/well/lindgren-ukbb/projects/ukbb-11867" | tail -n1 | tr -s " " | cut -d" " -f4)"
}

elapsed_time() {
  if [ ! -z $1 ]; then echo "elapsed time: $( echo "scale=2; $1/3600" | bc -l ) hrs "; fi
}

print_update() {
  local _message=$1
  set +o nounset # disable check for unbound variables temporarily
  local _duration=$2
  echo "${_message} $( elapsed_time ${_duration} )(job id: ${JOB_ID}.${SGE_TASK_ID} $( date ))"
  set -o nounset
}

file_size(){
  local file=${1}
  if [ -f ${file} ]; then
    echo "$( stat --printf="%s" ${file} )"
  else 
    raise_error "${file} does not exist."
  fi
}

from_sci() {
  sci=${1}
  result=$( echo "${sci}" | awk -F"E" 'BEGIN{OFMT="%10.20f"} {print $1 * (10 ^ $2)}' )
  if [ ! -z "${result}" ]; then
    echo ${result}
  fi
}

stopifnot_file_exists() {
  if [ ! -f $1 ]; then # check that VCF exists
    raise_error "$1 does not exist"
  elif [ ! -s $1 ]; then # check that VCF is not an empty file
    raise_error "$1 exists but is empty"
  fi
}

force_rm_bad_vcf() {
  local _vcf=${1}
  if [ -f ${_vcf} ]; then
    if [ ! -s ${_vcf} ]  || [ $( get_eof_error ${_vcf} ) -gt 0 ]; then
      echo "Removing bad VCF: '${_vcf}' (EOF error or empty file)"
      rm -f "${_vcf}" "${_vcf}.tbi"
    fi
 fi
}

#vcf_check() {
#  if [ ! -f $1 ]; then # check that VCF exists
#    raise_error "$1 does not exist"
#  elif [ ! -s $1 ]; then # check that VCF is not an empty file
#    raise_error "$1 exists but is empty"
#  elif [ $( bcftools view -h $1 2>&1 | head | grep "No BGZF EOF marker" | wc -l ) -gt 0 ]; then # check that VCF is not truncated
#    raise_error "$1 may be truncated"
#  fi
#}

#make_tabix() {
#  if [ ! -f $1.tbi ]; then
#    local _start_time=${SECONDS}
#    bcftools index $1 \
#      --tbi \
#      --threads $(( ${NSLOTS}-1 ))
#    local _duration=$(( ${SECONDS}-${_start_time} ))
#    print_update "finished tabix of $1" "${_duration}"
#  fi
#}

set_up_conda() {
  local __conda_setup="$('/apps/eb/skylake/software/Anaconda3/2020.07/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
  if [ $? -eq 0 ]; then
      set +o nounset # use to avoid "PS1: unbound variable" error
      eval "$__conda_setup"
      set -o nounset
  else
      if [ -f "/apps/eb/skylake/software/Anaconda3/2020.07/etc/profile.d/conda.sh" ]; then
          . "/apps/eb/skylake/software/Anaconda3/2020.07/etc/profile.d/conda.sh"
      else
          export PATH="/apps/eb/skylake/software/Anaconda3/2020.07/bin:$PATH"
      fi
  fi
  unset __conda_setup
}

set_up_shapeit5 () {
  module purge
  module load HTSlib/1.14-GCC-11.2.0
  local _version="${1:-0}"
  local _dir="/well/lindgren/flassen/software/SHAPEIT5/v${_version}/shapeit5"
  local phase_rare="${_dir}/phase_rare/bin/SHAPEIT5_phase_rare"
  local phase_common="${_dir}/phase_common/bin/SHAPEIT5_phase_common"
  local switch="${_dir}/switch/bin/SHAPEIT5_switch"
  SHAPEIT_phase_rare="${phase_rare}"
  SHAPEIT_phase_common="${phase_common}"
  SHAPEIT_switch="${switch}"
  >&2 echo "Loading SHAPEIT5 v${_version}."
}

#get_hail_memory() {
#  if [[ -z ${QUEUE} || -z ${NSLOTS} ]]; then
#    raise_error "QUEUE and NSLOTS must both be defined"
#  fi
#  if [[ "${QUEUE}" = *".qe" || "${QUEUE}" = *".qc" ]]; then
#    local _mem_per_slot=10
#  elif [[ "${QUEUE}" = *".qf" ]]; then
#    local _mem_per_slot=3
#  else
#    raise_error "QUEUE must end in either \".qe\", \".qc\", or \".qf\""
#  fi
#  echo $(( ${_mem_per_slot}*${NSLOTS} ))
#}

#set_up_hail() {
#  mkdir -p ${spark_dir} # directory for Hail's temporary Spark output files
#  module load Anaconda3/2020.07
#  module load java/1.8.0_latest
#  set_up_conda
#  conda activate jupyter-hail # Requires conda environment with Hail installed
#  local _mem=$( get_hail_memory ) 
#  if [ ! -z ${_mem} ]; then
#    export PYSPARK_SUBMIT_ARGS="--conf spark.local.dir=${spark_dir} --conf spark.executor.heartbeatInterval=1000000 --conf spark.network.timeout=1000000  --driver-memory ${_mem}g --executor-memory ${_mem}g pyspark-shell"
#  fi
#}



wait_for_path() {
  local path=${1}
  local thedir=$( dirname $path)
  local base=$( basename $path)
  local max_ticks=100
  local cur_ticks=0
  if [ "$(ls -l ${thedir} | grep ${base} |  wc -l)" -eq "0" ]; then
    echo "Waiting for '${path}' (${max_ticks} ticks).."
    while [  "$(ls -l ${thedir} | grep ${base} |  wc -l)" -eq "0" ]; do
      local cur_ticks=$(( ${cur_ticks} + 1 ))
      if [[ ${cur_ticks} -ge ${max_ticks} ]]; then
            >&2 echo "max ticks reached. Breaking loop.."
            break
        fi
      sleep 3
    done
  fi
}


set_up_RSAIGE() {
  module load Anaconda3/2020.07 
  local version="${1:-1.1.9}"
  local envs=$( conda env list | grep $version )
  local env_dir=$(echo $envs | cut -d" " -f2 )
  local env_saige="saige-v${version}"
  local ld_paths="${env_dir}/lib/R/etc/ldpaths"
  if [ -d "${env_dir}" ]; then
    set +eu
    module load java/1.8.0_latest
    source "/apps/eb/skylake/software/Anaconda3/2020.07/etc/profile.d/conda.sh"
    wait_for_path ${ld_paths}
    if [ $( file_size ${ld_paths} ) -ge 1 ]; then
      echo "Loading ${env_saige} (env dir: ${env_dir})"
      conda activate ${env_saige}
    else
      raise_error "${ld_paths} has zero bytes! Expected ~1500 bytes."
    fi
    set -eu
  else
    raise_error "${env_saige} does not exist."
  fi
}


set_up_rpy() {
  set +eu
  module load Anaconda3/2020.07
  module load java/1.8.0_latest
  source "/apps/eb/skylake/software/Anaconda3/2020.07/etc/profile.d/conda.sh"
  conda activate rpy
  set -eu
}

set_up_phaser() {
  set +eu
  module load Anaconda3/2020.07
  module load java/1.8.0_latest
  source "/apps/eb/skylake/software/Anaconda3/2020.07/etc/profile.d/conda.sh"
  conda activate phaser
  set -eu
  readonly PHASER_PATH="/well/lindgren/flassen/software/phaser/phaser/phaser/phaser.py"
}

set_up_whatshap() {
  set +eu
  module load Anaconda3/2020.07
  module load java/1.8.0_latest
  source "/apps/eb/skylake/software/Anaconda3/2020.07/etc/profile.d/conda.sh"
  conda activate whatshap
  set -eu
}

set_up_smartphase() {
  set +eu
  module load Anaconda3/2020.07
  module load java/10.0.1
  source "/apps/eb/skylake/software/Anaconda3/2020.07/etc/profile.d/conda.sh"
  conda activate smartphase
  readonly SMARTPHASE_PATH="/well/lindgren/flassen/software/smart-phase/smartPhase.jar"
  set -eu
}




set_up_ldpred2() {
  set +eu
  module load Anaconda3/2020.07
  module load java/1.8.0_latest
  source "/apps/eb/skylake/software/Anaconda3/2020.07/etc/profile.d/conda.sh"
  #conda activate bigsnpr-v1.11.6
  conda activate bigsnpr-v1.12.1
  set -eu
}

set_up_tensorflow() {
  module load Anaconda3/2020.07
  module load java/1.8.0_latest
  source "/apps/eb/skylake/software/Anaconda3/2020.07/etc/profile.d/conda.sh"
  conda activate tf
}


#set_up_vep() {
#  module load EnsEMBLCoreAPI/96.0-r20190601-foss-2019a-Perl-5.28.1 # required for LOFTEE
#  module load VEP/95.0-foss-2018b-Perl-5.28.0 # required FOR VEP (NOTE: this steps throws some errors since the above module is already loaded. It works nonetheless.)
#  module load samtools/1.8-gcc5.4.0 # required for LOFTEE 
#  export PERL5LIB=$PERL5LIB:/well/lindgren/flassen/software/VEP/plugins_grch38/
#}


# path to bonferonni corrected phenotypes
get_prs_path() {
  echo "/well/lindgren-ukbb/projects/ukbb-11867/flassen/projects/KO/wes_ko_ukbb/data/prs/validation/ldsc_summary_bonf_sig_phenos.txt"
}

# get path to currently used phenotype headers and phenotypes
get_pheno_header_path() {
 echo "/well/lindgren-ukbb/projects/ukbb-11867/flassen/projects/KO/wes_ko_ukbb/data/phenotypes/dec22_phenotypes_binary_200k_header.tsv"
}

get_pheno_header_path_200k() {
  echo "/well/lindgren-ukbb/projects/ukbb-11867/flassen/projects/KO/wes_ko_ukbb/data/phenotypes/dec22_phenotypes_binary_200k.tsv.gz"
}

# check if we allow conditioning om prs
pheno_allow_cond_prs() {
  local phenotype="${1}"
  echo "$( cat $(get_prs_path) | grep -w "${phenotype}" | wc -l)"
}

# check if phenotype
validate_phenotype() {
  local phenotype="${1}"
  local ok="$( cat $(get_pheno_header_path) | grep -w "${phenotype}" | wc -l)"
  if [ "${ok}" -eq "0" ]; then
    raise_error "${phenotype} not pheno path!"
  fi
}





