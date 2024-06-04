#!/usr/bin/env bash
set -eo pipefail
module purge

# Author: Vishal Koparde, Ph.D.; Samantha Chill, Ph.D.
# CCBR, NCI
# (c) 2021
#
# wrapper script to run the snakemake pipeline
# a) on an interactive node (runlocal) OR
# b) submit to the slurm load scheduler (run)
#
# DISCLAIMER: This wrapper only works on BIOWULF

# setting python and snakemake versions
PYTHON_VERSION="python/3.7"
SNAKEMAKE_VERSION="snakemake/7.7.0"
# set extra singularity bindings
EXTRA_SINGULARITY_BINDS="/lscratch, /data/CCBR_Pipeliner/"
SCRIPTNAME="$0"
SCRIPTDIRNAME=$(readlink -f $(dirname $0))
SCRIPTBASENAME=$(readlink -f $(basename $0))

# set times
log_time=`date +"%Y%m%d_%H%M"`

function get_git_commitid_tag() {

  # This function gets the latest git commit id and tag
  # @Input:
  #   @param1: PIPELINE_HOME folder which is a initialized git repo folder path
  # @Output:
  #   @param1: tab-delimited commit id and tag

  cd $1
  gid=$(git rev-parse HEAD)
  tag=$(git describe --tags $gid 2>/dev/null)
  echo -ne "$gid\t$tag"

}
echo "#################################################################
#################################################################"
PIPELINE_HOME=$(readlink -f $(dirname "$0"))
echo "Pipeline Dir: $PIPELINE_HOME"
SNAKEFILE="${PIPELINE_HOME}/workflow/Snakefile"
echo "Snakefile: $SNAKEFILE"
GIT_COMMIT_TAG=$(get_git_commitid_tag $PIPELINE_HOME)
echo "Git Commit/Tag: $GIT_COMMIT_TAG"
echo "#################################################################
#################################################################"
function usage_only() {

  # This script prints only the usage without any pre-amble

  cat '''
  #################################################################
  #################################################################
  USAGE:
    bash ${SCRIPTNAME} -m/--runmode=<RUNMODE> -w/--workdir=<WORKDIR>
  Required Arguments:
  1.  RUNMODE: [Type: String] Valid options:
      *) init : initialize workdir
      *) run : run with slurm
      *) reset : DELETE workdir dir and re-init it
      *) dryrun : dry run snakemake to generate DAG
      *) unlock : unlock workdir if locked by snakemake
      *) runlocal : run without submitting to sbatch
  2.  WORKDIR: [Type: String]: Absolute or relative path to the 
              output folder with write permissions.
  #################################################################
  #################################################################
  '''

}

function usage() { 

  # This function prints generic usage of the wrapper script.
  # @Input: None
  # @Output: Usage information about the script

  cat '''
  #################################################################
  #################################################################
  Running ${SCRIPTBASENAME} ...
  <Your Pipeline Short Name>.
  echo "Git Commit/Tag: $GIT_COMMIT_TAG"
  '''

  usage_only

}

function err() { 

# This is a generic error message function. It prints the message, then the 
# usage and exits with non-zero exit code.
# @Input:
#     @param1: Message to print before printing the usage and exiting
# @Ouput:
#     @param2: echo the error message with the usage to the screen

cat <<< "
#################################################################
##### ERROR ############ ERROR ########## ERROR #################
#################################################################
  $@
" && usage_only && exit 1 1>&2; 
}


function _set_config() {

  sed -e "s/PIPELINE_HOME/${PIPELINE_HOME//\//\\/}/g" \
    -e "s/WORKDIR/${WORKDIR//\//\\/}/g" \
    ${PIPELINE_HOME}/config/config.yaml > $WORKDIR/config.yaml

}

function init() {

  # This function initializes the workdir by:
  # 1. creating the working dir
  # 2. copying essential files like config.yaml and samples.tsv into the workdir
  # 3. setting up logs and stats folders

  # create output folder if it doesn't exist
  if [ ! -d $WORKDIR ];then mkdir -p $WORKDIR; fi
  
  #create directories
  dir_list=(logs stats scripts resources manifests)
  for pd in "${dir_list[@]}"; do 
    if [[ ! -d $WORKDIR/$pd ]]; then 
      mkdir -p $WORKDIR/$pd
      echo "$pd Dir: $WORKDIR/$pd"
    fi
  done

  dir_list=(logs/dryrun)
  for pd in "${dir_list[@]}"; do 
    if [[ ! -d $WORKDIR/$pd ]]; then 
      mkdir -p $WORKDIR/$pd
      echo "$pd Dir: $WORKDIR/$pd"
    fi
  done

  # copy configs, samples manifests, scripts, resources
  _set_config
  cp ${PIPELINE_HOME}/config/samples.tsv $WORKDIR/manifests
  cp ${PIPELINE_HOME}/config/contrasts.tsv $WORKDIR/manifests
  cp ${PIPELINE_HOME}/workflow/scripts/* $WORKDIR/scripts
  cp ${PIPELINE_HOME}/resources/* $WORKDIR/resources

  echo "
  #################################################################
  #################################################################
  Done Initializing $WORKDIR. 
  You can now edit $WORKDIR/config.yaml and $WORKDIR/manifests/samples.tsv
  #################################################################
  #################################################################
  "

}

function check_essential_files() {

# Checks if files essential to start running the pipeline exist in the workdir
# By default config.yaml and samples.tsv are considered essential files.

  if [ ! -d $WORKDIR ];then err "Folder $WORKDIR does not exist!"; fi
  for f in config.yaml manifests/samples.tsv; do
    if [ ! -f $WORKDIR/$f ]; then 
      err "Error: '${f}' file not found in workdir ... initialize first!"
    fi
  done

}

function reconfig(){

# Rebuild config file and replace the config.yaml in the WORKDIR
# this is only for dev purposes when new key-value pairs are being 
# updated in the config file in PIPELINE_HOME

  check_essential_files
  _set_config
  echo "$WORKDIR/config.yaml has been updated!"

}

function runcheck(){

  # Check "job-essential" files and load required modules

  check_essential_files
  module load $PYTHON_VERSION
  module load $SNAKEMAKE_VERSION

}

function dryrun() {

  # check essential files, load modules and do Dry-run

  runcheck
  run "--dry-run"

}

function unlock() {

  # check essential files, load modules and 
  # unlock the workdir if previous snakemake run ended abruptly

  runcheck
  run "--unlock"

}

function _exe_in_path() {

  name_of_exe=$1
  path_to_exe=$(which $name_of_exe 2>/dev/null)
 if [ ! -x "$path_to_exe" ] ; then
    err $path_to_exe NOT FOUND!
 fi

}


function set_singularity_binds(){

# this functions tries find what folders to bind
# "Biowulf specific"
# assumes that config.yaml and samples.tsv in the WORKDIR are essential
# files with the most uptodate information
# required dos2unix in path
  _exe_in_path dos2unix
  echo "$PIPELINE_HOME" > ${WORKDIR}/tmp1
  echo "$WORKDIR" >> ${WORKDIR}/tmp1
  grep -o '\/.*' <(cat ${WORKDIR}/config.yaml ${WORKDIR}/manifests/samples.tsv)| \
    dos2unix | \
    tr '\t' '\n' | \
    grep -v ' \|\/\/' | \
    sort | \
    uniq >> ${WORKDIR}/tmp1
  grep gpfs ${WORKDIR}/tmp1|awk -F'/' -v OFS='/' '{print $1,$2,$3,$4,$5}' | \
    grep "[a-zA-Z0-9]" | \
    sort | uniq > ${WORKDIR}/tmp2
  grep -v gpfs ${WORKDIR}/tmp1|awk -F'/' -v OFS='/' '{print $1,$2,$3}' | \
    grep "[a-zA-Z0-9]" | \
    sort | uniq > ${WORKDIR}/tmp3
  while read a;do 
    readlink -f $a
  done < ${WORKDIR}/tmp3 | grep "[a-zA-Z0-9]"> ${WORKDIR}/tmp4
  binds=$(cat ${WORKDIR}/tmp2 ${WORKDIR}/tmp3 ${WORKDIR}/tmp4 | sort | uniq | tr '\n' ',')
  rm -f ${WORKDIR}/tmp?
  binds=$(echo $binds | awk '{print substr($1,1,length($1)-1)}')
  SINGULARITY_BINDS="-B $EXTRA_SINGULARITY_BINDS,$binds"

}

function printbinds(){

# set the singularity binds and print them
# singularity binds are /lscratch,/data/CCBR_Pipeliner,
# plus paths deduced from config.yaml and samples.tsv using 
# set_singularity binds function

  set_singularity_binds
  echo $SINGULARITY_BINDS

}

function runlocal() {

  # If the pipeline is fired up on an interactive node (with sinteractive), 
  # this function runs the pipeline

  runcheck
  set_singularity_binds
  if [ "$SLURM_JOB_ID" == "" ];then err "runlocal can only be done on an interactive node"; fi
  module load singularity
  run "local"

}

function runslurm() {

  # Submit the execution of the pipeline to the biowulf job scheduler (slurm)

  runcheck
  set_singularity_binds
  run "slurm"

}

function create_runinfo() {

  # Create a runinfo.yaml file in the WORKDIR
  runinfo=${WORKDIR}/logs/$log_time/runinfo.yaml.${log_time}

  echo "Pipeline Dir: $PIPELINE_HOME" > $runinfo
  echo "Git Commit/Tag: $GIT_COMMIT_TAG" >> $runinfo
  userlogin=$(whoami)
  username=$(finger $userlogin|grep ^Login|awk -F"Name: " '{print $2}')
  echo "Login: $userlogin" >>  $runinfo
  echo "Name: $username" >>  $runinfo
  g=$(groups)
  echo "Groups: $g" >> $runinfo
  d=$(date)
  echo "Date/Time: $d" >> $runinfo

}

function create_logtime(){
  # create time log
  dir_list=(logs/$log_time)
  for pd in "${dir_list[@]}"; do 
    if [[ ! -d $WORKDIR/$pd ]]; then 
      mkdir -p $WORKDIR/$pd
    fi
  done
}

function run() {
  # RUN function
  # argument1 can be:
  # 1. local or
  # 2. dryrun or
  # 3. unlock or
  # 4. slurm

  if [ "$1" == "local" ];then

  create_logtime
  create_runinfo
  
  snakemake -s $SNAKEFILE \
  --directory $WORKDIR \
  --printshellcmds \
  --use-singularity \
  --singularity-args "$SINGULARITY_BINDS" \
  --use-envmodules \
  --latency-wait 120 \
  --configfile ${WORKDIR}/config.yaml \
  --cores all \
  --stats ${WORKDIR}/logs/$log_time/snakemake.stats \
  2>&1|tee ${WORKDIR}/logs/$log_time/snakemake.log

  if [ "$?" -eq "0" ];then
    snakemake -s $SNAKEFILE \
    --report ${WORKDIR}/runlocal_snakemake_report.html \
    --directory $WORKDIR \
    --configfile ${WORKDIR}/config.yaml 
  fi

  elif [ "$1" == "slurm" ];then
    create_logtime
    create_runinfo
    
    # if QOS is other than "global" and is supplied in the cluster.json file then add " --qos={cluster.qos}" to the 
    # snakemake command below
    cat > ${WORKDIR}/logs/$log_time/submit_script.sbatch << EOF
#!/bin/bash
#SBATCH --job-name="MAPLE"
#SBATCH --mem=10g
#SBATCH --partition="norm"
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=$WORKDIR/logs/${log_time}/00_%j_%x.out \
#SBATCH --mail-type=BEGIN,END,FAIL

module load $PYTHON_VERSION
module load $SNAKEMAKE_VERSION
module load singularity

cd \$SLURM_SUBMIT_DIR

snakemake -s $SNAKEFILE \
--directory $WORKDIR \
--use-singularity \
--singularity-args "$SINGULARITY_BINDS" \
--use-envmodules \
--printshellcmds \
--latency-wait 120 \
--configfile ${WORKDIR}/config.yaml \
--cluster-config ${WORKDIR}/resources/cluster.yaml \
--cluster "sbatch --gres {cluster.gres} --cpus-per-task {cluster.threads} -p {cluster.partition} -t {cluster.time} --mem {cluster.mem} --job-name {cluster.name} --output $WORKDIR/logs/$log_time/{cluster.output} --error  $WORKDIR/logs/$log_time/{cluster.error}" \
-j 500 \
--rerun-incomplete \
--keep-going \
--stats ${WORKDIR}/logs/$log_time/snakemake.stats \
2>&1|tee ${WORKDIR}/logs/$log_time/snakemake.log

if [ "\$?" -eq "0" ];then
  snakemake -s $SNAKEFILE \
  --directory $WORKDIR \
  --report ${WORKDIR}/logs/$log_time/runslurm_snakemake_report.html \
  --configfile ${WORKDIR}/config.yaml 
fi

bash <(curl https://raw.githubusercontent.com/CCBR/Tools/master/Biowulf/gather_cluster_stats.sh 2>/dev/null) ${WORKDIR}/logs/$log_time/snakemake.log > ${WORKDIR}/logs/$log_time/snakemake.log.HPC_summary.txt

EOF

  sbatch ${WORKDIR}/logs/$log_time/submit_script.sbatch
  
  # for unlock and dryrun 
  else 
    snakemake $1 -r -s $SNAKEFILE \
    --directory $WORKDIR \
    --use-envmodules \
    --printshellcmds \
    --verbose \
    --latency-wait 120 \
    --configfile ${WORKDIR}/config.yaml \
    --cluster-config ${WORKDIR}/resources/cluster.yaml \
    --cluster "sbatch --gres {cluster.gres} --cpus-per-task {cluster.threads} -p {cluster.partition} -t {cluster.time} --mem {cluster.mem} --job-name {cluster.name} --output $WORKDIR/logs/$log_time/{cluster.output} --error  $WORKDIR/logs/$log_time/{cluster.error}" \
    -j 500 \
    --rerun-incomplete \
    --scheduler greedy  \
    --keep-going \
    --stats ${WORKDIR}/snakemake.stats | tee ${WORKDIR}/logs/dryrun/dryrun.${log_time}.log
  fi

}

function reset() {

  # Delete the workdir and re-initialize it

  echo "Working Dir: $WORKDIR"
  if [ ! -d $WORKDIR ];then err "Folder $WORKDIR does not exist!";fi
  echo "Deleting $WORKDIR"
  rm -rf $WORKDIR
  echo "Re-Initializing $WORKDIR"
  init

}


function main(){

  # Main function which parses all arguments

  if [ $# -eq 0 ]; then usage && exit 1; fi

  for i in "$@"; do
  case $i in
      -m=*|--runmode=*)
        RUNMODE="${i#*=}"
      ;;
      -w=*|--workdir=*)
        WORKDIR="${i#*=}"
      ;;
      *)
        err "Unknown argument!"    # unknown option
      ;;
  esac
  done
  WORKDIR=$(readlink -f "$WORKDIR")
  echo "Working Dir: $WORKDIR"

  case $RUNMODE in
    init) init && exit 0;;
    dryrun) dryrun && exit 0;;
    unlock) unlock && exit 0;;
    run) runslurm && exit 0;;
    runlocal) runlocal && exit 0;;
    reset) reset && exit 0;;
    dry) dryrun && exit 0;;                      # hidden option
    local) runlocal && exit 0;;                  # hidden option
    reconfig) reconfig && exit 0;;               # hidden option for debugging
    printbinds) printbinds && exit 0;;           # hidden option
    *) err "Unknown RUNMODE \"$RUNMODE\"";;
  esac

}

# call the main function

main "$@"
