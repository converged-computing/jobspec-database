#!/usr/bin/env bash
# Author: Vishal Koparde, Ph.D.
# CCBR, NCI
# (c) 2021
#
# wrapper script to run the TOBIAS snakemake workflow
# run tobias
# https://github.com/loosolab/tobias/
# ## clone the pipeline to a folder
# ## git clone https://github.com/loosolab/TOBIAS.git

PYTHON_VERSION="python/3.10"
SNAKEMAKE_VERSION="snakemake/7.32.3"
SINGULARITY_VERSION="singularity"

SCRIPTNAME="$0"
SCRIPTBASENAME=$(readlink -f $(basename $0))

# set extra singularity bindings comma separated
EXTRA_SINGULARITY_BINDS="/lscratch"

# essential files
# these are relative to the workflows' base folder
# these are copied into the WORKDIR
ESSENTIAL_FILES="config/config.yaml config/uropa_base_config.yaml config/cluster.json config/tools.yaml"
ESSENTIAL_FOLDERS="scripts"

set -eo pipefail
module purge

function get_git_commitid_tag() {
  cd $1
  gid=$(git rev-parse HEAD)
  tag=$(git describe --tags $gid 2>/dev/null)
  echo -ne "$gid\t$tag"
}

# ## setting PIPELINE_HOME
PIPELINE_HOME=$(readlink -f $(dirname "$0"))
echo "Pipeline Dir: $PIPELINE_HOME"
# set snakefile
SNAKEFILE="${PIPELINE_HOME}/tobias.snakefile"

# get github commit tag
GIT_COMMIT_TAG=$(get_git_commitid_tag $PIPELINE_HOME)
echo "Git Commit/Tag: $GIT_COMMIT_TAG"


############################################################################
# FUNCTIONS
############################################################################


function usage() { cat << EOF
${SCRIPTNAME}: run CCBR TOBIAS workflow for ATAC seq data
USAGE:
  bash ${SCRIPTNAME} -m/--runmode=<MODE> -w/--workdir=<path_to_workdir>
Required Arguments:
1.  RUNMODE: [Type: String] Valid options:
    *) init : initialize workdir
    *) run : run with slurm
    *) reset : DELETE workdir dir and re-init it
    *) dryrun : dry run snakemake to generate DAG
    *) unlock : unlock workdir if locked by snakemake
    *) runlocal : run without submitting to sbatch
2.  WORKDIR: [Type: String]: Absolute or relative path to the output folder with write permissions.
EOF
}


############################################################################


function err() { cat <<< "
#
#
#
  $@
#
#
#
" && usage && exit 1 1>&2; }


############################################################################


function init() {

# This function initializes the workdir by:
# 1. creating the working dir
# 2. copying essential files like config.yaml and samples.tsv into the workdir
# 3. setting up logs and stats folders

# create output folder
if [ -d $WORKDIR ];then err "Folder $WORKDIR already exists!"; fi
mkdir -p $WORKDIR

# copy essential files
for f in $ESSENTIAL_FILES;do
echo "Copying essential file: $f"
fbn=$(basename $f)
sed -e "s/PIPELINE_HOME/${PIPELINE_HOME//\//\\/}/g" -e "s/WORKDIR/${WORKDIR//\//\\/}/g" ${PIPELINE_HOME}/$f > $WORKDIR/$fbn
done

# copy essential folders
for f in $ESSENTIAL_FOLDERS;do
  rsync -az --progress ${PIPELINE_HOME}/$f $WORKDIR/
done

#create log and stats folders
if [ ! -d $WORKDIR/logs ]; then mkdir -p $WORKDIR/logs;echo "Logs Dir: $WORKDIR/logs";fi
if [ ! -d $WORKDIR/stats ];then mkdir -p $WORKDIR/stats;echo "Stats Dir: $WORKDIR/stats";fi

echo "Done Initializing $WORKDIR. You can now edit $WORKDIR/config.yaml"

}


############################################################################


function runcheck(){
# Check "job-essential" files and load required modules

  check_essential_files
  module load $PYTHON_VERSION
  module load $SNAKEMAKE_VERSION

}


############################################################################


function dryrun() {
  runcheck
  if [ -f ${WORKDIR}/dryrun.log ]; then
    modtime=$(stat ${WORKDIR}/dryrun.log |grep Modify|awk '{print $2,$3}'|awk -F"." '{print $1}'|sed "s/ //g"|sed "s/-//g"|sed "s/://g")
    mv ${WORKDIR}/dryrun.log ${WORKDIR}/logs/dryrun.${modtime}.log
    if [ -f ${WORKDIR}/dryrun_git_commit.txt ];then
      mv ${WORKDIR}/dryrun_git_commit.txt ${WORKDIR}/logs/dryrun_git_commit.${modtime}.txt
    fi
  fi
  run "--dry-run" > ${WORKDIR}/dryrun.log && \
  cat ${WORKDIR}/dryrun.log && \
  run "--dry-run"
}


############################################################################


function unlock() {
  runcheck
  run "--unlock"  
}


############################################################################


function runlocal() {
  runcheck
  set_singularity_binds
  if [ "$SLURM_JOB_ID" == "" ];then err "runlocal can only be done on an interactive node"; exit 1; fi
  module load $SINGULARITY_VERSION
  run "--dry-run" && \
  echo "Dry-run was successful .... starting local execution" && \
  echo "Git Commit/Tag: $GIT_COMMIT_TAG" > ${WORKDIR}/run_git_commit.txt && \
  run "local"
}


############################################################################


function runslurm() {
  runcheck
  set_singularity_binds
#define cluster resource json file
# CLUSTERJSON=$(python <(curl -s https://raw.githubusercontent.com/CCBR/Tools/master/scripts/extract_value_from_yaml.py 2>/dev/null) -y ${WORKDIR}/config.yaml -k clusterjson)
# if "clusterjson" key is absent from config.yaml then it defaults to the cluster.json in WORKDIR... to make sure this happens comment the above line and use the following
  CLUSTERJSON=$(run "--dry-run"|grep "cluster.json"|awk '{print $NF}')
  run "--dry-run" && \
  echo "Dry-run was successful .... submitting to job-scheduler" && \
  echo "Git Commit/Tag: $GIT_COMMIT_TAG" > ${WORKDIR}/run_git_commit.txt && \
  run "slurm"
}


############################################################################


function preruncleanup() {
  # Cleanup function to rename/move files related to older runs to prevent overwriting them.
  echo "Running..."
  # check initialization
  check_essential_files 
  cd $WORKDIR

  ## Archive previous run files
  if [ -f ${WORKDIR}/snakemake.log ];then 
    modtime=$(stat ${WORKDIR}/snakemake.log |grep Modify|awk '{print $2,$3}'|awk -F"." '{print $1}'|sed "s/ //g"|sed "s/-//g"|sed "s/://g")
    mv ${WORKDIR}/snakemake.log ${WORKDIR}/logs/snakemake.${modtime}.log
    if [ -f ${WORKDIR}/snakemake.log.HPC_summary.txt ];then 
      mv ${WORKDIR}/snakemake.log.HPC_summary.txt ${WORKDIR}/stats/snakemake.${modtime}.log.HPC_summary.txt
    fi
    if [ -f ${WORKDIR}/snakemake.stats ];then 
      mv ${WORKDIR}/snakemake.stats ${WORKDIR}/stats/snakemake.${modtime}.stats
    fi
    if [ -f ${WORKDIR}/run_git_commit.txt ];then
      mv ${WORKDIR}/run_git_commit.txt ${WORKDIR}/logs/run_git_commit.${modtime}.txt
    fi
  fi
  nslurmouts=$(find ${WORKDIR} -maxdepth 1 -name "slurm-*.out" |wc -l)
  if [ "$nslurmouts" != "0" ];then
    for f in $(ls ${WORKDIR}/slurm-*.out);do gzip -n $f;mv ${f}.gz ${WORKDIR}/logs/;done
  fi

}


############################################################################


function run() {
# RUN function
# argument1 can be:
# 1. local or
# 2. dryrun or
# 3. unlock or
# 4. slurm

  if [ "$1" == "local" ];then

  preruncleanup

  snakemake -s $SNAKEFILE \
  --directory $WORKDIR \
  --printshellcmds \
  --use-singularity \
  --singularity-args "$SINGULARITY_BINDS" \
  --use-envmodules \
  --latency-wait 120 \
  --rerun-incomplete \
  --configfile ${WORKDIR}/config.yaml \
  --cores all \
  --stats ${WORKDIR}/snakemake.stats \
  2>&1|tee ${WORKDIR}/snakemake.log

  if [ "$?" -eq "0" ];then
    snakemake -s $SNAKEFILE \
    --report ${WORKDIR}/runlocal_snakemake_report.html \
    --directory $WORKDIR \
    --configfile ${WORKDIR}/config.yaml 
  fi

  elif [ "$1" == "slurm" ];then
  
  preruncleanup

#define partitions
BUYINPARTITIONS=$(bash <(curl -s https://raw.githubusercontent.com/CCBR/Tools/master/Biowulf/get_buyin_partition_list.bash 2>/dev/null))
PARTITIONS="norm"
if [ $BUYINPARTITIONS ];then PARTITIONS="norm,$BUYINPARTITIONS";fi

  cat > ${WORKDIR}/submit_script.sbatch << EOF
#!/bin/bash
#SBATCH --job-name="tobias"
#SBATCH --mem=10g
#SBATCH --partition="$PARTITIONS"
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=2

module load $PYTHON_VERSION
module load $SNAKEMAKE_VERSION
module load $SINGULARITY_VERSION

cd \$SLURM_SUBMIT_DIR

snakemake -s $SNAKEFILE \
--directory $WORKDIR \
--use-singularity \
--singularity-args "$SINGULARITY_BINDS" \
--use-envmodules \
--printshellcmds \
--latency-wait 120 \
--configfile ${WORKDIR}/config.yaml \
--cluster-config $CLUSTERJSON \
--cluster "sbatch --gres {cluster.gres} --cpus-per-task {cluster.threads} -p {cluster.partition} -t {cluster.time} --mem {cluster.mem} --job-name {cluster.name} --output {cluster.output} --error {cluster.error} --qos={cluster.qos}" \
-j 500 \
--rerun-incomplete \
--keep-going \
--stats ${WORKDIR}/snakemake.stats \
2>&1|tee ${WORKDIR}/snakemake.log

if [ "\$?" -eq "0" ];then
  snakemake -s $SNAKEFILE \
  --directory $WORKDIR \
  --report ${WORKDIR}/runslurm_snakemake_report.html \
  --configfile ${WORKDIR}/config.yaml 
fi

bash <(curl https://raw.githubusercontent.com/CCBR/Tools/master/Biowulf/gather_cluster_stats_biowulf.sh 2>/dev/null) ${WORKDIR}/snakemake.log > ${WORKDIR}/snakemake.log.HPC_summary.txt

EOF

  sbatch ${WORKDIR}/submit_script.sbatch

  else

# dryrun and unlock

snakemake $1 -s $SNAKEFILE \
--directory $WORKDIR \
--printshellcmds \
--rerun-incomplete \
-j 500 \
--configfile ${WORKDIR}/config.yaml

  fi

}


############################################################################


function reset() {
# Delete the workdir and re-initialize it
  echo "Working Dir: $WORKDIR"
  if [ ! -d $WORKDIR ];then err "Folder $WORKDIR does not exist!";fi
  echo "Deleting $WORKDIR"
  rm -rf $WORKDIR
  echo "Re-Initializing $WORKDIR"
  init
}


############################################################################


function check_essential_files() {

# Checks if files essential to start running the pipeline exist in the workdir

  if [ ! -d $WORKDIR ];then err "Folder $WORKDIR does not exist!"; fi
  for f in $ESSENTIAL_FILES; do
    fbn=$(basename $f)
    if [ ! -f $WORKDIR/$fbn ]; then err "Error: '${fbn}' file not found in workdir ... initialize first!";fi
  done

}


############################################################################


function reconfig(){
# Rebuild config file and replace the config.yaml in the WORKDIR
# this is only for dev purposes when new key-value pairs are being added to the config file

  check_essential_files
  sed -e "s/PIPELINE_HOME/${PIPELINE_HOME//\//\\/}/g" -e "s/WORKDIR/${WORKDIR//\//\\/}/g" ${PIPELINE_HOME}/config/config.yaml > $WORKDIR/config.yaml
  echo "$WORKDIR/config.yaml has been updated!"

}


############################################################################


function set_singularity_binds(){
# this functions tries find what folders to bind
# biowulf specific
  echo "$PIPELINE_HOME" > ${WORKDIR}/tmp1
  echo "$WORKDIR" >> ${WORKDIR}/tmp1
  grep -o '\/.*' <(cat ${WORKDIR}/config.yaml)|tr '\t' '\n'|grep -v ' \|\/\/'|sort|uniq >> ${WORKDIR}/tmp1
  grep gpfs ${WORKDIR}/tmp1|awk -F'/' -v OFS='/' '{print $1,$2,$3,$4,$5}' |sort|uniq > ${WORKDIR}/tmp2
  grep -v gpfs ${WORKDIR}/tmp1|awk -F'/' -v OFS='/' '{print $1,$2,$3}'|sort|uniq > ${WORKDIR}/tmp3
  while read a;do readlink -f $a;done < ${WORKDIR}/tmp3 > ${WORKDIR}/tmp4
  binds=$(cat ${WORKDIR}/tmp2 ${WORKDIR}/tmp3 ${WORKDIR}/tmp4|sort|uniq |tr '\n' ',')
  rm -f ${WORKDIR}/tmp?
  binds=$(echo $binds|awk '{print substr($1,1,length($1)-1)}')
  SINGULARITY_BINDS="-B $EXTRA_SINGULARITY_BINDS,$binds"
}


############################################################################


function printbinds(){
  set_singularity_binds
  echo $SINGULARITY_BINDS
}


############################################################################


function main(){

  if [ $# -eq 0 ]; then usage; exit 1; fi


  for i in "$@"
  do
  case $i in
      -m=*|--runmode=*)
        RUNMODE="${i#*=}"
      ;;
      -w=*|--workdir=*)
        WORKDIR="${i#*=}"
      ;;
      -h|--help)
        usage && exit 0;;
      *)
        echo "\"$i\"" && err "Unknown argument!"    # unknown option
      ;;
  esac
  done
  WORKDIR=$(readlink -f "$WORKDIR")
  echo "Working Dir: $WORKDIR"

  case $RUNMODE in
    init) init && exit 0;;
    dag) dag && exit 0;;
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


############################################################################


main "$@"




