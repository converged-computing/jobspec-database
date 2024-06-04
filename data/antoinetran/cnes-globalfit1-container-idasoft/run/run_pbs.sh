#!/bin/sh
#PBS -N lisa-trana-globalfit-idasoft
#PBS -l select=5:ncpus=40:mpiprocs=8:mem=100gb:os=rh7:generation=g2019
#PBS -l place=free:group=switch
#PBS -l walltime=00:05:00

# https://gitlab.cnes.fr/hpc/wikiHPC/-/wikis/pbs-exemple-job-mpi says that
# Il est nécessaire d'ajouter le paramètre -l place pour indiquer à PBS que l'on souhaite concaténer les processus sur le même switch infiniband.

#https://gitlab.cnes.fr/hpc/wikiHPC/-/wikis/pbs-ressources

scriptDir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

#test ! -f "${scriptDir}"/common.sh && echo "${scriptDir}/../common.sh not found." >&2 && exit 1
#. "${scriptDir}"/common.sh

logGeneric() {
  logLevel="$1"
  shift
  printf "$(date --iso-8601=ns -u) [${logLevel}] %s\n" "$*"
}

logInfo() {
  logGeneric INFO "$@"
}

logError() {
  logGeneric ERROR "$@" >&2
}

_prefixCmdDate() {
  while IFS= read -r line ; do
    printf "%s %s\n" "$(date --iso-8601=ns -u)" "${line}"
  done
}

prefixDate() {
  ( "$@" 2>&1 1>&3 3>&- | _prefixCmdDate ) 3>&1 1>&2 | _prefixCmdDate
}

runAndCheck() {
  "$@"
  exitCode="$?"
  test 0 != "${exitCode}" && logError Exit code: "${exitCode}" running: "$@" && exit "${exitCode}"
  return 0
}

# Nombre de threads OpenMP
export OMP_NUM_THREADS
# --no-burnin
globalfitExtraArgs="--debug"


loadModules() {
  module load singularity/3.10.0
  module load openmpi/3.1.4 #(version doit être la même que celle utilisée dans le conteneur pour compiler)
  module load monitoring/2.0
}

showHelp() {
  echo "-h|--help: show help"
  echo "The rest of arguments is provided by PBS run in pbsArgs(). Description of variables below."
  echo "globalFitMode: mode. Accept: full or simple"
  echo "runningMode: running mode. Accept: mpiGlobalFit, mpiHelloRingC, mpiHelloMpiTest"
  echo "singularityFile: singularity SIF file."
  echo "inputFile: for singularityMode=mpiGlobalFit only. Path of the input file. It must be in Sangria V2 format."
  echo "vgbFile: for runningMode=mpiGlobalFit,globalFitMode=full only. path of the auxiliary file VGB."
  echo "mbhDirectory: for singularityMode=mpiGlobalFit only. Path of the directory containing the auxiliary file MBH: search_sources.dat"
  echo "ucbDirectory: for singularityMode=mpiGlobalFit only. Path of the directory containing the auxiliary file ucb_frequency_spacing.dat."
  echo "steps: for singularityMode=mpiGlobalFit only. Default: 100000. Number of steps (excluding burnin phase) of MCMC."
  echo "chains: for singularityMode=mpiGlobalFit only. Must be a multiple of OpenMP threads."
}

parseArgs() {
  while test 0 != "$#" ; do
    case "$1" in
    -h|--help)
      showHelp
      shift
      ;;
    *)
      echo "Unknown argument $1" >&2
      exit 1
    esac
  done

  if test -z "${runningMode}" ; then
     logError "Missing runningMode"
     exit 11
  fi
}

pbsArgs() {
      pbsRss="${pbsRss}"
      globalFitMode="${globalFitMode}"
      runningMode="${runningMode}"
      singularityFile="${singularityFile}"
      inputFile="${inputFile}"
      #vgbFile="${vgbFile}"
      #mbhDirectory="${mbhDirectory}"
      #ucbDirectory="${ucbDirectory}"
      globalFitProfile="${globalFitProfile}"
      steps="${steps}"
      ompNumThreads="${ompNumThreads}"
      chains="${chains}"
      logInfo "pbsRss: ${pbsRss}"
      export OMP_NUM_THREADS="${ompNumThreads}"
      logInfo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
      logInfo "chains: ${chains}"
      logInfo "steps: ${steps}"
      logInfo "globalFitMode: ${globalFitMode}"
      logInfo "globalfitExtraArgs: ${globalfitExtraArgs}"
}

mpiRun() {
  if test -z "${PBS_NODEFILE}" ; then
    echo "[ERROR] missing variable PBS_NODEFILE. Did you launched with qsub?" >&2
    exit 11
  fi

  # nombre de processus MPI
  nb_procs=$(wc -l $PBS_NODEFILE | cut -d" " -f 1)
  echo "nb_procs = $nb_procs"
  echo PBS_NODEFILE
  cat $PBS_NODEFILE
  printAndRun mpirun -x OMP_NUM_THREADS -n "${nb_procs}" --hostfile "${PBS_NODEFILE}" --mca orte_base_help_aggregate 0 "$@"
  #printAndRun /work/SC/lisa/trana/dmtcp/bin/dmtcp_launch --rm -i 30 --no-gzip --ckptdir /work/SC/lisa/trana/dmtcpckpt/ --coord-logfile /work/SC/lisa/trana/dmtcp.log  mpirun --mca btl self,tcp -x OMP_NUM_THREADS -n "${nb_procs}" --hostfile "${PBS_NODEFILE}" --mca orte_base_help_aggregate 0 "$@"
  #cd /work/SC/lisa/trana/dmtcpckpt/
  #export DMTCP_COORD_HOST=$(hostname -f)
  #echo "DMTCP_COORD_HOST = $DMTCP_COORD_HOST"
  #printAndRun bash -x ./dmtcp_restart_script.sh --coord-host "${DMTCP_COORD_HOST}" --hostfile $PBS_NODEFILE --coord-logfile /work/SC/lisa/trana/dmtcp_restore.log --tmpdir "$TMPDIR"
}

helloworldMpiTest() {
  mpiRun singularity exec "${singularityFile}" /container/mpitest
}
helloworldRingC() {
  #mpiRun singularity exec "${singularityFile}" /container/ring_c
  mpiRun /work/SC/lisa/trana/cnes-globalfit1-container-idasoft/container/ring_c
}

printAndRun() {
  echo Running "$@"
  "$@"
}

setGlobalFitVars() {
  outputDir="${PBS_O_WORKDIR}"
  workDir="${PBS_O_WORKDIR}"
  if test "full" == "${globalFitMode}" ; then
    # In full mode, the ucb ucb_frequency_spacing.dat must be on working directory. Thus we set pwd.
    fullModeSingularityArg="--pwd /data/ucb --bind ${vgbFile}:/data/vgb --bind ${mbhDirectory}:/data/mbh --bind ${ucbDirectory}:/data/ucb"
    fullModeCmdArg="--known-sources /data/vgb --mbh-search-path /data/mbh"
    logInfo "Verification mode found."
  else
    fullModeSingularityArg=""
    fullModeCmdArg=""
    logInfo "globalFitMode ${globalFitMode} is not set to full mode."
  fi
  vgbFile=$PWD/run/auxiliaryfiles/${globalFitProfile}/ldc_sangria_vgb_list.dat,mbhDirectory=$PWD/run/auxiliaryfiles/${globalFitProfile}/,ucbDirectory=$PWD/run/auxiliaryfiles/${globalFitProfile}

  if test "nominal" == "${globalFitProfile}" ; then
    #### Full year
    #Tobs=31457280
    #padding=128
    ####
    #### 3 months
    Tobs=7864320
    padding=32
    ####
    sources=40
    Tstart=0
    fmin=0.0003
    samples=128
  elif test "profiling" == "${globalFitProfile}" ; then
    Tobs=7864320
    padding=32
    sources=40  
  fi
  test -n "${fmin}" && fminArg="--fmin ${fmin}"
  test -n "${Tstart}" && TstartArg="--start-time ${Tstart}"
  test -n "${samples}" && samplesArg="--samples ${samples}"

}

globalfit() {
  setGlobalFitVars

#export LDASOFT_PREFIX=${LDASOFT_PREFIX:-/work/SC/lisa/trana/lib/ldasoft}
#export MBH_HOME=${MBH_HOME:-/work/SC/lisa/trana/lib/mbh}
#export MPI_DIR=${MPI_DIR:-/work/SC/lisa/trana/lib/omp}
#export GSL_ROOT_DIR=${GSL_ROOT_DIR:-/work/SC/lisa/trana/lib/gsl}
#PATH=$PATH:$MPI_DIR/bin
#export LD_LIBRARY_PATH="$GSL_ROOT_DIR"/lib
# mpiRun /work/SC/lisa/trana/lib/ldasoft/bin/global_fit \
#     --rundir "${outputDir}" \
#     --h5-data "${inputFile}" --sangria \
#     --chains "${chains}" \
#     ${fminArg} \
#     ${TstartArg} \
#     --duration "${Tobs}" \
#     ${samplesArg} \
#     --padding "${padding}" \
#     --sources "${sources}" \
#     ${fullModeCmdArg} \
#     "$@"
# return


  mpiRun \
    singularity exec \
    --workdir "${workDir}" --bind "${workDir}":"${workDir}" --bind "${inputFile}":/data/input ${fullModeSingularityArg} \
    "${singularityFile}" /usr/local/lib/ldasoft/bin/global_fit \
      --rundir "${outputDir}" \
      --h5-data "/data/input" --sangria \
      --chains "${chains}" \
      ${fminArg} \
      ${TstartArg} \
      --duration "${Tobs}" \
      ${samplesArg} \
      --padding "${padding}" \
      --sources "${sources}" \
      ${fullModeCmdArg} \
      "$@"
}

# Goal: start and stop monitoring, and return the exit code only after stopping the monitoring.
# Takes the monitoring name as first argument, then the commands arguments.
monitoring() {
  monitoringName="$1"
  shift
  logInfo "Starting monitoring..."
  start_monitoring.sh --name "${monitoringName}"
  "$@"
  exitCode="$?"
  logInfo "Execution ended with exit code ${exitCode}. Stopping monitoring..."
  stop_monitoring.sh --name "${monitoringName}"
  return "${exitCode}"
}

main() {
  # If launched with qsub, this gets the variable with -v.
  pbsArgs
  if test -z "${runningMode}" ; then
    # Case without qsub.
    parseArgs "$@"
  fi
  loadModules
  if test mpiGlobalFit == "${runningMode}" ; then
    prefixDate runAndCheck monitoring job_"${PBS_JOBNAME}_${PBS_JOBID}" globalfit ${globalfitExtraArgs}  --steps "${steps}"
  elif test mpiHelloRingC == "${runningMode}" ; then
    prefixDate runAndCheck helloworldRingC
  elif test mpiHelloMpiTest == "${runningMode}" ; then
    prefixDate runAndCheck helloworldMpiTest
  fi
}




logInfo "Beginning computation $0 with steps ${steps}" "$@"

main "$@"

logInfo "End computation $0" "$@"


