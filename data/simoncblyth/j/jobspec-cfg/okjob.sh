#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --account=junogpu
#SBATCH --job-name=okjob
#SBATCH --ntasks=1
#SBATCH --output=/hpcfs/juno/junogpu/%u/okjob/%j.out
#SBATCH --error=/hpcfs/juno/junogpu/%u/okjob/%j.err
#SBATCH --mem-per-cpu=20480
#SBATCH --gres=gpu:v100:1

okjob-notes(){ cat << EON
okjob-notes
=============

Notice the lack of "bash -l" above : to retain sanity job scripts 
need to be somewhat isolated from the invoking environment. 

EON
}

okjob-paths()
{
    local vars="BASH_SOURCE FUNCNAME CMAKE_PREFIX_PATH PKG_CONFIG_PATH PATH LD_LIBRARY_PATH DYLD_LIBRARY_PATH PYTHONPATH MANPATH CPATH";
    local var;
    for var in $vars;
    do
        echo $var;
        echo ${!var} | tr ":" "\n";
        echo;
    done
}

okjob-setup-junosw-opticks-pre-release()
{
    source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120_opticks/Pre-Release/J23.1.0-rc6/setup.sh
}
okjob-setup-junosw-opticks-current()
{
    source $HOME/junotop/bashrc.sh
}


okjob-setup()
{
    local home=/hpcfs/juno/junogpu/$USER
    [ -d "$home" ] && export HOME=$home   
    export TMP=$HOME/tmp   ## override default /tmp/$USER/opticks as /tmp is blackhole (not same filesystem on GPU cluster and gateway)  
    mkdir -p $TMP          ## whether override or not, need to create 
    mkdir -p $HOME/okjob

    #okjob-setup-junosw-opticks-pre-release
    okjob-setup-junosw-opticks-current
}

okjob-head(){ 
   hostname 
   nvidia-smi   
   local vars="0 HOSTNAME USER BASH_SOURCE HOME OPTICKS_PREFIX CUDA_VISIBLE_DEVICES"
   for var in $vars ; do printf "%25s : %s \n" "$var" "${!var}" ; done 
   date 
}

okjob-ctest()
{
   : okjob.sh : caution this runs ctests installed by opticks-full which 
   : can become outdated vs actual tests during development 
   cd $OPTICKS_PREFIX/tests
   pwd
   echo === $BASH_SOURCE $FUNCNAME $PWD
   which ctest 
   ctest -N 
   ctest --output-on-failure
}

okjob-body()
{
   local msg="=== $FUNCNAME:"
   echo $msg TMP $TMP

   #okjob-ctest
   #opticks-t

   jok-tds
}

okjob-tail(){
   local rc=$?    # capture the return code of prior command
   echo $FUNCNAME : rc $rc              
   date  

   #okjob-check-tmp   
}

okjob-main(){
   okjob-setup
   okjob-head
   okjob-body
   okjob-tail
}


okjob-check-tmp()
{
   : /tmp on gpu cluster is a black hole 
   : instead of using /tmp write to TMP $TMP

   echo $FUNCNAME : /tmp files belonging to $USER
   [ -d /tmp/$USER ] && echo "find /tmp/$USER -type f" && find /tmp/$USER -type f 
}



source $HOME/j/jok.bash 

defarg="main"
arg=${1:-$defarg}

case $arg in 
   main) okjob-main ;; 
  ctest) okjob-ctest ;; 
  setup) okjob-setup ;; 
   head) okjob-head ;; 
   init) jok-init ;; 
  info)  jok-info ;;
   grab) jok-grab ;;
   scpmeta) jok-scpmeta ;;
   ana)  jok-ana ;;
report)  jok-report ;;
     *)  echo $BASH_SOURCE : arg $arg unhandled ;;
esac


