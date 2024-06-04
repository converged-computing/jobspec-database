#PBS -N "region-type"
#PBS -A nn2993k
#PBS -l mppwidth=1,walltime=00:20:00
#PBS -l mppmem=1000mb
#PBS -o region.out
#PBS -e region.err
#        
        

module load coreutils-cnl
module load matlab/2007b
# Make sure I am in the correct directory
cd $HOME/Realtime/jobdir
# This seems to be the safest way to run Matlab when no display is set.
# It starts a "fake" X window session in display number 9
XVFB_DISPLAY=$RANDOM
/usr/bin/Xvfb :9 -screen 0 1024x768x24 &

# The bulk of this routine is done by matlab script ice_type_maps.
# It is run inside the virtual frame buffer.
env DISPLAY=:9.0 matlab -nodesktop -nosplash -nojvm <<EOF
warning off MATLAB:FINITE:obsoleteFunction;
$proj_eval;
ice_type_maps;
exit;
EOF


pid=$(ps -C Xvfb -o pid --no-headers | tail -n1)
[ "$pid" == "" ] && alert_message "$ERROR" "Error when stopping Xvfb"
kill $pid


###!/bin/bash
###
### Give the job a name (optional)
###PBS -N "type-region-matlab"
###
### Specify the project the job should be accounted on (obligatory)
###PBS -A nn2993k  
###
### The job needs at most 60 hours wall-clock time on 1 CPU (obligatory)
###PBS -l mppwidth=XXX,mppnppn=1,mppmem=4000mb
##
###
### The job needs at most 1000mb of memory (obligatory)
###
###PBS -l mppmem=900mb
###
### Write the standard output of the job to file 'seqjob.out' (optional)
###PBS -o region_seqjob.out
###
### Write the standard error of the job to file 'seqjob.err' (optional)
###PBS -e region_seqjob.err
###
### Make sure I am in the correct directory
##cd ~/Realtime/rightdir
##
##
##matlab -nojvm -nosplash  -nodesktop -r "ice_type_maps"  
