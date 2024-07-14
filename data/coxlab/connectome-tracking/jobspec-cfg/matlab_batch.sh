#!/bin/sh

matlab_exe=matlab
dir=~/Documents/Research/connectome-tracking

cmd="${1}("
n=0
for arg in "$@" ; do
    n=$(($n + 1))
    if test $n -gt 1 ; then
        cmd="${cmd}'"$arg"',"
    fi
done

cmd="${cmd%?}"  # remove last comma
cmd="${cmd})"

echo "cd ${dir}; ${cmd}; exit;" > ${dir}/matlab_cmd_${1}.m
cat ${dir}/matlab_cmd_${1}.m
${matlab_exe} -nojvm -nodisplay -nosplash -nodesktop < ${dir}/matlab_cmd_${1}.m > matlab_cmd_${1}.out
rm ${dir}/matlab_cmd_${1}.m
