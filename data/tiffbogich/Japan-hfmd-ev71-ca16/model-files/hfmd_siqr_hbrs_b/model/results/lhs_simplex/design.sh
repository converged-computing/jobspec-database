#!/bin/bash


#PBS -l nodes=1:ppn=1,walltime=23:30:00







#PBS -t 0-499
h=$PBS_ARRAYID





path_root=/scratch/network/sballest/plom-data/hfmd/hfmd_siqr_hbrs_b/model
##all the following path are relative to path_root
path_theta=results/lhs_simplex/theta.json
path_results=results/lhs_simplex/ #(has to end with a /)
path_design=results/lhs_simplex/design.csv

cd $path_root
test -d $path_results || mkdir -p $path_results




##############################
## pipeline step 0 (Get the initial conditions (no forcing). Note that even if we turn off the forcing, there could be an Hopf bifucation => we compute the average rep on 1000 week)
##############################
path_read=${path_results}




    path_saved=${path_results}






plom pipe $path_theta -D $path_design -t $h -I -S e:all:guess:0.0 | ./simul ode -T 100000 -D 1000 --traj --freq W --quiet -p $path_saved -i $h


##############################
## pipeline step 1 (First simplex + rescale reporting rate)
##############################
path_read=${path_results}




    path_saved=${path_results}






plom pipe $path_theta -D $path_design -t $h -X ${path_read}hat_${h}.csv -r rep | ./simplex -M 10000 --no_trace --prior --quiet -p $path_saved -i $h


##############################
## pipeline step 2 (We chain simplex)
##############################
path_read=${path_results}


for (( i=0; i < 19; i++ )); do




    
    path_saved=${path_results}
    if [ $i -eq 18 ]; then
        ##last results of the pipeline are written in in path_results
        path_saved=$path_results
    fi
    





    if [ $i -gt 0 ]; then
        let im1=$i-1
##overwrite path_read within the repeatition loop
        path_read=${path_results}
    fi


plom pipe $path_theta -T ${path_read}trace_${h}.csv -u 0.01 | ./simplex -M 10000 --no_trace --prior --quiet -p $path_saved -i $h


done



