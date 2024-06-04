#!/bin/bash
# Example submission script for submitting 40 independent ABC simulations for
# each scratch assay dataset
#
# Author: David J. Warne (david.warne@qut.edu.au)
#         School of Mathematical Sciences
#         Queensland University of Technology
#

# note: after all simulations are complete data can be collated into a 
# a single file with the command e.g.,
# cat FKPP_pps_12_part*/sim.chkpnt > FKPP_pps_12.csv

# thesholding should then be preformed with e.g.,
# matlab -r "ABC_percentile('./FKPP_pps_12',2,0.001);quit;"

for S in 12 16 20
do
    for (( n=0; n<=40; n++))
    do
        cat > sub <<EOF
#!/bin/bash -l
#PBS -N FKPP$S
#PBS -l walltime=24:00:00
#PBS -l mem=1GB
#PBS -l ncpus=1

module load intel/2016b
cd \$PBS_O_WORKDIR
D=FKPP_pps_"$S"_part"$n"
mkdir \$D
cd \$D
../ABCREJ_FKPP ../../../../data/fbs"$S"000.csv 20000000 0 0 1e-5 5000 1 7e-3 --SSALOPT --seed $n > post_fbs"$S".csv
cd ../
EOF
    qsub sub
    rm sub
    done
done
