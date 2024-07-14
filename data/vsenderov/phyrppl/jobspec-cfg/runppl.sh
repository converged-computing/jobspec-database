#!/bin/bash
#
# Usage
#
#  runppl TEMPLATE-FILE ... NPARTS NSWEEPS NOMPTHREADS CUDAPARTPERTHREAD MODELNAME
#                            -5       -4      -3            -2               -1
# Some examples
#
#  ./runppl.sh CombineDS Anatinae 0.8709677419354839 1.0 1.0 1.0 0.5 1.0 1.0 0 1.0 3.0 0.1 0 1.0 3.0 0.1 false 2 0 1 10 50000 10 28 1 anadsGBM2
#


args=("$@")
for ((i=0; i < $#; i++))
do
    dir+="${args[$i]}"_
done

cp models/$1.tmpl results/$dir.cu

for ((i=0; i < $#; i++))
do
    if [ $i -gt 0 ]
    then 
	sed -i "s/__($i)__/${args[$i]}/g" results/$dir.cu
    fi
done

#rootppl clean
rppl -o $dir.out results/$dir.cu $RPPL_FLAGS
#                        NSWEEPS
for i in `seq "${args[$(($#-4))]}"`; do
#                                NPARTS               NOMPTHREADS         CUDAPARTPTHR
    \time -v ./$dir.out  "${args[$(($#-5))]}" 1 "${args[$(($#-3))]}" "${args[$(($#-2))]}";
done 1> logz.txt 2>> results/time.txt
mkdir $dir
mkdir $dir/plots
mkdir $dir/data
#                        MODELNAME
cp tools/treesim-"${args[$(($#-1))]}".js $dir/"${args[$(($#-1))]}".js
echo "${args[$(($#-1))]}" > $dir/model.txt
cp tools/pdf.py $dir/
cp tools/sample.py $dir/
cp tools/figtree.py $dir/
cp json-data/${args[1]}.tre.phyjson $dir/
mv logz.txt $dir/
mv results/* $dir/
cp $dir/logz.txt $dir/logz.uncleaned
(cd $dir/ ; sed -i '/Weight Sum is NaN, terminating.../d' logz.txt)
(cd $dir/ ; python3 pdf.py > estimates.csv)
(cd $dir/ ; python3 figtree.py "${args[0]}_header.csv" "${args[0]}.csv" "factors_end.csv" "logz.txt" "${args[1]}.tre.phyjson" "output.nex")
#mv log_norm_const.txt $dir/

rm $dir.out
