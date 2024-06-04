#!/bin/bash
#SBATCH -N 5 
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=8

runlist=$1
#module load apps/singularity/3.6.1
#module load compiler/devtoolset/7.3.1
#module load compiler/rocm/2.9
#module load mpi/hpcx/2.4.1/gcc-7.3.1

mkfifo testfifo
exec 1000<>testfifo
rm -fr testfifo
rm p02.log

for ((n=1;n<=5;n++))
do
    echo >&1000
done

start=`date "+%s"`


cat $runlist | while read line
do
    read -u1000
    {
        #echo `date` "Now runing with $line">>p02.log
	srun -N1 -n1 -c8  -o ${line}.out -e ${line}.error sh $line
	#echo `date` "Now Job $line is done">>Job.log
        echo >&1000
    }&
done
sleep 30m
sh p03.combin.sh
 
wait

end=`date "+%s"`

echo "TIME: `expr $end - $start `"

exec 1000>&-
exec 1000<&-

