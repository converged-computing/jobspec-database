#PBS -q cs6210
#PBS -l nodes=8:sixcore
#PBS -l walltime=00:15:00
#PBS -N Zizheng.mp_mpi
cd 6210_barrier/
for i in 2, 4, 6, 8;
do
  for j in 2, 4, 6, 8, 10, 12;
  do
    >&2 echo "processes "$i" threads "$j
    python run.py openmp_mpi $i $j
  done
done

