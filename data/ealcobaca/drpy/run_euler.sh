#PBS -N drpy  
#PBS -l select=1:ncpus=20
#PBS -l walltime=336:00:00 
#PBS -m abe
#PBS -M e.alcobaca@gmail.com

module load python/3.4.3 

cd /home/alcobaca/Studies/drpy-virtualenv/drpy
source ../bin/activate
make run
deactivate
