#PBS -N array
#PBS -l select=13:ncpus=8:mem=31gb:interconnect=1g,walltime=200:00:00


#The MEGAlib environment first needs to be sourced:
cd $tmp/zfs/astrohe/Software
source COSIMain_u2.sh

#Change to home directory and run job
module add gnu-parallel
module load anaconda3/2022.05-gcc/9.5.0
cd $PBS_O_WORKDIR
cat $PBS_NODEFILE > nodes.txt
seq 1058 1158 | parallel --delay=3 --sshloginfile nodes.txt -j8 "source $tmp/zfs/astrohe/Software/COSIMain_u2.sh; cd /scratch1/ckarwin/DC2/Background/Run_6;  python parallel.py {}"
rm nodes.txt
