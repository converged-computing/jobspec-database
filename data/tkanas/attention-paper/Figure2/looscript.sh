## shell script to run matlab sims

## Specify the shell for PBS ? required.
## Notice the double # for comments.
## You can also use /bin/csh.
# /usr/math/bin/uinit
 
#PBS -S /bin/sh
 
## Merge stderr to stdout (optional, otherwise they're in separate files)
#PBS -j oe

##Specify the output filename explicitly (optional; the default is named
## from the job ID, in the directory where qsub was run.)
#PBS -o /home/twiese/Clean_figs/Figure2/loo.out

##Requests 1 node to run 1 process in the queue.
#PBS -l nodes=1:ppn=1
#PBS -l walltime=10:00:00


## Request mail when job ends, or is aborted (optional, default is "a" only)
#PBS -m a

#PBS -t 1-4

# To start in a certain directory; you probably need to change.
cd $PBS_O_WORKDIR

## Below are the actual commands to be executed (i.e. to do the work).
echo "Test job starting at `date`"
echo "Greetings from job $PBS_ARRAY_INDEX"

/usr/local/bin/matlab -nodisplay -nosplash < data_analysis_leaveoneout.m > loo.log
echo "Test job finished at `date`"
#PBS -N /home/twiese/Clean_figs/Figure2
