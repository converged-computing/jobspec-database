#!/bin/bash
#
# watch all jobs     squeue -u sergio
# delete all jobs    scancel -u sergio
#
# if only doing things on taki
# /bin/rm Anomaly365_16/*/jac*.dat*
# /bin/rm Anomaly365_16/*/rad*.dat*
# or simply    rmer_anom.sc
#
# if running on the cluster, try Steve's suggestion
# rm_scratch_fat_files.sc
#
# run this with   sbatch --array=1-14600 sergio_matlab_anom.sbatch
# run this with   /bin/rm batchout*.* slurm* Anomaly365_16/*/individual_prof_convolved_kcarta* Anomaly365_16/*/rad.dat* Anomaly365_16/*/jac.dat*; sbatch --array=1-14600 sergio_matlab_anom.sbatch
# run this with   /bin/rm batchout*.* slurm* Anomaly365_16/*/individual_prof_convolved_kcarta* Anomaly365_16/*/rad.dat*; sbatch --array=1-Nmax%128 sergio_matlab_anom.sbatch
# run this with   rmer_slurm_JUNKrad.sc; sbatch --array=1-Nmax%128 sergio_matlab_anom.sbatch  
#

#  Name of the job:
#SBATCH --job-name=KCARTA_CLRJANOM_DRIVER

# requeue; if held type scontrol release JOBID
#SBATCH --requeue

#  N specifies that 1 job step is to be allocated per instance of matlab
#SBATCH -N1

#  This specifies the number of cores per matlab session will be
#available for parallel jobs
#SBATCH --cpus-per-task 1

#  Specify the desired partition develop/batch/prod
##SBATCH --partition=batch
#SBATCH --partition=high_mem

#  Specify the qos and run time (format:  dd-hh:mm:ss)

########################################################################
## 100 layer clouds
##SBATCH --qos=medium+
#SBATCH --time=8:59:00

# clear, or TwoSlab clouds, or fluxes
#SBATCH --qos=short+
#SBATCH --time=0:59:00

########################################################################

##  This is in MB, very aggressive but I have been running outta memory
##SBATCH --mem-per-cpu=24000
##  This is in MB, less aggressive
##SBATCH --mem-per-cpu=12000
##  This is in MB, very lean, don;t go below this as java starts crying, good for almost everything
##SBATCH --mem-per-cpu=4000
##  This is in MB, less aggressive
##SBATCH --mem-per-cpu=8000
##  This is in MB, less aggressive
#SBATCH --mem-per-cpu=32000

## defualt error output is to (by default slurm-{job id}_{array #}.out)
##  here we help spread the pain, into a directory
##SBATCH -o /home/sbuczko1/logs/sbatch/run_airibrad_rand_rtp-%A_%a.out
##SBATCH -e /home/sbuczko1/logs/sbatch/run_airibrad_rand_rtp-%A_%a.err
#######  here we help spread the pain, or into the dir where the call was initiated ... do this for fluu error reporting
##SBATCH -o batchout-%A_%a.out
##SBATCH -e batchout-%A_%a.err

########################################################################

##  Specify the job array (format:  start-stop:step)
####matlab -singleCompThread -nodisplay -r "LASTN = maxNumCompThreads(1); clust_do_kcarta_driver_anomaly; exit"
####matlab -singleCompThread -nodisplay -r "LASTN = maxNumCompThreads(1); clust_do_kcarta_driver_anomaly_filelist; exit"

#matlab -singleCompThread -nodisplay -r "clust_do_kcarta_driver_anomaly; exit"
matlab -singleCompThread -nodisplay -r "clust_do_kcarta_driver_anomaly_filelist; exit"
