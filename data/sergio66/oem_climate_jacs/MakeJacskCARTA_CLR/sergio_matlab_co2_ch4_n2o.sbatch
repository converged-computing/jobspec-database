#!/bin/bash

# watch squeue -u sergio
# run this with   sbatch --array=1-40 sergio_matlab_co2_ch4_n2o.sbatch
#
#  Name of the job:
#SBATCH --job-name=KCARTA_CO2JAC_DRIVER

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
##SBATCH --time=3:59:00

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
#SBATCH --mem-per-cpu=8000
##  This is in MB, less aggressive
##SBATCH --mem-per-cpu=32000

########################################################################

##  Specify the job array (format:  start-stop:step)
#matlab -nodisplay -r "clust_make_co2_370_385_400_415_all; clust_make_co2_coljac_370_385_400_415; clust_make_co2_coljac_370_385_400_415v2; exit"
matlab -nodisplay -r "clust_make_ch4_1700_2000_all; clust_make_ch4_coljac_1700_2000; clust_make_ch4_coljac_1700_2000v2; exit"
matlab -nodisplay -r "clust_make_n2o_315_340_all; clust_make_n2o_coljac_315_340; clust_make_n2o_coljac_315_340v2; exit"
#matlab -nodisplay -r "clust_make_co2_3lay_370_385_400_415_all; clust_make_co2_3layjac_370_385_400_415_all; exit"
