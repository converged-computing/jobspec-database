#!/bin/bash

#Powderday cluster setup convenience script for SLURM queue mananger
#on HiPerGator at the University of FLorida.  This sets up the model
#files for a cosmological simulation where we want to model many
#galaxies at once.

#Notes of interest:

#1. This does *not* set up the parameters_master.py file: it is
#assumed that you will *very carefully* set this up yourself.

#2. This requires bash versions >= 3.0.  To check, type at the shell
#prompt:

#> echo $BASH_VERSION

n_nodes=$1
model_dir=$2
model_dir_remote=$3
galaxy=${4}
snap=${5}
snap_redshift=${6}
tcmb=${7}
ngal=${8}
sed_fits_dir=$9

echo "processing model file for galaxy,snapshot:  $galaxy,$snap"


#clear the pyc files
rm -f *.pyc

echo "setting up the output directory in case it doesnt already exist"
echo "snap is: $snap"
echo "model dir is: $model_dir"
echo "ngalaxies is: $ngal"
echo "sed fits dir is: $sed_fits_dir"

mkdir sed_fits_dir



echo "writing slurm submission master script file"
qsubfile="$sed_fits_dir/sedfit_snap${snap}.job"
rm -f $qsubfile
echo $qsubfile

echo "#! /bin/bash" >>$qsubfile
echo "#SBATCH --job-name=prosp.snap${snap}" >>$qsubfile
echo "#SBATCH --mail-type=ALL" >>$qsubfile
echo "#SBATCH --mail-user=krahm581@agnesscott.edu" >>$qsubfile
echo "#SBATCH --time=48:00:00" >>$qsubfile
echo "#SBATCH --tasks-per-node=32">>$qsubfile
echo "#SBATCH --nodes=$n_nodes">>$qsubfile
echo "#SBATCH --output=sedfitting_job.snap${snap}.output" >>$qsubfile
echo "#SBATCH --error=setfitting_job.snap${snap}.error" >>$qsubfile
echo "#SBATCH --mem-per-cpu=3800">>$qsubfile
echo "#SBATCH --account=narayanan">>$qsubfile
echo "#SBATCH --qos=narayanan-b">>$qsubfile
echo "#SBATCH --array=0-${ngal}" >>$qsubfile

echo -e "\n">>$qsubfile
echo -e "\n" >>$qsubfile

echo "module purge">>$qsubfile
echo "module load git">>$qsubfile
echo "module load intel/2018.1.163">>$qsubfile
echo "module load hdf5">>$qsubfile
echo "module load openmpi/4.0.3">>$qsubfile
echo -e "\n">>$qsubfile

#echo "python pd_front_end.py $model_dir parameters_master snap${snap}_galaxy\$SLURM_ARRAY_TASK_ID">>$qsubfile
echo "python run_prosp.py \$SLURM_ARRAY_TASK_ID ${model_dir_remote}snap${snapnum_str_prefix}${snap}.galaxy\${SLURM_ARRAY_TASK_ID}.rtout.sed $sed_fits_dir">>$qsubfile
echo " ">>$qsubfile
echo "#cannot be run at the same time as run_prosp">>$qsubfile
echo "#python get_prosp_quantities.py \$SLURM_ARRAY_TASK_ID $model_dir_remote">>$qsubfile

#done
