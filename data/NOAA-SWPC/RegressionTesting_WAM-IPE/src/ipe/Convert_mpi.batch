#!/bin/bash
#set -ax
# ********* Settings that you should modify ********* #
# Set the MODEL variable to IPE or WAM-IPE
MODEL='WAM-IPE'
# Set the TAG to the appropriate Git Tag for the version of the code you are testing
TAG='V0.1'
# Set the path to the base WAM-IPE directory (containing IPELIB)
WAMIPEDIR=`pwd`/../..
# Set the path to the directory where your IPE output can be found
IPE_RUNDIR="/scratch3/NCEPDEV/swpc/noscrub/Joseph.Schoonover/WAM-IPE_20130306_1hr_ipe_nuopc_2d_petlayout/"
# Set the path to where you want the NetCDF output.
OUTDIR="/scratch3/NCEPDEV/swpc/noscrub/$USER/WAM-IPE_20130306_1hr_development/netcdf/"
# Set the path to where you want the eps files and the Latex file.
PLOTDIR="/scratch3/NCEPDEV/swpc/noscrub/$USER/WAM-IPE_20130306_1hr_development/plots/"
# Set the full path to the restart directory with the grid files can be found.
RESDIR="/scratch3/NCEPDEV/swpc/noscrub/ipe_initial_conditions/2013160300/grid_80x170/"
# *************************************************** #

# //////////////////////////////////////////////////////////////////////// #
# Do not modify below this point unless you really know what you are doing
# //////////////////////////////////////////////////////////////////////// #
# Clean house first, just in case
rm -rf tmp*

# Build the i2hg executable
module purge
module load intel/15.6.233 impi/5.0.3.048 netcdf-hdf5parallel
make --directory=${WAMIPEDIR}/IPELIB/src/plot/ i2hg_mpi -f i2hg_mpi.mak
export I2HG_EXE=${WAMIPEDIR}/IPELIB/bin/i2hg_mpi

# Build our list of files.
for plasmafile in ${IPE_RUNDIR}ipe_grid_plasma_params.*
do
  if [ -e $plasmafile ] ; then
    timestamp=$( echo $plasmafile | awk -F'.' '{print $NF}')
    neutralfile=${IPE_RUNDIR}ipe_grid_neutral_params.$timestamp
    echo $plasmafile  >> tmp_plasma
    echo $neutralfile >> tmp_neutral
  fi
done

# Create some more variables that are used below.
TASKS=`wc -l tmp_plasma | cut -d' ' -f 1`
export mcmd="IPEPlotsDriver( '${MODEL}', '${TAG}', '${OUTDIR}', '${PLOTDIR}' )"

# Bring in the IPE grid
ln -sf ${RESDIR}ipe_grid ./

# Bring in the IPE namelist parameter
cp ${IPE_RUNDIR}IPE.inp ./

# Make the output and plot directories if they don't already exist
[ ! -d $OUTDIR ]  && mkdir -p $OUTDIR
[ ! -d $PLOTDIR ] && mkdir -p $PLOTDIR


## Create job file.
tmp=tmp_job.sh
touch $tmp
chmod +x $tmp

cat >> $tmp << EOF
#!/bin/bash --login
#
#PBS -l procs=$TASKS
#PBS -l walltime=00:05:00
#PBS -q batch
#PBS -A swpc
#PBS -N convert_ipe
#PBS -j oe
set -ax
#
# change directory to the working directory of the job
# Use the if clause so that this script stays portable
#

if [ x\$PBS_O_WORKDIR != x ]; then
   cd \$PBS_O_WORKDIR
fi
module purge
module load intel/15.6.233 impi/5.0.3.048 netcdf-hdf5parallel

# Run the executable
ulimit -s unlimited
mpirun -np $TASKS $I2HG_EXE --plasma-file tmp_plasma --neutral-file tmp_neutral --output-dir $OUTDIR --grid-file ${RESDIR}GIP_Fixed_GEO_grid_lowres_corrected.bin

# Start plotting routine
module use -a /contrib/modulefiles
module load anaconda
#
## Run the matlab plotting script
python plot.py --input_directory $OUTDIR --output_directory $PLOTDIR

## Tar up the output directory
#tar -cvzf ${PLOTDIR::-1}.tar.gz ${PLOTDIR}

# Clean up this directory
rm ipe_grid IPE.inp fort.* tmp* *log

exit 0
EOF

qsub $tmp
