#!/bin/bash

# Script to launch CLASSIC model runs on either the supercomputers or the front-end clusters.

# Usage: Modify the "User specified parameter" section as necessary and then just execute this script
#        on the platform chosen to do the run.

# The final output netCDF files will be stored in the directory "netcdf_files" under the user specified
# output directory.

# If a configure.R file exists in the user specified output directory, a job to run AMBER will be
# automatically launched on one of the ppps at the end of the model run.

# Background:

# CLASSIC currently runs well in parallel on a single node, but not across multiple nodes.

# The work-around is to run the model on a subset of the domain on each node. The number of nodes
# required is automatically computed based on the number of land grid cells in the domain, the
# estimated time required to run the simulation and the maximum wallclock allowed per node.

# At the end of the run, the files from each node will be stored in separate directories of the form
# "netcdf_files_XX", below the user specified output directory. Each directory will correspond to a subset
# of the full domain where XX is numbered sequentially from 01, 02, etc. These are stitched together
# to produce the full domain and will be stored in the "netcdf_files" directory.

# Modifications:
#
# Ed Chan, Dec 2019: Migrate to new supercomputers and front-ends.
#   - Removed the option to specify the size of /tmp on the front-ends as the default size of almost 90 GB
#     should be more than sufficient.
#   - System versions of cdo/nco are sufficiently recent on the front-ends, so specifying the paths to 
#     these utilities are only needed when running on the supercomputers.
#   - The hostfile must now be specified for "rumpirun" on the front-ends.
#   - Specify the same number of available CPUs (i.e. 40) for both supercomputers and front-ends.
#   - Adjust PBS directives for the new platforms, including changing the read permission of the job output file.
#   - Load newer versions of libraries/modules into the model run environment.
#   - Automatically launch a job on the ppp to run AMBER if a configure.R script exists in the output directory.
# Ed Chan, May 2019.
#   - Added support for the supercomputers.
#   - Added check of all input files specified in the job options file.
#   - Revised execution method so that if one node runs out of time, the others
#     will still copy completed files to the output directory.
# Ed Chan, Jan 2019.
#   - Split latitude bands so that boundaries are half-way between latitudes.
# Ed Chan, Jan 2019.
#   - Replaced mpirun with rumpirun for OpenMPI on front-ends since it has better support.
#   - Added option to allow user to additionally request the max available wallclock time per node.
# Ed Chan, Dec 2018.
#   - Allow user to review job parameters and input a different number of nodes.
#   - Added stitching of netCDF output files at the end of the job (if necessary).
#   - Made required modifications to the mpirun command line after recent system changes.
#   - Cleaned up command line generation for regular grids.

# Initial implementation: Ed Chan, Oct 2018 (based on Vivek's script for supercomputers)

# ===============================================================================================

# Which test this is is read in.
test=$1

# -------------------------
# User specified parameters
# -------------------------

# If the job is to be submitted to a different cluster than the one running this script,
# then the next two parameters need to be specified. Otherwise, they can be commented out.
# HDNODE/RUNPATH are normally set by the CCCma enviroment by default.

#RUNPATH=/space/hall4/sitestore/eccc/crd/ccrn/users/rec001
#HDNODE=ppp4

# Location of output files.

output_directory=/space/hall3/sitestore/eccc/crd/ccrp/scrd530/classic_checksums/$test

# Location of job options file.
# *** NB: This script checks all input files (as specified in the job options file).
#         Files that are not needed for the run must either be commented out or specified as an empty string.

job_options_file=$CI_PROJECT_DIR/tools/regression_testing/submission_scripts/job_options_$test.txt

cdir=$( pwd )

# Location of compiled CLASSIC executable.

# *** NB: This script runs the model by first ensuring that the proper environment is set up.
#         The executable MUST be compiled using the same environment by running the script 
#         ~rec001/public/classic/make_classic.sh in the same directory as the Makefile.
#executable=$HOME/CLASSIC/bin/CLASSIC_supercomputer

# For the front-ends:
executable=~/tmp/CLASSIC/CLASSIC_ppp

# Specify the row/column limits OR the lat/lon limits of the domain/sub-domain.
# If not specified, defaults are set for the full domain.
# In the case of regular grids, the "full domain" is assumed to be global and
# ranges from latitudes -58 to 90 in order to avoid processing over Antarctica.

# Projected grids.
# This option is used only if the job options file contains the namelist parameter: projectedGrid=.true.
# Rows range from 1 to nlat and columns 1 to nlon.
# row1=1 ; row2=160 ; col1=1 ; col2=310

# Regular grids.
# Lons typically range from 0 to 360, but will depend on what is used in the input files.
# Lats are typically ordered South to North, but will also depend the order used in the input files.
lat1=-28 ; lat2=60 ; lon1=120 ; lon2=220

# Email to send job start/end/abort notifications.

email="joe.melton@canada.ca"

# ===============================================================================================

### ---------------------------------------------
### Lines below may require editing as necessary.
### ---------------------------------------------

# Ensure paths are set for nco and cdo utilities.

export PATH=/fs/ssm/hpco/exp/mib002/anaconda/anaconda-4.4.0/anaconda_4.4.0_ubuntu-14.04-amd64-64/envs/cdo-1.9.0/bin:$PATH
export PATH=/fs/ssm/hpco/exp/mib002/anaconda2/anaconda2-5.0.1-hpcobeta2/anaconda_5.0.1-hpcobeta2_ubuntu-14.04-amd64-64/x/envs/nco-4.7.3/bin:$PATH

# A metric "phi" is used below as an estimate of the model's performance on a given platform for a typical model configuration.
# *** If jobs time out, it may need to be adjusted (downwards), especially if options such as saving daily output are turned on.
#
# The metric is defined as the number of grid cells that can be run over the length of the simulation
# (in years) per hour (i.e. gridcells*years/hour) and can be estimated by timing a (short) model run.
# It is used to estimate the number of nodes required to run the entire simulation under the constraint
# of the max wallclock time permitted on each node.
#
# For the Canada domain, 18300 grid cells can run for 2 years in ~0.5 hours.
# If A = 18300, B = 2, and C = 0.5, then A*B/C = 73200 gridcell-years/hour.
# Choose a somewhat lower number to be safe.

phi=55000

# Set parameters according to the platform.

case "$HOSTID" in
  xc*)   platform=xc
         # Ensure paths are set for nco and cdo utilities.
         export PATH=/fs/ssm/hpco/exp/mib002/anaconda/anaconda-4.4.0/anaconda_4.4.0_ubuntu-14.04-amd64-64/envs/cdo-1.9.0/bin:$PATH
         export PATH=/fs/ssm/hpco/exp/mib002/anaconda2/anaconda2-5.0.1-hpcobeta2/anaconda_5.0.1-hpcobeta2_ubuntu-14.04-amd64-64/x/envs/nco-4.7.3/bin:$PATH
         max_wallclock=3 ;;
  *ppp*) platform=ppp
         max_wallclock=6 ;;
  *)     echo "Current platform $HOSTID is not supported. Please log into daley/banting or ppp3/ppp4 and try again." && exit -1 ;;
esac

# -----------------------------------------------------------------------------------------------

### ------------------------------------------------
### Lines below should not normally require editing.
### ------------------------------------------------

# ---------------------------------------
# Check and/or prepare directories/files.
# ---------------------------------------

# Clear screen.

#/usr/bin/clear

# Check if output netCDF directories exist and provide the option to either delete them or abort.

mkdir -p $output_directory
cd $output_directory

list=netcdf_files_[0-9]*
[ "$(echo $list)" = "$list" ] && unset list
[ -d netcdf_files ] && list="netcdf_files $list"

if [ -n "$list" ] ; then
  echo
  echo
  echo "The following directories in $output_directory will be deleted: "
  echo
  printf "%s\n" $list
  echo
  echo
#  read -s -n1 -p "Press any key to continue or Ctrl-C to cancel ..."
  rm -rf $list
  echo
fi

# Set namelist parameters defined in the job options file as local shell variables.
# NB: output_directory is defined in job options file and will override what is specified, so reset it.

new_out=$output_directory
eval "$(cat $job_options_file | sed -n -e 's/[&!]/#/g; s/,/;/g; s/ *= */=/gp')"
output_directory=$new_out

# Ensure all input files specified in the job options file are accessible.
# *** NB: If a file is specified, it will be checked, even if it isn't actually used
#         in the model. In that case, a user can either specify an empty string or just
#         comment it out in the job options file. While it's possible to check
#         switches to see if files are needed, this will be more involved as the
#         current switch definitions don't contain common strings (like [Ff]file)
#         as for input files.

flag=false
sed -i "/runparams_file/s|'.*'|'$cdir/configurationFiles/template_run_parameters.txt'| ; /xmlFile/s|'.*'|'$cdir/configurationFiles/outputVariableDescriptors.xml'|" $job_options_file
for file_var in $(compgen -v | grep -G '[Ff]ile')
do
  infile=$(eval echo \$$file_var)
  [ -n "$infile" -a ! -f "$infile" ] && echo $infile is not accessible. && flag=true
done

$flag && echo 'Error: Job options file contains paths to inaccessible file(s)' && exit -1

# Ensure model executable is accessible.

[ -n "$executable" -a ! -f "$executable" ] && echo $executable is not accessible. && exit -2

# Set the location of various files such as standard/error output from execution of the job,
# copies of scripts used for the run, etc. Contents may be useful for debugging purposes.

run_files_directory=$output_directory/run_files
rm -rf $run_files_directory
mkdir $run_files_directory
cd $run_files_directory

# Modify and save a copy of the job options file to use /tmp for the output_directory on compute nodes.

runname=${output_directory##*/}

sed "/output_directory/s|'.*'|'/tmp/$runname'| ; /rs_file_to_overwrite/s|'.*'|'/tmp/$runname/rsFile_modified.nc'|" $job_options_file > job_options_${runname}.txt

job_options_file=$run_files_directory/job_options_${runname}.txt

# -----------------------------------------------------------------------------------------------

# --------------------------------------------------
# Obtain grid lat/lon and/or row/column information.
# --------------------------------------------------

# Extract the ground cover mask from "init_file" to get grid-related information.

cdo -s -selvar,GC $init_file GC.nc 2>/dev/null

# If the namelist parameter "projectedGrid" is set to ".true." in the job options file,
# then following computations are based on rows/columns of a projected grid. Otherwise,
# they are done based on lats/lons of a regular grid.

[ -n "$projectedGrid" ] && projectedGrid=$(echo $projectedGrid | tr '[:upper:]' '[:lower:]')

if [ "$projectedGrid" = '.true.' ] ; then

  # Get the number of rows/columns of the grid.

  nlat=$(ncdump -h GC.nc | fgrep "lat =" | cut -d' ' -f3)
  nlon=$(ncdump -h GC.nc | fgrep "lon =" | cut -d' ' -f3)

  # Set the row/column limits and/or defaults.

  [ -z "$row1" ] && row1=1 ; [ -z "$row2" ] && row2=$nlat ; [ $row2 -gt $nlat ] && row2=$nlat
  [ -z "$col1" ] && col1=1 ; [ -z "$col2" ] && col2=$nlon ; [ $col2 -gt $nlon ] && col2=$nlon

  # Calculate the number of land grid cells in the domain/sub-domain.

  total_land_cells=$(cdo -s -output -fldsum -setrtoc2,-1.5,-0.5,1,0 -selindexbox,$col1,$col2,$row1,$row2 GC.nc)

else

  # Set the lat/lon limits and/or defaults.

  [ -z "$lat1" ] && lat1=-58 ; [ -z "$lat2" ] && lat2=90 ; [ $lat1 -lt -58 ] && lat1=-58
  [ -z "$lon1" ] && lon1=0   ; [ -z "$lon2" ] && lon2=360

  # Calculate the number of land grid cells in the domain/sub-domain.

  cdo -s -sellonlatbox,$lon1,$lon2,$lat1,$lat2 GC.nc GC_subdomain.nc
  total_land_cells=$(cdo -s -output -fldsum -setrtoc2,-1.5,-0.5,1,0 GC_subdomain.nc)

  # Get the row numbers (of the full domain) that correspond to the first/last latitudes of the sub-domain.

  lats=( $(ncks --cdl -C -v lat GC_subdomain.nc | fgrep 'lat =' | tail -1 | sed -e 's/[;,]//g; s/^ *lat = //') )
  lat1_subdomain=${lats[0]}

  lats=( $(ncks --cdl -C -v lat GC.nc | fgrep 'lat =' | tail -1 | sed -e 's/[;,]//g; s/^ *lat = //') )

  for i in ${!lats[@]}
  do
    [ $lat1_subdomain = ${lats[$i]} ] && row1=$((i+1))
  done

fi

# -----------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Determine the number of nodes, wallclock time, etc. required for the job.
# Also determine the number of grid rows to run on each node.
# -------------------------------------------------------------------------

# Compute the length of the run (years) based on information from the job options file.

run_length=$(echo "($readMetEndYear-$readMetStartYear+1)*$metLoop" | bc)

# Calculate the time required to do runs using the gridcell-years/hour metric.

total_time=$(echo "scale=2; $run_length*$total_land_cells/$phi" | bc)

# Estimate the number of nodes required.

#nodes=$(echo "$total_time/$max_wallclock" + 1 | bc)         # rounded up

nodes='3'
# Loop to compute job parameters, giving the user an option to choose a different number of nodes.

flag=false

while :
do

#  /usr/bin/clear

  # Compute initial values of parameters used to estimate the amount of run time required on each node.

  grid_cells_per_node=$(echo "$total_land_cells/$nodes" | bc)
  run_time=$(echo "scale=2; $run_length*$grid_cells_per_node/$phi" | bc)

  # If the estimated run time is within 1/2 hour of the "max_wallclock", then it is safer to
  # add an extra node and run fewer grid cells per node.

  if (( $(echo "$max_wallclock-$run_time < 0.5" | bc -l) )) ; then
    grid_cells_per_node=$(echo "$total_land_cells/$((nodes+1))" | bc)
  fi

  # Get number of land grid cells in each row/lat of the grid. Note: cdo will only compute a zonal sum
  # for regular grids, so impose one on the data for projected grids, just for this computation.

  if [ "$projectedGrid" = '.true.' ] ; then
    land_cells_per_row=( $(cdo -s -output -zonsum -setrtoc2,-1.5,-0.5,1,0 -selindexbox,$col1,$col2,$row1,$row2 -setgrid,r${nlon}x$nlat GC.nc) )
  else
    land_cells_per_row=( $(cdo -s -output -zonsum -setrtoc2,-1.5,-0.5,1,0 -sellonlatbox,$lon1,$lon2,$lat1,$lat2 GC.nc) )
  fi

  # Get the row indices where the cumulative number of land grid cells between the indices do not exceed the
  # number of grid cells to be run per node. So, e.g. if the number of land grid cells in rows 1-50 are less
  # than $grid_cells_per_node, then the 1st row index is 50. And if the same is true for rows 51-100, then
  # the next row index is 100. And if there are only the 2 indices, then rows 101-<last row> contains the
  # last set of land grid cells (i.e. the job is split across 3 nodes).

  sum=0; row_boundaries=( $((row1-1)) ) ; land_cells_per_node=()

  for i in ${!land_cells_per_row[@]}
  do
    sum0=$sum ; sum=$(( sum + land_cells_per_row[$i] ))
    if [ $sum -gt $grid_cells_per_node ] ; then
      row_boundaries+=( $((i+row1-1)) )
      land_cells_per_node+=( $sum0 )
      sum=${land_cells_per_row[$i]}
    fi
  done

  row_boundaries+=($row2)
  land_cells_per_node+=($sum)
  nodes=${#land_cells_per_node[@]}

  # Now that the number of nodes is determined, compute the wallclock time per node for the job.
  # If the estimated run time is within 1/2 hour of the wallclock time, then it is safer to
  # add an extra half-hour as long as the total doesn't exceed the "max_wallclock".

  grid_cells_per_node=$(echo "$total_land_cells/$nodes" | bc)
  half_hour_intervals=$(echo "$run_length*$grid_cells_per_node*2/$phi + 1" | bc) # rounded up

  if (( $(echo "$half_hour_intervals/2-$run_time < 0.5" | bc -l) && $half_hour_intervals < $max_wallclock*2 )) ; then
    half_hour_intervals=$((half_hour_intervals+1))
  fi

  run_time=$(echo "scale=2; $run_length*$grid_cells_per_node/$phi" | bc)
  wallclock=$( printf '%d:%02d:00\n' $(( half_hour_intervals/2 )) $(( (half_hour_intervals*30)%60 )) )

  # Output job parameters to the screen and obtain user feedback.

  echo "Total number of land grid cells: $total_land_cells"
  echo "Length of run (years): $run_length"
  echo
  echo "Total time required (hours): $total_time"
  echo "Number of nodes required: $nodes"
  echo "Estimated amount of time required per node (hours) = $run_time"
  echo "Amount of time requested per node (H:MM:SS) = $wallclock"
  echo "Maximum wallclock (hours): $max_wallclock"
#  echo
#  read -p 'Choose one of the following options:

#  - Enter/Return to continue with the above job configuration
#  - Redo configuration by entering a different number of nodes to use (approximate)
#  - Ctrl-C to cancel

#  Input: '

#  [ -z "$REPLY" ] && break

#  while [[ ! $REPLY =~ ^[0-9]+$ || $REPLY = 0 ]]
#  do
#    read -p "Only positive integers are allowed. Try again: "
#  done

#  nodes=$REPLY ; flag=true

   flag=true
   break
# TO UNDO AUTOMATION OF THIS SCRIPT: delete the 'nodes=3', and uncomment the above lines

done

# Allow user to also set the wallclock time requested per node to the max allowed,
# if requesting a different number of nodes.

if $flag ; then
#  echo
#  read -p "Override wallclock time requested per node to the max of $max_wallclock hours (y or any other key to skip)?"
#  if [[ $REPLY == y* ]] ; then
   wallclock="00:07:00"
#    echo
   echo "Amount of time requested per node (H:MM:SS) = $wallclock"
#  fi
fi

# -----------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# Construct the command lines used to run the model. Also save a table identifying the row/lat
# boundaries associated with each node. These can be used to reconstruct the full domain.
# --------------------------------------------------------------------------------------------

# Since parameters passed to the model are different depending on the type of grid, the appropriate command line is constructed
# first and saved to a variable. Note that the rumpirun -H (hosts) option must be specified otherwise all commands will target
# the same node. On the front-ends, the hosts are numbered sequentially from zero and are listed in the $RUMPIRUN_HOSTFILE at runtime.

if [ "$projectedGrid" = '.true.' ] ; then

  printf "%-20s %-20s %-20s %-20s\n" 'Node Number' 'First Row' 'Last Row' 'Number of Grid Cells' > domain_partitions.txt

  for i in ${!land_cells_per_node[@]}
  do
    printf "%-20d %-20d %-20d %-20d\n" $((i+1)) $((row_boundaries[$i]+1)) ${row_boundaries[$((i+1))]} ${land_cells_per_node[$i]} >> domain_partitions.txt
    if [ $platform = ppp ] ; then
      command_lines+="time rumpirun -hostfile \$RUMPIRUN_HOSTFILE -H $i -n 40 run_model_script $((i+1)) $col1 $col2 $((row_boundaries[$i]+1)) ${row_boundaries[$((i+1))]} &\n"
    else
      command_lines+="time aprun -n 40 -N 40 run_model_script $((i+1)) $col1 $col2 $((row_boundaries[$i]+1)) ${row_boundaries[$((i+1))]} &\n"
    fi
  done

else

  # Compute 1/2 the difference between each latitude (only valid for 2nd to last latitude).

  lats=( $lat1 ${lats[@]} )

  for i in ${!lats[@]}
  do
    delta_lat+=( $(echo "(${lats[$((i+1))]} - ${lats[$i]})/2" | bc -l) )
  done

  printf "%-20s %-20s %-20s %-20s\n" 'Node Number' 'First Lat' 'Last Lat' 'Number of Grid Cells' > domain_partitions.txt

  # Compute the boundaries for each latitude band and use these on the command lines. Using boundaries half-way between
  # latitudes prevents the possibility of round-off errors causing problems.

  boundaries=( $(printf "%.4f" $lat1) )

  for i in ${!land_cells_per_node[@]}
  do

    if [ $i -eq $(( ${#land_cells_per_node[@]} - 1)) ] ; then
      boundaries+=( $(printf "%.4f" $lat2) )
    else
      boundaries+=( $(printf "%.4f" $(echo "${lats[${row_boundaries[$((i+1))]}]} + ${delta_lat[${row_boundaries[$((i+1))]}]}" | bc -l)) )
    fi

    printf "%-20d %-20.4f %-20.4f %-20d\n" $((i+1)) ${boundaries[$i]} ${boundaries[$((i+1))]} ${land_cells_per_node[$i]} >> domain_partitions.txt

    if [ $platform = ppp ] ; then
      command_lines+="time rumpirun -hostfile \$RUMPIRUN_HOSTFILE -H $i -n 40 run_model_script $((i+1)) $lon1 $lon2 ${boundaries[$i]} ${boundaries[$((i+1))]} &\n"
    else
      command_lines+="time aprun -n 40 -N 40 run_model_script $((i+1)) $lon1 $lon2 ${boundaries[$i]} ${boundaries[$((i+1))]} &\n"
    fi

  done

fi

command_lines=$(printf "$command_lines")

echo
cat domain_partitions.txt
echo

# -----------------------------------------------------------------------------------------------

# ----------------------------------------
# Construct the script and submit the job.
# ----------------------------------------

if [ $platform = ppp ] ; then
  # NB: Jobs will stay queued if mem>160G on ppp4 and >180G on ppp3. In either case, only 160G is actually available.
  pbs_line="#PBS -l select=$nodes:ncpus=40:mem=160G:res_image=eccc/eccc_all_ppp_ubuntu-18.04-amd64_latest,place=free"
else
  pbs_line="#PBS -l select=$nodes:ncpus=40:vntype=cray_compute,place=scatter"
fi

# Use of mcp appears to give the best performance, but could use other utilities (e.g. sscp). 
copy=mcp

# NB: "date +%F_%T" : %T returns colons, which is a problem for jobsub/qsub on ppp3/ppp4.

cat <<endjob > CLASSIC_${runname}.job
#!/bin/bash
#PBS -S /bin/bash
#PBS -N $runname
#PBS -l walltime=$wallclock
#PBS -q development 
#PBS -j oe
#PBS -o $run_files_directory/${runname}_$(date +%F_%H-%M-%S)_${HDNODE#*-}.out
$pbs_line
#PBS -m abe -M $email
#PBS -W umask=022

# Set required environment variables.

if [ $platform = ppp ] ; then
  . ssmuse-sh -x hpco/exp/hdf5-netcdf4/parallel/openmpi-3.1.2/static/intel-19.0.3.199/01
  export RUMPIRUN_ENV="LD_LIBRARY_PATH OMP_NUM_THREADS UCX_NET_DEVICES"
else
  # Load performance profiling tools (CrayPat lite).
  #module load perftools-base # should already be loaded
  module load perftools-lite
  # Load modules with the required libraries. Note that cray-netcdf conflicts with cray-netcdf-hdf5parallel.
  #module load cray-mpich     # should already be loaded
  module swap cray-netcdf cray-netcdf-hdf5parallel
fi

cd $run_files_directory

# Construct scripts to be run via MPI.

cat <<'endcat' > run_model_script
#!/bin/bash

# Get the rank of the MPI task.

if [ $platform = ppp ] ; then
  MPI_RANK=\$OMPI_COMM_WORLD_LOCAL_RANK
else
  MPI_RANK=\$ALPS_APP_PE
fi

# Create run directory on tmpfs and get local copy of restart file (1 MPI task per node).

if [ "\$MPI_RANK" = 0 ] ; then
  #mkdir /tmp/$runname
  mkdir -p /tmp/$runname/checksums
  $copy $rs_file_to_overwrite /tmp/$runname/rsFile_modified.nc
  cd /tmp/$runname
fi

# Run the model (all MPI tasks per node). 

$executable $job_options_file \$2/\$3/\$4/\$5

# Copy files to output directory (1 MPI task per node).
# No need to delete files on tmpfs (should be done by the system).

if [ "\$MPI_RANK" = 0 ] ; then
  cd /tmp
  run_directory=netcdf_files_\$(echo 0\$1 | tail -c3)
  mv $runname \$run_directory
  $copy -r \$run_directory $output_directory
fi
endcat

chmod u+x run_model_script

# Distribute the execution of the model across all nodes. 

echo 'Executing CLASSIC'
$command_lines
wait

# If necessary, stitch row/lat bands split across multiple output directories onto the full domain.

echo 'Executing classic_stitch_netcdf.sh'
time ~rec001/public/classic/classic_stitch_netcdf.sh $output_directory 32

# If a configure.R file is present in the output directory, then submit job to run AMBER.

#front_end=\$HDNODE
#[ \$HDNODE == banting ] && front_end=ppp3
#[ \$HDNODE == daley ] && front_end=ppp4

#if [ -f $output_directory/configure.R ] ; then
#  ~rec001/public/classic/amber.sh -c \$front_end $output_directory
#fi

endjob

# Clean up.

rm -f GC.nc GC_subdomain.nc

# Submit the job.

if [ $platform = ppp ] ; then
  jobsub -c $HDNODE CLASSIC_${runname}.job | cut -d '.' -f 1 > $cdir/job_id.txt
else
  qsub CLASSIC_${runname}.job | cut -d '.' -f 1 > $cdir/job_id.txt
fi

echo
echo 'Job submission is complete.'
echo
echo 'Check job status with:'
echo '~rec001/public/classic/jobstat'
echo
echo 'Check job output files for model text and error messages in:'
echo $run_files_directory
echo

exit
