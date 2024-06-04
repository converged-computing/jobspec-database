#!/bin/bash
#
#==>  Schism routine model run
#
#     Wolfgang Koch June 2020
#
#SBATCH --job-name=GBwaddenrec        # Specify job name
#SBATCH --partition=pAll
#SBATCH --ntasks=900  # Specify number of tasks
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --account=cluster
#SBATCH --output=/gpfs/home/routine-ksd/job-out+err/%x.o%j       # File name for standard output
#
# in both cases use srun and bind to cores in order to not use HT
# Case1: Run MPI parallel program using Intel MPI
#

schism_exec=/gpfs/work/ksddata/code/schism/source_code/schism/schism_2021_wwm_HDF5/bin/pschism_strand_WWM_ANALYSIS_GOTM_TVD-SB # SCHISM executable to run with
schism_bindir=/gpfs/work/ksddata/code/schism/source_code/schism/schism20210420/bin # schism binary folder containing binaries to combine outputs and hotstarts

set +k
cdir=/gpfs/work/jacobb/data/RUNS/routine_GB_wave/
rundir=/gpfs/work/jacobb/data/RUNS/routine_GB_wave/
sfluxdir=/gpfs/work/routine-ksd/schism-routine/sflux/ # get already processed for schism DWD Atmospheric files from hydrodynmic forecast
amm15dir=${rundir}Downloads/Download/download_cmems_GB/

cd ${cdir}
#sdat=${sdat:-$(date +%Y%m%d)}
sdat=20210804
fluxth_start=$(date +%Y)0101 #${sdat:-$(date +%Y)}0101
ihot=${ihot-1}
#rnda=${rnda-3.01}
rnda=30
dta=($sdat $(while ((++i<=${rnda%.*}));do date -ud $sdat+$i\days +%Y%m%d;done))
dta="${dta[*]}"
#SBATCH --error=%x.e%j        # File name for standard error output


#BJ create SFLUX file appropriate for dates from climatology
(cd river
python extract_flux_th.py $fluxth_start  $sdat
cd ..
ln -sf river/flux.th-$sdat flux.th
)

#continuation run
if [ "$SLURM_JOB_ID" ];then  
  module load compilers/intel
  module load intelmpi
  module load netcdf
  module load hdf5
  #SBATCH --mail-user=benjamin.jacob@hereon.de  # Set your e-mail address
  #SBATCH --mail-type=FAIL       # Notify user by email in case of job failure
  #export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
  #export I_MPI_FABRICS=shm:dapl

  ((ihot==1))&&ln -sf outputs/hotstart_$sdat.nc hotstart.nc 
  sed "
   /^\s*rnday/s/=.*/=$rnda/
   /^\s*ihot/s/=.*/=$ihot/
   /^\s*start_year/s/=.*/=${sdat:0:4}/
   /^\s*start_month/s/=.*/=${sdat:4:2}/
   /^\s*start_day/s/=.*/=${sdat:6:2}/
   " param.nml-template > param.nml
   
  #update wwm dates 
  sed "
   /^\s*BEGTC/s/=.*/=${sdat}.000000/
   /^\s*ENDTC/s/=.*/=${dta[-1]}.000000/ 
   " wwminput.nml-template > wwminput.nml

   ln -sf WW4NBSbnd/ww3.????_??to??_spec.nc ww3.spec.nc
   
  (cd sflux
   for d in $dta;do
     ((i++))
     ii=$(printf "%4.4d" $i)
     for p in air prc rad;do
       #ln -sf $p\_$d.nc sflux_$p\_1.$ii.nc  # original
	    ln -sf $sfluxdir$p\_$d.nc sflux_$p\_1.$ii.nc   # link from hydrodynmic routine

       done
     done
   )
  for p in elev2D SAL_3D TEM_3D uv3D;do
    ln -sf $p\_nlslayer__${sdat:2}-${dta##* ??}.th.nc $p.th.nc
    done
  #srun --mpi=pmi2 -l --cpu_bind=verbose,cores /gpfs/work/$USER/schism/build/bin/pschism_PREC_EVAP_TVD-VL > error.out
  srun --mpi=pmi2 -l --cpu_bind=verbose,cores $schism_exec > error.out
  rm $(realpath *D.th.nc)
  [ -s error.out ]&&error.out-$sdat
  sbatch -J SgbP${sdat:4:4} --ntasks=$((${rnda%.*}+2)) <<- .
	#!/bin/bash
	#SBATCH --job-name=GBpost        # Specify job name
	#SBATCH --partition=pAll
	#SBATCH --ntasks=5  # Specify number of tasks on each node
	#SBATCH --time=08:00:00        # Set a limit on the total run time
	#SBATCH --account=cluster
	#SBATCH --output=/gpfs/home/routine-ksd/job-out+err/%x.o%j       # File name for standard output
	module load compilers/intel
	module load intelmpi
	module load netcdf
	module load hdf5
	ulimit -s unlimited
	set +k
	(cd $rundir/outputs
	 for n in \$(find . -newer param.out.nml -name 'hotstart_0000_*.nc'|sed 's/.*_//;s/.nc//');do
		$schism_bindir/combine_hotstart7 --iteration \$n
	   #../../schism/build/bin/combine_hotstart7 --iteration \$n
	   ((i=n/864))
	   touch -r hotstart_0000_\$n.nc hotstart_it=\$n.nc
	   mv hotstart_it=\$n.nc hotstart_\$(date -ud \$sdat+\$i\\days +%Y%m%d).nc
	   done&
	 for j in \$(find . -newer param.out.nml -name 'schout_0000_*.nc'|sed 's/.*_//;s/.nc//');do
	    ($schism_bindir/combine_output10 -b \$j -e \$j
	   #(../../schism/build/bin/combine_output10 -b \$j -e \$j
	    touch -r schout_0000_\$j.nc schout_\$j.nc
	    mv schout_\$j.nc schout_\$(date -ud \$sdat+\$((j-1))days +%Y%m%d).nc)&
	   done
	 wait
	 )
	#SBATCH --error=GBpost.e%j        # File name for standard error output
	.
#Intialisation run	
else 
  if [ "$TERM" == dumb ];then
    . /etc/profile
    . ~/.bash_profile
    fi
  # Atmospheric Boundary Forcing	
  #(cd sflux;make $(for d in $dta;do echo -n \ air_$d.nc;done)) &
  #(cd Download/download_cmems_GB
  # pf=$(date -ud $sdat-1day +%Y%m%d)
  # ./download_copernicus.sh $pf $((${rnda%.*}+1))
  # /usr/local/bin/matlab  -nosplash -nodisplay -nodesktop  -nojvm \
   # -r "gen_boundary_forcing($sdat,${dta##* }); quit"|/usr/bin/tr -cd '[:print:]\n')
  # Ocean Boundary Forcing
  #(cd Downloads/$sdat-1day
  # generate bounray forcing
  (cd Downloads/
   #pf=$(date -ud  +%Y%m%d)
   #pf=$(date -ud $sdat  +%Y%m%d)
   /usr/local/bin/matlab  -nosplash -nodisplay -nodesktop  -nojvm \
   -r "gen_boundary_forcing($sdat,${dta##* },'$rundir'); quit"|/usr/bin/tr -cd '[:print:]\n')
	# generate hotstart
  (cd Downloads/
   python genHot_arg.py $amm15dir $rundir $sdat; quit|/usr/bin/tr -cd '[:print:]\n'
   cd ..
   ln -sf GB_hot_$sdat.nc hotstart.nc
   )
   
#python genHot_arg <srcdir> <rundir> <date>
	
  wait
  exit
  sbatch -J Sgb${sdat:4:4} $0
  #sbatch --export=ALL,sdat=$sdat,ihot=$ihot,rnda=$rnda $0
 # exit
  #(cd Download/download_cmems_GB
   #for d in $pf $dta;do
    # mv -b metoffice_foam1_amm15_NWS_???_$d.nc alt
    # done)
  fi