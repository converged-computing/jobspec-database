#!/bin/bash --login
#SBATCH -J LETKF
#SBATCH -A gsd-fv3-dev
##SBATCH --open-mode=truncate
#SBATCH -o log.nemsio2nc
#SBATCH -e log.nemsio2nc
#SBATCH -n 1
##SBATCH --nodes=1
##SBATCH -q debug
#SBATCH -p service
#SBATCH -t 00:30:00


#module load matlab
#module load python/3.7.5
### Youhua's module
__conda_setup="$('/apps/intel-2020.2/intel-2020.2/intelpython3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/apps/intel-2020.2/intel-2020.2/intelpython3/etc/profile.d/conda.sh" ]; then
        . "/apps/intel-2020.2/intel-2020.2/intelpython3/etc/profile.d/conda.sh"
    else
	export PATH="/apps/intel-2020.2/intel-2020.2/intelpython3/bin:$PATH"
    fi
fi
unset __conda_setup

module unload python/3.9.2
conda activate /work2/noaa/da/ytang/py39
export PYTHONPATH=/work2/noaa/da/ytang/py39/lib/python3.9/site-packages
export __LMOD_REF_COUNT_PYTHONPATH=$PYTHONPATH
export CONDA_PYTHON_EXE=/work2/noaa/da/ytang/py39/bin/python

SDATE=20171125 # in YYYYMMDDyy
EDATE=20180105 # in YYYYMMDDyy
DATADIR="/work/noaa/gsd-fv3-dev/bhuang/JEDI-FV3/expRuns/MISC/VIIRS-AWS/data"
PYEXE="/work/noaa/gsd-fv3-dev/bhuang/JEDI-FV3/expRuns/MISC/VIIRS-AWS/viirs_aws_download_globalmode_v1_bo.py"

python ${PYEXE} ${SDATE} ${EDATE} ${DATADIR}

exit 0
