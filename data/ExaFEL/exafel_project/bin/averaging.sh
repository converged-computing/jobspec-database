#!/bin/bash -l
#SBATCH --partition=regular
#SBATCH --job-name=cxid9114_avg
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --image=docker:mlxd/xfel:lq79
#SBATCH --mail-type=ALL
#SBATCH --mail-user=loriordan@lbl.gov
#SBATCH --volume="/global/cscratch1/sd/mlxd/DataProc/cxid9114/cxi:/reg/d/psdm/CXI;/global/cscratch1/sd/mlxd/DataProc/cxid9114/cxi:/reg/d/psdm/cxi;/global/cscratch1/sd/mlxd/DataProc/cxid9114/regg/g:/reg/g"

if [ ! -f ./avg.sh ]; then
    echo "#\!/bin/bash" >> ./avg.sh
    echo "export SIT_DATA=/reg/g/psdm/data"  >> ./avg.sh
    echo "export SIT_PSDM_DATA=/reg/d/psdm"  >> ./avg.sh
    echo "source /build/setpaths.sh"  >> ./avg.sh
    echo "cxi.mpi_average -x cxid9114 -r \${1} -a CxiDs2.0:Cspad.0 -d 572 -v -g 6.85" >> ./avg.sh
fi

# submit jobs
srun -n 68 -c 1 --cpu_bind=cores shifter ./avg.sh ${1} -R
srun -n 68 -c 1 --cpu_bind=cores shifter ./avg.sh ${2}
