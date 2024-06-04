#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH -p workq
#SBATCH --time=6:00:00
#SBATCH --ntasks=6
#SBATCH --mem=64GB
#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --mail-user sirmcmissile47@gmail.com

start=`date +%s`

module load singularity
shopt -s expand_aliases
source /astro/mwasci/sprabu/aliases

set -x
{

obsnum=OBSNUM
base=BASE
myPath=MYPATH
link=

while getopts 'l:' OPTION
do
    case "$OPTION" in
        l)
            link=${OPTARG}
            ;;
    esac
done


cd ${base}/processing/
mkdir ${obsnum}
cd ${obsnum}

## move existing ms (for birli testing)
mv ${obsnum}.ms old${obsnum}.ms

wget -O ${obsnum}_ms.tar "${link}"
tar -xvf ${obsnum}_ms.tar

end=`date +%s`
runtime=$((end-start))
echo "the job run time ${runtime}"

}