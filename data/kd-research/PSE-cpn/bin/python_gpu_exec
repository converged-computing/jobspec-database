#!/bin/bash

if [ -e $2.out ]; then
  >&2 echo $2.out File already exists. Protectively reject submitting job
  exit 1
fi

touch $2.out

cat << HERE | sbatch
#!/bin/bash

#SBATCH --job-name=$2
#SBATCH --output=$2.out
##SBATCH --ntasks=2                     # batch setting
#SBATCH --ntasks=7   			# active setting
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --constraint="gtx1080ti"       # batch setting
#SBATCH --constraint="gtx1080ti|rtx2080"         # active setting
#SBATCH --mail-user=hukaidonghkd@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

module load singularity || true

set -xe
singularity exec --nv /home/hpc/hpcguest4/sifs/keras-230704.sif python3 $1 ${@:3}
HERE
