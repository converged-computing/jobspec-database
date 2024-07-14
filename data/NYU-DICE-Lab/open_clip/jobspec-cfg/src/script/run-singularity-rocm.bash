#!/bin/bash

args=''
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="$args \"${i//\"/\\\"}\""
done
echo $args
ls
if [ "$args" == "" ]; then args="/bin/bash"; fi

tmp=/tmp/$USER/$$
if [[ "$SLURM_TMPDIR" != "" ]]; then
    tmp="$SLURM_TMPDIR/miopen/$$"
fi
mkdir -p $tmp

singularity \
    exec --rocm \
    --bind $tmp:$HOME/.config/miopen \
  $(for sqf in /vast/work/public/ml-datasets/yfcc15m/data/*.sqf; do echo "--overlay $sqf:ro"; done) \
  --overlay /scratch/bf996/singularity_containers/openclip_env_rocm_25.ext3:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  --overlay /scratch/bf996/datasets/in100.sqf:ro \
  --overlay /scratch/bf996/datasets/laion100.sqf:ro \
  --overlay /scratch/bf996/datasets/openimages100.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-r.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-a.sqf:ro \
  --overlay /scratch/bf996/datasets/imagenet-sketch.sqf:ro \
  /scratch/work/public/singularity/rocm5.1.1-ubuntu20.04.4.sif \
  /bin/bash -c "
 source /ext3/env.sh
 $args 
"