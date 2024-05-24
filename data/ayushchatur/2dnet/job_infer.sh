#!/bin/bash
#SBATCH --job-name=ddnet
#SBATCH --nodes 1
#SBATCH --threads-per-core=1    # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8384                # total memory per node (4 GB per cpu-core is default)
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1             #GPU per node
#SBATCH --partition=v100_normal_q # slurm partition
#SBATCH --time=24:30:00          # time limit
#SBATCH -A HPCBIGDATA2           # account name

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes


export MASTER_PORT=$(comm -23 <(seq 20000 65535) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[0-9]\{1,5\}" | sort | uniq) | shuf | head -n 1)
echo "master port: $MASTER_PORT"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE=$WORLD_SIZE"
echo "slurm job: $SLURM_JOBID"
#expor job_id=$SLURM_JOBID
mkdir -p $SLURM_JOBID;cd $SLURM_JOBID
set | grep SLURM | while read line; do echo "# $line"; done > slurm.txt
env | grep -i -v slurm | sort > env.txt
#cp ../sparse_ddnet.py .
#cp ../sparse_ddnet_pp.py .
export batch_size=4
mkdir -p ./loss/
mkdir -p ./reconstructed_images/
mkdir -p ./reconstructed_images/val
mkdir -p ./reconstructed_images/test
mkdir -p ./visualize
mkdir -p ./visualize/val/
mkdir -p ./visualize/val/mapped/
mkdir -p ./visualize/val/diff_target_out/
mkdir -p ./visualize/val/diff_target_in/
mkdir -p ./visualize/val/input/
mkdir -p ./visualize/val/target/
mkdir -p ./visualize/test/
mkdir -p ./visualize/test/mapped/
mkdir -p ./visualize/test/diff_target_out/
mkdir -p ./visualize/test/diff_target_in/
mkdir -p ./visualize/test/input/
mkdir -p ./visualize/test/target/
cd ..
echo "tmpfs for this job at $TMPDIR"
echo "Staging full data per node"

#cd $TMPDIR/tmpfs
export dest_dir=$TMPDIR/tmpfs
#cp -r /projects/synergy_lab/garvit217/enhancement_data $dest_dir
echo "Staged full data per node on $dest_dir"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURMTMPDIR=$SLURMTMPDIR"

echo "working directory = "$SLURM_SUBMIT_DIR
echo "custom module path: $EASYBUILD_INSTALLPATH_MODULES"


echo "current dir: $PWD"
#chmod 755 * -R


: "${NEXP:=1}"
module list
module load site/infer-skylake_v100/easybuild/arc.arcadm
module load EasyBuild/4.6.1
module use $EASYBUILD_INSTALLPATH_MODULES

#module load numlib/cuDNN/8.4.1.50-CUDA-11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load CUDA/11.7.0
module load Anaconda3/2022.05
module load containers/singularity/3.8.5

echo "getting system info"
echo "cuda home: ${CUDA_HOME}"
alias nsys=$CUDA_HOME/bin/nsys
nsys --version
whereis nsys

if [ "$enable_gr" = "true" ]; then
  export imagefile="/projects/synergy_lab/ayush/containers/pytorch_2.0.sif"
else
  export imagefile="/projects/synergy_lab/ayush/containers/pytorch_22.04.sif"
fi

export infer_command="python ddnet_inference.py --filepath ${SLURM_JOBID} --batch ${batch_size} --epochs ${epochs} --out_dir ${SLURM_JOBID} --model ${model} --retrain ${retrain}"
export file="trainers.py"

export CMD="python ${file} --batch ${batch_size} --epochs ${epochs} --retrain ${retrain} --out_dir ${SLURM_JOBID} --amp ${mp} --num_w $num_data_w  --new_load ${new_load} --prune_amt $prune_amt --prune_t $prune_t  --wan $wandb --lr ${lr} --dr ${dr} --distback ${distback} --enable_profile ${enable_profile} --gr_mode ${gr_mode} --gr_backend ${gr_back} --enable_gr=${enable_gr} --schedtype ${schedtype} --model ${model} --port ${MASTER_PORT}"

export BASE="singularity exec --nv --writable-tmpfs --bind=/projects/synergy_lab/garvit217,/cm/shared,${TMPFS} ${imagefile} "

echo "CMD: ${BASE} ${CMD}"


if [ "$enable_profile" = "true"  ];then
  echo "cuda home: ${CUDA_HOME}"
  srun $BASE dlprof --output_path=${SLURM_JOBID} --nsys_base_name=nsys_${SLURM_PROCID} --profile_name=dlpro_${SLURM_PROCID} --mode=pytorch --nsys_opts="-t osrt,cuda,nvtx,cudnn,cublas,cusparse,mpi, --cuda-memory-usage=true" -f true --reports=all --delay 60 --duration 120 ${CMD}
else
  if [  "$inferonly"  == "true" ]; then
    export filepath=$1
    $BASE python ddnet_inference.py --filepath ${filepath} --batch ${batch_size} --epochs ${epochs} --out_dir ${filepath}
  else
    for _experiment_index in $(seq 1 "${NEXP}"); do
      (
    	echo "Beginning trial ${_experiment_index} of ${NEXP}"
    	srun --wait=120 --kill-on-bad-exit=0 --cpu-bind=none $BASE $CMD
      )
    done
    wait
    export infer_command="${BASE} ${infer_command}"
    echo "infer command: ${infer_command}"
    $infer_command
  fi
fi

