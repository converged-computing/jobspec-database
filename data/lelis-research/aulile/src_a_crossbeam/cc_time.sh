#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0:05:00
#SBATCH --array=1-2
#SBATCH --output=/scratch/hab1b/crossbeam-grow/slurm_outputs/nodup/grow50kt3600/%j.txt
#SBATCH --account=rrg-lelis

module load python/3.8 cuda cudnn
source $HOME/cb-env/bin/activate
cd /scratch/hab1b/crossbeam-grow

results_dir=augmented/bustle_results_nodup
maxni=3
maxsw=20
beam_size=10
data_root=crossbeam/data
models_dir=trained_models/bustle/
model=vw-bustle_sig-vsize

task_timeout=3600
max_values_explored=50000
dataset=sygus

argfile=/scratch/hab1b/crossbeam-grow/augmented/bustle_results_nodup/failed_${dataset}_${max_values_explored}.txt
args=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $argfile)
run=$(echo $args | cut -d' ' -f1)
task_name=$(echo $args | cut -d' ' -f2)

export CUDA_VISIBLE_DEVICES=0

grow_library=True
for attempts in 86400; do
	python3 -m crossbeam.experiment.run_crossbeam \
	  --task_name=${task_name} \
	  --grow_library=${grow_library} \
	  --stop_if_no_grow=False \
	  --attempts=${attempts} \
	  --seed=${run} \
	  --domain=bustle \
	  --model_type=char \
	  --max_num_inputs=$maxni \
	  --max_search_weight=$maxsw \
	  --data_folder=${data_root}/${dataset} \
	  --save_dir=${models_dir} \
	  --beam_size=$beam_size \
	  --gpu_list=0 \
	  --num_proc=1 \
	  --eval_every=1 \
	  --train_steps=0 \
	  --do_test=True \
	  --use_ur=True \
	  --task_timeout=${task_timeout} \
	  --attempt_timeout=${task_timeout} \
	  --max_values_explored=${max_values_explored} \
	  --load_model=${model}/model-best-valid.ckpt \
	  --io_encoder=bustle_sig --value_encoder=bustle_sig --encode_weight=True \
	  --json_results_file=${results_dir}/run_${run}.${max_values_explored}.attempt-${attempts}.${task_timeout}.grow-${grow_library}.${model}.${dataset}.${task_name}.json
done
