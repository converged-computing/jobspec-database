#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=4:00:00
#SBATCH --array=3-5
#SBATCH --output=/scratch/hab1b/crossbeam-grow/slurm_outputs/nodup/grow50kt600/%j.txt
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

export CUDA_VISIBLE_DEVICES=0

total_max_values_explored=50000
task_timeout=600
run=$SLURM_ARRAY_TASK_ID
for attempts in 86400; do
	#let max_values_explored=${total_max_values_explored}/${attempts}
	max_values_explored=$total_max_values_explored
	for dataset in sygus; do
		# Augmented (growing library) CrossBeam with UR for evaluation
		grow_library=True
		python3 -m crossbeam.experiment.run_crossbeam \
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
		  --json_results_file=${results_dir}/run_${run}.${max_values_explored}.attempt-${attempts}.${task_timeout}.grow-${grow_library}.${model}.${dataset}.json
	done
done
