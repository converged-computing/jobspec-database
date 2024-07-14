#!/bin/bash
# sbatch --priority=1000 launches/dpo-record.sh
#Resource Request 
#SBATCH --account=eleuther
#SBATCH --job-name=pythia
#SBATCH --output=slurm_%x_dpo-base-2.8b_%j.out ## filename of the output; the %j is equivalent to jobID; default is slurm-[jobID].out 
#SBATCH --partition=a40x ## the partitions to run in (comma seperated) 

#SBATCH --gpus=8 # number of gpus per task 
#SBATCH --cpus-per-gpu=8 
#SBATCH --nodes=1

#SBATCH --mail-type=ALL

##SBATCH --error=%x_%jerror.out    # Set this dir where you want slurm outs to go
##SBATCH --ntasks=1  ## number of tasks (analyses) to run 
##SBATCH --gpus-per-task=1 # number of gpus per task 
##SBATCH --ntasks-per-node=8

# module load cuda/11.7
module load cuda/12.1

export HYDRA_FULL_ERROR=1
# export HF_HOME='/admin/home-laura/.cache/huggingface/hub'
# export HF_DATASETS_CACHE='/admin/home-laura/.cache/huggingface/datasets'

# source /admin/home-laura/venvs/venv-direct-preference-optimization/bin/activate
source /admin/home-laura/venvs/venv-direct-preference-optimization310/bin/activate

# python new_scripts/hf_download.py

# want batch size constant: "batch_size" 32 * gradient_accumulation_steps 2 * num_gpus 8
##INFO check dataset processed correct, adjust eval_every 
##INFO upgraded to transformers==4.36.2
# python -u train.py loss.beta=0.1 model=pythia410m-sft seed=0 exp_name=pythia410m_dpo_seed0 batch_size=64 gradient_accumulation_steps=1
# python -u train.py loss.beta=0.1 model=pythia1-sft seed=0 exp_name=pythia1b_dpo_seed0 batch_size=64 gradient_accumulation_steps=1
# python -u train.py loss.beta=0.1 model=pythia14-sft seed=0 exp_name=pythia1.4b_dpo_seed0 batch_size=32 gradient_accumulation_steps=2 
# python -u train.py loss.beta=0.1 model=pythia28-sft seed=0 exp_name=pythia2.8b_dpo_seed0 batch_size=32 gradient_accumulation_steps=2 # 2 bs_per_gpu since "batch_size" 32 / (gradient_accumulation_steps 2 * num_gpus 8)

# python -u train.py loss.beta=0.1 n_epochs=3 model=pythia28 seed=0 exp_name=pythia2.8b_sfted0_dpo3_seed0 batch_size=32 gradient_accumulation_steps=2 revision="main" # 2 bs_per_gpu since "batch_size" 32 / (gradient_accumulation_steps 2 * num_gpus 8)
python -u train.py loss.beta=0.1 n_epochs=3 model=pythia28-sft-3-epochs seed=0 exp_name=pythia2.8b_sfted1_dpo3_seed0 batch_size=32 gradient_accumulation_steps=2 revision=epoch1-6000 # 2 bs_per_gpu since "batch_size" 32 / (gradient_accumulation_steps 2 * num_gpus 8)
# python -u train.py loss.beta=0.1 n_epochs=3 model=pythia28-sft-3-epochs seed=0 exp_name=pythia2.8b_sfted2_dpo3_seed0 batch_size=32 gradient_accumulation_steps=2 revision=epoch2-12000 # 2 bs_per_gpu since "batch_size" 32 / (gradient_accumulation_steps 2 * num_gpus 8)
# python -u train.py loss.beta=0.1 n_epochs=3 model=pythia28-sft-3-epochs seed=0 exp_name=pythia2.8b_sfted3_dpo3_seed0 batch_size=32 gradient_accumulation_steps=2 revision=main # 2 bs_per_gpu since "batch_size" 32 / (gradient_accumulation_steps 2 * num_gpus 8)

##NOTE see error {'rewards_train/chosen': 'nan', 'rewards_train/rejected': 'nan'
##INFO had to transformers==4.29.2 due to nan rewards for 70m, 160m (4.29.2)
##NOTE see error Some weights of GPTNeoXForCausalLM were not initializedx
# python -u train.py loss.beta=0.1 model=pythia70m-sft seed=0 exp_name=pythia70m_dpo_seed0 batch_size=64 gradient_accumulation_steps=1
# python -u train.py loss.beta=0.1 model=pythia160m-sft seed=0 exp_name=pythia160m_dpo_seed0 batch_size=64 gradient_accumulation_steps=1