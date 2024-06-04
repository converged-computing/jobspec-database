#!/bin/bash
#SBATCH --job-name=iVAE
#SBATCH --account=Project_2002842
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-cpu=8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

module purge
# module load gcc/8.3.0
module load gcc/8.3.0 cuda/10.1.168
module load pytorch/1.4

# srun python3 main.py --config binary-3-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_3_2.log
# srun python3 main.py --config binary-4-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_4_2.log
# srun python3 main.py --config binary-3-3-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_3_3.log
# srun python3 main.py --config binary-4-3-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_4_3.log
# srun python3 main.py --config binary-2-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_2_2.log
# srun python3 main.py --config binary-5-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_5_2.log
# srun python3 main.py --config binary-5-3-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_5_3.log
# srun python3 main.py --config binary-6-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2.log
# srun python3 main.py --config binary-6-3-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_3.log
# srun python3 main.py --config binary-6-4-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_4.log
# srun python3 main.py --config binary-6-5-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_5.log
# srun python3 main.py --config binary-7-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_7_2.log
# srun python3 main.py --config binary-7-3-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_7_3.log
# srun python3 main.py --config binary-8-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_8_2.log
# srun python3 main.py --config binary-8-3-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_8_3.log
# srun python3 main.py --config binary-9-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_9_2.log
# srun python3 main.py --config binary-9-3-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_9_3.log
# srun python3 main.py --config binary-10-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_10_2.log
# srun python3 main.py --config binary-10-3-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_10_3.log
# srun python3 main.py --config binary-10-4-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_10_4.log
# srun python3 main.py --config binary-9-4-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_9_4.log
# srun python3 main.py --config binary-100-10-lbfgs.yaml --n-sims 3 --m 2.0 --s 0
# srun python3 main.py --config binary-100-10-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_100_10.log
# srun python3 main.py --config binary-100-20-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_100_20.log
# srun python3 main.py --config binary-100-30-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_100_30.log
# srun python3 main.py --config binary-100-40-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_100_40.log
# srun python3 main.py --config binary-100-50-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_100_50.log
# srun python3 main.py --config binary-100-60-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_100_60.log
# srun python3 main.py --config binary-100-70-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_100_70.log
# srun python3 main.py --config binary-100-80-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_100_80.log
# srun python3 main.py --config binary-100-90-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_100_90.log
# srun python3 main.py --config binary-100-20-lbfgs.yaml --n-sims 3 --m 2.0 --s 0
# srun python3 main.py --config binary-100-20-gd.yaml --n-sims 3 --m 2.0 --s 0
# srun python3 main.py --config binary-6-2-lbfgs-20s.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2_20s.log
srun python3 main.py --config binary-6-2-fast_ica.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2.log
# srun python3 main.py --config binary-6-2-fast_ica.yaml --n-sims 3 --m 2.0 --s 0