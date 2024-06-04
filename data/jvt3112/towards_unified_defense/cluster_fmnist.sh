#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=jvt22 # required to send email notifcations - please replace <your_username> with your college login name or email address
#PBS -l walltime=20:00:00
export PATH=/vol/bitbucket/jvt22/myvenv/bin:$PATH
source activate
# source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
# TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
# running python script for federated learning protection mechanisms
# cd thesisCode/federated_learning_protection_mechanisms/
# FL no protection
python main.py --model=cnn --dataset=fmnist --adv=1 --dp=1 --wm=1 --gpu=1 --epochs=50 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --pgd_eps=0.15 --pgd_attack_steps=15 --pgd_step_size=0.01 --dp_epsilon=3 --adv_relax=1 --dp_relax=1
# python main.py --model=cnn --dataset=fmnist --adv=1 --dp=1 --wm=0 --gpu=1 --epochs=50 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --pgd_eps=0.15 --pgd_attack_steps=15 --pgd_step_size=0.01 --dp_epsilon=3 --adv_relax=1 --dp_relax=1
# python main.py --model=cnn --dataset=fmnist --adv=1 --dp=0 --wm=1 --gpu=1 --epochs=50 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --pgd_eps=0.15 --pgd_attack_steps=15 --pgd_step_size=0.01 --dp_epsilon=3 --adv_relax=1 --dp_relax=1
# python main.py --model=cnn --dataset=fmnist --adv=0 --dp=0 --wm=0 --gpu=1 --epochs=50 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --pgd_eps=0.15 --pgd_attack_steps=15 --pgd_step_size=0.01 --dp_epsilon=3 --adv_relax=1 --dp_relax=1
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=0 --dp_epsilon=3 --wm=0 --adv_relax=0 --dp_relax=0
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=0 --dp_epsilon=3 --wm=0 --adv_relax=0 --dp_relax=0

# FL with DP
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=0 --adv_relax=0 --dp_relax=0
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=32 --lr=0.01 --adv=0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=0 --adv_relax=0 --dp_relax=0

# FL with WM   
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=0 --dp_epsilon=3 --wm=1 --adv_relax=0 --dp_relax=0
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=0 --dp_epsilon=3 --wm=1 --adv_relax=0 --dp_relax=0

# FL with Adv
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=0 --dp_epsilon=3 --wm=0 --adv_relax=0 --dp_relax=0
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=0 --dp_epsilon=3 --wm=0 --adv_relax=0 --dp_relax=0

# FL with DP + WM
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=1 --adv_relax=0 --dp_relax=0
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=1 --adv_relax=0 --dp_relax=0

# FL with DP + WM + dp_relax
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=1 --adv_relax=0 --dp_relax=1
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=1 --adv_relax=0 --dp_relax=1

# FL with Adv + WM
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=0 --dp_epsilon=3 --wm=1 --adv_relax=0 --dp_relax=0
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=0 --dp_epsilon=3 --wm=1 --adv_relax=0 --dp_relax=0

# FL with Adv + WM + adv_relax
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=0 --dp_epsilon=3 --wm=1 --adv_relax=1 --dp_relax=0
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=0 --dp_epsilon=3 --wm=1 --adv_relax=1 --dp_relax=0
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=20 --num_clients=1 --frac=1 --local_ep=5 --local_bs=256 --lr=0.005 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=0 --dp_epsilon=3 --wm=1 --adv_relax=1 --dp_relax=0

# FL with Adv + DP 
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=200 --num_clients=50 --frac=0.5 --local_ep=1 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=0 --adv_relax=0 --dp_relax=0
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=200 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.1 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=0 --adv_relax=0 --dp_relax=0
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=0 --adv_relax=0 --dp_relax=0
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=32 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=0 --adv_relax=0 --dp_relax=0
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=10 --num_clients=50 --frac=0.2 --local_ep=5 --local_bs=32 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=8 --wm=0 --adv_relax=0 --dp_relax=0
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=10 --num_clients=50 --frac=0.2 --local_ep=5 --local_bs=32 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=12 --wm=0 --adv_relax=0 --dp_relax=0
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=10 --num_clients=50 --frac=0.2 --local_ep=5 --local_bs=32 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=16 --wm=0 --adv_relax=0 --dp_relax=0
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=10 --num_clients=50 --frac=0.2 --local_ep=5 --local_bs=32 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=20 --wm=0 --adv_relax=0 --dp_relax=0
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=10 --num_clients=50 --frac=0.2 --local_ep=5 --local_bs=32 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=24 --wm=0 --adv_relax=0 --dp_relax=0

# FL with Adv + DP + WM
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=1 --adv_relax=0 --dp_relax=0
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=1 --adv_relax=0 --dp_relax=0

# FL with Adv + DP + WM + dp_relax
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=1 --adv_relax=0 --dp_relax=1
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=1 --adv_relax=0 --dp_relax=1

# FL with Adv + DP + WM + adv_relax 
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=1 --adv_relax=1 --dp_relax=1
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=1 --adv_relax=1 --dp_relax=0

# FL with Adv + DP + WM + adv_relax + dp_relax
# python main.py --model=cnn --dataset=mnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=1 --adv_relax=1 --dp_relax=1
# python main.py --model=cnn --dataset=fmnist --gpu=1 --epochs=20 --num_clients=50 --frac=0.5 --local_ep=5 --local_bs=64 --lr=0.01 --adv=1 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --dp=1 --dp_epsilon=3 --wm=1 --adv_relax=1 --dp_relax=1
