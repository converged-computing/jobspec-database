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

python main.py --dataset=mnist --model=cnn --wm=1 --dp=0 --adv=0 --adv_relax=1 --dp_relax=1 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01  --epochs=100 --save=0
python main.py --dataset=mnist --model=cnn --wm=0 --dp=0 --adv=1 --adv_relax=1 --dp_relax=1 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01  --epochs=100 --save=0
python main.py --dataset=mnist --model=cnn --wm=0 --dp=1 --adv=0 --adv_relax=1 --dp_relax=1 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01  --epochs=100 --save=0
# python main.py --dataset=mnist --model=cnn --wm=1 --dp=1 --adv=1 --adv_relax=1 --dp_relax=1 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01  --epochs=100 --save=0
# python main.py --dataset=mnist --model=cnn --wm=1 --dp=1 --adv=1 --adv_relax=1 --dp_relax=1 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01  --epochs=100 --save=0
# python main.py --dataset=mnist --model=cnn --wm=1 --dp=1 --adv=1 --adv_relax=1 --dp_relax=1 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01  --epochs=100 --save=0
# python main.py --dataset=mnist --model=lenet --wm=0 --dp=0 --adv=1 --adv_relax=0 --dp_relax=0 --dp_epsilon=1 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.02  --epochs=200 --save=0
# python main.py --dataset=fmnist --model=lenet --wm=0 --dp=0 --adv=1 --adv_relax=0 --dp_relax=0 --dp_epsilon=1 --grad_norm=1.0 --pgd_eps=0.15 --pgd_attack_steps=15 --pgd_step_size=0.02  --epochs=200 --save=0
# python main.py --dataset=fmnist --model=lenet --wm=0 --dp=1 --adv=1 --adv_relax=0 --dp_relax=0 --dp_epsilon=1 --grad_norm=1.0 --pgd_eps=0.15 --pgd_attack_steps=15 --pgd_step_size=0.02  --epochs=200 --save=0
# python main.py --dataset=cifar --model=lenet --wm=0 --dp=1 --adv=1 --adv_relax=1 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.031 --pgd_attack_steps=10 --pgd_step_size=0.0078 --save=0 --epochs=100 --num=1
# python main.py --dataset=cifar --model=lenet --wm=0 --dp=1 --adv=1 --adv_relax=1 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.031 --pgd_attack_steps=10 --pgd_step_size=0.0078 --save=0 --epochs=100 --num=2
# python main.py --dataset=fmnist --model=lenet --wm=0 --dp=0 --adv=1 --adv_relax=1 --dp_relax=0 --dp_epsilon=1 --grad_norm=1.0 --pgd_eps=0.1 --pgd_attack_steps=10 --pgd_step_size=0.01 --save=0 --epochs=200
# python main.py --dataset=mnist --model=cnn --wm=0 --dp=1 --adv=1 --adv_relax=1 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01 --save=5 --epochs=100
# python main.py --dataset=fmnist --model=cnn --wm=1 --dp=0 --adv=1 --adv_relax=1 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.15 --pgd_attack_steps=15 --pgd_step_size=0.01 --save=1
# python main.py --dataset=fmnist --model=cnn --wm=1 --dp=0 --adv=0 --adv_relax=1 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.15 --pgd_attack_steps=15 --pgd_step_size=0.01 --save=0
# python main.py --dataset=fmnist --model=cnn --wm=1 --dp=0 --adv=0 --adv_relax=1 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.15 --pgd_attack_steps=15 --pgd_step_size=0.01
# python main.py --dataset=mnist --model=cnn --wm=0 --dp=0 --adv=1 --adv_relax=1 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.4 --pgd_attack_steps=40 --pgd_step_size=0.01
# python main.py --dataset=mnist --model=cnn --wm=0 --dp=0 --adv=1 --adv_relax=1 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.5 --pgd_attack_steps=40 --pgd_step_size=0.01
# python main.py --dataset=mnist --model=cnn --wm=0 --dp=0 --adv=1 --adv_relax=1 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.6 --pgd_attack_steps=40 --pgd_step_size=0.01

# python main.py --dataset=mnist --model=cnn --wm=1 --dp=0 --adv=1 --adv_relax=1 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01
# python main.py --dataset=mnist --model=small-cnn --wm=0 --dp=0 --adv=0 --adv_relax=0 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01
# python main.py --dataset=mnist --model=small-cnn --wm=0 --dp=1 --adv=0 --adv_relax=0 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01
# python main.py --dataset=mnist --model=small-cnn --wm=0 --dp=0 --adv=1 --adv_relax=0 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01
# python main.py --dataset=mnist --model=small-cnn --wm=1 --dp=0 --adv=0 --adv_relax=0 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01
# python main.py --dataset=mnist --model=small-cnn --wm=1 --dp=1 --adv=0 --adv_relax=0 --dp_relax=1 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01
# python main.py --dataset=mnist --model=small-cnn --wm=1 --dp=0 --adv=1 --adv_relax=1 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01

# python main.py --dataset=mnist --model=cnn --wm=0 --dp=0 --adv=0 --adv_relax=0 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01
# python main.py --dataset=mnist --model=cnn --wm=0 --dp=1 --adv=0 --adv_relax=0 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01
# python main.py --dataset=mnist --model=cnn --wm=0 --dp=0 --adv=1 --adv_relax=0 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01
# python main.py --dataset=mnist --model=cnn --wm=1 --dp=0 --adv=0 --adv_relax=0 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01
# python main.py --dataset=mnist --model=cnn --wm=1 --dp=1 --adv=0 --adv_relax=0 --dp_relax=1 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01
# python main.py --dataset=mnist --model=cnn --wm=1 --dp=0 --adv=1 --adv_relax=1 --dp_relax=0 --dp_epsilon=3 --grad_norm=1.0 --pgd_eps=0.25 --pgd_attack_steps=25 --pgd_step_size=0.01
