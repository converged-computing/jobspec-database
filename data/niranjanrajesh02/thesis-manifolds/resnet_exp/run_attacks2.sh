#! /bin/bash
#PBS -N CAMs_Attack2
#PBS -o /home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/attack_logs/out.log
#PBS -e /home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/attack_logs/err.log
#PBS -l ncpus=100
#PBS -q cpu


rm -r /home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/attack_logs
mkdir /home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/attack_logs



source /apps/compilers/anaconda3/etc/profile.d/conda.sh
conda activate cv

python  /home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/attacks2.py --dataset imagenet --n_classes 100 --many_models True