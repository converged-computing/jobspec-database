#! /bin/bash
#PBS -N CAMs_Attack
#PBS -o /home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/logs/out.log
#PBS -e /home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/logs/err.log
#PBS -l ncpus=104
#PBS -q cpu


rm -r /home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/logs
mkdir /home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/logs



source /apps/compilers/anaconda3/etc/profile.d/conda.sh
conda activate cv

python  /home/niranjan.rajesh_asp24/thesis-manifolds/resnet_exp/attacks.py --dataset imagenet --n_classes 100 --many_models True