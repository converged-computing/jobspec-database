#! /bin/sh
#SBATCH --job-name=gpujob
#SBATCH --qos="long"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=20:00:00
#SBATCH --partition g 
#SBATCH --gres=gpu:0
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err


singularity run --nv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest /users/maximilian.moser/DeepLeptonStuff/DeepLepton-Training/convert.sh

echo "done"
