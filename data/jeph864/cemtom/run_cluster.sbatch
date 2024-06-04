#!/bin/bash
#SBATCH -J ntms                      # the job name
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:V100:4
#SBATCH --cpus-per-task=8            # use 1 thread per taks
#SBATCH -N 1
#SBATCH --partition=informatik-mind
#SBATCH --output=output/ntms_%j_out.txt         # capture output
#SBATCH --error=output/ntms_%j_err.txt          # and error streams

#module purge
#module add nvidia/10.0
module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh


conda activate ntms
PROJECT="/scratch/$USER/thesis/tms/atlas/baselines/neural-topic-models/"
cd $PROJECT


data=${1-20ng} # amazoncat-13k, wiki10-31k
model=${2-etm} # etm, etm_dirichlet, etm_dirichlet_rsvi, prod_lda, nb_vae, dvae, dvae_rsvi
run_ntms_mayank(){
  PROJECT="/scratch/$USER/thesis/tms/atlas/baselines/neural-topic-models/"
  # shellcheck disable=SC2164
  cd "$PROJECT"
  python main.py \
    --data_name 20ng \
    --model_name ${1-etm} \
    --max_epochs 50
}

run_cluster_analysis(){
    PROJECT="/scratch/$USER/thesis/tms/atlas/baselines/Cluster-Analysis/"
    # shellcheck disable=SC2164
    cd "$PROJECT"
    python
}
run_ntms_mayank prod_lda
conda deactivate 
