#!/bin/bash
#
#SBATCH --job-name=news_poly # Όνομα για διαχωρισμό μεταξύ jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=56G
#SBATCH --output=runs/outputs/polyjuice_news_random.out.log
#SBATCH --error=runs/errors/polyjuice_news_random.error.log
#SBATCH --account=pa210503
#SBATCH -t 4-00:00:00 # Ζητούμενος χρόνος DD-HH:MM:SS
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


cd /users/pa21/ptzouv/tkaravangelis/mice
module purge
module load gnu/8 cuda/10.1.168 intelmpi/2018 pytorch/1.7.0
source /users/pa21/ptzouv/tkaravangelis/venv_polyjuice/bin/activate

start=$(date +%s.%N)
srun python ../scripts/polyjuice_newsgroups.py ADJ
deactivate

end=$(date +%s.%N)
runtime=$( echo "$end - $start" | bc -l )
echo "Total script time $runtime"
