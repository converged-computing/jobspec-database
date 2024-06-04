#!/bin/bash
#SBATCH --job-name=TrainLDAMP        
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4                  
#SBATCH --ntasks-per-node=1              
#SBATCH --mem=32GB                     
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=24:00:00                
#SBATCH --output=TrainLDAMP.%j.out      
#SBATCH --error=TrainLDAMP.%j.err      

module purge; 
module load anaconda3/2020.07
module load cuda/11.6.2
module load cudnn/8.6.0.163-cuda11
source imageenv/bin/activate
python -u TrainLDAMP.py --data_path "/scratch/nm4074/imageprocessing/D-AMP_Toolbox/LDAMP_TensorFlow/TrainingData" --model_path "/scratch/nm4074/imageprocessing/D-AMP_Toolbox/LDAMP_TensorFlow/saved_models/LDAMP"