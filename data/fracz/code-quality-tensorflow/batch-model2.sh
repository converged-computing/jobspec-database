#!/bin/bash -l
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=4GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=24:00:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A scqfracz
## Specyfikacja partycji
#SBATCH -p plgrid-gpu
## Parametr wyznaczający indeksy zadania tablicowego
#SBATCH --array=12
#SBATCH --gres=gpu:1

export DATASET1=100-diff10-java-strict
export DATASET2=100-diff10to50-java-strict
export DATASET3=100-diff10-java-strict-no-parenthesis
export DATASET4=100-diff10to50-java-strict-no-parenthesis
export DATASET5=100-diff10-php-strict
export DATASET6=100-diff10to50-php-strict
export DATASET7=100-diff10-php-strict-no-parenthesis
export DATASET8=100-diff10to50-php-strict-no-parenthesis
export DATASET9=200-diff5to100-java-strict-noonlyadddel
export DATASET10=100-diff10to50-java-strict-noonlyadddel
export DATASET11=code-fracz-291
export DATASET12=code-fracz-518


export CURRENT_DATASET_VARIABLE=DATASET$SLURM_ARRAY_TASK_ID
export DATASET=${!CURRENT_DATASET_VARIABLE}

cd /net/people/plgfracz/quality

unset PYTHONPATH
module load plgrid/apps/cuda/8.0
module load plgrid/tools/python/3.6.0
source venv/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/net/people/plgfracz/cudnn/cuda/lib64

stdbuf -oL python model2.py $DATASET --steps 50000 --numHidden 256 --tokensCount=129 &> logs/model2-$DATASET.log
