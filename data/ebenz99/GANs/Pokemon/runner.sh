#PBS -l select=1:ncpus=16:mem=20gb:ngpus=1:gpu_model=p100,walltime=3:00:00

cd ${PBS_O_WORKDIR}

source activate tf_env
python gan.py

