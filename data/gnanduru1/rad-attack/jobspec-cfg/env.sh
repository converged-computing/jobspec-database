export HF_HOME=/scratch/$USER/.cache
echo make sure to export HF_TOKEN=\"your_api_token\"
echo HF_HOME=$HF_HOME

module purge
module load gcc/11.4.0 openmpi/4.1.4 python/3.11.4
#module load intel-compilers/2023.1.0 impi/2021.9.0 python/3.9.16
export PYTHONPATH=$(pwd):$PYTHONPATH
python -m venv ENV
source ENV/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
