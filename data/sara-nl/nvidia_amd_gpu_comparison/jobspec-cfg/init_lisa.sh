module purge
module load 2019
module load 2020
module load Python/3.6.6-foss-2019b
module unload CUDA
module load cuDNN/7.3.1-CUDA-10.0.130

VIRTENV=PT_rocm
VIRTENV_ROOT=~/.virtualenvs

deactivate
conda deactivate

if [ ! -z $1 ] && [ $1 = 'create' ]; then
  echo "Creating virtual environment $VIRTENV_ROOT/$VIRTENV"
  yes | rm -r $VIRTENV_ROOT/$VIRTENV
  python3 -m venv $VIRTENV_ROOT/$VIRTENV
fi

source $VIRTENV_ROOT/$VIRTENV/bin/activate

if [ ! -z $1 ] && [ $1 = 'create' ]; then
  pip install --upgrade pip
  pip install torch==1.7.1 torchvision --no-cache-dir
  pip install tqdm --no-cache-dir
  pip install tensorboardX --no-cache-dir
  pip3 install efficientnet_pytorch --no-cache-dir
  pip3 install scipy --no-cache-dir
fi
