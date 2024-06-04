#!/bin/bash
#SBATCH --job-name=SDWebUI
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=<gatorusername>@ufl.edu   # Where to send mail
#SBATCH --nodes=1                     # Use one node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=1             # Use 1 core
#SBATCH --mem=19gb                    # Memory limit
#SBATCH --partition=gpu               # Required for GPUs
#SBATCH --gpus=a100:1                 # Specify GPU type
#SBATCH --time=04:00:00               # Time limit hrs:min:sec, or 7-0 for days + hrs
#SBATCH --account=art4659             # HiPerGator account
#SBATCH --qos=art4659                 # QOS Group
#SBATCH --output=webui_%j.out # Standard output and error log

date;hostname;pwd

# You should have previously created a module savelist called "webui" consisting of:
# 1) ufrc   2) git/2.30.1   3) ffmpeg/4.3.1   4) python/3.10   5) gcc/12.2.0   6) cuda/11.4.3
module r webui
# activate the python virtual environment created during initial install.
source venv/bin/activate

unset XDG_RUNTIME_DIR
unset LD_LIBRARY_PATH
# This helps with Out of Memory errors on HiPerGator A100s
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128'

port=$(shuf -i 20000-30000 -n 1)

echo -e "\nStarting Stable Diffusion WebUI on port ${port} on the $(hostname) server."
echo -e "\nSSH tunnel command:"
echo -e "\tssh -NL ${port}:$(hostname):${port} ${USER}@hpg.rc.ufl.edu"
echo -e "\nLocal browser URI:"
echo -e "\thttp://127.0.0.1:${port}"
host=$(hostname)

# Change USERNAME and PASSWD below to something you'll remember.
# This is also where you'll add any commandline arguments you need to run things.
python launch.py --listen --port ${port} --gradio-auth USERNAME:PASSWD --disable-safe-unpickle --enable-insecure-extension-access --lowram --xformers

# Since we have more VRAM than RAM using "--lowram" places model weights on the GPU rather than CPU.
