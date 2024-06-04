#!/bin/bash

# command to be executed when starting a job (can be changed later without modifying the docker image)
COMMAND="export EXPERIMENT_DIRECTORY=/app/shared && python /app/shared/viplanner/viplanner/multi_env.py"  # multi_env.py"  # m2f_overfit.py"

# define resource directories
CODE_DIR="/cluster/project/rsl/rothpa/vip_project/viplanner"
DATA_DIR="/cluster/scratch/rothpa/viplanner/data"
MODEL_DIR="/cluster/project/rsl/rothpa/vip_project/models"

# define directory where save the run scripts and log data
RUN_NAME="vip_train"
RUN_DIR="/cluster/scratch/rothpa/viplanner/runs"
LOG_DIR="/cluster/project/rsl/rothpa/vip_project/logs"

# The directory on the euler cluster where to docker container and related temporary files.
DOCKER_DIR="/cluster/scratch/rothpa/viplanner/docker"   # directory where docker container is stored
DOCKER_NAME="run_viplanner"                             # docker name
DOCKER_USE_CACHE=true                                   # use cache while building the docker container
DOCKER_BUILD_SINGULARITY_LOCAL=true                     # build singularity either locally (recommended) or on the cluster (only if resources issue on local machine)

# SSH access (user@domain). You should setup SSH access via private/public keys for this user!
SSH="rothpa@euler.ethz.ch"

### Create docker, singularity image and run script ###

# create run scripts and docker container

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$DIR"

mkdir "euler_scripts/"
cat <<EOT > euler_scripts/${RUN_NAME}_command.sh
#!/bin/bash
echo "Tar current code version and copy code ..."
mkdir -p \$TMPDIR/shared/$(basename "$CODE_DIR")
tar -C $CODE_DIR -cf $CODE_DIR.tar .
tar -xf $CODE_DIR.tar  -C \$TMPDIR/shared/$(basename "$CODE_DIR")

echo "Unpack Data ..."
mkdir -p \$TMPDIR/shared/data
for file in \$(ls $DATA_DIR/*.tar)
do
  echo "Unpacking \$file"
  tar -xf \$file  -C \$TMPDIR/shared/data
done

echo "Copy Docker Container ..."
tar -xf $DOCKER_DIR/$DOCKER_NAME.tar  -C \$TMPDIR

echo "Run Docker Container ..."
singularity exec --bind \$TMPDIR/shared:/app/shared --bind $MODEL_DIR:/app/shared/models --bind $LOG_DIR:/app/shared/logs --nv --writable --containall \$TMPDIR/$DOCKER_NAME.sif bash -c "$COMMAND"
EOT

chmod a+x euler_scripts/${RUN_NAME}_command.sh

cat <<END_OF_SCRIPT > euler_scripts/${RUN_NAME}_job.sh
#!/bin/bash

env2lmod
module load cuda cudnn nccl eth_proxy

cat <<EOT > job.sh
#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=END
#SBATCH --mail-user=roth.pascal@outlook.de
#SBATCH --job-name="imp_train-$(date +"%Y-%m-%dT%H:%M")"

sh $RUN_DIR/${RUN_NAME}_command.sh
EOT

sbatch < job.sh
rm job.sh
END_OF_SCRIPT
chmod a+x "euler_scripts/${RUN_NAME}_job.sh"

echo "Uploading Scripts ..."
cd "$DIR"
scp euler_scripts/* "$SSH:$RUN_DIR/"

echo "Building Docker Container ..."
cd "$DIR"
# NOTE: setting DOCKER_BUILDKIT=0 as bugfix because otherwise the gpu is not found during the build process even if default_runtime is nvidia 
# described here: https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime (notice at the end of the issue)
if $DOCKER_USE_CACHE 
then
    DOCKER_BUILDKIT=0 docker build ./ -t $DOCKER_NAME
else
    DOCKER_BUILDKIT=0 docker build ./ -t $DOCKER_NAME --no-cache
fi

if $DOCKER_BUILD_SINGULARITY_LOCAL
then
    echo "Create Singularity LOCAL ..."
    SINGULARITY_NOHTTPS=1 singularity build --sandbox $DOCKER_NAME.sif docker-daemon://$DOCKER_NAME:latest

    sudo tar -cvf $DOCKER_NAME.tar $DOCKER_NAME.sif
    scp $DOCKER_NAME.tar $SSH:$DOCKER_DIR

    echo "Clean-up ..."
    rm $DOCKER_NAME.tar
    rm $DOCKER_NAME.sif
    rm -rf euler_scripts/
else

echo "Uploading Docker Tarball ..."
cd "$DIR"
ssh -T "$SSH" << EOL
    mkdir -p $DOCKER_DIR/app/
    cd $DOCKER_DIR/
    rm -rf $DOCKER_NAME.tar $DOCKER_NAME
EOL

docker save $DOCKER_NAME -o $DOCKER_NAME.tar
scp $DOCKER_NAME.tar "$SSH:$DOCKER_DIR/"

echo "Creating Singularity Image on Remote Host ..."

ssh -T "$SSH" << EOL
    cd $DOCKER_DIR/
    singularity build $DOCKER_NAME docker-archive://$DOCKER_NAME.tar
EOL

echo "Cleaning Up ..."

rm $DOCKER_NAME.tar
rm -rf euler_scripts/
rm ${RUN_NAME}_command.sh

ssh -T "$SSH" << EOL
    rm $DOCKER_DIR/$DOCKER_NAME.tar
    apptainer cache clean -f
EOL

fi
