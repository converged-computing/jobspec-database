# instructions for setting up UAB infra
module load Anaconda3
conda create --name cwl python=3
source activate cwl
pip install wes-service --user
pip install cwltool --user
pip install Cython --user
pip install pyslurm --user
# pip install toil[all]

# Start server locally... 
# you need to start the server first, which will be serving on some host and port
# then use the client to actually run stuff
screen -S wes
source ~/.bash_profile
source activate cwl

export WES_API_HOST=localhost:8082
export WES_API_AUTH='Header: value'
export WES_API_PROTO=http

#wes-server --port 8082 --backend=wes_service.toil_wes --opt extra=--clean=never
# Can't use /data/user... for some reason, singularity pull doesn't like it
wes-server --backend=wes_service.cwl_runner --opt runner=cwltool --opt extra=--singularity --opt extra=--cachedir=$USER_DATA/cache_workflows/ --port 8082
#Use the key sequence Ctrl-a + Ctrl-d to detach from the screen session.
#Use the key sequence Ctrl-a + H to obtain logs



# Example run with workflow service cwl files
git clone https://github.com/common-workflow-language/workflow-service.git
cd workflow-service
wes-client --info
wes-client --attachments="testdata/dockstore-tool-md5sum.cwl,testdata/md5sum.input" testdata/md5sum.cwl testdata/md5sum.cwl.json --no-wait
wes-client --attachments="testtool.cwl,testtool.json" testtool.cwl testtool.json --no-wait
wes-client --list

# Example run with challeng workflow template cwl files
git clone https://github.com/Sage-Bionetworks/ChallengeWorkflowTemplates.git
cd ChallengeWorkflowTemplates
wes-client scoring_harness_workflow.cwl scoring_harness_workflow.yaml  --attachments="download_submission_file.cwl,validate_email.cwl,validate.cwl,score.cwl,score_email.cwl,download_from_synapse.cwl,check_status.cwl,annotate_submission.cwl" --no-wait

# export SINGULARITY_DOCKER_USERNAME=

# export SINGULARITY_DOCKER_PASSWORD=
# export SINGULARITY_BINDPATH=$USER_SCRATCH:/tmp
# export TMPDIR=$USER_DATA
# export SINGULARITY_CACHEDIR=$USER_SCRATCH/tmp
# export SINGULARITY_PULLFOLDER=$USER_SCRATCH/tmp
# export SINGULARITY_TEMPDIR=/data/scratch/robert.allaway@sagebionetworks.org/


# curl localhost:8082/ga4gh/wes/v1/runs/<jobid>/cancel


### Set up orchestrator

# Use screen to allow for continous running
screen -S syn
# Export all the values you use in your .env file
# these values are explained above
# export WES_ENDPOINT=http://localhost:8082/ga4gh/wes/v1
# export WES_SHARED_DIR_PROPERTY=$USER_DATA/orchestrator/
# export SYNAPSE_USERNAME=ra2dreamservice
# export SYNAPSE_PASSWORD=xxxxxx
# export WORKFLOW_OUTPUT_ROOT_ENTITY_ID=syn20803806
# # Remember to put quotes around the EVALUATION_TEMPLATES
# export EVALUATION_TEMPLATES='{"9614319": "syn20976528"}'
# export COMPOSE_PROJECT_NAME=workflow_orchestrator
# export MAX_CONCURRENT_WORKFLOWS=4
source ~/.bash_profile
source ~/.env
java -jar WorkflowOrchestrator-1.0-SNAPSHOT-jar-with-dependencies.jar 


# Example slurm bash script

#!/bin/bash
#SBATCH --partition=pascalnodes
#SBATCH --job-name=test
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=thomas.yu@sagebionetworks.org
#SBATCH --output=test.txt
#SBATCH --error=test_errors.txt
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --account=ra2_dream
module load Singularity/2.6.1-GCC-5.4.0-2.26
singularity exec --net --no-home --bind /cm/local/apps/cuda/libs --nv -B /data/project/RA2_DREAM/train:/train:ro -B /data/project/RA2_DREAM/test_leaderboard:/test:ro -B $HOME/test:/output:rw docker://docker.synapse.org/syn20545112/example-model@sha256:6cc6dd92462b946fe5fbe0020055e63ce712c70e70fc327207cca6b26954b823 /run.sh
