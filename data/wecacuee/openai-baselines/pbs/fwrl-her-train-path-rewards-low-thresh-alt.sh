#PBS -N fwrl-her-train-path-rewards-low-thresh-alt  # Any name to identify your job
#PBS -j oe                   # Join error and output files for convinience
#PBS -l walltime=1000:00:00     # Keep walltime big enough to finish the job
#PBS -l nodes=1:ppn=6:gpus=1 # nodes requested: Processor per node: gpus requested
#PBS -S /bin/bash            # Shell to use
#PBS -m a                  # Mail to <user>@umich.edu on abort, begin and end
#PBS -M dhiman@umich.edu     # Email id to alert
#PBS -o /z/home/dhiman/mid/floyd-warshall-rl/openai-baselines/her/pbs/
#
# #PBS -q fluxg              # Not required for blindspot but for flux
# #PBS -A jjcorso_fluxg      # Not required for blindspot but for flux
#PBS -t 0-39

echo "starting pbs script"
echo "running on $(hostname)"
VERSION=0.3.0
PROJNAME=openai-baselines
MID_DIR=/z/home/dhiman/mid/floyd-warshall-rl/openai-baselines/her/
for d in src-$VERSION pbs build; do mkdir -p $MID_DIR/$d; done
cd $MID_DIR/src-$VERSION/
git clone $MID_DIR/git/ $PROJNAME
cd $PROJNAME
git pull
. setup.sh
python <<EOF
from baselines.her.experiment.train_many import (exp_conf_path_reward_low_thresh_alt,
                                                 run_one_experiment)
exp_conf, common_kwargs = exp_conf_path_reward_low_thresh_alt(num_cpu = 6, gitrev = "be467df")
experiments = list(exp_conf.items())
experiment_id = 0 + $PBS_ARRAYID
print("Running experiment no {}/{} with {}".format(
    experiment_id, len(experiments), experiments[experiment_id]))
confname, conf = experiments[experiment_id]
run_one_experiment(confname, conf, common_kwargs)
EOF
echo "end of pbs script"
# dhiman@fovea:~/.../ext/openai-baselines$ python -c 'from baselines.her.experiment.train_many import exp_conf_path_reward_low_thresh_alt; print("\n".join(map(str, enumerate(exp_conf_path_reward_low_thresh_alt()[0].items()))))
# > '
# Choosing the latest nvidia driver: /usr/lib/nvidia-384, among ['/usr/lib/nvidia-352', '/usr/lib/nvidia-375', '/usr/lib/nvidia-384']
# Choosing the latest nvidia driver: /usr/lib/nvidia-384, among ['/usr/lib/nvidia-352', '/usr/lib/nvidia-375', '/usr/lib/nvidia-384']
# (0, ('FetchReach-v1-qlst', {'env': 'FetchReach-v1', 'loss_term': 'qlst'}))
# (1, ('FetchReach-v1-ddpg', {'env': 'FetchReach-v1', 'loss_term': 'ddpg'}))
# (2, ('FetchPush-v1-qlst', {'env': 'FetchPush-v1', 'loss_term': 'qlst'}))
# (3, ('FetchPush-v1-ddpg', {'env': 'FetchPush-v1', 'loss_term': 'ddpg'}))
# (4, ('FetchSlide-v1-qlst', {'env': 'FetchSlide-v1', 'loss_term': 'qlst'}))
# (5, ('FetchSlide-v1-ddpg', {'env': 'FetchSlide-v1', 'loss_term': 'ddpg'}))
# (6, ('FetchPickAndPlace-v1-qlst', {'env': 'FetchPickAndPlace-v1', 'loss_term': 'qlst'}))
# (7, ('FetchPickAndPlace-v1-ddpg', {'env': 'FetchPickAndPlace-v1', 'loss_term': 'ddpg'}))
# (8, ('HandReach-v0-qlst', {'env': 'HandReach-v0', 'loss_term': 'qlst'}))
# (9, ('HandReach-v0-ddpg', {'env': 'HandReach-v0', 'loss_term': 'ddpg'}))
# (10, ('HandManipulateBlockRotateXYZ-v0-qlst', {'env': 'HandManipulateBlockRotateXYZ-v0', 'loss_term': 'qlst'}))
# (11, ('HandManipulateBlockRotateXYZ-v0-ddpg', {'env': 'HandManipulateBlockRotateXYZ-v0', 'loss_term': 'ddpg'}))
# (12, ('HandManipulatePenRotate-v0-qlst', {'env': 'HandManipulatePenRotate-v0', 'loss_term': 'qlst'}))
# (13, ('HandManipulatePenRotate-v0-ddpg', {'env': 'HandManipulatePenRotate-v0', 'loss_term': 'ddpg'}))
# (14, ('HandManipulateEggFull-v0-qlst', {'env': 'HandManipulateEggFull-v0', 'loss_term': 'qlst'}))
# (15, ('HandManipulateEggFull-v0-ddpg', {'env': 'HandManipulateEggFull-v0', 'loss_term': 'ddpg'}))
# (16, ('FetchReachPR-v1-dqst', {'env': 'FetchReachPR-v1', 'loss_term': 'dqst'}))
# (17, ('FetchPushPR-v1-dqst', {'env': 'FetchPushPR-v1', 'loss_term': 'dqst'}))
# (18, ('FetchSlidePR-v1-dqst', {'env': 'FetchSlidePR-v1', 'loss_term': 'dqst'}))
# (19, ('FetchPickAndPlacePR-v1-dqst', {'env': 'FetchPickAndPlacePR-v1', 'loss_term': 'dqst'}))
# (20, ('HandReachPR-v0-dqst', {'env': 'HandReachPR-v0', 'loss_term': 'dqst'}))
# (21, ('HandManipulateBlockRotateXYZPR-v0-dqst', {'env': 'HandManipulateBlockRotateXYZPR-v0', 'loss_term': 'dqst'}))
# (22, ('HandManipulatePenRotatePR-v0-dqst', {'env': 'HandManipulatePenRotatePR-v0', 'loss_term': 'dqst'}))
# (23, ('HandManipulateEggFullPR-v0-dqst', {'env': 'HandManipulateEggFullPR-v0', 'loss_term': 'dqst'}))
# Running on job id 89021[0-23]
