#PBS -N fwrl-her-train-goal-noise  # Any name to identify your job
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
VERSION=0.2.0
PROJNAME=openai-baselines
MID_DIR=/z/home/dhiman/mid/floyd-warshall-rl/openai-baselines/her/
for d in src-$VERSION pbs build; do mkdir -p $MID_DIR/$d; done
cd $MID_DIR/src-$VERSION/
git clone $MID_DIR/git/ $PROJNAME
cd $PROJNAME
git pull
. setup.sh
python <<EOF
from baselines.her.experiment.train_many import (exp_conf_path_reward_goal_noise,
                                                 run_one_experiment)
exp_conf, common_kwargs = exp_conf_path_reward_goal_noise(num_cpu = 6, gitrev = "a048f9a")
experiments = list(exp_conf.items())
experiment_id = 0 + $PBS_ARRAYID
print("Running experiment no {}/{} with {}".format(
    experiment_id, len(experiments), experiments[experiment_id]))
confname, conf = experiments[experiment_id]
run_one_experiment(confname, conf, common_kwargs)
EOF
echo "end of pbs script"

# dhiman@fovea:~/.../openai-baselines/pbs$ python -c 'from baselines.her.experiment.train_many import exp_conf_path_reward_goal_noise; print("\n".join(map(str, enumerate(exp_conf_path_reward_goal_noise()[0].items()))))'
# Choosing the latest nvidia driver: /usr/lib/nvidia-384, among ['/usr/lib/nvidia-352', '/usr/lib/nvidia-375', '/usr/lib/nvidia-384']
# Choosing the latest nvidia driver: /usr/lib/nvidia-384, among ['/usr/lib/nvidia-352', '/usr/lib/nvidia-375', '/usr/lib/nvidia-384']
# (0, ('FetchReach-v1-dqst-uniform', {'env': 'FetchReach-v1', 'loss_term': 'dqst', 'goal_noise': 'uniform'}))
# (1, ('FetchReach-v1-dqst-zero', {'env': 'FetchReach-v1', 'loss_term': 'dqst', 'goal_noise': 'zero'}))
# (2, ('FetchReach-v1-ddpg-uniform', {'env': 'FetchReach-v1', 'loss_term': 'ddpg', 'goal_noise': 'uniform'}))
# (3, ('FetchReach-v1-ddpg-zero', {'env': 'FetchReach-v1', 'loss_term': 'ddpg', 'goal_noise': 'zero'}))
# (4, ('FetchReachPR-v1-dqst-uniform', {'env': 'FetchReachPR-v1', 'loss_term': 'dqst', 'goal_noise': 'uniform'}))
# (5, ('FetchReachPR-v1-dqst-zero', {'env': 'FetchReachPR-v1', 'loss_term': 'dqst', 'goal_noise': 'zero'}))
# (6, ('FetchReachPR-v1-ddpg-uniform', {'env': 'FetchReachPR-v1', 'loss_term': 'ddpg', 'goal_noise': 'uniform'}))
# (7, ('FetchReachPR-v1-ddpg-zero', {'env': 'FetchReachPR-v1', 'loss_term': 'ddpg', 'goal_noise': 'zero'}))
# (8, ('FetchPush-v1-dqst-uniform', {'env': 'FetchPush-v1', 'loss_term': 'dqst', 'goal_noise': 'uniform'}))
# (9, ('FetchPush-v1-dqst-zero', {'env': 'FetchPush-v1', 'loss_term': 'dqst', 'goal_noise': 'zero'}))
# (10, ('FetchPush-v1-ddpg-uniform', {'env': 'FetchPush-v1', 'loss_term': 'ddpg', 'goal_noise': 'uniform'}))
# (11, ('FetchPush-v1-ddpg-zero', {'env': 'FetchPush-v1', 'loss_term': 'ddpg', 'goal_noise': 'zero'}))
# (12, ('FetchPushPR-v1-dqst-uniform', {'env': 'FetchPushPR-v1', 'loss_term': 'dqst', 'goal_noise': 'uniform'}))
# (13, ('FetchPushPR-v1-dqst-zero', {'env': 'FetchPushPR-v1', 'loss_term': 'dqst', 'goal_noise': 'zero'}))
# (14, ('FetchPushPR-v1-ddpg-uniform', {'env': 'FetchPushPR-v1', 'loss_term': 'ddpg', 'goal_noise': 'uniform'}))
# (15, ('FetchPushPR-v1-ddpg-zero', {'env': 'FetchPushPR-v1', 'loss_term': 'ddpg', 'goal_noise': 'zero'}))
#
# Ran job ids 88959, 88958

