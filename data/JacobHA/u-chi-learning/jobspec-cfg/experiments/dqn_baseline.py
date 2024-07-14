import json
import sys
import copy
import yaml

import wandb
import argparse

sys.path.append('darer')
from stable_baselines3 import DQN
sys.path.append("darer")
from utils import env_id_to_envs, rllib_env_id_to_envs


exp_to_config = {
    # all of the 106 atari environments + hyperparameters. This will take a long time to train.
    "atari-v0": "dqn-atari-full-sweep.yml",
    # three of the atari environments
    "atari-mini": "dqn-atari-mini-sweep.yml",
    # pong only:
    "atari-pong": "dqn-atari-pong-sweep.yml"
}
int_hparams = {'batch_size', 'buffer_size', 'gradient_steps',
               'target_update_interval'}
device = None


def main(total_steps=1_200_000, hparams:dict=None):
    env_id = hparams.pop('env_id')
    env, eval_env = env_id_to_envs(env_id, render=False, is_atari=True, permute_dims=True)
    # rename to match stable baselines3:
    hparams['gamma'] = hparams.pop('discount_factor')
    hparams['exploration_fraction'] = hparams.pop('exploration_final_eps_frame') / total_steps
    # optimizer = torch.optim.RMSprop
    model = DQN('CnnPolicy', env, verbose=1, device='cuda',
                policy_kwargs={
                    'normalize_images': False,
                }, **hparams, tensorboard_log='ft/PongNoFrameskip-v4')

    model.learn(total_timesteps=total_steps, log_interval=10)


def wandb_train(local_cfg=None):
    """:param local: run wandb locally"""
    wandb_kwargs = {"project": project, "group": experiment_name}
    if local_cfg:
        local_cfg["controller"] = {'type': 'local'}
        wandb_kwargs['config'] = local_cfg
        hparams = dict()
        for k, v in local_cfg['parameters'].items():
            hparams[k] = v['values'][0]
        local_cfg['parameters'] = hparams
        print(f"dqn parameters: {local_cfg['parameters']}")
    with wandb.init(**wandb_kwargs, sync_tensorboard=True) as run:
        config = wandb.config.as_dict()
        main(total_steps=10_000_000, hparams=config['parameters'])


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--sweep", type=str, default=None)
    args.add_argument("--n_runs", type=int, default=10)
    args.add_argument("--proj", type=str, default="u-chi-learning-test")
    args.add_argument("--local-wandb", type=bool, default=True)
    args.add_argument("--exp-name", type=str, default="atari-pong")
    args.add_argument("--device", type=str, default='cuda')
    args = args.parse_args()
    project = args.proj
    experiment_name = args.exp_name
    device = args.device
    # load the default config
    with open("sweep_params/dqn-atari-default.yml", "r") as f:
        default_config = yaml.load(f, yaml.SafeLoader)
    # generate a new sweep if one was not passed as an argument
    if args.sweep is None and not args.local_wandb:
        sweep_id = wandb.sweep(default_config, project=project)
        print(f"created new sweep {sweep_id}")
        wandb.agent(sweep_id, project=args.proj,
                    count=args.n_runs, function=wandb_train)
    elif args.local_wandb:
        for i in range(args.n_runs):
            try:
                print(f"running local sweep {i}")
                wandb_train(local_cfg=copy.deepcopy(default_config))
            except Exception as e:
                print(f"failed to run local sweep {i}")
                print(e)
    else:
        print(f"continuing sweep {args.sweep}")
        wandb.agent(args.sweep, project=args.proj,
                    count=args.n_runs, function=wandb_train)
