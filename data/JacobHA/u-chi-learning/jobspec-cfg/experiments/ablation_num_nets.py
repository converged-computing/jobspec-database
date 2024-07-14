import sys
sys.path.append('darer')
import argparse
from CustomDQN import CustomDQN
from CustomPPO import CustomPPO
# from LogU import LogULearner
from UAgent import UAgent
from hparams import *
import time

# env = 'CartPole-v1'
# env = 'LunarLander-v2'
env = 'Acrobot-v1'
# env = 'MountainCar-v0'


def runner(device):
    if env == 'MountainCar-v0':
        configs = mcars
    elif env == 'CartPole-v1':
        configs = cartpoles
    elif env == 'LunarLander-v2':
        configs = lunars
    elif env == 'Acrobot-v1':
        configs = acrobots
    else:
        raise ValueError(f"env {env} not recognized.")

    # Now access the config for this algo:
    config = configs['u']

    ppi_hparams = {'use_ppi': False,
                        }

    model = UAgent(env, **config, tensorboard_log=f'experiments/ablations/{env}',
                 device=device, log_interval=1000, **ppi_hparams,
                 name=f'{NUM_NETS}nets',
                 num_nets=NUM_NETS
                 )#, aggregator='max')
    model.learn(total_timesteps=50_000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count', type=int, default=10)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-n', '--num_nets', type=int, default=2)
    args = parser.parse_args()

    NUM_NETS = args.num_nets

    start = time.time()
    for i in range(args.count):
        runner(args.device)
        print(f"Finished run {i+1}/{args.count}")
    print(f"trained in {time.time() - start}")
