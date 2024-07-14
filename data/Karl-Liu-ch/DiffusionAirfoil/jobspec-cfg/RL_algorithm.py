from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from AgentNet import AttnBlock
import numpy as np
import platform
from option import opt
from SB3_env import *
from utils import *
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import os

os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_id
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if platform.system().lower() == 'linux':
    path = '/work3/s212645/DiffusionAirfoil/checkpoint/'
elif platform.system().lower() == 'windows':
    path = 'H:/深度学习/checkpoint/'
base_airfoil = np.loadtxt('BETTER/20150114-50 +2 d.dat', skiprows=1)
base_airfoil = interpolate(base_airfoil, 256, 3)

class Net(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(Net, self).__init__(observation_space, features_dim)
        n_input_channels = 1
        self.cnn = nn.Sequential(
            nn.Linear(n_input_channels, features_dim), 
            AttnBlock(features_dim, 1, 1),
        )

        self.linear = nn.Sequential(nn.Linear(features_dim, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations.permute(0,2,1))).mean(dim=1)

policy_kwargs = dict(
    features_extractor_class=Net,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[dict(pi=[256, 256], vf=[256, 256])]
)
# env = OptimEnv()
env = AirfoilEnv()

if opt.agent == 'ppo':
    # model = PPO("MlpPolicy", env, batch_size=256, verbose=1)
    model = PPO("MlpPolicy", env, batch_size=256, policy_kwargs=policy_kwargs, verbose=1)
    path = '/work3/s212645/DiffusionAirfoil/PPO/stablebaseline_ppo'
    try:
        model.load(path)
    except Exception as e:
        print(e)
    model.learn(total_timesteps=100_000)
    model.save(path)

elif opt.agent == 'sac':
    policy_kwargs = dict(
        features_extractor_class=Net,
        features_extractor_kwargs=dict(features_dim=256)
    )
    model = SAC("MlpPolicy", env, learning_rate=1e-4, verbose=1)
    path = '/work3/s212645/DiffusionAirfoil/PPO/sac'
    # model = SAC("MlpPolicy", env, learning_rate=1e-4, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save(path)

elif opt.agent == 'td3':
    policy_kwargs = dict(
        features_extractor_class=Net,
        features_extractor_kwargs=dict(features_dim=256)
    )
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    path = '/work3/s212645/DiffusionAirfoil/PPO/td3'
    # model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model = TD3("MlpPolicy", env, policy_kwargs=policy_kwargs, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=10000, log_interval=10)
    model.save(path)