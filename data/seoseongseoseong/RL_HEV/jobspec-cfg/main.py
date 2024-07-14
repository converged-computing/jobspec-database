import os
import sys
from glob import glob
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
from fmpy.util import plot_result, download_test_file
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import random
import argparse
from collections import deque
from tqdm import tqdm
import scipy.signal
import time
import random
import copy
import gymnasium as gym
from gymnasium import spaces
import wandb

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import results_plotter
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import MultivariateNormal
from torch.distributions.normal import Normal
from torch import FloatTensor as FT
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader

import tensorflow

from env import *
# from utils import *

os.system('cls')

# parser = argparse.ArgumentParser(description='Train SAC model with specified parameters')
# parser.add_argument('--ent_coef', type=str, default='auto_1e-5', help='entropy coefficient')
# parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
# parser.add_argument('--batch_size', type=int, default=16, help='batch size')
# parser.add_argument('--tau', type=float, default=0.005, help='tau value')
# args = parser.parse_args()

project = 'SAC_0423'
name = 'SAC'
cwd = os.getcwd()
print(cwd)

print("cpu available:", os.cpu_count())

# GPU 준비
USE_CUDA = torch.cuda.is_available()
dev = torch.device("cuda:0" if USE_CUDA else "cpu")
print("Using Device:", dev)
torch.cuda.empty_cache()

seed_everything(seed=42)

fmu_filename = 'HEV_TMED_Simulator_WLTC_231005_Check.fmu'
fmu_name = 'HEV_TMED_Simulator_WLTC_231005_Check'
log = True
test_learn = True
test_eval = True
start_time = 0.0
stop_time = 1800.0
step_size = 1
soc_init = 67
limitation_coeff = 1
SoC_coeff = 10
BSFC_coeff = 0.1
NOx_coeff = 0.1
reward_coeff = 1
state_coeff = 1
alive_reward = 1
profile_name = 'wltp_1Hz.csv'
buffer_size = 50000
ent_coef= 'auto_1e-5'
learning_starts = 100
tau = 0.005
gamma = 0.99
learning_rate = 5e-5
batch_size = 256
save_dir = f'./model/{project}/'
monitor_dir = f'./monitor/{project}/'
checkpoints_dir = f'./checkpoints/{project}/'
log_dir = f'./logs/{project}/'
board_dir = f'./board/{project}/'
num_cpu = 1 #os.cpu_count()
episodes = 10000
total_timesteps = int(stop_time)*1
env_id = "HEV"
config = {"gamma": gamma, "batch_size": batch_size, "learning_rate": learning_rate}

vec_env = DummyVecEnv([make_env(fmu_filename, log, test_learn, start_time, step_size, limitation_coeff, SoC_coeff, BSFC_coeff, NOx_coeff, reward_coeff, state_coeff, alive_reward, 
                                project, name, config, monitor_dir, i) for i in range(num_cpu)])
# vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0, gamma=gamma, epsilon=1e-08)
checkpoint_callback = CheckpointCallback(save_freq=1800, save_path=checkpoints_dir, name_prefix="rl_model")
env = HEV(fmu_filename=fmu_filename, log=log, test=test_eval, start_time=start_time, step_size=step_size, 
            limitation_coeff=limitation_coeff, SoC_coeff=SoC_coeff, BSFC_coeff=BSFC_coeff, NOx_coeff=NOx_coeff, reward_coeff=reward_coeff, state_coeff=state_coeff, alive_reward=alive_reward, 
            project=project, name=name, config=config)
eval_env = DummyVecEnv([lambda: Monitor(env, monitor_dir+f"_eval")])
# eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0, gamma=gamma, epsilon=1e-08)
eval_callback = EvalCallback(eval_env, best_model_save_path=save_dir, log_path=log_dir, eval_freq=stop_time*1000, deterministic=True, render=False)

model = SAC("MlpPolicy", vec_env, buffer_size=buffer_size, learning_starts=learning_starts, ent_coef=ent_coef, verbose=1, device="cuda", batch_size=batch_size, tau=tau, gamma=gamma, learning_rate=learning_rate, tensorboard_log=board_dir)
# model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=1, device="cuda", batch_size=batch_size, gamma=gamma, learning_rate=learning_rate)

model.learn(total_timesteps = episodes*total_timesteps, callback = [checkpoint_callback, eval_callback])
del model


best_model = SAC.load(os.path.join(save_dir, "best_model"))
mean_reward, std_reward = evaluate_policy(best_model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")





# state = env.reset()
# action = np.array([0,0])
# done = False
# cum_reward = 0.0
# a_record = []
# r_record = []
# s_record = []
# t=0
# while not done:
#     next_state, reward, done, _, info = env.step(action)
#     cum_reward += reward
#     # a_record.append(np.array(action))
#     r_record.append(reward)
#     s_record.append(next_state)
#     state = next_state
# print(f"total reward: {cum_reward}")
# s_record = np.array(s_record)
# print(np.sum(np.log(1 - abs(0.67 - s_record[:,0]))))
# print(np.sum(0.1*s_record[:,1]))
# print(np.sum(s_record[:,2]))
# plt.plot(np.log(1 - abs(0.67 - s_record[:,0])), label='SoC')
# plt.plot(0.1*s_record[:,1], label='BSFC')
# plt.plot(s_record[:,2]/100, label='NOx_mgpkm')
# plt.plot(s_record[:,4], label='NOx_AFT')
# # plt.plot(r_record)
# plt.legend(loc='best')
# plt.show()