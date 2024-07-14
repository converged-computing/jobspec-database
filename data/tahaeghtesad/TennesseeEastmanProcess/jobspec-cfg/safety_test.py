import logging
import os
import sys

import gym
import tensorflow as tf

from agents.RLAgents import ZeroAgent, SimpleWrapperAgent
from envs.control.envs.safety import SafetyEnvAttacker, SafetyEnvDefender
from envs.control.threat.safety_threat import SafetyThreatModel
from safety_gym.envs.engine import Engine
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ppo2 import PPO2

temp_path = os.environ['TMPDIR']


def make_vec_env(env_id, n_envs=1, seed=None, start_index=0,
                 monitor_dir=None, wrapper_class=None,
                 env_kwargs=None, vec_env_cls=None, vec_env_kwargs=None):
    """
    Create a wrapped, monitored `VecEnv`.
    By default it uses a `DummyVecEnv` which is usually faster
    than a `SubprocVecEnv`.

    :param env_id: (str or Type[gym.Env]) the environment ID or the environment class
    :param n_envs: (int) the number of environments you wish to have in parallel
    :param seed: (int) the initial seed for the random number generator
    :param start_index: (int) start rank index
    :param monitor_dir: (str) Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: (gym.Wrapper or callable) Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: (dict) Optional keyword argument to pass to the env constructor
    :param vec_env_cls: (Type[VecEnv]) A custom `VecEnv` class constructor. Default: None.
    :param vec_env_kwargs: (dict) Keyword arguments to pass to the `VecEnv` class constructor.
    :return: (VecEnv) The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs

    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id)
                if len(env_kwargs) > 0:
                    logging.getLogger(__name__).warning("No environment class was passed (only an env ID) so `env_kwargs` will be ignored")
            else:
                env = env_id(**env_kwargs)

            if hasattr(env, 'config'):
                config = env.config
                config['placements_extents'] = [-2.0, -2.0, 2.0, 2.0]
                config['lidar_max_dist'] = 8 * config['placements_extents'][3]
                env = Engine(config)

            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            # env = Monitor(env)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env)
            return env
        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


def get_policy_class(policy_params):
    # policy params must have 'act_fun' and 'layers'
    class CustomPolicy(MlpPolicy):
        def __init__(self, *args, **_kwargs):
            super().__init__(*args, **_kwargs, **policy_params)

    return CustomPolicy


def train_attacker(name, env_name, defender):
    env = make_vec_env(SafetyEnvAttacker, env_kwargs=dict(env=env_name, defender=defender), n_envs=2)

    if os.path.isfile(f'{base_model_path}/{name}.zip'):
        model = PPO2.load(f'{base_model_path}/{name}.zip', env=env)
    else:
        model = PPO2(
            policy=get_policy_class(dict(
                net_arch=[dict(vf=[128, 64],
                               pi=[128, 128])],
                act_fun=tf.nn.tanh
            )),
            env=env,
            gamma=0.995,
            verbose=1,
            lam=0.97,
            tensorboard_log=f'{temp_path}/tb_logs/',
            full_tensorboard_log=True
        )

        model.learn(
            total_timesteps=train_length,
            callback=callback,
            tb_log_name=f'{name}'
        )
    model.save(f'{base_model_path}/{name}')
    agented_model = SimpleWrapperAgent(name, model)
    agent_reward = eval_agents(env_name, agented_model, defender)[0]
    print(f'Agent {name} evaluated: {agent_reward:.4f}')
    return agented_model


def train_defender(name, env_name, attacker):
    env = make_vec_env(SafetyEnvDefender, env_kwargs=dict(env=env_name, attacker=attacker), n_envs=2)

    if os.path.isfile(f'{base_model_path}/{name}.zip'):
        model = PPO2.load(f'{base_model_path}/{name}.zip', env=env)
    else:
        model = PPO2(
            policy=get_policy_class(dict(
                net_arch=[dict(vf=[128, 64],
                               pi=[64, 64])],
                act_fun=tf.nn.tanh
            )),
            env=env,
            gamma=0.995,
            verbose=1,
            lam=0.97,
            tensorboard_log=f'{temp_path}/tb_logs/',
            full_tensorboard_log=True
        )

        model.learn(
            total_timesteps=train_length,
            callback=callback,
            tb_log_name=f'{name}'
        )
        model.save(f'{base_model_path}/{name}')
    agented_model = SimpleWrapperAgent(name, model)
    agent_reward = eval_agents(env_name, attacker, agented_model)[1]
    print(f'Agent {name} evaluated: {agent_reward:.4f}')
    return agented_model


def train_nominal(name, env):
    env = make_vec_env(env, n_envs=2)

    if os.path.isfile(f'{base_model_path}/{name}.zip'):
        model = PPO2.load(f'{base_model_path}/{name}.zip', env=env)
    else:
        model = PPO2(
            policy=get_policy_class(dict(
                net_arch=[dict(vf=[128, 64],
                               pi=[64, 64])],
                act_fun=tf.nn.tanh
            )),
            env=env,
            gamma=0.995,
            verbose=1,
            lam=0.97,
            tensorboard_log=f'{temp_path}/tb_logs/',
            full_tensorboard_log=True
        )

        model.learn(
            total_timesteps=train_length,
            callback=callback,
            tb_log_name=f'{name}'
        )
        model.save(f'{base_model_path}/{name}')
    agented_model = SimpleWrapperAgent(name, model)
    agent_reward = eval_agents(env_name, ZeroAgent('Zero', 2), agented_model)[1]
    print(f'Agent {name} evaluated: {agent_reward:.4f}')
    return agented_model


def callback(locals_, globals_):
    self_ = locals_['self']
    if 'rewards' in locals_:
        if 'writer' in locals_ and locals_['writer'] is not None:
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=f'env/reward', simple_value=locals_['rewards'][0])])
            locals_['writer'].add_summary(summary, self_.model.num_timesteps)
    return True


def eval_agents(env, attacker, defender):
    epochs = 25
    print(f'Evaluating {attacker} vs. {defender} in environment {env} for {epochs} epoch(s).')
    env = SafetyThreatModel(env, attacker, defender)

    defender_rewards = []
    attacker_rewards = []
    lengths = []

    for epoch in range(epochs):
        done = False
        _ = env.reset()
        adversary_episode_reward = 0
        defender_episode_reward = 0
        step = 0
        while not done:
            _, (r_a, r_d), done, _ = env.step()
            adversary_episode_reward += r_a
            defender_episode_reward += r_d
            step += 1
        defender_rewards.append(defender_episode_reward)
        attacker_rewards.append(adversary_episode_reward)
        lengths.append(step)

    print(f'Average episode length: {sum(lengths)/len(lengths)}')
    print(f'Average episode reward for adversary: {sum(attacker_rewards) / len(attacker_rewards)}')
    print(f'Average episode reward for defender: {sum(defender_rewards) / len(defender_rewards)}')
    print(f'Average step reward for adversary: {sum(attacker_rewards) / sum(lengths)}')
    print(f'Average step reward for defender: {sum(defender_rewards) / sum(lengths)}')

    return sum(attacker_rewards)/len(attacker_rewards), sum(defender_rewards)/len(defender_rewards)


if __name__ == '__main__':

    robot = 'Car'
    env_name = f'Safexp-{robot}Goal0-v0'
    base_model_path = 'lee-models'
    repeat = 5
    train_length = 300_000

    if not os.path.isdir(base_model_path):
        os.makedirs(base_model_path, exist_ok=True)

    slurm_id = int(sys.argv[1])

    nominal = train_nominal(f'nominal-{slurm_id}', env_name)
    att1 = train_attacker(f'adversary-1-{slurm_id}', env_name, nominal)
    robust = train_defender(f'robust-{slurm_id}', env_name, att1)
    att2 = train_attacker(f'adversary-2-{slurm_id}', env_name, robust)
