import argparse
import importlib
import os
import sys

import numpy as np
import torch as th
import wandb
import time
import yaml
from huggingface_sb3 import EnvironmentName
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed

import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy


def collate() -> None:  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', default=0.0, type=float)
    parser.add_argument('--rho', default=0.05, type=float, help='rho of SAM')
    parser.add_argument("--optimize-choice", type=str, default="", choices=["base", "HERO", "SAM"])
    parser.add_argument("--env", help="environment ID", type=EnvironmentName, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False,
                        choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
             "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False,
        help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict,
        help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues"
    )
    parser.add_argument(
        "--track",
        action="store_true",
        default=False,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "-P",
        "--progress",
        action="store_true",
        default=False,
        help="if toggled, display a progress bar using tqdm and rich",
    )
    parser.add_argument("--wandb-project-name", type=str, default="", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument(
        "-tags", "--wandb-tags", type=str, default=[], nargs="+",
        help="Tags for wandb run, e.g.: -tags optimized pr-123"
    )
    args = parser.parse_args()

    rewards = []
    lengths = []
    if args.track:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
            ) from e
        if args.learning_rate != 0:
            run_name = f"PTQ_{args.env}_{args.algo}_{args.optimize_choice}_lr{args.learning_rate}_rho{args.rho}_time{int(time.time())}"
        else: # using suggested learning_rate
            run_name = f"PTQ_{args.env}_{args.algo}_{args.optimize_choice}_SuggestedLR_rho{args.rho}_time{int(time.time())}"
        if args.wandb_project_name:
            wandb_project_name = args.wandb_project_name
        else:
            wandb_project_name = "PTQ" + "_" + args.algo + "_" + args.env
        tags = [*args.wandb_tags, f"v{sb3.__version__}"]
        run = wandb.init(
            name=run_name,
            project=wandb_project_name,
            entity=args.wandb_entity,
            tags=tags,
            config=vars(args),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        args.tensorboard_log = f"runs/{run_name}"
    # set the bit list to iterate for reward of model
    bits_PTQ = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    for bit in bits_PTQ:
        optimize_choice = args.optimize_choice
        env_name: EnvironmentName = args.env
        algo = args.algo
        folder = args.folder + f"/{bit}"

        try:
            _, model_path, log_path = get_model_path(
                optimize_choice,
                bit,
                args.exp_id,
                folder,
                algo,
                env_name,
                args.load_best,
                args.load_checkpoint,
                args.load_last_checkpoint,
            )
        except (AssertionError, ValueError) as e:
            # Special case for rl-trained agents
            # auto-download from the hub
            if "rl-trained-agents" not in folder:
                raise e
            else:
                print(
                    "Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
                # Auto-download
                download_from_hub(
                    algo=algo,
                    env_name=env_name,
                    exp_id=args.exp_id,
                    folder=folder,
                    organization="sb3",
                    repo_name=None,
                    force=False,
                )
                # Try again
                _, model_path, log_path = get_model_path(
                    args.exp_id,
                    folder,
                    algo,
                    env_name,
                    args.load_best,
                    args.load_checkpoint,
                    args.load_last_checkpoint,
                )

        print(f"Loading {model_path}")

        # Off-policy algorithm only support one env for now
        off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

        if algo in off_policy_algos:
            args.n_envs = 1

        set_random_seed(args.seed)

        if args.num_threads > 0:
            if args.verbose > 1:
                print(f"Setting torch.num_threads to {args.num_threads}")
            th.set_num_threads(args.num_threads)

        is_atari = ExperimentManager.is_atari(env_name.gym_id)
        is_minigrid = ExperimentManager.is_minigrid(env_name.gym_id)

        stats_path = os.path.join(log_path, env_name)
        hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

        # load env_kwargs if existing
        env_kwargs = {}
        args_path = os.path.join(log_path, env_name, "args.yml")
        if os.path.isfile(args_path):
            with open(args_path) as f:
                loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
                if loaded_args["env_kwargs"] is not None:
                    env_kwargs = loaded_args["env_kwargs"]
        # overwrite with command line arguments
        if args.env_kwargs is not None:
            env_kwargs.update(args.env_kwargs)

        log_dir = args.reward_log if args.reward_log != "" else None

        env = create_test_env(
            env_name.gym_id,
            n_envs=args.n_envs,
            stats_path=maybe_stats_path,
            seed=args.seed,
            log_dir=log_dir,
            should_render=not args.no_render,
            hyperparams=hyperparams,
            env_kwargs=env_kwargs,
        )

        kwargs = dict(seed=args.seed)
        if algo in off_policy_algos:
            # Dummy buffer size as we don't need memory to enjoy the trained agent
            kwargs.update(dict(buffer_size=1))
            # Hack due to breaking change in v1.6
            # handle_timeout_termination cannot be at the same time
            # with optimize_memory_usage
            if "optimize_memory_usage" in hyperparams:
                kwargs.update(optimize_memory_usage=False)

        # Check if we are running python 3.8+
        # we need to patch saved model under python 3.6/3.7 to load them
        newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

        custom_objects = {}
        if newer_python_version or args.custom_objects:
            custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            }

        if "HerReplayBuffer" in hyperparams.get("replay_buffer_class", ""):
            kwargs["env"] = env

        model = ALGOS[algo].load(args.rho, bit, model_path, custom_objects=custom_objects, device=args.device,
                                 **kwargs)
        obs = env.reset()

        # Deterministic by default except for atari games
        stochastic = args.stochastic or (is_atari or is_minigrid) and not args.deterministic
        deterministic = not stochastic

        episode_reward = 0.0
        episode_rewards, episode_lengths = [], []
        ep_len = 0
        # For HER, monitor success rate
        successes = []
        lstm_states = None
        episode_start = np.ones((env.num_envs,), dtype=bool)

        generator = range(args.n_timesteps)
        if args.progress:
            if tqdm is None:
                raise ImportError("Please install tqdm and rich to use the progress bar")
            generator = tqdm(generator)

        try:
            for i in generator:
                action, lstm_states = model.predict(
                    obs,  # type: ignore[arg-type]
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=deterministic,
                )
                obs, reward, done, infos = env.step(action)
                print(f"step:{i},  reward:{reward},  state:{done}")
                episode_start = done

                if not args.no_render:
                    env.render("human")

                episode_reward += reward[0]
                ep_len += 1

                if args.n_envs == 1:
                    # For atari the return reward is not the atari score
                    # so we have to get it from the infos dict
                    if is_atari and infos is not None and args.verbose >= 1:
                        episode_infos = infos[0].get("episode")
                        if episode_infos is not None:
                            print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                            print("Atari Episode Length", episode_infos["l"])

                    if done and not is_atari and args.verbose > 0:
                        # NOTE: for env using VecNormalize, the mean reward
                        # is a normalized reward when `--norm_reward` flag is passed
                        print(f"Episode Reward: {episode_reward:.2f}")
                        print("Episode Length", ep_len)
                        episode_rewards.append(episode_reward)
                        episode_lengths.append(ep_len)
                        episode_reward = 0.0
                        ep_len = 0

                    # Reset also when the goal is achieved when using HER
                    if done and infos[0].get("is_success") is not None:
                        if args.verbose > 1:
                            print("Success?", infos[0].get("is_success", False))

                        if infos[0].get("is_success") is not None:
                            successes.append(infos[0].get("is_success", False))
                            episode_reward, ep_len = 0.0, 0

        except KeyboardInterrupt:
            pass
        # evaluate the policy
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
        print("mean_reward:", mean_reward, "std_reward:", std_reward)

        if args.verbose > 0 and len(successes) > 0:
            print(f"Success rate: {100 * np.mean(successes):.2f}%")

        if args.verbose > 0 and len(episode_rewards) > 0:
            print(f"{len(episode_rewards)} Episodes")
            print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

        if args.verbose > 0 and len(episode_lengths) > 0:
            print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

        rewards.append(mean_reward)
        if args.track:
            wandb.log({"PTQ/reward": mean_reward}, step=bit)

        env.close()
    if not os.path.exists('pngs'):
        os.mkdir("pngs")
    if not os.path.exists('pngs/PTQ'):
        os.mkdir("pngs/PTQ")
    png_path = "pngs/PTQ/"

    plt.title("PTQ reward")
    plt.plot(bits_PTQ, rewards, color='g', linestyle='-', linewidth=2, marker="o")
    plt.xlabel("bit")
    plt.ylabel('reward')
    plt.savefig(png_path + "outcome_{}_{}_{}_lr{}_rho{}.png".format(args.algo, args.env, args.optimize_choice, args.learning_rate, args.rho))


if __name__ == "__main__":
    collate()
