import torch
import numpy as np
import argparse
import os
import sys
import yaml
import time
import stable_baselines3 as sb3
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed
import scipy.stats as ss
import scipy
import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def kl_scipy(p, q):
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    p = p.flatten()
    q = q.flatten()
    p[p == 0] = np.finfo(float).eps
    q[q == 0] = np.finfo(float).eps
    if (len(p) > 1):
        # pg = ss.gaussian_kde(p)
        # qg = ss.gaussian_kde(q)
        # kl = scipy.stats.entropy(pg(p), qg(q))
        # print("p,q", scipy.stats.entropy(pg(p), qg(q)))
        print("len of p", len(p))
        # return kl
        return
    else:
        return 0


def Q(W, n):
    w_old = W
    if n >= 32:
        return W, 0
    assert (len(W.shape) <= 2)
    range = np.abs(np.min(W)) + np.abs(np.max(W))
    d = range / (2 ** (n))
    z = -np.min(W, 0) // d
    W = np.rint(W / d)
    W = d * (W)

    kl = kl_scipy(w_old, W)

    # plot histogram
    import matplotlib.pyplot as plt

    if not os.path.exists("weights-histogram"):
        os.mkdir("weights-histogram")
    # plt.hist(w_old.flatten(), bins=100)
    # plt.hist(W.flatten(), bins=100)

    # save plot
    # plt.savefig('weights-histogram/histogram_{}bit.png'.format(n))
    return W, kl


def conv_Q(W, n):
    if n >= 32:
        return W
    w_old = W
    newweight = np.zeros_like(W)
    # print(W.shape)
    for i in range(W.shape[-1]):
        range_i = np.abs(np.min(W[:, :, :, i])) + np.abs(np.max(W[:, :, :, i]))
        d = range_i / (2 ** (n))
        z = -np.min(W[:, :, :, i], 0) // d
        temp = np.rint(W[:, :, :, i] / d)
        newweight[:, :, :, i] += d * temp
    kl = kl_scipy(w_old, newweight)
    return newweight, kl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize-choice", type=str, default="", choices=["base", "HERO", "SAM"])
    parser.add_argument('--rho', default=0.05, type=float, help='rho of SAM')
    parser.add_argument("--quantized", help="quantization bit", type=int, default=32)
    parser.add_argument("--env", help="environment ID", type=EnvironmentName, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False,
                        choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=1, type=int)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument(
        "--norm-reward", action="store_true", default=False,
        help="Normalize reward if applicable (trained with VecNormalize)"

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

    args = parser.parse_args()

    env_name: EnvironmentName = args.env
    algo = args.algo
    folder = args.folder
    q = args.quantized

    try:
        _, model_path, log_path = get_model_path(
            args.optimize_choice,
            args.quantized,
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
                args.optimize_choice,
                args.quantized,
                folder,
                algo,
                env_name,
                args.load_best,
                args.load_checkpoint,
                args.load_last_checkpoint,
            )

    print(f"Loading {model_path}")

    set_random_seed(args.seed)

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
    # # overwrite with command line arguments
    # if args.env_kwargs is not None:
    #     env_kwargs.update(args.env_kwargs)
    model = ALGOS[algo].load(rho=args.rho, quantized=32,
                             path=model_path)  # PTQ loaded model no need to add fake quantization module in the network,so keep passing 32 bit the model
    data = model.get_parameters()  # like data.keys() : ['policy','policy.optimizer']  critics (value functions) and policies (pi functions).

    # start quantization process
    kl_array = []  # log KL of each layers
    for key in data.keys():
        if key == 'policy.optimizer':
            pass
        else:  # policy
            for param_key in data[key].keys():
                print('[', key, ']', '[', param_key, ']')
                if 'cnn' in param_key and 'weight' in param_key:  # like 'pi_features_extractor.cnn.2.weight'
                    if q == 16:
                        data[key][param_key] = data[key][param_key].cpu().numpy().astype(np.float16).astype(
                            np.float32)
                        data[key][param_key] = torch.from_numpy(data[key][param_key])
                    else:
                        data[key][param_key], kl = conv_Q(data[key][param_key].cpu().numpy(), q)
                        kl_array.append(kl)
                        try:
                            data[key][param_key] = torch.from_numpy(data[key][param_key])
                        except:
                            data[key][param_key] = torch.tensor(data[key][param_key])
                elif ('cnn' in param_key and 'bias' in param_key) or (  # like 'pi_features_extractor.cnn.2.bias'
                        'action' in param_key) or 'weight' in param_key or 'bias' in param_key:
                    if q == 16:
                        data[key][param_key] = data[key][param_key].cpu().numpy().astype(np.float16).astype(
                            np.float32)
                        data[key][param_key] = torch.from_numpy(data[key][param_key])
                    else:
                        data[key][param_key], kl = Q(data[key][param_key].cpu().numpy(), q)
                        kl_array.append(kl)
                        try:
                            data[key][param_key] = torch.from_numpy(data[key][param_key])
                        except:
                            data[key][param_key] = torch.tensor(data[key][param_key])

    save_path = 'quantized/{}/{}/{}'.format(q, algo, folder.split('_')[-1])
    os.makedirs(save_path, exist_ok=True)
    model.policy.load_state_dict(data['policy'])
    model.save(save_path + '/{}.zip'.format(env_name))

    print("model has succussfully saved to {}".format(save_path + '/{}.zip'.format(env_name)))
    # print("Kl divergence: ", sum(kl_array))
