import argparse
import os
import sys
import ast

from jax.lib import xla_bridge
import wandb
import os.path as osp

from benchmark import benchmark, sweep
from meta_train import meta_train
import tensorflow as tf

from mmengine.config import Config


def parse_args():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_type", type=str, choices=["benchmark", "meta-train","sweep"])
    parser.add_argument("--optimizer", type=str, choices=["sgd",
                                                          "adam", 
                                                          "fedavg", 
                                                          "fedavg-slowmo", 
                                                          "fedlopt", 
                                                          "fedlopt-adafac",
                                                          "delay-adafac",
                                                          "mlp",
                                                          "delay-mlp",
                                                          "fedlagg", 
                                                          "fedlagg-wavg", 
                                                          "fedlagg-adafac"])
    parser.add_argument("--task", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--local_learning_rate", type=float)
    parser.add_argument("--local_batch_size", type=int)
    parser.add_argument("--num_grads", type=int)
    parser.add_argument("--num_local_steps", type=int)
    parser.add_argument("--num_runs", type=int)
    parser.add_argument("--num_inner_steps", type=int)
    parser.add_argument("--num_outer_steps", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--sweep_config", type=str)
    parser.add_argument("--from_checkpoint", action="store_true")
    parser.add_argument("--test_checkpoint", type=str)
    parser.add_argument("--use_pmap", action="store_true")
    parser.add_argument("--num_devices", type=int)
    parser.add_argument("--num_tasks", type=int)
    parser.add_argument("--name_suffix", type=str)
    parser.add_argument("--slowmo_learning_rate", type=float)
    parser.add_argument("--wandb_checkpoint_id", type=str)
    parser.add_argument("--meta_loss_split", type=str)
    parser.add_argument("--test_project", type=str)
    parser.add_argument("--tfds_data_dir", type=str, default="./") # os.getenv("SLURM_TMPDIR")
    parser.add_argument("--wandb_dir", type=str, default=os.getenv("SCRATCH"))
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--truncation_schedule_min_length", type=int)
    parser.add_argument("--sweep_id", type=str)
    parser.add_argument("--needs_state", action="store_true")

    parser.add_argument("--delay", type=int, default=4)
    parser.add_argument("--delay_optim_test", action="store_true")
    #parser.add_argument("--delay_features", type=int, default=0) #Using additional specific delay features
    parser.add_argument('--delay_features', type=str, default='[0]', help='List of additional delay features')
    parser.add_argument("--delayed_compensation_method", type=str, default='None', choices=["None",
                                                                          "DC",
                                                                          "DC-diag",
                                                                          "SA",
                                                                          "GA"])
    parser.add_argument("--weight_prediction", action="store_true")     
    parser.add_argument("--momentum", type=float, default=0.0)


    parser.add_argument("--eta", type=float, default=1.0) #eta in ratio for now

    parser.add_argument("--slurm_id", type=int, default=-1)


    # fmt: on

    return parser.parse_args()


def assert_args(args):
    # fmt: off
    if args.run_type == "benchmark" and args.optimizer in ["fedlopt", "fedlopt-adafac", "fedlagg", "fedlagg-wavg", "fedlagg-adafac"]:
        assert os.path.exists(args.test_checkpoint), "need to meta-train learned optimizer before benchmarking"
        assert args.test_checkpoint.endswith('.pickle'), "optimizer checkpoints must be saved as .pickle files"
    if args.run_type == "meta-train":
        assert args.optimizer not in ["adam", "fedavg", "fedavg-slowmo"], "can't meta-train a non learned optimizer"
    # fmt: on


def download_wandb_checkpoint(cfg):
    api = wandb.Api()
    run = api.run(cfg.wandb_checkpoint_id)

    ckpts = [x for x in run.files() if "global_step" in x.name]
    if len(ckpts) > 1:
        print(ckpts)

    assert (
        len(ckpts) <= 1
    ), "multiple checkpoints exist can't determine which one to use"

    if len(ckpts) == 0:
        return None

    ckpts[0].download("/tmp", replace=True)
    return osp.join("/tmp", ckpts[0].name)


if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], "GPU")

    print(xla_bridge.get_backend().platform)

    args = parse_args()
    
    d_f_string = args.delay_features
    args.delay_features = ast.literal_eval(d_f_string)
    args.delay_features = [int(i) for i in args.delay_features]
    print("delay features", args.delay_features)

    sys.path.append(os.getcwd())
    os.environ["TFDS_DATA_DIR"] = args.tfds_data_dir
    os.environ["WANDB_DIR"] = args.wandb_dir
    print("wb dir", os.environ["WANDB_DIR"])
    
    #print('args', args)

    cfg = Config.fromfile(args.config)

    if args.slurm_id != -1:
        lr_list = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
        args.learning_rate = lr_list[args.slurm_id]

    
    # override args from the command line
    for k, v in vars(args).items():
        if v is not None:
            print("[INFO] Overriding config value: {}={}".format(k, v))
            cfg._cfg_dict[k] = v

    cfg.name = "{}{}_{}{}".format(
        cfg.optimizer, cfg.hidden_size, cfg.task, cfg.name_suffix
    )
    cfg.meta_train_name = "{}{}_{}_K{}_H{}_{}{}_feat{}_delay{}".format(
        cfg.optimizer,
        cfg.hidden_size,
        cfg.task,
        cfg.num_grads,
        cfg.num_local_steps,
        cfg.local_learning_rate,
        cfg.name_suffix,
        d_f_string,
        cfg.delay
    )

    if cfg.wandb_checkpoint_id is not None:
        cfg.test_checkpoint = download_wandb_checkpoint(cfg)

    args = argparse.Namespace(**cfg._cfg_dict)

    assert_args(args)

    print('args after', args)
    print('cfg after', cfg)

    run_types = {"benchmark": benchmark, "meta-train": meta_train, "sweep": sweep}

    run_types[args.run_type](args)
