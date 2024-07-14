import copy
import functools
import json
import os
import shutil

from ray import air, tune
from ray.tune.search.basic_variant import BasicVariantGenerator
import multiprocessing

from argparse import ArgumentParser
import time
import torch
import numpy as np
import ray
from gflownet.utils.misc import replace_dict_key, change_config, get_num_cpus
from gflownet.algo.config import TBVariant
from ray.tune.schedulers import ASHAScheduler

# Global main
from gflownet.tasks.main import main


def run_raytune(search_space, ray_dir):
    if os.path.exists(ray_dir):
        if search_space["overwrite_existing_exp"]:
            shutil.rmtree(ray_dir)
        else:
            raise ValueError(
                f"Log dir {ray_dir} already exists. Set overwrite_existing_exp=True to delete it."
            )

    os.makedirs(ray_dir)

    # Save the search space
    with open(os.path.join(ray_dir + "/" + time.strftime("%d.%m_%H:%M:%S") + ".json"), "w") as fp:
        json.dump(
            search_space,
            fp,
            sort_keys=True,
            indent=4,
            skipkeys=True,
            default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>",
        )

    # Save the search space by saving this file itself
    shutil.copy(__file__, os.path.join(ray_dir + "/ray.py"))

    # asha_scheduler = ASHAScheduler(
    # time_attr='training_iteration',
    # metric='loss',
    # mode='min',
    # max_t=100,
    # grace_period=10,
    # reduction_factor=3,
    # brackets=1,
    # )

    # Get absolute path
    abs_path = os.path.abspath(ray_dir)

    # # Add file scheme
    uri = 'file://' + abs_path

    np.random.seed(args.seed)
    tuner = tune.Tuner(
        tune.with_resources(functools.partial(main, use_wandb=True), resources=group_factory),
        # functools.partial(main,use_wandb=True),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            num_samples=args.num_samples,
            # scheduler=asha_scheduler,
            search_alg=BasicVariantGenerator(constant_grid_search=True),
            # search_alg=OptunaSearch(mode="min", metric="valid_loss_outer"),
            # search_alg=Repeater(OptunaSearch(mode="min", metric="valid_loss_outer"), repeat=2),
        ),
        run_config=air.RunConfig(name="details", verbose=2, storage_path=uri, local_dir=ray_dir, log_to_file=False),
    )

    # Start timing
    start = time.time()

    results = tuner.fit()

    # Stop timing
    end = time.time()
    print(f"Time elapsed: {end - start}")

    # Get a DataFrame with the results and save it to a CSV file
    df = results.get_dataframe()
    df.to_csv(os.path.join(ray_dir + "/" + "dataframe.csv"), index=False)

    # Generate txt files
    if results.errors:
        print("ERROR!")
        print(results.errors)
    else:
        print("No errors!")
    if results.errors:
        with open(os.path.join(ray_dir, "error.txt"), "w") as file:
            file.write(f"Experiment failed for with errors {results.errors}")

    with open(os.path.join(ray_dir + "/summary.txt"), "w") as file:
        for i, result in enumerate(results):
            if result.error:
                file.write(f"Trial #{i} had an error: {result.error} \n")
                continue

            file.write(f"Trial #{i} finished successfully with a {metric} metric of: {result.metrics[metric]} \n")

    config = results.get_best_result().config
    with open(os.path.join(ray_dir + "/best_config.json"), "w") as file:
        json.dump(
            config,
            file,
            sort_keys=True,
            indent=4,
            skipkeys=True,
            default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>",
        )


def method_config(training_objective):
    if training_objective == "FM":
        method = "FM"
        variant = TBVariant.TB
        method_name = "FM"
    elif training_objective == "TB":
        method = "TB"
        variant = TBVariant.TB
        method_name = "TB"
    elif training_objective == "SubTB1":
        method = "TB"
        variant = TBVariant.SubTB1
        method_name = "SubTB1"
    elif training_objective == "DB":
        method = "TB"
        variant = TBVariant.DB
        method_name = "DB"
    else:
        raise ValueError(f"Training objective {training_objective} not supported")

    return {"algo.method": method, "algo.helper": method_name, "algo.tb.variant": variant}


def task_config(task):
    if task == "seh_frag":
        task_name = "seh_frag"
        oracle = "qed"
    elif task == "seh_plus_frag":
        task_name = "seh_plus_frag"
        oracle = "qed"
    elif task == "qed_frag":
        task_name = "tdc_frag"
        oracle = "qed"

    elif task == "gsk3_frag":
        task_name = "tdc_frag"
        oracle = "gsk3"

    elif task == "drd2_frag":
        task_name = "tdc_frag"
        oracle = "drd2"

    elif task == "sa_frag":
        task_name = "tdc_frag"
        oracle = "sa"
    else:
        raise ValueError(f"Task {task} not supported")
    return {"task.name": task_name, "task.helper": task, "task.tdc.oracle": oracle}


def log_dir_config(name):
    return {
        "log_dir": f"./logs/",
        "exp_name": f"{name}",
        "project": f"{args.prepend_name}{args.experiment_name}"
    }


def exploration_config(exploration_strategy):
    if exploration_strategy == "e_random_action":
        return {
            "algo.train_random_action_prob": tune.choice([0.01, 0.05, 0.1, 0.15, 0.2]),
            "algo.valid_random_action_prob": 0,
        }
    
    elif exploration_strategy == "e_random_traj":
        return {
            "algo.train_random_traj_prob": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]), #TODO
        }

    elif exploration_strategy == "temp_fixed":
        return {
            "algo.sample_temp": tune.choice([0.5, 0.9, 1.1, 1.5, 2]),
            # "cond.temperature.dist_params": tune.choice([96, 48, 32, 16, 8]), #TODO: any other values to try?
            # "cond.temperature.num_thermometer_dim": 1,
        }
    
    elif exploration_strategy == "temp_cond":
        return {
            "cond.temperature.sample_dist": "discrete",
            "cond.temperature.dist_params": [1,2,4,8,16,32,64,96], 
            "cond.temperature.num_thermometer_dim": 50,
        }
    
    elif exploration_strategy == "no_exploration":
        return {}
    elif exploration_strategy == "temp_and_random_action":
        return {
            "algo.train_random_action_prob": tune.choice([0.01, 0.05, 0.1]),
            "algo.valid_random_action_prob": 0,
            "algo.sample_temp": tune.choice([0.9, 1, 1.1, 1.2]),
        }
    elif exploration_strategy == "temp_cond_log_uniform":
        return {
            "cond.temperature.sample_dist": "loguniform",
            "cond.temperature.dist_params": [4, 96], 
            "cond.temperature.num_thermometer_dim": 50,
        }
    else:
        raise ValueError(f"Exploration strategy {exploration_strategy} not supported")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="training_objectives")
    parser.add_argument("--ray_dir_base", type=str, default="./logs/")
    parser.add_argument("--idx", type=int, default=0, help="Run number in an experiment")
    parser.add_argument("--prepend_name", type=str, default="debug_")
    parser.add_argument("--num_cpus", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--placement_cpu", type=int, default=4)
    parser.add_argument("--placement_gpu", type=float, default=1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1212)
    args = parser.parse_args()

    # group_factory = tune.PlacementGroupFactory([{'CPU': 4.0, 'GPU': .25}])
    group_factory = tune.PlacementGroupFactory(
        [{"CPU": args.placement_cpu, "GPU": args.placement_gpu}]
    )
    num_workers = args.placement_cpu

    num_training_steps = 15650 #1000 # 15_650 #10_000
    validate_every = 1000  #100 # 1000 #1000
    num_final_gen_steps = 320 #100 # 320

    # metric = "val_loss"
    # mode = "min"
    metric = "avg_reward_in_topk_modes"
    mode = "max"

    training_objectives = ["FM", "DB", "SubTB1", "TB"]
    tasks = ["seh_plus_frag"]#, "qed_frag", "drd2_frag"]  #'sa_frag' gsk3_frag'

    exploration_strategies = ["e_random_action", "e_random_traj", "temp_fixed", "temp_cond", "no_exploration", "temp_and_random_action", "temp_cond_log_uniform"]

    buffer_samplings = ["uniform", "weighted", "quantile"]  # "weighted_power" 
    buffer_insertions = ["fifo", "reward", "diversity", "diversity_and_reward"]
    buffer_sizes = [1000, 10_000]

    ray.init(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
    )
    # num_cpus = get_num_cpus()
    # num_gpus = torch.cuda.device_count() #this doesn't always work on the cluster

    config = {
        "log_dir": f"./logs/debug_raytune",
        "exp_name": "test",
        "project": "test",
        "device": "cuda" if bool(args.num_gpus) else "cpu",
        "seed": 0,  
        "validate_every": validate_every,  # 1000,
        "print_every": 10,
        "num_training_steps": num_training_steps,  # 10_000,
        "num_workers": num_workers,
        "num_final_gen_steps": num_final_gen_steps,
        "overwrite_existing_exp": True,
        "exploration_helper": "no_exploration",
        "algo": {
            "method": "TB",
            "helper": "TB",
            "sampling_tau": 0.95,
            "sample_temp": 1.0,
            "online_batch_size": 64,
            "replay_batch_size": 32,
            "offline_batch_size": 0,
            "max_nodes": 9,
            "illegal_action_logreward": -75,
            "train_random_action_prob": 0.0,
            "valid_random_action_prob": 0.0,
            "train_random_traj_prob": 0.0,
            "valid_sample_cond_info": True,
            "tb": {
                "variant": TBVariant.TB,
                "Z_learning_rate": 1e-3,
                "Z_lr_decay": 50_000,
                "do_parameterize_p_b": False,
                "do_length_normalize": False,  ###TODO
                "epsilon": None,
                "bootstrap_own_reward": False,
                "cum_subtb": True,
            },
        },
        "model": {
            "num_layers": 4,
            "num_emb": 128,
        },
        "opt": {
            "opt": "adam",
            "learning_rate": 1e-4,
            "lr_decay": 20_000,
            "weight_decay": 1e-8,
            "momentum": 0.9,
            "clip_grad_type": "norm",
            "clip_grad_param": 10,
            "adam_eps": 1e-8,
        },
        "replay": {
            "use": True,
            "capacity": 1000,
            "warmup": 100,
            "hindsight_ratio": 0.0,
            "insertion": {
                "strategy": "diversity_and_reward",
                "sim_thresh": 0.7,
                "reward_thresh": 0.8,
            },
            "sampling": {
                "strategy": "uniform",
                "weighted": {
                    "reward_power": 1.0,
                },
                "quantile": {
                    "alpha": 0.1,
                    "beta": 0.5,
                },
            },
        },
        "cond": {
            "temperature": {
                "sample_dist": "constant",  # "uniform"
                "dist_params": [32.0],  # [0, 64.0],  #[16,32,64,96,128]
                "num_thermometer_dim": 1,
                "val_temp": 32.0,
            }
        },
        "task": {"name": "seh_frag", "helper": "seh_frag", "tdc": {"oracle": "qed"}},
        "evaluation": {
            "k": 100,
            "reward_thresh": 0.75,
            "distance_thresh": 0.3,
        },
    }

    learning_rate = tune.choice([3e-4, 1e-4, 3e-5, 1e-5])
    # lr_decay = tune.choice([20_000, 10_000, 1_000])
    Z_learning_rate = tune.choice([3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
    # Z_lr_decay = tune.choice([100_000, 50_000, 20_000, 1_000])

    shared_search_space = {
        # "opt.lr_decay": lr_decay,
        "opt.learning_rate": learning_rate,
        "algo.tb.Z_learning_rate": Z_learning_rate,
        # "algo.tb.Z_lr_decay": Z_lr_decay,
        "seed": tune.grid_search([42, 1010, 1335]),
    }

    search_spaces = []
    names = []
    if args.experiment_name == "training_objectives":
        for task in tasks: 
            for training_objective in training_objectives:
                name = f"{task}_{training_objective}"
                changes_config = {
                    **log_dir_config(name),
                    **task_config(task),
                    **method_config(training_objective),
                    **shared_search_space,
                }
                names.append(name)
                search_spaces.append(change_config(copy.deepcopy(config), changes_config))

    elif args.experiment_name == "buffer":
        for task in tasks:
            for buffer_size in buffer_sizes:
                for sampling in buffer_samplings:
                    for insertion in buffer_insertions:
                        name = f"{task}_{buffer_size}_{sampling}_{insertion}"
                        changes_config = {
                            **log_dir_config(name),
                            **task_config(task),
                            **shared_search_space,
                            "replay.use": True,
                            "replay.capacity": buffer_size,
                            "replay.insertion.strategy": insertion,
                            "replay.sampling.strategy": sampling,
                        }
                        names.append(name)
                        search_spaces.append(change_config(copy.deepcopy(config), changes_config))

            # No buffer control
            name = f"{task}_no_buffer"
            changes_config = {
                **log_dir_config(name),
                **task_config(task),
                **shared_search_space,
                "replay.use": False,
            }
            names.append(name)
            search_spaces.append(change_config(copy.deepcopy(config), changes_config))

    elif args.experiment_name == "exploration":
        for task in tasks:
            for exploration_strategy in exploration_strategies:
                name = f"{task}_{exploration_strategy}"
                changes_config = {
                    **log_dir_config(name),
                    **task_config(task),
                    **shared_search_space,
                    **exploration_config(exploration_strategy),
                    "exploration_helper": exploration_strategy,
                }
                names.append(name)
                search_spaces.append(change_config(copy.deepcopy(config), changes_config))
    ray_dir = os.path.join(args.ray_dir_base, f"{args.prepend_name}{args.experiment_name}/{names[args.idx]}")
    print(
        f"Running run number {args.idx} out of {len(search_spaces)} with log_dir {ray_dir}"
    )
    
    run_raytune(search_spaces[args.idx], ray_dir)
