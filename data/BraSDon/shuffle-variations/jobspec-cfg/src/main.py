import argparse
import copy
import os
import random
import sys
from time import sleep

import wandb
import yaml
import numpy as np
import torch
from torch.utils.data import Sampler, DistributedSampler
from torch.optim.lr_scheduler import StepLR, SequentialLR, LinearLR
import torch.distributed as dist

sys.path.insert(0, sys.path[0] + "/../")

from src.models.models import DummyModel, ANN, DeeperANN
from src.training.train import Trainer
from src.data.data import MyDataset
from src.training.custom_sampler import CustomDistributedSampler
from src.training.stratified_sampler import StratifiedSampler
from src.util.cases import CaseFactory


def main():
    # 1. Parse arguments
    system_config, run_config = parse_configs()

    # 2. Setup distributed training (if applicable)
    setup_distributed_training(
        system_config["system"], str(system_config["ddp"]["port"])
    )

    # 2a. Get device
    device = get_device(system_config["system"])

    # 2b. Setup logging
    run = setup_logging(run_config)

    # 3. Set seed
    set_seeds(run_config["seed"])

    # 4. Setup dataloaders
    batch_size = run_config["batch-size"]
    num_workers = run_config["num-workers"]
    my_dataset = get_dataset(system_config, run_config, device)
    train_sampler, test_sampler = get_samplers(
        my_dataset, run_config["case"], batch_size, run_config["seed"]
    )
    train_loader = my_dataset.get_train_loader(train_sampler, batch_size, num_workers)
    test_loader = my_dataset.get_test_loader(test_sampler, batch_size, num_workers)

    # 5. Setup model
    model = get_model_by_name(system_config, run_config)
    assert model is not None

    # 6. Setup training_objects (criterion, optimizer, scheduler)
    criterion = get_criterion(run_config["criterion"])
    optimizer = get_optimizer(model, run_config)
    scheduler = get_scheduler(optimizer, run_config)

    # 7. Perform sanity check before training
    sanity_check(system_config["system"])

    # 8. Setup trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=train_loader,
        test_loader=test_loader,
        system=system_config["system"],
        my_dataset=my_dataset,
        device=device,
    )

    # 9. Run training
    trainer.train(run_config["max-epochs"])

    # 10. Store model to wandb
    store_model_to_wandb(run, model, run_config)

    # 11. Free resources
    free_resources()


def parse_configs() -> tuple[dict, dict]:
    """
    Load the run-config and the system-config.
    The path to the run-config can be passed as a cmd-line argument, the
    system-config path is fixed. The case (e.g. pre-step-local) can be
    specified via cmd-line argument or the run-config. In case both are
    given, the config file will be prioritized.
    :return: system_config, run_config
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="run-configs/default-config.yaml"
    )
    parser.add_argument("--case", type=str, default="baseline")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        run_config = yaml.safe_load(f)
    with open("system-config.yaml", "r") as f:
        system_config = yaml.safe_load(f)

    if "case" not in run_config:
        run_config["case"] = args.case
    print(f"Running case: {run_config['case']}")

    return system_config, run_config


def setup_distributed_training(system: str, port: str) -> None:
    """
    Setup distributed training using torch.distributed.
    Differentiate between local execution and server execution.
    Local execution assumes a world size of four.
    Server execution assumes a slurm environment that sets the following:
    - SLURM_PROCID
    - SLURM_NPROCS
    - SLURM_LAUNCH_NODE_IPADDR
    :param system: (local | server)
    :param port: port to use for distributed training
    """
    if system == "local":
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
        dist.init_process_group(
            "gloo", rank=int(os.environ["LOCAL_RANK"]), world_size=4
        )
        return
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NPROCS"])
    addr = os.environ["SLURM_LAUNCH_NODE_IPADDR"]
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def get_device(system: str) -> torch.device:
    """
    Return the device to use for training.
    :param system: (local | server)
    :return: device
    """
    if system == "local":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        global_rank = int(os.environ["SLURM_PROCID"])
        device = global_rank % gpus_per_node
    return device


def setup_logging(run_config) -> wandb.run:
    """
    Setup weights and biases logging.
    :param run_config: the run_config
    :return: the wandb run
    """
    return wandb.init(
        project="paper",
        group=run_config["group"],
        name=f"{run_config['case']}-rank-{dist.get_rank()}",
        config=run_config,
    )


def set_seeds(seed: int) -> None:
    """Set the seed for torch, numpy and random."""
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_samplers(
    mydataset: MyDataset, case: str, batch_size: int, seed: int
) -> tuple[Sampler, Sampler]:
    """
    Return the samplers for the train and test dataset.
    :param mydataset: MyDataset instance, containing the train and test dataset
    :param case: the case to use for this run (e.g. pre-step-local)
    :param batch_size: the batch size
    :param seed: the initial seed
    :return: samplers for train and test dataset
    """
    if case == "baseline":
        train_sampler = DistributedSampler(mydataset.train_dataset, seed=seed)
    elif case == "stratified":
        train_sampler = StratifiedSampler(
            mydataset.train_dataset, batch_size=batch_size, seed=seed
        )
    else:
        case = CaseFactory.create_case(case)
        mydataset.sort_train_dataset()
        train_sampler = CustomDistributedSampler(mydataset.train_dataset, case, seed)
    test_sampler = DistributedSampler(mydataset.test_dataset)
    return train_sampler, test_sampler


def get_dataset(system_config: dict, run_config: dict, device) -> MyDataset:
    """
    Create and return a MyDataset instance based on the given config files.
    :param system_config: system-config file
    :param run_config: run-config file
    :param device: device to use for training
    :return: MyDataset instance
    """
    dataset_name = run_config["dataset"]
    path = system_config["datasets"][dataset_name]["path"]
    train_transformations = system_config["datasets"][dataset_name]["transforms"][
        "train"
    ]
    test_transformations = system_config["datasets"][dataset_name]["transforms"]["test"]
    load_function = system_config["datasets"][dataset_name]["load-function"]
    num_classes = system_config["datasets"][dataset_name]["num-classes"]

    return MyDataset(
        dataset_name,
        path,
        train_transformations,
        test_transformations,
        load_function,
        num_classes,
        device,
    )


def get_model_by_name(system_config: dict, run_config: dict) -> torch.nn.Module:
    """Return the model with the given name using torch.hub.load."""
    name = run_config["model"]
    repo = system_config["models"][name]["torch.hub.load"]["repo"]
    model = system_config["models"][name]["torch.hub.load"]["model"]
    num_classes = system_config["datasets"][run_config["dataset"]]["num-classes"]
    local_rank = dist.get_rank()
    if system_config["system"] == "server":
        local_rank = local_rank % int(os.environ["SLURM_GPUS_ON_NODE"])

    if name == "dummy":
        return DummyModel()
    elif name == "ann":
        return ANN()
    elif name == "deeper-ann":
        return DeeperANN()
    else:
        try:
            # Downstream code leads to an error when executed in parallel
            # for the first time, due to possible race conditions...
            # File "torch/hub.py", in _get_cache_or_reload:
            #     hub_dir = get_dir()
            #     if not os.path.exists(hub_dir):
            #         os.makedirs(hub_dir)
            if local_rank != 0:
                sleep(5)
            return torch.hub.load(repo, model, trust_repo=True, num_classes=num_classes)
        except:
            raise NotImplementedError(
                f"An error occurred while loading {model}" f" from torch.hub."
            )


def get_criterion(criterion: str) -> torch.nn.Module:
    """Return the criterion with the given name."""
    if criterion == "cross-entropy":
        return torch.nn.CrossEntropyLoss()
    elif criterion == "mse":
        return torch.nn.MSELoss()
    else:
        raise NotImplementedError(f"Criterion {criterion} not implemented.")


def get_optimizer(model: torch.nn.Module, run_config: dict) -> torch.optim.Optimizer:
    """Return the optimizer with the given name."""
    optimizer = run_config["optimizer"]["name"]
    kwargs = copy.deepcopy(run_config["optimizer"]["kwargs"])
    scaled_lr = get_scaled_lr(kwargs["lr"], run_config)

    # Remove lr from kwargs, as it is already scaled.
    kwargs.pop("lr")
    if optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=scaled_lr, **kwargs)
    elif optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=scaled_lr, **kwargs)
    elif optimizer == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=scaled_lr, **kwargs)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented.")


def get_scheduler(
    optimizer: torch.optim.Optimizer, run_config: dict
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create SequentialLR consisting of gradual warmup lr that implements the
    strategy of: https://arxiv.org/abs/1706.02677. And an optional scheduler.
    """
    scheduler_dict: dict = run_config["schedulers"]
    initial_lr = run_config["optimizer"]["kwargs"]["lr"]
    scaled_lr = get_scaled_lr(initial_lr, run_config)

    # If start_factor > 1 (initial_lr > scaled_lr), then...
    # 1. we need no warmup. CHECK
    # 2. the scaled_lr should be used from the start. CHECK
    start_factor = 1 if initial_lr > scaled_lr else initial_lr / scaled_lr
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=start_factor,
        total_iters=scheduler_dict["warmup-epochs"],
    )
    kwargs = scheduler_dict["kwargs"]
    if scheduler_dict["name"] == "step":
        scheduler = StepLR(optimizer, **kwargs)
    elif scheduler_dict["name"] == "none":
        return warmup_scheduler
    else:
        raise NotImplementedError(
            f"Scheduler {scheduler_dict['name']} not implemented."
        )
    return SequentialLR(
        optimizer, [warmup_scheduler, scheduler], [scheduler_dict["warmup-epochs"]]
    )


def get_scaled_lr(initial_lr, run_config) -> float:
    """Scale the initial lr based on the reference-kn."""
    return (
        initial_lr
        * run_config["batch-size"]
        * dist.get_world_size()
        / run_config["schedulers"]["reference-kn"]
    )


def sanity_check(device) -> None:
    """Perform a sanity check before training."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank: {rank}, world_size: {world_size}")
    dist.barrier()
    print(f"Running on {device}")


def store_model_to_wandb(run, model: torch.nn.Module, run_config: dict) -> None:
    """Store the model to wandb."""
    if dist.get_rank() != 0:
        return
    filename = (
        f"{run_config['model']}_{run_config['dataset']}_"
        f"{run_config['case']}_{run_config['seed']}.pth"
    )
    path = "trained_models/" + filename
    torch.save(model.state_dict(), path)
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(path)
    run.log_artifact(artifact)


def free_resources() -> None:
    wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
