import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from omegaconf import OmegaConf

import config

from utils.args_parser import arguments_parser
from utils.init_utils import init_process
from utils.train_utils import create_dataloader, create_criterions, create_optim_scheduler
from utils.logger_utils import attr_extractor
from utils.logging_tool import get_logger

# from arch import SomeModel
from models.basic_model import BasicModel


def main(args):
    logger = get_logger()
    device = args.local_rank

    # init wandb
    if args.rank == 0:
        wandb.init(project='lip-sync', dir=args.job_dir, name=args.job_dir.split('/')[-1],
                   config=OmegaConf.to_container(args))
    # Create job and tb_writer
    writer = SummaryWriter(args.job_dir) if args.rank == 0 else None

    # Load dataset
    logger.info(f"Load Dataset")
    train_data_loader, train_sampler, eval_data_loaders, eval_samplers = create_dataloader(args)

    # Create generator
    logger.info(f"Create Model")
    model = SomeModel(**args.model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"G-Model {model} :[Trainable Parameters: {trainable_params}]")

    # Loss function
    logger.info(f"Load loss function")
    criterion = create_criterions(args)

    # allocate model to gpu
    if args.distributed:
        logger.info("Distributed Training")
        model = DDP(model.to(device), device_ids=[device], output_device=device)
    else:
        model.to(device)

    # create optimizers and schedulers
    [optimizer], [scheduler] = create_optim_scheduler(model,
                                                      args=args,
                                                      num_batches=len(train_data_loader))
    trainer = BasicModel(**kwargs)

    # Load ckpt
    start_epoch = trainer.load_ckpt(args.ckpt,
                                    model=model, optimizer=optimizer, scheduler=scheduler)

    # Load state_dict
    trainer.load_model(model=g_model, ckpt_path=args.g_weight)
    trainer.load_model(model=d_model, ckpt_path=args.d_weight)

    logger.info(attr_extractor(args))

    if args.weight or args.ckpt:
        trainer.evaluating_epoch(epoch=start_epoch)
        if args.eval_only:
            return logger.info('Finish evaluation')

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Train
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        trainer.training_epoch(epoch=epoch)
        # Eval model
        trainer.evaluating_epoch(epoch=epoch)

        # save model weight
        trainer.save_model(os.path.join(args.job_dir, 'weights'))
        trainer.save_ckpt(os.path.join(args.job_dir, "ckpt"), epoch=epoch)

    logger.info(f"Finish Training")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arguments_parser(parser)

    # Parse arguments
    args = parser.parse_args()
    init_process(args)

    # read from config file
    args = config.update_params(args)

    # create logger
    logger = get_logger(file_path=args.job_dir)

    main(args)
