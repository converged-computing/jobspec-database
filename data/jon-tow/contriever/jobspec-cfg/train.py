# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import logging
import wandb
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler

from src.options import Options
from src import data, beir_utils, slurm, dist_utils, utils
from src import moco


logger = logging.getLogger(__name__)


def train(opt, model, optimizer, scheduler, step, wandb_run = None):
    run_stats = utils.WeightedAvgStats()

    logger.info("Data loading")
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        tokenizer = model.module.tokenizer
    else:
        tokenizer = model.tokenizer
    collator = data.Collator(opt=opt)
    train_dataset = data.load_data(opt, tokenizer)
    logger.warning(f"Data loading finished for rank {dist_utils.get_rank()}")
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        batch_size=opt.per_gpu_batch_size, 
        drop_last=True, 
        num_workers=opt.num_workers, 
        collate_fn=collator
    )

    log_embed_dir = Path(opt.log_embed_dir)
    if not log_embed_dir.exists():
        log_embed_dir.mkdir(parents=True, exist_ok=True)

    epoch = 1

    model.train()
    while step < opt.total_steps:
        train_dataset.generate_offset()

        logger.info(f'Start epoch {epoch}')
        for i, batch in enumerate(train_dataloader):
            step += 1

            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            # Store batch to get enough data for embedding logs
            if step % opt.log_embed_freq == opt.log_embed_freq - 1:
                log_batch = batch

            train_loss, iter_stats = model(**batch, stats_prefix='train')
            train_loss.backward()
            optimizer.step()

            scheduler.step()
            model.zero_grad()

            run_stats.update(iter_stats)

            if step % opt.log_embed_freq == 0:
                utils.log_train_embed(opt, step, model, [batch, log_batch], train_loss)
                del log_batch

            if step % opt.log_freq == 0:
                log = f'{step} / {opt.total_steps}'
                for k, v in run_stats.average_stats.items():
                    log += f' | {k}: {v:.3f}'
                    if dist_utils.is_main() and wandb_run is not None:
                        wandb_run.log({k: v}, step=step)
                log += f' | lr: {scheduler.get_last_lr()[0]:0.3g}'
                log += f' | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB'

                logger.info(log)
                run_stats.reset()

            if step % opt.eval_freq == 0:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    encoder = model.module.get_encoder()
                else:
                    encoder = model.get_encoder()
                evalmodel(
                    opt, 
                    query_encoder=encoder, 
                    doc_encoder=encoder, 
                    tokenizer=tokenizer,
                    wandb_run=wandb_run,
                    step=step
                )

                if dist_utils.is_main():
                    utils.save(model, optimizer, scheduler, step, opt, opt.output_dir, f"lastlog")

                model.train()

            if dist_utils.is_main() and step % opt.save_freq == 0:
                utils.save(model, optimizer, scheduler, step, opt, opt.output_dir, f"step-{step}")

            if step > opt.total_steps:
                break
        epoch += 1


def evalmodel(opt, query_encoder, doc_encoder, tokenizer, wandb_run, step):
    for datasetname in opt.eval_datasets:
        metrics = beir_utils.evaluate_model(
            query_encoder, 
            doc_encoder,
            tokenizer, 
            dataset=datasetname, 
            batch_size=opt.per_gpu_eval_batch_size,
            norm_doc=opt.norm_doc, 
            norm_query=opt.norm_query,
            beir_dir=opt.eval_datasets_dir,
            score_function=opt.score_function,
            lower_case=opt.lower_case,
            normalize_text=opt.eval_normalize_text,
            is_main=dist_utils.is_main(),
        )

        message = []
        if dist_utils.is_main():
            for metric in ['NDCG@10', 'Recall@10', 'Recall@100']:
                message.append(f"{datasetname}/{metric}: {metrics[metric]:.2f}")
                if dist_utils.is_main() and wandb_run is not None:
                    wandb_run.log({f"{datasetname}/{metric}": metrics[metric]}, step=step)
            logger.info(" | ".join(message))

if __name__ == "__main__":
    logger.info("Start")

    options = Options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    directory_exists = os.path.isdir(opt.output_dir)
    if dist.is_initialized():
        dist.barrier()
    os.makedirs(opt.output_dir, exist_ok=True)
    if not directory_exists and dist_utils.is_main():
        options.print_options(opt)
    if dist.is_initialized():
        dist.barrier()
    utils.init_logger(opt)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    model_class = moco.MoCo

    if not directory_exists and opt.model_path == "none":
        model = model_class(opt)
        model = model.cuda()
        optimizer, scheduler = utils.set_optim(opt, model)
        step = 0
    elif directory_exists and opt.model_path == "none":
        model_path = os.path.join(opt.output_dir, 'checkpoint', 'latest')
        model, optimizer, scheduler, opt_checkpoint, step = utils.load(
            model_class, model_path, opt, reset_params=False,
        )
        logger.info(f"Model loaded from {opt.output_dir}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step = utils.load(
            model_class, opt.model_path, opt, reset_params=False if opt.continue_training else True,
        )
        if not opt.continue_training:
            step = 0 
        # TODO: Fix utils.load to properly `step` and avoid this hard-coding
        # step = 100_000
        logger.info(f"Model loaded from {opt.model_path} on step {step}")

    logger.info(utils.get_parameters(model))

    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )
        dist.barrier()
 
    wandb_run = None
    if dist_utils.is_main():
        wandb_run = wandb.init(
            project=opt.wandb_project,
            entity=opt.wandb_entity,
            save_code=False,
            force=False,
            name=opt.name,
            resume=True if opt.wandb_id is not None else False,
            id=opt.wandb_id if opt.wandb_id is not None else None,
        )
   
    logger.info("Start training")
    train(opt, model, optimizer, scheduler, step, wandb_run)

    if dist_utils.is_main():
        # Make sure all processes are cleaned up
        wandb_run.finish()
