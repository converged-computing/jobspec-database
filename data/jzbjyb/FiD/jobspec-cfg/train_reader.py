# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import time
import sys
import random
import torch
import transformers
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.multiprocessing import set_start_method
#from fairscale.optim.oss import OSS
#from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP

from src.options import Options
import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
from src.util import WandbLogger, global_context


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_metric, checkpoint_path):
    tb_logger = None
    if opt.is_main:
        WandbLogger.init(opt)

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=opt.num_workers,
        collate_fn=collator)

    loss, curr_loss = 0.0, 0.0
    epoch = 0
    model.train()
    pbar = tqdm(total=opt.total_steps, disable=not opt.is_main)
    _step = step * opt.accumulation_steps  # the "real" step used for gradient accumulation
    while step <= opt.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            _step += 1
            if opt.accumulation_steps == 1 or _step % opt.accumulation_steps == 1:  # increase the optimization step
                step += 1
                WandbLogger.step(step)
                pbar.update(1)
            if step > opt.total_steps:
                break

            idx, labels, _, context_ids, context_mask, context_sep_mask, doc_ids, gold_doc_dist = batch
            context_sep_mask = context_sep_mask.cuda() if context_sep_mask is not None else context_sep_mask
            gold_doc_dist = gold_doc_dist.cuda() if gold_doc_dist is not None else gold_doc_dist
            '''
            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                attention_separate_mask=context_sep_mask,
                labels=labels.cuda(),
                input_doc_ids=doc_ids,
                gold_doc_dist=gold_doc_dist,
                accumulate_steps=opt.accumulation_for_ibn,
            )[0]
            '''
            train_loss = src.model.fid_run(
                model, 
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                attention_separate_mask=context_sep_mask,
                labels=labels.cuda(),
                input_doc_ids=doc_ids,
                gold_doc_dist=gold_doc_dist,
                accumulate_steps=opt.accumulation_for_ibn)[0]

            if _step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                train_loss = src.util.average_main(train_loss, opt)
                curr_loss += train_loss.item()
                WandbLogger.log_w_step({'train-loss': train_loss.item()})

                if step % opt.eval_freq == 0:
                    dev_metric = evaluate(model, eval_dataset, tokenizer, collator, opt)
                    model.train()
                    if opt.is_main:
                        if dev_metric > best_dev_metric:
                            best_dev_metric = dev_metric
                            src.util.save(model, optimizer, scheduler, step, best_dev_metric, opt, checkpoint_path, 'best_dev')
                        log = f"{step} / {opt.total_steps} |"
                        log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                        log += f"evaluation: {100*dev_metric:.2f}EM |"
                        log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                        logger.info(log)
                        WandbLogger.log_w_step({f'dev-{opt.metric}': dev_metric}, name='eval', interval=1)
                        if tb_logger is not None:
                            tb_logger.add_scalar("Evaluation", dev_metric, step)
                            tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                        curr_loss = 0.

                if opt.is_main and step % opt.save_freq == 0:
                    src.util.save(model, optimizer, scheduler, step, best_dev_metric, opt, checkpoint_path, f"step-{step}")

    if opt.is_main and opt.total_steps == 0:  # save the original model
        src.util.save(model, optimizer, scheduler, step, best_dev_metric, opt, checkpoint_path, f'step-{step}')

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    eval_batch_size = (opt.per_gpu_batch_size // opt.accumulation_for_ibn) if opt.accumulation_for_ibn else opt.per_gpu_batch_size
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=eval_batch_size,
        drop_last=False,
        num_workers=opt.num_workers,
        collate_fn=collator)
    model.eval()
    total = 0
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            idx, _, _, context_ids, context_mask, context_sep_mask, _, _ = batch
            context_sep_mask = context_sep_mask.cuda() if context_sep_mask is not None else context_sep_mask
            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                attention_separate_mask=context_sep_mask,
                max_length=opt.answer_maxlength)

            for k, o in enumerate(outputs):
                _, ans = src.util.extract_query_answer(o, tokenizer, query_in_decoder=opt.query_in_decoder)[:2]
                gold = dataset.get_example(idx[k])['answers']
                if opt.metric == 'em':
                    score = src.evaluation.ems(ans, gold)
                elif opt.metric == 'rougel':
                    score = src.evaluation.rougels(ans, gold)
                else:
                    raise NotImplementedError
                total += 1
                exactmatch.append(score)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch

if __name__ == "__main__":
    set_start_method('spawn')

    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)
    global_context['opt'] = opt

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    src.slurm.init_distributed_mode(opt, is_slurm_job=False)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    if opt.model_size in {'base', 'large'}:
        model_name = 't5-' + opt.model_size
    else:
        model_name = opt.model_size  # 'google/t5-base-lm-adapt'
    model_class = src.model.FiDT5

    #load data
    if opt.is_distributed and opt.global_rank != 0:  # load tokenizer
        torch.distributed.barrier()
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    if opt.is_distributed and opt.global_rank == 0:
        torch.distributed.barrier()
    collator = src.data.Collator(
      opt.text_maxlength,
      tokenizer,
      answer_maxlength=opt.answer_maxlength,
      separate_query_passage=opt.attention_mask,
      query_in_decoder=opt.query_in_decoder)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
        n_context=opt.n_context,
        use_gold_doc_dist=opt.use_gold_doc_dist,
    )
    train_dataset = src.data.Dataset(
        train_examples, opt.n_context, augmentation=opt.augmentation, join_multiple_answer=opt.join_multiple_answer)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
        n_context=opt.n_context,
        use_gold_doc_dist=opt.use_gold_doc_dist,
    )
    if opt.eval_num_examples:
        eval_examples = eval_examples[:opt.eval_num_examples]
    eval_dataset = src.data.Dataset(
        eval_examples, opt.n_context, augmentation=opt.augmentation, join_multiple_answer=opt.join_multiple_answer)

    if opt.is_distributed and opt.global_rank != 0:  # load model
        torch.distributed.barrier()
    if not checkpoint_exists and opt.model_path == "none":
        model = src.model.FiDT5.from_t5(
          model_name,
          n_layer_two_tower=opt.n_layer_two_tower,
          layer_for_retrieval=opt.layer_for_retrieval,
          attention_mask=opt.attention_mask,
          retrieval_aggregation_method=opt.retrieval_aggregation_method,
          query_in_decoder=opt.query_in_decoder,
          num_keep_ctx_in_decoder=opt.num_keep_ctx_in_decoder,
          combine_weight=opt.combine_weight,
          only_topk_n_context=opt.only_topk_n_context,
          keep_ctx_in_decoder_with_head=opt.keep_ctx_in_decoder_with_head,
          keep_ctx_in_decoder_head_tau=opt.keep_ctx_in_decoder_head_tau,
          head_weights_norm_func=opt.head_weights_norm_func,
          encoder_decoder_kl_ratio=opt.encoder_decoder_kl_ratio,
          encoder_decoder_kl_method=opt.encoder_decoder_kl_method,
          encoder_encoder_kl_method=opt.encoder_encoder_kl_method,
          in_batch_negative=opt.in_batch_negative,
          in_batch_negative_size=opt.in_batch_negative_size,
          in_batch_negative_max_num_query=opt.in_batch_negative_max_num_query,
          pairwise_loss=opt.pairwise_loss,
          memory_bank=opt.memory_bank,
          memory_bank_topk=opt.memory_bank_topk,
          memory_use_random=opt.memory_use_random,
          memory_bank_recompute=opt.memory_bank_recompute,
          memory_bank_additional_encode=opt.memory_bank_additional_encode,
          encoder_encoder_kl_ratio=opt.encoder_encoder_kl_ratio,
          encoder_encoder_kl_sparsity=opt.encoder_encoder_kl_sparsity,
          encoder_encoder_kl=opt.encoder_encoder_kl,
          decoder_attn_ctx_normalize=opt.decoder_attn_ctx_normalize,
          encoder_attention_pre_softmax=opt.encoder_attention_pre_softmax,
          max_over_head=opt.max_over_head,
          term_weight_parameter=opt.term_weight_parameter,
          embedding_normalize=opt.embedding_normalize,
          use_gold_doc_dist=opt.use_gold_doc_dist,
          retrieval_projection=opt.retrieval_projection,
          kl_loss_reduction=opt.kl_loss_reduction,
          no_qa=opt.no_qa,
          n_context_for_ibn=opt.n_context_for_ibn)
        if opt.init_from:
          logger.info(f'Init from {opt.init_from}')
          _model = model_class.from_pretrained(opt.init_from)
          model.load_from(_model)
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model, sharding=opt.use_sharding)
        step, best_dev_metric = 0, 0.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_metric = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:  # init from another checkpoint
        logger.info(f'Continue training from {opt.model_path}')
        model = model_class.from_pretrained(opt.model_path)
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model, sharding=opt.use_sharding)
        step, best_dev_metric = 0, 0.0
    if opt.is_distributed and opt.global_rank == 0:
        torch.distributed.barrier()

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        if opt.use_sharding:
            model = ShardedDDP(model, optimizer)
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                find_unused_parameters=False,
            )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_metric,
        checkpoint_path
    )
