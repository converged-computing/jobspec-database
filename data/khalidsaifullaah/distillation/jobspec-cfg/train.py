import os
import time
import argparse
import json
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import transformers

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from mpt_architectures.mpt_1B.gpt_blocks import GPTBlock
from mpt_architectures.mpt_7B_chat.blocks import MPTBlock

from transformers.models.gpt_neox import GPTNeoXTokenizerFast
from transformers.trainer_utils import seed_worker

from transformers.optimization import get_cosine_schedule_with_warmup
from lion_pytorch import Lion
from dataset import make_supervised_data_module
from accelerate.data_loader import skip_first_batches
import wandb

from utils import get_fsdp_wrapped_empty_model, load_model_opt_scheduler_states_fsdp, load_state_dict_fsdp, save_model_opt_scheduler_states_fsdp, load_fsdp_ckpt_with_accelerate

torch.backends.cuda.matmul.allow_tf32 = True  # For faster matmul (but less precise)
torch.backends.cudnn.benchmark = True  # To automate cudnn kernel choice

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_empty_model(model_config_path, add_tokens=1, wrapped_class=None, hack=False):
    model_config = transformers.AutoConfig.from_pretrained(model_config_path, trust_remote_code=True)
    model_config.vocab_size += add_tokens
    if "mpt" in model_config_path:
        model_config.attn_config['attn_impl'] = 'triton'
    return get_fsdp_wrapped_empty_model(model_config, wrapped_class, hack=hack)

def get_model_opt_scheduler(added_tokens, model_config_path, max_steps=1000, warmup_ratio=0.03, weight_decay=0.0, lr=2e-5, wrapped_class=None, hack=False):
    model = get_empty_model(model_config_path, add_tokens=added_tokens, wrapped_class=wrapped_class, hack=hack)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(opt, int(max_steps*warmup_ratio), num_training_steps=max_steps)
    return model, opt, scheduler
    
def get_dataloader_and_sampler(train_dataset, data_collator, batch_size, rank, world_size=4):
    sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    seed=0,
                )
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
    ), sampler

def get_class_from_class_name(class_name):
    if class_name == "LlamaDecoderLayer":
        return LlamaDecoderLayer
    elif class_name == "OPTDecoderLayer":
        return OPTDecoderLayer
    elif class_name == "GPTNeoXLayer":
        return GPTNeoXLayer
    elif class_name == "GPTBlock":
        return GPTBlock
    else:
        raise ValueError(f"Unknown class name {class_name}")

def get_clm_loss(labels, lm_logits):
    # move labels to correct device to enable model parallelism
    labels = labels.to(lm_logits.device)
    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

@record
def fsdp_main(rank, world_size, args):
    setup(rank, world_size, args.port) 
    if rank == 0:
        if args.wandb:
            wandb.init(project=args.wb_project, entity=args.wandb_entity, name=args.wb_name, config=args, resume=args.resume)
    
    torch.cuda.set_device(rank)
    wrapped_class = get_class_from_class_name(args.wrapped_class_name)
    model, opt, scheduler = get_model_opt_scheduler(
        added_tokens=args.added_tokens, 
        model_config_path=args.model_config_path,
        max_steps=args.max_steps, warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay, lr=args.lr,
        wrapped_class=wrapped_class, hack=args.hack)
    
    if args.logit_distillation_mode:
        if args.wrapped_class_name == "GPTBlock":
            teacher_wrapped_class = MPTBlock    # this is a hack to make the teacher model work with the MPT 1B model
        else:
            teacher_wrapped_class = wrapped_class
        teacher_model = get_empty_model(args.teacher_model_config_path, args.added_tokens, teacher_wrapped_class, args.hack)
        teacher_model = load_state_dict_fsdp(teacher_model, args.teacher_model_init_checkpoint_path)
        teacher_model.eval()
        # teacher_model = teacher_model.half()

    if args.resume:
        model, opt, scheduler, start_step_count = load_model_opt_scheduler_states_fsdp(model, opt, scheduler, args.checkpoint_path)
    else:
        model = load_state_dict_fsdp(model, args.init_checkpoint_path)
        # model = load_fsdp_ckpt_with_accelerate(args.init_checkpoint_path, args.model_config_path, args.model_config_path, wrapped_class)
        start_step_count = 0

    if args.act_checkpointing:
        check_fn = lambda submodule: isinstance(submodule, wrapped_class)
        apply_activation_checkpointing(
            model, check_fn=check_fn
        )
    
    if args.wrapped_class_name == "GPTNeoXLayer" or args.wrapped_class_name == "GPTBlock" or args.wrapped_class_name == "MPTBlock":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
                args.model_config_path,
                # model_max_length=model.config.max_seq_len,
                model_max_length=2048,
                padding_side="right",
            )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
                args.model_config_path,
                model_max_length=1024,
                padding_side="right",
                # use_fast=False,
            )
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        if "mpt" in args.model_config_path:
            special_tokens_dict["pad_token"] = tokenizer.eos_token
        else:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    tokenizer.add_special_tokens(special_tokens_dict) # no need to resize model embedding because its been resized during empty model loading

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_path=args.data_path, data_fraction=args.data_fraction, seed=args.sample_seed, efficient_load=True, filtering_method=args.filtering_method)
    train_dataset = data_module['train_dataset']
    data_collator = data_module['data_collator']
    dataloader_full, sampler = get_dataloader_and_sampler(train_dataset=train_dataset, data_collator=data_collator, batch_size=args.batch_size, rank=rank, world_size=world_size)
    # next(iter(dataloader_full)) # this is to make sure that the dataloader is initialized properly
    args.max_steps = (len(train_dataset) * args.num_epochs)//(args.batch_size*world_size*args.accumulation_steps)
    args.save_steps = ((len(train_dataset) * args.num_epochs)/(args.batch_size*world_size*args.accumulation_steps))//10
    # updating the dataloader to the right state
    step_count = start_step_count
    sub_step_count = step_count * args.accumulation_steps
    start_epoch = sub_step_count // len(dataloader_full)
    skip_steps = sub_step_count % len(dataloader_full)
    sampler.set_epoch(start_epoch)
    dataloader = skip_first_batches(dataloader_full, skip_steps)
    print("start_step_count", start_step_count, "step_count", step_count, "epoch", start_epoch, "skip_steps", skip_steps)
    
    accumulation_steps = args.accumulation_steps
    save_steps = args.save_steps
    epoch_iterator = iter(dataloader)
    start_time = time.time()

    def bergman_divergence(x, y, z):
        # term1 = -x / (1 + z * x)
        # term2 = x / ((1 + z * y) ** 2)
        # divergence = torch.sum(term1 + term2)
        term1 = -x / (1 + z * x)
        term2 = y / (1 + z * y)
        term3 = (x-y) / ((1 + z * y) ** 2)
        divergence = torch.sum(term1 + term2 + term3)
        return divergence

    for step_count in range(start_step_count, args.max_steps):
        train_loss = 0
        logit_distill_loss = 0
        clm_loss = 0
        for _ in range(accumulation_steps):
            try:
                data = next(epoch_iterator)
            except StopIteration:
                sampler.set_epoch(sampler.epoch + 1)
                dataloader = dataloader_full # we don't want the skipped version of the dataloader after the first epoch
                epoch_iterator = iter(dataloader)
                data = next(epoch_iterator)
            # calculate loss and backward
            # out = model(**data)
            out = model(input_ids=data['input_ids'], attention_mask=data['attention_mask'], output_attentions=args.use_attention_scores, output_hidden_states=args.use_hidden_states, return_dict=True)
            if args.logit_distillation_mode:
                with torch.no_grad():
                    teacher_out = teacher_model(**data, output_attentions=args.use_attention_scores, output_hidden_states=args.use_hidden_states, return_dict=True)
                label_idx = torch.where(data['labels'] != -100)[-1].tolist()
                if len(label_idx) > 0:
                    labels = data['input_ids'][0][label_idx[0]:].tolist()
                    teacher_out.logits[0, label_idx, labels] += args.spike_factor   # token spike
                if args.loss_type == "kl":
                    # loss_distill = F.kl_div(F.log_softmax(out.logits/args.tmp, dim=-1), F.softmax(teacher_out.logits/args.tmp, dim=-1), reduction="batchmean")
                    P = F.softmax(teacher_out.logits/args.tmp, dim=-1, dtype=torch.float32)
                    log_p = F.log_softmax(teacher_out.logits/args.tmp, dim=-1, dtype=torch.float32)
                    log_q = F.log_softmax(out.logits/args.tmp, dim=-1, dtype=torch.float32)
                    x = torch.sum(P * (log_p - log_q), dim=-1).view(-1)
                    mask = (data['labels'] != -100).cuda().int()
                    loss_distill = torch.mean(x * mask.view(-1), dim=0)
                elif args.loss_type == "bergman_div":
                    loss_distill = bergman_divergence(F.softmax(out.logits, dim=-1, dtype=torch.float32), F.softmax(teacher_out.logits, dim=-1, dtype=torch.float32), args.tmp)
                elif args.loss_type == "reverse_bergman_div":
                    loss_distill = bergman_divergence(F.softmax(teacher_out.logits, dim=-1, dtype=torch.float32), F.softmax(out.logits, dim=-1, dtype=torch.float32), args.tmp)
                elif args.loss_type == "reverse_kl":
                    # from paper
                    # R_t = F.kl_div(F.log_softmax(out.logits/args.tmp, dim=-1), F.softmax(teacher_out.logits/args.tmp, dim=-1), reduction="batchmean") # forward KL
                    # rho_t_theta = F.kl_div(F.log_softmax(teacher_out.logits/args.tmp, dim=-1), F.softmax(out.logits/args.tmp, dim=-1), reduction="batchmean") # reverse KL
                    # loss_distill = R_t * torch.min(rho_t_theta, torch.clamp(rho_t_theta, 1-0.2, 1+0.2))
                    
                    # from miniLLM repo
                    teacher_probs = F.softmax(teacher_out.logits, dim=-1, dtype=torch.float32)
                    inf_mask = torch.isinf(out.logits)
                    logprobs = F.log_softmax(out.logits, dim=-1, dtype=torch.float32)
                    prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
                    x = torch.sum(prod_probs, dim=-1).view(-1)
                    mask = (data['labels'] != -100).cuda().int()
                    if torch.sum(mask.view(-1), dim=0).item() == 0:
                        print("mask is all zero")
                        loss_distill = -torch.sum(x * mask.view(-1), dim=0) / 1.0
                    else:    
                        loss_distill = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

                    # manual implementation of reverse KL
                    # P = F.softmax(out.logits/args.tmp, dim=-1, dtype=torch.float32)
                    # log_p = F.log_softmax(out.logits/args.tmp, dim=-1, dtype=torch.float32)
                    # log_q = F.log_softmax(teacher_out.logits/args.tmp, dim=-1, dtype=torch.float32)
                    # x = torch.sum(P * (log_p - log_q), dim=-1).view(-1)
                    # mask = (data['labels'] != -100).cuda().int()
                    # loss_distill = torch.mean(x * mask.view(-1), dim=0)
                elif args.loss_type == "ce":
                    # normalizing teacher logits using softmax
                    loss_distill = torch.sum(F.softmax(teacher_out.logits, dim=-1) * (-1 *F.log_softmax(out.logits, dim=-1)))
                    # print(loss_distill)
                else:
                    raise ValueError(f"Unknown loss type {args.loss_type}")
                loss_clm = get_clm_loss(data['labels'], out.logits)
                loss = args.alpha * loss_distill + (1-args.alpha) * loss_clm
            else:
                loss = get_clm_loss(data['labels'], out.logits)
            # print(loss)
            (loss/accumulation_steps).backward()
            train_loss += loss.item()/accumulation_steps
            if args.logit_distillation_mode:
                logit_distill_loss += loss_distill.item()/accumulation_steps
                clm_loss += loss_clm.item()/accumulation_steps
        model.clip_grad_norm_(args.max_grad_norm)
        if rank == 0:
            time_so_far = (time.time() - start_time)/ 3600
            iteration_so_far = step_count - start_step_count
            remaining_iterations = args.max_steps - step_count
            estimated_time_per_iteration = time_so_far / (iteration_so_far+1)
            remaining_time = estimated_time_per_iteration * remaining_iterations
            previous_time = start_step_count * estimated_time_per_iteration
            total_estimated_time = time_so_far + remaining_time + previous_time
            metrics_dict = {"train/loss": train_loss, "train/learning_rate": scheduler.get_last_lr()[0], "train/global_step": step_count+1, 
                       "train/time_so_far": time_so_far, "train/remaining_time": remaining_time, 
                       "train/total_estimated_time": total_estimated_time, 
                       "train/train_steps_per_second": 1/(estimated_time_per_iteration*3600),
                       "train/epoch": sampler.epoch}
            if args.logit_distillation_mode:
                metrics_dict["train/logit_distill_loss"] = logit_distill_loss
                metrics_dict["train/clm_loss"] = clm_loss
            else:
                metrics_dict["train/clm_loss"] = train_loss
            if args.wandb:
                wandb.log(metrics_dict, step=step_count)
            print(json.dumps(metrics_dict, indent=4))
        opt.step()
        scheduler.step()
        opt.zero_grad()

        # save the model, optimizer, scheduler
        # if (step_count+1) % save_steps == 0 or (step_count+1) == args.max_steps:
        #     if rank == 0:
        #         print("saving checkpoint", step_count+1)
        #     save_model_opt_scheduler_states_fsdp(model, opt, scheduler, step_count, args.checkpoint_path, rank, dont_save_opt=args.dont_save_opt)
    if rank == 0:
        print("saving checkpoint", step_count+1)
    save_model_opt_scheduler_states_fsdp(model, opt, scheduler, step_count, args.checkpoint_path, rank, dont_save_opt=True)

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_checkpoint_path", type=str, default="/home/ksaifullah/redpajama_3B_sharded")
    parser.add_argument("--model_config_path", type=str, default="/home/ksaifullah/redpajama_3B_hf")
    parser.add_argument("--checkpoint_path", type=str, default="/home/ksaifullah/redpajama_3B_logits_distill")

    parser.add_argument("--teacher_model_init_checkpoint_path", type=str, default="/home/ksaifullah/redpajama_7B_chat_sharded")
    parser.add_argument("--teacher_model_config_path", type=str, default="/home/ksaifullah/redpajama_7B_chat_hf")
    parser.add_argument("--logit_distillation_mode", action='store_true', help="whether to use logit distillation mode")
    parser.add_argument("--alpha", type=float, default=0.5, help="the weight of the distillation loss")
    parser.add_argument("--loss_type", type=str, choices=["kl", "ce", "reverse_kl", "bergman_div", "reverse_bergman_div"], default="kl", help="the type of loss to use for distillation")
    parser.add_argument("--tmp", type=float, default=0.7, help="the temperature to use for softmax in distillation")
    parser.add_argument("--spike_factor", type=float, default=0.0, help="the weight of the distillation loss")
    parser.add_argument("--no_dist", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--use_hidden_states", action='store_true')
    parser.add_argument("--use_attention_scores", action='store_true')
    parser.add_argument("--filtering_method", type=str, choices=["random", "cluster"], default="random")

    parser.add_argument("--wrapped_class_name", type=str, choices=["LlamaDecoderLayer", "OPTDecoderLayer", "GPTNeoXLayer", "GPTBlock", "MPTBlock"], default="GPTBlock",
                        help="the name of the class that is wrapped by the FSDP module")
    parser.add_argument("--dont_save_opt",action='store_true', help="dont save optimizer and scheduler, this saves hard disk memory by trading off ability to resume the run")
    parser.add_argument("--added_tokens", type=int, default=1)
    parser.add_argument("--port", default=None)
    # the structure of the folder is as follows:
    # args.checkpoint_path/$step_count/model/shard_$rank.pt
    # args.checkpoint_path/$step_count/opt/shard_$rank.pt
    parser.add_argument("--data_path", type=str, default="datasets/alpaca-train.jsonl")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="fraction of data to use for training should be between 1 and 0")
    parser.add_argument("--sample_seed", type=int, default=42, help="the random seed used for sampling a fraction of the data")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--max_steps", type=int, default=52002*3//128)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--hack", action='store_true', 
                        help="This is a hack to make training behavior consistent with the huggingface trainer"
                        "For unknown reason, if we use model.bfloat16() when intializing the empty model,"
                        "we need to quadruple the learning to get the same training behavior, but if we do not"
                        "put model.bfloat16() when initializing the empty model, then we run out of memory"
                        "at llama7B, so this flag indicates that we are using a hack and quadruple the learning rate artificially...."
                        "We should investigate this issue in the future.")
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--act_checkpointing", action='store_true')
    parser.add_argument("--save_steps", type=int, default=(52002*3/128)//10)
    parser.add_argument("--accumulation_steps", type=int, default=32)

    # wandb associated arguments
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--wb_project", type=str, default="distillation")
    parser.add_argument("--wandb_entity", type=str, default="ksaifullah")
    parser.add_argument("--wb_name", type=str, default="logit_distillation")
    parser.add_argument("--wb_id", type=str, default="adslifjaoeihgaaa")
    args = parser.parse_args()
    print(args)
    WORLD_SIZE = torch.cuda.device_count()
    if args.port is None:
        args.port = str(random.randint(1024, 65353)) # randomly generate ports if not specified
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)