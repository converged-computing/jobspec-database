#coding=utf-8
import argparse
from ast import parse
from fsspec import spec
from pytorch_lightning.accelerators import accelerator
import tokenizers
from torch._C import device
from tqdm import tqdm
import sys

import wandb
import torch
from OutGener import OutGenerModel
from utils.dataset import OutGenDataModule, OutGenDataset
from pytorch_lightning.trainer import Trainer
import os
import csv
from pytorch_lightning.utilities import seed
from pytorch_lightning.callbacks import ModelCheckpoint


from torch.utils.data import DataLoader, dataloader

from transformers import (
    BertTokenizer,
    XLNetTokenizer
)

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--root", type=str, default=os.path.join("/userhome/whzhong/code/GPT2LOT"), help="project root path")
    parser.add_argument("--data_root", type=str, default=os.path.join("LOTdatasets/outgen"), help="Dataset root path")
    parser.add_argument("--model_path", type=str, default="models/BART",  help="Path to save or load the model")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to output generating files")
    parser.add_argument("--ckpt_dir", type=str, default="ckpts", help="Path to output generating files")
    parser.add_argument("--ckpt_load_dir", type=str, default="ckpts", help="Path to load model checkpoints for training")
    parser.add_argument("--test_model", type=str, default="BART-epoch= 2.ckpt", help="test model file name")
    parser.add_argument("--resume_dir", type=str, default=None, help="resume training from this path")
    

    parser.add_argument("--test_pth", type=str, default="test_dataset.pth", help="test model file name")

    parser.add_argument("--do_train", action='store_true', default=True, help="whether do training")
    parser.add_argument("--cont_train", action='store_true', default=False, help="whether do training")
    parser.add_argument("--do_test", action='store_true', default=False, help="whether do generation")

    parser.add_argument("--model_name", type=str, default="BART", help="Path to output generating files")
    parser.add_argument("--group_name", type=str, default="DDP", help="group name for ddp training")
    
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--min_length", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--epoch_num", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)

    parser.add_argument("--eos_token", type=str, default="[EOS]")
    parser.add_argument("--bos_token", type=str, default="[BOS]")
    parser.add_argument("--sep_token", type=str, default="[PAD]")
    parser.add_argument("--delimeter_token", type=str, default="<DELIMETER>")
    parser.add_argument("--pad_token", type=str, default="[PAD]")
    
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--lm_coef", type=float, default=0.9)
    parser.add_argument("--n_valid", type=int, default=374)
    parser.add_argument('--length_penalty', default=1, type=float, required=False, help='long text > 1; short text < 1')

    parser.add_argument("--random_seed", type=int, default=42)
    

    args = parser.parse_args()
    
    args.data_root = os.path.join(args.root, args.data_root)
    args.model_path = os.path.join(args.root, args.model_path)
    args.output_dir = os.path.join(args.root, args.output_dir)
    args.ckpt_dir = os.path.join(args.root, args.ckpt_dir)
    args.ckpt_load_dir = os.path.join(args.root, args.ckpt_load_dir)
    if args.resume_dir is not None:
        args.resume_dir = os.path.join(args.root, args.resume_dir)


    return args



def redefine_tokenizer(args):

    if args.model_name == 'BART' or args.model_name == 'CPT':
        
        special_tokens = {'delimeter': args.delimeter_token, 'eos': args.eos_token, 'bos': args.bos_token, 'sep': args.sep_token}
        addtional_tokens = {'bos_token': special_tokens['bos'], 'eos_token': special_tokens['eos'], 'additional_special_tokens':\
            ['<DELIMETER>', '<SEPE>', '<WORD>']}
        tokenizer = BertTokenizer.from_pretrained(args.model_path)
        tokenizer.add_tokens(["“", "”"])
    else:
        special_tokens = {'delimeter': args.delimeter_token, 'eos': args.eos_token, 'bos': args.bos_token, 'sep': args.sep_token, 'pad': args.pad_token}
        addtional_tokens = {'bos_token': special_tokens['bos'], 'eos_token': special_tokens['eos'], 'additional_special_tokens':\
            ['<DELIMETER>', '<SEPE>'], 'pad_token': special_tokens['pad']}
        
        tokenizer = XLNetTokenizer.from_pretrained(args.model_path)
        tokenizer.padding_side = 'right'
    # spacial_tokens = ['START', 'DELIMITER']
    #tokenizer.add_tokens(list(special_tokens.values()))
    #tokenizer.eos_token = special_tokens['eos']
    #tokenizer.bos_token = special_tokens['bos']
    #tokenizer.pad_token = '[PAD]'
    tokenizer.add_special_tokens(addtional_tokens)
    return tokenizer, special_tokens

def batch_generation(args):
    
    args = parse_args()
    
    seed.seed_everything(args.random_seed)

    tokenizer, special_tokens = redefine_tokenizer(args)
    dataset = DataLoader(OutGenDataset(args.data_root, 'test', special_tokens), \
    batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # args, tokenizer, special_tokens, batch_num
    print(os.path.join(args.ckpt_dir, args.test_model))
    
    #import ipdb; ipdb.set_trace()
    model = OutGenerModel.load_from_checkpoint(os.path.join(args.ckpt_dir, args.test_model), \
        args=args, tokenizer=tokenizer, special_tokens=special_tokens)
    model.to(device)
    model.eval()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    with open(os.path.join(args.output_dir, f'{args.model_name}_test.json'), 'w') as f:
        for item in tqdm(dataset):
            context_batch = item['source']
            # use the predefined model length 
            encoded_batch = tokenizer(context_batch, padding=True, max_length=args.max_length, truncation=True)
            sentence_ids = torch.tensor(encoded_batch.input_ids)
            sentence_ids = sentence_ids.to(device)
            output = model(sentence_ids, max_length=args.max_length)
            if args.model_name == "BART" or args.model_name == "CPT":
                inference_results = [tokenizer.decode(sample, skip_special_tokens=True)\
                    .replace(" ", "").replace(special_tokens['delimeter'], "") for sample in output]
                for inference_result in inference_results:
                    f.write(inference_result + "\n")    
            else:
                inference_result = tokenizer.decode(output[0][len(sentence_ids[0]):], skip_special_tokens=True)\
                    .replace(" ", "").replace("\n", "")
                f.write(inference_result + "\n")    

def generation(args):
    
    args = parse_args()
    
    seed.seed_everything(args.random_seed)

    tokenizer, special_tokens = redefine_tokenizer(args)

    dataset = OutGenDataset(args.data_root, 'test', special_tokens)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # args, tokenizer, special_tokens, batch_num
    model = OutGenerModel.load_from_checkpoint(os.path.join("~/code/GPT2LOT",args.ckpt_dir, args.test_model), \
        args=args, tokenizer=tokenizer, special_tokens=special_tokens)
    model.to(device)
    model.eval()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    with open(os.path.join(args.output_dir, f'{args.model_name}_test.json'), 'w') as f:
        for item in tqdm(dataset):
            line = item['source']
            sentence_ids = torch.tensor([tokenizer(line).input_ids])
            sentence_ids = sentence_ids.to(device)
            output = model(sentence_ids, max_length=args.max_length)
            if args.model_name == "BART" or args.model_name == "CPT":
                inference_result = tokenizer.decode(output[0], skip_special_tokens=True)\
                .replace(" ", "").replace(special_tokens['delimeter'], "")
            else:
                inference_result = tokenizer.decode(output[0][len(sentence_ids[0]):], skip_special_tokens=True)\
                    .replace(" ", "").replace("\n", "")
            f.write(inference_result + "\n")

def main(args):

    seed.seed_everything(args.random_seed)

    wandb.login(key='618360cab03417467d48fa9773513264ef8f4794')
    wandb_run = wandb.init(project="BART-outgen", group=args.group_name)

    tokenizer, special_tokens = redefine_tokenizer(args)

    dataloader = OutGenDataModule(args.data_root, tokenizer, args, special_tokens)
    checkpoint_callback = ModelCheckpoint(
        monitor="bleu-1",
        dirpath=args.ckpt_dir,
        filename=args.model_name+"-{epoch:2d}",
        save_top_k=-1,
        every_n_epochs=1,
        mode="max"
    )

    if args.cont_train:
        model = OutGenerModel.load_from_checkpoint(os.path.join(args.ckpt_load_dir, args.\
           test_model), args=args, tokenizer=tokenizer, special_tokens=special_tokens, runs = wandb_run)
    else:
        model = OutGenerModel(args, tokenizer, special_tokens, wandb_run)
    # accelerator="ddp", 
    #trainer = Trainer(gpus=-1, accelerator="ddp", callbacks=[checkpoint_callback], max_epochs=args.epoch_num)
   # trainer = Trainer(gpus=1, callbacks=[checkpoint_callback], max_epochs=args.epoch_num, precision=16)
    if args.resume_dir is not None:
        trainer = Trainer(resume_from_checkpoint=os.path.join(args.resume_dir, args.test_model), gpus=-1, accelerator="ddp", callbacks=[checkpoint_callback], max_epochs=args.epoch_num, precision=16)
    else:
        trainer = Trainer(gpus=-1, accelerator="ddp", callbacks=[checkpoint_callback], max_epochs=args.epoch_num, precision=16)
    trainer.fit(model, datamodule=dataloader)

    wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    if args.do_test:
        batch_generation(args)
    else:
        main(args)