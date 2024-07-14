
import argparse
import logging
import os
import random
import torch
import numpy as np
import torch

from datetime import datetime
from transformers import AdamW
from torch.nn import functional as F
from src.wiki_data_loaders import WIKI_Data_Loaders
from src.model_wrapper import DocLvlEventExt_Model


logger = logging.getLogger(__name__)


def cross_entropy_loss(logits, labels):
    return F.nll_loss(logits, labels)


def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    device = torch.device('cuda')

    # Required parameters
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=['constrained-gen']
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['KAIROS', 'KAIROS0']
    )
    parser.add_argument('--tmp_dir', type=str)
    parser.add_argument(
        "--ckpt_name",
        default=None,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--load_ckpt",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--val_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default=None,
    )
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--coref_dir', type=str,
                        default='data/kairos/coref_outputs')
    parser.add_argument('--use_info', action='store_true', default=False,
                        help='use informative mentions instead of the nearest mention.')
    parser.add_argument('--mark_trigger', action='store_true')
    parser.add_argument('--sample-gen', action='store_true',
                        help='Do sampling when generation.')
    parser.add_argument('--knowledge-pair-gen', action='store_true',
                        help='decoding based on constraint pairs.')
    parser.add_argument('--sim_train', action='store_true',
                        help='train with most similar template as additionl context.')
    parser.add_argument('--adv', action='store_true', help='adv test')
    parser.add_argument("--train_batch_size", default=8,
                        type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--eval_only", action="store_true",
    )
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_clip_val", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--gpus", default=1, help='-1 means train on all gpus')
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--threads", type=int, default=1,
                        help="multiple threads for converting example to features")

    args = parser.parse_args()
    # Set seed
    seed_everything(args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("Training/evaluation parameters %s", args)

    if not args.ckpt_name:
        d = datetime.now()
        time_str = d.strftime('%m-%dT%H%M')
        args.ckpt_name = '{}_{}lr{}_{}'.format(args.model,  args.train_batch_size * args.accumulate_grad_batches,
                                               args.learning_rate, time_str)

    args.ckpt_dir = os.path.join(f'./checkpoints/{args.ckpt_name}')

    os.makedirs(args.ckpt_dir)

    dlee_dl = WIKI_Data_Loaders(args)
    dlee_dl.preprocess_wiki_data()

    model = DocLvlEventExt_Model(args)
    model = model.to(device)

# *************************optimizer******************************
    if args.max_steps < 0:
        args.max_epochs = args.min_epochs = args.num_train_epochs

    train_len = len(dlee_dl.train_dataloader_dlee())
    print("------------------>", train_len)  # dss needs to delete afterwards
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // train_len // args.accumulate_grad_batches + 1
    else:
        t_total = train_len // args.accumulate_grad_batches * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)

    
# *************************optimizer******************************

###########################---In pyTorch---###################################
    num_epochs = args.num_train_epochs
    args.ckpt_dir = os.path.join(f'./checkpoints/{args.ckpt_name}/')
    if args.load_ckpt:
        model.load_state_dict(torch.load(
            args.load_ckpt, map_location=torch.device('cuda'))['model_state_dict'])

# ---------------- testing ----------------
    if args.eval_only:
        smp_opts = []
        model.eval()
        with torch.no_grad():
            for batch_idx, tst_batch in enumerate(dlee_dl.test_dataloader_dlee()):
                tst_batch['input_token_ids'] = tst_batch['input_token_ids'].to(
                    device)
                tst_batch['input_attn_mask'] = tst_batch['input_attn_mask'].to(
                    device)
                tst_batch['tgt_token_ids'] = tst_batch['tgt_token_ids'].to(
                    device)
                tst_batch['tgt_attn_mask'] = tst_batch['tgt_attn_mask'].to(
                    device)
                smp_opts.append(model.test_step(tst_batch, batch_idx))
                # smp_opts = [smp_opts]
                # print("sample output:--",smp_opts)
        # print("sample output:--",smp_opts)
        model.test_epoch_end(smp_opts)
        # print("test complete",batch_idx)
        # if batch_idx == 10:
        #     break

    else:
        # outs = []
        for epoch in range(num_epochs):
            print("epoch:", epoch)
            model.train()
            for batch_idx, train_btch in enumerate(dlee_dl.train_dataloader_dlee()):

                train_btch['input_token_ids'] = train_btch['input_token_ids'].to(
                    device)
                train_btch['input_attn_mask'] = train_btch['input_attn_mask'].to(
                    device)
                train_btch['tgt_token_ids'] = train_btch['tgt_token_ids'].to(
                    device)
                train_btch['tgt_attn_mask'] = train_btch['tgt_attn_mask'].to(
                    device)
                loss = model.training_step(train_btch, batch_idx)
                optimizer.zero_grad()
                loss['loss'].backward()
                optimizer.step()

                print(" running complete for:", batch_idx)
                # if batch_idx == 20:
                #     break

            # epoch_metric = torch.mean(torch.stack([x for x in outs]))
            # print(epoch_metric) #just printing what stores
            print("saving check point.....")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'system_rng': random.getstate(),
                'numpy_rng': np.random.get_state(),
                'torch_rng': torch.random.get_rng_state(),
            }, args.ckpt_dir+str(epoch)+".ckpt")
            print("check point saved successfully")


if __name__ == "__main__":
    main()
