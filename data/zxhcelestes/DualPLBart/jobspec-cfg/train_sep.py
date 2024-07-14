import argparse

import torch
from transformers import PLBartTokenizer
from transformers import Trainer, TrainingArguments
from torch.optim import AdamW

from dual_model import DualModel, PLBartForConditionalGeneration
from utils import create_dataset, check_folder


def collect_fn(features: dict):
    to_inp = {"code": [f['code'] for f in features],
              "summ": [f['summ'] for f in features]}
    return to_inp
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_dual", action="store_true")
    parser.add_argument("--code_lang", default="java", choices=['java', 'python'],
                        help="type of programming language")
    
    parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument('--epoches', default=100)
    parser.add_argument("--max_grad_norm", default=5.)
    parser.add_argument('--cp_steps', default=125000)

    parser.add_argument("--reuse", action="store_true",
                        help="if continue training with trained model")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--test_only",
                        default=False,
                        type=bool,
                        help="only test")
    parser.add_argument("--model_name",
                        default="model",
                        type=str,
                        help="name of model")
    parser.add_argument("--save_dir",
                        default="java_results/",
                        type=str,
                        help="directory to save model and log")

    parser.add_argument("--train_code_path",
                        default="data/java/train/code.original")
    parser.add_argument("--train_summ_path",
                        default="data/java/train/javadoc.original")
    parser.add_argument("--dev_code_path",
                        default="data/java/tmp/code.original")
    parser.add_argument("--dev_summ_path",
                        default="data/java/tmp/javadoc.original")
    parser.add_argument("--test_code_path",
                        default="data/java/tmp/code.original")
    parser.add_argument("--test_summ_path",
                        default="data/java/tmp/javadoc.original")

    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight deay if we apply some.")

    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        args.gpu = True
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    training_args = TrainingArguments(
        output_dir=args.save_dir,  # output directory 结果输出地址
        num_train_epochs=args.epoches,  # total # of training epochs 训练总批次
        evaluation_strategy="no",
        per_device_train_batch_size=args.
        train_batch_size,  # batch size per device during training 训练批大小
        per_device_eval_batch_size=args.
        test_batch_size,  # batch size for evaluation 评估批大小
        logging_dir=args.save_dir +
        "/train_log",  # directory for storing logs 日志存储位置
        learning_rate=args.lr,
        save_strategy='epoch',
        save_total_limit=5,
        max_grad_norm=args.max_grad_norm,
        label_names=['labels'],
        gradient_accumulation_steps=3,
        disable_tqdm=True,
        report_to="none")
    
    cs_model = PLBartForConditionalGeneration.from_pretrained("./pretrained_cache/plbart-base")
    cs_tokenizer = PLBartTokenizer.from_pretrained("./pretrained_cache/plbart-base",
                                                   src_lang=args.code_lang,
                                                   tgt_lang="__en_XX__")
    cg_model = PLBartForConditionalGeneration.from_pretrained("./pretrained_cache/plbart-base")
    cg_tokenizer = PLBartTokenizer.from_pretrained("./pretrained_cache/plbart-base",
                                                   src_lang="__en_XX__",
                                                   tgt_lang=args.code_lang)
    dual_model = DualModel(cs_model=cs_model,
                           cg_model=cg_model,
                           cs_tokenizer=cs_tokenizer,
                           cg_tokenizer=cs_tokenizer,
                           device=device)
    optimizer = AdamW(dual_model.parameters(), lr=args.lr)
    train_dataset = create_dataset(code_fp=args.train_code_path,
                                   summ_fp=args.train_summ_path)
    
    trainer = Trainer(
        model=dual_model,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, None),
        data_collator=collect_fn
    )
    torch.cuda.empty_cache()
    
    if args.reuse and check_folder(args.save_dir):
        print("Reuse trained checkpoint")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
