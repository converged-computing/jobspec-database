"""
This module is to experiment for analysis of "How is the performance of In-context Learning?"
"""

import sys
import logging
import json
import pickle
from datasets import load_dataset
import argparse
from datetime import datetime

import tqdm
import wandb
import pandas as pd
import torch
from accelerate import infer_auto_device_map, init_empty_weights

try:
    from transformers import AutoTokenizer, MT5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoModelForCausalLM, \
        LlamaTokenizer, LlamaForCausalLM
except ImportError:  # old huggingface does not have LlmaTimeForCausalLM
    from transformers import AutoTokenizer, MT5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from few_shot_prompt_utility import format_prompt_from_demo_pairs, prompt_llm, evaluate_response_summaries, \
    generate_tf_idf_keywords, generate_control_length, generate_focus_planning, generate_numerical_keywords

wandb.login(key='3138e1b24deb278ed045d0dedb39511d3a96245b')

# set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='google/mt5-base', help='the name of the model')
parser.add_argument('-k', type=int, default=1, help='the number of few-shot examples')
parser.add_argument('--demonstration_file', type=str, default=None, help='the pre-generated demonstration file')
parser.add_argument('--dataset', type=str, default='samsum', help='the dataset to evaluate on')
parser.add_argument('--keywords', type=str, default=None, choices=['tfidf', 'numeric'],
                    help='the types of keywords to use')
parser.add_argument('--keyword_num', type=int, default=None, help='the number of keywords to use')
parser.add_argument('--log', type=bool, default=True, help='whether to log the results to wandb')
parser.add_argument('--control', type=str, default=None, choices=['length', 'entity', 'focus'],
                    help='the type of control')
parser.add_argument('--replace_name', type=bool, default=False, help='whether to replace the speaker name with '
                                                                     '#Person1# as DialogSum')
parser.add_argument('--add_instruction', type=bool, default=False)
parser.add_argument('--random_label', type=bool, default=False, help='whether to use random labels')
parser.add_argument('--omit_control_in_demo', type=bool, default=False, help='True means not to add control in demonstrations'
                                                                           ' (for ablation study)')
parser.add_argument('--flip_label', type=bool, default=False,
                    help='whether to flip the labels in demonstrations\' summaries')

# parse the arguments
args = parser.parse_args()


# set up log files
logging.basicConfig(
        level=logging.INFO,  # otherwise huggingface has many debug logs
        handlers=[
                logging.FileHandler("{}.log".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))),
                logging.StreamHandler(sys.stdout)
                ]
        )

# start a new wandb run to track this script
if args.log:
    wandb.init(
            project="In-context-learning for Dialogue Summarization",
            # track hyperparameters and run metadata
            config={
                    'model_type': args.model,
                    'k': args.k,
                    'dataset': args.dataset,
                    'keywords': args.keywords,
                    'keyword_num': args.keyword_num,
                    'control': args.control
                    },
            group='performance_in_context_learning',
            job_type='evaluation'
            )

if args.keywords == 'numeric' and args.keyword_num is not None:
    logging.warning('The number of keywords is not used for numerical keywords.')

if args.demonstration_file is None and args.k == 0:
    test_dataset = load_dataset(args.dataset, split='test')
    run_id = 0
    results = {0: {}}
    # iterate over test samples to generate k demonstrations
    for test_sample in tqdm.tqdm(test_dataset, total=len(test_dataset)):
        test_id = test_sample['id']
        results[run_id][test_id] = (test_sample, {})
    run_id2demo_pairs = results
else:
    # load the demonstration pickle file
    with open(args.demonstration_file, 'rb') as f:
        run_id2demo_pairs = pickle.load(f)

if args.control == 'entity':
    if args.keywords == 'tfidf':
        run_id2demo_pairs = generate_tf_idf_keywords(run_id2demo_pairs, args.keyword_num)
    elif args.keywords == 'numeric':
        run_id2demo_pairs = generate_numerical_keywords(run_id2demo_pairs)
    else:
        raise NotImplementedError
elif args.control == 'length':
    run_id2demo_pairs = generate_control_length(run_id2demo_pairs)
elif args.control == 'focus':
    assert args.dataset == 'samsum', 'Only samsum dataset has focus control'
    run_id2demo_pairs = generate_focus_planning(run_id2demo_pairs)
else:
    # issue warning
    logging.warning('No control signal is used.')

run_id2prompts, run_id2gold_summaries = format_prompt_from_demo_pairs(run_id2demo_pairs, args.model, args.replace_name,
                                                                      args.add_instruction,
                                                                      is_focus_planning=args.control == 'focus',
                                                                      is_random_label=args.random_label,
                                                                      is_numerical_label=args.keywords == 'numeric',
                                                                      is_flipped_label=args.flip_label,
                                                                      is_add_control_signals_in_demon=
                                                                      not args.omit_control_in_demo)

logging.info("load the model {}".format(args.model))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info('device count = {}'.format(torch.cuda.device_count()))
if args.model in ['google/mt5-xl', 'google/mt5-base']:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = MT5ForConditionalGeneration.from_pretrained(args.model).to(DEVICE)
elif args.model in ['google/mt5-xxl', 'google/flan-t5-xl']:
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
elif 'llama' in args.model or 'alpaca' in args.model:
    model = LlamaForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map='auto')
    if 'opt' in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        logging.info("device_map: {}".format(model.hf_device_map))
    except AttributeError:
        # send model to GPU
        model = model.to(DEVICE)

logging.info("Start to prompt the model")
run_id2pred_summaries = {}
run_id2raw_outputs = {}
for run_id, prompts in run_id2prompts.items():
    logging.info("Prompting the model for Run {}".format(run_id))
    run_id2pred_summaries[run_id] = []
    run_id2raw_outputs[run_id] = []
    for prompt in tqdm.tqdm(prompts):
        is_gpt_style = args.model not in ['google/mt5-xl', 'google/mt5-base', 'google/mt5-xxl']
        try:
            if is_gpt_style:
                response = prompt_llm(model, tokenizer, prompt, is_gpt_style)
            else:
                if isinstance(prompt, list):  # for mt5 with keywords
                    assert len(prompt) == 2
                    response, raw_output = prompt_llm(model, tokenizer, prompt[0], is_gpt_style,
                                                      spans_to_fill=prompt[1])
                else:  # direct prompt mt5
                    response, raw_output = prompt_llm(model, tokenizer, prompt, is_gpt_style)
                run_id2raw_outputs[run_id].append(raw_output)
        except Exception as e:  # in case any error happens
            logging.info("Exception: {}".format(e))
            logging.info("Prompt: {}".format(prompt))
            # log the complete list of tokens id
            try:
                tokens = tokenizer.encode(prompt, return_tensors="pt")
                # log the complete list of tokens id
                logging.info("Tokens: {}".format(tokens.tolist()))
            except TypeError as e:
                logging.info('[Error during tokenization] {}'.format(e))
            response = None
        run_id2pred_summaries[run_id].append(response)

logging.info("Start to evaluate the performance")
if 'mt5' in args.model:
    # check if the value of run_id2prompts is nested list
    if isinstance(list(run_id2prompts.values())[0][0], list):
        run_id2prompts = {k: [x[0] for x in v] for k, v in run_id2prompts.items()}
    summary_table, summary_text_table = evaluate_response_summaries(run_id2pred_summaries, run_id2gold_summaries,
                                                                    run_id2prompts, run_id2raw_outputs)
else:
    summary_table, summary_text_table = evaluate_response_summaries(run_id2pred_summaries, run_id2gold_summaries,
                                                                    run_id2prompts)

# save the summary table to wandb
summary_table = pd.DataFrame(summary_table)
summary_text_table = pd.DataFrame(summary_text_table)
summary_table = wandb.Table(dataframe=summary_table)
summary_text_table = wandb.Table(dataframe=summary_text_table)
wandb.log({"Evaluation metrics Table": summary_table})
wandb.log({"Summaries Table": summary_text_table})
wandb.finish()
