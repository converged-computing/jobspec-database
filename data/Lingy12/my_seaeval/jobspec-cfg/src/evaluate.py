#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Monday, July 24th 2023, 5:55:08 pm
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###


import os
import fire
import json

import logging

from tqdm import trange
import torch
from dataset import Dataset
from model   import Model
from metric  import Metric

from transformers import set_seed
set_seed(42) # ensure reproducability
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from config.model_path import (
    MODEL_LANG,
)

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
LANG_TAG = ['[ZH]', '[EN]', '[VI]', '[ES]']
TAG_MAP = {
    "chinese": "[ZH]", 
    "english": "[EN]",
    "vietnamese": "[VI]",
    "spanish": "[ES]"
}

def main(
        dataset_name: str = "",
        model_name  : str = "",
        batch_size  : int = 1,
        prompt_index: int = 0,
        eval_mode   : str = "public_test",
        target_folder: str = None,
        tag: bool = False
):

    
    if target_folder is None:
        raise Exception("Please specify target folder")
    
    logger.info("Dataset name: {}".format(dataset_name))
    logger.info("Model name: {}".format(model_name))
    logger.info("Batch size: {}".format(batch_size))
    logger.info("Prompt index: {}".format(prompt_index))

    logger.info("")
    
    dataset = Dataset(dataset_name, eval_mode, prompt_index, support_langs=['english', 'chinese', 'vietnamese', 'spanish'])
    model   = Model(model_name)
    metric  = Metric(dataset_name)

    results, model_predictions, all_soft_answer = do_evaluation(dataset, model, metric, batch_size, tag)
    model_name = os.path.basename(os.path.normpath(model_name))
    all_samples_with_model_predictions = []
    for sample, model_prediction, model_soft_answer in zip(dataset.data, model_predictions, all_soft_answer):
        sample['model_prediction'] = model_prediction.encode('utf-8', 'ignore').decode('utf-8')
        #sample['model_prediction'] = model_prediction
        sample['model_soft_answer'] = model_soft_answer
        all_samples_with_model_predictions.append(sample)
    
    pred_folder = os.path.join(target_folder, "log_predictions")
    log_folder = os.path.join(target_folder, "log")
    os.makedirs(pred_folder, exist_ok=True)  
    os.makedirs(log_folder, exist_ok=True)  
    with open(os.path.join(pred_folder, '{}_{}_p{}.json'.format(model_name, dataset_name, prompt_index)), 'w') as f:
        
        if model_name in ['chatglm-6b', 'chatglm2-6b', 'chatglm3-6b']:
            json.dump(all_samples_with_model_predictions, f, indent=4, sort_keys=False, ensure_ascii=True)
        else:
            json.dump(all_samples_with_model_predictions, f, indent=4, sort_keys=False, ensure_ascii=False)


    logger.info("Results: {0}".format(json.dumps(results, indent=4)))

    with open(os.path.join(log_folder, '{}_p{}.json'.format(dataset_name, prompt_index)), 'w') as f:
        json.dump(results, f, indent=4)


def do_evaluation(dataset, model, metric, batch_size, tag):

    all_inputs = [sample['input'] for sample in dataset.data]
    all_ids = [sample['id'] for sample in dataset.data]
    # import pdb; pdb.set_trace()
    # input = ['我国最大的岛屿是什么？']
    # print(model.generate(input))

    predictions = []
    for i in trange(0, len(all_inputs), batch_size, leave=False):
        batch_ids = all_ids[i : i + batch_size]
        batch_tag = list(map(lambda x: TAG_MAP[x.rsplit('_', 1)[-1]], batch_ids))
        batch_inputs  = all_inputs[i:i+batch_size]

        if tag == True:
            batch_outputs = model.generate(batch_inputs, batch_tag)
        else:
            batch_outputs = model.generate(batch_inputs, None)

        # import pdb; pdb.set_trace()
        predictions.extend(batch_outputs)


    results, all_soft_answer = metric.compute(dataset.data, predictions.copy())

    return results, predictions, all_soft_answer

    


if __name__ == "__main__":
    fire.Fire(main)
