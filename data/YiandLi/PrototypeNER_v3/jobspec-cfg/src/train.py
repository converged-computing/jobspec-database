import argparse
import logging
import os
import sys
import warnings
from copy import deepcopy

import torch
from transformers import BertTokenizer, BertModel

warnings.filterwarnings('ignore')
sys.path.append("./")

from src.utils import set_seeds, print_args, get_model_path_name
from src.data_utils import read_and_load_data

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--use_reptile", action="store_true")
parser.add_argument("--use_prior", action="store_true")
parser.add_argument("--do_source_train", action="store_true")
parser.add_argument("--do_target_eval", action="store_true")
parser.add_argument("--froze_encoder", action="store_true")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--downsize_lr", default=4e-4, type=float)
parser.add_argument("--encoder_lr", default=3e-5, type=float)
parser.add_argument("--max_output_loops", default=100, type=int)
parser.add_argument("--update_per_tasks", default=1, type=int)
parser.add_argument("--eval_per_update", default=1, type=int)
parser.add_argument("--support_test_epoch", default=1, type=int)
parser.add_argument("--inner_identify_loops", default=100, type=int)  # 每个 task 训练多少轮 identify ( circle loss
parser.add_argument("--inner_classify_loops", default=1, type=int)  # 每个 task 训练多少轮 classifier ( prototype
parser.add_argument("--inner_batch_size", default=32, type=int)  # 每个 task 多少 setence 进行训练，对 proto 很关键，要大于 n way 数量
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--encoder_type", default="bert-base-uncased", type=str)
# parser.add_argument("--distribution_loss", default="contrastive_L2", type=str, help="in # {contrastive_dot, contrastive_L2, margin_loss}")
parser.add_argument("--save_model_dir", default="model_dir", type=str)
parser.add_argument("--output_dir", default="output", type=str)
parser.add_argument("--source_dir", default="data/episode-data/inter", type=str)
parser.add_argument("--target_dir", default="data/episode-data/inter", type=str)
parser.add_argument("--way_num", default=5, type=int)
parser.add_argument("--shot_num", default=1, type=int)
parser.add_argument("--down_size", default=128, type=int)
parser.add_argument("--post_avg_by_class", action="store_true")
parser.add_argument("--prior_avg_by_class", action="store_true")

for _ in range(5):
    logger.info("===")
args = parser.parse_args()
args.source_model_path = get_model_path_name(args)
print_args(args)
set_seeds(args)

from src.train_utils import get_names_and_params, test_meta, load_weights, froze_model, \
    train_meta, update_weight
from src.SourceModel import SourceModel

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained(args.encoder_type)

# Get model
encoder = BertModel.from_pretrained(args.encoder_type)
model = SourceModel(args, encoder).to(device)
model = froze_model(args, model)

# FOR TEST
model = torch.load(args.source_model_path, map_location='cpu')

# Train on Source
if args.do_source_train:
    # Get data
    train_tasks = read_and_load_data(args, "train", tokenizer)
    dev_tasks = read_and_load_data(args, "dev", tokenizer)
    logger.info(
        f"Have processed data from cache/train_{args.way_num}_{args.shot_num}, totally {len(train_tasks)} tasks.\n"
        f"Have processed data from cache/dev_{args.way_num}_{args.shot_num}, totally {len(dev_tasks)} tasks.")
    
    # Begin Training by epoch level
    best_update_loss = 1e99
    best_update_f1 = -1
    
    patience = 0
    outer_update_nums = 0
    
    # Save model's initial parameters
    names, params = get_names_and_params(model)
    initial_weights = deepcopy(params)
    all_tasks_grad = []  # task_num, para_num
    
    while outer_update_nums < args.max_output_loops:
        
        for task_id, train_task in enumerate(train_tasks):  # 一个 task 一个 task 进行训练的
            # logger.info(f"\t task id: {task_id}")
            # # train on one task for n times
            # label2id = train_task['type2id']
            # identify_updated_params, classify_updated_params = train_meta(args, label2id, train_task, model, device)
            #
            # # get its grad, save in all_tasks_grad
            # this_task_grad = []
            # for after_para, init_para in zip(identify_updated_params, initial_weights):
            #     # for after_para, init_para in zip(classify_updated_params, initial_weights):
            #     this_task_grad.append(after_para - init_para)
            # assert len(this_task_grad) == len(initial_weights)
            # all_tasks_grad.append(this_task_grad)
            #
            # # re-load model for next support task
            # load_weights(model, names, initial_weights)
            
            # TODO: Update with the grad
            if (task_id + 1) % args.update_per_tasks == 0:
                # outer_update_nums += 1
                #
                # # update model every several tasks within episode and parameter
                # update_weight(model, names, all_tasks_grad)
                #
                # # refresh saved cache
                # names, params = get_names_and_params(model)
                # initial_weights = deepcopy(params)
                # all_tasks_grad = []
                
                # TODO: Do eval on dev_task
                if outer_update_nums % args.eval_per_update == 0:
                    _prior_f1, _prior_precision, _prior_recall = 0., 0., 0.
                    _ent_f1, _ent_precision, _ent_recall = 0., 0., 0.
                    
                    for _, dev_task in enumerate(dev_tasks[:10]):
                        prior_f1, prior_precision, prior_recall, \
                        ent_f1, ent_precision, ent_recall = test_meta(args, device, model, dev_task)
                        
                        # re-load model for next query task
                        load_weights(model, names, initial_weights)
                        
                        _prior_f1 += prior_f1
                        _prior_precision += prior_precision
                        _prior_recall += prior_recall
                        _ent_f1 += ent_f1
                        _ent_precision += ent_precision
                        _ent_recall += ent_recall
                    
                    avg_pri_f1 = _prior_f1 / 10
                    avg_prior_precision = _prior_precision / 10
                    avg_prior_recall = _prior_recall / 10
                    avg_sample_f1 = _ent_f1 / 10
                    avg_precision = _ent_precision / 10
                    avg_recall = _ent_recall / 10
                    
                    # entity identify & NER
                    logger.info("Outer Update Step {}: \n"
                                "EI_f1 {:.3f} EI_recall {:.3f} EI_precision {:.3f}\n"
                                "EC_f1 {:.3f} EC_recall {:.3f} EC_precision {:.3f}\n".
                                format(outer_update_nums,
                                       avg_pri_f1, avg_prior_recall, avg_prior_precision,
                                       avg_sample_f1, avg_recall, avg_precision))
                    
                    # Save model
                    if avg_sample_f1 > best_update_f1:
                        best_update_f1 = avg_sample_f1
                        if not os.path.exists(os.path.dirname(args.source_model_path)):
                            os.makedirs(os.path.dirname(args.source_model_path))
                        torch.save(model, args.source_model_path)
                        logger.info("Best f1: {:.3f} gotten".format(best_update_f1))
                        logger.info("Source Model saved in {}".format(args.source_model_path))
                        patience = 0
                    else:
                        logger.info(
                            "Outer Update Step {} f1: {:.3f}, not updated ; Best f1: {:.3f}".format(outer_update_nums,
                                                                                                    avg_sample_f1,
                                                                                                    best_update_f1))
                        patience += 1
                        if patience > 10:
                            break

if args.do_target_eval:
    # Load Model
    model = torch.load(args.source_model_path, map_location='cpu')
    model.to(device)
    logger.info(f"Reload Model from {args.source_model_path}")
    
    # Get data
    target_tasks = read_and_load_data(args, "test", tokenizer)
    
    # Save model's initial parameters
    names, params = get_names_and_params(model)
    initial_weights = deepcopy(params)
    
    # Test indicator results list
    f1_list, pre_list, recall_list = [], [], []
    
    # Test
    for i, target_task in enumerate(target_tasks):
        sample_f1, precision, recall = test_meta(args, device, model, target_task)
        f1_list.append(sample_f1)
        pre_list.append(precision)
        recall_list.append(recall)
        logger.info("\tNo{}/{} - f1: {:.3f} ; precision: {:.3f}; recall: {:.3f}".format(i, len(target_tasks),
                                                                                        sample_f1, precision, recall))
        load_weights(model, names, initial_weights)  # reload parameters
    
    logger.info("Avg f1: {:.3f} ; precision: {:.3f}; recall: {:.3f}".
                format(sum(f1_list) / len(f1_list), sum(pre_list) / len(pre_list), sum(recall_list) / len(recall_list)))
