
# modified from https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_3.ipynb
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pprint

# External module imports

from torch_geometric import seed_everything
from torch_geometric.data.makedirs import makedirs
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.config import (dump_cfg, 
                                             makedirs_rm_exist)
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from graphgps.train.opt_train import Trainer, Trainer_Saint
from graphgps.network.custom_gnn import create_model
from data_utils.load import load_data_nc, load_data_lp
from core.graphgps.utility.utils import create_optimizer, config_device, \
        init_model_from_pretrained, create_logger, set_cfg
from gcns.gsaint_main import get_loader_RW
import wandb 
from sklearn.metrics import *
from core.embedding.tune_utils import (

    FILE_PATH
)
import argparse

set_float = lambda result: float(result.split(' ±')[0])


def merge_cfg_from_sweep(cfg, wandb_config):
    for ind, k in wandb_config.items():
        if hasattr(cfg.model, ind):
            cfg.model.ind = k
        if hasattr(cfg.train, ind):
            cfg.train.ind = k
            
    pprint.pprint(cfg.model)
    pprint.pprint(cfg.train)
    return cfg


def wandb_record_files(path):
    record_or_not = False
    record_lst = [cfg_sweep,
                  cfg_config,
                  'core/gcns/wb_tune.py'
                  ]
    
    for recorded in record_lst:
        if recorded in path:
            record_or_not = True
            break
    return record_or_not

def save_results_to_file(result_dict, cfg, wandb, output_dir):
    """
    Saves the results and the configuration to a CSV file.
    """
    # Create a DataFrame from the result dictionary
    result_df = pd.DataFrame([result_dict])
    
    # Add configuration details as columns
    result_df['ModelType'] = cfg.model.type
    result_df['BatchSize'] = wandb.batch_size
    result_df['BatchSizeSampler'] = wandb.batch_size_sampler
    result_df['HiddenChannels'] = wandb.hidden_channels
    result_df['OutChannels'] = wandb.out_channels
    result_df['NumSteps'] = wandb.num_steps
    result_df['SampleCoverage'] = wandb.sample_coverage
    result_df['WalkLength'] = wandb.walk_length
    result_df['LearningRate'] = wandb.lr
    result_df['Epochs'] = cfg.train.epochs
    
    # Specify the output file path
    output_file = os.path.join(output_dir, 'results_summary.csv')
    
    # Check if file exists to append or write header
    if os.path.exists(output_file):
        result_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        result_df.to_csv(output_file, mode='w', header=True, index=False)
    
    print(f"Results saved to {output_file}")

def run_experiment():  # sourcery skip: avoid-builtin-shadow

    output_dir = os.path.join(FILE_PATH, "results_vgae")
    os.makedirs(output_dir, exist_ok=True)

    id = wandb.util.generate_id()
    
    wandb.init(id=id, config=cfg_sweep, settings=wandb.Settings(_service_wait=300), save_code=True)

    wandb_config = wandb.config
    
    wandb.log(dict(wandb_config))   
    for k, val in wandb_config.items():
        print(k, val)
        
    # merge model param
    cfg = merge_cfg_from_sweep(cfg_config, cfg_sweep)
    
    torch.set_num_threads(cfg.run.num_threads)
    splits, _, data = load_data_lp[cfg.data.name](cfg.data)
    cfg.model.in_channels = splits['train'].x.shape[1]

    cfg.train.batch_size = wandb_config.batch_size
    model = create_model(cfg)
    
    wandb.watch(model, log="all",log_freq=10)
    optimizer = create_optimizer(model, cfg)
    loggers = create_logger(1)

    seed_everything(cfg.seed)
    cfg = config_device(cfg)

    # LLM: finetuning
    if cfg.train.finetune: 
        model = init_model_from_pretrained(model, cfg.train.finetune,
                                            cfg.train.freeze_pretrained)

    print(cfg.model.type)
    if cfg.model.sampler == 'gsaint':
            trainer = Trainer_Saint(
        FILE_PATH=FILE_PATH,
        cfg=cfg, 
        model=model,
        emb=None,
        data=data,
        optimizer=optimizer,
        splits=splits, 
        run=0, 
        repeat=args.repeat,
        loggers=loggers,
        print_logger=None,
        device=cfg.device,
        gsaint=get_loader_RW, 
        batch_size_sampler=wandb_config.batch_size_sampler, 
        walk_length=wandb_config.walk_length, 
        num_steps=wandb_config.num_steps, 
        sample_coverage=wandb_config.sample_coverage
        )
    else:
        trainer = Trainer(FILE_PATH,
                    cfg,
                    model, 
                    None, 
                    data,
                    optimizer,
                    splits,
                    0, 
                    args.repeat,
                    loggers,
                    None, 
                    cfg.device)

    for epoch in range(1, cfg.train.epochs + 1):
        loss = trainer.train_func[cfg.model.type]()
        
        if epoch % 100 == 0:
            results_rank = trainer.merge_result_rank()
            # print(results_rank)
            
            if results_rank["AUC"][1] > best_auc:
                best_auc = results_rank["AUC"][1]
            elif results_rank['Hits@100'][1] > best_hit100:
                best_hits100 = results_rank['Hits@100'][1]
                
            for key, result in results_rank.items():
                trainer.loggers[key].add_result(0, result)
                
            for key, result in results_rank.items():
                if key in ['Hits@20', 'AUC', 'MRR']:
                    train_hits, valid_hits, test_hits = result
                    print(
                        f'Run: {0 + 1:02d}, '
                            f'Key: {key}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_hits:.2f}%, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
            print('---')
                    
    result_dict = {}
    for key in loggers:
        _, _, _, valid_test, _, _ = trainer.loggers[key].calc_all_stats()
        result_dict.update({key: valid_test})
    
    
    trainer.save_result(result_dict)
    save_results_to_file(result_dict, cfg, wandb_config, output_dir)
    wandb.log({'Hits@20': set_float(result_dict['Hits@100'])})
    
    wandb.log({'best hits100': best_hits100})
    wandb.log({'best auc': best_auc})
    return  



# cfg_sweep= 'core/yamls/cora/gcns/gae_sp1.yaml'
# cfg_config = 'core/yamls/cora/gcns/gae.yaml'

# cfg_sweep= 'core/yamls/pubmed/gcns/gae_sp1.yaml'
# cfg_config = 'core/yamls/pubmed/gcns/gae.yaml'

# cfg_sweep= 'core/yamls/arxiv_2023/gcns/gae_sp1.yaml'
# cfg_config = 'core/yamls/arxiv_2023/gcns/gae.yaml'


# cfg_sweep= 'core/yamls/ogbn-arxiv/gcns/gae_sp1.yaml'
# cfg_config = 'core/yamls/ogbn-arxiv/gcns/gae.yaml'

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default= 'core/yamls/pubmed/gcns/gat.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/pubmed/gcns/gat_sp1.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=False, default='ogbn-arxiv',
                        help='name of data for hyper tune.')   
    parser.add_argument('--repeat', type=int, default=4,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('--device',  type=int, default=2,
                        help='device index.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()

import torch

args = parse_args()

print(args)

cfg_sweep = set_cfg(FILE_PATH, args.sweep_file)
cfg_config = set_cfg(FILE_PATH, args.cfg_file)

cfg_config.data.name = args.data
cfg_config.data.device = args.device
cfg_config.model.type = 'GAE'

pprint.pprint(cfg_config)
sweep_id = wandb.sweep(sweep=cfg_sweep, project=f"{cfg_config.model.type}-sweep-{cfg_config.data.name}")



wandb.agent(sweep_id, run_experiment, count=30)
# CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID

