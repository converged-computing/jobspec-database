#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np

from config import Config
from utils import setup_custom_logger, read_data
from embed import multilevel_embed, call_baselines
from coarsen import mile_match, confmile_match
from refine_model import GCN, GCN_conf
from evaluate import *
import multiprocessing
import tensorflow as tf
import importlib
import json


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--task', required=False, default='nc', choices=['nc', 'lp'],
                        help='Downstream task (nc or lp)')
    parser.add_argument('--data', required=False, default='pokec-n',
                        choices=['citeseer', 'pokec-n'],
                        help='Input graph file')
    parser.add_argument('--embed-dim', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--basic-embed', required=False, default='node2vec',
                        choices=['deepwalk', 'node2vec', 'grarep', 'netmf'],
                        help='The basic embedding method. If you added a new embedding method, please add its name to choices')
    parser.add_argument('--baseline', action='store_true',
                        help='Call the method stand-alone.')
    parser.add_argument('--train-ratio', default=0.5, type=float,
                        help='MAX number of levels of coarsening.')
    parser.add_argument('--valid', action='store_true', default=False,
                        help='Use validation data.')

    ''' Parameters for Coarsening '''
    parser.add_argument('--coarsen-level', default=8, type=int,
                        help='MAX number of levels of coarsening.')

    ''' Parameters for Refinement'''
    parser.add_argument('--refine-type', required=False, default='conf',
                        choices=['gcn', 'conf'],
                        help='The method for refining embeddings.')
    parser.add_argument('--lambda-fl', default=0.99, type=float,
                        help='Weight of cross entropy.')
    parser.add_argument('--workers', default=multiprocessing.cpu_count(), type=int,
                        help='Number of workers.')
    parser.add_argument('--double-base', action='store_true',
                        help='Use double base for training')
    parser.add_argument('--epoch', default=1000, type=int,
                        help='Epochs for training the refinement model')
    parser.add_argument('--report-epoch', default=100, type=int,
                        help='Frequency of showing training losses')
    parser.add_argument('--valid-epoch', default=100, type=int,
                        help='Frequency of showing validation losses')
    parser.add_argument('--learning-rate', '--lr', default=0.001, type=float,
                        help='Learning rate of the refinement model')
    parser.add_argument('--self-weight', default=0.05, type=float,
                        help='Self-loop weight for GCN model.')  # usually in the range [0, 1]

    ''' Parameters for Conformal Predictions '''
    parser.add_argument('--alpha', default=0.1, type=float,
                        help='Miscoverage rate')
    parser.add_argument(
        "--use_aps_epsilon",
        action="store_true",
        default=True,
        help="Use uniform probability for APS score"
    )
    parser.add_argument(
        "--diffusion_param",
        default=0.1,
        type=float,
        help="diffusion parameter for DAPS",
    )


    ''' Storing result and embedding'''
    parser.add_argument('--seed', type=int, default=20,
                        help='Random seed.')
    parser.add_argument('--jobid', default=0, type=int,
                        help='slurm job id')
    parser.add_argument('--store-embed', action='store_true',
                        help='Store the embeddings.')
    parser.add_argument('--no-eval', action='store_true',
                        help='Evaluate the embeddings.')
    parser.add_argument('--only-eval', action='store_true',
                        help='Evaluate existing embeddings.')
    parser.add_argument('--no-json', action='store_true',
                        help='No json of results stored.')

    args = parser.parse_args()
    return args


def sync_config_args(ctrl, args, graph):
    # General
    ctrl.dataset = args.data
    ctrl.logger = setup_custom_logger('ConfMILE')
    ctrl.embed_dim = args.embed_dim
    ctrl.coarsen_level = args.coarsen_level
    ctrl.seed = args.seed

    # Coarsening
    ctrl.coarsen_to = max(1, graph.node_num // (2 ** args.coarsen_level))
    ctrl.max_node_wgt = int((5.0 * graph.node_num) / ctrl.coarsen_to)

    # Embedding
    ctrl.basic_embed = args.basic_embed
    ctrl.workers = args.workers

    # Refinement
    ctrl.refine_type = args.refine_type
    ctrl.refine_model.num_classes = graph.label_range
    ctrl.refine_model.double_base = args.double_base
    ctrl.refine_model.use_valid = args.valid
    ctrl.refine_model.epoch = args.epoch
    ctrl.refine_model.report_epoch = args.report_epoch
    ctrl.refine_model.valid_epoch = args.valid_epoch
    ctrl.refine_model.learning_rate = args.learning_rate
    ctrl.refine_model.lda = args.self_weight
    ctrl.refine_model.lambda_fl = args.lambda_fl

    # Baselines
    ctrl.baseline = args.baseline

    # Evaluation
    ctrl.alpha = args.alpha
    ctrl.use_aps_epsilon = args.use_aps_epsilon
    ctrl.diffusion_param = args.diffusion_param
    ctrl.only_eval = args.only_eval

    ctrl.logger.info(args)



def select_base_embed(ctrl):
    mod_path = "base_embed_methods." + ctrl.basic_embed
    embed_mod = importlib.import_module(mod_path)
    embed_func = getattr(embed_mod, ctrl.basic_embed)
    return embed_func


def select_refine_model(ctrl):
    refine_model = None
    if ctrl.refine_type == 'gcn':
        refine_model = GCN
    elif ctrl.refine_type == 'conf':
        refine_model = GCN_conf
    else:
        raise NotImplementedError
    return refine_model


def evaluate_embeddings(ctrl, truth_mat, embeddings, sens, sens_num, sens_dim, attr_range):
    '''
    Evaluation for node classification
    :param ctrl:
    :param truth_mat:
    :param embeddings:
    :param sens:
    :param sens_num:
    :param sens_dim:
    :param attr_range:
    :return:
    '''
    # idx_arr = truth_mat[:, 0].reshape(-1)  # this is the original index
    # raw_truth = truth_mat[:, 1:]  # multi-class result
    # embeddings = embeddings[idx_arr, :]  # in the case of yelp, only a fraction of data contains label.
    # # res, entry = eval_oneclass_clf(ctrl, embeddings, truth, sens, sens_num, sens_dim, attr_range, fold=fold, no_sample=truth.shape[1] == 1)
    # if len(np.unique(raw_truth)) == 2:
    #     res, entry = eval_oneclass_clf(ctrl, embeddings, raw_truth, sens, sens_num, sens_dim, attr_range, seed=ctrl.seed)
    # else:
    #     res, entry = eval_multilabel_clf(ctrl, embeddings, raw_truth, sens, sens_num, sens_dim, attr_range, seed=ctrl.seed)
    # print(res)
    # return entry
    pass


def get_filename(args):
    if not os.path.exists('results'):
        os.mkdir('results')
    return f'results/{args.data}_' + \
               f'ConfMILE-{args.coarsen_level}-' + \
               args.basic_embed + \
               f'_{args.embed_dim}_{args.lambda_fl}_{args.seed}'

def store_files(args, probs, embeddings):
    filename = get_filename(args)
    embed_name = filename + '.embeddings'
    with open(embed_name, 'wb') as f:
        np.save(f, embeddings)
    probs_name = filename + '.probs'
    with open(probs_name, 'wb') as f:
        np.save(f, probs)


if __name__ == '__main__':
    ctrl = Config()
    args = parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    graph = read_data(args)
    fair_eval = False
    if os.path.exists(os.path.join('datasets', args.data, 'processed/sens.npy')):
        fair_eval = True
        graph.sens = np.load(os.path.join('datasets', args.data, 'processed/sens.npy'))
    sync_config_args(ctrl, args, graph)
    ctrl.fair_eval = fair_eval

    if ctrl.only_eval:
        ctrl.embed_time = 0
        embeddings = np.load(get_filename(args) + '.embeddings')
        probs = np.load(get_filename(args) + '.probs')
    else:
        base_embed = select_base_embed(ctrl)
        if ctrl.baseline is False:
            # Select coarsening method and refinement model
            coarse_method = confmile_match
            refine_model = select_refine_model(ctrl)

            # generate embeddings
            embeddings, probs = multilevel_embed(ctrl, graph, coarse_method, base_embed, refine_model)
        else:
            raise NotImplementedError

    # Store embeddings
    if not ctrl.only_eval and args.store_embed:
        store_files(args, probs, embeddings)
    # evaluate
    entry = None
    if not args.no_eval:
        entry = conformal_evaluation(ctrl, probs, graph)
        print_summary(ctrl, graph, entry)

    # write to json
    if entry and not args.no_json:
        filename = os.path.join('results', 'json', f'{args.jobid}.json')
        if os.path.exists(filename):
            fr = open(filename)
            jd = json.load(fr)
            fr.close()
        else:
            jd = json.loads("""{"results": []}""")

        entry['dataset'] = args.data
        entry['baseline'] = 'ConfMILE'
        # ConfMILE specific parameters
        entry['method'] = args.basic_embed
        entry['dimension'] = args.embed_dim
        entry['c-level'] = '*' if args.baseline else args.coarsen_level
        entry['lr'] = args.learning_rate
        entry['epoch'] = args.epoch
        entry['lambda'] = args.lambda_fl
        entry['seed'] = args.seed

        jd['results'].append(entry)
        js = json.dumps(jd, indent=2)

        fw = open(filename, 'w')
        fw.write(js)
        fw.close()
