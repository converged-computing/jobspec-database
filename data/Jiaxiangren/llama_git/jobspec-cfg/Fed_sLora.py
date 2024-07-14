from flearn.experiments.slora import *
from flearn.utils.options import flArguments
from transformers import HfArgumentParser
import sys
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path', help='Description for foo argument', required=False)
    parser.add_argument('--select_method', help="how to select layers", required=True)
    parser.add_argument('--select_layer_num', help='how many layers to choose', required=True)
    parser.add_argument('--lr', help='learning rate', required=False)
    parser.add_argument('--momentum', help="how to select layers", required=False)
    parser.add_argument('--prune_ratio', help='percentage of parameters to choose', required=False)
    parser.add_argument('--mask_epochs', help='learning rate', required=False)
    parser.add_argument('--trainable_size', required=False)
    parser.add_argument("--sort_type", help="the data sort_type", required=False)
    parser.add_argument("--data_peace", required=False)
    parser.add_argument("--heter_degree", required=False)
    parser.add_argument("--clients_num", required=False)
    parser.add_argument("--init_ratio", required=False)
    args = parser.parse_args()

    return args




def main():

    arguments = sys.argv
    if len(arguments) < 2:
        assert len(arguments) == 2, "please input the config file..."
    
    hyper_params = get_parser()

    # config_path = arguments[1]
    config_path = hyper_params.path
    parser = HfArgumentParser((flArguments))
    # print(parser)
    training_args = parser.parse_json_file(config_path)[0]

    training_args.select_method = hyper_params.select_method
    training_args.select_layer_num = int(hyper_params.select_layer_num)

    if hyper_params.init_ratio:
        training_args.init_ratio = float(hyper_params.init_ratio)

    if hyper_params.clients_num:
        training_args.num_clients = int(hyper_params.clients_num)

    if hyper_params.data_peace:
        training_args.data_peace_func = hyper_params.data_peace
        
    if hyper_params.heter_degree:
        training_args.dirichlet_alpha = int(hyper_params.heter_degree)

    if hyper_params.lr:
        training_args.learning_rate = float(hyper_params.lr)
        training_args.transfer_learning_rate = 0.5 * training_args.learning_rate
    
    if hyper_params.momentum:
        training_args.momentum = float(hyper_params.momentum)

    if hyper_params.prune_ratio:
        training_args.prune_ratio = float(hyper_params.prune_ratio)
    
    if hyper_params.mask_epochs:
        training_args.mask_epochs = int(hyper_params.mask_epochs)
    
    if hyper_params.trainable_size:
        training_args.trainable_size = int(hyper_params.trainable_size)
    
    if hyper_params.sort_type:
        training_args.sort_type = hyper_params.sort_type
        
    print(training_args)


    t = CentralTraining(training_args, share_percent=training_args.share_percent, iid=training_args.iid, unequal=training_args.unequal, result_dir=training_args.result_dir)
    t.train()


if __name__ == '__main__':
    main()