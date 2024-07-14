import time
import os
import torch
import sys
sys.path.append(os.path.abspath('../..'))
from utils import get_dataset, average_weights, exp_details
from flServer import FLServer
import argparse
import wandb
import warnings
warnings.filterwarnings("ignore")

# argument parser
def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=5,
                        help="number of rounds of training")
    parser.add_argument('--num_clients', type=int, default=10,
                        help="number of clients: K")
    parser.add_argument('--frac', type=float, default=0.4,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=5,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', type=int, default=1, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--gpu_id', type=int, default=0, help='random seed')
    # mechanisms arguments 

    # DP arguments
    parser.add_argument('--dp', type=int, default=0, help='whether to use dp')
    parser.add_argument('--dp_batch_size', type=int, default=10,
                        help="the batch size for dp model")
    parser.add_argument('--dp_epsilon', type=float, default=5.0,
                        help="the epsilon for dp model")
    parser.add_argument('--dp_relax', type=int, default=0,
                        help="whether to use relaxed dp")

    # Adversarial training arguments (PGD)
    parser.add_argument('--adv', type=int, default=0, help='whether to use adversarial training')
    parser.add_argument('--pgd_eps', type=float, default=0.25, help='epsilon for PGD')
    parser.add_argument('--pgd_attack_steps', type=int, default=25, help='number of steps for PGD')
    parser.add_argument('--pgd_step_size', type=float, default=0.01, help='step size for PGD')
    # relaxed adversarial training arguments
    parser.add_argument('--adv_relax', type=int, default=0, help='whether to use relaxed adversarial training')
    parser.add_argument('--adv_percent', type=float, default=1, help='percentage of adversarial examples in a batch')
    # Watermarking arguments
    parser.add_argument('--wm', type=int, default=0, help='whether to use watermarking')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    args = args_parser()
    # initiating wandb
    wandb.init(
        project="fl-final-all",
        name= str(args.dataset)+'_'+str(args.model)+'_'+str(args.num_clients)+'_'+str(args.frac)+'_'+str(args.local_ep)+'_'+str(args.local_bs)+'_'+str(args.lr)+'_'+str(args.dp)+'_'+str(args.adv)+'_'+str(args.wm),
        config={
            "epochs": args.epochs,
            "num_clients": args.num_clients,
            "frac": args.frac,
            "local_ep": args.local_ep,
            "local_bs": args.local_bs,
            "lr": args.lr,
            "model": args.model,
            "dp_epsilon": args.dp_epsilon,
            "dataset": args.dataset,
            "optimizer": args.optimizer,
            "pgd_epsilon": args.pgd_eps,
            "pgd_attack_steps": args.pgd_attack_steps,
            "pgd_step_size": args.pgd_step_size,
            "dp_relax": args.dp_relax,
            "dp": args.dp,
            "adv_percent": args.adv_percent,
            "adv_relax": args.adv_relax,
            "adv": args.adv,
            "wm": args.wm
        }
    )

    # checking and initiliasing GPU
    if args.gpu:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # FL training
    global_server = FLServer(args) # initialising FL server
    global_server.train() # training the global model
    global_server.evaluate() # evaluating the performance on the test dataset 
    if args.adv: # evaluating the performance on the adversarial dataset 
        global_server.evaluate_adversarial() 
    if args.wm: # evaluating the performance on the watermarking dataset 
        global_server.evaluate_watermark()
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    wandb.log({"Total Run Time": time.time()-start_time})
    wandb.finish()