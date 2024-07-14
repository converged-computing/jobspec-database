import sys
import time
import wandb
import logging
import datetime
import argparse
from torch.backends import cudnn
import torch.nn as nn

from utils import util
from utils.util import *
from Trainer_nc import Trainer
from model import model_nc
from imbalance_data.data import get_dataset, get_dataset_balanced

best_acc1 = 0


def get_model(args):
    if args.arch.lower() == 'mlp':
        model = model_nc.MLP(hidden=args.width, depth=args.depth, fc_bias=args.bias, num_classes=args.num_classes)
    else:
        model = model_nc.ResNet(pretrained=False, num_classes=args.num_classes, backbone=args.arch, args=args)
    return model


def main(args):
    print(args)
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

    os.environ["WANDB_API_KEY"] = "0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee"
    os.environ["WANDB_MODE"] = "online" #"dryrun"
    os.environ["WANDB_CACHE_DIR"] = "/scratch/lg154/sseg/.cache/wandb"
    os.environ["WANDB_CONFIG_DIR"] = "/scratch/lg154/sseg/.config/wandb"
    wandb.login(key='0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee')
    wandb.init(project="NC2_"+str(args.dataset),
               name= args.store_name.split('/')[-1]
               )
    wandb.config.update(args)
    main_worker(args.gpu, wandb.config)


def main_worker(gpu, args):
    global best_acc1

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M')
    fh = logging.FileHandler(os.path.join(args.root_model + args.store_name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)
    logging.info(args)

    # ==================== create model
    model = get_model(args)
    _ = print_model_param_nums(model=model)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # ================= Data loading code
    if args.imbalance_type == 'exp' or args.imbalance_type == 'step':
        train_dataset, val_dataset = get_dataset(args)
    else:
        train_dataset, val_dataset = get_dataset_balanced(args)
    # num_classes = len(np.unique(train_dataset.targets))
    # assert num_classes == args.num_classes

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, persistent_workers=True, pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False,
                                             num_workers=args.workers, persistent_workers=True, pin_memory=True)

    start_time = time.time()
    print("Training started!")
    trainer = Trainer(args, model=model, train_loader=train_loader, val_loader=val_loader, log=logging)
    trainer.train_base()
    end_time = time.time()
    print("It took {} to execute the program".format(hms_string(end_time - start_time)))


if __name__ == '__main__':
    # train set
    parser = argparse.ArgumentParser(description="Global and Local Mixture Consistency Cumulative Learning")
    parser.add_argument('--dataset', type=str, default='cifar100', help="cifar10,cifar100,stl10")
    parser.add_argument('--root', type=str, default='../dataset/', help="dataset setting")
    parser.add_argument('--aug', default='null', help='data augmentation')  # null | pc (padded_random_crop)
    parser.add_argument('--coarse', default=False, action='store_true')
    parser.add_argument('--imbalance_rate', type=float, default=1.0)
    parser.add_argument('--imbalance_type', type=str, default='null')  # null | step | exp

    # model structure
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32', choices=('resnet18', 'resnet34', 'resnet32', 'resnet50', 'resnext50_32x4d'))
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes ')
    parser.add_argument('--norm', type=str, default='bn', help='Type of norm layer')  # bn|gn

    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--scheduler', type=str, default='ms')
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--end_lr', type=float, default=0.00001)  # poly LRD
    parser.add_argument('--power', type=float, default=2.0)       # poly LRD
    parser.add_argument('--decay_epochs', type=int, default=400)

    parser.add_argument('--epochs', default=800, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-3、2e-4、1e-4)', dest='weight_decay')

    parser.add_argument('--loss', type=str, default='ce')   # ce|ls|ceh|hinge
    parser.add_argument('--eps', type=float, default=0.05)  # for ls loss
    parser.add_argument('--etf_cls', default=False, action='store_true')
    parser.add_argument('--mixup', type=int, default=-1, help='flag for using mixup, -1 means no mixup')
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='alpha parameter for mixup')

    # MLP settings (only when using mlp and res_adapt(in which case only width has effect))
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--no-bias', dest='bias', default=True, action='store_false')

    # etc.
    parser.add_argument('--seed', default=3407, type=int, help='seed for initializing training. ')
    parser.add_argument('-p', '--print_freq', default=1000, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--root_model', type=str, default='./result2/')
    parser.add_argument('--store_name', type=str, default='name')
    parser.add_argument('--debug', type=int, default=10)
    args = parser.parse_args()

    if args.dataset == 'cifar10' or args.dataset == 'fmnist':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        if args.coarse:
            args.num_classes = 20
        else:
            args.num_classes = 100
    elif args.dataset == 'stl10':
        args.num_classes = 10
    elif args.dataset == 'ImageNet-LT':
        args.num_classes = 1000
    elif args.dataset == 'iNaturelist2018':
        args.num_classes = 8142
    elif args.dataset == 'tinyi':
        args.num_classes = 200
    
    if args.batch_size < 64: 
        args.lr = args.lr*(args.batch_size/64)
        print("has modified the learning rate to {}".format(args.lr))

    curr_time = datetime.datetime.now()
    file_name = args.store_name
    args.store_name = '{}_{}/{}'.format(
        args.dataset, args.arch,
        file_name
    )

    main(args)