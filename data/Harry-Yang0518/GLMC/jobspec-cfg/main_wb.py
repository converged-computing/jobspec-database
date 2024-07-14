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
from Trainer import Trainer
from model import Resnet_LT
from model import ResNet_new
from imbalance_data import cifar10Imbanlance, cifar100Imbanlance, dataset_lt_data
from imbalance_data.data import get_dataset

best_acc1 = 0


def get_model(args):
    if args.dataset == "ImageNet-LT" or args.dataset == "iNaturelist2018":
        print("=> creating model '{}'".format('resnext50_32x4d'))
        net = Resnet_LT.resnext50_32x4d(num_classes=args.num_classes)
        return net
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch == 'resnet50':
            net = ResNet_new.resnet50(args=args)
        if args.arch == 'iresnet50':
            net = ResNet_new.iresnet50(args=args)
        elif args.arch == 'resnet18':
            net = ResNet_new.resnet18(args=args)
        elif args.arch == 'mresnet32':
            net = ResNet_new.mresnet32(args=args)
        elif args.arch == 'resnet34':
            net = ResNet_new.resnet34(args=args)

        if args.dataset == 'fmnist':
            net.conv1[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        return net


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
    wandb.init(project='lg3_'+args.dataset,
               name= args.store_name.split('/')[-1]
               )
    wandb.config.update(args)
    main_worker(wandb.config)


def main_worker(args):
    global best_acc1

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
    if torch.cuda.is_available():
        model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # ================= Data loading code =================
    train_dataset, val_dataset = get_dataset(args)
    # train_dataset_minority = copy(train_dataset)
    # change train_dataset_minority.data to keep the minory class only
    num_classes = len(np.unique(train_dataset.targets))
    assert num_classes == args.num_classes

    # ================= Default Loader
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None), num_workers=args.workers,
                                               persistent_workers=True, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, persistent_workers=True, pin_memory=True)

    # ================= weighted_loader
    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    cls_num_list = np.array(cls_num_list)

    cls_weight = 1.0 / (cls_num_list ** args.resample_weighting)
    cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
    samples_weight = np.array([cls_weight[t] for t in train_dataset.targets])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    weighted_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                        num_workers=args.workers, persistent_workers=True,
                                                        pin_memory=True, sampler=weighted_sampler)

    start_time = time.time()
    print("Training started!")
    trainer = Trainer(args, model=model, train_loader=train_loader, val_loader=val_loader,
                      weighted_train_loader=weighted_train_loader, per_class_num=cls_num_list, log=logging)
    trainer.train_base()
    end_time = time.time()
    print("It took {} to execute the program".format(hms_string(end_time - start_time)))


if __name__ == '__main__':
    # train set
    parser = argparse.ArgumentParser(description="Global and Local Mixture Consistency Cumulative Learning")
    parser.add_argument('--dataset', type=str, default='cifar100', help="cifar10,cifar100,ImageNet-LT,iNaturelist2018")
    parser.add_argument('--root', type=str, default='../dataset/', help="dataset setting")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32') # , choices=('resnet18', 'resnet34', 'resnet32', 'resnet50', 'resnext50_32x4d'))
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes ')
    parser.add_argument('--imbalance_rate', default=1.0, type=float, help='imbalance factor')
    parser.add_argument('--imbalance_type', default='null', type=str, help='imbalance type')

    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=5e-3, type=float, metavar='W', help='weight decay (default: 5e-3、2e-4、1e-4)', dest='weight_decay')

    parser.add_argument('--resample_weighting', default=0.0, type=float, help='weighted for sampling probability (q(1,k))')
    parser.add_argument('--label_weighting', default=1.0, type=float, help='weighted for Loss')
    parser.add_argument('--contrast_weight', default=10, type=int, help='Mixture Consistency  Weights')
    parser.add_argument('--beta', type=float, default=0.5, help="augment mixture")

    parser.add_argument('--feat', type=str, default='none')  # none|nn1|nn2
    parser.add_argument('--norm', default=False, action='store_true')  # none|nn1|nn2
    parser.add_argument('--bias', default=False, action='store_true')  # none|nn1|nn2
    parser.add_argument('--loss', type=str, default='ce')  # ce|ls|ceh|hinge
    parser.add_argument('--margins', type=str, default='1.0_0.5_0.0')
    parser.add_argument('--eps', type=float, default=0.05)  # for ls loss
    parser.add_argument('--etf_cls', default=False, action='store_true')
    parser.add_argument('--aug', default='null', help='data augmentation')  # null | pc (padded_random_crop)
    parser.add_argument('--mixup', type=int, default=-1, help='flag for using mixup, -1 means no mixup')
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='alpha parameter for mixup')

    # etc.
    parser.add_argument('--seed', default=3407, type=int, help='seed for initializing training. ')
    parser.add_argument('-p', '--print_freq', default=1000, type=int, metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--root_model', type=str, default='./output/')
    parser.add_argument('--store_name', type=str, default='name')
    parser.add_argument('--debug', type=int, default=10)
    parser.add_argument('--knn', default=False, action='store_true')
    args = parser.parse_args()

    if args.dataset == 'cifar10' or args.dataset == 'fmnist':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'ImageNet-LT':
        args.num_classes = 1000
    elif args.dataset == 'iNaturelist2018':
        args.num_classes = 8142
    elif args.dataset == 'tinyi':
        args.num_classes = 200
    if args.imbalance_rate < 1.0: 
        args.knn = True
    args.margins = [float(a) for a in args.margins.split('_')]

    curr_time = datetime.datetime.now()
    file_name = args.store_name
    args.store_name = '{}_{}/{}/{}'.format(
        args.dataset, args.arch,
        str(args.imbalance_rate),
        file_name
    )

    main(args)