import numpy as np

import data_providers as data_providers
from arg_extractor import get_args
from data_augmentations import Cutout
from experiment_builder import ExperimentBuilder
from model_architectures import ConvolutionalNetwork

args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

from torchvision import transforms
import torch

torch.manual_seed(seed=args.seed)  # sets pytorch's seed

##original transforms
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     Cutout(n_holes=1, length=14),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

#trial - no transforms except normalizing
transform_train = transforms.Compose([
    transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = data_providers.CIFAR10(root='data', set_name='train', download=False, transform=transform_train)
styled = data_providers.AugmentedCIFAR10(path='./data/Monet',transform=transform_train)

concatenated = torch.utils.data.ConcatDataset([trainset, styled])

train_data = torch.utils.data.DataLoader(concatenated, batch_size=100, shuffle=True, num_workers=4)

valset = data_providers.CIFAR10(root='data', set_name='val', download=False, transform=transform_test)
val_data = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=4)

testset = data_providers.CIFAR10(root='data', set_name='test', download=False, transform=transform_test)
test_data = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_output_classes = 10

custom_conv_net = ConvolutionalNetwork(num_output_classes=num_output_classes)  # initialize our network object, in this case a ConvNet


conv_experiment = ExperimentBuilder(network_model=custom_conv_net, use_gpu=args.use_gpu,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    learning_rate=args.learning_rate,
                                    weight_decay_coefficient = args.weight_decay,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data)  # build an experiment object
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
