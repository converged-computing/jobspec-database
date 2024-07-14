import os
import sys
import argparse
import pprint
from data import dataloader
from run_networks import model
import warnings
import logging
import numpy as np
import sklearn.model_selection
from sklearn import preprocessing 

from utils import source_import
from shuffler.lib.utils import testUtils
from shuffler.interface.pytorch import datasets

import torch
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

# logging.basicConfig(level=1)


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/stamps/stage_1.py', type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
args = parser.parse_args()

test_mode = args.test
test_open = args.test_open

if test_open:
    test_mode = True

output_logits = args.output_logits

config = source_import(args.config).config
training_opt = config['training_opt']
# change
relatin_opt = config['memory']
dataset = training_opt['dataset']

db_file = training_opt['db_file']
rootdir = training_opt['rootdir']

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

pprint.pprint(config)

if not test_mode:

    sampler_defs = training_opt['sampler']
    if sampler_defs:
        sampler_dic = {'sampler': source_import(sampler_defs['def_file']).get_sampler(), 
                       'num_samples_cls': sampler_defs['num_samples_cls']}
    else:
        sampler_dic = None


    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    

    data = datasets.ObjectDataset(db_file,
                            rootdir=rootdir,
                            where_object="name NOT LIKE '%page%'",
                            mode='r',
                            used_keys=['image', 'objectid', 'name', 'num_instances', 'name_id'],
                            transform_group={'image': transform, 'name_id': lambda x: int(x)})
    dataset_size = data.__len__()


    print("\nTotal number of samples", dataset_size)

    training_opt['num_classes'] = max([item["name_id"] for item in data])
    print(training_opt['num_classes'])

    open_set = [item for item in data if item["name_id"]==-1]
    train_val_set = [item for item in data if item["name_id"]!=-1]

    validation_split = .2
    random_seed = 1
    shuffle = True
    indices = list(range(train_val_set.__len__()))
    split = int(np.floor(validation_split * train_val_set.__len__()))
    # if shuffle :
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    print("Training samples ", len(train_sampler))
    print("Validation samples ", len(valid_sampler))
    #print("Testing samples ", len(test_y))

    full_dataloader = torch.utils.data.DataLoader(train_val_set,
                                              batch_size=training_opt['batch_size'],
                                              num_workers=training_opt['num_workers'])

    train_dataloader = torch.utils.data.DataLoader(train_val_set,
                                              batch_size=training_opt['batch_size'],
                                              num_workers=training_opt['num_workers'], sampler=train_sampler)

    

    val_dataloader = torch.utils.data.DataLoader(train_val_set,
                                              batch_size=training_opt['batch_size'],
                                              num_workers=training_opt['num_workers'], sampler=valid_sampler)


    training_model = model(config, full_dataloader, train_dataloader, val_dataloader, test=False)

    training_model.train()
    #dataset.close()


else:

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    print('Under testing phase, we load training data simply to calculate training data number for each class.')

    # data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')], dataset=dataset, phase=x,
    #                                 batch_size=training_opt['batch_size'],
    #                                 sampler_dic=None, 
    #                                 test_open=test_open,
    #                                 num_workers=training_opt['num_workers'],
    #                                 shuffle=False)
    #         for x in ['train', 'test']}

    
    # training_model = model(config, data, test=True)
    # training_model.load_model()
    # training_model.eval(phase='test', openset=test_open)
    # #print(training_model)
    
    # if output_logits:
    #     training_model.output_logits(openset=test_open)
        
print('ALL COMPLETED.')
