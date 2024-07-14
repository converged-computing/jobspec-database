import os
import sys
import argparse
import pprint
import torch
from torch.autograd import Variable
from data import dataloader
from run_networks import model
import warnings
from utils import source_import

from utils import source_import
from shuffler.lib.utils import testUtils
from shuffler.interface.pytorch import datasets

import torch
from torchvision import transforms

# ================
# LOAD CONFIGURATIONS


db_file = '/path/to/file'
rootdir = '/path/to/dir'

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/stamps/stage_2_meta_embedding.py', type=str)
parser.add_argument('--test', default=True, action='store_true')
parser.add_argument('--test_open', default=True, action='store_true')
parser.add_argument('--output_logits', default=True)
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
training_opt['batch_size'] = 1

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

data = datasets.ObjectDataset(db_file,
                            rootdir=rootdir,
                            mode='w',
                            used_keys=['image', 'objectid'],
                            transform_group={'image': transform})
 

data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=training_opt['batch_size'],
                                          shuffle=False,
                                          num_workers=1)

training_model = model(config, data_loader, data_loader, data_loader, test=True)
training_model.load_model()

memory = config['memory']
training_model.infer(data, phase='test', openset=test_open)

# if not os.path.isdir(training_opt['log_dir']):
#     os.makedirs(training_opt['log_dir'])

# print('Loading dataset from: %s' % data_root[dataset.rstrip('_LT')])
# pprint.pprint(config)


# print('Under testing phase, we load training data simply to calculate training data number for each class.')

# data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')], dataset=dataset, phase=x,
#                                 batch_size=training_opt['batch_size'],
#                                 sampler_dic=None, 
#                                 test_open=test_open,
#                                 num_workers=training_opt['num_workers'],
#                                 shuffle=False)
#         for x in ['train', 'test']}


# training_model = model(config, data, test=True)
# training_model.load_model()
#for training_model in training_model.networks.values():
#    training_model.eval()
#training_model.eval()

# memory = config['memory']
# training_model.eval(phase='test', openset=test_open)

#for inputs, labels, paths in (data['test']):
#    with torch.no_grad():
#        inputs = Variable(inputs)
#        labels = Variable(labels)
        #print(inputs)
#        print(labels)
#        print(paths)
#        preds = training_model(inputs, centroids=memory['centroids'])
#        print(preds)
        


#training_model.eval(phase='test', openset=test_open)

    
# if output_logits:
#     training_model.output_logits(openset=test_open)
    
print('ALL COMPLETED.')
