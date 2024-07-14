
import os
import numpy as np
from cub2011 import Cub2011
from pytorch_mlp_framework.experiment_builder import ExperimentBuilder
from pytorch_mlp_framework.arg_extractor import get_args

from PIL import Image
import timm

import torch
from torchvision import transforms, datasets
from torchvision.models import resnet50, vit_b_16, ViT_B_16_Weights, ResNet50_Weights

def get_dataloaders(dataloader, train_transform, test_transform, batch_size=32):
    if dataloader == 'aircrafts':
        train_data = datasets.FGVCAircraft(root='data', download=False, transform=train_transform, split='train')
        val_data = datasets.FGVCAircraft(root='data', download=False, transform=train_transform, split='val')
        test_data = datasets.FGVCAircraft(root='data', download=False, transform=test_transform, split='test')
        targets = train_data.classes
        print("Number of training samples: ", len(train_data))
        print("Number of validation samples: ", len(val_data))
        print("Number of testing samples: ", len(test_data))
        
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
        
    elif dataloader == 'birds':
        train_data = Cub2011(root='data', train=True, transform=train_transform, download=False)
        targets = train_data.classes
        # Split train_bird_data into training and validation sets
        train_size = int(0.75 * len(train_data))
        val_size = len(train_data) - train_size
        train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])
        test_data = Cub2011(root='data', train=False, transform=test_transform, download=False)
        print("Number of training samples: ", len(train_data))
        print("Number of validation samples: ", len(val_data))
        print("Number of testing samples: ", len(test_data))
        # Create the bird DataLoader
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    # print(train_data._labels)
    return train_dataloader, val_dataloader, test_dataloader, targets

def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location='cpu')
    
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print(
                    'could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model

def get_models(model, pretrain_scheme, num_classes=None):
    weight_path = 'data/imagenet21k_weights/'
    if model == 'resnet50':
        if pretrain_scheme == 'base':
            model = resnet50()
        elif pretrain_scheme == 'imagenet':
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif pretrain_scheme == 'imagenet21k':
            model = resnet50()
            weight_path = os.path.join(weight_path, 'resnet50_miil_21k.pth')
            model = load_model_weights(model, weight_path)
    elif model == 'vitb16':
        if pretrain_scheme == 'base':
            model = vit_b_16()
        elif pretrain_scheme == 'imagenet':
            model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        elif pretrain_scheme == 'imagenet21k':
            model_kwargs = dict(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False)
            model = timm.models.vision_transformer._create_vision_transformer('vit_base_patch16_224_in21k',
                                                                            pretrained=False,
                                                                            num_classes=num_classes, **model_kwargs)
            weight_path = os.path.join(weight_path, 'vit_base_patch16_224_miil_21k.pth')
            model = load_model_weights(model, weight_path)
    
    return model

args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed)  # sets pytorch's seed

if __name__ == '__main__':
    crop_size = 0.875 * args.height
    
    if args.model == 'vitb16':
        train_transform = transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.CenterCrop(crop_size),
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    train_dataloader, val_dataloader, test_dataloader, classes = get_dataloaders(dataloader=args.dataloader, 
                                                                                train_transform=train_transform,
                                                                                test_transform=test_transform, 
                                                                                batch_size=args.batch_size)
    
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")
    
    model = get_models(args.model, args.pretrain, num_classes=num_classes)
    
    if args.model == 'vitb16' and args.pretrain in ['imagenet', 'base']:
        model.heads[0].out_features = num_classes
    elif args.model == 'resnet50':
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
    conv_experiment = ExperimentBuilder(network_model=model,
                                        experiment_name=args.experiment_name,
                                        num_epochs=args.num_epochs,
                                        weight_decay_coefficient=args.weight_decay_coefficient,
                                        learning_rate=args.learning_rate,
                                        use_gpu=args.use_gpu,
                                        continue_from_epoch=args.continue_from_epoch,
                                        train_data=train_dataloader, 
                                        val_data=val_dataloader,
                                        test_data=test_dataloader)  # build an experiment object
    experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
