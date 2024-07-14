import torch
import torch.nn as nn
from dataset_loaders.load_dataset import get_loader
from models.models import get_model

from dataset_loaders.datasset_config import DatasetConfig

from utils.data_utils import get_dataset_constant
from utils.model_utils import get_model_constant, load_model, freeze_layers_except_last
from utils.env_utils import get_env_constant

import argparse
from torch.cuda.amp import autocast

import time


def eval(model, val_loader, metric_path):
    model.eval()

    i = 0

    total = 0
    num_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to('cuda'), labels.to('cuda')

            with autocast(dtype=torch.float16):
                outputs = model(images)
                predicted = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1)
                num_correct += (predicted == labels).sum().item()
            
            total += len(labels)
            i += 1

            if i % 10 == 0:
                with open(metric_path, 'a+') as f:
                    f.write(f'accuracy {num_correct/total:.4f}, batch: {i}\n')

            print(f'accuracy {num_correct/total:.4f}, batch: {i}')

    with open(metric_path, 'a+') as f:
        f.write(f'accuracy {num_correct/total:.4f}, completed\n')

    print(f"Validation Accuracy: {num_correct/total:.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet", "imagenet_real", "flowers102", "oxford_iiit"], default="imagenet_real",
                        help="Which model to evaluate on.", required=True)
    
    parser.add_argument("--model_type", choices=["vit_l_16", "vit_h_14", "eff_net_l2", "bit_l_res"], default="vit_l_16", help="Which variant to use.", required=True)
    
    parser.add_argument("--env", choices=["local", "newton"], default="local", help="Which variant to use.", required=True)

    parser.add_argument("--pretrained_dir", type=str, default=None, help="Where to search for pretrained ViT models.")
    
    parser.add_argument("--metric_path", default="models/metrics", type=str, help="The output directory where metrics will be written.", required=True)

    
    args = parser.parse_args()

    dataset_name = get_dataset_constant(args.dataset)
    model_name = get_model_constant(args.model_type)
    env_name = get_env_constant(args.env)
    
    dataset_config = DatasetConfig(dataset_name, env_name)
    model = get_model(model_name, dataset_name)

    if args.pretrained_dir:
        model = load_model(model, args.pretrained_dir)

    model = freeze_layers_except_last(model)

    val_loader = get_loader(dataset_config, False)

    eval(model, val_loader, args.metric_path)

if __name__ == '__main__':
    main()