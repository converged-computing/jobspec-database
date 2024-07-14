import torch
import torch.nn as nn
from dataset_loaders.load_dataset import get_loader
from models.models import get_model

from utils.data_utils import get_dataset_constant
from utils.env_utils import get_env_constant
from utils.model_utils import get_model_constant, load_model, save_model, freeze_layers_except_last

from dataset_loaders.datasset_config import DatasetConfig

import argparse

from torch.cuda.amp import GradScaler, autocast

def train(model, train_loader, metric_path, checkpoint_path, num_epochs, optimizer, criterion):
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for images, labels in train_loader:
            images, labels = images.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()
            with autocast(dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += len(images)


        # Calculate and print training accuracy and loss
        train_accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

        with open(metric_path, 'a+') as f:
            f.write(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss:.4f}, Training Accuracy: {train_accuracy:.4f}\n")

        if epoch % 5 == 0:
            save_model(model, checkpoint_path)
    
    save_model(model, checkpoint_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet", "imagenet_real", "flowers102", "oxford_iiit"], default="imagenet_real",
                        help="Which model to evaluate on.", required=True)
    
    parser.add_argument("--model_type", choices=["vit_l_16", "vit_h_14", "eff_net_l2", "bit_l_res"], default="ViT-B_16", help="Which variant to use.", required=True)
    
    parser.add_argument("--env", choices=["local", "newton"], default="local", help="Which variant to use.", required=True)

    parser.add_argument("--pretrained_dir", type=str, default=None, help="Where to search for pretrained ViT models.")
    
    parser.add_argument("--metric_path", default="models/metrics", type=str, help="The output directory where metrics will be written.", required=True)

    parser.add_argument("--checkpoint_path", default="models/checkpoints", type=str, help="The output directory where checkpoints will be written.", required=True)

    parser.add_argument("--num_epochs", default=10, type=int, help="Number of epochs/times to iterate through the dataset.")

    
    args = parser.parse_args()

    dataset_name = get_dataset_constant(args.dataset)
    model_name = get_model_constant(args.model_type)
    env_name = get_env_constant(args.env)
    
    dataset_config = DatasetConfig(dataset_name, env_name)
    model = get_model(model_name, dataset_name)
    
    
    if args.pretrained_dir:
        model = load_model(model, args.pretrained_dir)
    
    model = freeze_layers_except_last(model)
    

    train_loader = get_loader(dataset_config, True)
    
    learning_rate = 0.001

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, args.metric_path, args.checkpoint_path, args.num_epochs, optimizer, criterion)



if __name__ == '__main__':
    main()