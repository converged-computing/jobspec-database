import argparse
import torch
import sys
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR

from attention_models import ViT, CrossViT   # rename the skeleton file for your implementation / comment before testing for ResNet
from utils import EarlyStopping

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to classify CIFAR10')
    parser.add_argument('--model', type=str, default='vit', help='model to train (default: r18)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (default: 0.003)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    print(parser.parse_args())
    return parser.parse_args()

def train(model, trainloader, optimizer, criterion, device, epoch):
    model.train()
    
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
    
        output = model(data)

        loss = criterion(output, target) / len(output)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


def test(model, device, test_loader, criterion, epoch, set="Test", early_stop=None):
    model.eval()

    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    early_stop(test_loss, model, epoch)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        set, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def run(args):
    # Download and load the training data
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomInvert(),
        transforms.ToTensor(),  
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
           
    ])

    # TODO: adjust folder
    dataset = datasets.CIFAR10('cifar10/train', download=False, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    # Download and load the test data
    # TODO: adjust folder
    testset = datasets.CIFAR10('cifar10/test', download=False, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=True)

    # Build a feed-forward network
    print(f"Using {args.model}")
    if args.model == "r18":
        model = models.resnet18(pretrained=False)

    elif args.model == "vit":
        model = ViT(image_size=32, patch_size=32, num_classes=10, dim=128,
                    depth=6, heads=8, mlp_dim=2048, dropout=0.1,
                    emb_dropout=0.1)

    # CrossViT15 parameters
    elif args.model == "cvit":
        model = CrossViT(
                        image_size = 32,
                        num_classes = 10,
                        sm_dim = 192, 
                        lg_dim = 384,
                        sm_patch_size = 12,
                        sm_enc_depth = 1,
                        sm_enc_heads = 5,
                        sm_enc_mlp_dim = 2048, 
                        sm_enc_dim_head = 64,
                        lg_patch_size = 16, 
                        lg_enc_depth = 5,
                        lg_enc_heads = 5, 
                        lg_enc_mlp_dim = 2048,
                        lg_enc_dim_head = 64,
                        cross_attn_depth = 2,
                        cross_attn_heads = 3,
                        cross_attn_dim_head = 64,
                        depth = 4,
                        dropout = 0.1,
                        emb_dropout = 0.1)

    # Define the loss
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Change from the original SGD optimizer to AdamW
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Implement learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # Standard pytorch early stopping
    early_stop = EarlyStopping(patience=15)

    for epoch in range(1, args.epochs + 1):
        train(model, trainloader, optimizer, criterion, device, epoch)
        test(model, device, valloader, criterion, epoch, set="Validation", early_stop=early_stop)
        scheduler.step()

        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        if early_stop.early_stop:
            print("Early stopping")
            break
            
    model.load_state_dict(torch.load('checkpoint.pt'))
    test(model, device, testloader, criterion)

if __name__ == '__main__':
    args = parse_args()
    run(args)
