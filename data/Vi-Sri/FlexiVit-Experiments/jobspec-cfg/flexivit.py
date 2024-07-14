from timm import create_model
from timm.layers.pos_embed import resample_abs_pos_embed
from flexivit_pytorch import pi_resize_patch_embed
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import gc
from tqdm import tqdm
import pickle


def create_vit_with_patch_size(new_patch_size):
    state_dict = create_model("vit_base_patch16_224", pretrained=True).state_dict()
    state_dict["patch_embed.proj.weight"] = pi_resize_patch_embed(
        patch_embed=state_dict["patch_embed.proj.weight"], new_patch_size=new_patch_size
    )
    image_size = 224
    grid_size = image_size // new_patch_size[0]
    state_dict["pos_embed"] = resample_abs_pos_embed(
        posemb=state_dict["pos_embed"], new_size=[grid_size, grid_size]
    )
    net = create_model(
        "vit_base_patch16_224", img_size=image_size, patch_size=new_patch_size
    )
    net.load_state_dict(state_dict, strict=True)
    return net


def train_flexified_vit_and_track_accuracy(model, trainloader, testloader, epochs=10, freeze_layers=None, label=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

        epoch_accuracies = []
        epoch = 0

        for epoch in tqdm(range(epochs)):
            model.train()
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Evaluation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            epoch_accuracy = 100 * correct / total
            epoch_accuracies.append(epoch_accuracy)
            print(f'Epoch for {label} - {epoch + 1}, Accuracy: {epoch_accuracy}%')
        
        torch.save(model.state_dict(), f"saved_models/{label}_fine_tuned.pth")
        print('Finished Training for patch size : ',label)

        with open(f"saved_accuracies/{label}_accuracies_fine_tune.pkl", "wb") as f:
            pickle.dump(epoch_accuracies, f)

        del model
        torch.cuda.empty_cache()
        gc.collect()
        return epoch_accuracies


if __name__ == "__main__":

    net_32x32 = create_vit_with_patch_size((32, 32))
    # net_16x16 = create_vit_with_patch_size((16, 16))
    net_8x8 = create_vit_with_patch_size((8, 8))
    # net_4x4 = create_vit_with_patch_size((4, 4))
    # net_2x2 = create_vit_with_patch_size((2, 2))

    # Data augmentation and normalization for training
    # Just normalization for validation
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR10 training and test datasets
    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True)

    testset = datasets.CIFAR10(root='./data', train=False,
                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False)

    accuracies_32x32_full = train_flexified_vit_and_track_accuracy(net_32x32, trainloader, testloader, epochs=10, freeze_layers=True, label="32x32")
    # accuracies_16x16_full = train_flexified_vit_and_track_accuracy(net_16x16, trainloader, testloader, epochs=10, freeze_layers=True, label="16x16")
    accuracies_8x8_full = train_flexified_vit_and_track_accuracy(net_8x8, trainloader, testloader, epochs=10, freeze_layers=True, label="8x8")
    # accuracies_4x4_full = train_flexified_vit_and_track_accuracy(net_4x4, trainloader, testloader, epochs=10, freeze_layers=True, label="4x4")

    # Assuming accuracies for each model are already calculated
    epochs = range(1, 11)  # Assuming 10 epochs
    plt.plot(epochs, accuracies_32x32_full, label='32x32')
    # plt.plot(epochs, accuracies_16x16_full, label='16x16')
    plt.plot(epochs, accuracies_8x8_full, label='8x8')
    # plt.plot(epochs, accuracies_4x4_full, label='4x4')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs for Different Patch Sizes')
    plt.legend()

    # Save the plot
    plt.savefig('vit_patch_size_accuracy_comparison_fine_tune.png')

    # Show the plot
    plt.show()