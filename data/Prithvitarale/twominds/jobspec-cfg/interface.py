from main import Classifier as c
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as t
from torchvision import datasets, transforms
from process_cub import Cub2011
import pickle

def load_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.ConvertImageDtype(t.float),
                                    transforms.Resize([256, 256])])
    # transforms.ConvertImageDtype(t.float)
    # train_data = datasets.FashionMNIST("./data", download=True, train=True, transform=transform)
    # test_data = datasets.FashionMNIST("./data", download=True, train=False, transform=transform)
    # train_data = datasets.CelebA("./data", download=True, split='train', transform=transform)
    # test_data = datasets.CelebA("./data", download=True, split='test', transform=transform)
    # train_data = datasets.CIFAR10("./data", download=True, train=True, transform=transform)
    # test_data = datasets.CIFAR10("./data", download=True, train=False, transform=transform)
    # train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    cub_data = Cub2011("data/", train=True, download=False)
    train_loader, test_loader = cub_data.get_data_loader(transform)
    print(train_loader, test_loader)
    return train_loader, test_loader


def train():
    model = c()
    loss_calc = nn.CrossEntropyLoss()
    # loss_calc = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=7e-3, momentum=0.9)
    epochs = 10
    train_loader, test_loader = load_data()
    dataiter = iter(test_loader)

    running_loss = 0
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        for i, data in enumerate(train_loader, 0):
            x, y = data
            print(x[0])
            print(y[0])
            print(len(x))
            print(len(y))
            # exit()
            optimizer.zero_grad()
            print("1")
            y_hat = model(x)
            print("2")

            loss = loss_calc(y_hat, y)
            print("3")

            # print(loss.item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(loss.item())

            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        # images, labels = next(dataiter)
        # print(images.shape)
        # labels_hat = model(images)
        # print(labels_hat.shape)
        # print(type(labels_hat[0][0]))
        # # exit()
        # test_loss = loss_calc(labels, labels_hat)
        # print(f"Test loss at epoch {epoch}: {test_loss.item()}")
    running_loss = 0
    for i, data in enumerate(test_loader, 0):
        x, y = data
        y_hat_pred = model(x)
        y_hat = t.argmax(y_hat_pred, dim=1)
        running_loss += t.sum(t.eq(y_hat, y))
    print(f'Final Test Loss: {running_loss / (i * 32)}')
    t.save(model, "./saved_models/preliminary_faces.pt")


with open("concepts/concepts.pkl", "rb") as fp:
    concepts = pickle.load(fp)

with open("concepts/visibility.pkl", "rb") as fp2:
    visibility = pickle.load(fp2)
print(concepts[2][4])
print(visibility[2][3])
exit()

#
# cub_data = Cub2011("data/")
# cub_data.get_concepts()
# exit()
#
#
#
# train()
# exit()
# model = t.load("./saved_models/preliminary_cifar10.pt")
# model.eval()


td, ted = load_data()
features = []
for i, data in enumerate(td):
    x, y = data
    features = model.get_features(x[5])
    print(y[5])
    print(y[4])
    print(y[6])
    break

imager = transforms.ToPILImage()
# features = t.mean(features, dim=0)

print(features.shape)
# img = imager(features)
# img.save("./features/mean.PNG")
# exit()
for i, f in enumerate(features):
    # print(f.shape)
    # break
    img = imager(f)
    img.save(f'./features/{i}.png')
exit()
# running_loss = 0
# for i, data in enumerate(ted, 0):
#     x, y = data
#     y_hat_pred = model(x)
#     y_hat = t.argmax(y_hat_pred, dim=1)
#     running_loss+=t.sum(t.eq(y_hat, y))
# print(f'Final Test Loss: {running_loss/(i*32)}')

# all_weights = []
# for layer in model.layers:
#    w = layer.get_weights()
#    all_weights.append(w)
imager = transforms.ToPILImage()
# for i, f in enumerate(all_weights):
#     # print(f.shape)
#     # break
#     img = imager(f)
#     img.save(f'./features/{i}.png')
# a = model.graph
# for i, data in enumerate(td, 0):
#     x, y = data
#     conv2_tensor = model.graph.get_tensor_by_name('conv2')
#     _, conv_val = model.run([conv2_tensor],
#                                       {'x': x})
#     break
# print(conv_val.shape)
    # img = imager(f)
    # img.save(f'./features/conv2.png')

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
nodes, _ = get_graph_node_names(model)
feature_extractor = create_feature_extractor(
	model, return_nodes=[nodes[3]])
# `out` will be a dict of Tensors, each representing a feature map
out = feature_extractor(t.zeros(16, 1, 11, 11))
print(out['conv2'].shape)
exit()
img = imager(out['conv2'])
img.save(f'./features/conv2.png')
print(out)

# 6 or 7 conv2 layers
# keep the h, w same but more channels
# decrease h, w and more channels
# also check without pooling
#batch normalization