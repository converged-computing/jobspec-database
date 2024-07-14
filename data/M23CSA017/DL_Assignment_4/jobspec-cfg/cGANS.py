import argparse
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# matplotlib inline
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import glob
from torch.autograd import Variable
import torch.autograd as autograd

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
# parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=7, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
args = parser.parse_args()
print(args)


ngpu = torch.cuda.device_count()
print('num gpus available: ', ngpu)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

image_dir = "dataset/dl_assignment_4/Train_data"
sketch_dir = "dataset/dl_assignment_4/Train/Contours"

labels_df = "dataset/dl_assignment_4/Train/Train_labels.csv"



image_size = 256
batch_size = args.batch_size
stats_image = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
stats_sketch = (0,), (1)


def add_gaussian_noise(image, mean=0, stddev=1):

    noise = torch.randn_like(image)

    noisy_image = image + noise

    return noisy_image


class ImageSketchDataset(torch.utils.data.Dataset):
    def __init__(
        self, image_dir, sketch_dir, labels_df, transform_image, transform_sketch
    ):
        self.image_dir = image_dir
        self.sketch_dir = sketch_dir
        self.labels_df = pd.read_csv(labels_df)
        self.transform_image = transform_image
        self.transform_sketch = transform_sketch
        self.all_sketches = glob.glob1(
            self.sketch_dir, "*.png"
        )  # return .jpg or .png files

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        # print(self.labels_df,"here")
        image_filename = self.labels_df.iloc[index]["image"]  # Get image filename

        label_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
        label = self.labels_df.loc[index, label_cols].values.astype(
            "float32"
        )  # Load and convert labels

        image_path = os.path.join(self.image_dir, image_filename + ".jpg")
        sketch_filename = np.random.choice(
            self.all_sketches
        )  # Assuming sketch filenames start with 'sketch_'
        sketch_path = os.path.join(self.sketch_dir, sketch_filename)

        image = Image.open(image_path)
        # print(image)

        sketch = Image.open(sketch_path)

        if self.transform_image:

            image = self.transform_image(image)

        if self.transform_sketch:
            sketch = self.transform_sketch(sketch)

        # Convert images to NumPy arrays
        image_np = np.array(image)

        sketch_np = np.zeros_like(sketch)
        sketch_np[np.all(sketch) == 255] = 1.0
        sketch_np = sketch_np.astype(np.float32)
        # Add Gaussian noise to the sketch

        # print(image_np, noisy_sketch_np)
        # print(image_filename,label)
        return (
            torch.from_numpy(image_np),
            torch.from_numpy(sketch_np),
            torch.from_numpy(label),
        )


# Transformations
transform_image = T.Compose(
    [
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(*stats_image),
    ]
)

transform_sketch = T.Compose(
    [
        T.Resize(image_size),
        T.CenterCrop(image_size),
        # T.ToTensor(),
        # T.Normalize(*stats_sketch)
    ]
)
train_ds = ImageSketchDataset(
    image_dir,
    sketch_dir,
    labels_df,
    transform_image=transform_image,
    transform_sketch=transform_sketch,
)



train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    # num_workers=2,
    pin_memory=True,
)




def denorm(img_tensors):
    return img_tensors * stats_image[1][0] + stats_image[0][0]


# def show_images(images, nmax=64):
#   fig, ax = plt.subplots(figsize=(8, 8))
#   ax.set_xticks([]); ax.set_yticks([])
#   ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

# def show_batch(dl, nmax=64):/
#   for images, _ , _ in dl:
# # # #     show_images(images, nmax)
#     break


# class Discriminator(nn.Module):
#     def __init__(self, num_classes,ngpu=0):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=0, bias=False),
#         )

#         self.flatten = nn.Flatten()

#         # Output layer
#         self.fc = nn.Linear(56, 1)  # Add an extra dimension for the class labels

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, labels):

#         x = self.main(x)
#         x = self.flatten(x)

#         # Concatenate labels with the features
#         concatenated = torch.cat((x, labels), dim=1)
#         # print(concatenated.shape, x.shape, labels.shape)
#         x = self.fc(concatenated)
#         # x = self.sigmoid(x)
#         return x

class Discriminator(nn.Module):
    def __init__(self, num_classes, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=0, bias=False),
        )
        self.flatten = nn.Flatten()
        
        # Output layers
        self.fc_dis = nn.Linear(49, 1)
        self.fc_aux = nn.Linear(49, num_classes)  # Classifier for auxiliary task
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x,labels):
        x = self.main(x)
        x = self.flatten(x)
        
        # realfake = self.sigmoid(self.fc_dis(x)).view(-1, 1).squeeze(1)
        realfake = self.fc_dis(x)

        classes = self.softmax(self.fc_aux(x))
        
        return realfake, classes



num_classes = len(train_ds.labels_df.columns) - 1


# """Generator Network"""


def double_convolution(in_channels, out_channels):

    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    return conv_op


class Generator(nn.Module):
    def __init__(self,ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution__2 = double_convolution(8, 4)
        self.down_convolution__1 = double_convolution(4, 8)
        self.down_convolution_0 = double_convolution(8, 1)
        self.down_convolution_1 = double_convolution(1, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)
        self.down_convolution_5 = double_convolution(512, 1024)
        # Expanding path.

        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        # Below, `in_channels` again becomes 1024 as we are concatinating.
        self.up_convolution_1 = double_convolution(1024, 512)

        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up_convolution_2 = double_convolution(512, 256)

        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up_convolution_3 = double_convolution(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.up_convolution_4 = double_convolution(128, 64)
        # output => `out_channels` as per the number of classes.
        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

    def forward(self, x):
        down__2 = self.down_convolution__2(x)
        down__1 = self.down_convolution__1(down__2)
        down_0 = self.down_convolution_0(down__1)
        down_1 = self.down_convolution_1(down_0)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)
        down_9 = self.down_convolution_5(down_8)

        up_1 = self.up_transpose_1(down_9)
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))
        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))
        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))
        out = self.out(x)
        return out

discriminator = Discriminator(num_classes, ngpu).to(device)
generator = Generator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    generator = nn.DataParallel(generator, list(range(ngpu)))
    discriminator = nn.DataParallel(discriminator, list(range(ngpu)))

def sample_sketches(num_sketches):
    sketches = []
    for i in range(num_sketches):
        sketch = torch.zeros(1, 1, 256, 256)
        # Randomly sample 50% of the pixels
        sketch[
            0,
            0,
            torch.randint(0, 256, (int(0.5 * 256 * 256),)),
            torch.randint(
                0,
                256,
                (
                    int(
                        0.5 * 256 * 256,
                    )
                ),
            ),
        ] = 1.0
        sketches.append(sketch)
    return sketches


def Generate_Fakes(sketches):
    noisy_sketchs = add_gaussian_noise(sketches)
    noisy_sketchs_ = []
    fake_labels = torch.randint(0, 7, (sketches.size(0), ), device=sketches.device)
    for noisy_sketch, fake_label in zip(noisy_sketchs, fake_labels):
        channels = torch.zeros(
            size=(7, *noisy_sketch.shape), device=noisy_sketch.device
        )
        channels[fake_label] = 1.0
        noisy_sketch = torch.cat((noisy_sketch.unsqueeze(0), channels), dim=0)
        noisy_sketchs_.append(noisy_sketch)

    noisy_sketchs = torch.stack(noisy_sketchs_)

    # convert fake_labels to one-hot encoding
    # fake_labels = F.one_hot(fake_labels, num_classes=7).squeeze(1).float().to(device)

    return noisy_sketchs, fake_labels



sample_dir = "generated_wcgan"
os.makedirs(sample_dir, exist_ok=True)


def save_samples(index, generator, train_dl, show=True):
    real_images, sketches, real_labels = next(iter(train_dl))
    latent_input, gen_labels = Generate_Fakes(sketches=sketches)
    fake_images = generator(latent_input.to(device))

    fake_fname = "generated-images-{0:0=4d}.png".format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print("Saving", fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))


# fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
adversarial_loss = torch.nn.MSELoss()
aux_criterion = nn.NLLLoss()
Tensor = torch.cuda.FloatTensor if (device.type == 'cuda') else torch.FloatTensor

def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()
    generator.train()
    discriminator.train()

    # Losses and scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    k = 2
    p = 6

    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):

        for idx, (real_images, sketches, real_labels) in tqdm(enumerate(train_dl), 
                                                              desc= "Training", dynamic_ncols=True,total=len(train_dl)):  # Ensure that real_labels are provided
            # Configure input
            real_images  = Variable(real_images.type(Tensor).to(device), requires_grad=True)
            sketches = sketches.to(device)
            real_labels = torch.argmax(real_labels.to(device), dim=1)
            # Adversarial ground truths
            batch_size = real_images.shape[0]
            
            # valid  = torch.full((batch_size,1), 1.0, dtype=torch.float, device=device)
            # fake = torch.full((batch_size,1), 0.0, dtype=torch.float, device=device)

            # generate fake input
            latent_input, gen_labels = Generate_Fakes(sketches=sketches)
            
            latent_input =  Variable(latent_input.to(device))
            # ----------------------
            # Train Discriminator
            # ----------------------
            
            opt_d.zero_grad()
            
            fake_images = generator(latent_input)

            #  real images
            validity_real, real_aux_output = discriminator(real_images, real_labels)
            #  fake images
            validity_fake, fake_aux_output = discriminator(fake_images, gen_labels)

             # Compute W-div gradient penalty
            real_grad_out = Variable(Tensor(real_images.size(0), 1).fill_(1.0), requires_grad=False)
            
            real_grad = autograd.grad(validity_real, real_images, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
            
            real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

            fake_grad_out = Variable(Tensor(fake_images.size(0), 1).fill_(1.0), requires_grad=False)

            fake_grad = autograd.grad(validity_fake, fake_images, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True,allow_unused=True)[0]
            
            fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

            div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

               # Adversarial loss
            loss_d_wgan = -torch.mean(validity_real) + torch.mean(validity_fake) + div_gp
            # print(fake_aux_output.shape,gen_labels.shape,real_aux_output.shape,real_labels.shape)
            loss_d_aux = aux_criterion(fake_aux_output, gen_labels) + aux_criterion(real_aux_output, real_labels)
            
            loss_d = loss_d_wgan + loss_d_aux

            # real_loss_d = adversarial_loss(validity_real, valid)
            # real_score =torch.mean(validity_real).item()

            

            # fake_loss_d = adversarial_loss(validity_fake, fake)
            # fake_score = torch.mean(validity_fake).item()
            
            # Total discriminator loss
            # loss_d = (real_loss_d + fake_loss_d) / 2
            loss_d.backward()
            opt_d.step()

            # Train the generator every n_critic steps
            if idx % args.n_critic == 0:
                # ------------------
                # Train generator
                # ------------------
                opt_g.zero_grad()
                fake_images = generator(latent_input)
                validity_fake, fake_aux_output = discriminator(fake_images, gen_labels)
                # loss_g = adversarial_loss(validity, valid)
                loss_g = -torch.mean(validity_fake) + aux_criterion(fake_aux_output, gen_labels) 
                loss_g.backward()
                opt_g.step()


                print(
                    "Epoch [{}/{}], Batch [{}/{}], loss_g:{:.4f}, loss_d:{:.4f}, real_scores:{:.4f}, fake_score:{:.4f}".format(
                        epoch + 1, epochs, idx, len(train_dl), loss_g, loss_d, 0, 0
                    )
                )
                batches_done = epoch * len(train_dl) + idx
                if batches_done % args.sample_interval == 0:
                    save_samples(epoch + start_idx, generator, train_dl, show=False)
                
                batches_done += args.n_critic
                
                losses_d.append(loss_d.item())
                losses_g.append(loss_g.item())
                # real_scores.append(real_score)
                # fake_scores.append(fake_score)

    return losses_g, losses_d, real_scores, fake_scores


lr = args.lr #0.0002
epochs = args.n_epochs

history = fit(epochs, lr)

losses_g, losses_d, real_scores, fake_scores = history

# plt.plot(losses_d, '-')
# plt.plot(losses_g, '-')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['Discriminator', 'Generator'])
# plt.title('Losses')

# plt.plot(real_scores, '-')
# plt.plot(fake_scores, '-')
# plt.xlabel('epoch')
# plt.ylabel('score')
# plt.legend(['Real', 'Fake'])
# plt.title('Scores')
