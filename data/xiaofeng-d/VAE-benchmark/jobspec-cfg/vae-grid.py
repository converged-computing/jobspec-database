
# Import libraries
import torch
import torchvision                             # contains image datasets and many functions to manipulate images
import torchvision.transforms as transforms    # to normalize, scale etc the dataset
from torch.utils.data import DataLoader,Subset      # to load data into batches (for SGD)
from torchvision.utils import make_grid        # Plotting. Makes a grid of tensors
from torchvision.datasets import MNIST         # the classic handwritten digits dataset
import matplotlib.pyplot as plt                # to plot our images
import numpy as np

import torch.nn as nn                          # Class that implements a model (such as a Neural Network)
import torch.nn.functional as F                # contains activation functions, sampling layers etc
import torch.optim as optim                    # For optimization routines such as SGD, ADAM, ADAGRAD, etc

from torchsummary import summary

from sklearn.decomposition import PCA

def run_sim(latent_dim, num_train_samples, flag='train', e_hidden=500, d_hidden=500):


    # Define the number of training samples you want to use
    # num_train_samples = 1000  # Example: Use 1000 training samples

    # Create the full MNIST training and test datasets
    full_trainset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    testset = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # Generate indices for the subset of the training set
    subset_indices = torch.arange(0, num_train_samples)

    # Create a subset of the training dataset
    trainset = Subset(full_trainset, subset_indices)

    # Batch size
    batch_size = 100


    # Create Dataset object.s Notice that ToTensor() transforms images to pytorch tensors AND scales the pixel values to be within [0, 1]. Also, we have separate Dataset
    # objects for training and test sets. Data will be downloaded to a folder called 'data'.
    # trainset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    # testset  = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())


    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(testset, batch_size=batch_size, shuffle=True)

    # trainset

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)




    # e_hidden = 500        # Number of hidden units in the encoder. See AEVB paper page 7, section "Marginal Likelihood"
    # d_hidden = 500        # Number of hidden units in the decoder. See AEVB paper page 7, section "Marginal Likelihood"
    # latent_dim = 2        # Dimension of latent space. See AEVB paper, page 7, section "Marginal Likelihood"
    learning_rate = 0.001 # For optimizer (SGD or Adam)
    weight_decay = 1e-5   # For optimizer (SGD or Adam)
    epochs = 200          # Number of sweeps through the whole dataset

    class VAE(nn.Module):
        def __init__(self, latent_dim):
            """Variational Auto-Encoder Class"""
            super(VAE, self).__init__()
            # Encoding Layers
            self.latent_dim = latent_dim
            self.e_input2hidden = nn.Linear(in_features=784, out_features=e_hidden)
            self.e_hidden2mean = nn.Linear(in_features=e_hidden, out_features=self.latent_dim)
            self.e_hidden2logvar = nn.Linear(in_features=e_hidden, out_features=self.latent_dim)

            # Decoding Layers
            self.d_latent2hidden = nn.Linear(in_features=self.latent_dim, out_features=d_hidden)
            self.d_hidden2image = nn.Linear(in_features=d_hidden, out_features=784)

        def encoder(self, x):
            # Shape Flatten image to [batch_size, input_features]
            x = x.view(-1, 784)
            # Feed x into Encoder to obtain mean and logvar
            x = F.relu(self.e_input2hidden(x))
            return self.e_hidden2mean(x), self.e_hidden2logvar(x)

        def decoder(self, z):
            return torch.sigmoid(self.d_hidden2image(torch.relu(self.d_latent2hidden(z))))

        def forward(self, x):
            # Encoder image to latent representation mean & std
            mu, logvar = self.encoder(x)

            # Sample z from latent space using mu and logvar
            if self.training:
                z = torch.randn_like(mu).mul(torch.exp(0.5*logvar)).add_(mu)
            else:
                z = mu

            # Feed z into Decoder to obtain reconstructed image. Use Sigmoid as output activation (=probabilities)
            x_recon = self.decoder(z)

            return x_recon, mu, logvar

    # Loss
    def vae_loss(image, reconstruction, mu, logvar):
        """Loss for the Variational AutoEncoder."""
        # Binary Cross Entropy for batch
        BCE = F.binary_cross_entropy(input=reconstruction.view(-1, 28*28), target=image.view(-1, 28*28), reduction='sum')
        # Closed-form KL Divergence
        KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE - KLD



    # Instantiate VAE with Adam optimizer
    vae = VAE(latent_dim=latent_dim)
    vae = vae.to(device)    # send weights to GPU. Do this BEFORE defining Optimizer
    optimizer = optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=weight_decay)

    input_size = (1,28,28)
    summary(vae, input_size)    

    # Train
    if flag == 'train':
        
        vae.train()            # tell the network to be in training mode. Useful to activate Dropout layers & other stuff
        losses = []

        for epoch in range(epochs):
            # Store training losses & instantiate batch counter
            losses.append(0)
            number_of_batches = 0

            # Grab the batch, we are only interested in images not on their labels
            for images, _ in trainloader:
                # Save batch to GPU, remove existing gradients from previous iterations
                images = images.to(device)
                optimizer.zero_grad()

                # Feed images to VAE. Compute Loss.
                reconstructions, latent_mu, latent_logvar = vae(images)
                loss = vae_loss(images, reconstructions, latent_mu, latent_logvar)

                # Backpropagate the loss & perform optimization step with such gradients
                loss.backward()
                optimizer.step()

                # Add loss to the cumulative sum
                losses[-1] += loss.item()
                number_of_batches += 1

            # Update average loss & Log information
            losses[-1] /= number_of_batches
            print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, epochs, losses[-1]))



        # After your training loop, save the model's state_dict
        # model_save_path = './content/vae_model_'+'latent_'+str(latent_dim)+'.pth'  # Temporary path in Google Colab
        # torch.save(vae.state_dict(), model_save_path)
        model_save_path = './content/vae_model_'+'latent_'+str(latent_dim)+'_ehidden_'+str(e_hidden)+'_dhidden_'+str(d_hidden)+'.pth'  

        torch.save(vae.state_dict(), model_save_path)
        np.save('./loss_data/'+str(latent_dim)+'_ehidden_'+str(e_hidden)+'_dhidden_'+str(d_hidden),losses)


    # Visualize
    if flag == 'test':
        
        # load model weights
        model_save_path = './content/vae_model_'+'latent_'+str(latent_dim)+'.pth'  # Temporary path in Google Colab
        vae.load_state_dict(torch.load(model_save_path))
        vae.eval()
    
        latent_mus = []
        labels = []
        test_loss, number_of_batches = 0.0, 0
        for test_images, test_labels in testloader:
            
            
            # Do not track gradients
            with torch.no_grad():

                # Send images to the GPU/CPU
                test_images = test_images.to(device)

                # Feed images through the VAE to obtain their reconstruction & compute loss
                reconstructions, latent_mu, latent_logvar = vae(test_images)
                loss = vae_loss(test_images, reconstructions, latent_mu, latent_logvar)



                latent_mus.append(latent_mu.cpu().numpy())  # Assuming latent_mu is a tensor
                labels.append(test_labels.cpu().numpy())

                # Cumulative loss & Number of batches
                test_loss += loss.item()
                number_of_batches += 1

                # # Now divide by number of batches to get average loss per batch
                # test_loss /= number_of_batches
                # print('average reconstruction error: %f' % (test_loss))

    # Convert lists to numpy arrays
        latent_mus = np.concatenate(latent_mus, axis=0)
        labels = np.concatenate(labels, axis=0)

        # Perform PCA to reduce dimensions to 2
        pca = PCA(n_components=2)
        latent_mus_2d = pca.fit_transform(latent_mus)


        # Plotting
        plt.figure(figsize=(10, 8))
        for i in range(10):  # MNIST has 10 classes (digits 0-9)
            indices = labels == i
            plt.scatter(latent_mus_2d[indices, 0], latent_mus_2d[indices, 1], label=str(i))  # Assuming 2D latent space

        plt.xlabel('Latent Variable 1')
        plt.ylabel('Latent Variable 2')
        plt.title('Latent Space Visualization')
        plt.legend()
        plt.savefig('./latentvis_'+str(latent_dim)+'_ehidden_'+str(e_hidden)+'_dhidden_'+str(d_hidden)+'.png',dpi=300)


# for latent_dim in [3,5,10,20,200]:
#     run_sim(latent_dim=latent_dim, num_train_samples= 60000,flag='train')

# for latent_dim in [2]: #: #[3,5,10,20,200]
#     run_sim(latent_dim=latent_dim, num_train_samples= 60000,flag='test')

# for latent_dim in [2]:
#     run_sim(latent_dim=latent_dim, num_train_samples= 60000,flag='train')

if __name__ == '__main__':

    for hidden_dim in  [50, 100, 200, 400, 800]:
        print('------now testing hidden dimensions: ', hidden_dim)
        run_sim(latent_dim=latent_dim, num_train_samples= 60000,flag='train', e_hidden=hidden_dim, d_hidden= hidden_dim)
