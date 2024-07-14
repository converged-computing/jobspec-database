from load_data import load_data
from baseline import AE
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.decomposition import PCA
from torch import nn
import numpy as np
import os
import pandas as pd


base_dir = os.getcwd()
study = 'schiebinger2019/raw_data_log'
# study = 'cao2019'
ttp = 12
ttp_p = 11
gene_cap = 2000
cell_cap = 2000
latent_size = 100
count = []
dense_mat_list, _ = load_data(base_dir, study, ttp, gene_cap, cell_cap)
for i in dense_mat_list:
    count.append(len(i))
print(f"Count si {count}")
mat = np.concatenate(dense_mat_list)
print("Mat is:")
print(mat.shape)
pca = PCA(n_components= latent_size)
pca.fit(mat)
transformed = pca.transform(mat)
print(f"Transformed: {transformed.shape}")
# print(len(dense_mat_list))
sum=0
output = []
for i in count:
    adding = transformed[sum:sum+i,:]
    print(adding.shape)
    output.append(adding)
    sum+=i

for i in range(len(output)):
    cur = output[i]
    print(len(cur))
    df = pd.DataFrame(cur)
    file_name = 'schiebinger2019/large_test/pca/' + str(i+1) + '.tsv.gz'
    df.to_csv(file_name, sep='\t', index=False, compression='gzip')

# for i in range(len(output)):
#     # Extract the (c,g) matrix for the i-th t
#     matrix = dense_mat_list[i]
    
#     # Convert the numpy array to a pandas DataFrame
#     df = pd.DataFrame(matrix)
    
#     # Construct the file name
#     file_name = 'schiebinger2019/raw_data_origin/raw_data/original_' + str(i+1) + '.tsv.gz'
    
#     # Save the DataFrame to a compressed TSV file
#     df.to_csv(file_name, sep='\t', index=False, compression='gzip')

exit()


# Convert tensor
input_tensor = torch.tensor(np.concatenate(dense_mat_list), dtype=torch.float32)

# Shuffle 
input_tensor = input_tensor[torch.randperm(input_tensor.size()[0])]

# Split
train_size = int(0.7 * len(input_tensor))
valid_size = int(0.15 * len(input_tensor))
test_size = len(input_tensor) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(input_tensor, [train_size, valid_size, test_size])

print(f"Dataset preprocessed.")


# Initialization
[cell_num, gene_num] = input_tensor.size()
print(f"Cell and gene: {cell_num}, {gene_num}")
print(f"Cuda? {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ae = AE(gene_num, latent_size).to(device)          #Testing.
print(f"AE initialized.")
# Recon
criterion = nn.MSELoss()
optimizer = optim.Adam(ae.parameters(), lr=0.001)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True).to(device)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True).to(device)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True).to(device)

# Train
def train(model, data_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data in data_loader:
            inputs = data.unsqueeze(1)  # Add a channel dimension
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}')

# Eval
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            inputs = data.unsqueeze(1)  # Add a channel dimension
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
    return total_loss / len(data_loader)

# Train
print("Start training.")
train(ae, train_loader, optimizer, criterion, epochs=10)
print("Training completed.")

latent = ae.latent(gene_num, input_tensor)
print(f"Latent representation: {latent.size()}")
# Evaluating the model
valid_loss = evaluate(ae, valid_loader, criterion)
test_loss = evaluate(ae, test_loader, criterion)

print(f'Validation Loss: {valid_loss}, Test Loss: {test_loss}')
torch.save(ae.state_dict(), "Output_baseline.pt")
