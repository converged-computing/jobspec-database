#import sys
#sys.path.append('/scratch/project_462000599/kostis/libs/analysator')
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import pytools as pt
torch.cuda.empty_cache()


#Reads in a VDF from cid CellID in a 3D  32 bit numpy array
def extract_vdf(file, cid, box=-1):
    assert cid > 0
    f = pt.vlsvfile.VlsvReader(file)
    # -- read phase space density
    vcells = f.read_velocity_cells(cid)
    keys = list(vcells.keys())
    values = list(vcells.values())

    # -- generate a velocity space
    size = f.get_velocity_mesh_size()
    vids = np.arange(4 * 4 * 4 * int(size[0]) * int(size[1]) * int(size[2]))

    # -- put phase space density into array
    dist = np.zeros_like(vids, dtype=float)
    dist.fill(np.NaN)
    dist.fill(0)
    dist[keys] = values

    # -- sort vspace by velocity
    v = f.get_velocity_cell_coordinates(vids)

    i = np.argsort(v[:, 0], kind="stable")
    v = v[i]
    # vids = vids[i]
    dist = dist[i]

    j = np.argsort(v[:, 1], kind="stable")
    v = v[j]
    # vids = vids[j]
    dist = dist[j]

    k = np.argsort(v[:, 2], kind="stable")
    v = v[k]
    # vids = vids[k]
    dist = dist[k]
    dist = dist.reshape(4 * int(size[0]), 4 * int(size[1]), 4 * int(size[2]))
    vdf = dist
    i, j, k = np.unravel_index(np.nanargmax(vdf), vdf.shape)
    len = box
    data = vdf[(i - len) : (i + len), (j - len) : (j + len), (k - len) : (k + len)]
    return np.array(data, dtype=np.float32)

def extract_vdfs(file , cids,box):
    vdfs=[]
    for cid in cids:
        vdfs.append(extract_vdf(file, cid,box))
    return np.stack(vdfs);

class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        # See Section 4.1 of "Neural Discrete Representation Learning".
        layers = []
        for i in range(num_residual_layers):
            layers.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv3d(
                        in_channels=num_hiddens,
                        out_channels=num_residual_hiddens,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.Conv3d(
                        in_channels=num_residual_hiddens,
                        out_channels=num_hiddens,
                        kernel_size=1,
                    ),
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = h + layer(h)

        # ResNet V1-style.
        return torch.relu(h)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_downsampling_layers,
        num_residual_layers,
        num_residual_hiddens,
    ):
        super().__init__()
        # See Section 4.1 of "Neural Discrete Representation Learning".
        # The last ReLU from the Sonnet example is omitted because ResidualStack starts
        # off with a ReLU.
        conv = nn.Sequential()
        for downsampling_layer in range(num_downsampling_layers):
            if downsampling_layer == 0:
                out_channels = num_hiddens // 2
            elif downsampling_layer == 1:
                (in_channels, out_channels) = (num_hiddens // 2, num_hiddens)

            else:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)

            conv.add_module(
                f"down{downsampling_layer}",
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
            conv.add_module(f"relu{downsampling_layer}", nn.ReLU())

        conv.add_module(
            "final_conv",
            nn.Conv3d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=5,
                padding=1,
            ),
        )
        self.conv = conv
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )

    def forward(self, x):
        h = self.conv(x)
        return self.residual_stack(h)


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_hiddens,
        num_upsampling_layers,
        num_residual_layers,
        num_residual_hiddens,
    ):
        super().__init__()
        # See Section 4.1 of "Neural Discrete Representation Learning".
        self.conv = nn.Conv3d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=1,
        )

        self.tconv = nn.ConvTranspose3d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=5,
            stride=1,
            padding=1,
        )
        
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )
        upconv = nn.Sequential()
        for upsampling_layer in range(num_upsampling_layers):
            if upsampling_layer < num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)

            elif upsampling_layer == num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens // 2)

            else:
                (in_channels, out_channels) = (num_hiddens // 2, 1)#3)

            upconv.add_module(
                f"up{upsampling_layer}",
                nn.ConvTranspose3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    #output_padding=(1,1,1),
                ),
            )
            if upsampling_layer < num_upsampling_layers - 1:
                upconv.add_module(f"relu{upsampling_layer}", nn.ReLU())

        self.upconv = upconv

    def forward(self, x):
        h = self.conv(x)
        h = self.residual_stack(h)
        h = self.tconv(h)
        h = nn.ReLU()(h)
        x_recon = self.upconv(h)
        return x_recon


class SonnetExponentialMovingAverage(nn.Module):
    # See: https://github.com/deepmind/sonnet/blob/5cbfdc356962d9b6198d5b63f0826a80acfdf35b/sonnet/src/moving_averages.py#L25.
    # They do *not* use the exponential moving average updates described in Appendix A.1
    # of "Neural Discrete Representation Learning".
    def __init__(self, decay, shape):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros(*shape))
        self.register_buffer("average", torch.zeros(*shape))

    def update(self, value):
        self.counter += 1
        with torch.no_grad():
            self.hidden -= (self.hidden - value) * (1 - self.decay)
            self.average = self.hidden / (1 - self.decay ** self.counter)

    def __call__(self, value):
        self.update(value)
        return self.average


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, use_ema, decay, epsilon):
        super().__init__()
        # See Section 3 of "Neural Discrete Representation Learning" and:
        # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L142.

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.use_ema = use_ema
        # Weight for the exponential moving average.
        self.decay = decay
        # Small constant to avoid numerical instability in embedding updates.
        self.epsilon = epsilon

        # Dictionary embeddings.
        limit = 3 ** 0.5
        e_i_ts = torch.FloatTensor(embedding_dim, num_embeddings).uniform_(
            -limit, limit
        )
        if use_ema:
            self.register_buffer("e_i_ts", e_i_ts)
        else:
            self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))

        # Exponential moving average of the cluster counts.
        self.N_i_ts = SonnetExponentialMovingAverage(decay, (num_embeddings,))
        # Exponential moving average of the embeddings.
        self.m_i_ts = SonnetExponentialMovingAverage(decay, e_i_ts.shape)

    def forward(self, x):
        flat_x = x.permute(0, 2, 3, 4, 1).reshape(-1, self.embedding_dim)
        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )
        encoding_indices = distances.argmin(1)
        quantized_x = F.embedding(
            encoding_indices.view(x.shape[0], *x.shape[2:]), self.e_i_ts.transpose(0, 1)
        ).permute(0, 4, 1, 2, 3)

        # See second term of Equation (3).
        if not self.use_ema:
            dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()
        else:
            dictionary_loss = None

        # See third term of Equation (3).
        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()
        # Straight-through gradient. See Section 3.2.
        quantized_x = x + (quantized_x - x).detach()

        if self.use_ema and self.training:
            with torch.no_grad():
                # See Appendix A.1 of "Neural Discrete Representation Learning".

                # Cluster counts.
                encoding_one_hots = F.one_hot(
                    encoding_indices, self.num_embeddings
                ).type(flat_x.dtype)
                n_i_ts = encoding_one_hots.sum(0)
                # Updated exponential moving average of the cluster counts.
                # See Equation (6).
                self.N_i_ts(n_i_ts)

                # Exponential moving average of the embeddings. See Equation (7).
                embed_sums = flat_x.transpose(0, 1) @ encoding_one_hots
                self.m_i_ts(embed_sums)

                # This is kind of weird.
                # Compare: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L270
                # and Equation (8).
                N_i_ts_sum = self.N_i_ts.average.sum()
                N_i_ts_stable = (
                    (self.N_i_ts.average + self.epsilon)
                    / (N_i_ts_sum + self.num_embeddings * self.epsilon)
                    * N_i_ts_sum
                )
                self.e_i_ts = self.m_i_ts.average / N_i_ts_stable.unsqueeze(0)

        return (
            quantized_x,
            dictionary_loss,
            commitment_loss,
            encoding_indices.view(x.shape[0], -1),
        )


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_downsampling_layers,
        num_residual_layers,
        num_residual_hiddens,
        embedding_dim,
        num_embeddings,
        use_ema,
        decay,
        epsilon,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )
        self.pre_vq_conv = nn.Conv3d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1
        )
        self.vq = VectorQuantizer(
            embedding_dim, num_embeddings, use_ema, decay, epsilon
        )
        self.decoder = Decoder(
            embedding_dim,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )

    def quantize(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        (z_quantized, dictionary_loss, commitment_loss, encoding_indices) = self.vq(z)
        return (z_quantized, dictionary_loss, commitment_loss, encoding_indices)

    def forward(self, x):
        (z_quantized, dictionary_loss, commitment_loss, _) = self.quantize(x)
        x_recon = self.decoder(z_quantized)
        return {
            "dictionary_loss": dictionary_loss,
            "commitment_loss": commitment_loss,
            "x_recon": x_recon,
        }

# TODO: reading vlsv file
import sys
cids=[1,2,3,4,5,6,7,8]
filename=sys.argv[1]
# input_array=extract_vdfs(filename,cids,25) # 25-> half the mesh dimension
# input_array=input_array.squeeze();
# print(input_array.shape)


device = 'cuda'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print('Using device:', device)

use_tb = True # Use Tensorboard (optional)

# Initialize model
use_ema = True # Use exponential moving average
model_args = {
    "in_channels":1,
    "num_hiddens": 128,
    "num_downsampling_layers": 2,
    "num_residual_layers": 2,
    "num_residual_hiddens": 32,
    "embedding_dim": 64,
    "num_embeddings": 512,
    "use_ema": use_ema,
    "decay": 0.99,
    "epsilon": 1e-5,
}
model = VQVAE(**model_args).to(device)

class VDFDataset():
    def __init__(self, cids,filename):
        self.cids=cids
        self.filename=filename

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, idx):
        vdf=extract_vdf(self.filename, self.cids[idx],box=25)
        vdf_norm = (vdf - vdf.min())/(vdf.max() - vdf.min())
        return torch.tensor(vdf_norm).unsqueeze(0).to(device)
        

# Initialize dataset
batch_size = 2
workers = 0

# input_norm = (input_array - input_array.min())/(input_array.max() - input_array.min()) # MinMax normalization
# input_tensor = torch.tensor(input_norm, dtype=torch.float32).unsqueeze(0)#.unsqueeze(0)#.to(device)  # Add batch and channel dimensions, move to device
# print(input_tensor.shape)
# train_dataset = input_tensor
VDF_Data=VDFDataset(cids,filename)
train_loader = DataLoader(
    dataset=VDF_Data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    pin_memory=False,
)

# Multiplier for commitment loss. See Equation (3) in "Neural Discrete Representation Learning"
beta = 0.25

# Initialize optimizer
train_params = [params for params in model.parameters()]
lr = 3e-4
optimizer = optim.Adam(train_params, lr=lr)
criterion = nn.MSELoss()

# Train model
epochs = 3
eval_every = 1
best_train_loss = float("inf")
model.train()

# Training
for epoch in tqdm(range(epochs)):
    total_train_loss = 0
    total_recon_error = 0
    n_train = 0
    for (batch_idx, train_tensors) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs = train_tensors[0].unsqueeze(0).to(device)
        out = model(imgs)
        recon_error = criterion(out["x_recon"], imgs) #/ train_data_variance
        total_recon_error += recon_error.item()
        loss = recon_error + beta * out["commitment_loss"]
        if not use_ema:
            loss += out["dictionary_loss"]

        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_train += 1

        if ((batch_idx + 1) % eval_every) == 0:
            print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
            total_train_loss /= n_train
            if total_train_loss < best_train_loss:
                best_train_loss = total_train_loss

            print(f"best_train_loss: {best_train_loss}")
            print(f"recon_error: {total_recon_error / n_train}\n")
            total_train_loss = 0
            total_recon_error = 0
            n_train = 0



# with torch.no_grad():
#     encoded = model.encoder(input_tensor)

