import numpy as np, pandas as pd
import torch, os, logging

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from preprocessing import load_dataset, normalize, compute_minimum_distances

# Set up Python logging
logging.basicConfig(level=logging.ERROR)

# Create a log directory with a timestamp to keep different runs separate
logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(logdir)

pd.set_option('display.precision', 20)

# torch.manual_seed(1234)
torch.autograd.set_detect_anomaly(True)
#torch.set_printoptions(edgeitems=1000)

if __name__ == "__main__":

    # Load Data (change path if needed)
    path = ["airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32"]
    train_data, len_list = load_dataset(path, 
                                       n_random_sampling = 0
                                       )

    Uinf, alpha, gamma_1, gamma_2, gamma_3 = float(path[0].split('_')[2]), float(path[0].split('_')[3])*np.pi/180, float(path[0].split('_')[4]), float(path[0].split('_')[5]), float(path[0].split('_')[6])
    print(f"Uinf: {Uinf}, alpha: {alpha}")
    
    u_inlet, v_inlet = np.cos(alpha)*Uinf, np.sin(alpha)*Uinf
    
    df_train_input = pd.DataFrame(train_data[0].x_train, columns=["x", "y", "sdf", "x_n", "y_n"])
    df_train_target = pd.DataFrame(train_data[0].y_train, columns=["u", "v", "p", "nut"])
    df_train = pd.concat([df_train_input, df_train_target], axis=1) 

    df_train['u_inlet'] = u_inlet
    df_train['v_inlet'] = v_inlet
    df_train['gamma_1'] = gamma_1
    df_train['gamma_2'] = gamma_2
    df_train['gamma_3'] = gamma_3

    df_box = df_train.iloc[len_list[0]:len_list[0]+len_list[1]+len_list[2],:]


    # df = df_box[(df_box['x'] <= 4) & (df_box['x'] >= 1.5)]

    # Define the grid limits
    x_min, x_max = -2, 4
    y_min, y_max = -1.5, 1.5

    # Define the resolution of the grid
    grid_resolution_x = 2000  # Number of points along x
    grid_resolution_y = 1000  # Number of points along y

    # Create grid points
    grid_x = np.linspace(x_min, x_max, grid_resolution_x)
    grid_y = np.linspace(y_min, y_max, grid_resolution_y)
    grid = np.meshgrid(grid_x, grid_y)

    # Create grid points
    grid_points = np.vstack([grid[0].ravel(), grid[1].ravel()]).T

    # For demonstration, interpolating 'x_n' feature
    features = df_box[['x', 'y', "sdf", "x_n", "y_n", 'u_inlet', 'v_inlet', 'gamma_1', 'gamma_2', 'gamma_3']].to_numpy()
    outputs = df_box[['x', 'y', 'u', 'v', 'p', 'nut']].to_numpy()

    # Perform interpolation
    features_grid = griddata(features[:, :2], features, grid_points, method='linear')
    outputs_grid = griddata(features[:, :2], outputs, grid_points, method='linear')

    # Reshape the results to the grid shape
    interpolated_feature = features_grid.reshape((10, grid_resolution_x, grid_resolution_y))
    output_interpolated_reshaped = outputs_grid.reshape((6, grid_resolution_x, grid_resolution_y))

    x_y_tensor = torch.from_numpy(interpolated_feature[:2]).float().requires_grad_(True)
    other_features_tensor = torch.from_numpy(interpolated_feature[2:]).float()

    input_tensor_torch = torch.cat([x_y_tensor, other_features_tensor], dim=0).unsqueeze(0)
    output_tensor_torch = torch.from_numpy(output_interpolated_reshaped[np.newaxis, ...])

    print("Datasets Loaded.")

    from FourierNeuralOperatorNN import FourierNeuralOperatorNN

    # Train the model
    model = FourierNeuralOperatorNN(input_tensor_torch, output_tensor_torch)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
        
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    print(f"Started Training.")
    model.train(1)
    print(f"Finished Training.")