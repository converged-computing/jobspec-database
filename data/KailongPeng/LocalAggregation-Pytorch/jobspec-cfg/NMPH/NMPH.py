from __future__ import print_function
import os
from glob import glob
import numpy as np
from tqdm import tqdm
import sys
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import random

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

testMode = False
testBatchNum = 50
if testMode:
    jobID = 1
else:
    jobID = int(float(sys.argv[1]))

print(f"jobID={jobID}")
if jobID == 1:
    exp_name = "imagenet_la"
elif jobID == 2:
    exp_name = "imagenet_ft"
elif jobID == 3:
    exp_name = "imagenet_ir"
elif jobID == 4:
    exp_name = "imagenet_la_layer_norm"
elif jobID == 5:
    exp_name = "imagenet_ft_layer_norm"
elif jobID == 6:
    exp_name = "imagenet_ir_layer_norm"
else:
    raise Exception("jobID not found")

print(f"exp_name={exp_name}")

directory_path = f'/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/{exp_name}/weights_difference/numpy/'


def dataPrepare():
    # selected_channel_penultimate_layer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 selected units
    # randomly select 10 units from the penultimate layer
    # seed
    random.seed(131)
    selected_channel_penultimate_layer = random.sample(range(0, 512), 512)
    selected_channel_last_layer = random.sample(range(0, 128), 128)

    totalBatchNum = 0
    epochBatchNum = {}
    startFromEpoch = 0
    totalEpochNum = 1

    # directory_path = '/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/weights_difference/numpy/'
    for epoch in range(startFromEpoch, totalEpochNum):
        print(f'{directory_path}/activation_lastLayer_epoch{epoch}_batch_i*.npy')
        files = glob(f'{directory_path}/activation_lastLayer_epoch{epoch}_batch_i*.npy')
        if testMode:
            files = files[:testBatchNum]
            epochBatchNum[epoch] = len(files)
            totalBatchNum += len(files)
        else:
            epochBatchNum[epoch] = len(files) - 1
            totalBatchNum += len(
                files) - 1  # minus 1 since the final batch usually has a different size, e.g. for batch size=128, the final batch only has 15 images
    print(f"totalBatchNum={totalBatchNum}")
    fc1_activations = np.zeros(
        (totalBatchNum, 128, len(selected_channel_penultimate_layer)))  # [#batch, batch size, #selected units]
    fc2_activations = np.zeros(
        (totalBatchNum, 128, len(selected_channel_last_layer)))
    weight_changes = np.zeros(
        (totalBatchNum, len(selected_channel_last_layer), len(selected_channel_penultimate_layer)))

    currBatchNum = 0
    for epoch in range(startFromEpoch, totalEpochNum):
        for batch_i in tqdm(range(0, epochBatchNum[epoch])):
            # torch.save(weights_difference,
            #            f'{weights_difference_folder}/weights_difference_epoch{self.current_epoch}_batch_i{batch_i}.pth.tar')
            # torch.save(activation_lastLayer,
            #            f'{weights_difference_folder}/activation_lastLayer_epoch{self.current_epoch}_batch_i{batch_i}.pth.tar')
            # torch.save(activation_secondLastLayer,
            #            f'{weights_difference_folder}/activation_secondLastLayer_epoch{self.current_epoch}_batch_i{batch_i}.pth.tar')
            # load activations and weights
            activation_secondLastLayer = np.load(
                f'{directory_path}/activation_secondLastLayer_epoch{epoch}_batch_i{batch_i}.npy')
            activation_lastLayer = np.load(
                f'{directory_path}/activation_lastLayer_epoch{epoch}_batch_i{batch_i}.npy')
            weight_change = np.load(
                f'{directory_path}/weights_difference_epoch{epoch}_batch_i{batch_i}.npy')  # .detach().numpy()

            fc1_activations[currBatchNum, :, :] = activation_secondLastLayer[:,
                                                  selected_channel_penultimate_layer]  # (128 batch#, 512)
            fc2_activations[currBatchNum, :, :] = activation_lastLayer[:,
                                                  selected_channel_last_layer]  # (128 batch#, 128)
            weight_changes[currBatchNum, :, :] = weight_change[selected_channel_last_layer, :][:,
                                                 selected_channel_penultimate_layer]  # (128 channel#, 512 channel#)
            currBatchNum += 1
    print(f"fc1_activations.shape={fc1_activations.shape}")
    print(f"fc2_activations.shape={fc2_activations.shape}")
    print(f"weight_changes.shape={weight_changes.shape}")
    # fc1_activations.shape=(16, 64, 10)  # (#batch, batch size, #selected units)
    # fc2_activations.shape=(16, 64, 5)
    # fc2_partial_weights.shape=(16, 5, 10)
    # weight_changes.shape=(16, 5, 10)

    # Calculate the row-wise differences
    # fc2_partial_weights_differences = fc2_partial_weights[1:] - fc2_partial_weights[:-1]

    # obtain all the co-activation and changes.
    co_activations_flatten = []
    weight_changes_flatten = []
    pairIDs = []
    for curr_fc1_feature in tqdm(range(len(selected_channel_penultimate_layer))):  # 512*128 = 65536 pairs
        for curr_fc2_feature in range(len(selected_channel_last_layer)):
            activation_lastLayer = fc1_activations[:, :, curr_fc1_feature]  # [#batch, batch size， channel#]
            activation_secondLastLayer = fc2_activations[:, :, curr_fc2_feature]
            weight_change = weight_changes[:, curr_fc2_feature, curr_fc1_feature]
            weight_changes_flatten.append(weight_change)  # each batch has a single weight change
            # Calculate the co-activation
            co_activation = np.multiply(activation_lastLayer, activation_secondLastLayer)
            # print(f"co_activation.shape={co_activation.shape}")
            # each batch has a single weight change but multiple co-activations, average across the batch to obtain a batch specific co-activation
            co_activation = np.mean(co_activation, axis=1)
            # print(f"np.mean(co_activation, axis=1).shape={co_activation.shape}")
            co_activations_flatten.append(co_activation)
            pairIDs.append(
                [selected_channel_penultimate_layer[curr_fc1_feature], selected_channel_last_layer[curr_fc2_feature]])
    return co_activations_flatten, weight_changes_flatten, pairIDs


co_activations_flatten_, weight_changes_flatten_, pairIDs_ = dataPrepare()
# mkdir(f'{directory_path}/temp')
if not os.path.exists(f'{directory_path}/temp'):
    os.mkdir(f'{directory_path}/temp')

if not testMode:
    np.save(f'{directory_path}/temp/co_activations_flatten_.npy',
            co_activations_flatten_)  # shape = [pair#, batch#]
    np.save(f'{directory_path}/temp/weight_changes_flatten_.npy',
            weight_changes_flatten_)  # shape = [pair#, batch#]
    np.save(f'{directory_path}/temp/pairIDs_.npy',
            pairIDs_)  # shape = [pair#, [ID1, ID2]]


# co_activations_flatten_ = np.load(f'{directory_path}/temp/co_activations_flatten_.npy',
#                                   allow_pickle=True)  # shape = [pair#, batch#]
# weight_changes_flatten_ = np.load(f'{directory_path}/temp/weight_changes_flatten_.npy',
#                                   allow_pickle=True)  # shape = [pair#, batch#]
# pairIDs_ = np.load(f'{directory_path}/temp/pairIDs_.npy',
#                    allow_pickle=True)  # shape = [pair#, [ID1, ID2]]


def cubic_fit_correlation_with_params(x, y, n_splits=10, random_state=42, return_subset=True):
    def cubic_function(_x, a, b, c, d):
        return a * _x ** 3 + b * _x ** 2 + c * _x + d

    # Function to compute correlation coefficient
    def compute_correlation(observed, predicted):
        return pearsonr(observed, predicted)[0]

    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Shuffle indices for k-fold cross-validation
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    # Initialize arrays to store correlation coefficients and parameters
    correlation_coefficients = []
    fitted_params = []

    for curr_split in range(n_splits):
        # Split data into training and testing sets
        split_size = len(x) // n_splits
        test_indices = indices[curr_split * split_size: (curr_split + 1) * split_size]
        train_indices = np.concatenate([indices[:curr_split * split_size], indices[(curr_split + 1) * split_size:]])

        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Perform constrained cubic fit on the training data
        params, _ = curve_fit(cubic_function, x_train, y_train)

        # Predict y values on the test data
        y_pred = cubic_function(x_test, *params)

        # Compute correlation coefficient and store it
        correlation_coefficient = compute_correlation(y_test, y_pred)
        correlation_coefficients.append(correlation_coefficient)

        # Store fitted parameters
        fitted_params.append(params)

    # Average correlation coefficients and parameters across folds
    mean_correlation = np.mean(correlation_coefficients)
    mean_params = np.mean(fitted_params, axis=0)

    if return_subset:
        # Randomly choose 9% of the data for future visualization
        subset_size = 10  # int(0.09 * len(x))
        subset_indices = random.sample(range(len(x)), subset_size)
        return mean_correlation, mean_params, x[subset_indices], y[subset_indices]
    else:
        return mean_correlation, mean_params


def run_NMPH(co_activations_flatten, weight_changes_flatten, pairIDs, rows=None, cols=None, plotFig=False):
    if plotFig:
        if rows is None:
            rows = int(np.ceil(np.sqrt(len(co_activations_flatten))))
        if cols is None:
            cols = int(np.sqrt(len(co_activations_flatten)))

        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))  # Create a subplot matrix
        cmap = get_cmap('viridis')  # Choose a colormap (you can change 'viridis' to your preferred one)
    else:
        axs = None
        cmap = None

    mean_correlation_coefficients = []
    # recorded_data = []  # Store recorded data for visualization
    mean_parameters = []
    x_partials = []
    y_partials = []
    for i in tqdm(range(len(co_activations_flatten))):
        if testMode:
            x__ = co_activations_flatten[i][:testBatchNum]
            y__ = weight_changes_flatten[i][:testBatchNum]
            pairID = pairIDs[i]
        else:
            x__ = co_activations_flatten[i]
            y__ = weight_changes_flatten[i]
            pairID = pairIDs[i]
        mean_correlation_coefficient, mean_parameter, x_partial, y_partial = cubic_fit_correlation_with_params(
            x__, y__,
            n_splits=10,
            random_state=42,
            return_subset=True
        )
        mean_correlation_coefficients.append(mean_correlation_coefficient)
        mean_parameters.append(mean_parameter)
        x_partials.append(x_partial)
        y_partials.append(y_partial)

        if plotFig:
            row = i // cols
            col = i % cols

            ax = axs[row, col]  # Select the appropriate subplot

            # Color the dots based on a sequence
            sequence = np.linspace(0, 1, len(x__))  # Create a sequence of values from 0 to 1
            colors = cmap(sequence)  # Map the sequence to colors using the chosen colormap

            ax.scatter(x__, y__, s=10, c=colors)  # 's' controls the size of the points, 'c' sets the colors

            # Add labels and a title to each subplot
            ax.set_title(f'pairID: {pairID}')

            # Hide x and y-axis ticks and tick labels
            ax.set_xticks([])
            ax.set_yticks([])

    if plotFig:
        plt.tight_layout()  # Adjust subplot layout for better visualization
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    mean_correlation_coefficients = np.array(mean_correlation_coefficients)
    p_value = np.nanmean(mean_correlation_coefficients < 0)
    print(f"{exp_name} p value = {p_value}")

    # Return mean_correlation_coefficients along with recorded_data
    return mean_correlation_coefficients, np.array(mean_parameters), np.array(x_partials), np.array(y_partials)


if testMode:
    mean_correlation_coefficients_, mean_parameters_, x_partials_, y_partials_ = run_NMPH(
        co_activations_flatten_[:4], weight_changes_flatten_[:4], pairIDs_[:4])
else:
    mean_correlation_coefficients_, mean_parameters_, x_partials_, y_partials_ = run_NMPH(
        co_activations_flatten_, weight_changes_flatten_, pairIDs_)


if not testMode:
    np.save(f'{directory_path}/temp/mean_correlation_coefficients_.npy', mean_correlation_coefficients_)
    np.save(f'{directory_path}/temp/mean_parameters_.npy', mean_parameters_)
    np.save(f'{directory_path}/temp/x_partials_.npy', x_partials_)
    np.save(f'{directory_path}/temp/y_partials_.npy', y_partials_)


x_partials_ = x_partials_.flatten()
y_partials_ = y_partials_.flatten()
mean_parameters_avg = np.mean(mean_parameters_, axis=0)


def plot_scatter_and_cubic(x_partials, y_partials, mean_parameters):
    def cubic_function(_x, a, b, c, d):
        print(f"a={a}, b={b}, c={c}, d={d}")
        return a * _x ** 3 + b * _x ** 2 + c * _x + d
    # Scatter plot
    plt.scatter(x_partials, y_partials, label='Data Points', color='green', marker='o', s=30)

    # Fit cubic curve using curve_fit
    # popt, _ = curve_fit(cubic_function, x_partials_, y_partials_)

    # Generate points for the fitted cubic curve
    x_fit = np.linspace(min(x_partials), max(x_partials), 100)
    y_fit = cubic_function(x_fit, *mean_parameters)

    # Plot the fitted cubic curve
    plt.plot(x_fit, y_fit, label='Fitted Cubic Curve', color='red')

    # Add labels and a legend
    plt.xlabel('X Partials')
    plt.ylabel('Y Partials')
    plt.legend()

    # Show the plot
    plt.show()


# plot_scatter_and_cubic(x_partials_, y_partials_, mean_parameters_avg)


# # Example usage
# x__ = np.array([-0.01337063, -0.01693626, -0.03028395, -0.03174323, -0.04779751, -0.05754384, -0.06399068, -0.07061033, -0.09292604, -0.08798023, -0.095779, -0.08514577, -0.09195771, -0.09968637, -0.08177572, -0.08894159, -0.09934128, -0.08480207, -0.08318947, -0.09024453, -0.08300266, -0.10122141, -0.10633385, -0.10665936, -0.10781459, -0.1259854, -0.13187735, -0.13555743, -0.140931, -0.14450621, -0.1364753, -0.15242977, -0.14012327, -0.10739904])
# y__ = np.array([1.71735883e-06, -1.57840550e-05, 3.84114683e-05, 4.14438546e-05, 1.91703439e-05, 4.61861491e-05, 3.67611647e-05, 1.37425959e-05, -5.71087003e-06, -2.46353447e-05, -1.38916075e-05, 7.54334033e-05, 7.71433115e-05, 4.81046736e-05, 2.15470791e-05, -2.89455056e-06, -1.20028853e-05, -2.79136002e-05, 1.59293413e-05, -6.66454434e-06, -2.77347863e-05, -4.52324748e-05, -2.76193023e-05, -4.45954502e-05, -1.90772116e-05, -3.65227461e-05, -2.76044011e-05, -1.98855996e-05, -1.44354999e-05, -3.10726464e-05, 1.36781484e-04, 1.21731311e-04, 1.04047358e-04, 7.56606460e-05])
# mean_correlation_coefficients, mean_params = cubic_fit_correlation(x__, y__)
# print(f"The averaged correlation coefficients for the 10 folds are: {mean_correlation_coefficients}")

