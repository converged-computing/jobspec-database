#==============================================================================
# IMPORT DEPENDENCIES
#==============================================================================

print("\nImporting dependencies ---------------------------------------------------------\n")

from contextlib import redirect_stdout
import io
from time import time

from src import physical_models
program_start_time = time()
import numpy as np
from LRFutils import logs, archive, color, progress
import os
import yaml
import json
import matplotlib.pyplot as plt
import random
from src import mltools
import datetime

os.environ["HDF5_USE_FILE_LOCKINGS"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
os.environ["http_proxy"] = "http://11.0.0.254:3142/"
os.environ["https_proxy"] = "http://11.0.0.254:3142/"
os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

import tensorrt
import tensorflow as tf
# tf.config.experimental.enable_tensor_float_32_execution(enabled=True)
tf.random.set_seed(0)
# tf.config.run_functions_eagerly(True)
from keras.layers import Input, Dense, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Reshape, Conv3DTranspose, Flatten, Concatenate, Dropout
import keras.backend

print("\nEnd importing dependencies -----------------------------------------------------\n")

archive_path = archive.new(verbose=True)
print("")

#==============================================================================
# LOAD DATASET
#==============================================================================

# Load dataset ----------------------------------------------------------------

cpt = 0
dataset_path = "/scratch-local/vforiel/dataset"

logs.info("Loading dataset...")
bar = progress.Bar(max=len(os.listdir(dataset_path)))

dataset = {
    "dust_wavelenght": [],
    "dust_cube": [],
    "dust_map_at_250um": [],
    "CO_velocity": [],
    "CO_cube": [],
    "N2H_velocity": [],
    "N2H_cube": [],
    "space_range": [],
    "total_mass": [],
    "max_temperature": [],
    "plummer_max": [],
    "plummer_radius": [],
    "plummer_slope": [],
    "plummer_slope_log": [],
    "plummer_profile_1D": [],
}

for file in os.listdir(dataset_path):

    cpt += 1
    if not cpt%1 == 0:
        continue

    bar(cpt, prefix=mltools.sysinfo.get())
    
    file_path = os.path.join(dataset_path, file)

    data = np.load(file_path)

    # if not os.path.exists("test"):
    #     os.mkdir("test")
    # plt.figure()
    # plt.imshow(data["dust_cube"][np.argmin(np.abs(data["dust_freq"]-2.9979e8/250.0e-6)),:,:])
    # plt.colorbar()
    # plt.title(f"nH={data['n_H']:.2e}, r={data['r']:.2e}, p={data['p']:.2e}")
    # plt.savefig(f"test/{cpt}.png")
    # plt.close()

    dataset["dust_wavelenght"].append(np.array([250.,])) # [um]
    dataset["dust_cube"].append(data["dust_cube"].reshape(*data["dust_cube"].shape, 1)) # adding a channel dimension
    dataset["dust_map_at_250um"].append(data["dust_cube"][np.argmin(np.abs(data["dust_freq"]-2.9979e8/250.0e-6)),:,:].reshape(*data["dust_cube"].shape[1:], 1))
    dataset["CO_velocity"].append(data["CO_v"])
    dataset["CO_cube"].append(data["CO_cube"].reshape(*data["CO_cube"].shape, 1)) # adding a channel dimension
    dataset["N2H_velocity"].append(data["N2H_v"])
    dataset["N2H_cube"].append(data["N2H_cube"].reshape(*data["N2H_cube"].shape, 1)) # adding a channel dimension
    dataset["space_range"].append(data["space_range"])
    dataset["total_mass"].append(np.array([data["mass"]]))
    dataset["max_temperature"].append(np.array([np.amax(data["dust_temperature"])]))
    dataset["plummer_max"].append(np.array([data["n_H"]]))
    dataset["plummer_radius"].append(np.array([data["r"]]))
    dataset["plummer_slope"].append(np.array([data["p"]]))
    dataset["plummer_slope_log"].append(np.array([np.log10(data["p"])]))
    dataset["plummer_profile_1D"].append(
        np.array([physical_models.plummer(data["space_range"], data["n_H"], data["r"], data["p"])])
    )

logs.info("Dataset loaded. ✅")

# Process dataset -------------------------------------------------------------

# Fraction of the dataset used for validation and test
val_frac = 0.3 # managed by keras in fit() method
test_frac = 0.1

# Number of data in the test dataset
num_data = len(dataset[list(dataset.keys())[0]])
num_test = int(num_data * test_frac)

# Indices of the data for each dataset
test_indices = random.sample(set(range(num_data)), num_test)
train_indices = set(range(num_data)) - set(test_indices)

# Create train and test datasets
train_dataset = {}
test_dataset = {}
for label in dataset.keys():
    train_dataset[label] = np.array(dataset[label])[list(train_indices)]
    test_dataset[label] = np.array(dataset[label])[list(test_indices)]


# train_dataset, val_dataset, test_dataset = dataset.split(val_frac, test_frac)

#==============================================================================
# BUILD MODEL
#==============================================================================

# Design model ----------------------------------------------------------------

# Inputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

input_fields = [
    "dust_map_at_250um",
]

inputs = {}
normalized_inputs = {}
for label in input_fields:
    inputs[label] = Input(shape=dataset[label][0].shape, name=label)
    normalized_inputs[label] = tf.keras.layers.Normalization(axis=None, mean=np.mean(dataset[label]), variance=np.mean(dataset[label]), invert=False)(inputs[label])

# Network ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	

x = Conv2D(16, (5, 5), activation='linear', padding='same')(normalized_inputs["dust_map_at_250um"])
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
x = Dense(32, activation='linear')(x)
x = Dropout(0.3)(x, training=True)
# Pmax = Dense(128, activation='linear')(x)
Prad = Dense(32, activation='linear')(x)
# Pslope = Dense(128, activation='linear')(x)
# Pslopelog = Dense(32, activation='linear')(x)
# P1d = Dense(128, activation='linear')(x)

# Outputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

normalized_outputs = {
    # "total_mass": Dense(1, activation='linear')(total_mass),
    # "max_temperature": Dense(1, activation='linear')(x),
    # "plummer_max": Dense(1, activation='linear')(Pmax),
    "plummer_radius": Dense(1, activation='linear')(Prad),
    # "plummer_slope": Dense(1, activation='linear')(Pslope),
    # "plummer_slope_log": Dense(1, activation='linear')(Pslopelog),
    # "plummer_profile_1D": Dense(64, activation='linear')(P1d),
}

outputs = {}
for label, value in normalized_outputs.items():
    outputs[label] = tf.keras.layers.Normalization(axis=None, mean=np.mean(dataset[label]), variance=np.std(dataset[label]), invert=True, name=label)(value)

model = tf.keras.Model(inputs, outputs)


# Compile, show and train -----------------------------------------------------

logs.info("Building model...")

loss="mean_squared_error"
optimizer="adam"
metrics=[tf.keras.metrics.MeanAbsoluteError(name="MAE"),]

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
model.summary()

# Filter dataset
train_x = {}
train_y = {}
test_x = {}
test_y = {}

for label in dataset.keys():
    if label in model.input_names:
        train_x[label] = train_dataset[label]
        test_x[label] = test_dataset[label]
    if label in model.output_names:
        train_y[label] = train_dataset[label]
        test_y[label] = test_dataset[label]

logs.info("Model built. ✅")

#==============================================================================
# TRAIN MODEL
#==============================================================================

epochs = 5000
batch_size=100

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.bar = progress.Bar(epochs)
    def on_epoch_end(self, epoch, logs=None):
        self.bar(epoch+1, prefix = f"Loss: {logs['loss']:.2e} | {mltools.sysinfo.get()}")

log_dir = f"{archive_path}/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


start_training = time()
history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_split=val_frac, verbose=0, callbacks=[CustomCallback(), tensorboard_callback])
trining_time = time() - start_training

#==============================================================================
# PREDICTION
#==============================================================================

print("\n\nPredictions --------------------------------------------------------------------\n\n")

N = 1000
p_list = []
for i in range(N):
    # print("Round", i, "of", N, "-----------------------------------------------")
    p = model.predict(test_x)
    p_list.append(p)
    # for label in p.keys():
    #     print("   ", label)
    #     for j in range(len(p[label])):
    #         print("      Expected:", test_y[label][j].item(), "Predicted:", p[label][j].item())

expectations = {}
predictions = {}
for label in test_y.keys():
    expectations[label] = []
    predictions[label] = []
    for i in range(num_test):
        expectations[label].append(test_y[label][i].item())
        predictions[label].append([])
        for j in range(N):
            predictions[label][i].append(p_list[j][label][i].item())


#==============================================================================
# SAVE RESULTS
#==============================================================================

# Save model reference --------------------------------------------------------

def save_reference(model, reference_path, archive_path):

    # Create reference folder ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if not os.path.isdir(os.path.join(reference_path, "output")):
        os.makedirs(os.path.join(reference_path, "output"))
    if not os.path.isdir(os.path.join(reference_path, "problem")):
        os.makedirs(os.path.join(reference_path, "problem"))

    # Getting mode id ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    try:
        with open(os.path.join(reference_path, "model_list.yml"), "r") as f:
            models = yaml.safe_load(f)

        last_model = int(list(models.keys())[-1], 16) # get last model id from hexa
        new_model = last_model + 1
        new_model = hex(new_model)[2:] # convert to hexa
        new_model = "0"*(4-len(new_model)) + new_model # pad with 0 to reach 4 hexa digits

    except FileNotFoundError:
        models = {}
        new_model = "0000"

    # Save model reference ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    models[str(new_model)] = archive_path

    # Global list
    with open(os.path.join(reference_path, "model_list.yml"), "w") as f:
        yaml.dump(models, f)

    # Output list
    for output in model.output_names:
        with open(os.path.join(reference_path, f"output/{output}.yml"), "a") as f:
            f.write(f"'{new_model}': {archive_path}\n")

    # Problem list
    problem = ",".join(model.input_names) + "---" + ",".join(model.output_names)
    with open(os.path.join(reference_path, f"problem/{problem}.yml"), "a") as f:
        f.write(f"'{new_model}': {archive_path}\n")

    return new_model

new_model = save_reference(model, "data/model_comparison", archive_path)

# Convert metrics to strings
for i, metric in enumerate(metrics):
    if not isinstance(metric, str):
        metrics[i] = metric.name

# Save model ------------------------------------------------------------------

model.save(os.path.join(archive_path,"model.h5"))

# History treatment ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if len(model.output_names) == 1:
    output_name = model.output_names[0]
    keys = list(history.history.keys())
    for key in keys:
        if key != "loss" and key != "val_loss":
            if key.startswith("val_"):
                new_key = "val_" + output_name + "_" + key[4:]
            else:
                new_key = output_name + "_" + key
            history.history[new_key] = history.history[key]
            del history.history[key]

N = len(history.history)
N1 = int(np.sqrt(N))
N2 = N1
if N1*N2 < N:
    N2 += 1
if N1*N2 < N:
    N1 += 1

fig, axs = plt.subplots(N1, N2, figsize=(N2*5, N1*5))
axs = axs.flatten()
i = 0
for key, value in history.history.items():
    if key.startswith("val_"):
        continue
    axs[i].plot(value, label=key)
    axs[i].plot(history.history[f"val_{key}"], label=f"val_{key}")
    axs[i].set_title(key)
    axs[i].legend()
    axs[i].set_xlabel("Epoch")
    axs[i].set_ylabel(key)
    i += 1

    axs[i].plot(value, label=key)
    axs[i].plot(history.history[f"val_{key}"], label=f"val_{key}")
    axs[i].set_title(key + " in log:log scale")
    axs[i].legend()
    axs[i].set_xlabel("Epoch")
    axs[i].set_ylabel(key)
    axs[i].set_yscale("log")
    axs[i].set_xscale("log")    
    i += 1

fig.savefig(f"{archive_path}/history.png")

# Save metadata ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def summary(model):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    return short_model_summary

np.savez_compressed(f"{archive_path}/history.npz",
    history=history.history,
    training_time=trining_time,
)

np.savez_compressed(f"{archive_path}/metadata.npz",
    summary=summary(model),
    archive_path=archive_path,
    val_frac=val_frac,
    test_frac=test_frac,
    epochs=epochs,
    batch_size=batch_size,
    loss=loss,
    optimizer=optimizer,
    metrics=metrics,
    id=new_model,
    dataset_size=num_data,
)

np.savez_compressed(f"{archive_path}/inference.npz", test_x=test_x, test_y=test_y, predictions=predictions, expectations=expectations)

def plot(model, archive_path):
    trainable_count = np.sum([keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([keras.backend.count_params(w) for w in model.non_trainable_weights])

    return tf.keras.utils.plot_model(
        model,
        to_file=f"{archive_path}/model.png",
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
        layer_range=None,
        show_layer_activations=True,
    )

plot(model, archive_path)

# End of program
spent_time = time() - program_start_time
logs.info(f"End of program. ✅ Took {int(spent_time//60)} minutes and {spent_time%60:.2f} seconds \n -> Results dans be found in {archive_path} folder.")
