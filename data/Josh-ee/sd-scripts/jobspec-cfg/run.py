import subprocess
import os
import json
import toml
import requests

"""
to plot the logs, paste below into terminal:
tensorboard --logdir=logs --port=6006
"""



model_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors"
model_filename = os.path.basename(model_url)

# Check if the model file exists in the current directory
if not os.path.exists(model_filename):
    print(f"Downloading {model_filename}...\n This may take 5min")
    response = requests.get(model_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Save the model file in the current directory
        with open(model_filename, 'wb') as f:
            f.write(response.content)
        print(f"{model_filename} downloaded successfully.")
    else:
        print(f"Failed to download {model_filename}. Status code: {response.status_code}")
else:
    print(f"{model_filename} already exists in the current directory.")

dataset_config_file_path = "configs/dataset_config.toml"


config_file_path = "configs/training_config.toml"


# Define the path to train_network.py
train_network_path = "train_network.py"

# Construct the command to run train_network.py with subprocess, including dataset_config_file_path
command = [
    "python", train_network_path,
    "--config_file", config_file_path,
    "--dataset_config", dataset_config_file_path
    # Add more arguments as needed
]

# Execute the command
subprocess.run(command)

# # Optionally, clean up the temporary files after training
# os.remove(config_file_path)
# os.remove(dataset_config_file_path)
