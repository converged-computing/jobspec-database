from time import sleep
import time
from func.dataFunctions import *
from matplotlib import pyplot as plt, scale
from func.LSTM_pytorch import *
#from mpi4py import MPI
import torch
import torch.nn as nn

# For testing 
from mpi4py.futures import MPIPoolExecutor as Executor

if __name__ == "__main__":
    print('Program started')

    # Check for GPU availability
    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        print("CUDA (GPU support) is available on this machine.")
        print("Number of GPUs available:", torch.cuda.device_count())
        print("Name of the GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA (GPU support) is not available on this machine.")


    path_amzn = "data/Daily_NLP/merged_Amazon_AMZN.csv"
    path_apple = "data/Daily_NLP/merged_Apple_AAPL.csv"
    path_neflx = "data/Daily_NLP/merged_Netflix_NFLX.csv"
    path_tsla = "data/Daily_NLP/merged_Tesla_Inc_TSLA.csv"
    path_MSFT = "data/Daily_NLP/merged_Microsoft_Corporation_MSFT.csv"

    file_paths = [path_amzn,path_apple,path_neflx,path_tsla,path_MSFT]

    data_list_array, list_of_scalers = read_data_in_parallel_future(file_paths)
    train_and_save_model_on_data(data_list_array, list_of_scalers, num_epochs=200, learning_rate=0.001)


