import torch as T
import torchvision as TV
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from utils import checkfolders
from config import Hyper, Constants
from train import train

# It all starts here
def main():
    print("\n"*10)
    print("-"*100)
    print("Start of Person Detection in Images")
    Hyper.display()
    checkfolders()
    train()
    print("-"*100)
    #train()

    print("\n"*5)  
    print("-"*100)
    Hyper.display()
    print("End of Person Detection in Images")
    print("-"*100)
    
if __name__ == "__main__":
    main()