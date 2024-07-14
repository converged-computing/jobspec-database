import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from Data import genData as genData
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from src import train
from src import loader
from src.models import model as mainmodel
from src import custom_plots as cp


def main():


    for dyn_type in ["Motion","Scale","Intensity"]:

        print("Dynamics type: ", dyn_type)

        if dyn_type == "Motion":
            data_folder = np.load('Data/dataset_motion.npy')
        if dyn_type == "Scale":
            data_folder = np.load('Data/dataset_Scale_nu.npy')
        if dyn_type == "Intensity":
            data_folder = np.load('Data/dataset_intensity.npy')

        data_train = data_folder
        dt = 0.2
        train_dataloader, test_dataloader, train_x, val_x  = loader.getLoader_folder(data_train, split=True)
        

        a = []
        b = []

        gamma1 = []	
        gamma2 = []

        plt.figure()

        
        for inits in [ -10.0, -5.0, -1.0,0.0, 1.0, 5.0, 10.0]:
        
            latentEncoder = mainmodel.EndPhys(dt = dt,  
                                pmodel = "Damped_oscillation",
                                init_phys = inits, 
                                initw=True)

            latentEncoder, log  = train.train(latentEncoder, 
                                            train_dataloader, 
                                            test_dataloader,
                                            init_phys = inits,                                 
                                            loss_name='latent_loss')
            
            gamma1.append(latentEncoder.pModel.alpha[0].detach().cpu().numpy().item())
            gamma2.append(latentEncoder.pModel.beta[0].detach().cpu().numpy().item())
            
            a.append( [element["alpha"] for element in log  ])
            b.append( [element["beta"] for element in log  ] )

        a = np.array(a)
        b = np.array(b)
        cp.plotAreas(a, 3.99, dyn_type+"_gamma1")
        cp.plotAreas(b, 0.08, dyn_type+"_gamma2")

        gamma1 = np.array(gamma1)
        gamma2 = np.array(gamma2)

        print("Best gamma1: ", gamma1.mean() , "\pm", gamma1.std())
        print("Best gamma2: ", gamma2.mean() , "\pm", gamma2.std())
    
if __name__ == "__main__":
    main()
