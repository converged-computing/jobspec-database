"""
Author : Maniraman Periyasamy

This module is the main python file which generates and loads the U-Net model based on the
hyperparameters preset in the yaml file given as command line argument.This file is structured in such a way that all combination of parametrs given in yaml file as a list will be run sequentially.
for example, if the epoch and batchsize parameters in the yaml file are [e1,e2,e3] and [b1,b2,b3] respectively,
then 9 differents model will be train with 3X3 combinations sequentially.
Each model and its results will be saved in a different folder.

This was done so that multiple model can be trained one after the other without waiting for the user to start training of the next model.

"""




import yaml
import os
import time
import argparse

# This function pre-process, augument and split the dataset.
from data_preprocess import generateData



from data_generator import trainGenerator
from model import unet

import numpy as np
import pandas as pd
import shutil


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter", help="Enter the link to your yaml file which includes the hyperparameter",
                        default="hyperparameters/hyperparameters_reference.yaml")
    args = parser.parse_args()

    # All the hyperparameters list are loded as a dictionary named hyperPar
    with open(args.parameter) as f:
        hyperPar = yaml.load(f)

    for epoch in hyperPar['epochs']: # iterate over different epoch values
        for patchSize in hyperPar['patchSize']: # iterate over different patch sizes to be tested
            for batchSize in hyperPar['batchSize']: # iterate over different batch sizes
                for optimizer in hyperPar['optimizer']: # iterate over different optimizers to be tested
                    for unetType in hyperPar['unetType']: # iterate over different types of U-Net model being tested
                        for loss_weight in hyperPar['loss_weight']: # iterate over different combination of weights for BCE and Dice loss function if the loss type is 'combined' else ignored
                            for loss in hyperPar['loss']: # iterate over different type of loss function tested


                                # If the train data does not exist, generate the train function by pre-processing, augmenting and patching the raw SAR images.
                                if not os.path.exists(hyperPar['trainPath']):
                                    generateData(hyperPar['rawImage'], hyperPar['inputFolder'])
                                if len(os.listdir(hyperPar['trainPath'])) == 0 :
                                    generateData(hyperPar['rawImage'], hyperPar['inputFolder'])

                                START = time.time()
                                print(loss + '_' + optimizer)
                                trainSampleList = os.listdir(hyperPar['trainPath'] + '/images')
                                valSampleList = os.listdir(hyperPar['valPath'] + '/images')
                                steps_per_epoch = np.ceil(len(trainSampleList) / batchSize)
                                validation_steps = np.ceil(len(valSampleList) / batchSize)


                                # Create keras data generators from the dataset.

                                data_gen_args = dict(horizontal_flip=True, fill_mode='nearest')

                                train_Generator = trainGenerator(batch_size=batchSize,
                                                                 train_path=hyperPar['trainPath'],
                                                                 image_folder='images',
                                                                 mask_folder='masks_zones',
                                                                 aug_dict=data_gen_args,
                                                                 save_to_dir=None)

                                val_Generator = trainGenerator(batch_size=batchSize,
                                                               train_path=hyperPar['valPath'],
                                                               image_folder='images',
                                                               mask_folder='masks_zones',
                                                               aug_dict=None,
                                                               save_to_dir=None)


                                # Create a unique folder where all the results of this current model is going to be saved.
                                outputPath =  hyperPar['outputPath'] + "Data/" + '_' + str(
                                    epoch) + '_' + str(patchSize) \
                                             + '_' + str(batchSize) + '_' + optimizer + '_' + loss + '_' + \
                                             str(loss_weight) + unetType+ str(hyperPar['dilationRate'][0]) + \
                                             str(hyperPar['dropout']) + "/"

                                if not os.path.exists(outputPath): os.makedirs(outputPath)

                                outputPathCheck = outputPath + "unet_zone.hdf5"

                                # generate the model based on the hyperparameters.
                                model = unet(trainGenerator=train_Generator, valGenerator=val_Generator,
                                                 stepsPerEpoch=steps_per_epoch, validationSteps=validation_steps,
                                                 patchSize=patchSize, outputPathCheck=outputPathCheck,
                                                 outputPath=outputPath,
                                                 testPath=hyperPar['testPath'], epochs=epoch, unetType=unetType,
                                                 loss=loss,metrics=hyperPar['metrics'], optimizer=optimizer,
                                                 validationPath= hyperPar["valPath"],
                                                 threshold= 0.4, patience=30, dilationRate= hyperPar['dilationRate'][0],
                                             dropout=hyperPar['dropout'],lossWeights=loss_weight,lossType=loss)

                                # train the model and returs the train history, along with a boolean flag.
                                # All the model are restricted to stop training after 24 hours due to LME cluster restriction.
                                # This timeFlag indicates whether the training was completed or stopped due to time restriction.

                                history, timeFlag = model.train()
                                df = pd.DataFrame(history.history)
                                df.to_csv(outputPath + 'history_first.csv', index_label= False)

                                # copy the hyperparameter file into the output directory for future reference
                                shutil.copy(args.parameter,outputPath)

                                if not timeFlag:
                                    # If the model's training is completed successfully.
                                    model.test()
                                    END = time.time()
                                    print('Execution Time: ', END - START)
                                    print("simulation done completely")

                                    # An indicator to see whether the training got completed or not outside the program.
                                    with open(outputPath+"simulation1Resutl.txt","w") as f:
                                        f.write("Completed")
                                else:
                                    # If the model's training is stopped due to time restriction.
                                    END = time.time()
                                    print('Execution Time: ', END - START)
                                    print("simulation done with timed stop")
                                    with open(outputPath+"simulation1Resutl.txt","w") as f:
                                        f.write("Not Completed")





