print('111')

import numpy as np
import math
import glob
import sys

print('222')

import tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

print('333')

from sklearn.utils import class_weight

print('444')

import IvysaurusModel
import IvysaurusModel_VGG
import IvysaurusModel_VGG_BN
import IvysaurusModel_VGG_SEP
import IvysaurusModel_VGG_EX
import IvysaurusModel_VGG_RES

print('555')

###########################################################
###########################################################

def WhoseThatPokemon(ndimensions, nclasses, ntrackvars, nshowervars, mode) :
    
   if (mode == '0') :
      print('IvysaurusModel_VGG')
      return IvysaurusModel_VGG.IvysaurusIChooseYou(ndimensions, nclasses, ntrackvars, nshowervars)
   elif (mode == '1') :
      print('IvysaurusModel_VGG_BN')
      return IvysaurusModel_VGG_BN.IvysaurusIChooseYou(ndimensions, nclasses, ntrackvars, nshowervars)
   elif (mode == '2') :
      print('IvysaurusModel_VGG_SEP')
      return IvysaurusModel_VGG_SEP.IvysaurusIChooseYou(ndimensions, nclasses, ntrackvars, nshowervars)
   elif (mode == '3') :
      print('IvysaurusModel_VGG_EX')
      return IvysaurusModel_VGG_EX.IvysaurusIChooseYou(ndimensions, nclasses, ntrackvars, nshowervars)
   elif (mode == '4') :
      print('IvysaurusModel_VGG_RES')
      return IvysaurusModel_VGG_RES.IvysaurusIChooseYou(ndimensions, nclasses, ntrackvars, nshowervars)

   return IvysaurusModel_VGG.IvysaurusIChooseYou(ndimensions, nclasses, ntrackvars, nshowervars)
      

###########################################################
###########################################################

print('AAAAAAA')

physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0 :
    for gpu in physical_devices :
       tensorflow.config.experimental.set_memory_growth(gpu, True)

print('1111111')

###########################################################

# VGG, VGG_BN, VGG_SEP, VGG_EX, VGG_RES
MODE_VGG = sys.argv[1]

print('MODE_VGG: ', MODE_VGG)

dimensions = 24
nClasses = 5

nTrackVars = 10 # nTrackChildren, nShowerChildren, nGrandChildren, nChildHits, childEnergy, childTrackScore, trackLength, trackWobble, trackScore, momComparison
nShowerVars = 3 # displacement, dca, trackStubLength

nEpochs = 10
batchSize = 64
learningRate = 1e-4

###########################################################

# Here we'll get our information...

print('BBBBBBB')

startGridU_train = np.empty((0, dimensions, dimensions, 1))
startGridV_train = np.empty((0, dimensions, dimensions, 1))
startGridW_train = np.empty((0, dimensions, dimensions, 1))

endGridU_train = np.empty((0, dimensions, dimensions, 1))
endGridV_train = np.empty((0, dimensions, dimensions, 1))
endGridW_train = np.empty((0, dimensions, dimensions, 1))

trackVars_train = np.empty((0, nTrackVars))
showerVars_train = np.empty((0, nShowerVars))

y_train = np.empty((0, nClasses))

startGridU_test = np.empty((0, dimensions, dimensions, 1))
startGridV_test = np.empty((0, dimensions, dimensions, 1))
startGridW_test = np.empty((0, dimensions, dimensions, 1))

endGridU_test = np.empty((0, dimensions, dimensions, 1))
endGridV_test = np.empty((0, dimensions, dimensions, 1))
endGridW_test = np.empty((0, dimensions, dimensions, 1))

trackVars_test = np.empty((0, nTrackVars))
showerVars_test = np.empty((0, nShowerVars))

y_test = np.empty((0, nClasses))

trainFileNames = glob.glob('/storage/users/mawbyi1/Ivysaurus/files/grid24/*/ivysaurus_*.npz')
print(trainFileNames)

for trainFileName in trainFileNames :
    print('Reading file: ', str(trainFileName),', This may take a while...')
    
    data = np.load(trainFileName)

    startGridU_train = np.concatenate((startGridU_train, data['startGridU_test']), axis=0)
    startGridV_train = np.concatenate((startGridV_train, data['startGridV_test']), axis=0)
    startGridW_train = np.concatenate((startGridW_train, data['startGridW_test']), axis=0)
    
    startGridU_test = np.concatenate((startGridU_test, data['startGridU_train']), axis=0)
    startGridV_test = np.concatenate((startGridV_test, data['startGridV_train']), axis=0) 
    startGridW_test = np.concatenate((startGridW_test, data['startGridW_train']), axis=0)
    
    endGridU_train = np.concatenate((endGridU_train, data['endGridU_test']), axis=0)
    endGridV_train = np.concatenate((endGridV_train, data['endGridV_test']), axis=0)
    endGridW_train = np.concatenate((endGridW_train, data['endGridW_test']), axis=0)
    
    endGridU_test = np.concatenate((endGridU_test, data['endGridU_train']), axis=0)
    endGridV_test = np.concatenate((endGridV_test, data['endGridV_train']), axis=0)
    endGridW_test = np.concatenate((endGridW_test, data['endGridW_train']), axis=0)
    
    trackVars_train = np.concatenate((trackVars_train, data['trackVars_test']), axis=0)
    trackVars_test = np.concatenate((trackVars_test, data['trackVars_train']), axis=0)

    showerVars_train = np.concatenate((showerVars_train, data['showerVars_test']), axis=0)
    showerVars_test = np.concatenate((showerVars_test, data['showerVars_train']), axis=0)
    
    y_train = np.concatenate((y_train, data['y_test']), axis=0)
    y_test = np.concatenate((y_test, data['y_train']), axis=0)

print('CCCCCC')

###########################################################

print('startGridU_train: ', startGridU_train.shape)
print('startGridV_train: ', startGridV_train.shape)
print('startGridW_train: ', startGridW_train.shape)
print('startGridU_test: ', startGridU_test.shape)
print('startGridV_test: ', startGridV_test.shape)
print('startGridW_test: ', startGridW_test.shape)
   
print('endGridU_train: ', endGridU_train.shape)    
print('endGridV_train: ', endGridV_train.shape)
print('endGridW_train: ', endGridW_train.shape)
print('endGridU_test: ', endGridU_test.shape)     
print('endGridV_test: ', endGridV_test.shape)     
print('endGridW_test: ', endGridW_test.shape) 
    
print('trackVars_train: ', trackVars_train.shape)    
print('trackVars_test: ', trackVars_test.shape)

print('showerVars_train: ', showerVars_train.shape)    
print('showerVars_test: ', showerVars_test.shape)

print('y_train: ', y_train.shape)
print('y_test', y_test.shape)

###########################################################

# Set the most deviating graph to zero?
'''
nU_train = (startGridU_train > 0.00000000000001).sum(axis=1).sum(axis=1)
nV_train = (startGridV_train > 0.00000000000001).sum(axis=1).sum(axis=1)
nW_train = (startGridW_train > 0.00000000000001).sum(axis=1).sum(axis=1)

nU_test = (startGridU_test > 0.00000000000001).sum(axis=1).sum(axis=1)
nV_test = (startGridV_test > 0.00000000000001).sum(axis=1).sum(axis=1)
nW_test = (startGridW_test > 0.00000000000001).sum(axis=1).sum(axis=1)

deltaUV_train = np.fabs(nU_train - nV_train)
deltaUW_train = np.fabs(nU_train - nW_train)
deltaVU_train = np.fabs(nU_train - nV_train)
deltaVW_train = np.fabs(nV_train - nW_train)
deltaWU_train = np.fabs(nU_train - nW_train)
deltaWV_train = np.fabs(nV_train - nW_train)

deltaUV_test = np.fabs(nU_test - nV_test)
deltaUW_test = np.fabs(nU_test - nW_test)
deltaVU_test = np.fabs(nU_test - nV_test)
deltaVW_test = np.fabs(nV_test - nW_test)
deltaWU_test = np.fabs(nU_test - nW_test)
deltaWV_test = np.fabs(nV_test - nW_test)

deltaU_train = deltaUV_train + deltaUW_train
deltaV_train = deltaVU_train + deltaVW_train
deltaW_train = deltaWU_train + deltaWV_train

deltaU_test = deltaUV_test + deltaUW_test
deltaV_test = deltaVU_test + deltaVW_test
deltaW_test = deltaWU_test + deltaWV_test

brokenUIndex_train = (deltaU_train > deltaV_train) & (deltaU_train > deltaW_train)
brokenVIndex_train = (deltaV_train > deltaU_train) & (deltaV_train > deltaW_train)
brokenWIndex_train = (deltaW_train > deltaU_train) & (deltaW_train > deltaV_train)
ambiguousIndex_train = (brokenUIndex_train == False) & (brokenVIndex_train == False) & (brokenWIndex_train == False) 

brokenUIndex_test = (deltaU_test > deltaV_test) & (deltaU_test > deltaW_test)
brokenVIndex_test = (deltaV_test > deltaU_test) & (deltaV_test > deltaW_test)
brokenWIndex_test = (deltaW_test > deltaU_test) & (deltaW_test > deltaV_test)
ambiguousIndex_test = (brokenUIndex_test == False) & (brokenVIndex_test == False) & (brokenWIndex_test == False) 

startGridU_train[np.where(ambiguousIndex_train)[:][:]] = 0
startGridU_train[np.where(brokenUIndex_train)[:][:]] = 0
startGridV_train[np.where(brokenVIndex_train)[:][:]] = 0
startGridW_train[np.where(brokenWIndex_train)[:][:]] = 0
endGridU_train[np.where(ambiguousIndex_train)[:][:]] = 0
endGridU_train[np.where(brokenUIndex_train)[:][:]] = 0
endGridV_train[np.where(brokenVIndex_train)[:][:]] = 0
endGridW_train[np.where(brokenWIndex_train)[:][:]] = 0

startGridU_test[np.where(ambiguousIndex_test)[:][:]] = 0
startGridU_test[np.where(brokenUIndex_test)[:][:]] = 0
startGridV_test[np.where(brokenVIndex_test)[:][:]] = 0
startGridW_test[np.where(brokenWIndex_test)[:][:]] = 0
endGridU_test[np.where(ambiguousIndex_test)[:][:]] = 0
endGridU_test[np.where(brokenUIndex_test)[:][:]] = 0
endGridV_test[np.where(brokenVIndex_test)[:][:]] = 0
endGridW_test[np.where(brokenWIndex_test)[:][:]] = 0
'''
###########################################################

mirrored_strategy = tensorflow.distribute.MultiWorkerMirroredStrategy()

print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

with mirrored_strategy.scope():
   # VGG, VGG_BN, VGG_SEP, VGG_EX, VGG_RES
   ivysaurusCNN = WhoseThatPokemon(dimensions, nClasses, nTrackVars, nShowerVars, MODE_VGG)
   ivysaurusCNN.summary()

   # Define the optimiser and compile the model
   optimiser = Adam(learning_rate=learningRate)
   ivysaurusCNN.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])

###########################################################

# Create class weights

indexVector = np.argmax(y_test, axis=1)

# muons = 0, protons = 1, pions = 2, electrons = 3, photons = 4, other = 5

nMuons = np.count_nonzero(indexVector == 0)    
nProtons = np.count_nonzero(indexVector == 1)  
nPions = np.count_nonzero(indexVector == 2)  
nElectrons = np.count_nonzero(indexVector == 3)  
nPhotons = np.count_nonzero(indexVector == 4)  

# Normalise to largest
maxParticle = max(nMuons, nProtons, nPions, nElectrons, nPhotons)

classWeights = {0: maxParticle/nMuons, 1: maxParticle/nProtons, 2: maxParticle/nPions, 3: maxParticle/nElectrons, 4: maxParticle/nPhotons}

print('Class Weights: ')
print(classWeights)

###########################################################

# Fit that model!

# Reduce the learning rate by a factor of ten when required
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6, verbose=1)
history = ivysaurusCNN.fit([startGridU_train, endGridU_train, startGridV_train, endGridV_train, startGridW_train, endGridW_train, trackVars_train, showerVars_train], y_train, 
    batch_size = batchSize, validation_data=([startGridU_test, endGridU_test, startGridV_test, endGridV_test, startGridW_test, endGridW_test, trackVars_test, showerVars_test], y_test), 
    shuffle=True, epochs=nEpochs, class_weight=classWeights, callbacks=[reduce_lr], verbose=2) 

###########################################################

# Save the model

print('Saving model...')

if (MODE_VGG == '0') :
   ivysaurusCNN.save('/home/hpc/30/mawbyi1/Ivysaurus/files/grid24/my_model_VGG')
elif (MODE_VGG == '1') :
   ivysaurusCNN.save('/home/hpc/30/mawbyi1/Ivysaurus/files/grid24/my_model_VGG_BN')
elif (MODE_VGG == '2') :
   ivysaurusCNN.save('/home/hpc/30/mawbyi1/Ivysaurus/files/grid24/my_model_VGG_SEP')
elif (MODE_VGG == '3') :
   ivysaurusCNN.save('/home/hpc/30/mawbyi1/Ivysaurus/files/grid24/my_model_VGG_EX')
elif (MODE_VGG == '4') :
   ivysaurusCNN.save('/home/hpc/30/mawbyi1/Ivysaurus/files/grid24/my_model_VGG_RES')

