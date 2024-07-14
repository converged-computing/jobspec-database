import numpy as np
import json
import os

import keras
import tensorflow as tf
from keras.layers import Input, Layer
from keras_applications import inception_v3, inception_resnet_v2, resnet
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import glob
from math import log, inf
import itertools
import pickle
from datetime import datetime

def load_images():
    images = dict()

    for label in LABELS:
        files = glob.glob("images/square_%dp/squares_%dp_%s_*.png" % (CENTER_SIZE, CENTER_SIZE, label))
        print('images for label', label, len(files))
        images[label] = [image.load_img(f, target_size=(CENTER_SIZE, CENTER_SIZE)) for f in files[:MAX_IMAGES_PER_CLASS]]

    return images

def classify(img, model='I_V3'):

    # ugly
    if model == 'RN50':
        model = rn50_model
        application = resnet
    elif model == 'RN152':
        model = rn152_model
        application = resnet
    elif model == 'I_V3':
        model = i_v3_model
        application = inception_v3
    elif model == 'I_RN_V2':
        model = i_rn_v2_model
        application = inception_resnet_v2
        
    # preprocess
    img_prep = image.img_to_array(img)
    img_prep = np.expand_dims(img_prep, axis=0)
    img_prep = application.preprocess_input(img_prep,
        backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    # predict
    predictions = model.predict(img_prep)
    #results
    return application.decode_predictions(predictions, top=2, utils=keras.utils)[0]

def evaluate(model_name:str, images: {str: []}, adv_program: []):
    results = {}
    start = int((IMAGE_SIZE - CENTER_SIZE) / 2)
    end = int((IMAGE_SIZE - CENTER_SIZE) / 2 + CENTER_SIZE)
    for k, class_images in images.items():
        class_results = []
        for img in class_images:
            adv_program[start:end, start:end, :] = img
#             Append best match
            class_results.append(classify(adv_program, model_name)[0][1])
        
        results[k] = class_results
    return results

flatten = lambda l: [item for sublist in l for item in sublist]
value_dict = dict()
key_dict = dict()

def results_to_matrix(results: {str:[]}) -> [[]]:
    keys = results.keys()
    for key in keys:
        if(key not in key_dict.keys()):
            key_dict[key]=len(key_dict.keys())
    values = set(flatten(results.values()))
    for val in values:
        if(val not in value_dict.keys()):
            value_dict[val]=len(value_dict.keys())
 #   if(isinstance(matrix, list)):
    matrix = np.zeros((len(keys), len(value_dict.keys())))
    for i in results.keys():
        for j in results[i]:
            matrix[key_dict[i]][value_dict[j]]+=1
    #for value in results.values():
    #    el=[]
    #    for c in classes: el.append(value.count(c))
    #    matrix.append(el)
    return matrix

def compute_matrix_loss(matrix: [[]]) -> float:
    matrix = matrix/matrix.sum(axis=1)[:,None]
    usedLabels = []
    loss=0
    matrix = matrix.transpose()
    for i in matrix:
        highest_probability = 0
        probability_id = inf
        for j in range(0,len(i)):
            if(j not in usedLabels):
                if(i[j]>=highest_probability):
                    highest_probability=i[j]
                    probability_id=j
        usedLabels.append(probability_id)
        if(highest_probability==0):
            loss = loss+10
        else:
            loss = loss - log(highest_probability)
    loss = loss+(len(matrix[1])-len(matrix))*10
    return loss

def train(model_name:str, images:{str:[]}) -> ([[]], [[]], float):
    best_adv_program = []
    best_loss = inf
    
    best_matrix = []
    try:
        for i in itertools.count(0):

            adv_program = np.random.rand(IMAGE_SIZE,IMAGE_SIZE,3) * 255
            adv_program = adv_program.astype(int)

            result = evaluate(model_name, images, adv_program)
            mresult = results_to_matrix(result)

            loss = compute_matrix_loss(mresult)
            if loss < best_loss:
                best_loss = loss
                best_adv_program = adv_program
                best_matrix = mresult

            print('round: ', i, 'best loss: ', best_loss)

            if best_loss < BEST_LOSS_GOAL:
                return (best_adv_program, best_matrix, best_loss)

    except KeyboardInterrupt as e:
        print('\nkeyboardInterupt:', e)
        return (best_adv_program, best_matrix, best_loss)
    except Exception as e:
        print('\nexception:', e)
        return (best_adv_program, best_matrix, best_loss)


if __name__ == "__main__":
    ### SETUP PARAMETERS ###
    CENTER_SIZE = 35
    IMAGE_SIZE = 299
    LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    MAX_IMAGES_PER_CLASS = 10
    BEST_LOSS_GOAL = 0.01
    ### END SETUP PARAMETERS ###
    
    # Load Images
    images = load_images()

    # Load model
    i_v3_model = inception_v3.InceptionV3(weights='imagenet', input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)

    try:
        best_program, best_matrix, best_loss = train('I_V3', images)

        if not os.path.exists("results/random/"):
            os.makedirs("results/random/")

        now = datetime.now()
        now_string = now.strftime("%d-%m-%Y_%H-%M-%S")

        pickle.dump(best_program, open('results/random/adv_program-%s-%.3f' % (now_string, best_loss), 'wb'))
        pickle.dump(best_matrix, open('results/random/best_matrix-%s-%.3f' % (now_string, best_loss), 'wb'))
    except Exception as e:
        print('error', e)

