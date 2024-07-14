#!/usr/bin/env python
# coding: utf-8
'''
El siguiente script está diseñado para ejecutar redes de transfer learning para la detección de edema macular.

AUTOR: Nuria Velasco Pérez
FECHA: Abril 2023
TRABAJO FIN DE GRADO
'''
#Importaciones necesarias.

import os
import sys
import math
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report  
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_VGG16

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_VGG19

from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input as preprocess_Xception

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_ResNet50V2

from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_ResNet101_152

from tensorflow.keras.applications import ResNet152

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_InceptionV3

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_InceptionResNetV2

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_MobileNet

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_DenseNet121

from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_DenseNet201

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_EfficientNetB0


#Obtención de los argumentos de entrada.

modelo = getattr(sys.modules[__name__], sys.argv[1])
proc = sys.argv[2]
unlock = sys.argv[3]
batch = int(sys.argv[4])
K_test = sys.argv[5]

#Selección del directorio a emplear en función a la elección sobre el preprocesamiento indicada como parámetro de entrada.

if (proc=='False'):
    directorio = 'datasets/Datos cross validation base EMD'
else:
    directorio = 'datasets/Datos cross validation EMD'

#Definición de hiperparámetros del modelo a partir de la red seleccionada (diccionarios para escala y preprocesamiento) o del valor indicado como entrada.

color = sys.argv[6] #rgba / grayscale
dic_escala = {VGG16:224,VGG19:224,Xception:299,ResNet50V2:224,ResNet101:224,ResNet152:224,InceptionResNetV2:299,MobileNet:224,DenseNet121:224,DenseNet201:224,EfficientNetB0:224,InceptionV3:299}
dic_preprocesado = {VGG16:preprocess_VGG16,
                VGG19:preprocess_VGG19,
                Xception:preprocess_Xception,
                ResNet50V2:preprocess_ResNet50V2,
                ResNet101:preprocess_ResNet101_152,
                ResNet152:preprocess_ResNet101_152,
                InceptionResNetV2:preprocess_InceptionResNetV2,
                MobileNet:preprocess_MobileNet,
                DenseNet121:preprocess_DenseNet121,
                DenseNet201:preprocess_DenseNet201,
                EfficientNetB0:preprocess_EfficientNetB0,
                InceptionV3:preprocess_InceptionV3}
'''
combine_gen() permitirá unir en un solo objeto una lista de generadores. La idea es poder unir los generadores de imágenes que hayamos obtenido de distintos origenes, para utilizar imágenes variadas en un mismo entrenamiento o en una validación o en una predicción. 
'''
def combine_gen(gens):
    while True:
        tuplas = []
        for i in gens:
            tuplas.append(next(i))
        arrays_images = []
        arrays_labels = []
        for i in tuplas:
            arrays_images.append(i[0])
            arrays_labels.append(i[1])
        images = np.concatenate(arrays_images)
        labels = np.concatenate(arrays_labels)

        yield(tuple((images,labels)))
'''
num() tiene como objetivo conocer de cuántas imágenes disponemos en total para hacer el entrenamiento y la validación, habiendo quitado la caja de test que empleemos en cada caso. 
'''
def num(directorio,K_test,K_val):
    total = 0
    lista_K = ['K1','K2','K3','K4','K5']
    lista_K.remove(K_test)
    lista_K.remove(K_val)
    for K in lista_K:
            for origen in ['iPhone','OCT','Samsung']:
                for grado in ['EMD', 'NO EMD']:
                    total += len(os.listdir(directorio + '/' + K + '/' + origen + '/' + grado + '/' + grado + '/'))

    num_train = int(total*0.8)
    num_val = int(total*0.2)
    return num_train,num_val
'''
generador_train() crea la estructua de un generador de imágenes que emplearemos tanto para el conjunto de entrenamiento como para el de validación. El beneficio que nos proporciona la función ImageDataGenerator de keras es que permite crear generadores de grandes cantidades de datos sin necesidad de que estos se carguen en memoria y, de tal forma, disminuya el rendimiento del algoritmo. Además, aprovechamos para hacer data augmentation con los parámetros de esta función: rotando horizontalmente, rotando verticalmente, cambiando el brillo y el color.
'''
def generador_train(modelo,directorio,K,origen,grado,escala):
    train_datagen = ImageDataGenerator(
        preprocessing_function = dic_preprocesado[modelo],
        horizontal_flip=True,
        vertical_flip=True,
        zca_whitening=True,
        zca_epsilon=1e-06
    )

    generator = train_datagen.flow_from_directory(
        directory = directorio + '/' + K + '/' + origen + '/' + grado,
        target_size = (escala,escala),
        color_mode = color,
        class_mode='categorical',
        batch_size = 1,
        seed = 42
    )  
    return generator

'''
train_gen() sirve para unir los generadores correspondientes a todas las subcarpetas que contienen imágenes que se van a utilizar en el entrenamiento. Como parámetros únicamente deben especificarse: la caja de test para no incluirla en este subset y la escala a la que deben estar las imágenes.
'''
def train_gen(modelo,directorio,K_test,K_val,escala=224):
    lista_K = ['K1','K2','K3','K4','K5']
    lista_K.remove(K_test)
    lista_K.remove(K_val)
    generadores = []
    for K in lista_K:
        for origen in ['iPhone','OCT','Samsung']:
            for grado in ['EMD', 'NO EMD']:
                generadores.append(generador_train(modelo,directorio,K,origen,grado,escala))
                
    generador_combinado = combine_gen(generadores)
    return generador_combinado

'''
generador_val() permitirá crear un generador de imágenes considerando las especificaciones generales indicadas en "train_datagen" de una subcarpeta en concreto. O sea, carga las imágenes de la subcarpeta correspondiente a los datos indicados como parámetros, que son: caja, dispositivo de origen y grado. Estos generadores contendrán exlucisvamente las imágenes de validación. 
'''
def generador_val(modelo,directorio,K_val,origen,grado,escala=224):
    val_datagen = ImageDataGenerator(
        preprocessing_function = dic_preprocesado[modelo]
    )

    generator = val_datagen.flow_from_directory(
        directory = directorio + '/' + K_val + '/'+ origen + '/' + grado,
        target_size = (escala,escala),
        color_mode = color,
        class_mode='categorical',
        batch_size = 1,#vamos a equilibrar proporciones, 1/80 ya que tenemos 4 cajas de train * 4 orígenes * 5 grados
        seed = 42
    )
    
    return generator

'''
val_gen() sirve para unir los generadores correspondientes a todas las subcarpetas que contienen imágenes que se van a utilizar en la validación. Como parámetros únicamente deben especificarse: la caja de test para no incluirla en este subset y la escala a la que deben estar las imágenes.
'''
def val_gen(modelo,directorio,K_val,escala=224):
    generadores = []
    for origen in ['iPhone','OCT','Samsung']:
    	for grado in ['EMD', 'NO EMD']:
        	generadores.append(generador_val(modelo,directorio,K_val,origen,grado,escala))
                
    generador_combinado = combine_gen(generadores)
    return generador_combinado
'''
transferLearning() entrena una red de transfer learning haciendo predicciones sobre la caja indicada como primer parámetro. Las otras cuatro cajas restantes se destinarán a train y val.
'''
def transfer_learning(directorio,K_test,red,batch,unlock):
    if (K_test == 'K1'):
        K_val = 'K5'
    elif (K_test == 'K2'):
        K_val = 'K1'
    elif (K_test == 'K3'):
        K_val = 'K2'
    elif (K_test == 'K4'):
        K_val = 'K3'
    elif (K_test == 'K5'):
        K_val = 'K4'
   
    #Creación de los generadores de datos de: entrenamiento, validación y test.
    train_generator = train_gen(red,directorio,K_test,K_val,escala = dic_escala[red])
    val_generator = val_gen(red,directorio,K_val,escala = dic_escala[red])
    
    #para el generador de tipo test debemos definir antes un nuevo ImageDataGenerator
    test_datagen = ImageDataGenerator(preprocessing_function=dic_preprocesado[red])
    
    #definimos un generador para Samsung, que coja únicamente las imágenes de Samsung de la caja de test
    test_Samsung = test_datagen.flow_from_directory(
        directory = directorio + '/' + K_test + '/Samsung/',
        target_size = (dic_escala[red],dic_escala[red]),
        color_mode = color,
        shuffle = False,
        class_mode='categorical',
        batch_size=1,
        seed = 42
    )
    
    #y lo mismo para iPhone, que solo coja las imágenes de iPhone de la caja de test
    test_iPhone = test_datagen.flow_from_directory(
        directory = directorio + '/' + K_test + '/iPhone/',
        target_size = (dic_escala[red],dic_escala[red]),
        color_mode = color,
        shuffle = False,
        class_mode='categorical',
        batch_size=1,
        seed = 42
    )
    
    #Definimos el modelo base de transfer-learning
    base_model = red(weights=None, include_top=False, input_shape=(dic_escala[red],dic_escala[red],3))
    base_model.load_weights('Pesos/' + str(red).split(' ')[1] + '.h5')
    if unlock == 'All':
        base_model.trainable = True ## Trainable weights
    else:
        base_model.trainable = False ## Not trainable weights
    
    #Definición últimas capas que se ajustarán al dominio de datos nuevo, fine tunning.
    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(1024, activation='relu')
    dense_layer_2 = layers.Dense(512, activation='relu')
    prediction_layer = layers.Dense(2, activation='softmax')
    
    #Unión de todas las capas creadas para el modelo.
    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    
    
    #Compilar el modelo e indicar ciertos hiperparámetros
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    es = EarlyStopping(monitor='val_loss', mode='min', patience=20,  restore_best_weights=True)

    #Entrenar el modelo
    history = model.fit(
        x = train_generator,
        batch_size=batch,
        epochs=200,
        steps_per_epoch = math.ceil(num(directorio,K_test,K_val)[0] / batch),
        callbacks=[es],
        validation_data = val_generator,
        validation_steps = math.ceil(num(directorio,K_test,K_val)[1] / batch)
    )
    
    #Parate del entrenamiento exclusiva del desbloqueo parcial de los pesos.
    if unlock == 'Parcial_all':
        base_model.trainable = True
        #es importante volver a compilar el modelo, para que estos cambios de descongelación sean tenidos en cuenta
        model.compile(optimizer=keras.optimizers.Adam(1e-5),  #Very low learning rate
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(
            x = train_generator,
            batch_size=batch,
            epochs=50,
            steps_per_epoch = math.ceil(num_train / batch),
            callbacks=[es],
            validation_data = val_generator,
            validation_steps = math.ceil(num_val / batch)
     
    #Cálculo de métricas de evaluación para las predicciones de los modelos ya entrenados comparando siempre con un baseline.   
    print('_________________________________________________________________________')
    print(f'MÉTRICAS DE EVALUACIÓN\n *CrossValidation\n *Balanced Generator\n *Red: {red}\n *K_test: {K_test}')
    print('_________________________________________________________________________')
    
    print('___________________________________________________________________________________')
    print('TEST: iPHONE')
    print('___________________________________________________________________________________')
       
    #Testeo y métrica para iPhone.     
    score_test_iphone = model.evaluate(x = test_iPhone, verbose = 0)
    print("Test loss:", score_test_iphone[0])
    print("Test accuracy:", score_test_iphone[1])
    
    y_test_iphone = test_iPhone.labels
    
    y_test_iphone_prob = model.predict(test_iPhone)
    y_test_iphone_pred = np.where(y_test_iphone_prob > 0.5, 1, 0)
            
    report_iphone = classification_report(y_test_iphone, y_test_iphone_pred, output_dict=True)

    matrix_iphone = confusion_matrix(y_test_iphone, y_test_iphone_pred)
      
    #Almacenamiento de las métricas en un dataframe para luego generar un csv visible por el programador.      
    df_cm = pd.DataFrame(matrix_iphone.reshape(1, -1), columns=['TN', 'FP', 'FN', 'TP'])
    df_cm.insert(loc=0, column='Modelo', value=modelo)
    df_cm.insert(loc=0, column='Dispositivo', value='iPhone')
    df_cm.insert(loc=0, column='Bloqueo', value=unlock)
    df_cm.insert(loc=0, column='Batch', value=batch)
    df_cm.insert(loc=0, column='Color', value=color)
    df_cm.insert(loc=1, column='Test', value=K_test)
    df_cm.insert(loc=2, column='Fecha', value=date_time)
    df_cm.insert(len(df_cm.columns), 'Accuracy', accuracy_score(y_test_iphone, y_test_iphone_pred))
    df_cm.insert(len(df_cm.columns), 'Precision', report_iphone['weighted avg']['precision'])
    df_cm.insert(len(df_cm.columns), 'Recall', report_iphone['weighted avg']['recall'])
    df_cm.insert(len(df_cm.columns), 'F1-score', report_iphone['weighted avg']['f1-score'])
    df_cm.insert(len(df_cm.columns), 'AUC', roc_auc_score(y_test_iphone, y_test_iphone_prob))
    
    #Calculo a parte de geometric means
    precision, recall, _, _ = precision_recall_fscore_support(y_test_iphone, y_test_iphone_pred, average=None)
    gm_per_class = [np.sqrt(precision[i] * recall[i]) for i in range(len(precision))]
    gm_total = np.prod(gm_per_class) ** (1/len(gm_per_class))
    df_cm.insert(len(df_cm.columns), 'g-means', gm_total)
    
    #Cálculo del tiempo conjunto de entrenamiento y ejecución del modelo probado.
    fin = time.time()
    df_cm.insert(len(df_cm.columns), 'time', (fin-inicio))

    #Guardar el fichero de datos generado en la carpeta "Resultados" y el csv correspondiente, si ya existe se escribirá debajo.
    fichero_resultado = r"resultados/Resultados.csv"
    if os.path.isfile(fichero_resultado):
        df_existing = pd.read_csv(fichero_resultado)
        df_combined = pd.concat([df_existing, df_cm])
        df_combined.to_csv(fichero_resultado, index=False, sep=',', mode='w+')
    else:
        df_cm.to_csv(fichero_resultado, index=False, sep=',', mode='w+')

    print(df_cm)

    print('___________________________________________________________________________________')
    print('TEST: Samsung')
    print('___________________________________________________________________________________')
    
    #Testeo y métricas para Samsung.
    score_test_samsung = model.evaluate(x = test_samsung, verbose = 0)
    print("Test loss:", score_test_samsung[0])
    print("Test accuracy:", score_test_samsung[1])
    
    y_test_samsung = test_samsung.labels
    
    y_test_samsung_prob = model.predict(test_samsung)
    y_test_samsung_pred = np.where(y_test_samsung_prob > 0.5, 1, 0)
            
    report_samsung = classification_report(y_test_samsung, y_test_samsung_pred, output_dict=True)

    matrix_samsung = confusion_matrix(y_test_samsung, y_test_samsung_pred)
            
    df_cm = pd.DataFrame(matrix_samsung.reshape(1, -1), columns=['TN', 'FP', 'FN', 'TP'])
    df_cm.insert(loc=0, column='Modelo', value=modelo)
    df_cm.insert(loc=0, column='Dispositivo', value='Samsung')
    df_cm.insert(loc=0, column='Bloqueo', value=unlock)
    df_cm.insert(loc=0, column='Batch', value=batch)
    df_cm.insert(loc=0, column='Color', value=color)
    df_cm.insert(loc=1, column='Test', value=K_test)
    df_cm.insert(loc=2, column='Fecha', value=date_time)
    df_cm.insert(len(df_cm.columns), 'Accuracy', accuracy_score(y_test_samsung, y_test_samsung_pred))
    df_cm.insert(len(df_cm.columns), 'Precision', report_samsung['weighted avg']['precision'])
    df_cm.insert(len(df_cm.columns), 'Recall', report_samsung['weighted avg']['recall'])
    df_cm.insert(len(df_cm.columns), 'F1-score', report_samsung['weighted avg']['f1-score'])
    df_cm.insert(len(df_cm.columns), 'AUC', roc_auc_score(y_test_samsung, y_test_samsung_prob))
    
    precision, recall, _, _ = precision_recall_fscore_support(y_test_iphone, y_test_iphone_pred, average=None)
    gm_per_class = [np.sqrt(precision[i] * recall[i]) for i in range(len(precision))]
    gm_total = np.prod(gm_per_class) ** (1/len(gm_per_class))
    df_cm.insert(len(df_cm.columns), 'g-means', gm_total)

    fin = time.time()
    df_cm.insert(len(df_cm.columns), 'time', (fin-inicio))
    
    fichero_resultado = r"resultados/Resultados.csv"
    if os.path.isfile(fichero_resultado):
        df_existing = pd.read_csv(fichero_resultado)
        df_combined = pd.concat([df_existing, df_cm])
        df_combined.to_csv(fichero_resultado, index=False, sep=',', mode='w+')
    else:
        df_cm.to_csv(fichero_resultado, index=False, sep=',', mode='w+')

    print(df_cm)

    #Guardar los pesos de los modelos entrenados para poder comenzar a hacer predicciones directamente en la carpeta "Modelos".
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    fichero_modelo = f"modelos/{modelo}_test{K_test}_{date_time}.h5"
    model.save(fichero_modelo)

transfer_learning(directorio,K_test,modelo,batch,unlock)

