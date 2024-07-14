import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
from datetime import datetime
import os
import random
import sys
from zipfile import ZipFile
import zipfile

import data_transforms
from tcn import TCN

from train_sklearn_models import classifier_train

from classification_methods import (
    random_forest_classifier,
    knn_classifier,
    logistic_regression,
    sgd_classifier,
    ridge_classifier,
    svm_classifier,
    gaussian_nb_classifier,
    # xgboost_classifier,
)

from sklearn.model_selection import ParameterSampler
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

import tensorflow as tf
from tensorflow import keras
import tensorboard
from tensorflow.keras.models import model_from_json

# stop warnings from sklearn
# https://stackoverflow.com/questions/32612180/eliminating-warnings-from-scikit-learn
# https://stackoverflow.com/a/15778297
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.simplefilter(action='ignore', category=FutureWarning)


print("TensorFlow version: ", tf.__version__)
print("Keras version: ", keras.__version__)
print("Tensorboard version:", tensorboard.__version__)

#### setup the zip files and extract them
# location of the zip folders the tensorflow encoder models

#!#!# INPUT FOLDER NAME HERE THAT CONTAINS ZIP FILES OF ENCODERS
folder_to_get_data = "2020.04.22_results_1_zip"

zip_path = (
    Path("/home/tvhahn/projects/def-mechefsk/tvhahn/milling_data/results")
    / folder_to_get_data
)

# # zip path if run on local computer
# zip_path = (
#     Path("/home/tim/Documents/Milling-Files")
#     / folder_to_get_data
# )

#### setup the location where the zip files will be extracted to on the scratch
# folder location will be created if does not already exist
Path("/home/tvhahn/scratch/interim_data_encoder").mkdir(parents=True, exist_ok=True)
scratch_path = Path("/home/tvhahn/scratch/interim_data_encoder")

# Path("/home/tim/Documents/Milling-Files/_temp").mkdir(parents=True, exist_ok=True)
# scratch_path = Path("/home/tim/Documents/Milling-Files/_temp")

file_name = sys.argv[1]
file_folder_index = file_name.split(sep=".")[0]

# extract zip file
with ZipFile(zip_path / file_name, "r") as zip_file:
    # setup the location where the split cut data will be stored.
    # folder location will be created if does not already exist
    zip_file.extractall(path=(scratch_path / file_folder_index))

# location of all the split signals (these are all the pickles that were created in the create_split_data.py)
saved_model_dir = scratch_path / file_folder_index


# processed_data = Path('/home/tim/Documents/milling/data/processed/scale_0_to_1_mashed')
processed_data = Path(
    "/home/tvhahn/projects/def-mechefsk/tvhahn/milling_data/processed/scale_0_to_1_mashed"
)

date_time = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")

(
    X_train,
    y_train,
    X_train_slim,
    y_train_slim,
    X_val,
    y_val,
    X_val_slim,
    y_val_slim,
    X_test,
    y_test,
) = data_transforms.load_train_test(processed_data)

# rename the classes so it is a binary problem
class_to_remove = np.array([2], dtype="uint8")

new_y_val = []
for i in y_val:
    if i in class_to_remove:
        new_y_val.append(1)
    else:
        new_y_val.append(-1)
new_y_val = np.array(new_y_val)

new_y_train = []
for i in y_train:
    if i in class_to_remove:
        new_y_train.append(1)
    else:
        new_y_train.append(-1)
new_y_train = np.array(new_y_train)

new_y_test = []
for i in y_test:
    if i in class_to_remove:
        new_y_test.append(1)
    else:
        new_y_test.append(-1)
new_y_test = np.array(new_y_test)

# for keras models
K = keras.backend


class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean


# rounded accuracy for the metric
def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


# sweep through models
df_all = pd.DataFrame()
counter = 1
for folder_name in os.listdir(saved_model_dir):
    if "encoder" in folder_name:
        date_model_ran = folder_name.split("_")[0]
        print(date_model_ran, counter)

        loaded_json = open(
            r"{}/{}/model.json".format(saved_model_dir, folder_name), "r"
        ).read()
        encoder = model_from_json(
            loaded_json, custom_objects={"TCN": TCN, "Sampling": Sampling}
        )

        print('Beginning encoder output for {}....'.format(file_name))
        _, _, bvae_latent_train = encoder.predict(X_train, batch_size=64)
        _, _, bvae_latent_val = encoder.predict(X_val, batch_size=64)
        print('bvae_latent_train max:', np.max(bvae_latent_train))
        print('bvae_latent_train has nan:', np.isnan(bvae_latent_train))
        print('bvae_latent_val max:', np.max(bvae_latent_val))
        print('bvae_latent_val has nan:', np.isnan(bvae_latent_val))


        no_iterations = 40
        # sampler_seed = random.randint(0, 2 ** 16)
        sampler_seed = 11
        no_k_folds = 3

        # list of classifiers to test
        classifier_list_all = [
            random_forest_classifier,
            knn_classifier,
            # logistic_regression,
            # sgd_classifier,
            # ridge_classifier,
            # svm_classifier,
            # gaussian_nb_classifier,
            # xgboost_classifier,
        ]

        imbalance_ratios = [1]

        over_under_sampling_methods = [
            # "random_over",
            # "random_under",
            # "random_under_bootstrap",
            # "smote",
            # "adasyn",
            None,
        ]

        parameters_sample_dict = {
            "parameter_sampler_random_int": sp_randint(0, 2 ** 16),
            # "parameter_sampler_random_int": [16],
            "classifier_used": classifier_list_all,
            "uo_method": over_under_sampling_methods,
            "imbalance_ratio": imbalance_ratios,
        }

        # generate the list of parameters to sample over
        p_list = list(
            ParameterSampler(
                parameters_sample_dict, n_iter=no_iterations, random_state=sampler_seed
            )
        )

        #############################################################################
        # run models with each of the parameters

        for k, p in enumerate(p_list):
            print(p)

            # set random.seed
            random.seed(p["parameter_sampler_random_int"])

            # get specific parameters
            clf_name = str(p["classifier_used"]).split(" ")[1]

            parameter_sampler_random_int = p["parameter_sampler_random_int"]
            clf_function = p["classifier_used"]
            uo_method = p["uo_method"]
            imbalance_ratio = p["imbalance_ratio"]

            # build dictionary to store parameter results and other info
            parameter_values = {
                "clf_name": clf_name,
                "uo_method": uo_method,
                "imbalance_ratio": imbalance_ratio,
                "parameter_sampler_seed": parameter_sampler_random_int,
                "initial_script_seed": sampler_seed,
            }

            # save the general parameters values
            df_gpam = pd.DataFrame.from_dict(parameter_values, orient="index").T

            # instantiate the model
            clf, classifier_parameters = clf_function(parameter_sampler_random_int)

            # save classifier parameters into dataframe
            df_cpam = pd.DataFrame.from_dict(classifier_parameters, orient="index").T

            result_dict, final_clf = classifier_train(
                bvae_latent_train,
                new_y_train,
                bvae_latent_val,
                new_y_val,
                clf,
                k_fold_no=no_k_folds,
                print_results=False,
                train_on_all=True,
            )

            df_result_dict = pd.DataFrame.from_dict(result_dict, orient="index").T
            # df_result_dict.astype("float16").dtypes

            if k == 0:
                df_results = pd.concat([df_gpam, df_cpam, df_result_dict], axis=1)
                df_results["model_date"] = date_model_ran
            else:
                df_results = df_results.append(
                    pd.concat([df_gpam, df_cpam, df_result_dict], axis=1)
                )
                df_results["model_date"] = date_model_ran
        df_all = df_all.append(df_results)

        save_folder_name = 'temp_results_{}'.format(folder_to_get_data)
        # root_folder = Path('/home/tim/Documents/Milling-Files')
        root_folder = Path('/home/tvhahn/Milling-Files')

        Path(root_folder / save_folder_name).mkdir(parents=True, exist_ok=True)

        df_all.to_csv(root_folder / save_folder_name / "interim_encoder_results_{}_{}.csv".format(file_folder_index,date_time))
        counter += 1
