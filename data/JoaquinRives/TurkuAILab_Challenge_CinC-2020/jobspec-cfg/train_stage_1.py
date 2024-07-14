"""
Training of any of the models of the stage 1
"""

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold
import os

from utils import custom_multiclass_f1, multiclass_f1
import shutil
import ntpath
import numpy as np
from preprocess_and_segmentation import load_data, segment_all_dict_data, reshape_segmented_arrays
from preprocessor import preprocess_input_data
from utils import encode_labels
import tensorflow.keras as keras
import pandas as pd
import sys
import matplotlib.pyplot as plt
# from tensorflow_addons.metrics import FBetaScore
from metrics import MetricsCallback
from data_generator import DataGenerator

from models.model_1 import stage_1_1
from models.model_2 import ResNet1D
from models.model_architecture_V01 import Inc_ResNet_LSTM_v01, Inc_ResNet_LSTM_v02


class SWA(keras.callbacks.Callback):
    """ Stochastic Weight Averaging """

    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch

    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))

    def on_epoch_end(self, epoch, logs=None):

        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()

        elif epoch > self.swa_epoch:
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] *
                                       (epoch - self.swa_epoch) + self.model.get_weights()[i]) / (
                                              (epoch - self.swa_epoch) + 1)
        else:
            pass

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')


def cross_validation(arr_of_segments, arr_of_labels, arr_of_IDs, ids_labels):
    """ Subject cross-validation """

    global epochs, batch_size, n_folds, exp_path, aug_prob, swa_epochs

    # split the subjects on n folds keeping balance
    ids = ids_labels['subject']
    skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=True)
    subj_folds = [(ids[test_index]) for train_index, test_index in skf.split(ids_labels['subject'],
                                                                             ids_labels['label']
                                                                             )]

    # true labels of each subject
    subject_labels = {ID: None for ID in list(ids)}
    for ID, label in zip(arr_of_IDs, arr_of_labels):
        subject_labels[ID[0]] = label

    # to save the predictions of each subject
    subject_predictions = {ID: [] for ID in list(ids)}

    # to save the f1-score of each fold
    scores = {}
    scores_custom = {}

    for i, validation_fold in enumerate(subj_folds):
        print(f"\n\nFold {i} ------------------------------------------------- \n")

        # selector
        selector = np.isin(arr_of_IDs.squeeze(), validation_fold)

        # validation
        arr_seg_validation = arr_of_segments[selector]
        arr_labels_validation = arr_of_labels[selector]
        arr_IDs_validation = arr_of_IDs[selector]

        # train
        arr_seg_train = arr_of_segments[np.invert(selector)]
        arr_labels_train = arr_of_labels[np.invert(selector)]
        arr_IDs_train = arr_of_IDs[np.invert(selector)]

        # TODO
        # Up-balance 'STE' (3x)
        add_to_input = []
        add_to_labels = []
        add_to_IDs = []
        for j in range(len(arr_labels_train)):
            if arr_labels_train[j][8] == 1:
                add_to_input.append(arr_seg_train[j])
                add_to_labels.append(arr_labels_train[j])
                add_to_IDs.append(arr_IDs_train[j])

        arr_seg_train_balanced = np.concatenate([add_to_input, arr_seg_train, add_to_input])
        arr_labels_train_balanced = np.concatenate([add_to_labels, arr_labels_train, add_to_labels])
        arr_IDs_train_balanced = np.concatenate([add_to_IDs, arr_IDs_train, add_to_IDs])

        dg_train = DataGenerator(arr_seg_train_balanced, arr_labels_train_balanced, batch_size=batch_size, n_classes=9,
                                 augmentation=use_aug, aug_prob=aug_prob)

        # callbacks

        # to show the f1 and fbeta during training
        metrics_callback = MetricsCallback(validation_data=(arr_seg_validation, arr_labels_validation))

        #  NOT use early stopping if SWA
        # earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', patience=10, verbose=0, mode='max')
        mcp_save = ModelCheckpoint(os.path.join(exp_path, f"model_fold_{i}.h5"),
                                   save_best_only=True, monitor='val_categorical_accuracy', mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, patience=7, verbose=1,
                                           epsilon=1e-4,
                                           mode='max')
        ################################################################################################################
        # Model

        if model_to_train == 'model_1_1':
            # model 1
            model = stage_1_1(segment_size, n_channels, 9)
            # TODO settings = # load lr, momentum...
        elif model_to_train == 'model_1_2':
            # model 2
            model = ResNet1D(input_shape=(segment_size, n_channels), classes=9)
        elif model_to_train == 'model_1_3':
            # model 3
            model = Inc_ResNet_LSTM_v01(segment_size, n_channels, classes=9)
        elif model_to_train == 'model_1_4':
            # model 4
            model = Inc_ResNet_LSTM_v02(segment_size, n_channels, classes=9)
        else:
            raise ValueError(f"Model {model_to_train} not known.")

        # opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy']) # , fbeta_score])

        ################################################################################################################
        # stocastic weight averaging of the last n epochs
        swa = SWA(os.path.join(exp_path, f"model_fold_{i}_weights_SWA.h5"), epochs - swa_epochs)

        # history = model.fit(arr_seg_train_balanced, arr_labels_train_balanced, epochs=epochs, batch_size=batch_size,
        #                     verbose=2, validation_data=(arr_seg_validation, arr_labels_validation), shuffle=True,
        #                     callbacks=[mcp_save, reduce_lr, swa, metrics_callback])

        history = model.fit_generator(dg_train, epochs=epochs, verbose=2,
                                      validation_data=(arr_seg_validation, arr_labels_validation), shuffle=True,
                                      callbacks=[mcp_save, reduce_lr, swa, metrics_callback])

        # TODO: load the best model from the ModelCheckpoint callback and compare to the SWA model created from the
        #  the last n epochs:
        # del model
        # model = load_model(os.path.join(exp_path, f"model_fold_{i}.h5"))

        # re-load best model weights
        model.load_weights(os.path.join(exp_path, f"model_fold_{i}_weights_SWA.h5"))

        results = model.evaluate(arr_seg_validation, arr_labels_validation, batch_size=batch_size, verbose=1)
        predictions = model.predict(arr_seg_validation, verbose=2)

        # print fold results
        print("Validation (loss | acc):", results)

        f1_score, f1_score_list = multiclass_f1(arr_labels_validation, predictions, return_list=True)
        print("\nf1 score:", f1_score)
        print(f1_score_list)

        f1_score_custom, f1_score_custom_list = custom_multiclass_f1(arr_labels_validation, predictions,
                                                                     return_list=True)
        print("\nf1 score (custom):", f1_score_custom)
        print(f1_score_custom_list)

        # save predictions
        for ID, pred in zip(arr_IDs_validation, predictions):
            subject_predictions[ID[0]].append(pred)

        # save f1-score
        scores[f"fold_{i}"] = f1_score
        scores_custom[f"fold_{i}"] = f1_score_custom

        # save f1-score list (text file):
        with open(os.path.join(exp_path, "scores.txt"), 'a') as f:
            f.write(f"Fold {str(i)}:\n"
                    f"{str(f1_score_list)} (f1-score by class) \n"
                    f"{str(f1_score_custom_list)} (f1 score (custom) by class) \n")

    # Average f-1 score
    m, s = np.mean([v for v in scores.values()]), np.std([v for v in scores.values()])
    m_c, s_c = np.mean([v for v in scores_custom.values()]), np.std([v for v in scores_custom.values()])

    # save labels (to disk)
    np.save(os.path.join(exp_path, "subject_labels.npy"), subject_labels)

    # save predictions (to disk)
    np.save(os.path.join(exp_path, "subject_predictions.npy"), subject_predictions)

    # save f1-scores (to disk)
    np.save(os.path.join(exp_path, "scores.npy"), scores)
    np.save(os.path.join(exp_path, "scores_custom.npy"), scores_custom)

    print("\n==========================================================\n")
    print(f"CV f1-score: {str(m)} (+/- {str(s)}) \nCV f1-score (custom): {str(m_c)} (+/- {str(s_c)})")

    # save f1-scores (text file)
    with open(os.path.join(exp_path, "scores.txt"), 'a') as f:
        f.write("\n\n ==> Score by CV:")
        f.write(f"\n{str(scores)} (f1-score) \n{str(scores_custom)} (f1-score (custom))")
        f.write("\n\n ==> Average score CV:")
        f.write(f"\nCV f1-score: {str(m)} (+/- {str(s)}) \nCV f1-score (custom): {str(m_c)} (+/- {str(s_c)})\n\n")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(exp_path, f"{model_to_train}_loss_plot.png"))


if __name__ == '__main__':

    # Config
    model_to_train = sys.argv[1]  # slurm job array !!!
    # model_to_train = 'model_1_1'
    experiment_name = "exp002"
    experiments_dir = "experiments/stage_1"
    data_dir = 'data/train_balanced'
    epochs = 30
    swa_epochs = 4
    n_folds = 10
    segment_size = 3000
    overlap = 0.5
    batch_size = 64
    freq = 500  # Hz
    ch_spectrum = []
    max_freq_spectrum = 175
    n_channels = 12 + len(ch_spectrum)
    use_aug = True
    aug_prob = 0.2
    exp_path = os.path.join(experiments_dir, f"{model_to_train}_{experiment_name}")

    # create directory for the experiment
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    # else:
    #     raise NameError(f"Already exist an experiment with the name '{experiment_name}'"
    #                     f" in the '{experiments_dir}' directory.")

    # save a copy of the script
    shutil.copy(__file__, os.path.join(exp_path, ntpath.basename(__file__)))

    # load data
    print('loading data...')
    data = load_data(data_dir)

    # create array with the label of each subject (it is used to keep the balance of the labels
    # in the folds of the cross-validation

    dic_labels = {}
    for k, v in data.items():
        print(k)
        dic_labels[k] = data[k]['info']['Dx']

    ids_labels = pd.Series(dic_labels).reset_index()
    ids_labels.columns = ['subject', 'label']

    # pre-process signals
    print('processing data...')
    data = preprocess_input_data(data)

    # segment signal
    data = segment_all_dict_data(data, segment_size, overlap)

    arr_of_segments, arr_of_labels, arr_of_IDs = reshape_segmented_arrays(data,
                                                                          shuffle_IDs=True,
                                                                          # Do not shuffle the segments to keep the
                                                                          # order in time of the predictions
                                                                          shuffle_segments=False,
                                                                          segment_standardization_flag=True)
    # power spectrum
    # arr_of_segments = add_spectral(arr_of_segments, segment_size=segment_size, channels=ch_spectrum,
    #                                hz=freq, max_freq=max_freq_spectrum)

    # Encode labels
    arr_of_labels = np.array([i[0]['Dx'] for i in arr_of_labels])
    arr_of_labels = encode_labels(arr_of_labels)

    # Cross-validation
    cross_validation(arr_of_segments, arr_of_labels, arr_of_IDs, ids_labels)
