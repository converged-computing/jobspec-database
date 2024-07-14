import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import time
import os
import dataUtilities as dt
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import IsolationForest


def get_model(shape, nn_1, nn_2, lr, drop):
    """
    Returns a Tensorflow model compiled with the input parameters
    :param shape: int sample dimensions
    :param nn_1: int number of neurons for layer 1
    :param nn_2: int number of neurons for layer 2
    :param lr: float learning rate
    :param drop: float dropout rate
    :return: Sequential
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=shape, name='Input'),
        tf.keras.layers.Dense(nn_1, activation=tf.keras.activations.relu, name='Dense1'),
        tf.keras.layers.Dropout(drop, name='Dropout'),
        tf.keras.layers.BatchNormalization(name='BatchNorm1'),
        tf.keras.layers.Dense(nn_2, activation=tf.keras.activations.relu, name='Dense2'),
        tf.keras.layers.BatchNormalization(name='BatchNorm2'),
        tf.keras.layers.Dense(2, activation='softmax', name='Classification')])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True, beta_1=0.5, beta_2=0.8),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tfa.metrics.F1Score(num_classes=2, average='macro'),
                           tf.keras.metrics.SpecificityAtSensitivity(0.5),
                           tf.keras.metrics.SensitivityAtSpecificity(0.5),
                           tf.keras.metrics.Precision()])
    return model


def loadHistory(pathToHistory):
    """
    Loads the complete history of a hypertuner to continue where the last tuning ended
    :param pathToHistory: string
    :return: ndarray hypertuner outputs
    """
    history = []
    for root, dirs, files in os.walk(pathToHistory):
        for file in files:
            hist = np.load(os.path.join(root, file))
            history.append(hist.reshape(-1, 5))
    history = np.concatenate(history, axis=0)
    return history


def hyperModel(hp):
    """
    Returns the Tensorflow model prepared to be used with the keras hypertuner
    :param hp: tensorflow hypermodel
    :return: Sequential
    """
    hp_units_first = hp.Int('units', min_value=2, max_value=128, step=2)
    hp_units_second = hp.Int('units', min_value=2, max_value=128, step=2)
    hp_drop = hp.Float('rate', min_value=0.0, max_value=0.5, step=0.05)
    hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(95,), name='Input'),
        tf.keras.layers.Dense(hp_units_first, activation='sigmoid', name='Dense1'),
        tf.keras.layers.Dropout(hp_drop, name='Dropout'),  # neu
        tf.keras.layers.BatchNormalization(name='BatchNorm1'),  # neu
        tf.keras.layers.Dense(hp_units_second, activation='sigmoid', name='Dense2'),
        tf.keras.layers.BatchNormalization(name='BatchNorm2'),  # neu
        tf.keras.layers.Dense(1, activation='sigmoid', name='Classification')])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=[tfa.metrics.F1Score(num_classes=1, average='macro'),
                           tf.keras.metrics.SpecificityAtSensitivity(0.5),
                           tf.keras.metrics.SensitivityAtSpecificity(0.5)])
    return model


def timeit(modelTriaging, Xtrain, yTrain, Xval, yVal):
    """
    Measures the time for 1 training epoch
    :param modelTriaging: build tensorflow model
    :param Xtrain: ndarray training samples
    :param yTrain: ndarray training labels
    :param Xval: ndarray validation samples
    :param yVal: ndarray validation labels
    :return: average computation time in s
    """
    start = time.time()
    for i in range(0, 100):
        modelTriaging.fit(Xtrain, yTrain, validation_data=(Xval, yVal), epochs=25, batch_size=500, shuffle=True)
    end = time.time()
    return np.divide(np.subtract(end, start), 25)


def loadData(pathToCrossValData):
    """
    Loads the data splitted into four cross validation sets
    :param pathToCrossValData: string
    :return: list of data using all features, list of data using features identified as flags only
    """
    dataFlags = []
    labelsFlags = []
    dataAll = []
    labelsAll = []
    valAllData = np.load(pathToCrossValData + 'validation_all_data.npy')
    valAllLabel = np.load(pathToCrossValData + 'validation_all_label.npy')
    valFlagsData = np.load(pathToCrossValData + 'validation_flags_data.npy')
    valFlagsLabel = np.load(pathToCrossValData + 'validation_flags_label.npy')
    for i in range(0, 3):
        currentAllDataPath = pathToCrossValData + 'fold_' + str(i) + '_all_data.npy'
        currentAllLabelPath = pathToCrossValData + 'fold_' + str(i) + '_all_label.npy'
        currentFlagsDataPath = pathToCrossValData + 'fold_' + str(i) + '_flags_data.npy'
        currentFlagsLabelPath = pathToCrossValData + 'fold_' + str(i) + '_flags_label.npy'
        dataFlags.append(np.load(currentFlagsDataPath))
        labelsFlags.append(np.load(currentFlagsLabelPath))
        dataAll.append(np.load(currentAllDataPath))
        labelsAll.append(np.load(currentAllLabelPath))
    return [dataAll, labelsAll, valAllData, valAllLabel], [dataFlags, labelsFlags, valFlagsData, valFlagsLabel]


def hypertuning():
    """
    Runs the hypertuning of an Neural Network
    :return:  None
    """
    pathToCrossData = './Data/crossval/'
    all, flags = loadData(pathToCrossData)
    datasets = [all, flags]
    pathForCallbacks = './Data/nn/cb'
    pathHistory = './Data/nn/history'
    layer1 = np.arange(2, 200, 5)  # 2 - 200 - 2
    layer2 = np.arange(2, 200, 5)
    drop = np.arange(0.25)
    max_steps = drop.size * layer1.size * layer2.size
    print("Max steps: " + str(max_steps))
    counter = 0
    best_f1 = 0
    best_combination = ""
    # hist = loadHistory(pathHistory) is used for pause and continue training
    for index, dset in enumerate(datasets):
        for i in range(0, 3):
            xTrain = dset[0][i]
            yTrain = tf.keras.utils.to_categorical(dset[1][i])
            xVal = dset[2]
            yVal = tf.keras.utils.to_categorical(dset[3])
            for neurons1 in layer1:
                history = []
                for neurons2 in layer2:
                    for droprate in drop:
                        tempHist = np.asarray([index, i, neurons1, neurons2, droprate])
                        found = False
                        # for row in hist:
                        #    if np.equal(row, tempHist).all():
                        #        found = True
                        #        break
                        if found:
                            counter = counter + 1
                            continue
                        if index == 0:
                            saveString = 'all_l1_' + str(neurons1) + '_l2_' + str(neurons2) + '_drop_' + str(
                                droprate) + '_fold_' + str(i)
                        else:
                            saveString = 'flags_l1_' + str(neurons1) + '_l2_' + str(neurons2) + '_drop_' + str(
                                droprate) + '_fold_' + str(i)
                        cp_callback = tf.keras.callbacks.ModelCheckpoint(
                            filepath=pathForCallbacks + '/nn_hp_' + saveString + '/checkpoints/{epoch}/',
                            save_weights_only=True,
                            verbose=1,
                            save_best_only=True,
                            monitor='val_f1_score',
                            mode='max')
                        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', min_delta=0.0001,
                                                                       patience=25, mode='max')
                        plt_callback = tf.keras.callbacks.CSVLogger(
                            pathForCallbacks + '/logs_oo/nn_hypertuning_' + saveString + '.log', append=True,
                            separator=';')
                        modelTriaging = get_model(xTrain.shape[1], neurons1, neurons2, 0.01, droprate)
                        result = modelTriaging.fit(xTrain, yTrain, validation_data=(xVal, yVal), epochs=200,
                                                   batch_size=500,
                                                   shuffle=True,
                                                   callbacks=[es_callback, plt_callback], verbose=0)
                        history.append([index, i, neurons1, neurons2, droprate])
                        f1 = result.history['f1_score'][-1]
                        if best_f1 < f1:
                            best_f1 = f1
                            best_combination = saveString
                        if counter % (max(1, int(max_steps / 100))) == 0:
                            print("Curr percent: " + str(counter * 100 / max_steps) + " (" + str(counter) + "/" + str(
                                max_steps) + ")")
                        counter = counter + 1
                # if history:
                # saveHist = np.concatenate(history, axis=0)
    print("Hypertuning finished")
    print("Best f1: " + str(best_f1))
    print("Save string: " + str(best_combination))
    print("Created entries: " + str(counter) + " (" + str(max_steps) + ")")
    return


def sensitivity(y_true, y_pred):
    """
       computes the sensitivity for each label and prints it
       :param y_true: ndarray true labels
       :param y_pred: ndarray predicted labels
       :return:
       """
    sens = []
    labels = np.unique(y_true)
    print('Sensitivity')
    for label in labels:
        print('Label: ' + str(label))
        tp_indices = np.where(y_true == label)[0]
        tp = np.sum(y_pred[tp_indices] == label)
        fn = np.sum(np.logical_not(y_pred[tp_indices] == label))
        sens.append(np.divide(tp, np.add(tp, fn)))
        print(np.divide(tp, np.add(tp, fn)))
    return sens


def specificity(y_true, y_pred):
    """
       computes the specificity  for each label and prints it
       :param y_true: ndarray true labels
       :param y_pred: ndarray predicted labels
       :return:
       """
    specs = []
    labels = np.unique(y_true)
    print('Specificity')
    for label in labels:
        print('Label: ' + str(label))
        tn_indices = np.where(y_true != label)[0]
        tn = np.sum(np.logical_not(y_pred[tn_indices] == label))
        fp = np.sum(y_pred[tn_indices] == label)
        specs.append(np.divide(tn, np.add(tn, fp)))
        print(np.divide(tn, np.add(tn, fp)))
    return


def f_score(y_true, y_pred):
    """
    computes the f score for each label and prints it
    :param y_true: ndarray true labels
    :param y_pred: ndarray predicted labels
    :return:
    """
    fscores = []
    labels = np.unique(y_true)
    print('F1-Score')
    for label in labels:
        print('Label: ' + str(label))
        tn_indices = np.where(y_true != label)[0]
        tp_indices = np.where(y_true == label)[0]
        tn = np.sum(np.logical_not(y_pred[tn_indices] == label))
        fp = np.sum(y_pred[tn_indices] == label)
        tp = np.sum(y_pred[tp_indices] == label)
        score = np.divide(tp, np.add(tp, np.multiply(0.5, np.add(fp, tp))))
        fscores.append(score)
        print(score)
    return


def computeMetrics(testset, trainedModel):
    """
    Computes and prints the specificty and sensitivity on the testset
    :param testset: ndarray of the test samples with the labels in the last column
    :param trainedModel: the trained sklearn classifier
    :return: None
    """
    pred = trainedModel.predict(testset[:, 0:-1])
    pred = np.argmax(pred, axis=1)
    specificity(testset[:, -1], pred)
    sensitivity(testset[:, -1], pred)
    f_score(testset[:, -1], pred)
    return


def onclasssvm():
    """
    Runs a one class svm for outlier detection and prints the confusion matrix
    :return: None
    """
    data = dt.loadReplacedSetCrossval()
    normal = data[data['Behandlungskategorie'] == 0]
    outlier = data[data['Behandlungskategorie'] == 1]
    normal = normal.drop(columns=['Behandlungskategorie'])
    outlier = outlier.drop(columns=['Behandlungskategorie'])
    clf = OneClassSVM(kernel='rbf', nu=np.divide(outlier.shape[0], normal.shape[0]), gamma='scale')
    clf.fit(normal[100:])
    pred = clf.predict(np.concatenate([normal[:100], outlier], axis=0))
    y_true = np.ones(100 + outlier.shape[0])
    y_true[outlier.shape[0]:] = -1
    conf_matrix = confusion_matrix(y_true, pred, normalize='true')
    print(conf_matrix)
    return


def isoforest():
    """
    Runs an isolation forest for outlier detection and prints the confusion matrix
    :return: None
    """
    data = dt.loadReplacedSetCrossval()
    normal = data[data['Behandlungskategorie'] == 0]
    outlier = data[data['Behandlungskategorie'] == 1]
    normal = normal.drop(columns=['Behandlungskategorie'])
    outlier = outlier.drop(columns=['Behandlungskategorie'])
    train_data = np.concatenate([normal[25:], outlier[:25]], axis=0)
    pca = PCA(0.99)
    train_data = pca.fit_transform(train_data)
    clf = IsolationForest(n_estimators=32, n_jobs=8)
    clf.fit(train_data)
    pred = clf.predict(pca.transform(np.concatenate([normal[:25], outlier[25:]], axis=0)))
    y_true = np.ones(25 + outlier.shape[0] - 25)
    y_true[outlier.shape[0] - 25:] = -1
    conf_matrix = confusion_matrix(y_true, pred, normalize='true')
    print(conf_matrix)
    return


def example():
    """
    Runs an example for neural networks
    :return:
    """
    hypertuning()
    return


example()
