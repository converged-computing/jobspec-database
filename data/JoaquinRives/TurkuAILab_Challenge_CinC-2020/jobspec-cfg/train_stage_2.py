"""
LSTM model (stage_2)
"""

from keras import Input, Model
import numpy as np
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, LSTM, Dropout, LeakyReLU, BatchNormalization, Masking, Bidirectional
from keras.models import load_model
from keras.optimizers import Adam
import tensorflow.keras as keras
# from logger import Logger
from metrics import MetricsCallback
from preprocess_and_segmentation import load_data, segment_all_dict_data, reshape_segmented_arrays
from preprocessor import preprocess_input_data
from utils import custom_multiclass_f1, split_train_validation_part_2, multiclass_f1, encode_labels
import shutil
import ntpath


# import sys


def build_model(n_timesteps, n_features, n_outputs):
    # model
    input = Input(shape=(n_timesteps, n_features), dtype='float32')  # TODO half float16 cuda
    x = Masking(mask_value=0.)(input)
    x = Bidirectional(LSTM(units=n_neurons, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Bidirectional(LSTM(units=n_neurons))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    output = Dense(n_outputs, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)
    # opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

    return model


if __name__ == '__main__':

    # Config
    experiment_name = "stage_2_exp_training_with_only_known_new_data"
    experiments_dir = "experiments/stage_2/"

    stage_1_exp = "local-known"

    data_dir_1 = f"experiments/stage_1/model_1_1_{stage_1_exp}"
    data_dir_2 = f"experiments/stage_1/model_1_2_{stage_1_exp}"
    data_dir_3 = f"experiments/stage_1/model_1_3_{stage_1_exp}"
    data_dir_4 = f"experiments/stage_1/model_1_4_{stage_1_exp}"

    data_test_dir = 'data/test_balanced'

    labels_file = 'subject_labels.npy'
    predictions_file = 'subject_predictions.npy'
    scores_file = 'scores_custom.npy'

    segment_size = 3000
    overlap = 0.5
    epochs = 30
    batch_size = 18
    n_timesteps = 120
    n_features = 36
    n_outputs = 9
    n_neurons = 85

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # create directory for the experiment
    if not os.path.exists(os.path.join(experiments_dir, experiment_name, 'best_models')):
        os.makedirs(os.path.join(experiments_dir, experiment_name, 'best_models'))
    # else:
    #     raise NameError(f"Already exist an experiment with the name '{experiment_name}'"
    #                     f" in the '{experiments_dir}' directory.")

    # save a copy of the script
    shutil.copy(__file__, os.path.join(experiments_dir, experiment_name, ntpath.basename(__file__)))

    # # Log stdout
    # log_file = os.path.join(experiments_dir, experiment_name, 'logfile.log')
    # sys.stdout = Logger(log_file)

    subject_labels = np.load(os.path.join(data_dir_1, labels_file), allow_pickle=True).item()
    print("xxxxxxxxx", len(subject_labels))
    subject_predictions = np.load(os.path.join(data_dir_1, predictions_file), allow_pickle=True).item()
    print("xxxxxxxxx", len(subject_predictions))

    scores_1 = np.load(os.path.join(data_dir_1, scores_file), allow_pickle=True).item()

    # subject_labels_2 = np.load(os.path.join(data_dir_2, labels_file), allow_pickle=True).item()  # TODO
    subject_predictions_2 = np.load(os.path.join(data_dir_2, predictions_file), allow_pickle=True).item()
    scores_2 = np.load(os.path.join(data_dir_2, scores_file), allow_pickle=True).item()

    # subject_labels_3 = np.load(os.path.join(data_dir_1, labels_file), allow_pickle=True).item()  # TODO
    subject_predictions_3 = np.load(os.path.join(data_dir_3, predictions_file), allow_pickle=True).item()
    scores_3 = np.load(os.path.join(data_dir_3, scores_file), allow_pickle=True).item()

    # subject_labels_4 = np.load(os.path.join(data_dir_1, labels_file), allow_pickle=True).item()  # TODO
    subject_predictions_4 = np.load(os.path.join(data_dir_4, predictions_file), allow_pickle=True).item()
    scores_4 = np.load(os.path.join(data_dir_4, scores_file), allow_pickle=True).item()

    # pad inputs
    subject_predictions_padded = {k: np.zeros((n_timesteps, 9)) for k in subject_predictions.keys()}
    for k, v in subject_predictions.items():
        subject_predictions_padded[k][-len(v):, :] = v

    subject_predictions_padded_2 = {k: np.zeros((n_timesteps, 9)) for k in subject_predictions_2.keys()}
    for k, v in subject_predictions_2.items():
        subject_predictions_padded_2[k][-len(v):, :] = v

    subject_predictions_padded_3 = {k: np.zeros((n_timesteps, 9)) for k in subject_predictions_3.keys()}
    for k, v in subject_predictions_3.items():
        subject_predictions_padded_3[k][-len(v):, :] = v

    subject_predictions_padded_4 = {k: np.zeros((n_timesteps, 9)) for k in subject_predictions_4.keys()}
    for k, v in subject_predictions_4.items():
        subject_predictions_padded_4[k][-len(v):, :] = v

    # concatenate predictions of stages 1
    for k, v in subject_predictions.items():
        subject_predictions_padded[k] = np.concatenate([subject_predictions_padded[k],
                                                        subject_predictions_padded_2[k],
                                                        subject_predictions_padded_3[k],
                                                        subject_predictions_padded_4[k]], axis=1)

    ###################################################################################################################
    ## Stage 1 on test

    # Get the best stage_1 models
    best_fold_1 = [(k, v) for k, v in sorted(scores_1.items(), key=lambda item: item[1], reverse=True)][0][0]
    best_fold_2 = [(k, v) for k, v in sorted(scores_2.items(), key=lambda item: item[1], reverse=True)][0][0]
    best_fold_3 = [(k, v) for k, v in sorted(scores_3.items(), key=lambda item: item[1], reverse=True)][0][0]
    best_fold_4 = [(k, v) for k, v in sorted(scores_4.items(), key=lambda item: item[1], reverse=True)][0][0]

    # Models stage 1
    model_stage_1_1 = keras.models.load_model(os.path.join(data_dir_1, f"model_{best_fold_1}.h5"))
    model_stage_1_2 = keras.models.load_model(os.path.join(data_dir_2, f"model_{best_fold_2}.h5"))
    model_stage_1_3 = keras.models.load_model(os.path.join(data_dir_3, f"model_{best_fold_3}.h5"))
    model_stage_1_4 = keras.models.load_model(os.path.join(data_dir_4, f"model_{best_fold_4}.h5"))

    # save best models apart
    model_stage_1_1.save(os.path.join(experiments_dir, experiment_name, 'best_models', f"stage_1_1_{best_fold_1}.h5"))
    model_stage_1_2.save(os.path.join(experiments_dir, experiment_name, 'best_models', f"stage_1_2_{best_fold_2}.h5"))
    model_stage_1_3.save(os.path.join(experiments_dir, experiment_name, 'best_models', f"stage_1_3_{best_fold_3}.h5"))
    model_stage_1_4.save(os.path.join(experiments_dir, experiment_name, 'best_models', f"stage_1_4_{best_fold_4}.h5"))

    # Load test data
    data_test = load_data(data_test_dir)
    data_test = preprocess_input_data(data_test)
    data_test = segment_all_dict_data(data_test, segment_size, overlap)
    arr_of_segments, arr_of_labels, arr_of_IDs = reshape_segmented_arrays(data_test,
                                                                          shuffle_IDs=False,
                                                                          # Do not shuffle the segments to keep the
                                                                          # order in time of the predictions
                                                                          shuffle_segments=False,
                                                                          segment_standardization_flag=True)
    # Encode labels
    arr_of_labels = np.array([i[0]['Dx'] for i in arr_of_labels])
    arr_of_labels = encode_labels(arr_of_labels)

    # Predictions stages 1
    predictions_s1_1 = model_stage_1_1.predict(arr_of_segments, verbose=1)
    predictions_s1_2 = model_stage_1_2.predict(arr_of_segments, verbose=1)
    predictions_s1_3 = model_stage_1_3.predict(arr_of_segments, verbose=1)
    predictions_s1_4 = model_stage_1_4.predict(arr_of_segments, verbose=1)

    # Score stages 1
    f1_score_s1_1, f1_score_p1_list_1 = multiclass_f1(arr_of_labels, predictions_s1_1, return_list=True)
    f1_score_custom_s1_1, f1_score_custom_s1_list_1 = custom_multiclass_f1(arr_of_labels, predictions_s1_1,
                                                                           return_list=True)
    print("\nStage 1 1 f1-score: ", f1_score_s1_1)
    print(f1_score_p1_list_1)
    print("\nStage 1 1 f1-score (custom):", f1_score_custom_s1_1)
    print(f1_score_custom_s1_list_1, "\n\n")

    f1_score_s1_2, f1_score_p1_list_2 = multiclass_f1(arr_of_labels, predictions_s1_2, return_list=True)
    f1_score_custom_s1_2, f1_score_custom_s1_list_2 = custom_multiclass_f1(arr_of_labels, predictions_s1_2,
                                                                           return_list=True)
    print("\nStage 1 2 f1-score: ", f1_score_s1_2)
    print(f1_score_p1_list_2)
    print("\nStage 1 2 f1-score (custom):", f1_score_custom_s1_2)
    print(f1_score_custom_s1_list_2, "\n\n")

    f1_score_s1_3, f1_score_p1_list_3 = multiclass_f1(arr_of_labels, predictions_s1_3, return_list=True)
    f1_score_custom_s1_3, f1_score_custom_s1_list_3 = custom_multiclass_f1(arr_of_labels, predictions_s1_3,
                                                                           return_list=True)
    print("\nStage 1 3 f1-score: ", f1_score_s1_3)
    print(f1_score_p1_list_3)
    print("\nStage 1 3 f1-score (custom):", f1_score_custom_s1_3)
    print(f1_score_custom_s1_list_3, "\n\n")

    f1_score_s1_4, f1_score_p1_list_4 = multiclass_f1(arr_of_labels, predictions_s1_4, return_list=True)
    f1_score_custom_s1_4, f1_score_custom_s1_list_4 = custom_multiclass_f1(arr_of_labels, predictions_s1_4,
                                                                           return_list=True)

    print("\nStage 1 4 f1-score: ", f1_score_s1_4)
    print(f1_score_p1_list_4)
    print("\nStage 1 4 f1-score (custom):", f1_score_custom_s1_4)
    print(f1_score_custom_s1_list_4, "\n\n")

    # concatenate predictions of stages 1
    predictions_stages_1 = np.concatenate([predictions_s1_1, predictions_s1_2, predictions_s1_3, predictions_s1_4],
                                          axis=-1)

    # Group by subject & padding:

    # true labels of each subject
    subject_labels_test = {ID: None for ID in list(np.unique(arr_of_IDs))}
    for ID, label in zip(arr_of_IDs, arr_of_labels):
        subject_labels_test[ID[0]] = label

    # stages 1 predictions for each subject
    subject_predictions_test = {ID: [] for ID in list(np.unique(arr_of_IDs))}
    for ID, pred in zip(arr_of_IDs, predictions_stages_1):
        subject_predictions_test[ID[0]].append(pred)

    # pad inputs
    subject_predictions_padded_test = {k: np.zeros((n_timesteps, n_features)) for k in subject_predictions_test.keys()}
    for k, v in subject_predictions_test.items():
        subject_predictions_padded_test[k][-len(v):, :] = v

    # convert to array
    X_val, y_val, _, _ = split_train_validation_part_2(subject_predictions_padded_test, subject_labels_test,
                                                       n_variables=n_features, split=0)

    ## end stage 1 on test
    ###################################################################################################################
    # Stage 2

    # convert to array
    X_train, y_train, _, _ = split_train_validation_part_2(subject_predictions_padded, subject_labels,
                                                           n_variables=n_features, split=0)

    # Model
    model_stage_2 = build_model(n_timesteps, n_features, n_outputs)

    # callbacks
    earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', patience=16, verbose=0, mode='max')
    mcp_save = ModelCheckpoint(os.path.join(experiments_dir, experiment_name, f"model_stage_2.h5"),
                               save_best_only=True, monitor='val_categorical_accuracy', mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, patience=10, verbose=1,
                                       epsilon=1e-4,
                                       mode='max')

    # to show the f1 and fbeta during training
    metrics_callback = MetricsCallback(validation_data=(X_val, y_val))

    # train stage 2
    model_stage_2.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True,
                      validation_data=(X_val, y_val),
                      callbacks=[earlyStopping, mcp_save, reduce_lr_loss, metrics_callback])

    # reloading the best model
    del model_stage_2
    model_stage_2 = load_model(os.path.join(experiments_dir, experiment_name, f"model_stage_2.h5"))

    # final predictions
    _, accuracy = model_stage_2.evaluate(X_val, y_val, verbose=1)
    final_predictions = model_stage_2.predict(X_val, verbose=1)

    # save test predictions & labels
    np.save(os.path.join(experiments_dir, experiment_name, "labels.npy"), y_val)
    np.save(os.path.join(experiments_dir, experiment_name, "final_predictions.npy"), final_predictions)

    print(f"\nAccuracy: {accuracy}")

    score, score_list = multiclass_f1(y_val, final_predictions, return_list=True)
    print(f"\nf1-score: {score}")
    print(score_list)

    # f1-score
    score_custom, score_custom_list = custom_multiclass_f1(y_val, final_predictions, return_list=True)
    print(f"\nf1-score (custom): {score_custom}")
    print(score_custom_list)

    # save f1-score
    with open(os.path.join(experiments_dir, experiment_name, "score_stage_2.txt"), 'w') as f:
        f.write(f"f1-score: {str(score)} \n f1-score (custom): {str(score_custom)}")

    # Save info and results test
    with open(os.path.join(experiments_dir, experiment_name, "info.txt"), 'w') as f:
        f.write(f"Model stage 1 1: {data_dir_1}\n")
        f.write(f"Model stage 2 2: {data_dir_2}\n")
        f.write(f"Model stage 2 3: {data_dir_3}\n")
        f.write(f"Model stage 2 4: {data_dir_4}\n")

        f.write(f"\n\nStage 1 1 f1-score: {str(f1_score_s1_1)}\n")
        f.write(str(f1_score_p1_list_1))
        f.write(f"\n\nStage 1 1 f1-score (custom): {str(f1_score_custom_s1_1)}\n")
        f.write(str(f1_score_custom_s1_list_1))

        f.write(f"\n\nStage 1 2 f1-score: {str(f1_score_s1_2)}\n")
        f.write(str(f1_score_p1_list_2))
        f.write(f"\n\nStage 1 2 f1-score (custom): {str(f1_score_custom_s1_2)}\n")
        f.write(str(f1_score_custom_s1_list_2))

        f.write(f"\n\nStage 1 3 f1-score: {str(f1_score_s1_3)}\n")
        f.write(str(f1_score_p1_list_3))
        f.write(f"\n\nStage 1 3 f1-score (custom): {str(f1_score_custom_s1_3)}\n")
        f.write(str(f1_score_custom_s1_list_3))

        f.write(f"\n\nStage 1 4 f1-score: {str(f1_score_s1_4)}\n")
        f.write(str(f1_score_p1_list_4))
        f.write(f"\n\nStage 1 4 f1-score (custom): {str(f1_score_custom_s1_4)}\n")
        f.write(str(f1_score_custom_s1_list_4))

        f.write(f"\n\nStage 2 f1-score: {str(score)}\n")
        f.write(str(score_list))
        f.write(f"\n\nStage 2 f1-score (custom): {str(score_custom)}\n")
        f.write(str(score_custom_list))

    # from sklearn.metrics import multilabel_confusion_matrix
    #
    #
    # pred = np.where(predictions > 0.5, 1, 0)
    # true = y_validation.copy()
    #
    # confusion = multilabel_confusion_matrix(true, pred)
