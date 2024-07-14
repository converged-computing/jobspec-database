"""
Model training
"""
import matplotlib

matplotlib.use('Agg')  # necessary in order to be able to produce plots on grid

import os
import argparse
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from datetime import datetime

# suppress useless warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# import other important procedures and utilities
import network.utils.data_generator_base
from network.utils.file_data_generator import FileDataGenerator
from network.utils.text_cleaner import TextCleaner
from network.utils.code_cleaner import CodeCleaner
from elasticsearch_dsl.connections import connections
from data.documents import ES_HOSTS, ES_LOGIN

from network.utils import constants

from network.metrics.f_scores import F1Score
from network.utils.train_utils import load_idx2word, load_word2idx, load_word2vec_embeddings, plot_graphs
from network.utils.train_utils import TRAIN_DS_TEXT_FN, DEV_DS_TEXT_FN, EMBEDDING_MATRIX_CUSTOM_CODE_FN
from network.utils.train_utils import TRAIN_DS_CODE_FN, DEV_DS_CODE_FN
from network.utils.train_utils import WORD_TO_IDX_CUSTOM_CODE_FN, IDX_TO_WORD_CUSTOM_CODE_FN

# hyperparameters values
hparams = {
    constants.HP_FIRST_DROPOUT: 0.5,
    constants.HP_SECOND_DROPOUT: 0.35,
    constants.HP_BATCH_SIZE: 256,
    constants.HP_SEQUENCE_LENGTH: 150,
    constants.HP_L2_REGULARIZATION: 0.05,
}

BUFFER_SIZE = 1000
EPOCHS = 15
PREFETCH_TRAIN = 2
PREFETCH_VAL = 2

if __name__ == '__main__':
    """
    Train the model selected by command line paramters
    """
    connections.create_connection(hosts=ES_HOSTS, http_auth=ES_LOGIN)

    # parse cmdline args
    parser = argparse.ArgumentParser("Evaluate model performance on test data and creates confusion matrix.")
    parser.add_argument("--model", required=True, choices=constants.ALL_MODELS_CONFIG.keys(),
                        help="Model to be used.")
    parser.add_argument("--embedding", required=True, choices=constants.TEXT_EMBEDDINGS.keys(),
                        help="Embedding to be used.")
    parser.add_argument("--loss", required=True, choices=["f1_loss", "sparse_categorical_crossentropy"],
                        help="Loss function to be used")
    args = parser.parse_args()

    # load idx to word dictionary and it's inversion
    text_word2idx = load_word2idx(constants.TEXT_EMBEDDINGS[args.embedding][1])
    text_idx2word = load_idx2word(constants.TEXT_EMBEDDINGS[args.embedding][2])

    # load pretrained embeddings
    text_embeddings = load_word2vec_embeddings(constants.TEXT_EMBEDDINGS[args.embedding][0])

    # configure rest of the hyperparameters
    hparams[constants.HP_MODEL] = args.model
    hparams[constants.HP_LOSS] = args.loss
    hparams[constants.HP_EMBEDDING] = args.embedding

    # define log and checkpoint paths
    now = datetime.now()
    CHECKPOINT_PATH = "network/checkpoints/cp-{epoch:04d}.ckpt"
    LOG_DIR = "network/logs/fit/" + args.model + "_" + args.loss + "_" + now.strftime("%d_%m_%Y_%H%M%S")
    CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)

    # define hparams statistics
    with tf.summary.create_file_writer(LOG_DIR).as_default():
        hp.hparams_config(
            hparams=[constants.HP_FIRST_DROPOUT, constants.HP_SECOND_DROPOUT, constants.HP_BATCH_SIZE,
                     constants.HP_SEQUENCE_LENGTH, constants.HP_MODEL, constants.HP_LOSS,
                     constants.HP_EMBEDDING, constants.HP_L2_REGULARIZATION],
            metrics=[hp.Metric('epoch_accuracy', display_name='Accuracy', group="validation"),
                     hp.Metric('epoch_f1_score', display_name='F1Score', group="validation")],
        )

    # is 2 or 3 class model chosen
    is_3_class_model = True if constants.ALL_MODELS_CONFIG[args.model][1] == 3 else False

    # is selected model using code embedding
    is_code_embedding = constants.ALL_MODELS_CONFIG[args.model][2]
    if is_code_embedding:
        code_word2idx = load_word2idx(WORD_TO_IDX_CUSTOM_CODE_FN)
        code_idx2word = load_idx2word(IDX_TO_WORD_CUSTOM_CODE_FN)
        code_embeddings = load_word2vec_embeddings(EMBEDDING_MATRIX_CUSTOM_CODE_FN)

    # -----------------------------TRAIN DATA-------------------------------------------------------------
    if not is_code_embedding:
        train_generator = FileDataGenerator(TRAIN_DS_TEXT_FN, text_word2idx, text_idx2word,
                                            max_sentence_length=hparams[constants.HP_SEQUENCE_LENGTH],
                                            use_3_cls=is_3_class_model)
        train_ds = tf.data.Dataset.from_generator(train_generator.generate,
                                                  args=[],
                                                  output_types=((tf.int32, tf.int32), tf.int32),
                                                  output_shapes=(((hparams[constants.HP_SEQUENCE_LENGTH],),
                                                                  (hparams[constants.HP_SEQUENCE_LENGTH],)), ()))
    else:
        train_generator = FileDataGenerator(TRAIN_DS_TEXT_FN, text_word2idx, text_idx2word,
                                            max_sentence_length=hparams[constants.HP_SEQUENCE_LENGTH],
                                            use_3_cls=is_3_class_model, embed_code=True, code2idx=code_word2idx,
                                            idx2code=code_idx2word, csv_code_file=TRAIN_DS_CODE_FN,
                                            max_code_length=hparams[constants.HP_SEQUENCE_LENGTH])
        train_ds = tf.data.Dataset.from_generator(train_generator.generate,
                                                  args=[],
                                                  output_types=((tf.int32, tf.int32, tf.int32, tf.int32), tf.int32),
                                                  output_shapes=(((hparams[constants.HP_SEQUENCE_LENGTH],),
                                                                  (hparams[constants.HP_SEQUENCE_LENGTH],),
                                                                  (hparams[constants.HP_SEQUENCE_LENGTH],),
                                                                  (hparams[constants.HP_SEQUENCE_LENGTH],)),
                                                                 ())
                                                  )

    train_ds = train_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    train_ds = train_ds.batch(hparams[constants.HP_BATCH_SIZE], drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    # --------------------------------------------------------------------------------------------------
    # -----------------------------DEV DATA-------------------------------------------------------------
    if not is_code_embedding:
        val_generator = FileDataGenerator(DEV_DS_TEXT_FN, text_word2idx, text_idx2word,
                                          max_sentence_length=hparams[constants.HP_SEQUENCE_LENGTH],
                                          use_3_cls=is_3_class_model)
        val_ds = tf.data.Dataset.from_generator(val_generator.generate,
                                                args=[],
                                                output_types=((tf.int32, tf.int32), tf.int32),
                                                output_shapes=(((hparams[constants.HP_SEQUENCE_LENGTH],),
                                                                 (hparams[constants.HP_SEQUENCE_LENGTH],)), ()))
    else:
        val_generator = FileDataGenerator(DEV_DS_TEXT_FN, text_word2idx, text_idx2word,
                                          max_sentence_length=hparams[constants.HP_SEQUENCE_LENGTH],
                                          use_3_cls=is_3_class_model, embed_code=True, code2idx=code_word2idx,
                                          idx2code=code_idx2word, csv_code_file=DEV_DS_CODE_FN,
                                          max_code_length=hparams[constants.HP_SEQUENCE_LENGTH])
        val_ds = tf.data.Dataset.from_generator(val_generator.generate,
                                                args=[],
                                                output_types=((tf.int32, tf.int32, tf.int32, tf.int32), tf.int32),
                                                output_shapes=(((hparams[constants.HP_SEQUENCE_LENGTH],),
                                                                 (hparams[constants.HP_SEQUENCE_LENGTH],),
                                                                 (hparams[constants.HP_SEQUENCE_LENGTH],),
                                                                 (hparams[constants.HP_SEQUENCE_LENGTH],)),
                                                                ())
                                                )

    val_ds = val_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    val_ds = val_ds.batch(hparams[constants.HP_BATCH_SIZE], drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    # --------------------------------------------------------------------------------------------------

    # create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        verbose=1,
        save_best_only=True,
        monitor="val_f1_score",
        mode="max",
        save_weights_only=True,
        save_freq='epoch'
    )

    # create tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

    # create early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=3, mode="max", verbose=1)

    # create chosen model
    if is_code_embedding:
        model = constants.ALL_MODELS_CONFIG[args.model][0](text_embeddings,
                                                           hparams,
                                                           constants.TEXT_EMBEDDINGS[args.embedding][3],
                                                           code_embeddings)
    else:
        model = constants.ALL_MODELS_CONFIG[args.model][0](text_embeddings,
                                                           hparams,
                                                           constants.TEXT_EMBEDDINGS[args.embedding][3])

    # create HParams callback
    hparams_callback = hp.KerasCallback(LOG_DIR, hparams)

    # instantiate chosen loss
    if args.loss == "f1_loss":
        class_nr = 3 if is_3_class_model else 2
        loss = constants.LOSSES[args.loss](class_nr)
    else:
        loss = constants.LOSSES[args.loss]()

    # Compile the model with Adam optimizer, crossentropy and macro F1 score + accuracy metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss,
                  metrics=[F1Score(num_classes=constants.ALL_MODELS_CONFIG[args.model][1], average="macro"),
                           tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])

    # train the model
    history = model.fit(train_ds, epochs=EPOCHS,
                        callbacks=[tensorboard_callback, cp_callback, early_stopping, hparams_callback],
                        validation_data=val_ds)

    # plot training graphs
    plot_graphs(history, "accuracy", args.model + " - " + args.loss, LOG_DIR + "/accuracy.png")
    plot_graphs(history, "f1_score", args.model + " - " + args.loss, LOG_DIR + "/f1_score.png")
    plot_graphs(history, "loss", args.model + " - " + args.loss, LOG_DIR + "/loss.png")
