#!/usr/bin/env python3

import sys
import math
import numpy as np
from os.path import isfile
import csv
import json

from scipy.sparse import lil_matrix
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from collections import Counter
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Dense, Layer, LSTM, TimeDistributed, Bidirectional, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC, Accuracy
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.config import experimental as tfconf_exp
from tensorflow_addons.metrics import F1Score

from transformers import AutoConfig, AutoTokenizer, TFAutoModel, TFAutoModelForSequenceClassification
from transformers.optimization_tf import create_optimizer

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from logging import warning

from readers import READERS, get_reader
from common import timed


# Parameter defaults
DEFAULT_BATCH_SIZE = 8
DEFAULT_SEQ_LEN = 512
DEFAULT_LR = 5e-5
DEFAULT_WARMUP_PROPORTION = 0.1
DEFAULT_MAX_LINES = 250

def init_tf_memory():
    gpus = tfconf_exp.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tfconf_exp.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', default=None,
                    help='pretrained model name')
    ap.add_argument('--train', metavar='FILE', required=False,
                    help='training data')
    ap.add_argument('--dev', metavar='FILE', required=False,
                    help='development data')
    ap.add_argument('--test', metavar='FILE', required=False,
                    help='test data', default=None)
    ap.add_argument('--pred', metavar='FILE', required=False,
                    help='file to be predicted')
    ap.add_argument('--batch_size', metavar='INT', type=int,
                    default=DEFAULT_BATCH_SIZE,
                    help='batch size for training')
    ap.add_argument('--epochs', metavar='INT', type=int, default=1,
                    help='number of training epochs')
    ap.add_argument('--lr', '--learning_rate', metavar='FLOAT', type=float,
                    default=DEFAULT_LR, help='learning rate')
    ap.add_argument('--seq_len', metavar='INT', type=int,
                    default=DEFAULT_SEQ_LEN,
                    help='maximum input sequence length')
    ap.add_argument('--max_lines', metavar='INT', type=int,
                    default=DEFAULT_MAX_LINES)
    ap.add_argument('--warmup_proportion', metavar='FLOAT', type=float,
                    default=DEFAULT_WARMUP_PROPORTION,
                    help='warmup proportion of training steps')
    ap.add_argument('--input_format', choices=READERS.keys(),
                    default='json',
                    help='input file format')
    ap.add_argument('--multiclass', default=False, action='store_true',
                    help='task has exactly one label per text')
    ap.add_argument('--multilabel', default=False, action='store_true',
                    help='task has more than two possible labels')
    ap.add_argument('--output_file', default=None, metavar='FILE',
                    help='save model to file')
    ap.add_argument('--save_predictions', default=None, metavar='FILE',
                    help='save predictions and labels for dev set, or for test set if provided')
    #ap.add_argument('--save_predictions', default=False, action='store_true',
    #                help='save predictions and labels for dev set, or for test set if provided')
    ap.add_argument('--load_model', default=None, metavar='FILE',
                    help='load model from file')
    ap.add_argument('--lstm_model', default=None, metavar='FILE',
                    help='load lstm model from file')
    ap.add_argument('--log_file', default="train.log", metavar='FILE',
                    help='log parameters and performance to file')
    ap.add_argument('--threshold', metavar='FLOAT', type=float, default=None, 
                    help='fixed threshold for multilabel prediction')
    ap.add_argument('--test_log_file', default="test.log", metavar='FILE',
                    help='log parameters and performance on test set to file')
    return ap



def load_pretrained(options):
    name = options.model_name
    config = AutoConfig.from_pretrained(name)
    config.return_dict = True
    tokenizer = AutoTokenizer.from_pretrained(name, config=config, use_fast=False)#slow for transformers 4.7
    model = TFAutoModel.from_pretrained(name, config=config)
    #model = TFAutoModelForSequenceClassification.from_pretrained(name, config=config)

    if options.seq_len > config.max_position_embeddings:
        warning(f'--seq_len ({options.seq_len}) > max_position_embeddings '
                f'({config.max_position_embeddings}), using latter')
        options.seq_len = config.max_position_embeddings

    return model, tokenizer, config


def get_optimizer(num_train_examples, options):
    steps_per_epoch = math.ceil(num_train_examples / options.batch_size)
    num_train_steps = steps_per_epoch * options.epochs
    num_warmup_steps = math.floor(num_train_steps * options.warmup_proportion)

    # Mostly defaults from transformers.optimization_tf
    optimizer, lr_scheduler = create_optimizer(
        options.lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        min_lr_ratio=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay_rate=0.01,
        power=1.0,
    )
    return optimizer

# two separate build functions
def build_transformer(pretrained_model, num_labels, optimizer, options, embedding_output):
    seq_len = options.seq_len
    input_ids = Input(
        shape=(seq_len,), dtype='int32', name='input_ids')
#    token_type_ids = Input(
#        shape=(seq_len,), dtype='int32', name='token_type_ids')
    attention_mask = Input(
        shape=(seq_len,), dtype='int32', name='attention_mask')
#    inputs = [input_ids, attention_mask, token_type_ids]
    inputs = [input_ids, attention_mask]

    pretrained_outputs = pretrained_model(inputs)
    #pooled_output = pretrained_outputs[1]
    pooled_output = pretrained_outputs['last_hidden_state'][:,0,:] #CLS

    # TODO consider Dropout here
    if embedding_output:
        output = pooled_output
        loss = BinaryCrossentropy()
        metrics = [#F1Score(name='f1_th0.4', num_classes=num_labels, average='micro', threshold=0.4),
                   CategoricalAccuracy(name='categorical_accuracy', dtype=None)
        ]
    else:
        output = Dense(num_labels, activation='softmax')(pooled_output)
        loss = BinaryCrossentropy()
        metrics = [
            #F1Score(name='f1_th0.3', num_classes=num_labels, average='micro', threshold=0.3),
            #F1Score(name='f1_th0.4', num_classes=num_labels, average='micro', threshold=0.4)#,
#            F1Score(name='f1_th0.5', num_classes=num_labels, average='micro', threshold=0.5),
            CategoricalAccuracy(name='categorical_accuracy', dtype=None)
            #AUC(name='auc', multi_label=True)
        ]
    #output = pretrained_outputs # test
    model = Model(
        inputs=inputs,
        outputs=[output]
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model




def build_lstm(pretrained_model, num_labels, optimizer, options, MAX_LINES):
    seq_len = options.seq_len
#    if options.lstm_model is not None:
#        MAX_LINES = max([len(text) for text in test_texts])
#    else:
#        MAX_LINES = options.max_lines

    input_ids = Input(shape=(MAX_LINES, 768), dtype='float32', name='input_ids') #1024 for roberta-large
    attention_mask = Input(shape=(MAX_LINES,), dtype='bool', name='attention_mask')
    inputs = [input_ids, attention_mask]
    dropout = Dropout(0.15)(input_ids)
    lstm = Bidirectional(LSTM(768, return_sequences=True))(inputs=dropout, mask=attention_mask)
#    lstm = Bidirectional(LSTM(768, return_sequences=True))(inputs=input_embeddings, mask=mask)
    output = TimeDistributed(Dense(2, activation='softmax'))(lstm)

#    loss = BinaryCrossentropy()
    loss = CategoricalCrossentropy()
    metrics = [
#        F1Score(name='f1_th0.5', num_classes=num_labels, average='micro', threshold=0.5)
        CategoricalAccuracy(name='categorical_accuracy', dtype=None)
        ]
    model = Model(
        inputs=inputs,
        outputs=output
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model


@timed
def load_data(fn, options, max_chars=None):
    read = get_reader(options.input_format)
    texts, labels, ids = [], [], []
    with open(fn) as f:
        for ln, (text, text_labels) in enumerate(read(f, fn), start=1):
            if options.multiclass and not text_labels:
                raise ValueError(f'missing label on line {ln} in {fn}: {l}')
            elif options.multiclass and len(text_labels) > 1:
                raise ValueError(f'multiple labels on line {ln} in {fn}: {l}')
            words = 0
            for l in text:
                words += len(l.split())
            # skipt texts that contain less than 75 words or have different amount of lines and labels
            if len(text) != len(text_labels) or words < 75:
                continue
                #raise ValueError(f'labels and lines do not match')
#            if not options.load_model and len(text) > options.max_lines:
#                texts.append(text[:options.max_lines])
#                labels.append(labels[:options.max_lines])
#                ids.append(ln)

            # if text is longer than max_lines, chop it up for the lstm model training while preserving doc boundaries
            elif len(text) > options.max_lines and options.load_model:
                if len(text)%options.max_lines == 0:
                    x = int(len(text)/options.max_lines)
                else:
                    x = int(len(text)/options.max_lines) + 1
                for i in range(x):
                    if len(text) > (i+1)*options.max_lines:
                        texts.append(text[i*options.max_lines:(i+1)*options.max_lines])
                        labels.append(text_labels[i*options.max_lines:(i+1)*options.max_lines])
                        ids.append(ln)
                    else:
                        texts.append(text[i*options.max_lines:])
                        labels.append(text_labels[i*options.max_lines:])
                        ids.append(ln)
            else:
                texts.append(text)
                labels.append(text_labels)
                ids.append(ln)
    print(f'loaded {len(texts)} examples from {fn}', file=sys.stderr)
    return texts, labels, ids



class DataGenerator(Sequence):
    def __init__(self, data_path, tokenize_func, options, max_chars=None, label_encoder=None):
        texts, labels, ids = load_data(data_path, options, max_chars=max_chars)
        self.num_examples = len(texts)
        self.batch_size = options.batch_size
        #self.seq_len = options.seq_len
        self.X = tokenize_func(texts)

        if label_encoder is None:
            self.label_encoder = MultiLabelBinarizer()
            self.label_encoder.fit(labels)
        else:
            self.label_encoder = label_encoder

        self.Y = self.label_encoder.transform(labels)
        self.num_labels = len(self.label_encoder.classes_)

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(self.num_examples)
        np.random.shuffle(self.indexes)

    def __len__(self):
        return self.num_examples//self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_X = {}
        for key in self.X:
            batch_X[key] = np.empty((self.batch_size, *self.X[key].shape[1:]))
            for j, idx in enumerate(batch_indexes):
                batch_X[key][j] = self.X[key][idx]

        batch_y = np.empty((self.batch_size, *self.Y.shape[1:]), dtype=int)
        for j, idx in enumerate(batch_indexes):
            batch_y[j] = self.Y[idx]

        return batch_X, batch_y


def make_tokenization_function(tokenizer, options):
    seq_len = options.seq_len
    @timed
    def tokenize(text):
        tokenized = tokenizer(
            text,
            max_length=seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='np'
        )
        # Return dict b/c Keras (2.3.0-tf) DataAdapter doesn't apply
        # dict mapping to transformer.BatchEncoding inputs
        return {
            'input_ids': tokenized['input_ids'],
#            'token_type_ids': tokenized['token_type_ids'],
            'attention_mask': tokenized['attention_mask'],
        }
    return tokenize

def le(text_labels, labels, MAX_LINES, count):
    xlmr_Y = []
    lstm_Y = []
    for text in text_labels:
        lines = []
        for label in text: # is this needed?
            xlmr_Y.append(labels[label])
            lines.append(labels[label])
        lstm_Y.append(lines)
    count.update(xlmr_Y)
#    print('label distribution', Counter(xlmr_Y))
    lstm_Y = np.array(lstm_Y, dtype='object')
    lstm_Y = to_categorical(pad_sequences(lstm_Y, maxlen=MAX_LINES, padding='post', truncating='post'), num_classes=2, dtype='int')
    label_encoder = LabelEncoder()
    label_encoder.fit(xlmr_Y)
 #   print("LSTM Y SHAPE", np.shape(lstm_Y))
 #   print("Y SHAPE:", np.shape(xlmr_Y))
    xlmr_Y = label_encoder.transform(xlmr_Y)
    xlmr_Y = to_categorical(xlmr_Y)
    return xlmr_Y, lstm_Y

# wrangle the embeddings from the transformer model into sequential data for the lstm    
def wrangle_data(transformer_output, doc_ids, MAX_LINES):
    #MAX_LINES = options.max_lines
    #TO DO: track ids instead of indexes?
    last_doc_ID = -1
    lstm_input_data = np.zeros((doc_ids[-1]+1, MAX_LINES, 768)) #768 for xlmr base
    current_doc_idx = -1
    current_line_idx = -1
    mask = np.zeros((doc_ids[-1]+1, MAX_LINES), dtype='bool')
    for line_idx in range(transformer_output.shape[0]):
        if doc_ids[line_idx] == last_doc_ID:
            current_line_idx += 1
            if current_line_idx < MAX_LINES:                
                lstm_input_data[doc_ids[line_idx], current_line_idx] = transformer_output[line_idx]
                mask[doc_ids[line_idx], current_line_idx] = True
#            current_line_idx += 1
            # check that current_line_idx < MAX_LINES
            # insert transformer_output[line_idx] into lstm_input_data[current_doc_idx, current_line_idx]
        else:
            current_line_idx = 0
            current_doc_idx += 1
            last_doc_ID = doc_ids[line_idx]
            lstm_input_data[doc_ids[line_idx], current_line_idx] = transformer_output[line_idx]
            mask[doc_ids[line_idx], current_line_idx] = True
            # then insert as above

    return {
        'input_ids': lstm_input_data,
        'attention_mask': mask
    }


@timed
def prepare_classifier(num_train_examples, num_labels, options):
    optimizer = get_optimizer(num_train_examples, options)
    pretrained_model, tokenizer, config = load_pretrained(options)
    model = build_classifier(pretrained_model, num_labels, optimizer, options, num_train_examples)
    return model, tokenizer, optimizer


def optimize_threshold(model, train_X, train_Y, test_X, test_Y, options=None, epoch=None, save_pred_to=None):
    labels_prob = model.predict(train_X, verbose=1)#, batch_size=options.batch_size)

    best_f1 = 0.
    print("Optimizing threshold...\nThres.\tPrec.\tRecall\tF1")
    for threshold in np.arange(0.1, 0.9, 0.05):
        labels_pred = lil_matrix(labels_prob.shape, dtype='b')
        labels_pred[labels_prob>=threshold] = 1
        precision, recall, f1, _ = precision_recall_fscore_support(train_Y, labels_pred, average="micro")
        print("%.2f\t%.4f\t%.4f\t%.4f" % (threshold, precision, recall, f1), end="")
        if f1 > best_f1:
            print("\t*")
            best_f1 = f1
            #best_f1_epoch = epoch
            best_f1_threshold = threshold
        else:
            print()

    #print("Current F_max:", best_f1, "epoch", best_f1_epoch+1, "threshold", best_f1_threshold, '\n')
    #print("Current F_max:", best_f1, "threshold", best_f1_threshold, '\n')

    test_labels_prob = model.predict(test_X, verbose=1)#, batch_size=options.batch_size)
    test_labels_pred = lil_matrix(test_labels_prob.shape, dtype='b')
    test_labels_pred[test_labels_prob>=best_f1_threshold] = 1
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_Y, test_labels_pred, average="micro")
    if epoch:
        epoch_str = ", epoch %d" % epoch
    else:
        epoch_str = ""
    print("\nValidation/Test performance at threshold %.2f%s: Prec. %.4f, Recall %.4f, F1 %.4f" % (best_f1_threshold, epoch_str, test_precision, test_recall, test_f1))

    if save_pred_to is not None:
        if epoch is None:
            print("Saving predictions to", save_pred_to+".*.npy" % epoch)
            np.save(save_pred_to+".preds.npy", test_labels_pred.toarray())
            np.save(save_pred_to+".gold.npy", test_Y)
            #np.save(save_pred_to+".class_labels.npy", label_encoder.classes_)
        else:
            print("Saving predictions to", save_pred_to+"-epoch%d.*.npy" % epoch)
            np.save(save_pred_to+"-epoch%d.preds.npy" % epoch, test_labels_pred.toarray())
            np.save(save_pred_to+"-epoch%d.gold.npy" % epoch, test_Y)
            #np.save(save_pred_to+"-epoch%d.class_labels.npy" % epoch, label_encoder.classes_)

    return test_f1, best_f1_threshold, test_labels_pred


def test_threshold(model, test_X, test_Y, threshold=0.4, options=None, epoch=None, return_auc=False, save_pred_to=None):
    test_labels_prob = model.predict(test_X, verbose=1)#, batch_size=options.batch_size)
    test_labels_pred = lil_matrix(test_labels_prob.shape, dtype='b')
    test_labels_pred[test_labels_prob>=threshold] = 1
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_Y, test_labels_pred, average="micro")
    if epoch:
        epoch_str = ", epoch %d" % epoch
    else:
        epoch_str = ""
    print("\nValidation/Test performance at threshold %.2f%s: Prec. %.4f, Recall %.4f, F1 %.4f" % (threshold, epoch_str, test_precision, test_recall, test_f1))
    if save_pred_to is not None:
        if epoch is None:
            print("Saving predictions to", save_pred_to+".*.npy")
            np.save(save_pred_to+".preds.npy", test_labels_pred.toarray())
            np.save(save_pred_to+".gold.npy", test_Y)
            #np.save(save_pred_to+".class_labels.npy", label_encoder.classes_)
        else:
            print("Saving predictions to", save_pred_to+"-epoch%d.*.npy" % epoch)
            np.save(save_pred_to+"-epoch%d.preds.npy" % epoch, test_labels_pred.toarray())
            np.save(save_pred_to+"-epoch%d.gold.npy" % epoch, test_Y)
            #np.save(save_pred_to+"-epoch%d.class_labels.npy" % epoch, label_encoder.classes_)

    if return_auc:
        auc = roc_auc_score(test_Y, test_labels_prob, average = 'micro')
        return test_f1, threshold, test_labels_pred, auc
    else:
        return test_f1, threshold, test_labels_pred


def test_auc(model, test_X, test_Y):
    labels_prob = model.predict(test_X, verbose=1)
    return roc_auc_score(test_Y, labels_prob, average = 'micro')


class Logger:
    def __init__(self, filename, model, params):
        self.filename = filename
        self.model = model
        self.log = dict([('p%s'%p, v) for p, v in params.items()])

    def record(self, epoch, logs):
        for k in logs:
            self.log['_%s' % k] = logs[k]
        self.log['_Epoch'] = epoch
        self.write()

    def write(self):
        file_exists = isfile(self.filename)
        with open(self.filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, sorted(self.log.keys()))
            if not file_exists:
                print("Creating log file", self.filename, flush=True)
                writer.writeheader()
            writer.writerow(self.log)


class EvalCallback(Callback):
    def __init__(self, model, train_X, train_Y, dev_X, dev_Y, test_X=None, test_Y=None, logfile="train.log", test_logfile=None, save_pred_to=None, params={}):
        self.model = model
        self.train_X = train_X
        self.train_Y = train_Y
        self.dev_X = dev_X
        self.dev_Y = dev_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.logger = Logger(logfile, self.model, params)
        if test_logfile is not None:
            print("Setting up test set logging to", test_logfile, flush=True)
            self.test_logger = Logger(test_logfile, self.model, params)
        if save_pred_to is not None:
            self.save_pred_to = save_pred_to
        else:
            self.save_pred_to = None

    def on_epoch_end(self, epoch, logs={}):
        print("Validation set performance:")
        logs['f1'], _, _ = optimize_threshold(self.model, self.train_X, self.train_Y, self.dev_X, self.dev_Y, epoch=epoch)
        logs['rocauc'] = test_auc(self.model, self.dev_X, self.dev_Y)
        print("AUC dev:", logs['rocauc'])

        self.logger.record(epoch, logs)
        if self.test_X is not None:
            print("Test set performance:")
            test_f1, _, _ = optimize_threshold(self.model, self.train_X, self.train_Y, self.test_X, self.test_Y, epoch=epoch, save_pred_to=self.save_pred_to)
            auc = test_auc(self.model, self.test_X, self.test_Y)
            print("AUC test:", auc)
            self.test_logger.record(epoch, {'f1': test_f1, 'rocauc': auc})

# f-score per document
class EvaluateDocs(Callback):
    def __init__(self, model, train_X, train_Y, dev_X, dev_Y, dev_labels, test_X=None, test_Y=None, test_labels=None, logfile="train.log", test_logfile=None, save_pred_to=None, params={}\
):
        self.model = model
        self.train_X = train_X
        self.train_Y = train_Y
        self.dev_X = dev_X
        self.dev_Y = dev_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.test_labels = test_labels
        self.dev_labels = dev_labels
        self.logger = Logger(logfile, self.model, params)
        if test_logfile is not None:
            print("Setting up test set logging to", test_logfile, flush=True)
            self.test_logger = Logger(test_logfile, self.model, params)
        if save_pred_to is not None:
            self.save_pred_to = save_pred_to
        else:
            self.save_pred_to = None

    def on_epoch_end(self, epoch, logs={}):
#        self.losses.append(logs.get('loss'))
        preds = self.model.predict(self.dev_X)
#        print('preds shape', np.shape(preds))
        f1s = []
        for i, doc in enumerate(self.dev_labels):
            if len(doc) < 250:
                dev_Y = np.argmax(self.dev_Y[i],axis=-1)[:len(doc)]
            else:
                dev_Y = np.argmax(self.dev_Y[i],axis=-1)
            pred_Y = np.argmax(preds[i], axis = -1)
            pred_Y = pred_Y[:len(dev_Y)]
            test_acc = accuracy_score(dev_Y, pred_Y)
            f1s.append(test_acc)
#            tn, fp, fn, tp = confusion_matrix(dev_Y, pred).ravel()
#            test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(np.argmax(self.dev_Y[i], axis = 1), np.argmax(pred, axis=1), average=None)
#            print("\nPrec. %.4f, Recall %.4f, F1 %.4f" % (test_precision, test_recall, test_f1))
#            f1s.append(test_f1)
        logs['custom_accuracy'] = np.average(f1s)
#        logs['f1'], _, _ = optimize_threshold(self.model, self.train_X, self.train_Y, self.dev_X, self.dev_Y, epoch=epoch)
#        logs['rocauc'] = test_auc(self.model, self.dev_X, self.dev_Y)
#        print("AUC dev:", logs['rocauc'])

        self.logger.record(epoch, logs)
        
        if self.test_X is not None:
            testf1s = []
            testpreds = self.model.predict(self.test_X)
            print("Test set performance:")
            for i, doc in enumerate(self.test_labels):            
#                test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(np.argmax(self.test_Y[i], axis = 1), np.argmax(pred, axis=1), average=None)
#                testf1s.append(test_f1)
                if len(doc) < 250:
                    test_Y = np.argmax(self.test_Y[i],axis=-1)[:len(doc)]
                else:
                    test_Y = np.argmax(self.test_Y[i],axis=-1)
                pred_Y = np.argmax(testpreds[i], axis = -1)
                pred_Y = pred_Y[:len(test_Y)]
                test_acc = accuracy_score(test_Y, pred_Y)
                f1s.append(test_acc)
            
            '''
            test_f1, _, _ = optimize_threshold(self.model, self.train_X, self.train_Y, self.test_X, self.test_Y, epoch=epoch, save_pred_to=self.save_pred_to\
)
            auc = test_auc(self.model, self.test_X, self.test_Y)
            print("AUC test:", auc)
            '''
            self.test_logger.record(epoch, {'custom_accuracy': np.average(testf1s)})
                
        return

def main(argv):
    init_tf_memory()
    options = argparser().parse_args(argv[1:])
    MAX_LINES = options.max_lines
    if options.train is not None:
        train_texts, train_labels, train_ids = load_data(options.train, options, max_chars=25000)
        print("TEXTS", len(train_texts))
        num_train_texts = len(train_texts)
    if options.dev is not None:
        dev_texts, dev_labels, dev_ids = load_data(options.dev, options, max_chars=25000)
        print('dev texts', len(dev_texts))
    if options.test is not None:
        test_texts, test_labels, test_ids = load_data(options.test, options, max_chars=25000)
        print('test texts', len(test_texts))
    if options.pred is not None:
#        print("fetching file")
        pred_texts = []
        pred_ids = []
        with open(options.pred, 'r') as f:
            data = json.load(f)
            for k, key in enumerate(data):
                try:
                    text = data[key]['text']
                    lines = text.split('\n')
                    text_id = data[key]['id']
#                    labels = data[key]['labels']
                except:
                    continue
                if len(lines) > MAX_LINES:
                    if len(lines)%options.max_lines == 0:
                        x = int(len(lines)/options.max_lines)
                    else:
                        x = int(len(lines)/options.max_lines) + 1
                    for i in range(x):
                        if len(lines) > (i+1)*options.max_lines:
                            pred_texts.append(lines[i*options.max_lines:(i+1)*options.max_lines])
                            pred_ids.append(text_id)
                        else:
                            pred_texts.append(lines[i*options.max_lines:])
                            pred_ids.append(text_id)
                else:
                    pred_texts.append(lines)
                    pred_ids.append(text_id)
#                    pred_labels.append(labels)
        print(f"loaded {len(set(pred_ids))} pred texts")
        #print('test texts', len(test_texts))
#    print(train_texts[0])

        # split into batches to be predicted
        pred_texts = [pred_texts[i:i+50] for i in range(0,len(pred_texts),50)]
        pred_ids = [pred_ids[i:i+50] for i in range(0,len(pred_ids),50)]
#        pred_labels = [pred_labels[i:i+50] for i in range(0,len(pred_labels),50)]

        print(f"loaded {len(pred_ids)} pred batches")

    num_labels = 2
    labels = {'0':0, '1':1}
    count = Counter()

    # label encoding
    if options.train is not None:
        #print("train labels")
        xlmr_train_Y, lstm_train_Y = le(train_labels, labels, MAX_LINES, count)
    if options.dev is not None:
        #print("dev labels")
        xlmr_dev_Y, lstm_dev_Y = le(dev_labels, labels, MAX_LINES, count)
    if options.test is not None:
        #print("test labels")
        xlmr_test_Y, lstm_test_Y = le(test_labels, labels, MAX_LINES, count)
    
    print(count)
#    print(xlmr_test_Y)
    
    if options.train is not None:
        xlmr_num_train_examples = len(xlmr_train_Y)
        lstm_num_train_examples = len(lstm_train_Y)
        train_X = []
        train_doc_ids = []
        for i, doc in enumerate(train_texts):
            for line in doc:
                train_X.append(line)
                train_doc_ids.append(i)
#    print("DOC IDS:",train_doc_ids)
        print("TRAIN SHAPE", np.shape(train_X))
    if options.dev is not None:
        dev_X = []
        dev_doc_ids = []
        for i, doc in enumerate(dev_texts):
            for line in doc:
                dev_X.append(line)
                dev_doc_ids.append(i)
    if options.test is not None:
            test_X = []
            test_doc_ids = []
            for i, doc in enumerate(test_texts):
                for line in doc:
                    test_X.append(line)
                    test_doc_ids.append(i)
            xlmr_num_train_examples = len(test_X)
            lstm_num_train_examples = len(test_texts)
            xlmr_texts = test_X
    if options.pred is not None:
            pred_Xs = []
            pred_doc_ids = []
            for b in pred_texts:
                pred_X = []
                pred_doc_id = []
                for i, doc in enumerate(b):
                    for line in doc:
                        pred_X.append(line)
                        pred_doc_id.append(i)
                pred_Xs.append(pred_X)
                pred_doc_ids.append(pred_doc_id)
            xlmr_num_train_examples = 256
            lstm_num_train_examples = 256
    xlmr_optimizer = get_optimizer(xlmr_num_train_examples, options)
    lstm_optimizer = get_optimizer(lstm_num_train_examples, options)
    pretrained_model, tokenizer, config = load_pretrained(options)
    xlmr_model = build_transformer(pretrained_model, num_labels, xlmr_optimizer, options, embedding_output=False)

    if options.load_model is not None and options.lstm_model is None:
        print("Loading xlmr model...")
        xlmr_model.load_weights(options.load_model)
        xlmr_emb = build_transformer(pretrained_model, num_labels, xlmr_optimizer, options, embedding_output=True)
        for i in range(len(xlmr_model.layers)-1):
            xlmr_emb.layers[i].set_weights(xlmr_model.layers[i].get_weights())
        lstm_model = build_lstm(pretrained_model, num_labels, lstm_optimizer, options, MAX_LINES)
        print("create lstm model")
        classifier = lstm_model
    elif options.lstm_model is not None:
        print("Loading xlmr model...")
        xlmr_model.load_weights(options.load_model)
        xlmr_emb = build_transformer(pretrained_model, num_labels, xlmr_optimizer, options, embedding_output=True)
        for i in range(len(xlmr_model.layers)-1):
            xlmr_emb.layers[i].set_weights(xlmr_model.layers[i].get_weights())
        lstm_model = build_lstm(pretrained_model, num_labels, lstm_optimizer, options, MAX_LINES)
        lstm_model.load_weights(options.lstm_model)
        classifier = lstm_model

    else:
        classifier = xlmr_model
    
    tokenize = make_tokenization_function(tokenizer, options)
    if options.train is not None:
        train_X = tokenize(train_X)
    if options.dev is not None:
        dev_X = tokenize(dev_X)

    #train_gen = DataGenerator(options.train, tokenize, options, max_chars=25000)
    #dev_gen = DataGenerator(options.dev, tokenize, options, max_chars=25000, label_encoder=train_gen.label_encoder)
    if options.test is not None:
        test_X = tokenize(test_X)
    if options.pred is not None:
        for i, pred_X in enumerate(pred_Xs):
            pred_Xs[i] = tokenize(pred_X)
#    print(f'train shape: {np.shape(train_X['input_ids'])}, test shape: {np.shape(test_X['input_ids'])}')

    # check xlmr model performance
#    if options.load_model is not None and options.test is not None:
#        xlmr_preds = xlmr_model.predict(test_X, verbose=0)
#        print("loaded model accuracy:", accuracy_score(np.argmax(xlmr_test_Y,axis=1),np.argmax(xlmr_preds,axis=1)))
#        tn, fp, fn, tp = confusion_matrix(np.argmax(xlmr_test_Y,axis=1),np.argmax(xlmr_preds,axis=1)).ravel()
#        print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
        # if you want to see the predictions with this model:
#        for i, doc in enumerate(xlmr_texts):
#            for l, line in enumerate(doc):
#                if l >= MAX_LINES:
#                   break
#            print(f'{np.argmax(xlmr_preds[i], axis = -1)}\t{xlmr_preds[i][1]}\t{np.argmax(xlmr_test_Y[i], axis=-1)}\t{xlmr_texts[i]}')
#            print()

    # set up lstm model training
    if options.load_model is not None and options.lstm_model is None:
        xlmr_preds = xlmr_model.predict(test_X, verbose=0)
#        xlmr_preds = (xlmr_preds > 0.5)
#        print("xlmr preds:",xlmr_preds)
#        print("test y:", xlmr_test_Y)
#        print("test x:",np.argmax(xlmr_test_Y,axis=1))
        print("loaded model accuracy:", accuracy_score(np.argmax(xlmr_test_Y,axis=1),np.argmax(xlmr_preds,axis=1)))
        print("Getting the embeddings...")
        transformer_output_train = xlmr_emb.predict(train_X, verbose=0)
        transformer_output_dev = xlmr_emb.predict(dev_X, verbose=0)
        transformer_output_test = xlmr_emb.predict(test_X,verbose=0)
        print("Wrangling the data...")
        train_X = wrangle_data(transformer_output_train, train_doc_ids, MAX_LINES)
        print("TRAIN SHAPE", np.shape(train_X))
#        print('lstm train shape', train_X.shape)
#    lstm_input_data_train = lstm_input_data_train.reshape(lstm_input_data_train.shape[0]*lstm_input_data_train.shape[1],lstm_input_data_train.shape[2])
#    print('lstm train shape', lstm_input_data_train.shape)
#        print(train_X)
        dev_X = wrangle_data(transformer_output_dev, dev_doc_ids, MAX_LINES)
        test_X = wrangle_data(transformer_output_test, test_doc_ids, MAX_LINES)
        train_Y = lstm_train_Y
        dev_Y = lstm_dev_Y
        test_Y = lstm_test_Y
    # set up lstm evaluation
    elif options.lstm_model is not None and options.test is not None:
        xlmr_preds = xlmr_model.predict(test_X, verbose=0)
        print("loaded model accuracy:", accuracy_score(np.argmax(xlmr_test_Y,axis=1),np.argmax(xlmr_preds,axis=1)))
        #print("Getting the embeddings...")
        transformer_output_test = xlmr_emb.predict(test_X,verbose=0)
        #print("Wrangling the data...")
        test_X = wrangle_data(transformer_output_test, test_doc_ids, MAX_LINES)
        test_Y = lstm_test_Y
    # predict only
    elif options.pred is not None:
        result = {}
        pred = []
        lines = []
        probs = []
        for i, pred_X in enumerate(pred_Xs):
            xlmr_preds = xlmr_model.predict(pred_X, verbose=0)
            #print("loaded model accuracy:", accuracy_score(np.argmax(xlmr_test_Y,axis=1),np.argmax(xlmr_preds,axis=1)))
            print("Getting the embeddings...")
            transformer_output_test = xlmr_emb.predict(pred_X,verbose=0)
            print("Wrangling the data...")
            pred_X = wrangle_data(transformer_output_test, pred_doc_ids[i], MAX_LINES)
            print("wrangled")
            current_id = -1
            print("Predicting batch", i)
            preds = classifier.predict(pred_X)
            for j, doc in enumerate(pred_texts[i]):
                if pred_ids[i][j] != current_id:
                    if options.save_predictions is not None and current_id != -1:
                        result[current_id] = {}
                        result[current_id]['id'] = current_id
                        result[current_id]['text'] = lines
                        result[current_id]['linelabels'] = ['ok' if p==1 else 'boilerplate' for p in pred]
                        result[current_id]['probs'] = probs
                        pred = []
                        lines = []
                        probs = []
                    print()
                    print("##ID:", pred_ids[i][j])
#                    print("##Labels:", pred_labels[i][j])
                    current_id = pred_ids[i][j]
                for l, line in enumerate(doc):
                    if options.save_predictions is not None:
                        pred.append(np.argmax(preds[j][l],axis=-1))
                        probs.append("{:.5f}".format(preds[j][l][1]))
                        lines.append(pred_texts[i][j][l].strip())
                    print("\t".join([str(np.argmax(preds[j][l],axis=-1)),"{:.5f}".format(preds[j][l][1]),pred_texts[i][j][l].strip()]))
        print(result)
        result = json.dumps(result)
        with open(options.save_predictions,'w') as f:
            f.write(result)
  #      test_Y = lstm_test_Y
    # set up xlm-r training
    else:
        train_Y = xlmr_train_Y
        dev_Y = xlmr_dev_Y
        test_Y = xlmr_test_Y
#    print("Lstm input")
#    print(train_X)
#    print("LSTM INPUT DATA SHAPE", lstm_input_data_train.shape[0])
#    lstm_model = build_lstm(pretrained_model, num_labels, optimizer, options)
#    lstm_model.summary()
    

    if options.lstm_model is None:
        callbacks = [] #[ModelCheckpoint(options.output_file+'.{epoch:02d}', save_weights_only=True)]
        if options.threshold is None:
            if options.test is not None and options.test_log_file is not None:
                print("Initializing evaluation with dev and test set...")
                callbacks.append(EvalCallback(classifier, train_X, train_Y, dev_X, dev_Y, test_X=test_X, test_Y=test_Y,
                                              logfile=options.log_file,
                                              test_logfile=options.test_log_file,
                                              save_pred_to=options.output_file,
                                              params={'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size}))
            else:
                print("Initializing evaluation with dev set...")
                callbacks.append(EvalCallback(classifier, train_X, train_Y, dev_X, dev_Y,
                                              logfile=options.log_file,
                                              params={'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size}))
        else:

            if options.test is not None and options.test_log_file is not None:
                print("Initializing evaluation with dev and test set...")
                callbacks.append(EvaluateDocs(classifier, train_X, train_Y, dev_X, dev_Y, test_X=test_X, test_Y=test_Y, dev_labels=dev_labels, test_labels=test_labels,
                                              logfile=options.log_file,
                                              test_logfile=options.test_log_file,
                                              save_pred_to=options.output_file,
                                              params={'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size}))
            else:
                print("Initializing evaluation with dev set...")
                callbacks.append(EvaluateDocs(classifier, train_X, train_Y, dev_X, dev_Y,dev_labels=dev_labels,
                                              logfile=options.log_file,
                                              params={'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size}))
        
#        print("Initializing early stopping criterion...")
#        callbacks.append(EarlyStopping(monitor='val_loss', verbose=1, patience=5, mode='min', restore_best_weights=True))
#        callbacks.append(EarlyStopping(monitor='val_f1_th0.5', verbose=1, patience=5, mode="max", restore_best_weights=True))
        callbacks.append(EarlyStopping(monitor='val_categorical_accuracy', verbose=1, patience=5, mode="max", restore_best_weights=True))

        history = classifier.fit(
            train_X,
            train_Y,
            epochs=options.epochs,
            batch_size=options.batch_size,
            validation_data=(dev_X, dev_Y),
            callbacks=callbacks
        )
            
        if options.threshold is not None and options.load_model is None:
            if options.dev is not None:
#                f1, _, _, rocauc = test_threshold(lstm_model, lstm_input_data_dev, lstm_dev_Y, return_auc=True, threshold=options.threshold)

                f1, _, _, rocauc = test_threshold(classifier, dev_X, dev_Y, return_auc=True, threshold=options.threshold)
                print("Restored best checkpoint, F1: %.6f, AUC: %.6f" % (f1, rocauc))

                logger = Logger(options.log_file, None, {'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size})
                try:
                    epoch = len(history.history['loss'])-1
                except:
                    epoch = -1
                logger.record(epoch, {'f1': f1, 'rocauc': rocauc})

            if options.test is not None:
                '''
                for i, (test_X, test_Y) in enumerate(zip(test_Xs, test_Ys)):
                    test_log_file = options.test_log_file.split(';')[i]
                    print("Evaluating on test set %d..." % i)
                    test_f1, test_th, test_pred, test_rocauc = test_threshold(classifier, test_X, test_Y, threshold=options.threshold, return_auc=True)
                    print("AUC test:", test_rocauc)
                    print("Logging to", test_log_file)
                    test_logger = Logger(options.test_log_file.split(';')[i], None, {'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size})
                    try:
                        epoch = len(history.history['loss'])-1
                    except:
                        epoch = -1
                    test_logger.record(epoch, {'f1': test_f1, 'rocauc': test_rocauc})
                '''
                print("Evaluating on test set...")
                test_f1, test_th, test_pred = test_threshold(lstm_model, lstm_input_data_test, lstm_test_Y, return_auc=True, threshold=options.threshold)

#            test_f1, test_th, test_pred = test_threshold(classifier, test_X, test_Y, threshold=options.threshold, return_auc=True)
                print("AUC test:", test_rocauc)
                print("Logging to", test_log_file)
                test_logger = Logger(options.test_log_file.split(';')[i], None, {'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size})
                try:
                    epoch = len(history.history['loss'])-1
                except:
                    epoch = -1
                test_logger.record(epoch, {'f1': test_f1, 'rocauc': test_rocauc})
        else:
            print("Evaluate on test data")
            results = classifier.evaluate(test_X, test_Y)
            print("Keras test loss, test acc:", results)
    
#    test_X, lstm_test_Y, test_texts = shuffle(test_X, lstm_test_Y, test_texts) 
        if options.load_model is not None and options.test is not None:
            preds = classifier.predict(test_X)
            print("PREDS SHAPE", np.shape(preds))
            f1s = []
            for i, doc in enumerate(test_texts):
                if len(doc) < MAX_LINES:
                    test_Y = np.argmax(lstm_test_Y[i],axis=-1)[:len(doc)]
                else:
                    test_Y = np.argmax(lstm_test_Y[i],axis=-1)
                pred_Y = np.argmax(preds[i], axis = -1)[:len(test_Y)]
                test_acc = accuracy_score(test_Y, pred_Y)
                f1s.append(test_acc)
            print("AVG OF ACCURACY",np.average(f1s))
        
            for i, doc in enumerate(test_texts):
                print("ACCURACY", f1s[i])
                for l, line in enumerate(doc):
                    if l >= MAX_LINES:
                        break
                    print(f'{np.argmax(preds[i][l], axis = -1)} {preds[i][l][1]}     {np.argmax(lstm_test_Y[i][l], axis=-1)} {test_texts[i][l]}')
                print()
    
        try:
            if options.output_file:
                print("Saving model to %s" % options.output_file)
                classifier.save_weights(options.output_file)
        except:
            pass

    if options.lstm_model is not None and options.test is not None:
        current_id = -1
        if options.pred is None:
            print("Evaluate on test data")
            results = classifier.evaluate(test_X, test_Y)
            print("Keras test loss, test acc:", results)
            true_Y = []
            pred_Y = []
            preds = classifier.predict(test_X)
            for i, doc in enumerate(test_texts):
                if test_ids[i] != current_id:
                    print("ACCURACY", accuracy_score(true_Y, pred_Y))
                    print()
                    print("INDX", test_ids[i])
                    current_id = test_ids[i]
                    true_Y = []
                    pred_Y = []
                for l, line in enumerate(doc):
#                    if l >= MAX_LINES:
#                        break
                    pred_Y.append(np.argmax(preds[i][l], axis=-1))
                    true_Y.append(np.argmax(lstm_test_Y[i][l], axis=-1))
                        
                    print("\t".join([str(np.argmax(preds[i][l], axis = -1)),"{:.5f}".format(preds[i][l][1]),str(np.argmax(lstm_test_Y[i][l], axis=-1)),test_texts[i][l]]))
            print("ACCURACY", accuracy_score(true_Y, pred_Y))

    return 0
    

if __name__ == '__main__':
    sys.exit(main(sys.argv))

