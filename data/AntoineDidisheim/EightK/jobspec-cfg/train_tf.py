import didipack as didi
import numpy as np
import pandas as pd
import os

import tqdm

from parameters import *
from data import *
from utils_local.nlp_ticker import *
from didipack.utils_didi.ridge import run_efficient_ridge
from didipack.trainer.trainer_ridge import TrainerRidge
# from didipack.trainer.trainer_logistic_elastic_net import TrainerLogisticElasticNet
from didipack.trainer.trainer_logistic_elastic_net import TrainerLogisticElasticNet
from didipack.trainer.train_splitter import get_start_dates, get_chunks, get_tasks_for_current_node
import psutil
from utils_local.general import *
from utils_local.trainer_specials import *
from experiments_params import get_main_experiments
import tensorflow as tf


class PipelineTrainer:
    def __init__(self, par: Params):
        self.par = par
        self.norm_params = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.test_dataset_with_id = None
        self.input_dim = None
        self.best_hyper = None
        self.model = None
        self.best_hyper_value = None

    @staticmethod
    def compute_mean_variance(dataset):
        running_sum = 0
        running_squared_sum = 0
        num_samples = 0

        for batch, _ in tqdm.tqdm(dataset, 'Computing M-V on training sample'):
            dense_batch = tf.sparse.to_dense(batch)  # Convert sparse tensor to dense tensor
            batch_size = dense_batch.shape[0]
            running_sum += tf.reduce_sum(dense_batch, axis=0)
            running_squared_sum += tf.reduce_sum(tf.square(dense_batch), axis=0)
            num_samples += batch_size

        mean = running_sum / num_samples
        variance = (running_squared_sum / num_samples) - tf.square(mean)

        return mean.numpy(), variance.numpy()

    def parse_tfrecord(self, example_proto):
        if self.par.enc.news_source in [NewsSource.NEWS_REF_ON_EIGHT_K]:
            feature_description = {
                'vec': tf.io.VarLenFeature(tf.float32),
                'id': tf.io.FixedLenFeature([], tf.string),
                'index': tf.io.FixedLenFeature([], tf.int64),
                'timestamp': tf.io.FixedLenFeature([], tf.string),
                'alert': tf.io.FixedLenFeature([], tf.int64),
                'ticker': tf.io.FixedLenFeature([], tf.string),
                'date': tf.io.FixedLenFeature([], tf.string),
                'permno': tf.io.FixedLenFeature([], tf.int64),
                'news0': tf.io.FixedLenFeature([], tf.int64),
                'ret': tf.io.FixedLenFeature([], tf.float32),
                'abret': tf.io.FixedLenFeature([], tf.float32),
                'cosine': tf.io.FixedLenFeature([], tf.float32),
                'm_cosine': tf.io.FixedLenFeature([], tf.float32),
                'prn': tf.io.FixedLenFeature([], tf.int64),
                'reuters': tf.io.FixedLenFeature([], tf.int64),
            }
        else:
            feature_description = {
                'vec': tf.io.VarLenFeature(tf.float32),
                'id': tf.io.FixedLenFeature([], tf.string),
                'index': tf.io.FixedLenFeature([], tf.int64),
                'timestamp': tf.io.FixedLenFeature([], tf.string),
                'alert': tf.io.FixedLenFeature([], tf.int64),
                'ticker': tf.io.FixedLenFeature([], tf.string),
                'date': tf.io.FixedLenFeature([], tf.string),
                'permno': tf.io.FixedLenFeature([], tf.int64),
                'ret': tf.io.FixedLenFeature([], tf.float32),
                'ret_m': tf.io.FixedLenFeature([], tf.float32),
                'reuters': tf.io.FixedLenFeature([], tf.int64),
            }

        parsed_features = tf.io.parse_single_example(example_proto, feature_description)

        if self.norm_params is not None:
            if self.par.train.norm == Normalisation.ZSCORE:
                parsed_features['vec'] = (parsed_features['vec'] - self.norm_params['mean']) / tf.sqrt(self.norm_params['var'] + 1e-7)
        return parsed_features

    @tf.autograph.experimental.do_not_convert
    def filter_start_year(self, x, const_start_year):
        return tf.greater_equal(tf.strings.to_number(tf.strings.substr(x['date'], 0, 4), out_type=tf.int32), const_start_year)

    @tf.autograph.experimental.do_not_convert
    def filter_end_year(self, x, const_end_year):
        return (tf.less_equal(tf.strings.to_number(tf.strings.substr(x['date'], 0, 4), out_type=tf.int32), const_end_year))

    @tf.autograph.experimental.do_not_convert
    def filter_sample_based_on_par(self, x):
        reuters_condition = tf.constant(True, dtype=tf.bool) if self.par.train.filter_on_reuters is None else tf.equal(x['reuters'], par.train.filter_on_reuters)
        prn_condition = tf.constant(True, dtype=tf.bool) if self.par.train.filter_on_prn is None else tf.equal(x['prn'], par.train.filter_on_prn)
        alert_condition = tf.constant(True, dtype=tf.bool) if self.par.train.filter_on_alert is None else tf.equal(x['alert'], par.train.filter_on_alert)
        cosine_condition = tf.constant(True, dtype=tf.bool) if self.par.train.filter_on_cosine is None else tf.greater(x['cosine'], x['m_cosine'] * (1 + tf.constant(par.train.filter_on_cosine, dtype=tf.float32)))
        combined_conditions = tf.logical_and(tf.logical_and(tf.logical_and(reuters_condition, prn_condition), alert_condition), cosine_condition)
        return combined_conditions

    @tf.autograph.experimental.do_not_convert
    def extract_input_and_label(self, x):
        return (x['vec'], tf.where(x[self.par.train.abny] >= 0, 1, 0))

    @tf.autograph.experimental.do_not_convert
    def extract_with_id(self, x):
        return (x['vec'], tf.where(x[self.par.train.abny] >= 0, 1, 0), x['id'], x['date'], x['permno'])

    @tf.autograph.experimental.do_not_convert
    def load_dataset(self,data_id, tfrecord_files, batch_size, start_year, end_year, return_id_too=False, shuffle=False):
        dataset = tf.data.TFRecordDataset(tfrecord_files)

        # Parse the dataset using the provided function
        dataset = dataset.map(self.parse_tfrecord)


        # Convert start_year and end_year to TensorFlow constants
        const_start_year = tf.constant(start_year, dtype=tf.int32)
        const_end_year = tf.constant(end_year, dtype=tf.int32)

        # Filter the dataset based on years
        dataset = dataset.filter(lambda x: self.filter_start_year(x, const_start_year))
        dataset = dataset.filter(lambda x: self.filter_end_year(x, const_end_year))

        # If this particular dataset must be filtered, we apply the filter at the tfrecords level.
        if self.par.train.apply_filter is not None:
            if data_id in self.par.train.apply_filter:
                dataset = dataset.filter(lambda x: self.filter_sample_based_on_par(x))

        # Extract input x (vec_last) and label (sign of abret)
        if return_id_too:
            dataset = dataset.map(self.extract_with_id)
        else:
            dataset = dataset.map(self.extract_input_and_label)


        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)  # You can adjust the buffer size as needed

        # Batch the dataset
        dataset = dataset.batch(batch_size)
        return dataset

    def train_model(self, tr_data, val_data, reg_to_use):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor=self.par.train.monitor_loss, patience=self.par.train.patience, restore_best_weights=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.par.train.adam_rate)  # Using AMSGrad variant
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(self.input_dim,), kernel_regularizer=reg_to_use)
        ])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        # Train the model with early stopping
        history = model.fit(tr_data, validation_data=val_data, epochs=self.par.train.max_epoch, callbacks=[early_stop])
        if val_data is not None:
            # Get the best validation loss for this regularizer
            assert 'val_loss' in history.history.keys(), 'Validation set empty, problem with data splitting most likely. '
            min_val_loss = min(history.history['val_loss'])
            return model, min_val_loss
        else:
            return model

    def compute_parameters_for_normalisation(self):
        mean_vec_last, variance_vec_last = self.compute_mean_variance(self.train_dataset)
        self.norm_params = {}
        self.norm_params['mean'] = mean_vec_last
        self.norm_params['var'] = variance_vec_last
        print('Parameters for normalisation estimated on train sample', flush=True)

    def def_create_the_datasets(self):

        path_with_records = self.par.get_training_dir()
        if (socket.gethostname() == '3330L-214940-M') & (self.par.train.sanity_check is None):
            path_with_records = path_with_records[:-1] + '_debug/'

        end_val = self.par.grid.year_id - 1
        start_val = end_val - self.par.train.T_val + 1
        start_train = self.par.grid.year_id - self.par.train.T_train
        end_train = start_val - 1
        start_test = self.par.grid.year_id
        end_test = self.par.grid.year_id - 1 + self.par.train.testing_window
        tfrecord_files = [os.path.join(path_with_records, f) for f in os.listdir(path_with_records) if '.tfrecord' in f]
        self.train_dataset = self.load_dataset('train',tfrecord_files, self.par.train.batch_size, start_train, end_train, shuffle=True)
        self.val_dataset = self.load_dataset('val',tfrecord_files, self.par.train.batch_size, start_val, end_val)
        self.test_dataset = self.load_dataset('test',tfrecord_files, self.par.train.batch_size, start_test, end_test)
        self.test_dataset_with_id = self.load_dataset('test',tfrecord_files, self.par.train.batch_size, start_test, end_test, return_id_too=True)

        sample_batch = next(iter(self.train_dataset.take(1)))
        self.input_dim = sample_batch[0].shape[1]

    def train_to_find_hyperparams(self):
        best_reg = None
        best_loss = float('inf')
        best_reg_value = None
        for reg_value in self.par.train.shrinkage_list:
            if self.par.train.l1_ratio[0] == 1.0:
                reg = tf.keras.regularizers.l2(reg_value)
            if self.par.train.l1_ratio[0] == 0.0:
                reg = tf.keras.regularizers.l1(reg_value)
            if self.par.train.l1_ratio[0] == 0.5:
                reg = tf.keras.regularizers.l1_l2(reg_value, reg_value)
            model, min_val_loss = self.train_model(tr_data=self.train_dataset, val_data=self.val_dataset, reg_to_use=reg)
            model.evaluate(self.val_dataset)
            model.evaluate(self.train_dataset)

            if min_val_loss < best_loss:
                best_loss = min_val_loss
                best_reg = reg
                best_reg_value = reg_value
            self.best_hyper = best_reg
            self.best_hyper_value = best_reg_value

        print('Selected best penalisation', best_reg_value, flush=True)
    def train_on_val_and_train_with_best_hyper(self):
        print('Start the final training (train+val)', flush=True)
        # Combine train and validation datasets
        combined_dataset = self.train_dataset.concatenate(self.val_dataset)
        # Train the model using the best hyperparameters found
        self.model = self.train_model(tr_data=combined_dataset, val_data=None, reg_to_use=self.best_hyper)

    def get_prediction_on_test_sample(self):
        @tf.autograph.experimental.do_not_convert
        def extract_features_for_prediction(x, y, ids, dates, tickers):
            return x

        @tf.autograph.experimental.do_not_convert
        def extract_features_and_labels(x, y, ids, dates, tickers):
            return x, y

        # Now use this function in your map operation
        predictions = self.model.predict(self.test_dataset_with_id.map(extract_features_for_prediction))

        true_labels = np.concatenate([y_batch.numpy() for _, y_batch, _, _, _ in self.test_dataset_with_id], axis=0)
        ids = np.concatenate([id_batch.numpy().astype(str) for _, _, id_batch, _, _ in self.test_dataset_with_id], axis=0)
        tickers = np.concatenate([ticker_batch.numpy().astype(int) for _, _, _, _, ticker_batch in self.test_dataset_with_id], axis=0)
        dates = np.concatenate([date_batch.numpy().astype(str) for _, _, _, date_batch, _ in self.test_dataset_with_id], axis=0)

        # 3. Compute the accuracy
        predicted_labels = (predictions > 0.5).astype(int).flatten()

        # Saving the results with corresponding IDs, dates, and tickers
        df = pd.DataFrame({
            'id': ids,
            'date': dates,
            'ticker': tickers,
            'y_true': true_labels,
            'y_pred': predicted_labels,
            'y_pred_prb': predictions.flatten()
        })
        df['accuracy'] = df['y_pred'] == df['y_true']

        _, accuracy_evaluate = self.model.evaluate(self.test_dataset_with_id.map(extract_features_and_labels))
        print('####### SANITY CHECK')
        print("Accuracy from .evaluate:", accuracy_evaluate)
        print("Accuracy from .df:", df['accuracy'].mean().round(6), flush=True)

        return df


if __name__ == '__main__':
    args = didi.parse()
    par = get_main_experiments(args.a, train_gpu=args.cpu == 0)

    start = time.time()

    # if socket.gethostname() == '3330L-214940-M':
    #     par.train.max_epoch = 1
    # else:
    #     par.train.max_epoch = 1
    #     par.train.T_train = 2
    #     par.train.T_val = 1
    #
    temp_save_dir = par.get_res_dir()
    print(temp_save_dir, flush=True)
    already_processed = os.listdir(temp_save_dir)
    save_name = f'{par.grid.year_id}.p'
    if save_name in already_processed:
        print(f'Already processed {save_name}', flush=True)
    else:
        trainer = PipelineTrainer(par)
        trainer.def_create_the_datasets()
        if par.train.norm == Normalisation.ZSCORE:
            trainer.compute_parameters_for_normalisation()
        # train to find which penalisaiton to use
        trainer.train_to_find_hyperparams()
        trainer.train_on_val_and_train_with_best_hyper()
        end = time.time()
        print('Ran it all in ', np.round((end - start) / 60, 5), 'min', flush=True)
        df = trainer.get_prediction_on_test_sample()
        df.to_pickle(temp_save_dir + save_name)

        print(df, flush=True)
        print('saved to', temp_save_dir + save_name, flush=True)
        print('We used',trainer.best_hyper_value)
