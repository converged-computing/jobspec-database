"""

"""
# -- Built-in modules -- #
import os

# -- Third-party modules -- #
import random
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from dotenv import load_dotenv
from tensorflow import keras
from keras import models, layers
from keras import backend as K
from keras_tuner.engine import trial as trial_module

from init import OPTIONS

load_dotenv()  # take environment variables from .env.

project_name = os.getenv('PROJ_NAME')

print(tf.__version__, flush=True)
root_dir = os.getenv('OUTPUT_DIR')
result_model_dir = f'{root_dir}/model_results/{project_name}'
tensorboard_dir = f'{root_dir}/tensorboard/{project_name}'
search_results_dir = f'{root_dir}/search_results'


# Charts in the dataset
CHARTS = ['SIC']
path_to_data = os.getenv('PATH_TO_DATA')


def get_scene_lists(scene_name_orig: bool = False):
    """
    Get training and validation train lists. Removes duplicates and invalid scenes.

    Parameters
    ----------
    scene_name_orig : bool, optional
        Should the scene name have original names or _pro.nc. The default is False.

    Returns
    -------
    train_list : List[str]
        List of scene names used for training.
    validate_list : List[str]
        List of scene names used for validation.
    """
    # Check file names from .env file
    train_files = os.getenv('TRAIN_FILES')
    if train_files is not None:
        val_files = os.getenv('VAL_FILES')
        train_files = train_files.split('\n')
        val_files = val_files.split('\n')
        return train_files, val_files

    list_path = OPTIONS['project_path'] + '/datalists'

    train_list = np.loadtxt(list_path + '/train_alice.txt', dtype=str)
    validate_list = set(np.loadtxt(list_path + '/validate_alice.txt', dtype=str))
    # - Add extra validation scenes
    if 'validate_list_extra.txt' in os.listdir(list_path):
        validate_list |= set(np.loadtxt(list_path + '/validate_list_extra.txt', dtype=str))
        validate_list = list(validate_list)

    # [:15] in the file name corresponds to YYYYMMDDThhmmss, which is the Sentinel-1 acquisition.
    # timestamp. The files have been named YYYYMMDDThhmmss_pro - pro implying processed.
    suffix = '_pro.nc'
    if scene_name_orig is False:
        train_list = [file[:15] + suffix for file in train_list]
        validate_list = [file[:15] + suffix for file in validate_list]

    # List of the invalid scenes
    if 'invalid_list.txt' in os.listdir(list_path):
        invalid_list = np.loadtxt(list_path + '/invalid_list.txt', dtype=str)

        if scene_name_orig is False:
            invalid_list = [file[:15] + suffix for file in invalid_list]

        # Compares scenes in invalid_list and train_list, discards invalid_scenes
        train_list = [file for file in train_list if file not in invalid_list]

    # Select scenes in train_list not occurring in validate_list
    train_list = [file for file in train_list if file not in validate_list]
    train_list.sort()
    validate_list.sort()

    return train_list, validate_list

train_list, val_list = get_scene_lists()
train_list = [elem[0:15] for elem in train_list]
val_list = [elem[0:15] for elem in val_list]


# Sea Ice Concentration (SIC) code to class conversion lookup table.
SIC_LOOKUP = {
    'polygon_idx': 0,  # Index of polygon number.
    'total_sic_idx': 1,  # Total Sea Ice Concentration Index, CT.
    'sic_partial_idx': [2, 5, 8],  # Partial SIC polygon code index. CA, CB, CC.
    0: 0,
    1: 0,
    2: 0,
    55: 0,
    10: 1,  # 10 %
    20: 2,  # 20 %
    30: 3,  # 30 %
    40: 4,  # 40 %
    50: 5,  # 50 %
    60: 6,  # 60 %
    70: 7,  # 70 %
    80: 8,  # 80 %
    90: 9,  # 90 %
    91: 10,  # 100 %
    92: 10,  # Fast ice
    'mask': 255,
    'n_classes': 13,
    # 'n_classes': 12,
    # 255: 11  # Added by HvM
}

# Names of the SIC classes.
SIC_GROUPS = {
    0: 0,
    1: 10,
    2: 20,
    3: 30,
    4: 40,
    5: 50,
    6: 60,
    7: 70,
    8: 80,
    9: 90,
    10: 100,
    11: 'unknown'  # Added
}

SCENE_VARIABLES = [
    # -- Sentinel-1 variables -- #
    'sar_primary',
    'sar_secondary',
    # 'nersc_sar_primary',
    # 'nersc_sar_secondary',
]

ICE_STRINGS = {
    'SIC': 'Sea Ice Concentration [%]',
    'SOD': 'Stage of Development',
    'FLOE': 'Floe Size'
}

NUM_CLASSES = len(SIC_GROUPS)

SAR_VARIABLES = [variable for variable in SCENE_VARIABLES if 'sar' in variable or 'map' in variable]
FULL_VARIABLES = np.hstack((CHARTS, SAR_VARIABLES))
print(FULL_VARIABLES, flush=True)

NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
MAX_TRIALS = int(os.getenv('MAX_TRIALS'))
N_TRAIN_PATCHES = int(os.getenv('N_TRAIN_PATCHES'))
N_VAL_PATCHES = int(os.getenv('N_VAL_PATCHES'))


def calculate_patches(rows, cols, patch_size):
    '''Calulates the start position of the row and column for the patches and returns this in a tuple'''

    n_patches_rows = 1 + ((rows - 1) // patch_size)
    n_patches_cols = 1 + ((cols - 1) // patch_size)

    latest_patch_row_start = rows - patch_size
    latest_patch_col_start = cols - patch_size

    if latest_patch_row_start < 0 or latest_patch_col_start < 0:
        return []

    shift_value_rows = 0
    if n_patches_rows > 1:
        shift_value_rows = latest_patch_row_start // (n_patches_rows - 1)

    shift_value_cols = 0
    if n_patches_cols > 1:
        shift_value_cols = latest_patch_col_start // (n_patches_cols - 1)

    result = []
    for row in range(n_patches_rows):
        row_start = row * shift_value_rows
        for col in range(n_patches_cols):
            col_start = col * shift_value_cols
            patch = (row_start, col_start)
            result.append(patch)

    return result


def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)

    return x


def downsample_block(x, n_filters, dropout):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(dropout)(p)

    return f, p


def upsample_block(x, conv_features, n_filters, dropout):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(dropout)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x


class R2_Score(tf.keras.metrics.Metric):

    def __init__(self, mean, name="sparse_r_squared", **kwargs):
        super(R2_Score, self).__init__(name=name, **kwargs)
        self.mean = mean
        self.sum_of_rss = self.add_weight(name="sum_of_rss", initializer="zeros")
        self.sum_of_tss = self.add_weight(name="sum_of_tss", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        print('\nUPDATE+STATE', y_true.shape, y_pred.shape, flush=True)

        # From softmax output to labels
        y_pred_labels = tf.argmax(y_pred, -1, output_type=tf.int32)
        y_pred_labels = tf.expand_dims(y_pred_labels, -1)

        y_true = tf.reshape(y_true, [-1])  # flatten
        y_pred_labels = tf.reshape(y_pred_labels, [-1])

        # mask, all elements not eq 11
        mask = tf.math.not_equal(y_true, 11)

        y_true = tf.boolean_mask(y_true, mask)
        y_pred_labels = tf.boolean_mask(y_pred_labels, mask)

        y_true = K.cast(y_true, dtype=tf.int32)
        rss = K.sum(K.square(y_true - y_pred_labels))

        y_true = K.cast(y_true, dtype=tf.float32)

        tss = K.sum(K.square(y_true - self.mean))
        rss = K.cast(rss, dtype=tf.float32)

        self.sum_of_rss.assign_add(rss)
        self.sum_of_tss.assign_add(tss)

    def result(self):
        print('\nGET RESULT', flush=True)
        return 1 - self.sum_of_rss / self.sum_of_tss


def set_shapes_factory(patch_size):
    def set_shapes(x, y):
        x.set_shape((patch_size, patch_size, 2))
        y.set_shape((patch_size, patch_size, 1))
        return x, y

    return set_shapes


def set_shapes_with_weights_factory(patch_size):
    def set_shapes(x, y, weights):
        x.set_shape((patch_size, patch_size, 2))
        y.set_shape((patch_size, patch_size, 1))
        weights.set_shape((patch_size, patch_size, 1))
        return x, y, weights

    return set_shapes


def extract_patch(x, y, patch, patch_size):
    row_start, col_start = patch
    row_end = row_start + patch_size
    col_end = col_start + patch_size
    x1 = x[row_start:row_end, col_start:col_end, :]
    y1 = y[row_start:row_end, col_start:col_end]

    # for all pixels with 0.0 for both channels, we assign unknown to corresponding y value
    # To fix map errors
    invalid_x_values = x1 == 0.0
    invalid_pixels_mask = np.sum(invalid_x_values, axis=2) > 1
    y1[invalid_pixels_mask] = 11

    y1[y1 == 255] = 11

    # - Discard patches with too many meaningless pixels (optional) - 10% pixels with data at least needed.
    if (y1 != 11).sum() < (patch_size * patch_size) / 10:
        return None, None

    y1 = np.expand_dims(y1, axis=-1)

    x1 = np.expand_dims(x1, axis=0)
    y1 = np.expand_dims(y1, axis=0)
    return x1, y1


def load_files_py(file_name_tensor, data_dir_tensor, patch_size_tensor, load_data_randomly_tensor, noise_reduction_tensor, shuffle_tensor):
    '''Returns a batch of pathes. Each patch is a 2D npy array'''

    file_name = str(file_name_tensor.numpy(), 'utf-8')
    data_dir = str(data_dir_tensor.numpy(), 'utf-8')
    noise_reduction = str(noise_reduction_tensor.numpy(), 'utf-8')
    patch_size = patch_size_tensor.numpy()
    load_data_randomly = load_data_randomly_tensor.numpy()
    shuffle = shuffle_tensor.numpy()

    file_path = os.path.join(data_dir, file_name)
    x = np.load(file_path + f'-{noise_reduction}-x.npy')
    y = np.load(file_path + '-y.npy')

    rows, cols = y.shape

    patches = calculate_patches(rows, cols, patch_size)
    if shuffle:
        random.shuffle(patches)

    x0 = np.zeros((0, patch_size, patch_size, 2), dtype=np.float32)
    y0 = np.zeros((0, patch_size, patch_size, 1), dtype=np.uint8)

    x_patches = [x0]
    y_patches = [y0]

    for patch in patches:
        x1, y1 = extract_patch(x, y, patch, patch_size)
        if x1 is not None:
            x_patches.append(x1)
            y_patches.append(y1)

    if load_data_randomly:
        n_required_patches = len(x_patches)
        x_patches = [x0]
        y_patches = [y0]

        tries = 0
        while len(x_patches) < n_required_patches and tries < 1000:
            row_start = np.random.randint(rows - patch_size + 1)
            col_start = np.random.randint(cols - patch_size + 1)

            x1, y1 = extract_patch(x, y, (row_start, col_start), patch_size)
            if x1 is not None:
                tries = 0
                x_patches.append(x1)
                y_patches.append(y1)
            else:
                tries += 1
                if tries >= 20:
                    print("\nTries is now: ", tries, "!!!!\n", flush=True)
                if tries == 1000:
                    print("\n\n*****\n*****\nCOULD NOT FIND PATCH AFTER 1000 TRIES\n*****\n*****\n\n", flush=True)

    xxx = np.concatenate(x_patches, axis=0)
    yyy = np.concatenate(y_patches, axis=0)
    return xxx, yyy


def patch_loader_factory(data_dir, patch_size, load_data_randomly, noise_reduction, shuffle):
    '''Returns a function which can load patches'''

    def patch_loader(file_name):
        return tf.py_function(load_files_py, inp=[file_name, data_dir, patch_size, load_data_randomly, noise_reduction, shuffle],
                              Tout=(
                                  tf.TensorSpec(shape=(None, patch_size, patch_size, 2), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None, patch_size, patch_size, 1), dtype=tf.uint8)),
                              )
    return patch_loader


class WeightsAdder:
    '''The WeightsAdder class is used to add weights to the loss of the model, this way certain classes can have 
        different weights applied to them'''

    def __init__(self, weights):
        class_weights = tf.constant(weights)
        self.class_weights = class_weights / tf.reduce_sum(class_weights)

    def add_sample_weights(self, image, label):
        '''Create an image of `sample_weights` by using the label at each pixel as an index into the `class weights`'''

        sample_weights = tf.gather(self.class_weights, indices=tf.cast(label, tf.int32))

        return image, label, sample_weights


def get_data_set(data_dir, file_list, patch_size, n_samples=None, load_data_randomly=False, noise_reduction='sar', apply_weights='None', shuffle=False):
    '''Retrieves a dataset and sets the weights for the dataset if needed'''

    patch_loader = patch_loader_factory(data_dir, patch_size, load_data_randomly, noise_reduction, shuffle)
    set_shapes = set_shapes_factory(patch_size)
    set_shapes_with_weights = set_shapes_with_weights_factory(patch_size)
    ds = tf.data.Dataset.from_tensor_slices(file_list)

    if shuffle:
        ds = ds.shuffle(10000, reshuffle_each_iteration=True)
    ds = ds.map(patch_loader, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.unbatch()

    if apply_weights == 'Ignore_11':
        weights_adder = WeightsAdder([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        ds = ds.map(weights_adder.add_sample_weights, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(set_shapes_with_weights, num_parallel_calls=tf.data.AUTOTUNE)
    elif apply_weights == 'Custom_weights':
        weights_adder = WeightsAdder([0.039, 1.413, 0.907, 0.925, 1.089, 1.401, 1.233, 1.154, 0.702, 0.369, 0.099, 0])
        ds = ds.map(weights_adder.add_sample_weights, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(set_shapes_with_weights, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)

    if n_samples is not None:
        ds = ds.take(N_TRAIN_PATCHES)
    return ds


def calc_mean():
    '''Calculates the mean of a the dataset'''

    patch_size = 768
    batch_size = 16

    val_dataset = get_data_set(path_to_data, val_list, patch_size, N_VAL_PATCHES, shuffle=False)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.unbatch()

    n_sum = 0
    val_sum = 0
    for d in val_dataset:
        y_true = d[1]
        y1_flattened = tf.reshape(y_true, [-1])

        # mask, all elements not eq 11
        mask = tf.math.not_equal(y1_flattened, 11)
        y1_flattened = tf.boolean_mask(y1_flattened, mask)

        n = len(y1_flattened)
        val = np.sum(y1_flattened)
        n_sum += n
        val_sum += val
        print('* n_sum', n, n_sum)

    print('n_sum', n_sum)
    print('val_sum', val_sum)

    mean = val_sum / n_sum
    print('mean', mean)
    return mean


class MyHyperModel(kt.HyperModel):
    '''This class can build and fit a U-Net neural network'''

    def build(self, hp):
        '''This function builds and returns a custom U-Net neural network given a set of hyperparameters'''

        image_size = (None, None)
        # image_size=(PATCH_SIZE, PATCH_SIZE)
        n_channels = len(SAR_VARIABLES)
        inputs = keras.Input(shape=(image_size[0], image_size[1], n_channels))

        num_layers = hp.Int(name='num_layers', min_value=4, max_value=8, step=1)
        learning_rate = hp.Choice(name='learning_rate', values=[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1])
        dropout = hp.Choice(name='dropout', values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        max_layer_size = hp.Choice(name='max_layer_size', values=[32, 64, 128])
        hp.Choice(name='patch_size', values=[768])
        hp.Choice(name='batch_size', values=[16])
        hp.Choice(name='load_data_randomly', values=[False, True])
        hp.Choice(name='noise_reduction', values=['sar', 'nersc'])
        hp.Choice(name='apply_weights', values=['None', 'Ignore_11', 'Custom_weights'])

        base_filter = [16, 32]
        if max_layer_size == 128:
            base_filter.append(64)
        extra_filters = [max_layer_size for n in range(num_layers - len(base_filter))]
        filter_values = base_filter + extra_filters

        # downsample
        down_layers = []
        next_input = inputs
        for l in range(num_layers):
            layer, output = downsample_block(next_input, filter_values[l], dropout)
            down_layers.append((layer, l))
            next_input = output

        # 5 - bottleneck
        bottleneck = double_conv_block(next_input, filter_values[-1])

        # upsample
        next_input = bottleneck
        for conv in reversed(down_layers):
            output = upsample_block(next_input, conv[0], filter_values[conv[1]], dropout)
            next_input = output

        # outputs
        outputs = layers.Conv2D(NUM_CLASSES, 1, padding="same", activation="softmax")(next_input)

        # unet model with Keras Functional API
        unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        mean = calc_mean()
        metric = R2_Score(mean)
        unet_model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[metric, "accuracy"],
        )

        return unet_model


class CustomBayesianSearch(kt.BayesianOptimization):
    '''A custom Bayesian search algorithm based on the KerasTuner Bayesian search.'''

    def search(self, *fit_args, **fit_kwargs):
        """Performs a search for best hyperparameter configuations.

        Args:
            *fit_args: Positional arguments that should be passed to
              `run_trial`, for example the training and validation data.
            **fit_kwargs: Keyword arguments that should be passed to
              `run_trial`, for example the training and validation data.
        """
        if "verbose" in fit_kwargs:
            self._display.verbose = fit_kwargs.get("verbose")
        self.on_search_begin()
        while True:
            self.pre_create_trial()

            trial = self.oracle.create_trial(self.tuner_id)
            if trial.status == trial_module.TrialStatus.STOPPED:
                # Oracle triggered exit.
                tf.get_logger().info("Oracle triggered exit")
                break
            if trial.status == trial_module.TrialStatus.IDLE:
                # Oracle is calculating, resend request.
                continue
            self.on_trial_begin(trial)
            self._try_run_and_update_trial(trial, *fit_args, **fit_kwargs)
            self.on_trial_end(trial)
            one_trail_worker_mode = os.environ.get("KERASTUNER_ONE_TRIAL_WORKER_MODE", "0")
            print('##### TRIAL END', flush=True)
            if one_trail_worker_mode == '1':
                print('##### STOP WORKER', flush=True)
                break  # Stop worker
        self.on_search_end()




tuner = CustomBayesianSearch(
    hypermodel=MyHyperModel(),
    objective=kt.Objective("val_sparse_r_squared", direction="max"),
    max_trials=MAX_TRIALS,
    distribution_strategy=tf.distribute.MirroredStrategy(),
    directory=search_results_dir,
    project_name=project_name,
    max_retries_per_trial=1,
    overwrite=False,
)

print('loading done...')

tuner.search_space_summary()

tuner.oracle.reload()

tuner.results_summary()

best_trials = tuner.oracle.get_best_trials(50)
print('Number of best trials:', len(best_trials))
for trial in best_trials:
    best_hps = trial.hyperparameters
    nl = best_hps.get('num_layers')
    lr = best_hps.get('learning_rate')
    ps = best_hps.get('patch_size')
    bs = best_hps.get('batch_size')
    d = best_hps.get('dropout')
    mls = best_hps.get('max_layer_size')
    ldr = best_hps.get('load_data_randomly')
    nr = best_hps.get('noise_reduction')
    aw = best_hps.get('apply_weights')
    score = trial.score
    best_step = trial.best_step
    status = trial.status
    message = trial.message

    print(f' - trial: {trial.trial_id}  -- num_layers: {nl} - rate: {lr}  - patch_size: {ps}  - batch_size: {bs}  - dropout: {d}  - layer_size: {mls}  - load_randomly: {ldr}  - noise_reduction: {nr}  - weights: {aw}', flush=True)
    print(f'   (score: {score}, best_step: {best_step}, status: {status}, msg: {message})', flush=True)


print('')
print('')
all_trials = [t for t in tuner.oracle.trials.values()]
print('All trials', len(all_trials))
for trial in all_trials:
    score = trial.score
    best_step = trial.best_step
    status = trial.status
    message = trial.message
    print(f' - trial: {trial.trial_id}  -- score: {score}, best_step: {best_step}, status: {status}, msg: {message})', flush=True)

print('')
print('')

best_models = tuner.get_best_models(num_models=1)

unet_model = best_models[0]

unet_model.summary()

print("StandAlone or CHIEF, Saving model in dir:", result_model_dir)
unet_model.save(result_model_dir)



