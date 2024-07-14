"""File initiating scene variables, paths to files etc. and options used for training models."""

# -- File info -- #
__author__ = ['Andreas R. Stokholm', 'Andrzej S. Kucik']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__version__ = '0.1.15'
__date__ = '2022-01-20'

# -- Built-in modules -- #
import os
import time

# -- Third-party modules -- #
import matplotlib.pyplot as plt
import numpy as np
from functions.utils import SIC_LOOKUP, SOD_LOOKUP, FLOE_LOOKUP, get_dict_int_values

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['NCCL_DEBUG'] = 'INFO'

# Make the directory for the bins
os.makedirs('misc/scene_pro_bins/', exist_ok=True)

# Model Training options
OPTIONS = {
    # -- Path to project
    'project_path': '/home/example_path/AutoML4SeaIce',

    # -- Paths to data
    'path_to_data': '/home/example_path/seaice_source',
    'path_to_processed_data': '/home/example_path/seaice_preprocessed',
    'path_to_npy_data': '/home/example_path/seaice_npy',

    # -- Path to result directory
    'path_to_results': '/home/example_path/results',

    # -- Training options -- #
    'patch_size': 768,  # crop size for patches used in training
    'batch_size': 32,  # number of patches for each forward/backward pass
    'epochs': 2,  # number of epochs to train
    'epoch_len': 100,  # number of batches in each epoch

    # variables to train on, should be np.array
    'train_variables': np.array(('nersc_sar_primary', 'nersc_sar_secondary')),
    'num_workers': 1,  # number of threads used in the data loader
    'n_patches': 1,  # number of patches extracted per scene. Picks new scene afterwards
    'model_name': f'model_{int(time.time())}',  # name for the model if not specified in parse.
    'chart': 'SIC',  # What chart to use as reference. Available: 'SIC', 'SOD', 'FLOE'.
    'charts': ['SIC', 'SOD', 'FLOE'],  # For multi chart parameter models. Not implemented yet.

    # -- Model options -- #
    'n_classes': {  # number of total classes in the reference charts.
        'SIC': max(get_dict_int_values(SIC_LOOKUP)) + 2,
        'SOD': max(get_dict_int_values(SOD_LOOKUP)) + 2,
        'FLOE': max(get_dict_int_values(FLOE_LOOKUP)) + 2
    },
    'kernel_size': (3, 3),  # size of convolutional kernel.
    'stride_rate': (1, 1),  # convolutional striding rate.
    'dilation_rate': (1, 1),
    'padding': (1, 1),  # Number of padding pixels.
    'padding_style': 'zeros',  # Style of applied padding. e.g. zero_pad, replicate.

    # -- ResUNet/Unet specific options -- #
    'filters': [24, 32, 64, 64, 64, 64, 64, 64],

    # -- Optimizer options -- #
    'lr': .0001,  # initial learning rate
    'optimizer': 'adam',  # choice of optimizer, adam and adamW is currently implemented
    'betas': (.5, .999),
    'weight_decay': 0,  # L2 weight Decay
    'fill_threshold': 1,  # minimum number of pixels not belonging to fill_value

    # -- Data Augmentation, imgaug parameters -- #
    'shear': 10,
    'scale': .3,
    'translate': 0.3,
    'sometimes': .5,  # probability of applying between 1-4 of shear, scale, translate, and +- 45 degree rotation

    # -- Misc -- #
    'cmap_style': {  # Color map name for plt plotting.
        'SIC': 'viridis',
        'SOD': 'viridis',
        'FLOE': 'viridis'
    },
    'primary_color': 'darkred',  # Primary color for some figures, e.g. histograms.
    'secondary_color': 'black',  # Secondary color for some figures, e.g. histograms.
    'epoch_before_val': 25,  # Epochs before validating model. Speeds up training. To disable -> 0

    # -- Nan fill values for instrument and chart data. -- #
    'train_fill_value': 0,  # Nan fill value for SAR/AMSR2 training data.
    'class_fill_value': 11,
    'class_fill_values': {  # Nan fill value for class/reference data.
        'SIC': max(get_dict_int_values(SIC_LOOKUP)) + 1,  # NaN fill value for Sea Ice Concentration.
        'SOD': max(get_dict_int_values(SOD_LOOKUP)) + 1,  # NaN fill value for Stage Of Development.
        'FLOE': max(get_dict_int_values(FLOE_LOOKUP)) + 1,  # NaN fill value for floe/form of sea ice
    },
    'class_fill_weight': 0,
    'class_fill_weights': {  # Weight multiplier for mask class. Intended to be 0 or 1.
        'SIC': 0,  # Weight multiplier for SIC.
        'SID': 0,  # Weight multiplier for SOD.
        'FLOE': 0  # Weight multiplier for floe/form of sea ice.
    },
    'pixel_spacing': 80,  # Original is 40. Should be 80, 160 etc. Only used in preprocessing.
    'normalize_range': [-1, 1],  # Used for preprocessing.

    # -- vmin/vmax display values for charts -- #
    'vmin': {  # Colorbar display minimum.
        'SIC': 0,
        'SOD': 0,
        'FLOE': 0
    },
    'vmax': {  # Colorbar display maximum.
        'SIC': max(get_dict_int_values(SIC_LOOKUP)),
        'SOD': max(get_dict_int_values(SOD_LOOKUP)),
        'FLOE': max(get_dict_int_values(FLOE_LOOKUP))
    },
    'cmap': {}
}
for chart in OPTIONS['charts']:
    OPTIONS['cmap'][chart] = plt.get_cmap(OPTIONS['cmap_style'][chart], OPTIONS['n_classes'][chart] - 1)
