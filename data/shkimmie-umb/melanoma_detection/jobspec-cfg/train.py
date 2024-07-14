import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path
import itertools
import glob
import argparse

import melanoma as mel

CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--DB",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="+",  # 0 or more values expected => creates a list
  type=str,
  default=[],  # default if nothing is provided
)
CLI.add_argument(
  "--IMG_SIZE",
  nargs="+",
  type=int,  # any type/callable can be used here
  action="store"
)
CLI.add_argument(
  "--CLASSIFIER",
  nargs="?",
  type=str,  # any type/callable can be used here
  
)

# CLI.add_argument(
#   "--SELF_AUG",
#   nargs="?",
#   type=int,
#   default=0,

# )

CLI.add_argument(
  "--JOB_INDEX",
  nargs="?",
  type=str,  # any type/callable can be used here

)

# parse the command line
args = CLI.parse_args()
# access CLI options

check_DBs = [db.name for db in mel.DatasetType]
check_Classifiers = [c.name for c in mel.NetworkType]

assert set(args.DB).issubset(check_DBs)
assert any(args.CLASSIFIER in item for item in check_Classifiers)

print(f"DB: {args.DB}")
print(f"IMG_SIZE: {args.IMG_SIZE}")
print(f"CLASSIFIER: {args.CLASSIFIER}")
# print(f"SELF_AUG: {args.SELF_AUG}")
print(f"JOB_INDEX: {args.JOB_INDEX}")

DB = args.DB
IMG_SIZE = tuple(args.IMG_SIZE)
CLASSIFIER = args.CLASSIFIER
# SELF_AUG = args.SELF_AUG
JOB_INDEX = args.JOB_INDEX

DBname = '+'.join(DB)

rootpath = '/hpcstor6/scratch01/s/sanghyuk.kim001'
img_size = IMG_SIZE # height, width

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.callbacks import EarlyStopping
# from mel import SilentTrainingCallback as silent_callback

# optimizer1 = Adam(learning_rate=1e-5)
optimizer2 = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
red_lr= ReduceLROnPlateau(monitor='val_loss', patience=5 , verbose=1, factor=0.8)
cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = 10)

CFG = dict(
			batch_size            =  64,   # 8; 16; 32; 64; bigger batch size => moemry allocation issue
			epochs                =  180,   # 5; 10; 20;
			last_trainable_layers =   0,
			verbose               =   0,   # 0; 1
			fontsize              =  14,
			num_classes           =  2, # binary
      apply_aug             = True,

			# Images sizes
			img_height = img_size[0],   # Original: (450h, 600w)
      img_width = img_size[1],

			# Images augs
			ROTATION_RANGE        =   0.0,
			ZOOM_RANGE            =   0.0,
			HSHIFT_RANGE          =   0.0, # randomly shift images horizontally
			WSHIFT_RANGE          =   0.0, # randomly shift images vertically
			HFLIP                 = False, # randomly flip images
			VFLIP                 = False, # randomly flip images

			# Model settings
			pretrained_weights = 'imagenet',
			model_optimizer = optimizer2,
			# loss='binary_crossentropy',
			loss='categorical_crossentropy',
			metrics=['accuracy'],
			callbacks = [mel.SilentTrainingCallback(), red_lr, cb_early_stopper],

			# Postprocessing
			stopper_patience      =  0,   # 0.01; 0.05; 0.1; 0.2;
			# run_functions_eagerly = False,
            
      # save
      snapshot_path = '/raid/mpsych/MELANOMA/snapshot',
      # snapshot_path = '/hpcstor6/scratch01/s/sanghyuk.kim001/snapshot',
      experiment_noaug = f'{DBname}_noaug_{CLASSIFIER}_{IMG_SIZE[0]}h_{IMG_SIZE[1]}w_{JOB_INDEX}',
			experiment_aug = f'{DBname}_aug_{CLASSIFIER}_{IMG_SIZE[0]}h_{IMG_SIZE[1]}w_{JOB_INDEX}',
		)
# base_model = mel.CNN(CFG=CFG)
# commondata = mel.CommonData()


# Training DBs with Networks
dbpath = f'/hpcstor6/scratch01/s/sanghyuk.kim001/melanomaDB/customDB/uniform01'

ori_dbs = list(itertools.chain.from_iterable([glob.glob(f'{dbpath}/{db}_{IMG_SIZE[0]}h_{IMG_SIZE[1]}w*', recursive=True) for db in DB]))
aug_dbs = list(itertools.chain.from_iterable([glob.glob(f'{dbpath}/{db}_augmented*_{IMG_SIZE[0]}h_{IMG_SIZE[1]}w*', recursive=True) for db in DB]))

assert len(ori_dbs) == len(aug_dbs)


print("Start training augmented images")

combined_data = mel.Util.combineDatasets(aug_dbs)

# Test, Val sets must not be augmented


# Original images training (No augmentation)
model_name = CFG['experiment_aug']
# model = mel.CNN.transfer(commondata.classifierDict[CLASSIFIER], CFG)
model = mel.CommonData.model_caller(CLASSIFIER, CFG)

trainimages = combined_data['trainimages']
trainlabels = combined_data['trainlabels']
validationimages = combined_data['validationimages']
validationlabels = combined_data['validationlabels']




history_noaug = mel.CNN.fit_model(
  CFG = CFG,
  model = model,
  model_name = model_name,
  trainimages = trainimages,
  trainlabels = trainlabels,
  validationimages = validationimages,
  validationlabels = validationlabels,
)

    # visualizer = mel.Visualizer()
    # visualizer.visualize_model(model = model, plot_path=CFG['snapshot_path'], model_name = model_noaug_name)

    # visualizer.visualize_performance(
    #     model_name = model_noaug_name,
    #     plot_path=CFG['snapshot_path'],
    #     history = history_noaug
    # )

print("End of augmented training")


print("Finish")

    # visualizer.visualize_model(model = model, plot_path=CFG['snapshot_path'], model_name = model_aug_name)

    # visualizer.visualize_performance(
    #     model_name = model_aug_name,
    #     plot_path=CFG['snapshot_path'],
    #     history = history_aug
    # )