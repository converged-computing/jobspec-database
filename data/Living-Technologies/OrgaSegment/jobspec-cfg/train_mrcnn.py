#Init logger
import logging
import logging.config
logging.config.fileConfig('conf/logging.conf')
logger = logging.getLogger(__file__)

#Import Mask RCNN packages
import mrcnn.model as modellib

#Import OrgaSwell functions
from lib import OrganoidDataset, config_to_dict
import importlib

#Import other packages
import tensorflow as tf
import sys
import shutil
import os

#Set Tensorflow logging
logger.info(f'Tensorflow version: {tf.VERSION}')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Check Tensorflow GPU
if tf.test.is_gpu_available():
    devices = tf.config.experimental.list_physical_devices('GPU')
    logger.info(f'GPU devices: {devices}')
    for i in devices:
        logger.info(f'Current memory growth setting for device {str(i)}: {tf.config.experimental.get_memory_growth(i)}')
        logger.info('Set memory growth to TRUE')
        tf.config.experimental.set_memory_growth(i, True)
        logger.info(f'New memory growth setting for device {str(i)}: {tf.config.experimental.get_memory_growth(i)}')
else:
    logger.error(f'No GPUs available')
    exit(1)

#Get Job ID
job_id=sys.argv[1]

#Get config
config_path=sys.argv[2]
spec = importlib.util.spec_from_file_location('TrainConfig', config_path)
modulevar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulevar)
config = modulevar.TrainConfig()

#Set log_dir
log_dir = None

def main():  
    #display config
    logger.info(config.display())

    #Get data
    logger.info('Preparing data')
    data_train = OrganoidDataset()
    data_train.load_data(config.TRAIN_DIR,
                         config.CLASSES,
                         config.IMAGE_FILTER,
                         config.MASK_FILTER,
                         config.COLOR_MODE)
    data_train.prepare()

    data_val = OrganoidDataset()
    data_val.load_data(config.VAL_DIR,
                       config.CLASSES,
                       config.IMAGE_FILTER,
                       config.MASK_FILTER,
                       config.COLOR_MODE)
    data_val.prepare()

    #Compile model
    logger.info('Compiling model')
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=config.MODEL_DIR)
    model.load_weights(config.PRETRAINED_WEIGHTS,
                       by_name=True,
                       exclude=config.EXCLUDE_LAYERS)

    #Update log_dir
    global log_dir
    log_dir = model.log_dir
    name = os.path.basename(log_dir)

    ##Train model
    logger.info('Start training heads')
    model.train(data_train, data_val, 
                learning_rate=config.LEARNING_RATE,
                epochs=config.EPOCHS_HEADS,
                layers='heads',
                workers=config.WORKERS,
                use_multiprocessing=config.MULTIPROCESSING,
                class_weight=config.CLASS_WEIGHTS)

    logger.info('Start training all layers')
    model.train(data_train, data_val, 
                learning_rate=config.LEARNING_RATE,
                epochs=config.EPOCHS_ALL_LAYERS,
                layers='all',
                workers=config.WORKERS,
                use_multiprocessing=config.MULTIPROCESSING,
                class_weight=config.CLASS_WEIGHTS)

if __name__ == "__main__":
    logger.info('Start training...')
    main()
    logger.info('Training completed!')
    ##Copy logging to model log dir
    shutil.copy(f'log/JobName.{job_id}.out', f'{log_dir}/JobName.{job_id}.out')
    shutil.copy(f'log/JobName.{job_id}.err', f'{log_dir}/JobName.{job_id}.err')