#Init logger
import logging
import logging.config
logging.config.fileConfig('conf/logging.conf')
logger = logging.getLogger(__file__)

#Import Mask RCNN packages
import mrcnn.model as modellib
from mrcnn import utils

#Import OrgaSegment functions
from lib import OrganoidDataset, mask_projection, average_precision, config_to_dict
import importlib

#Import other packages
import tensorflow as tf
import sys
import os
import shutil
from skimage.io import imsave, imread
from skimage.color import label2rgb 
import pandas as pd
import numpy as np
from pathlib import Path
import re

#Set Tensorflow logging
logger.info(f'Tensorflow version: {tf.VERSION}')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Check Tensorflow GPU
if tf.test.is_gpu_available():
    logger.info(f'GPU devices: {tf.config.experimental_list_devices()}')
else:
    logger.error(f'No GPUs available')
    exit(1)

#Get Job ID
job_id=sys.argv[1]

#Get config
config_path=sys.argv[2]
spec = importlib.util.spec_from_file_location('EvalConfig', config_path)
modulevar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulevar)
config = modulevar.EvalConfig()

#Model
model_path=sys.argv[3]

#Set log_dir
log_dir = None

def main():
    #Get config, display and save config
    # config = EvalConfig()
    logger.info(config.display())

    #Get data
    logger.info('Preparing data')
    data_eval = OrganoidDataset()
    data_eval.load_data(config.EVAL_DIR,
                        config.CLASSES,
                        config.IMAGE_FILTER,
                        config.MASK_FILTER,
                        config.COLOR_MODE)
    data_eval.prepare()

    #Load model
    logger.info('Loading model')
    model = modellib.MaskRCNN(mode='inference', 
                              config=config,
                              model_dir=config.MODEL_DIR)
    
    if os.path.isfile(model_path) and model_path.endswith('.h5'):
        model_name = model_path
        model.load_weights(model_path, by_name=True)
        logger.info(f'Model loaded: {model_name}')
    else:
        model_name = model.find_last()
        model.load_weights(model_name, by_name=True)
        logger.info(f'Model loaded: {model_name}')
    
    #Create output data folder
    output_path = re.sub('mask.*\.h5', f'eval/{config.EVAL_DATASET}/', model_name)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    #Update log_dir
    global log_dir
    log_dir = model.log_dir
    name = os.path.basename(log_dir)

    #Create empty data frame for results
    evaluation =  pd.DataFrame({'image': pd.Series([], dtype='str'),
                                'class_id': pd.Series([], dtype=np.double),
                                'class_name': pd.Series([], dtype='str'),
                                'threshold': pd.Series([], dtype=np.double),
                                'ap': pd.Series([], dtype=np.double),
                                'tp': pd.Series([], dtype=np.double),
                                'fp': pd.Series([], dtype=np.double),
                                'fn': pd.Series([], dtype=np.double)})

    # Compute Average Precisions based on Cellpose paper
    for i in data_eval.image_ids:
        # Load image and ground truth data
        logger.info(f'Load image {i}')
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(data_eval, config,
                                       i, use_mini_mask=False)
        # Run object detection
        logger.info(f'Run object detection for image {i}')
        results = model.detect([image], verbose=1)
        r = results[0]

        #Save  eval image, gt and mask
        logger.info(f'Save output for image {i}')
        image_name = re.search(f'^{config.EVAL_DIR}(.*){config.IMAGE_FILTER}\..*$', data_eval.info(i)['path']).group(1)
        imsave(f'{output_path}{image_name}{config.IMAGE_FILTER}.jpg', image)
        
        for class_id in list(set(gt_class_id)):
            class_name = config.CLASSES[class_id - 1]
            #Get gt per class id
            gt_indices = [i for i, u in enumerate(gt_class_id) if u == class_id] #get gt indices where label is equal to i
            gt = [gt_mask[:,:,i] for i in gt_indices] #get gt masks where label is equal to i
            gt = mask_projection(np.stack(gt, axis=-1))

            #Get prediction per class id
            p_indices = [i for i, u in enumerate(r['class_ids']) if u == class_id] #get prediction indices where label is equal to i
            scores = [r['scores'][i] for i in p_indices] #get scores where label is equal to i
            p_masks = [r['masks'][:,:,i] for i in p_indices] #get prediction masks where label is equal to i

            #Remove masks with a low score
            s_indices = [i for i, u in enumerate(scores) if u >= config.CONFIDENCE_SCORE_THRESHOLD] #get prediction indices where score is higher than thresholds
            p = [p_masks[i] for i in s_indices] #get prediction masks where score is higher than threshold
            p = mask_projection(np.stack(p, axis=-1))

            #Save gt and mask
            imsave(f'{output_path}{image_name}_gt_class-{class_id}.png', gt)
            imsave(f'{output_path}{image_name}_pred_class-{class_id}.png', p)

            #Combine image and mask and create preview
            preview_path = f'{output_path}{image_name}_preview_class-{class_id}.jpg'
            combined = label2rgb(p, imread(data_eval.info(i)['path']), bg_label = 0)
            imsave(preview_path, combined)

            # Compute AP
            ap, tp, fp, fn = average_precision(gt, p, config.AP_THRESHOLDS)
        
            #Combine information
            for t in range(config.AP_THRESHOLDS.size):
                info = {'image': data_eval.info(i)['path'],
                        'class_id': class_id,
                        'class_name': class_name,
                        'threshold': round(config.AP_THRESHOLDS[t], 2),
                        'ap': ap[t],
                        'tp': tp[t],
                        'fp': fp[t],
                        'fn': fn[t]}
                evaluation = evaluation.append(info, ignore_index=True)

    #Save results
    eval_name = re.search('^.*/(.*)\.h5', model_name).group(1) + '_evaluation.csv'
    evaluation.to_csv(output_path + eval_name, index=False)

if __name__ == "__main__":
    logger.info('Start evaluation...')
    main()
    logger.info('Evaluation completed!')
    ##Copy logging to model log dir
    shutil.copy(f'log/JobName.{job_id}.out', f'{log_dir}/JobName.{job_id}.out')
    shutil.copy(f'log/JobName.{job_id}.err', f'{log_dir}/JobName.{job_id}.err')