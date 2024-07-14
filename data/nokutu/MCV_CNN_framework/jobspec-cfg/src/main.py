import time

import torch

from config.configuration import Configuration
from dataloader.dataloader_builder import DataLoaderBuilder
from models.model_builder import ModelBuilder
from tasks.classification_manager import ClassificationManager
from tasks.detection_manager import DetectionManager
from tasks.semantic_segmentation_manager import SemanticSegmentationManager
from utils.logger import Logger


def main():
    print('Using GPU: ', torch.cuda.get_device_name(0))
    start_time = time.time()
    # Prepare configuration
    config = Configuration()
    cf = config.load()
    # Enable log file
    logger_debug = Logger(cf.log_file_debug)

    logger_debug.write('\n ---------- Init experiment: ' + cf.exp_name + ' ---------- \n')
    # Model building
    logger_debug.write('- Building model: ' + cf.model_name + ' <--- ')
    model = ModelBuilder(cf)
    model.build()
    print(model.net)

    # Problem type
    if cf.problem_type == 'segmentation':
        problem_manager = SemanticSegmentationManager(cf, model)
    elif cf.problem_type == 'classification':
        problem_manager = ClassificationManager(cf, model)
    elif cf.problem_type == 'detection':
        problem_manager = DetectionManager(cf, model)
    else:
        raise ValueError('Unknown problem type')

    # Create dataloader builder
    dataloader = DataLoaderBuilder(cf, model)

    if cf.train:
        model.net.train()  # enable dropout modules and others
        train_time = time.time()
        logger_debug.write('\n- Reading Train dataset: ')
        dataloader.build_train()
        if (cf.valid_dataset_path is not None or (
                cf.valid_images_txt is not None and cf.valid_gt_txt is not None)) and cf.valid_samples_epoch != 0:
            logger_debug.write('\n- Reading Validation dataset: ')
            dataloader.build_valid(cf.valid_samples_epoch, cf.valid_images_txt, cf.valid_gt_txt,
                                   cf.resize_image_valid, cf.valid_batch_size)
            problem_manager.trainer.start(dataloader.train_loader, dataloader.train_set,
                                          dataloader.valid_set, dataloader.valid_loader)
        else:
            # Train without validation inside epoch
            problem_manager.trainer.start(dataloader.train_loader, dataloader.train_set)
        train_time = time.time() - train_time
        logger_debug.write('\t Train step finished: %ds ' % train_time)

    if cf.validation:
        model.net.eval()
        valid_time = time.time()
        if not cf.train:
            logger_debug.write('- Reading Validation dataset: ')
            dataloader.build_valid(cf.valid_samples, cf.valid_images_txt, cf.valid_gt_txt,
                                   cf.resize_image_valid, cf.valid_batch_size)
        else:
            # If the Dataloader for validation was used on train, only update the total number of images to take
            dataloader.valid_set.update_indexes(cf.valid_samples,
                                                valid=True)  # valid=True avoids shuffle for validation
        logger_debug.write('\n- Starting validation <---')
        problem_manager.validator.start(dataloader.valid_set, dataloader.valid_loader, 'Validation')
        valid_time = time.time() - valid_time
        logger_debug.write('\t Validation step finished: %ds ' % valid_time)

    if cf.test:
        model.net.eval()
        test_time = time.time()
        logger_debug.write('\n- Reading Test dataset: ')
        dataloader.build_valid(cf.test_samples, cf.test_images_txt, cf.test_gt_txt,
                               cf.resize_image_test, cf.test_batch_size)
        logger_debug.write('\n - Starting test <---')
        problem_manager.validator.start(dataloader.valid_set, dataloader.valid_loader, 'Test')
        test_time = time.time() - test_time
        logger_debug.write('\t Test step finished: %ds ' % test_time)

    if cf.predict_test:
        model.net.eval()
        pred_time = time.time()
        logger_debug.write('\n- Reading Prediction dataset: ')
        dataloader.build_predict()
        logger_debug.write('\n - Generating predictions <---')
        problem_manager.predictor.start(dataloader.predict_loader)
        pred_time = time.time() - pred_time
        logger_debug.write('\t Prediction step finished: %ds ' % pred_time)

    total_time = time.time() - start_time
    logger_debug.write('\n- Experiment finished: %ds ' % total_time)
    logger_debug.write('\n')


if __name__ == "__main__":
    main()
