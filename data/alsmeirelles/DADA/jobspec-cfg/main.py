#!/usr/bin/env python3
#  -*- coding: utf-8

import os
import argparse
import sys
import importlib
import multiprocessing as mp
from multiprocessing import Process

#Filter warnings
import warnings
warnings.filterwarnings('ignore')
    
#  Project imports
from Preprocessing import Preprocess
from Utils import Exitcodes, CacheManager
from Testing import TrainTest, DatasourcesTest, PredictionTest, ActiveLearningTest
from Trainers import GenericTrainer, Predictions, ALTrainer
    
#  Supported image types
img_types = ['svs', 'dicom', 'nii', 'tif', 'tiff', 'png']

def main_exec(config):
    """
    Main execution line. Dispatch processes according to parameter groups.
    Multiple processes here prevent main process from consuming too much memory.
    """

    if not os.path.isdir(config.bdir):
        os.mkdir(config.bdir)

    if not os.path.isdir(config.weights_path):
        os.mkdir(config.weights_path)
        
    if not os.path.isdir(config.model_path):
        os.mkdir(config.model_path)

    if not os.path.isdir(config.cache):
        os.mkdir(config.cache)

    if not os.path.isdir(config.logdir):
        os.mkdir(config.logdir)
        
    if config.preprocess:
        if config.img_type is None:
            imgt = img_types
        else:
            imgt = config.img_type
            
        if config.multiprocess:
            proc = Process(target=Preprocess.preprocess_data, args=(config, imgt))
            proc.start()
            proc.join()

            if proc.exitcode != Exitcodes.ALL_GOOD:
                print("System did not end well. Check logs or enhace verbosity level.")
                sys.exit(proc.exitcode)
        else:
            Preprocess.preprocess_data(config,imgt)
        
    if config.train:
        if config.multiprocess:
            ctx = mp.get_context('spawn')
            cache_m = CacheManager()
            proc = ctx.Process(target=GenericTrainer.run_training, args=(config, cache_m.getLocations()))
            proc.start()
            proc.join()

            if proc.exitcode != Exitcodes.ALL_GOOD:
                print("System did not end well. Check logs or enhace verbosity level.")
                sys.exit(proc.exitcode)
        else:
            GenericTrainer.run_training(config,None)

    if config.al:            
        if config.multiprocess:
            ctx = mp.get_context('spawn')
            cache_m = CacheManager()
            proc = ctx.Process(target=ALTrainer.run_training, args=(config, cache_m.getLocations()))
            proc.start()
            proc.join()

            if proc.exitcode != Exitcodes.ALL_GOOD:
                print("System did not end well. Check logs or enhace verbosity level.")
                sys.exit(proc.exitcode)
        else:
            ts = importlib.import_module('Trainers', config.strategy)
            getattr(ts, config.strategy).run_training(config, None)
            
    if config.pred:
        if config.multiprocess:
            ctx = mp.get_context('spawn')
            cache_m = CacheManager()
            proc = Process(target=Predictions.run_prediction, args=(config, cache_m.getLocations()))
            proc.start()
            proc.join()

            if proc.exitcode != Exitcodes.ALL_GOOD:
                print("System did not end well. Check logs or enhace verbosity level.")
                sys.exit(proc.exitcode)
        else:
            Predictions.run_prediction(config, None)
            
    if config.postproc:
        pass

    if config.runtest:
        if config.tmode == 0:
            pass
        elif config.tmode == 1:
            #  Run train test
            TrainTest.run(config)
        elif config.tmode == 2:
            DatasourcesTest.run(config)
        elif config.tmode == 3:
            PredictionTest.run(config)
        elif config.tmode == 4:
            ActiveLearningTest.run(config)

    if not (config.preprocess or config.train or config.postproc or config.pred or config.runtest):
        print("The problem begins with choice: preprocess, train, postprocess or predict")

if __name__ == "__main__":

    #  Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Convolunional Neural \
        Network for Active Learning.')

    ##  Preprocessing options
    pre_args = parser.add_argument_group('Preprocessing','Includes image format usage, tiling and normalization')
    arg_groups.append(pre_args)
    
    pre_args.add_argument('--pre', action='store_true', dest='preprocess', default=False, 
        help='Run preprocess steps')
    pre_args.add_argument('-tile', action='store_true', dest='tile', default=False, 
        help='Make tiles from input images')
    pre_args.add_argument('-tcga', action='store_true', dest='tcga', default=False, 
        help='Input is a TCGA image base.')    
    pre_args.add_argument('-presrc', dest='presrc', type=str, default='',
        help='Input image or directory of images (runs recursively)', required=False)
    pre_args.add_argument('-predst', dest='predst', type=str, default='tiles',
        help='Output tiles go to this directory. Use this option as training source.')
    pre_args.add_argument('-img_type', dest='img_type', nargs='+', type=str, 
        help='Input image types to consider (list): svs, dicom, nii.', default=None)
    pre_args.add_argument('-mag', dest='magnification', type=int, 
        help='For SVS images only, use specific magnification level.',
        choices=[2,4,8,10,20,40],default=40)
    pre_args.add_argument('-tdim', dest='tdim', nargs='+', type=int, 
        help='Tile width and heigth, optionally inform the number of channels (Use: 200 200 for SVS 50 um).', 
        default=None, metavar=('Width', 'Height'))
    pre_args.add_argument('-norm', dest='normalize', type=str, nargs='?', default=None, const='Preprocessing/target_40X.png',
        help='Normalize tiles based on reference image (given)')
    

    ##Training options
    train_args = parser.add_argument_group('Training','Common network training options')
    arg_groups.append(train_args)

    train_args.add_argument('--train', action='store_true', dest='train', default=False, 
        help='Train model')
    train_args.add_argument('-net', dest='network', type=str, default='', help='Network name which should be trained.\n \
    Check documentation for available models.')
    train_args.add_argument('-data', dest='data', type=str, help='Dataset name to train model.\n \
    Check documentation for available datasets.', default='')
    train_args.add_argument('-b', dest='batch_size', type=int, 
        help='Batch size (Default: 8).', default=8)
    train_args.add_argument('-lr', dest='learn_r', type=float, 
        help='Learning rate (Default: 0.00005).', default=0.00005)
    train_args.add_argument('-e', dest='epochs', type=int, 
        help='Number of epochs (Default: 1).', default=1)
    train_args.add_argument('-tn', action='store_true', dest='new_net',
        help='Do not use older weights file.', default=False)
    train_args.add_argument('-nsw', action='store_false', dest='save_w',
        help='Do not save intermediate weights as a callback.', default=True)
    train_args.add_argument('-tnorm', action='store_true', dest='batch_norm',
        help='Applies batch normalization during training.', default=False)
    train_args.add_argument('-aug', action='store_true', dest='augment',
        help='Applies data augmentation during training.', default=False)
    train_args.add_argument('-plw', action='store_true', dest='plw',
        help='Preload Imagenet weights after single model build.', default=False)
    train_args.add_argument('-lyf', dest='lyf', type=int, 
        help='Freeze this number of layers for training (Default=0).', default=0)
    train_args.add_argument('-wpath', dest='weights_path',
        help='Use weights file contained in path - usefull for sequential training (Default: None).',
        default='ModelWeights')
    train_args.add_argument('-split', dest='split', nargs=3, type=float, 
        help='Split data in as much as 3 sets (Default: 80%% train, 10%% validation, 10%% test). If AL experiment, test set can be defined as integer.',
        default=(0.8, 0.1, 0.1), metavar=('Train', 'Validation', 'Test'))
    train_args.add_argument('-f1', dest='f1period', type=int, 
        help='Execute F1 and ROC AUC calculations every X epochs (Default: 0).', default=0)
    train_args.add_argument('-sample', dest='sample', type=float, 
        help='Use a sample of the whole data for training (Default: 100.0%% - use floats [0.0-1.0]).',
        default=1.0)
    train_args.add_argument('-pos_rt', dest='pos_rt', type=float, 
        help='When sampling, keep a positive rate among samples (Default: not set - use floats [0.0-1.0]).',
        default=None)
    
    ##Active Learning options
    al_args = parser.add_argument_group('AL','Active Learning options')
    arg_groups.append(al_args)

    al_args.add_argument('--al', action='store_true', dest='al', default=False, 
        help='Train model')
    al_args.add_argument('-strategy', dest='strategy', type=str,
       help='Which strategy to use: ALTrainer, EnsembleTrainer, etc.', default='ALTrainer')
    al_args.add_argument('-init_train', dest='init_train', type=int, 
        help='Initial training set size (Default: 1000).', default=1000)
    al_args.add_argument('-ac_function', dest='ac_function',type=str,
       help='Acquisition function. Check documentation for available functions.', default=None)
    al_args.add_argument('-un_function', dest='un_function', type=str,
       help='Uncertainty function to be used with KM. Check documentation for available functions.', default='bayesian_varratios')
    al_args.add_argument('-ac_steps', dest='acquisition_steps', type=int, 
        help='Run active learning for this many cycles (Default: 10).', default=10)
    al_args.add_argument('-emodels', dest='emodels', type=int, 
        help='Number of ensemble submodels (Default: 3).', default=3)
    al_args.add_argument('-acquire', dest='acquire', type=int, 
        help='Acquire this many samples at each acquisition step (Default: 1000).', default=1000)
    al_args.add_argument('-dropout_steps', dest='dropout_steps', type=int, 
        help='For Bayesian CNNs, sample the network this many times (Default: 100).', default=100)
    al_args.add_argument('-bal', action='store_true', dest='balance',
        help='Balance dataset samples between classes.', default=False)
    al_args.add_argument('-sv', action='store_true', dest='save_var',
        help='Save aquisition variations/probability/clusters arrays and selected items indexes.', default=False)
    al_args.add_argument('-clusters', dest='clusters', type=int, 
        help='Number of clusters to form in similarity selections (Default 20).', default=20)
    al_args.add_argument('-recluster', dest='recluster', type=int, 
        help='Re-cluster data every X acquisitions (Default:all).', default=0)
    al_args.add_argument('-ffeat', dest='ffeat', type=str,
        help='Use a fixed pre-trained model to extract features.', default=None)
    al_args.add_argument('-pca', dest='pca', type=int, 
        help='Apply PCA to extracted features before clustering (Default: 0 (not used)).',default=0)
    al_args.add_argument('-load_train', dest='load_train', action='store_true', default=False,
        help='Use the same initial training set as produced by a previous experiment.')
    al_args.add_argument('-spool', dest='spool', type=int, 
        help='Regenerate the pool every spool acquisitions. Default = 0 (do not regenerate).', default=0)
    al_args.add_argument('-spool_f', dest='spool_f', type=str, 
        help='Use this function to regenerate - check docs. Default is None (random sample).', default=None)
    al_args.add_argument('-restore', dest='restore', action='store_true', default=False,
        help='Restore a previous experimet data and continue from where it left of.')
    al_args.add_argument('-tnet', dest='tnet', type=str, default=None, help='Target network for AL Transfer.\n \
    Check documentation for available models.')
    al_args.add_argument('-tnpred', dest='tnpred', type=int, 
        help='Train the target network and run prediction on this interval of AL iterations. Default = 0 (do not train/test).', default=0)
    al_args.add_argument('-phi', dest='phi', type=int, 
        help='Phi defines network architecture reduction. Values bigger than 0 reduce nets by 1/phi. Default = 0 (use original sizes).', default=0)
    al_args.add_argument('-tnphi', dest='tnphi', type=int, 
        help='Phi defines target network architecture reduction. Values bigger than 0 reduce nets by 1/phi. Default = 0 (use original sizes).', default=0)
    al_args.add_argument('-dye', dest='dye', action='store_true', default=False,
        help='Apply dynamic epoch count adjustment.')
    
    ##Postprocessing options
    post_args = parser.add_argument_group('Postprocessing', 'Generate bounding boxes or other operation')
    arg_groups.append(post_args)

    post_args.add_argument('--post', action='store_true', dest='postproc', default=False, 
        help='Run postprocess steps')
    post_args.add_argument('-postsrc', dest='postsrc', type=str, default='tiles',
        help='Input image or directory of images (runs recursively)')
    post_args.add_argument('-postdst', dest='postdst', type=str, default='',
        help='Output tiles to directory. If empty, output to same directory as input')
    
    ##Model selection
    model_args = parser.add_argument_group('Model')
    arg_groups.append(model_args)

    model_args.add_argument('-model_dir', dest='model_path',
        help='Save trained models in dir (Default: TrainedModels).',
        default='TrainedModels')
    
    ##Hardware configurations
    hd_args = parser.add_argument_group('Hardware')
    arg_groups.append(hd_args)

    hd_args.add_argument('-gpu', dest='gpu_count', type=int, 
        help='Number of GPUs available (Default: 0).', default=0)
    hd_args.add_argument('-cpu', dest='cpu_count', type=int, 
        help='Number of CPU cores available (Default: 1).', default=1)

    ##Runtime options
    parser.add_argument('-out', dest='bdir', type=str, default='',
        help='Base dir to store all temporary data and general output', required=True)
    parser.add_argument('-cache', dest='cache', type=str, default='cache',
        help='Keeps caches in this directory', required=False)
    parser.add_argument('-v', action='count', default=0, dest='verbose',
        help='Amount of verbosity (more \'v\'s means more verbose).')
    parser.add_argument('-i', action='store_true', dest='info', default=False, 
        help='Return general info about data input, the CNN, etc.')
    parser.add_argument('-logdir', dest='logdir', type=str, default='logs',
        help='Keep logs of current execution instance in dir.')
    parser.add_argument('-mp', action='store_true', dest='multiprocess', default=False, 
        help='[TODO] Preprocess multiple images at a time (memory consuming - multiple processes).')
    parser.add_argument('-pb', action='store_true', dest='progressbar', default=False, 
        help='Print progress bars of processing execution.')
    parser.add_argument('-k', action='store_true', dest='keepimg', default=False, 
        help='Keep loaded images in memory.')
    parser.add_argument('-d', action='store_true', dest='delay_load', default=False, 
        help='Delay the loading of images to the latest moment possible (memory efficiency).')
    parser.add_argument('-db', action='store_true', dest='debug',
        help='Runs debugging procedures.', default=False)
    
    ##  Run prediction options
    parser.add_argument('--pred', action='store_true', dest='pred', default=False, 
                        help='Runs prediction with a given model (use -net parameter).')
    parser.add_argument('-print', action='store_true', dest='print_pred', default=False, 
                        help='Prints stored prediction results.')
    parser.add_argument('-pred_size', dest='pred_size', type=int,
                        help='Limit test set size to this number of images.', default=0)
    parser.add_argument('-pconf', dest='pconf', type=float,
                        help='Confidence threshold to use in predictions (Default 0.0 for optimal).', default=0.0)
    parser.add_argument('-test_dir', dest='testdir', type=str, default=None,
                        help='Runs prediction on a different set of images stored in dir.')
    parser.add_argument('-wsi_split', dest='wsi_split', type=int,
                        help='Randomly select this number of WSIs to choose test patches from (Default 0, means use all).', default=0)
    parser.add_argument('-wsilist', dest='wsilist', type=str, nargs='+', default=None)
    parser.add_argument('-wsimax', dest='wsimax', type=float, nargs='+', help="Define % of patches of each slide to use in testing",default=None)
        
    ##  System tests
    test_args = parser.add_argument_group('Tests')
    arg_groups.append(test_args)
    
    parser.add_argument('-t', action='store_true', dest='runtest', default=False, 
        help='Run tests.')
    test_args.add_argument('-tmode', dest='tmode', type=int, 
        help='Run tests for individual subsystems: \n \
        0 - Run all tests; \n \
        1 - Run training test; \n \
        2 - Run Datasources test; \n \
        3 - Run Prediction test; \n \
        4 - Run AL test.',
       choices=[0, 1, 2, 3, 4], default=0)
    parser.add_argument('-tlocal', action='store_true', dest='local_test', default=False, 
        help='Test is local (assumes a small dataset).')
    
    config, unparsed = parser.parse_known_args()
    
    files = {
        'datatree.pik': os.path.join(config.cache, '{}-datatree.pik'.format(config.data)),
        'tcga.pik': os.path.join(config.cache, 'tcga.pik'),
        'metadata.pik': os.path.join(config.cache, '{0}-metadata.pik'.format(config.data)),
        'sampled_metadata.pik': os.path.join(config.cache, '{0}-sampled_metadata.pik'.format(config.data)),
        'testset.pik': os.path.join(config.cache, '{0}-testset.pik'.format(config.data)),
        'initial_train.pik': os.path.join(config.cache, '{0}-inittrain.pik'.format(config.data)),
        'split_ratio.pik': os.path.join(config.cache, '{0}-split_ratio.pik'.format(config.data)),
        'clusters.pik': os.path.join(config.cache, '{0}-clusters.pik'.format(config.data)),
        'data_dims.pik': os.path.join(config.cache, '{0}-data_dims.pik'.format(config.data)),
        'tiles.pik': os.path.join(config.predst, 'tiles.pik'),
        'test_pred.pik': os.path.join(config.logdir, 'test_pred.pik'),
        'cae_model.h5': os.path.join(config.model_path, 'cae_model.h5'),
        'vgg16_weights_notop.h5': os.path.join('PretrainedModels', 'vgg16_weights_notop.h5')}

    cache_m = CacheManager(locations=files)    

    config.split = tuple(config.split)
    #  Run main program
    main_exec(config)
