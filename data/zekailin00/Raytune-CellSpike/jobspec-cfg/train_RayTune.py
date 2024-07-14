#!/usr/bin/env python3
import sys,os,time
sys.path.append(os.path.abspath("toolbox"))
from Plotter_CellSpike import Plotter_CellSpike
from Deep_CellSpike import Deep_CellSpike
from Util_IOfunc import write_yaml

import tensorflow as tf
Lpd=tf.config.list_physical_devices('GPU')
print('GPU info (train_Raytune)')
print('Lpd, devCnt=',len(Lpd), Lpd)
#gpus-per-task * nodes

import argparse

# get_parser copied from train_CellSpike.py. This will allow us to specify the same arguments as when we run train_CellSpike.py

def get_parser():
    '''This is the same function (same name) from train_CellSpike.py. No modifications needed for now.'''

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--restoreFile', dest='restoreFile', default="searcher_state.pkl",
                        help="name of searcher state file")
    
    parser.add_argument('--restorePath', dest='restorePath', default=False,
                        help="restore previous Ray Tune Training")
    
    parser.add_argument("-n","--nodes",help="running on CPU or GPU nodes",
        default='CPU')
    
    parser.add_argument("--numCPU",help="number of CPUs for each trial",
        default=30)
    parser.add_argument("--numGPU",help="numuber of GPUs for each trial",
        default=1)
    
    parser.add_argument('--rayResult', dest='rayResult', default='./ray_results',
        help="the output directory of raytune")
    parser.add_argument('--numHparams', dest='numHparams', default='5',
        help="the number of Raytune Samples")

        
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "--noHorovod", dest='useHorovod',  action='store_false', default=True, help="disable Horovod to run on 1 node on all CPUs")
    parser.add_argument("--designPath", default='./',help="path to hpar-model definition")

    parser.add_argument('--design', dest='modelDesign', default='smt190927_b_ontra3', help=" model design of the network")
    parser.add_argument('--venue', dest='formatVenue', default='prod', choices=['prod','poster'],help=" output quality and layout")
    parser.add_argument("--cellName", type=str, default=None, help="cell shortName , alternative for 1-cell training")
    
    parser.add_argument("--seedWeights",default=None, help="seed weights only, after model is created")

    parser.add_argument("-d","--dataPath",help="path to input",
        default='/global/homes/b/balewski/prjn/neuronBBP-pack8kHzRam/probe_3prB8kHz/ontra3/etype_8inhib_v1') 
    
    parser.add_argument("--probeType",default='8inhib157c_3prB8kHz',  help="data partition")
    parser.add_argument("--numFeature",default=None, type=int, help="if defined, reduces num of input probes.")
   
    parser.add_argument("-o","--outPath", default='out',help="output path for plots and tables")

    parser.add_argument( "-X","--noXterm", dest='noXterm', action='store_true', default=False,  help="disable X-term for batch mode")

    parser.add_argument("-e", "--epochs", type=int, default=5, dest='goalEpochs', help="training  epochs assuming localSamples*numRanks per epoch")
    parser.add_argument("-b", "--localBS", type=int, default=None,  help="training local  batch_size")
    parser.add_argument( "--localSamples", type=int, required=True,  help="samples per worker, it defines 1 epoch")
    parser.add_argument("--initLRfactor", type=float, default=1., help="scaling factor for initial LRdrop")
    
    parser.add_argument("--dropFrac", type=float, default=None, help="drop fraction at all FC layers, default=None, set by hpar")
    parser.add_argument("--earlyStop", type=int, dest='earlyStopPatience', default=20, help="early stop:  epochs w/o improvement (aka patience), 0=off")
    parser.add_argument( "--checkPt", dest='checkPtOn', action='store_true',default=True,help="enable check points for weights")
    
    parser.add_argument( "--reduceLR", dest='reduceLRPatience', type=int, default=5,help="reduce learning at plateau, patience")

    parser.add_argument("-j","--jobId", default=None, help="optional, aux info to be stored w/ summary")
    parser.add_argument('--maxConcurrent', type = int, default=None, dest = 'max_concurrent',
                        help="max concurrent trials set by ConcurrencyLimit")

    args = parser.parse_args()
    args.train_loss_EOE=False #True # 2nd loss computation at the end of each epoch
    args.shuffle_data=True # use False only for debugging

    args.prjName='cellSpike'
    args.dataPath+='/'
    args.outPath+='/'

    if not args.useHorovod:
        for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    return args


# Imports the TuneReportCheckpointCallback class, which will handle checkpointing and reporting for us.

import ray
from ray import tune, init
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.keras import TuneReportCheckpointCallback
from ray.tune.suggest import ConcurrencyLimiter

from ray.tune.integration.horovod import DistributedTrainableCreator

import numpy as np

import os, time, datetime
import warnings
import socket  # for hostname
import copy
import hashlib, json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

startT0 = time.time()
from Util_IOfunc import write_yaml, read_yaml
from Util_CellSpike import MyLearningTracker, MyEpochEndLoss

from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint , ReduceLROnPlateau
from tensorflow.python.keras.layers import Dense, Dropout,  Input, Conv1D,MaxPool1D,Flatten, Lambda, BatchNormalization
from  tensorflow.python.keras.layers.advanced_activations import LeakyReLU

# pick one of this data streamers:
#from InpGenDisc_CellSpike import  CellSpike_input_generator
#from InpGenRam_CellSpike import  CellSpike_input_generator

import tensorflow as tf

import numpy as np
import h5py

from numpy import linalg as LA  # for eigen vals
from pprint import pprint



class RayTune_CellSpike(Deep_CellSpike):



    @classmethod #........................
    def trainer(cls, config, args):
        print('Cnst:train')
        obj=cls(**vars(args))
        # initialize Horovod - if requested
        if obj.useHorovod:
            obj.config_horovod()
            if obj.verb:
                for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
        else:
            obj.myRank=0
            obj.numRanks=1
            obj.hvd=None
            print('Horovod disabled')

        if obj.verb:
            print('Cnst:train, myRank=',obj.myRank)
            print('deep-libs imported TF ver:',tf.__version__,' elaT=%.1f sec,'%(time.time() - startT0))
            
            print('GPU info (Deep_CellSpike)')
            Lpd=tf.config.list_physical_devices('GPU')
            print('Lpd, devCnt=',len(Lpd), Lpd)
            
        obj.read_metaInp(obj.dataPath+obj.metaF)
        obj.train_hirD={'acc': [],'loss': [],'lr': [],'val_acc': [],'val_loss': []}
        obj.sumRec={}
        
        obj.hparams = config


        # overwrite some hpar if provided from command line
        if obj.dropFrac!=None:  
            obj.hparams['fc_block']['dropFrac']=obj.dropFrac
        
        if obj.localBS!=None:
            obj.hparams['train_conf']['localBS']=obj.localBS

        if obj.numFeature!=None:
            obj.hparams['inpShape'][1]=obj.numFeature

        LRconf=obj.hparams['train_conf']['LRconf']
        LRconf['init']*=obj.initLRfactor
        if obj.verb: print('use initLR=%.3g'%LRconf['init'])
            
        # sanity checks
        assert obj.hparams['fc_block']['dropFrac']>=0
        assert obj.hparams['train_conf']['localBS']>=1
        assert obj.hparams['inpShape'][1]>0
        
        return obj



    def train_model(self):
        '''
        ---- This method overrides the one from Deep_CellSpike. It is almost exactly identical
        ---- except the callbacks_list now contains a
        ---- TuneReportCheckpointCallback() object The TuneReportCheckpointCallback
        ---- class handles reporting metrics and checkpoints. The metric it reports is validation loss.

        '''
        if self.verb==0: print('train silently, myRank=',self.myRank)
        hpar=self.hparams
        callbacks_list = [TuneReportCheckpointCallback(metrics = ['val_loss'])]

        if self.useHorovod:
            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            callbacks_list.append(self.hvd.callbacks.BroadcastGlobalVariablesCallback(0))
            # Note: This callback must be in the list before the ReduceLROnPlateau,
            # TensorBoard or other metrics-based callbacks.
            callbacks_list.append(self.hvd.callbacks.MetricAverageCallback())
            #Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` before
            # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.

            mutliAgent=hpar['train_conf']['multiAgent']
            if mutliAgent['warmup_epochs']>0 :
                callbacks_list.append(self.hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=mutliAgent['warmup_epochs'], verbose=self.verb))
                if self.verb: print('added LearningRateWarmupCallback(%d epochs)'%mutliAgent['warmup_epochs'])
        
        lrCb=MyLearningTracker()
        callbacks_list.append(lrCb)

        trlsCb=None
        if self.train_loss_EOE:
            print('enable  end-epoch true train-loss as callBack')
            assert self.cellName==None  # not tested
            genConf=copy.deepcopy(self.sumRec['inpGen']['train'])
            # we need much less stats for the EOE loss:
            genConf['numLocalSamples']//=8 # needs less data
            genConf['name']='EOF'+genConf['name']

            inpGen=CellSpike_input_generator(genConf,verb=1)
            trlsCb=MyEpochEndLoss(inpGen)
            callbacks_list.append(trlsCb)

        if self.checkPtOn and self.myRank==0:
            outF5w=self.outPath+'/'+self.prjName+'.weights_best.h5'
            chkPer=1
            ckpt=ModelCheckpoint(outF5w, save_best_only=True, save_weights_only=True, verbose=1, period=chkPer,monitor='val_loss')
            callbacks_list.append(ckpt)
            if self.verb: print('enabled ModelCheckpoint, save_freq=',chkPer)

        LRconf=hpar['train_conf']['LRconf']
        if LRconf['patience']>0:
            [pati,fact]=LRconf['patience'],LRconf['reduceFactor']
            redu_lr = ReduceLROnPlateau(monitor='val_loss', factor=fact, patience=pati, min_lr=0.0, verbose=self.verb,min_delta=LRconf['min_delta'])
            callbacks_list.append(redu_lr)
            if self.verb: print('enabled ReduceLROnPlateau, patience=%d, factor =%.2f'%(pati,fact))

        if self.earlyStopPatience>0:
            earlyStop=EarlyStopping(monitor='val_loss', patience=self.earlyStopPatience, verbose=self.verb, min_delta=LRconf['min_delta'])
            callbacks_list.append(earlyStop)
            if self.verb: print('enabled EarlyStopping, patience=',self.earlyStopPatience)

        #pprint(hpar)
        if self.verb: print('\nTrain_model  goalEpochs=%d'%(self.goalEpochs),'  modelDesign=', self.modelDesign,'localBS=',hpar['train_conf']['localBS'],'globBS=',hpar['train_conf']['localBS']*self.numRanks)

        fitVerb=1   # prints live:  [=========>....] - ETA: xx s 
        if self.verb==2: fitVerb=1
        if self.verb==0: fitVerb=2  # prints 1-line summary at epoch end

        if self.numRanks>1:  # change the logic
            if self.verb :
                fitVerb=2  
            else:
              fitVerb=0 # keras is silent  

        startTm = time.time()
        hir=self.model.fit(  # TF-2.1
            self.inpGenD['train'],callbacks=callbacks_list, 
            epochs=self.goalEpochs, max_queue_size=10, 
            workers=1, use_multiprocessing=False,
            shuffle=self.shuffle_data, verbose=fitVerb,
            validation_data=self.inpGenD['val']
        )
        fitTime=time.time() - startTm
        hir=hir.history
        totEpochs=len(hir['loss'])
        earlyStopOccured=totEpochs<self.goalEpochs
        
        #print('hir keys',hir.keys(),lrCb.hir)
        for obs in hir:
            rec=[ float(x) for x in hir[obs] ]
            self.train_hirD[obs].extend(rec)

        # this is a hack, 'lr' is returned by fit only when --reduceLr is used
        if 'lr' not in  hir:
            self.train_hirD['lr'].extend(lrCb.hir)

        if trlsCb: # add train-loss for Kris
            nn=len(self.train_hirD['loss'])
            self.train_hirD['train_loss']=trlsCb.hir[-nn:]

        self.train_hirD['stepTimeHist']=self.inpGenD['train'].stepTime
                    
        #report performance for the last epoch
        lossT=self.train_hirD['loss'][-1]
        lossV=self.train_hirD['val_loss'][-1]
        lossVbest=min(self.train_hirD['val_loss'])

        if self.verb:
            print('end-hpar:')
            pprint(hpar) 
            print('\n End Val Loss=%s:%.3f, best:%.3f'%( hpar['train_conf']['lossName'],lossV,lossVbest),', %d totEpochs, fit=%.1f min, earlyStop=%r'%(totEpochs,fitTime/60.,earlyStopOccured))
        self.train_sec=fitTime

        xxV=np.array(self.train_hirD['loss'][-5:])
        trainLoss_avr_last5=-1
        if xxV.shape[0]>3: trainLoss_avr_last5=xxV.mean()

        # add info to summary
        rec={}
        rec['earlyStopOccured']=int(earlyStopOccured)
        rec['fitTime_min']=fitTime/60.
        rec['totEpochs']=totEpochs
        rec['trainLoss_avr_last5']=float(trainLoss_avr_last5)
        rec['train_loss']=float(lossT)
        rec['val_loss']=float(lossV)
        rec['steps_per_epoch']=self.inpGenD['train'].__len__()
        rec['state']='model_trained'
        rec['rank']=self.myRank
        rec['num_ranks']=self.numRanks
        rec['num_open_files']=len(self.inpGenD['train'].conf['cellList'])
        self.sumRec.update(rec)
        

args=get_parser()

def training_initialization():
    """
    The parent function count the number of times a model is created
    which then creates a folder for the output of each model
    """
    
    model_path = args.outPath
    print("current work dir = " + os.getcwd())
    print("All output of Deep_CellSpike model will be saved to: " + str(model_path))


    def training_function(config, checkpoint_dir = None):
        
        """
        
        # Removed due to being incompatible with horovod rank
        
        config['myID'] = 'id1' + ('%.11f'%np.random.uniform())[2:]
        config["id"] = "id_base_ontra3_HyperOpt" + config['myID']
        config.pop("myID")
        """
        
        # Hashing the config dictionary into a 8-digit ID
        trialID = int(hashlib.sha1(json.dumps(config).encode("utf-8")).hexdigest(), 16) % (10 ** 8)
        config["id"] = args.rayResult.split('/')[-1]+"-{date:%Y\%m\%d_%H:%M}".format(date=datetime.datetime.now())+"-ID"+str(trialID)
        print("Training Initialized - " + config["id"])
        
        args.outPath = model_path + "/" + config["id"] + "/"
        try:
            if not os.path.exists(args.outPath):
                    os.makedirs(args.outPath)
        except:
            print("The folder has already been created.")
            
        config['conv_kernel'] = int(config['conv_kernel'])
        config['pool_len'] = int(config['pool_len'])
        
        conv_filter = []
        if not config["conv_filter"]:
            cf_num_layers = int(config["cf_num_layers"])
            filter_size = int(np.exp(config["filter_1_pre"]))
            conv_filter.append(filter_size)
            config.pop("filter_1_pre")
            for i in range(2, cf_num_layers + 1):
                multiplier = int(config[f"filter_{i}"])
                filter_size *= multiplier
                conv_filter.append(filter_size)
                config.pop(f"filter_{i}")
            for i in range(cf_num_layers + 1, 8):
                config.pop(f"filter_{i}")
            config["conv_filter"] = conv_filter
            config.pop("cf_num_layers")
        
        
        if not config["fc_dims"]:
            fc_dims = []
            fc_num_layers = int(config["fc_num_layers"])
            fc_size = int(np.exp(config["fc_1_pre"]))
            conv_filter.append(fc_size)
            config.pop("fc_1_pre")
            for i in range(2, fc_num_layers + 1):
                multiplier2 = config[f"fc_{i}"]
                fc_size *= multiplier2
                fc_dims.insert(0, min(512, fc_size))
                config.pop(f"fc_{i}")
            for i in range(fc_num_layers + 1, 8):
                config.pop(f"fc_{i}")
            config["fc_dims"] = fc_dims
            config.pop("fc_num_layers")
        
        if not config["optimizer"]:
            optimizer = [config["optName"], 0.001, 1.1e-7]
            config["optimizer"] = optimizer
            config.pop("optName")
        
        if not config["batch_size"]:
            batch_size = 1<<int(config["batch_size_j"])
            config["batch_size"] = batch_size
            config.pop("batch_size_j")
            
        #NEW HPAR GROUPING
        
        config["cnn_block"] = {}
        config["fc_block"] = {}
        config["train_conf"] = {}
        
        #block inits
        cnn_block = config["cnn_block"]
        fc_block = config["fc_block"]
        train_conf = config["train_conf"]
        
        #convolutional block
        cnn_block["filters"] = config["conv_filter"]
        config.pop("conv_filter")
        cnn_block["kernel"] = config["conv_kernel"]
        config.pop("conv_kernel")
        cnn_block["pool_size"] = config["pool_len"]
        config.pop("pool_len")
        
        #fully connected block
        fc_block["dropFrac"] = np.exp(config["dropFrac_pre"])
        config.pop("dropFrac_pre")
        fc_block["units"] = config["fc_dims"]
        config.pop("fc_dims")
        
        #training block
        train_conf["LRconf"] = {"init": np.exp(config["initLR_pre"]), "min_delta": np.exp(config["min_deltaLR_pre"]), "patience": 8, 
                                "reduceFactor": np.exp(config["reduceLR_pre"])} 
        config.pop("initLR_pre")
        config.pop("min_deltaLR_pre")
        config.pop("reduceLR_pre")
        train_conf["lossName"] = config["lossName"]
        config.pop("lossName")
        train_conf["multiAgent"] = {"warmup_epochs": 0}
        # to-do: LearningRateWarmupCallback is having some issues, so set this to 0 for now
        train_conf["optName"] = config["optimizer"][0]
        config.pop("optimizer")
        train_conf["localBS"] = config["batch_size"]
        config.pop("batch_size")
        
        


        #NEW HPAR
        config["inpShape"] = [1600, 3]
        config["outShape"] = 19
        
            
        print("DEBUG: ", config)
        print("training func starts")
        print("current work dir = " + os.getcwd())
        deep=RayTune_CellSpike.trainer(config, args)
        print("trainer constructed")
        
        
        
        if deep.myRank==0:
            plot=Plotter_CellSpike(args,deep.metaD )
        deep.init_genetors() # before building model, so data dims are known
        print("init_genetors gets called")

        try:
            deep.build_model()
        except:
            print('M: deep.build_model failed') #probably HPAR were pathological
            raise Exception('Trial failed due to bad hyper parameters')
            #exit(0)

        if deep.myRank==0 and args.goalEpochs >10: deep.save_model_full() # just to get binary dump of the graph
            
        if args.seedWeights=='same' :  args.seedWeights=args.outPath
        if args.seedWeights:
            deep.load_weights_only(path=args.seedWeights)

        if deep.myRank==0:
            sumF=deep.outPath+deep.prjName+'.sum_train.yaml'
            write_yaml(deep.sumRec, sumF) # to be able to predict while training continus
        
        
        
        deep.train_model()
        
        """
        # does not work with Ray Horovod integration, remove exit(0)
        
        if deep.myRank>0: exit(0)
        
        deep.save_model_full()

        try:
            plot.train_history(deep,figId=10)
        except:
            deep.sumRec['plots']='failed'
            print('M: plot.train_history failed')

        write_yaml(deep.sumRec, sumF)

        plot.display_all('train')
        """
        
        if deep.myRank==0: 
        
            deep.save_model_full()

            try:
                plot.train_history(deep,figId=10)
            except:
                deep.sumRec['plots']='failed'
                print('M: plot.train_history failed')

            write_yaml(deep.sumRec, sumF)

            plot.display_all('train')
        
        
        

    return training_function

#custom sampler functions for search space. The logic for these functions came from genHPar_CellSpike.py

# sampling conv_filter
def get_conv_filter(spec):
    numLyr=np.random.randint(3,8)
    filt1=int(np.random.choice([8,16,32,64,128]))
    filtL=[filt1]
    for i in range(1,numLyr):
        fact=np.random.choice([1,2,4])
        filt1=int(min(256,filt1*fact))
        filtL.append(filt1)

    return filtL

# sampling fc_dims
def get_fc_dims(spec):
    numLyr=np.random.randint(2,8)
    nOut = 31
    filt1=nOut*np.random.choice([1,2,4,8,16])
    #print('oo',nOut,filt1)
    filtL=[int(filt1)]
    for i in range(1,numLyr):
        fact=np.random.choice([1,2,4])
        filt1=int(min(512,filt1*fact))
        filtL.insert(0,filt1)

    return filtL

# generates batch size
def get_batch_size(spec):
    j=np.random.randint(5,8)
    bs=1<<j
    return bs

#generates LR factor
def get_reduceLR(spec):
    xx=np.random.uniform(0.2,0.8)
    lrReduce=xx*xx
    return lrReduce

#form opt list
def get_opt(spec):
    optName = str(np.random.choice(['adam','nadam']))
    return [optName, 0.001, 1.1e-7]



#UNCOMMENT WHEN USING SLURM FILES


print("Connecting to Ray head @ "+os.environ["ip_head"])
#init(address=os.environ["ip_head"])
ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0], _redis_password=os.environ["redis_password"])
print("Connected to Ray")


if args.nodes == "GPU":
    # Using raytune on a Slurm cluster
    print('GPU info (read from ray)')
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print(ray.cluster_resources())



'''
#mutation space (for pbt). NOTE: tune.sample_from is not allowed in the mutation space, so for now I will not put them here.

mutation_config = {'conv_kernel' : tune.randint(3, 10),
          'conv_repeat' : tune.randint(3, 10),
          'pool_len' : tune.randint(1, 4),
          'dropFrac' : tune.choice([0.01, 0.02, 0.05, 0.10]),
          'lossName' : tune.choice(['mse', 'mae']),
          }

#mutation custom explore functions

def mutation_custom_explore(config):
    config['conv_filter'] = tune.sample_from(get_conv_filter)
    config['fc_dims'] = tune.sample_from(get_fc_dims)
    config['optimizer'] = tune.sample_from(get_opt)
    config['batch_size'] = tune.sample_from(get_batch_size)
    config['reduceLR_factor'] = tune.sample_from(get_reduceLR)
    return config


# Instatiating our PBT object. Metric is validation loss and mode is minimize since we are trying to minimize loss.
# NOTE: The hyperparam_mutations dictionary was taken from an example and does not apply to our code. I am still figuring out what
# is the difference between hyperparam_mutations and config since they are both search spaces.

pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="val_loss",
    mode="min",
    hyperparam_mutations=mutation_config,
    custom_explore_fn=mutation_custom_explore
    )
'''
# search space
'''
config = {'myID' : tune.sample_from(lambda spec: 'id1' + ('%.11f'%np.random.uniform())[2:]),
          'inp_batch_norm': True,
          'junk1' : [1],
          'numFeature' : None,
          'conv_filter' : tune.sample_from(get_conv_filter),
          'conv_kernel' : tune.randint(3, 10),
          'conv_repeat' : tune.randint(3, 10),
          'pool_len' : tune.randint(1, 4),
          'fc_dims' : tune.sample_from(get_fc_dims),
          'lastAct' : 'tanh',
          'outAmpl' : 1.2,
          'dropFrac' : tune.choice([0.01, 0.02, 0.05, 0.10]),
          'lossName' : tune.choice(['mse', 'mae']),
          'optimizer' : tune.sample_from(get_opt),
          'batch_size' : tune.sample_from(get_batch_size),
          'reduceLR_factor' : tune.sample_from(get_reduceLR),
          'steps' : None,
          }
'''
#narrow space
config = {'conv_filter' : None,
          "cf_num_layers": tune.choice([3, 4, 5]),
          "filter_1_pre": tune.uniform(np.log(15), np.log(40)),
          "filter_2": tune.choice([2, 3]),
          "filter_3": tune.choice([2, 3]),
          "filter_4": tune.choice([2, 3]),
          "filter_5": tune.choice([2, 3]),
          "filter_6": tune.choice([2, 3]),
          "filter_7": tune.choice([2, 3]),
          'conv_kernel' : tune.choice([2, 3, 4]),
          'pool_len' : tune.choice([2, 3, 4]),
          'fc_dims' : None,
          'fc_num_layers': tune.choice([3, 4, 5]),
          'fc_1_pre': tune.uniform(np.log(40), np.log(200)),
          'fc_2': tune.choice([1, 2, 3]),
          'fc_3': tune.choice([1, 2, 3]),
          'fc_4': tune.choice([1, 2, 3]),
          'fc_5': tune.choice([1, 2, 3]),
          'fc_6': tune.choice([1, 2, 3]),
          'fc_7': tune.choice([1, 2, 3]),
          'dropFrac_pre' : tune.uniform(np.log(0.02), np.log(0.1)),
          'lossName' : tune.choice(['mse']),
          'optimizer' : None,
          'optName': tune.choice(['adam']),
          'batch_size' : None,
          'batch_size_j': tune.quniform(5, 8, 1),
          'initLR_pre' : tune.uniform(np.log(3e-4), np.log(4e-3)),
          'reduceLR_pre' : tune.uniform(np.log(0.05), np.log(0.4)),
          'min_deltaLR_pre': tune.uniform(np.log(3e-5), np.log(1e-4)),
          'steps' : None,
          'batch_norm_cnn' : tune.choice([False, True]),
          'batch_norm_flat' : tune.choice([False, True])
         }


#wide space
config_wide1 = {'conv_filter' : None,
                "cf_num_layers": tune.choice([3, 4, 5, 6]),
                "filter_1_pre": tune.uniform(np.log(10), np.log(50)),
                "filter_2": tune.choice([1, 2, 3, 4]),
                "filter_3": tune.choice([1, 2, 3, 4]),
                "filter_4": tune.choice([1, 2, 3, 4]),
                "filter_5": tune.choice([1, 2, 3, 4]),
                "filter_6": tune.choice([1, 2, 3, 4]),
                "filter_7": tune.choice([1, 2, 3, 4]),
                'conv_kernel' : tune.choice([2, 3, 4, 5, 6]),
                'pool_len' : tune.choice([2, 3, 4]),
                'fc_dims' : None,
                'fc_num_layers': tune.choice([2, 3, 4, 5]),
                'fc_1_pre': tune.uniform(np.log(20), np.log(200)),
                'fc_2': tune.choice([1, 2, 3, 4]),
                'fc_3': tune.choice([1, 2, 3, 4]),
                'fc_4': tune.choice([1, 2, 3, 4]),
                'fc_5': tune.choice([1, 2, 3, 4]),
                'fc_6': tune.choice([1, 2, 3, 4]),
                'fc_7': tune.choice([1, 2, 3, 4]),
                'dropFrac_pre' : tune.uniform(np.log(0.02), np.log(0.4)),
                'lossName' : tune.choice(['mse']),
                'optimizer' : None,
                'optName': tune.choice(['adam']),
                'batch_size' : None,
                'batch_size_j': tune.quniform(5, 12, 1),
                'initLR_pre' : tune.uniform(np.log(3e-4), np.log(1e-2)),
                'reduceLR_pre' : tune.uniform(np.log(0.05), np.log(0.4)),
                'min_deltaLR_pre': tune.uniform(np.log(3e-5), np.log(1e-4)),
                'steps' : None,
                'batch_norm_cnn' : tune.choice([False, True]),
                'batch_norm_flat' : tune.choice([False, True])
               }

# ASHA Scheduler
asha = AsyncHyperBandScheduler(time_attr='training_iteration',
                               metric="val_loss",
                               mode="min",
                               grace_period=6)

# HyperOpt
'''
initial_best_config = [{'conv_filter' : [32, 64, 64, 64, 128, 128],
                        'conv_kernel' : 7,
                        'conv_repeat' : 1,
                        'pool_len' : 3,
                        'fc_dims' : [256, 256, 240],
                        'dropFrac' : 0.01,
                        'lossName' : 'mse',
                        'optimizer' : ['nadam', 0.001, 1.1e-07],
                        'batch_size' : 128,
                        'reduceLR_factor' : 0.448298765780917
                       },
                       {'conv_filter' : [32, 64, 128, 256, 256],
                        'conv_kernel' : 3,
                        'conv_repeat' : 1,
                        'pool_len' : 4,
                        'fc_dims' : [256, 256, 256, 120, 30],
                        'dropFrac' : 0.05,
                        'lossName' : 'mae',
                        'optimizer' : ['nadam', 0.001, 1.1e-07],
                        'batch_size' : 128,
                        'reduceLR_factor' : 0.40318386178240434,
                        'batch_norm_cnn' : 1,
                        'batch_norm_flat' : 1,
                       }
                      ]


initial_best_config2 = [{'conv_filter' : None,
                        "cf_num_layers": 6.0,
                        "filter_1": 5,
                        "filter_2": 1,
                        "filter_3": 0,
                        "filter_4": 0,
                        "filter_5": 1,
                        "filter_6": 0,
                        "filter_7": 0,
                        'conv_kernel' : 7.0,
                        'conv_repeat' : 1.0,
                        'pool_len' : 3.0,
                        'fc_dims' : None,
                        'fc_num_layers': 3.0,
                        'fc_1': 8,
                        'fc_2': 0,
                        'fc_3': 0,
                        'fc_4': 0,
                        'fc_5': 0,
                        'fc_6': 0,
                        'fc_7': 0,
                        'lastAct' : 'tanh',
                        'outAmpl' : 1.2,
                        'dropFrac' : 0,
                        'lossName' : 0,
                        'optimizer' : None,
                        'optName': 1,
                        'batch_size' : None,
                        'batch_size_j': 7.0,
                        'reduceLR_factor' : None,
                        'reduceLR_x' : 0.66955116741,
                        'steps' : None,
                        'batch_norm_cnn' : 1,
                        'batch_norm_flat' : 1,
                        },
                        {'conv_filter' : None,
                        "cf_num_layers": 5.0,
                        "filter_1": 5,
                        "filter_2": 2,
                        "filter_3": 1,
                        "filter_4": 1,
                        "filter_5": 0,
                        "filter_6": 0,
                        "filter_7": 0,
                        'conv_kernel' : 3.0,
                        'conv_repeat' : 1.0,
                        'pool_len' : 4.0,
                        'fc_dims' : None,
                        'fc_num_layers': 5.0,
                        'fc_1': 5,
                        'fc_2': 2,
                        'fc_3': 1,
                        'fc_4': 0,
                        'fc_5': 0,
                        'fc_6': 0,
                        'fc_7': 0,
                        'lastAct' : 'tanh',
                        'outAmpl' : 1.2,
                        'dropFrac' : 2,
                        'lossName' : 1,
                        'optimizer' : None,
                        'optName': 1,
                        'batch_size' : None,
                        'batch_size_j': 7.0,
                        'reduceLR_factor' : None,
                        'reduceLR_x' : 0.63496760687,
                        'steps' : None,
                        'batch_norm_cnn' : 1,
                        'batch_norm_flat' : 1,
                        },
                        {'conv_filter' : None,
                        "cf_num_layers": 4.0,
                        "filter_1": 6,
                        "filter_2": 2,
                        "filter_3": 0,
                        "filter_4": 0,
                        "filter_5": 0,
                        "filter_6": 0,
                        "filter_7": 0,
                        'conv_kernel' : 6.0,
                        'conv_repeat' : 1.0,
                        'pool_len' : 3.0,
                        'fc_dims' : None,
                        'fc_num_layers': 4.0,
                        'fc_1': 7,
                        'fc_2': 0,
                        'fc_3': 2,
                        'fc_4': 0,
                        'fc_5': 0,
                        'fc_6': 0,
                        'fc_7': 0,
                        'lastAct' : 'tanh',
                        'outAmpl' : 1.2,
                        'dropFrac' : 2,
                        'lossName' : 0,
                        'optimizer' : None,
                        'optName': 0,
                        'batch_size' : None,
                        'batch_size_j': 10.0,
                        'reduceLR_factor' : None,
                        'reduceLR_x' : 0.63245,
                        'steps' : None,
                        'batch_norm_cnn' : 1,
                        'batch_norm_flat' : 1,
                        }
                       ]

'''  

initial_best_config3 = [{'conv_filter' : None,
                     "cf_num_layers": 0,
                     "filter_1_pre": 3.41,
                     "filter_2": 1,
                     "filter_3": 0,
                     "filter_4": 0,
                     "filter_5": 0,
                     "filter_6": 0,
                     "filter_7": 0,
                     'conv_kernel' : 2,
                     'pool_len' : 2,
                     'fc_dims' : None,
                     'fc_num_layers': 2,
                     'fc_1_pre': 4.853,
                     'fc_2': 1,
                     'fc_3': 1,
                     'fc_4': 0,
                     'fc_5': 0,
                     'fc_6': 0,
                     'fc_7': 0,
                     'dropFrac_pre' : -2.99573227355,
                     'lossName' : 0,
                     'optimizer' : None,
                     'optName': 0,
                     'batch_size' : None,
                     'batch_size_j': 7,
                     'initLR_pre' : -7.60090245954,
                     'reduceLR_pre' : -2.65926003693,
                     'min_deltaLR_pre': -10.4143131763,
                     'steps' : None,
                     'batch_norm_cnn' : 1,
                     'batch_norm_flat' : 1
         }
]


# Metric and mode are redundant so RayTune said to remove them from either pbt or tune.run. Num_samples is the number of trials (different hpam combinations?), #
# which is set to 10 for now. Scheduler is the PBT object instatiated above.


if args.nodes == "CPU":
    resources = {"cpu":int(args.numCPU)}
else:
    resources = {"cpu":10,"gpu":int(args.numGPU)}
    
    
#to-do: update  points_to_evaluate for ray 1.3.0
hyperopt = HyperOptSearch(metric="val_loss", mode="min", 
                          #points_to_evaluate=initial_best_config3, 
                          n_initial_points=4)

local_dir = ""
if args.restorePath:
    path = args.restorePath + "/experiment/" + args.restoreFile
    print('Restore from ' + path)
    hyperopt.restore(path)
    local_dir = args.restorePath
    print("Training logs will be saved to " + local_dir)
else:
    local_dir = args.rayResult
        
    
    
hyperopt_limited = ConcurrencyLimiter(hyperopt, max_concurrent=args.max_concurrent)

trainable = DistributedTrainableCreator(training_initialization(), num_slots=int(args.numGPU), use_gpu=True)


analysis = tune.run(
    trainable,
    #resources_per_trial=resources,
    scheduler=asha,
    search_alg=hyperopt_limited,
    num_samples=int(args.numHparams),
    config=config,
    name='experiment',
    local_dir = local_dir) 


print("Searcher_state is saved to "+local_dir + "/experiment/searcher_state.pkl")
hyperopt.save(local_dir + "/experiment/searcher_state.pkl")
