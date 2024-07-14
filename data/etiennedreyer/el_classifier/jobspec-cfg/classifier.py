# IMPORT PACKAGES AND FUNCTIONS
import tensorflow as tf, tensorflow.keras.callbacks as cb
import numpy      as np, multiprocessing as mp, os, sys, h5py, pickle
from   argparse   import ArgumentParser
from   tabulate   import tabulate
from   itertools  import accumulate
from   utils      import validation, make_sample, sample_composition, apply_scaler, load_scaler
from   utils      import balance_sample, compo_matrix, class_weights, cross_valid, valid_results
from   utils      import sample_analysis, sample_weights
from   models     import multi_CNN


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--n_train'     , default =  1e6,  type = float )
parser.add_argument( '--n_valid'     , default =  1e6,  type = float )
parser.add_argument( '--batch_size'  , default =  5e3,  type = float )
parser.add_argument( '--n_epochs'    , default =  100,  type = int   )
parser.add_argument( '--n_classes'   , default =    2,  type = int   )
parser.add_argument( '--n_tracks'    , default =    5,  type = int   )
parser.add_argument( '--bkg_ratio'   , default =    2,  type = int   )
parser.add_argument( '--n_folds'     , default =    1,  type = int   )
parser.add_argument( '--n_gpus'      , default =    4,  type = int   )
parser.add_argument( '--verbose'     , default =    1,  type = int   )
parser.add_argument( '--patience'    , default =   10,  type = int   )
parser.add_argument( '--sbatch_var'  , default =    1,  type = int   )
parser.add_argument( '--l2'          , default = 1e-8,  type = float )
parser.add_argument( '--dropout'     , default = 0.05,  type = float )
parser.add_argument( '--FCN_neurons' , default = [200, 200], type = int, nargs='+')
parser.add_argument( '--weight_type' , default = 'none'              )
parser.add_argument( '--train_cuts'  , default = ''                  )
parser.add_argument( '--valid_cuts'  , default = ''                  )
parser.add_argument( '--NN_type'     , default = 'CNN'               )
parser.add_argument( '--images'      , default = 'ON'                )
parser.add_argument( '--scalars'     , default = 'ON'                )
parser.add_argument( '--resampling'  , default = 'OFF'               )
parser.add_argument( '--scaling'     , default = 'ON'                )
parser.add_argument( '--plotting'    , default = 'OFF'               )
parser.add_argument( '--metrics'     , default = 'val_accuracy'      )
parser.add_argument( '--data_file'   , default = ''                  )
parser.add_argument( '--output_dir'  , default = 'outputs'           )
parser.add_argument( '--model_in'    , default = ''                  )
parser.add_argument( '--model_out'   , default = 'model.h5'          )
parser.add_argument( '--scaler_in'   , default = 'scaler.pkl'        )
parser.add_argument( '--scaler_out'  , default = 'scaler.pkl'        )
parser.add_argument( '--results_in'  , default = ''                  )
parser.add_argument( '--results_out' , default = ''                  )
args = parser.parse_args()


# VERIFYING ARGUMENTS
for key in ['n_train', 'n_valid', 'batch_size']: vars(args)[key] = int(vars(args)[key])
if args.weight_type not in ['flattening', 'match2s', 'match2b', 'none']:
    print('\nweight_type: \"',args.weight_type,'\" not recognized, resetting it to none!!!')
    args.weight_type = 'none'
if '.h5' not in args.model_in and args.n_epochs < 1 and args.n_folds==1:
    print('\nERROR: weight file required with n_epochs < 1 -> exiting program\n'); sys.exit()


# DATAFILE
for path in list(accumulate([folder+'/' for folder in args.output_dir.split('/')])):
    try: os.mkdir(path)
    except FileExistsError: pass
data_file = '/opt/tmp/godin/el_data/2020-05-28/el_data.h5'
#data_file = '/project/def-arguinj/dgodin/el_data/2020-05-28/el_data.h5'
if args.data_file != '': data_file= args.data_file


# CNN PARAMETERS
CNN = {(56,11):{'maps':[200,200], 'kernels':[ (3,3) , (3,3) ], 'pools':[ (2,2) , (2,2) ]},
        (7,11):{'maps':[200,200], 'kernels':[(2,3,7),(2,3,1)], 'pools':[(1,1,1),(1,1,1)]},
        (5,13):{'maps':[200,200], 'kernels':[ (1,1) , (1,1) ], 'pools':[ (1,1) , (1,1) ]}}


# TRAINING VARIABLES
images    = ['em_barrel_Lr0'  , 'em_barrel_Lr1'  , 'em_barrel_Lr2'  , 'em_barrel_Lr3', 'em_barrel_Lr1_fine',
             'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3', 'tracks_image']
scalars   = ['p_Eratio', 'p_Reta'   , 'p_Rhad'     , 'p_Rphi'  , 'p_TRTPID' , 'p_numberOfSCTHits'  ,
             'p_ndof'  , 'p_dPOverP', 'p_deltaEta1', 'p_f1'    , 'p_f3'     , 'p_deltaPhiRescaled2',
             'p_weta2' , 'p_d0'     , 'p_d0Sig'    , 'p_qd0Sig', 'p_nTracks', 'p_sct_weight_charge']
#scalars  += ['p_eta'   , 'p_et_calo']
others    = ['mcChannelNumber', 'eventNumber', 'p_TruthType', 'p_iffTruth'   , 'p_TruthOrigin', 'p_LHValue',
             'p_LHTight'      , 'p_LHMedium' , 'p_LHLoose'  , 'p_ECIDSResult', 'p_eta'        , 'p_et_calo']
#others   += ['p_firstEgMotherTruthType', 'p_firstEgMotherTruthOrigin']
train_var = {'images' :images  if args.images =='ON' else [], 'tracks':[],
             'scalars':scalars if args.scalars=='ON' else []}
variables = {**train_var, 'others':others}; scalars = train_var['scalars']


# SAMPLES SIZES AND APPLIED CUTS ON PHYSICS VARIABLES
sample_size  = len(h5py.File(data_file, 'r')['eventNumber'])
args.n_train = [0, min(sample_size, args.n_train)]
args.n_valid = [args.n_train[1], min(args.n_train[1]+args.n_valid, sample_size)]
if args.n_valid[0] == args.n_valid[1]: args.n_valid = args.n_train
#args.valid_cuts += ' & (abs(sample["p_eta"] > 0.6)'
#args.valid_cuts += ' & (sample["p_et_calo"] > 4.5) & (sample["p_et_calo"] < 20)'


# OBTAINING PERFORMANCE FROM EXISTING VALIDATION RESULTS
if os.path.isfile(args.output_dir+'/'+args.results_in):
    variables = {'others':others, 'scalars':scalars, 'images':[]}
    validation(args.output_dir, args.results_in, args.plotting, args.n_valid, data_file, variables)
if args.results_in != '': sys.exit()


# MULTI-GPU DISTRIBUTION
n_gpus  = min(args.n_gpus, len(tf.config.experimental.list_physical_devices('GPU')))
devices = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']
tf.debugging.set_log_device_placement(False)
strategy = tf.distribute.MirroredStrategy(devices=devices[:n_gpus])
with strategy.scope():
    if tf.__version__ >= '2.1.0' and len(variables['images']) >= 1:
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    sample, _ = make_sample(data_file, variables, [0,1], args.n_tracks, args.n_classes)
    func_args = (args.n_classes, args.NN_type, sample, args.l2, args.dropout, CNN, args.FCN_neurons)
    model     = multi_CNN(*func_args, **train_var)
    print('\nNEURAL NETWORK ARCHITECTURE'); model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# ARGUMENTS AND VARIABLES TABLES
args.scaler_in  = args.scaler_in  if '.pkl' in args.scaler_in  else ''
args.model_in   = args.model_in   if '.h5'  in args.model_in   else ''
args.results_in = args.results_in if '.h5'  in args.results_in else ''
args.NN_type    = 'FCN' if train_var['images'] == [] else args.NN_type
args.scaling    = (args.scaling == 'ON' and scalars != [])
if args.NN_type == 'CNN':
    print('\nCNN ARCHITECTURES:')
    for shape in CNN: print(format(str(shape),'>8s')+':', str(CNN[shape]))
print('\nPROGRAM ARGUMENTS:'); print(tabulate(vars(args).items(), tablefmt='psql'))
print('\nTRAINING VARIABLES:')
headers = [          key  for key in train_var if train_var[key]!=[]]
table   = [train_var[key] for key in train_var if train_var[key]!=[]]
length  = max([len(n) for n in table])
table   = list(map(list, zip(*[n+(length-len(n))*[''] for n in table])))
print(tabulate(table, headers=headers, tablefmt='psql')); print()


# GENERATING VALIDATION SAMPLE AND LOADING PRE-TRAINED WEIGHTS
print('CLASSIFIER: loading valid sample', args.n_valid, end=' ... ', flush=True)
func_args = (data_file, variables, args.n_valid, args.n_tracks, args.n_classes, args.valid_cuts)
valid_sample, valid_labels = make_sample(*func_args)
valid_sample.update({'eta':valid_sample['p_eta'], 'pt':valid_sample['p_et_calo']})
#sample_analysis(valid_sample, valid_labels, scalars, args.scaler_in, args.output_dir); sys.exit()
if args.model_in != '':
    print('CLASSIFIER: loading pre-trained weights from', args.output_dir+'/'+args.model_in, '\n')
    model.load_weights(args.output_dir+'/'+args.model_in)
    if args.scaling: valid_sample = load_scaler(valid_sample, scalars, args.output_dir+'/'+args.scaler_in)


# TRAINING LOOP
if args.n_epochs > 0:
    print(  'CLASSIFIER: train sample:'   , format(args.n_train[1] -args.n_train[0], '8.0f'), 'e')
    print(  'CLASSIFIER: valid sample:'   , format(args.n_valid[1] -args.n_valid[0], '8.0f'), 'e')
    print('\nCLASSIFIER: using TensorFlow', tf.__version__ )
    print(  'CLASSIFIER: using'           , n_gpus, 'GPU(s)')
    print('\nCLASSIFIER: using'           , args.NN_type, 'architecture with', end=' ')
    print([group for group in train_var if train_var[group] != [ ]])
    print('\nCLASSIFIER: loading train sample', args.n_train, end=' ... ', flush=True)
    func_args = (data_file, variables, args.n_train, args.n_tracks, args.n_classes, args.train_cuts)
    train_sample, train_labels = make_sample(*func_args); sample_composition(train_sample)
    sample_weight = sample_weights(train_sample,train_labels,args.n_classes,args.weight_type,args.output_dir)
    if args.resampling == 'ON': train_sample, train_labels = balance_sample(train_sample, train_labels)
    if args.scaling:
        if args.model_in == '':
            scaler_out = args.output_dir+'/'+args.scaler_out
            train_sample, valid_sample = apply_scaler(train_sample, valid_sample, scalars, scaler_out)
        else: train_sample = load_scaler(train_sample, scalars, args.output_dir+'/'+args.scaler_in)
    compo_matrix(valid_labels, train_labels=train_labels); print()
    model_out   = args.output_dir+'/'+args.model_out
    check_point = cb.ModelCheckpoint(model_out, save_best_only =True, monitor=args.metrics, verbose=1)
    early_stop  = cb.EarlyStopping(patience=args.patience, restore_best_weights=True, monitor=args.metrics)
    training    = model.fit( train_sample, train_labels, validation_data=(valid_sample,valid_labels),
                             callbacks=[check_point,early_stop], epochs=args.n_epochs, verbose=args.verbose,
                             class_weight=class_weights(train_labels, bkg_ratio=args.bkg_ratio),
                             sample_weight=sample_weight, batch_size=max(1,n_gpus)*int(args.batch_size) )
    model.load_weights(model_out)
else: train_labels = []; training = None


# RESULTS AND PLOTTING SECTION
if args.n_folds > 1:
    valid_probs = cross_valid(valid_sample, valid_labels, scalars, model, args.output_dir, args.n_folds)
    print('MERGING ALL FOLDS AND PREDICTING CLASSES ...')
else:
    print('\nValidation sample', args.n_valid, 'class predictions:')
    valid_probs = model.predict(valid_sample, batch_size=20000, verbose=args.verbose); print()
valid_results(valid_sample, valid_labels, valid_probs, train_labels, training, args.output_dir, args.plotting)
if args.results_out != '':
    print('Saving validation results to:', args.output_dir+'/'+args.results_out, '\n')
    if args.n_folds > 1 and args.valid_cuts == '': valid_data = (valid_probs,)
    else: valid_data = ({key:valid_sample[key] for key in others}, valid_labels, valid_probs)
    pickle.dump(valid_data, open(args.output_dir+'/'+args.results_out,'wb'))
