########## Direct Data input in model.fit version
import sys
print(sys.executable)
from F_modules import *
#from F_functions import *
from F_plotting import *
from F_utils import *
from F_models import *

# %%     init paths parameters
Save_pictures_path,local_dataset_path,Keras_models_path = F_init_paths()
Record_data_flag='read_data_time_domain2'

# Load pretrained CNN weights=0,train CNN=1
flag_train_model = 1
flag_single_gpu=0
if flag_single_gpu==0:
    import horovod.tensorflow.keras as hvd
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
#  Start logging to file
log_save_const = F_calculate_log_number(Save_pictures_path, 'log', '')
Save_pictures_path = Save_pictures_path + '/log' + str(log_save_const)
F_create_folder(Save_pictures_path)
logname = '/log' + str(log_save_const) + '.txt'
logpath = Save_pictures_path
f = open(logpath + logname, 'w')
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)
print('Writing log to ' + logpath + logname)
T1 = datetime.datetime.now()
print(T1)
print('Record_data_flag=', Record_data_flag)

# %%   dataset loading
dataset_to_load_name='dataset'
dataset_to_load_name='dataset_vl_gen'
# dataset_to_load_name='dataset_4'
data_path=local_dataset_path+dataset_to_load_name
Nn=-1
# Nn=10000
# Nn=30
if os.path.exists(data_path):
    print('Start loading of dataset ' + data_path)       
    files_all=fnmatch.filter(os.listdir(data_path),'*.npz')
    files=sorted(files_all)
    N=len(files)
    print('Files number=',N)
    data=np.load(data_path+'/'+files[0],allow_pickle=True)
    dx=data['dx']
    dz=dx
    data.close()
    if flag_train_model==0:
        Nn=30
        train_frac = 0.6
        val_frac = 0.4
    else:
        train_frac= 0.9
        val_frac= 0.1
    if Nn>N or Nn==-1:
        N=N
    else:
        N=Nn
    ################# dataset splitting, no shuffling
    # Train_models = np.arange(0,N * train_frac, dtype=int)   
    # Valid_models = np.setdiff1d(np.arange(0, N, 1), Train_models)
    ################# dataset splitting, shuffling
    all_models=np.arange(0,N, dtype=int).tolist()
    Train_models=random.sample(all_models,int(N*train_frac))
    Valid_models =list(set(all_models)-set(Train_models))
    Train_models=np.array(Train_models)
    Valid_models=np.array(Valid_models)
    Test_models=[]
    Test_models.append(  files_all.index('augmented_marmousi_10_it__Overthrust.npz') )
    Test_models.append(  files_all.index('augmented_marmousi_10_it__Marmousi.npz') )
    ###############################################################
    for ii in Test_models:
        if [ii] in Train_models:
            Train_models=np.delete(Train_models,np.where(ii==Train_models))
        if [ii] in Valid_models:
            Valid_models=np.delete(Valid_models,np.where(ii==Valid_models))
    ######## Overfitting case for four testing models
    list_train=[]
    for i in range(len(Train_models)):
        list_train.append(data_path+'/'+files[Train_models[i]])
    list_valid=[]
    for i in range(len(Valid_models)):
        list_valid.append(data_path+'/'+files[Valid_models[i]])
    list_test=[]
    for i in range(len(Test_models)):
        list_test.append(data_path+'/'+files_all[Test_models[i]])
    list_all=list_train+list_valid+list_test
    N=len(list_all)
else:
    print('dataset does not exist')
    exit()
print('Models for training:',  len(Train_models), ' out of ',N, 'Overall models')
print('Models for validation:',len(Valid_models), ' out of ',N, 'Overall models')
print('Models for testing:',   len(Test_models), ' out of ',N, 'Overall models')
# %%   CNN hyperparameters
batch_size=2
epochs=2
test_status=0
#   data loader for training, validation
training_generator=DataLoader(list_train,batch_size,shuffle=True,   to_fit=True)
validation_generator=DataLoader(list_valid,batch_size,shuffle=True, to_fit=True)
testing_generator=DataLoader(list_test,batch_size=1,shuffle=True,to_fit=True)

pars_h={'epochs':epochs,'batch_size':batch_size,'learning_rate':0.001,
          'training_type': 2, 'loss': 'mse',
          'patch_sz_x':training_generator.Nx_in,'patch_sz_z':training_generator.Nz_in, 
          'strides_x':training_generator.Nx_in, 'strides_z': training_generator.Nz_in,
          'reg_value': 0.0, 'Nm_to_invert': 200,
          'val_shuffle': True, 'flag_parallel': False}
save_best_only=True
# %%     callbacks for training
filename=Keras_models_path + '/pars_' + str(log_save_const) + '.hdf5'
early_stopping = EarlyStopping(monitor='val_coeff_determination', patience=14, min_delta=0.000010, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.001, factor=0.5, patience=15, min_lr=1e-8, verbose=1,mode='auto')
terminate = TerminateOnNaN()
log_dir = "logs/profile/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,profile_batch='10, 15')
checkpoint = ModelCheckpoint(Keras_models_path + '/weights_best' + str(log_save_const) + '.hdf5', verbose=1,
    monitor='val_loss',mode='min',save_best_only=save_best_only, save_weights_only=True,save_freq='epoch')
# Callbacks = [terminate,checkpoint,early_stopping]  #add early_stopping??
# Callbacks = [terminate,checkpoint,early_stopping,tensorboard_callback]  #add early_stopping??
Callbacks = [terminate,checkpoint]  #add early_stopping??
print(pars_h);  print(Callbacks)
# %%     DL MODEL initialization
input_shape=(training_generator.Nx_in,training_generator.Nz_in,1)
print('input picture shape=',input_shape)
model = auto_encoder2_deep4(input_shape)
model.summary()

flag_input_pipeline='dataloader'
flag_input_pipeline='tfdataset'

data_train=training_generator.__getdataset__()
x_train=data_train[0]
t_train=data_train[1]
data_valid=validation_generator.__getdataset__()
x_valid=data_valid[0]
t_valid=data_valid[1]
data_test=testing_generator.__getdataset__()
x_test=data_test[0]
t_test=data_test[1]
flag_do_scaling=1
if flag_do_scaling==1:
    # from sklearn.externals.joblib import dump,load
    from joblib import dump,load
    if flag_train_model==1:
        x_train_,scaler_x=scaling_data(x_train)
        x_valid_=transforming_data(x_valid,scaler_x)
        x_test_=transforming_data(x_test,scaler_x)
        t_train_,scaler_t=scaling_data(t_train)
        t_valid_=transforming_data(t_valid,scaler_t)
        t_test_=transforming_data(t_test,scaler_t)
        dump(scaler_x,Keras_models_path+'/scaler_x_'+str(log_save_const)+'.bin',compress=True)
        dump(scaler_t,Keras_models_path+'/scaler_t_'+str(log_save_const)+'.bin',compress=True)
    else:
        scaler_x=load(Keras_models_path+'/scaler_x_'+str(Model_to_load_const)+'.bin')
        scaler_t=load(Keras_models_path+'/scaler_t_'+str(Model_to_load_const)+'.bin')
        x_train_=transforming_data(x_train,scaler_x)
        x_valid_=transforming_data(x_valid,scaler_x)
        x_test_=transforming_data(x_test,scaler_x)
        t_train_=transforming_data(t_train,scaler_t)
        t_valid_=transforming_data(t_valid,scaler_t)
        t_test_=transforming_data(t_test,scaler_t)
else:
    x_train_=x_train
    x_valid_=x_valid
    x_test_=x_test
    t_train_=t_train
    t_valid_=t_valid
    t_test_=t_test
       
if flag_train_model == True:
    metrics=['MAPE',coeff_determination]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_,t_train_))
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid_,t_valid_))
    train_dataset=train_dataset.shuffle(Nn+2000).batch(batch_size)
    valid_dataset=valid_dataset.shuffle(Nn+2000).batch(batch_size)
    if flag_single_gpu==1:
        a=1
    else:
        #########multi gpu-version
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)
        # if gpus:
        #     tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        # Optimizer
        scaled_lr = pars_h['learning_rate'] * hvd.size()
        opt = tf.optimizers.Nadam(scaled_lr)
        opt = hvd.DistributedOptimizer(opt)
        # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
        # uses hvd.DistributedOptimizer() to compute gradients.
        model.compile(loss='mse',
                    optimizer=opt,
                    # metrics=['accuracy'],
                    metrics=metrics,
                    experimental_run_tf_function=False)
        callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        # Horovod: average metrics among workers at the end of every epoch.
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),
        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1),]

        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if hvd.rank() == 0:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(Save_pictures_path+'/checkpoint-{epoch}.h5'))
        # Horovod: write logs on worker 0.
        verbose = 1 if hvd.rank() == 0 else 0
        # Train the model.
        # Horovod: adjust number of steps based on number of GPUs.
        # model.fit(dataset, steps_per_epoch=500 // hvd.size(), callbacks=callbacks, epochs=24, verbose=verbose)
        print('start of training')
        T3 = datetime.datetime.now()
        history=model.fit(x=train_dataset,validation_data=valid_dataset,steps_per_epoch=500 // hvd.size(), callbacks=callbacks, epochs=pars_h['epochs'], verbose=verbose)
        T4 = datetime.datetime.now()
        print('Training time', T4 - T3)
        # model.fit(x=train_dataset,
        #     validation_data=valid_dataset,
        #     epochs=pars_h['epochs'],
        #     verbose=2,shuffle=True,callbacks=callbacks)
    history = history.history
    print(history)
    F_save_history_to_file(history,Keras_models_path,log_save_const)
    Model_to_load_const=log_save_const
else:
    fname=Keras_models_path + '/weights_best' + str(Model_to_load_const)+'.hdf5';print(fname)
    model.load_weights(fname,by_name=1)
    history = F_load_history_from_file(Keras_models_path,Model_to_load_const)
    predict_using_model=1

print('predictions')
if flag_do_scaling==1:
    t_predicted_train_ =model.predict(x_train_,verbose=1)
    t_predicted_valid_ =model.predict(x_valid_,verbose=1)
    t_predicted_test_ = model.predict(x_test_,verbose=1)
    t_predicted_train=transforming_data_inverse(t_predicted_train_,scaler_t)
    t_predicted_valid=transforming_data_inverse(t_predicted_valid_,scaler_t)
    t_predicted_test=transforming_data_inverse(t_predicted_test_,scaler_t)
else:
    t_predicted_train_ =model.predict(x_train_,verbose=1)
    t_predicted_valid_ =model.predict(x_valid_,verbose=1)
    t_predicted_test_ = model.predict(x_test_,verbose=1)
    t_predicted_train=t_predicted_train_
    t_predicted_valid=t_predicted_valid_
    t_predicted_test=t_predicted_test_
