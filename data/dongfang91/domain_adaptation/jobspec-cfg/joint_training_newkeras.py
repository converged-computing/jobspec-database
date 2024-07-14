import numpy as np
import os
import h5py

np.random.seed(20170915)
from keras.layers.wrappers import Bidirectional
from keras.layers import GRU, Dropout, Embedding, Dense,Input
from keras.regularizers import l2
from keras.models import Model
from keras.callbacks import CSVLogger,ModelCheckpoint
import keras
from keras import optimizers


def load_hdf5(filename,labels):
    data = list()
    with h5py.File(filename + '.hdf5', 'r') as hf:
        print("List of datum in this file: ", hf.keys())
        for label in labels:
            x = hf.get(label)
            x_data = np.array(x)
            del x
            print("The shape of datum "+ label +": ",x_data.shape)
            data.append(x_data)
    return data

def trainging(storage,exp,sampleweights,char_x,pos_x,unicate_x,trainy_interval,trainy_operator_ex,trainy_operator_im,
                                        char_x_cv,pos_x_cv,unicate_x_cv,cv_y_interval,cv_y_operator_ex,cv_y_operator_im,batchsize,epoch_size,
                                        n_char,n_pos,n_unicate,n_vocab,reload = False,modelpath = None,embedding_size_char =64,
                                        embedding_size_pos = 48, embedding_size_unicate = 32,embedding_size_vocab =32,
                                        gru_size1 = 128,gru_size2 = 160):

    seq_length = char_x.shape[1]
    type_size_interval = trainy_interval.shape[-1]
    type_size_operator_ex = trainy_operator_ex.shape[-1]
    type_size_operator_im = trainy_operator_im.shape[-1]



    if not os.path.exists(storage):
        os.makedirs(storage)

    CharEmbedding = Embedding(output_dim=embedding_size_char, input_dim=n_char, input_length=seq_length,
                        embeddings_regularizer = l2(.01),mask_zero=True)

    PosEmbedding = Embedding(output_dim=embedding_size_pos, input_dim=n_pos, input_length=seq_length,
                        embeddings_regularizer = l2(.01),mask_zero=True)

    UnicateEmbedding = Embedding(output_dim=embedding_size_unicate, input_dim=n_unicate, input_length=seq_length,
                        embeddings_regularizer = l2(.01),mask_zero=True)

    Gru_out_1 = Bidirectional(GRU(gru_size1,return_sequences=True,input_shape=(seq_length, embedding_size_char+embedding_size_pos+embedding_size_unicate)))

    Gru_out_2 = GRU(gru_size2, return_sequences=True)

    Interval_output = Dense(type_size_interval, activation='softmax', kernel_regularizer=l2(.01), name='dense_1')

    Gru_out_3 = Bidirectional(GRU(gru_size1,return_sequences=True))

    Gru_out_4 = GRU(gru_size2, return_sequences=True)
    #
    Explicit_operator = Dense(type_size_operator_ex, activation='softmax', kernel_regularizer=l2(.01), name='dense_2')

    Gru_out_5 = Bidirectional(GRU(gru_size1,return_sequences=True))

    Gru_out_6 = GRU(gru_size2, return_sequences=True)

    Implicit_operator = Dense(type_size_operator_im, activation='softmax', kernel_regularizer=l2(.01), name='dense_3')



    char_input = Input(shape=(seq_length,), dtype='int8', name='character')

    pos_input = Input(shape=(seq_length,), dtype='int8', name='pos')

    unicate_input = Input(shape=(seq_length,), dtype='int8', name='unicate')

    char_em  = Dropout(0.25)(CharEmbedding(char_input))

    pos_em = Dropout(0.15)(PosEmbedding(pos_input))

    unicate_em = Dropout(0.15)(UnicateEmbedding(unicate_input))

    merged = keras.layers.concatenate([char_em,pos_em,unicate_em],axis=-1)

    gru_out1= Gru_out_1(merged)
    gru_out2 = Gru_out_2(gru_out1)
    interval_output  = Interval_output(gru_out2)

    gru_out3 = Gru_out_3(merged)
    gru_out4 = Gru_out_4(gru_out3)
    explicit_operator  = Explicit_operator(gru_out4)

    gru_out5 = Gru_out_5(merged)
    gru_out6 = Gru_out_6(gru_out5)
    implicit_operator  = Implicit_operator(gru_out6)

    model = Model(inputs=[char_input, pos_input,unicate_input],
                  outputs=[interval_output, explicit_operator, implicit_operator])

    sgd = optimizers.SGD(lr=0.1,decay=1e-6, momentum=0.9,nesterov=True, clipnorm=0.5)
    model.compile(optimizer='rmsprop',
                  loss={'dense_1': 'categorical_crossentropy',
                        'dense_2': 'categorical_crossentropy',
                        'dense_3': 'categorical_crossentropy'},
                  loss_weights={'dense_1': 1.0, 'dense_2': 0.75, 'dense_3': 0.5},
                  metrics=['categorical_accuracy'],
                  sample_weight_mode="temporal")

    print(model.summary())




    filepath = storage + "/weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
    csv_logger = CSVLogger('training_%s.csv' % exp)

    callbacks_list = [checkpoint, csv_logger]

    hist = model.fit(x ={'character': char_x, 'pos': pos_x,'unicate':unicate_x},
                     y={'dense_1': trainy_interval, 'dense_2': trainy_operator_ex,'dense_3': trainy_operator_im}, epochs=epoch_size,
                     batch_size=batchsize, callbacks=callbacks_list,validation_data =({'character': char_x_cv, 'pos': pos_x_cv,'unicate':unicate_x_cv},
                    {'dense_1': cv_y_interval, 'dense_2': cv_y_operator_ex, 'dense_3': cv_y_operator_im}),sample_weight=sampleweights)
    model.save(storage + '/model_result.hdf5')
    np.save(storage + '/epoch_history.npy', hist.history)



file_path = "/extra/dongfangxu9/domain_adaptation/data/data_mixed"

#file_path= "data/data_mixed"

portion = str(6)
char_x,pos_x,unicate_x = load_hdf5(file_path + "/news_colon_1_" + portion + "_train_input", ["char", "pos", "unic"])
char_x_cv,pos_x_cv,unicate_x_cv= load_hdf5(file_path + "/cvnews_train_input", ["char", "pos", "unic"])

trainy_interval,trainy_operator_ex,trainy_operator_im = load_hdf5(file_path + "/news_colon_1_" + portion + "_train_target", ["interval_softmax","explicit_operator","implicit_operator"])
cv_y_interval,cv_y_operator_ex,cv_y_operator_im = load_hdf5(file_path + "/cvnews_train_target", ["interval_softmax","explicit_operator","implicit_operator"])


n_pos = 49
n_char = 89
n_unicate = 15
n_vocab = 16
epoch_size = 800
batchsize =250



#sampleweights = list(np.load(file_path+"/news_colon_1_"+ portion  +"_sample_weights.npy"))
sampleweights = list(np.load(file_path+"/news_colon_1_"+ portion  +"_sample_weights.npy"))


path = "/xdisk/dongfangxu9/domain_adaptation/"

exp = "news_colon_1_" + portion + "_devon_news_newkeras_rmsprop_batch250"
storage = path + exp
trainging(storage,exp,sampleweights,char_x,pos_x,unicate_x,trainy_interval,trainy_operator_ex,trainy_operator_im,
                                        char_x_cv,pos_x_cv,unicate_x_cv,cv_y_interval,cv_y_operator_ex,cv_y_operator_im,batchsize,epoch_size,
                                        n_char,n_pos,n_unicate,n_vocab,reload = False,modelpath = "med_3softmax_5_29_pretrainedmodel/weights-improvement-231",embedding_size_char =128,
                                        embedding_size_pos = 32, embedding_size_unicate = 64,embedding_size_vocab =16,
                                        gru_size1 =256,gru_size2 = 150)
