from collections import deque
import tensorflow as tf
import os
import numpy as np
import cv2
from utils import *
import pickle
import os
from model import *
from loss import *
from generate_data_list import generate_data_list
#from generate_data_list_aux import generate_data_list_aux
import time
from datetime import datetime
import shutil
import argparse
from vgg import *
import math

#slim = tensorflow.contrib.slim


############## arguments ##############
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="filename of train_set",default='None')
parser.add_argument("--dataset", help="filename of train_set",default='None')
parser.add_argument("--train_set", help="filename of train_set",default='/Google-Bris/train0115/train/')
parser.add_argument("--train_set_label", help="filename of train_set_label", default="/Google-Bris/train0115/train_label/")
parser.add_argument("--test_set", help="filename of train_set",default="/Google-Bris/test/")
parser.add_argument("--test_set_label", help="filename of train_set_label", default="/Google-Bris/test-gt/")

parser.add_argument("--aux", help="aux_train or not",default=False)
parser.add_argument("--aux_train_set", help="filename of aux_train_set",default="/Google-Bris/train0115/train_all/")
parser.add_argument("--aux_train_set_label", help="filename of aux_train_set_label", default="/Google-Bris/train0115/train_label/")
parser.add_argument("--new_label_update_dir", help="filename of aux_train_set_label", default="/scratch/po21/jz1585/1117/label")


parser.add_argument("--model", default='FPN_RFA_3', help="model")
parser.add_argument("--block", type=int,default=4, help="block")

parser.add_argument("--vgg_checkpoint", default='/scratch/po21/jz1585/VGG/model-Bris/', help="directory with checkpoints")

parser.add_argument("--checkpoint", default='None', help="directory with checkpoints")
parser.add_argument("--max_epochs", type=int, default=400,help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=20, help="update summaries every summary_freq steps")
parser.add_argument("--save_freq", type=int, default=50, help="save model every save_freq epochs")

parser.add_argument("--batch_size",type=int, default=8, help="number of images in batch")
parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
parser.add_argument("--decay_rate", type=float, default=0.8, help="decay_rate")
parser.add_argument("--decay_ep", type=int, default=15, help="decay_epoch")

parser.add_argument("--loss", required=True, choices=["CE","WCE","BSGCRF","BSWCE","GCRF"])
#CRF LOSS CONFIG
parser.add_argument("--crf_w", type=float, default=0.3, help="GCRF_w")
parser.add_argument("--kernels_radius", type=int, default=11, help="GCRF_kernels_radius")

parser.add_argument("--image_size", type=int, default=256)

parser.add_argument("--self_pace", default=False, help="Self_paced_learning")
parser.add_argument("--lamda", type=float, default=0.008, help="initial_lamda")
parser.add_argument("--lamda_decay_ep", type=int, default=8, help="lamda_decay_ep")
parser.add_argument("--lamda_decay_rate", type=float, default=0.95, help="lamda_decay_rate")

parser.add_argument("--label_update", default=True, help="update_labels_in_Self_Paced_learning")
parser.add_argument("--label_update_dir", default='/scratch/po21/jz1585/0118ACT/label/', help="update_labels_dir")
Config=parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"



WT_CRF_loss_config = [{'weight':0.6,'xy':6,'rgb':11},{'weight':0.1,'xy':6},{'weight':0.3,'texture':11}]
CRF_loss_config = [{'weight':0.9,'xy':6,'rgb':16},{'weight':0.1,'xy':6}]

mask = generate_kenerl_mask(Config.kernels_radius)




if Config.aux:
    TRAIN_SET_DIR_solar =  Config.data_dir + Config.train_set
    TRAIN_SET_DIR_label =  Config.data_dir + Config.train_set_label
else:
    TRAIN_SET_DIR_solar =  Config.data_dir + Config.aux_train_set
    TRAIN_SET_DIR_label =  Config.data_dir + Config.aux_train_set_label

TEST_SET_DIR_solar =  Config.data_dir + Config.test_set
TEST_SET_DIR_label =  Config.data_dir + Config.test_set_label
generate_data_list(TRAIN_SET_DIR_solar,TRAIN_SET_DIR_label,TEST_SET_DIR_solar,TEST_SET_DIR_label)



localtime=time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
ckpt_save_dir = localtime


if not os.path.exists(ckpt_save_dir):      
    os.makedirs(ckpt_save_dir) 
    os.makedirs(ckpt_save_dir+'/logs')
    os.makedirs(ckpt_save_dir+'/model')



training_config = 'model '+str(Config.model)+'\r\n'\
    +'dataset '+str(Config.dataset)+'\r\n'\
    +'block '+str(Config.block)+'\r\n'\
    +'initial_learning_rate '+str(Config.lr)+'\r\n'\
    +'image_size '+str(Config.image_size) +'\r\n'\
    +'batch_size'+str(Config.batch_size)+'\r\n'\
    +'max_epochs '+str(Config.max_epochs)+ '\r\n'\
    +'summary_freq '+str(Config.summary_freq)+ '\r\n'\
    +'save_freq '+str(Config.save_freq)+ '\r\n'\
    +'decay_ep '+str(Config.decay_ep)+ '\r\n'\
    +'decay_rate '+str(Config.decay_rate)+'\r\n' \
    +'loss '+str(Config.loss)+'\r\n' \
    +'GCRF_kernels_radius '+str(Config.kernels_radius)+'\r\n' \
    +'GCRF_w '+str(Config.crf_w)+'\r\n' \
    +'train_set '+str(Config.train_set) +'\r\n'\
    +'train_label_dir '+str(Config.train_set_label) +'\r\n'\
    +'aux_train '+str(Config.aux) +'\r\n'\
    +'aux_train_set '+str(Config.aux_train_set) +'\r\n'\
    +'aux_train_label_dir '+str(Config.aux_train_set_label) +'\r\n'\
    +'self_pace '+str(Config.self_pace)+ '\r\n'\
    +'lamda '+str(Config.lamda)+ '\r\n'\
    +'lamda_decay_ep '+str(Config.lamda_decay_ep)+'\r\n' \
    +'lamda_decay_rate '+str(Config.lamda_decay_rate)+'\r\n' \
    +'label_update '+str(Config.lamda_decay_rate)+'\r\n' \
    +'label_update_dir '+str(Config.label_update_dir)+'\r\n' \
    +'test_image_dir'+str(Config.test_set) +'\r\n'\
    +'test_label_dir'+str(Config.test_set_label) +'\r\n'\
    +'ckpt_restore_dir '+str(Config.checkpoint)+'\r\n'\
    +'CRF_loss_config_rgb' +str(CRF_loss_config[0]['rgb'])



save_config(Config.checkpoint, ckpt_save_dir,'config.txt',training_config)

try:
    with open('train_set_list.pickle', 'rb') as f:
            train_set_list = pickle.load(f)
except:
    raise EnvironmentError('Data list not existed. Please run generate_data_list.py first.')
random.shuffle(train_set_list)
train_set_queue = deque(train_set_list)
train_set_size = len(train_set_list)
#for d in train_set_list:
    #img = load_image(d[0],Config.image_size)
    #print(d[0],img.shape)
#del train_set_list

print ('Training set built. Size: '+str(train_set_size))


try:
    with open('test_set_list.pickle', 'rb') as f:
            test_set_list = pickle.load(f)
except:
    raise EnvironmentError('Data list not existed. Please run generate_data_list.py first.')
random.shuffle(test_set_list)
test_set_queue = deque(test_set_list)
test_set_size = len(test_set_list)


print ('Testing set built. Size: '+str(test_set_size))

test_image_list = [load_image_test(d[0],Config.image_size) for d in test_set_list]
test_label_list = [convert_label(d[1]) for d in test_set_list]
test_label_list, test_image_list = np.array(test_label_list), np.array(test_image_list)
test_label_list, test_image_list = test_label_list[:Config.batch_size,:,:,:],test_image_list[:Config.batch_size,:,:,:]


del test_set_list
#print(test_label)
with tf.Graph().as_default():
    image = tf.placeholder(tf.float32, [Config.batch_size, Config.image_size, Config.image_size, 3],name='input')
    label = tf.placeholder(tf.float32, [Config.batch_size, Config.image_size, Config.image_size, 2],name = 'train_label')
    test_loss = tf.placeholder(tf.float32,name='loss_test')
    test_image = tf.placeholder(tf.float32, [None, Config.image_size, Config.image_size, 3],name = 'test_image')
    test_label = tf.placeholder(tf.float32, [None, Config.image_size, Config.image_size, 1],name = 'test_label')
    test_output = tf.placeholder(tf.float32, [None, Config.image_size, Config.image_size, 1],name = 'test_output')
    kenerl_mask = tf.placeholder(tf.float32, [2*Config.kernels_radius+1, 2*Config.kernels_radius+1],name = 'kernels_mask')
    global_step = tf.placeholder(tf.int32,name='global_step')
    global_ep = tf.placeholder(tf.int32,name='global_ep')
    lamda_n = tf.placeholder(tf.float32,name='lamda')
    img_num = tf.placeholder(tf.int32,name='img_num')


    vgg = vgg16(image*255.0)


    if Config.model =='FPN':
        model = FPN(image, block_num = Config.block)
    elif Config.model =='Unet':
        model = unet(image, block_num = Config.block)
    elif Config.model =='FPN_RFA':
        model = FPN_RFA(image, block_num = Config.block)
    elif Config.model =='FPN_RFA_BS':
        model = FPN_RFA_BS(image, score = vgg.probs[:,1],block_num = Config.block)
    elif Config.model =='FPN_RFA_BS_2':
        model = FPN_RFA_BS_2(image, score = vgg.probs[:,1],block_num = Config.block)
    elif Config.model =='FPN_RFA_BS_3':
        model = FPN_RFA_BS_3(image, score = vgg.probs[:,1],block_num = Config.block)
    elif Config.model == 'FPN_RFA_3':
        model = FPN_RFA_3(image, score = vgg.probs[:,1],block_num = Config.block)
    elif Config.model == 'FPN_RFA_4':
        model = FPN_RFA_4(image, score = vgg.probs[:,1],block_num = Config.block)
    elif Config.model == 'FPN_RFA_BS_4':
        model = FPN_RFA_BS_4(image, score = vgg.probs[:,1],block_num = Config.block)

    if Config.loss =='CE':
        loss = cross_entropy(label, model.finalpre)
        tf.summary.scalar('train_loss', loss)
    elif Config.loss =='WCE':
        ce_loss = weighted_cross_entropy_s(label, model.finalpre)
        loss_v = ce_loss
        loss = tf.math.reduce_mean(loss_v)
        tf.summary.scalar('train_loss', loss)  

    elif Config.loss =='GCRF':
        ce_loss = weighted_cross_entropy_s(label, model.finalpre)
        crf_loss = CRF_loss_s(model.finalpre,image,Config.kernels_radius,CRF_loss_config,kenerl_mask,mask_dst=None,mask_src=None,compatibility=None)
        loss_v = ce_loss+Config.crf_w*crf_loss
        loss = tf.math.reduce_mean(loss_v)
        tf.summary.scalar('ce_loss', tf.math.reduce_mean(ce_loss))
        tf.summary.scalar('crf_loss', tf.math.reduce_mean(crf_loss))
        tf.summary.scalar('train_loss', loss)  

    elif Config.loss =='BSWCE':
        ce_loss = weighted_cross_entropy_s(label, model.finalpre)
        loss_v = ce_loss
        loss = tf.math.reduce_mean(vgg.probs[:,1]*ce_loss)
        tf.summary.scalar('ce_loss', tf.math.reduce_mean(ce_loss))
        tf.summary.scalar('train_loss', loss)  
    
    elif Config.loss =='BSGCRF':
        ce_loss = weighted_cross_entropy_s(label, model.finalpre)
        crf_loss = CRF_loss_s(model.finalpre,image,Config.kernels_radius,CRF_loss_config,kenerl_mask,mask_dst=None,mask_src=None,compatibility=None)
        loss_t = ce_loss+Config.crf_w*crf_loss
        loss_v = vgg.probs[:,1]*loss_t
        loss = tf.math.reduce_mean(vgg.probs[:,1]*loss_t)
        tf.summary.scalar('ce_loss', tf.math.reduce_mean(ce_loss))
        tf.summary.scalar('crf_loss', tf.math.reduce_mean(crf_loss))
        tf.summary.scalar('train_loss', loss)  



    elif Config.loss =='WGCRF':
        ce_loss = weighted_cross_entropy(label, model.finalpre)
        crf_loss = W_CRF_loss(model.finalpre,image,Config.kernels_radius,vgg.probs[:,1],CRF_loss_config,kenerl_mask,mask_dst=None,mask_src=None,compatibility=None)
        loss = ce_loss+Config.crf_w*crf_loss
        tf.summary.scalar('ce_loss', ce_loss)
        tf.summary.scalar('crf_loss', crf_loss)
        tf.summary.scalar('train_loss', loss)    
    #loss = ce_loss 

    decay_steps = Config.decay_ep*train_set_size//Config.batch_size


    lr=tf.train.exponential_decay(Config.lr,global_step = global_step, decay_steps = decay_steps, decay_rate = Config.decay_rate,staircase = 'True')
 

    variables_all = tf.trainable_variables()
    #print(variables)
    variables_model = [v for v in variables_all if v.name.split('/')[0]=='FPN_RFA_BS']
    variables_vgg = [v for v in variables_all if v.name.split('/')[0]!='FPN_RFA_BS']
    #train_step = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss,var_list=variables_model)
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,var_list=variables_model)
    #train_step = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    grads, variables = zip(*optimizer.compute_gradients(loss))
    # grads, global_norm = tf.clip_by_global_norm(grads, 5)
    # train_step = optimizer.apply_gradients(zip(grads, variables))


    
    tf.summary.scalar('test_loss', test_loss)
    tf.summary.scalar('img_num', img_num)
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('lamda', lamda_n)
    tf.summary.image('train_images', image)
    tf.summary.image('test_images', test_image)
    tf.summary.image('labels', label[:,:,:,:1])
    tf.summary.image('test_label', test_label*255)
    tf.summary.image('output', test_output)
    tf.summary.image('finalpre', model.finalpre[:,:,:,:1])


    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    # Summarize all gradients
    # grads = tf.gradients(loss, tf.trainable_variables())
    # grads = list(zip(grads, tf.trainable_variables()))

    for i in range(len(grads)):
        if grads[i] is not None:
            tf.summary.histogram(variables[i].name + '/gradient', grads[i])
    summary_step = tf.summary.merge_all()


    #config = tf.ConfigProto(allow_soft_placement=True)
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    #config.gpu_options.allow_growth = True


    step = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(ckpt_save_dir+'/logs', sess.graph)


        saver = tf.train.Saver(variables_all,max_to_keep = 10)
        saver1 = tf.train.Saver(variables_model)
        # restore old checkpoint
        checkpoint = tf.train.get_checkpoint_state(Config.checkpoint)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver1.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        
        #VGG
        saver2 = tf.train.Saver(variables_vgg)
        # restore old checkpoint
        checkpoint = tf.train.get_checkpoint_state(Config.vgg_checkpoint)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver2.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded VGG:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")


        new_train_list = []
        new_train_set_size =0

        generate_initial_label(TRAIN_SET_DIR_label,Config.label_update_dir)

        for ep in range(Config.max_epochs):

            if Config.self_pace and ep%Config.lamda_decay_ep==0:
                batch_size,image_size = Config.batch_size,Config.image_size
                new_train_list=[]
                batch_num = len(train_set_list)//batch_size
                train_set_queue = deque(train_set_list)
                for iter in range(0,batch_num):
                    minibatch = []
                    for count in range(0, batch_size):
                        element = train_set_queue.pop()
                        minibatch.append(element)
                        train_set_queue.appendleft(element)

                    image_list = [load_image(d[0],image_size) for d in minibatch]
                    label_list = [convert_label(os.path.join(Config.label_update_dir, os.path.basename(d[1]))) for d in minibatch]
                    
                    image_batch = np.array(image_list)
                    label_batch = np.array(label_list)

                    image_batch = np.reshape(image_batch, [batch_size, image_size, image_size, 3])
                    label_batch = np.reshape(label_batch, [batch_size, image_size, image_size, 2])
                    print(image_batch.shape,label_batch.shape)
                
                    net_out, loss_for_sp = sess.run([model.finalpre, loss_v], feed_dict = {image: image_batch, label: label_batch,kenerl_mask: mask,global_step:step,global_ep:ep})

                    for i in range(0,batch_size):
                        new_lamda = Config.lamda*math.exp(float(ep)/200)
                        print(new_lamda)
                        if loss_for_sp[i] <= new_lamda:
                            new_train_list.append(minibatch[i])
                            generate_new_label(net_out[i,:,:,0],os.path.basename(minibatch[i][0]), Config.label_update_dir)
                    
            #if len(new_train_list)==new_train_set_size:
                #continue

            new_train_set_size_1 = len(new_train_list)
            new_train_set_size = len(train_set_list)

            batch_num = new_train_set_size//Config.batch_size
            #train_set_queue =  deque(new_train_list)
            train_set_queue = deque(train_set_list)
            random.shuffle(train_set_queue)
            for j in range(0,batch_num):
                minibatch = []
                for count in range(0, Config.batch_size):
                    element = train_set_queue.pop()
                    minibatch.append(element)
                    train_set_queue.appendleft(element)
                #print(minibatch)
                image_list = [load_image(d[0],Config.image_size) for d in minibatch]
                #label_list = [convert_label(d[1],Config.image_size) for d in minibatch]
                label_list = [convert_label(os.path.join(Config.label_update_dir, os.path.basename(d[1]))) for d in minibatch]

                image_batch = np.array(image_list)
                label_batch = np.array(label_list)
                #print(image_list)
                #print(label_list)
                #print(image_batch.shape)

                image_batch = np.reshape(image_batch, [Config.batch_size, Config.image_size, Config.image_size, 3])
                label_batch = np.reshape(label_batch, [Config.batch_size, Config.image_size, Config.image_size, 2])
                

                _, train_loss= sess.run([train_step,loss], feed_dict = {image: image_batch, label: label_batch,kenerl_mask: mask,global_step:step,global_ep:ep})
                #print(probs,temp,CRFloss)
                #for key in layers.keys():
                    #print(layers[key]) 


                step = step +1            

                if step%Config.summary_freq ==0 and step>0:

                    loss_test,output = sess.run([loss,model.finalpre],  feed_dict = {image: test_image_list, label: test_label_list,kenerl_mask: mask,global_step:step,global_ep:ep})
                    #print(output[1,:,:,1].shape)
                    #print(test_label_list[1,:,:,0].shape)
                    printstr = ('%s: step %d, train_loss = %.5f, test_loss = %5f')
                    print(printstr % (datetime.now(), step, train_loss, loss_test))
                    #print(step,train_loss, loss_test)
                    output = np.reshape(output[:,:,:,0],[-1,Config.image_size,Config.image_size,1])
                    test_labels = np.reshape(test_label_list[:,:,:,:1],[-1,Config.image_size, Config.image_size,1])
                    #print(test_label.shape)
                    summary = sess.run([summary_step],  feed_dict = {image: image_batch, label: label_batch, test_image: test_image_list, kenerl_mask: mask,\
                                                                    test_label:test_labels, test_loss: loss_test,global_step:step, test_output:output,global_ep:ep, \
                                                                    lamda_n: new_lamda, img_num:new_train_set_size_1 })
                    summary_writer.add_summary(summary[0], step)
            
            if ep % Config.save_freq == 0 and ep>0:

                saver.save(sess, ckpt_save_dir+'/model/'+str(ep)+'.ckpt')

                
        saver.save(sess, ckpt_save_dir+'/model/'+str(ep)+'.ckpt')