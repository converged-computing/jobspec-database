""" Generate lists containing filepaths and labels for training, validation and evaluation. """
import pickle
import os.path
import random

def generate_data_list(TRAIN_SET_DIR_solar,TRAIN_SET_DIR_label,TEST_SET_DIR_solar,TEST_SET_DIR_label):
##### generate training set #####
    #TRAIN_SET_DIR_solar = '/root/google-esay-256/train/residential/'
    #TRAIN_SET_DIR_label = '/root/PSL/result_train_gradcam/conv4_3/'
    train_set_list = []
    pos_num = 0

    for i in range(1, 15000):
        img_path = os.path.join(TRAIN_SET_DIR_solar, '1-'+ str(i)+'.png')
        label_path = os.path.join(TRAIN_SET_DIR_label, '1-'+ str(i)+'.png')
        if  os.path.exists(img_path) and os.path.exists(label_path):
            train_set_list.append((img_path, label_path))
            pos_num += 1
            
    # for i in range(1, 6000):
    #     img_path = os.path.join(TRAIN_SET_DIR_solar, '0-'+ str(i)+'.png')
    #     label_path = os.path.join(TRAIN_SET_DIR_label, '0-'+ str(i)+'.png')
    #     if  os.path.exists(img_path) and os.path.exists(label_path):
    #         train_set_list.append((img_path, label_path))
    #         pos_num += 1
    # random.shuffle(train_set_list)
    #print(train_set_list)
    with open('train_set_list.pickle', 'wb') as f:
        pickle.dump(train_set_list, f)

    print ('Train set list done. #  samples: '+str(pos_num))

    ##### generate test set #####
    #TEST_SET_DIR_solar = '/root/google-esay-256/test/residential/'
    #TEST_SET_DIR_label = '/root/google-esay-256/test_gt/'
    test_set_list = []
    pos_num = 0

    for i in range(1, 15000):
        img_path = os.path.join(TEST_SET_DIR_solar, '1-'+ str(i)+'.png')
        label_path = os.path.join(TEST_SET_DIR_label, '1-'+ str(i)+'.png')
        if  os.path.exists(img_path) and os.path.exists(label_path):
            test_set_list.append((img_path, label_path))
            pos_num += 1


    for i in range(1, 15000):
        img_path = os.path.join(TEST_SET_DIR_solar, '0-'+ str(i)+'.png')
        label_path = os.path.join(TEST_SET_DIR_label, '0-'+ str(i)+'.png')
        if  os.path.exists(img_path) and os.path.exists(label_path):
            test_set_list.append((img_path, label_path))
            pos_num += 1

    #random.shuffle(test_set_list)
    #print(test_set_list)
    with open('test_set_list.pickle', 'wb') as f:
        pickle.dump(test_set_list, f)

    print ('Test set list done. # positive samples: '+str(pos_num))

