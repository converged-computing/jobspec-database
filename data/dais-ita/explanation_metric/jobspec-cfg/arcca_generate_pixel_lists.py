import numpy as np 
import tensorflow as tf
from keras import backend as K

import pickle

import sys
import os
import time
import random

"""
Takes in parameters needed to generate a pixel importance list for a proportion of a provided range of a dataset. 
Saves the pixel list as a pickle file to disk.

args:
experiment_id           :string

data_range_index_start  :int
data_range_index_end    :int

explanation_name        :string

generate_for_test_data  :bool

dataset_name            :string
model_name              :string


NOT IMPLEMENTED USING DEFAULTS ONLY:
path_to_spilt_file      :string
path_to_model_weights   :string
normalise_data          :bool

outputs:
---

"""

def CreateOrderedPixelsList(attribution_map):
    pixel_weight_list = []
    for i in range(attribution_map.shape[0]):
        for j in range(attribution_map.shape[1]):
            #TODO: decide best way to aggregate colour attribution
            if(len(attribution_map[i][j].shape) > 0):
                attribution_value = sum(attribution_map[i][j])
            else:
                attribution_value = attribution_map[i][j]
            pixel_weight_list.append( (i,j,attribution_value) )
    #TODO: confirm not taking the abs is correct
    return sorted(pixel_weight_list,key=lambda x: x[2],reverse=True)


def CreatePixelListForAllData(data_x, data_y, dataset_name, model_instance, explanation_instance,additional_args=None,framework_tool=None, train_x = None, train_y=None, denormalise_function=None):
    # if creating pixel list for data which isn't the training data, you must also pass the training data so it can be used by some explanation types. 
    # If training x and y not passed then data_x and data_y are the datasets training data.
    if train_x is None:
        train_x = data_x
    
    if train_y is None:
        train_y = data_y
    #default arguments for various explanation techniques 
    if(additional_args is None):
        additional_args = {
        "num_samples":100,
        "num_features":300,
        "min_weight":0.01, 
        "num_background_samples":50,
        "train_x":train_x,
        "train_y":train_y,
        "max_n_influence_images":9,
        "dataset_name":dataset_name,
        "background_image_pool":train_x,
        }
    if(not denormalise_function is None):
        additional_args["denormalise_function"] = denormalise_function
    
    total_imgs = len(data_x)
    dataset_pixel_weight_lists = [] 
    start = time.clock()

    verbose_every_n_steps = 5
    
    reset_session_every = 1 
    #Some explanation implementations cause slow down if they are used repeatidly on the same session.
    #if reset_session_every is trTrueue on the explanation instance, then the session will be cleared and refreshed every 'reset_session_every' steps.

    #TODO: Could be parallelized
    for image_i in range(total_imgs):
        if(image_i % verbose_every_n_steps == 0):
            print(time.clock() - start)

            start = time.clock()
            print("Generating Explanation for Image: "+str(image_i)+ " to "+ str(min(image_i+verbose_every_n_steps, total_imgs))+"/" + str(total_imgs))
            print("")
        
        if(image_i % reset_session_every == 0):
            if(explanation_instance.requires_fresh_session==True):
                if(not framework_tool is None):
                    print("___")
                    print("Resetting Session")
                    model_load_path = model_instance.model_dir
                    del model_instance
                    del explanation_instance
                    tf.reset_default_graph() 
                    tf.keras.backend.clear_session()
                    # print("Releasing GPU")
                    # cuda.select_device(0)
                    # cuda.close()
                    # print("GPU released")
                    model_instance = framework_tool.InstantiateModelFromName(model_name,model_save_path_suffix,dataset_json,additional_args = {"learning_rate":model_train_params["learning_rate"]})
                    model_instance.LoadModel(model_load_path)
                    explanation_instance = framework_tool.InstantiateExplanationFromName(explanation_name,model_instance)
                    print("session restarted")
                    print("___")
                    print("")    
        
        additional_outputs = None
        
        
        image_x = data_x[image_i]
        _, _, _, additional_outputs = explanation_instance.Explain(image_x,additional_args=additional_args) 
        
        attribution_map =  np.array(additional_outputs["attribution_map"])
        pixel_weight_list = CreateOrderedPixelsList(attribution_map)
        
        dataset_pixel_weight_lists.append(pixel_weight_list)


    return dataset_pixel_weight_lists


def SavePixelList(dataset_name,explanation_name,pixel_lists,data_range_index_start,data_range_index_end):
    pixel_out_path = os.path.join("pixel_lists",dataset_name+"_"+explanation_name +"_"+ str(format(data_range_index_start, '07d')) +"_"+ str(format(data_range_index_end, '07d')) +"_"+str(int(time.time()))+".pkl")
    with open(pixel_out_path,"wb") as f:
        pickle.dump(pixel_lists, f)


####
#initialise parameters from args


experiment_id=sys.args[1]


data_range_index_start = 0
if(len(sys.args) > 2):
    data_range_index_start = int(sys.args[2])

data_range_index_end = None
if(len(sys.args) > 3):
    data_range_index_end = int(sys.args[3])

explanation_name = "LIME"
if(len(sys.args) > 4):
    explanation_name = sys.args[4]

generate_for_test_data = False
if(len(sys.args) > 5):
    generate_for_test_data = bool(sys.args[5])

dataset_name = "CIFAR-10"
if(len(sys.args) > 6):
    dataset_name = sys.args[6]

model_name = "vgg16"
if(len(sys.args) > 7):
    model_name = sys.args[7]

framework_path = "/media/harborned/ShutUpN/repos/dais/interpretability_framework"
if(len(sys.args) > 8):
    framework_path = sys.args[8]



path_to_model_weights = dataset_name.lower().replace(" ","_")+"_"


normalise_data = True

print("Pixel Generation Parameters:")
print("experiment_id",experiment_id)
print("data_range_index_start",data_range_index_start)
print("data_range_index_end",data_range_index_end)
print("explanation_name",explanation_name)
print("generate_for_test_data",generate_for_test_data)
print("dataset_name",dataset_name)
print("model_name",model_name)
print("path_to_model_weights",path_to_model_weights)
print("normalise_data",normalise_data)
print("")



####
#initialise framework and misc parameters
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

###UPDATE FRAMEWORK PATH
sys.path.append(framework_path)

from DaisFrameworkTool import DaisFrameworkTool


np.random.seed(42)
tf.set_random_seed(1234)
random.seed(1234)

framework_tool = DaisFrameworkTool(explicit_framework_base_path=framework_path)


####
#initialise dataset tool
dataset_json, dataset_tool = framework_tool.LoadFrameworkDataset(dataset_name)

label_names = [label["label"] for label in dataset_json["labels"]] # gets all labels in dataset. To use a subset of labels, build a list manually

#LOAD DATA
#load all train images as model handles batching
print("load training data")
print("")
source = "train"
#TODO change batch sizes to -1 , 256 , 256
train_x, train_y = dataset_tool.GetBatch(batch_size = -1,even_examples=True, y_labels_to_use=label_names, split_batch = True, split_one_hot = True, batch_source = source, shuffle=False)
print("num train examples: "+str(len(train_x)))


#validate on up to 256 images only
source = "validation"
val_x, val_y = dataset_tool.GetBatch(batch_size = 256,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
print("num validation examples: "+str(len(val_x)))


#load train data
source = "test"
test_x, test_y = dataset_tool.GetBatch(batch_size = -1,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source, shuffle=False)
print("num test examples: "+str(len(test_x)))

print("First and Last Images in Data:")
print("Train:")
print(dataset_tool.live_training[0])
print(dataset_tool.live_training[-1])
print("")
print("Test:")
print(dataset_tool.live_test[0])
print(dataset_tool.live_test[-1])
print("")


print("calculating dataset mean")
dataset_mean = dataset_tool.GetMean()
print(dataset_mean)

if(normalise_data):
    denormalise_function = dataset_tool.CreateDestandardizeFuntion()



####
#initialise model
model_train_params ={
    "learning_rate": 0.0001
    ,"batch_size":128
    ,"num_train_steps":150
    ,"experiment_id":experiment_id
    }

model_save_path_suffix = ""

model_instance = framework_tool.InstantiateModelFromName(model_name,model_save_path_suffix,dataset_json,additional_args = model_train_params)

model_load_path = model_instance.model_dir

model_instance.LoadModel(model_load_path)
        

####
#initialise explanation
explanation_instance = framework_tool.InstantiateExplanationFromName(explanation_name,model_instance)


####
#generate pixel list for images from [data_range_index_start to data_range_index_end)
if(generate_for_test_data):
    working_data_x = np.copy(test_x)
    working_data_y = np.copy(test_y)
    pickle_prefix = "TEST_"
else:
    working_data_x = np.copy(train_x)
    working_data_y = np.copy(train_y)
    pickle_prefix = "TRAIN_"

if(data_range_index_end is None):
    data_range_index_end = len(working_data_x)

working_data_x = working_data_x[data_range_index_start:data_range_index_end]

working_data_x = model_instance.CheckInputArrayAndResize(working_data_x,model_instance.min_height,model_instance.min_width)

num_pixels_in_padded = working_data_x[0].shape[:-1][0] * working_data_x[0].shape[:-1][1]

pixel_lists = []

print("Creating Pixel List")
if(normalise_data):
    pixel_lists = CreatePixelListForAllData(dataset_tool.StandardizeImages(working_data_x), working_data_y, dataset_name, model_instance, explanation_instance,additional_args=None,framework_tool=framework_tool, train_x=dataset_tool.StandardizeImages(train_x), train_y=train_y, denormalise_function=denormalise_function)
else:
    pixel_lists = CreatePixelListForAllData(working_data_x, working_data_y, dataset_name, model_instance, explanation_instance,additional_args=None,framework_tool=framework_tool, train_x=train_x, train_y=train_y)


print("saving pixel lists")
SavePixelList(pickle_prefix+experiment_id,explanation_name,pixel_lists,data_range_index_start,data_range_index_end)

