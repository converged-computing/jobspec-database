
import os
import argparse
import tensorflow as tf
import tensorflow_hub as tfhub
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
import numpy as np
import functools
import cv2
import time
import concurrent.futures

import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
from PIL import Image

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def prepare_img(img):
    max_dim = 512
    img = tf.convert_to_tensor(img)
    #img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
    tensor = tensor[0]
    return tensor #PIL.Image.fromarray(tensor)

def get_args():
    parser = argparse.ArgumentParser(description='ArtsyML')
    parser.add_argument('--style_img', default='/home/haicu/ruolin.shen/projects/ArtsyML/images_style/style1.jpg')
    return parser.parse_args()

def style_model_part(content_image,style_image):
    print('style model begin ......')
    begin = time.time()
    style_image_tensor = style_model(tf.constant(content_image), tf.constant(style_image))[0]
    style_img = tensor_to_image(style_image_tensor)
    end = time.time()
    
    print('style model:',end-begin)
    
    return style_img
    
def seg_model_part(frame):
    print('segmentation model begin ......')
    begin_seg = time.time()
    input_tensor = preprocess(frame)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device=device)

    with torch.no_grad():
        seg_output = seg_model(input_batch)['out'][0]
        seg_output_predictions = seg_output.detach().argmax(0)
    # edit segmentation mask to binary to keep people only
    seg_mask =  seg_output_predictions.cpu().data.numpy() 
    
    end_seg = time.time()
    print('seg model:',end_seg-begin_seg)
    
    return seg_mask

def f(x):  # (nums,frame,content_image,style_image)
    #print('threading...',nums,x[1].shape,x[2].shape,x[3].shape)
    if x[0] == 0: # gpu 0
        return style_model_part(x[2],x[3])
    else:
        return seg_model_part(x[1])

if __name__ == '__main__':

    args = get_args()
    style_path = args.style_img
    
    # Torch seg_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE', device)
    torch.cuda.set_per_process_memory_fraction(0.5)
    seg_model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
    seg_model = seg_model.to(device=device)
    seg_model.eval()

    # Tensorflow style model
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
    style_image = load_img(style_path)
    style_image = tf.stack([style_image[:,:,:,2],style_image[:,:,:,1],style_image[:,:,:,0]],axis = 3)
    style_model = tfhub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    #style_model = Net(ngf=128)
    #style_model.load_state_dict(torch.load('21styles.model'))
#     print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    print('loaded model')
    #   cap = cv2.VideoCapture(0)
    arr = np.load('/home/haicu/ruolin.shen/projects/ArtsyML/video.npy')
    prev_capture = time.time()
    for i in range(10):

        capture_time = time.time()
        print('Time between captures: ', capture_time - prev_capture)
        prev_capture = capture_time

        # Capture frame-by-frame
#         ret, frame = cap.read()
        frame = arr[i]
        content_image = prepare_img(frame)
        frame = cv2.resize(frame, (content_image.shape[2], content_image.shape[1]))
        
        # paralization
        b = time.time()
#         style_img = f(0, frame,content_image,style_image)
#         seg_mask = f(1, frame,content_image,style_image)
        nums = [(0, frame,content_image,style_image), (1, frame,content_image,style_image)]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            r_m = [val for val in executor.map(f, nums)]
        style_img = r_m[0]
        seg_mask = r_m[1]
        e = time.time()

        print('parallel time:',e-b)
        

        # Preparing the frame for the style net
#         tik = time.time()
#         content_image = prepare_img(frame)
#         tok = time.time()
#         print('Time for style preparing: ', tok-tik)
#         style_image_tensor = style_model(tf.constant(content_image), tf.constant(style_image))[0]
#         tik = time.time()
#         print('Time for style model: ', tik-tok)
#         style_img = tensor_to_image(style_image_tensor)
#         tok = time.time()
#         print('Time for style tensor to image: ', tok-tik)        



        # Preparing the frame for the segmentation net 
        # resize to same shape as output of style net
#         tik = time.time()
#         frame = cv2.resize(frame, (content_image.shape[2], content_image.shape[1]))
#         input_tensor = preprocess(frame)
#         tok = time.time()
#         print('Time for segment prep: ', tok-tik)
#         # create a mini-batch as expected by the model
#         input_batch = input_tensor.unsqueeze(0)
#         input_batch = input_batch.to(device=device)
#         tik = time.time()
#         print('Time for segment to gpu: ', tik-tok)

#         with torch.no_grad():
#             seg_output = seg_model(input_batch)['out'][0]
#             seg_output_predictions = seg_output.detach().argmax(0)
#         tok = time.time()
#         print('Time for segmentation model: ', tok-tik)

#         # edit segmentation mask to binary to keep people only
#         seg_mask =  seg_output_predictions.cpu().data.numpy() 
#         print(seg_mask.dtype)
#         tik = time.time()
#         print('Time for seg to cpu: ', tik-tok)
        seg_mask[seg_mask!=15] = 0
        seg_mask[seg_mask==15] = 1

        # keep people only from style image and background only from original frame
        style_img =  (1-seg_mask[:,:,None])*frame + seg_mask[:,:,None]*style_img
        style_img = style_img.astype(np.uint8)
        
#         tok = time.time()
#         print('Time for rest: ', tok-tik)
        print(' ')

        # Display the resulting frame
#         cv2.imshow('Style Transfer', style_img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    # When everything done, release the capture
    #cap.release()
    #cv2.destroyAllWindows()
