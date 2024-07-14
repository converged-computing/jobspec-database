import cv2
import numpy as np
import os
from tqdm import tqdm
# reserved for future use: no oversampling; directly add the poisson noise and concatenate images
# #read images into greayscale and add poisson noise
# def read_images(image_path, image_count):
#     for i in range(image_count):
#         if i % 50 == 0:
#             if i != 0:
#                 #concatenate images
#                 images_np = np.concatenate(images, axis=2)
#                 print(images_np.shape)
#                 #save to npy file
#                 np.save(image_path + "images_" + str(i//50-1) + ".npy", images_np)
#             images = []
#         tmp = cv2.imread(image_path + "frame%d.jpg" % i, 0)
#         #magnitude normalization (to 0.5)
#         tmp = tmp / np.max(tmp) * 0.5
#         #add poisson noise
#         tmp = np.random.poisson(tmp, tmp.shape)
#         tmp = (tmp > 0).astype(np.float32)
#         # #save tmp as image
#         # cv2.imwrite("test.jpg", tmp.astype(np.uint8)*255)
#         # exit()
#         images.append(np.expand_dims(tmp,2))

fixed_idx = False
if fixed_idx:
    idx_w = 200
    idx_h = 100

def crop_random(img, scale_factor, w, h=None):
    """randomly crop a patch shaped patch_size*patch_size, with a upscale factor"""
    h = w if h is None else h
    nw = img.shape[1] - w*scale_factor
    nh = img.shape[0] - h*scale_factor
    if nw < 0 or nh < 0:
        raise RuntimeError("Image is to small {} for the desired size {}". \
                                format((img.shape[1], img.shape[0]), (w*scale_factor, h*scale_factor))
                          )
    
    if not fixed_idx:
        idx_w = np.random.randint(0, nw+1)
        idx_h = np.random.randint(0, nh+1)

    scaled_patch = img[idx_h:idx_h+h*scale_factor, idx_w:idx_w+w*scale_factor]
    # print(scaled_patch.shape)

    patch = cv2.resize(scaled_patch, (w, h), interpolation=cv2.INTER_CUBIC)
    # print(patch.shape)
    return patch

def center_crop(img, crop_size):
    h, w = img.shape[0], img.shape[1]
    x = (w - crop_size[0]) // 2
    y = (h - crop_size[1]) // 2
    return img[y:y+crop_size[1], x:x+crop_size[0]]


#convert mp4 to list of images
def convert_video_to_images(video_paths, image_path):
    count = 0
    for video_path in video_paths:
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        print(success)
        while success:
            #! moved the crop to the dataset loader
            # image = crop_random(image, 2, 640, 640)
            cv2.imwrite(image_path + "frame%d.jpg" % count, image)  # save frame as JPEG file
            success, image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1
    return count
def convert_video_to_images_with_subfolder(video_paths, image_path,image_size = (192,192)):
    video_frame_count = {}
    count = 0
    for video_path in tqdm(video_paths):
        count = 0
        os.mkdir(image_path + video_path.split("/")[-1])
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        while success:
            #! moved the crop to the dataset loader
            # image = crop_random(image, 2, 640, 640)
            image = center_crop(image, (720,720))
            image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(image_path + video_path.split("/")[-1] + "/frame%d.jpg" % count, image)  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
        video_frame_count[video_path.split("/")[-1]] = count
    return count

def convert_video_to_numpy(video_paths, image_path, batch_size, overlap_size=0, skip_size=0):
    batch_count = 0
    count = 0
    for video_path in video_paths:
        success = True
        image_list = []
        count = 0
        while success:
            vidcap = cv2.VideoCapture(video_path)
            success, image = vidcap.read()
            if success:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(1,image.shape[0], image.shape[1])
                image_list.append(image)
                count += 1
                if (count) % batch_size == 0:
                    image = np.concatenate(image_list, axis=0)
                    np.save(image_path + "frame%d.npy" % batch_count, image)
                    print("batch {} saved".format(batch_count), ", ", image.shape)
                    batch_count += 1
                    if overlap_size > 0:
                        image_list = image_list[-overlap_size:]
                    else:
                        image_list = []
                    count = count - (batch_size-overlap_size)
    return batch_count
                    

#read images into greayscale and add poisson noise
def read_images(image_path, train_path, label_path, image_count, oversampled_rate=50):
    if not os.path.exists(train_path):
            os.makedirs(train_path)
    if not os.path.exists(label_path):
            os.makedirs(label_path)
    for i in range(image_count):
        tmp = cv2.imread(image_path + "frame%d.jpg" % i, 0)
        np.save(label_path + "frame%d.npy" % i, tmp)
        #magnitude normalization (to 0.5)
        if np.max(tmp) == 0:
            tmp = tmp * 0.0
        else:
            tmp = tmp / np.max(tmp) * 0.5
        images = []
        for j in range(oversampled_rate):
            #add poisson noise
            tmp_poisson = np.random.poisson(tmp, tmp.shape)
            tmp_poisson = (tmp_poisson > 0).astype(np.float32)
            # if i ==144:
            #     #save tmp as image
            #     cv2.imwrite(train_path+"/test_"+str(i)+"_"+str(j)+".jpg", tmp_poisson.astype(np.uint8)*255)
            
            #concatenate the channels: the last dimension
            images.append(np.expand_dims(tmp_poisson,2))
        #concatenate images
        images_np = np.concatenate(images, axis=2)
        print("frame%d.npy" % i, images_np.shape)
        #save to npy file
        np.save(train_path + "frame%d.npy" % i, images_np)

video_list=[]
for f in os.listdir("/scratch/yz87/original_high_fps_videos/"):
    print(f)
    if "720p_240fps_3.mov" not in f and "GOPR9650.mp4" not in f:
        video_list.append("/scratch/yz87/original_high_fps_videos/"+f)



#training images
img_count = convert_video_to_images_with_subfolder(video_list, "/scratch/yz87/test_images/")
print("total images: ", img_count)

#evaluation images
img_count = convert_video_to_images_with_subfolder(["/scratch/yz87/original_high_fps_videos/GOPR9650.mp4", "/scratch/yz87/original_high_fps_videos/720p_240fps_3.mov"], "/scratch/yz87/eval_images/")
print("total images: ", img_count)


# #training images
# img_count = convert_video_to_numpy(video_list, "../test_images_np/",257)
# print("total batches: ", img_count)

# #evaluation images
# img_count = convert_video_to_numpy(["../original_high_fps_videos/GOPR9646.mp4"], "../eval_images_np/", 257)
# print("total batches: ", img_count)

# #training images
# img_count = convert_video_to_images(video_list, "../test_images/")
# print("total images: ", img_count)

# #evaluation images
# img_count = convert_video_to_images(["../original_high_fps_videos/GOPR9646.mp4"], "../eval_images/")
# print("total images: ", img_count)


