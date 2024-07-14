import os

BASE = "TAR_Real_Train"
dir = os.listdir(BASE)

folder_list = set([f[:-2] for f in dir])

pair = []
id = [1, 2, 3, 4, 5]
for f in folder_list:
    for i in id[:-1]:
        img_folder = f"{BASE}/{f}-{i}"
        tgt_folder = f"{BASE}/{f}-{i+1}"
        imgs = os.listdir(img_folder)
        for img in imgs:
            tar_img  = img[:10] + str(i+1)+"_"+img[12:] # 01_08_03_02_0202.bmp
            if os.path.isfile(os.path.join(tgt_folder, tar_img)):
                pair.append((os.path.join(img_folder, img), os.path.join(tgt_folder, tar_img)))

print(len(pair))

import random
random.shuffle(pair)

train_pair = pair[:int(0.8*len(pair))]
val_pair = pair[int(0.8*len(pair)):]
import shutil

for i, t in enumerate(train_pair):
    shutil.copy2(t[1], os.path.join("data", "train","input_crops" , str(i)+".bmp"))
    shutil.copy2(t[0], os.path.join("data", "train","target_crops", str(i)+".bmp"))

for i, t in enumerate(val_pair):
    shutil.copy2(t[1], os.path.join("data", "val", "input_crops" ,str(i)+".bmp"))
    shutil.copy2(t[0], os.path.join("data", "val", "target_crops",str(i)+".bmp"))
