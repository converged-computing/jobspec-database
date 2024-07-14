import pandas as pd
from tifffile import imread
from tqdm import tqdm
import numba
import numpy as np
from multiprocessing import Pool

images = pd.read_csv("/nfs/turbo/umms-tocho/code/achowdur/experiment/vqmodel/test2.csv")

def image_size(image_path):
    img = imread(image_path)
    return img.shape[-1]
tqdm.pandas()
images["fullpath"] = images["file_name"].progress_apply(lambda x: "/nfs/turbo/umms-tocho/root_srh_db"+x)

split_dfs = np.array_split(images,24)

for idx, df in enumerate(split_dfs):
    df["split_val"]=idx

total_df = pd.concat(split_dfs)

def image_size(image_path):
    img = imread(image_path)
    size = img.shape[-1]
    if size != 300:
        print(image_path)
        f = open("/nfs/turbo/umms-tocho/code/achowdur/log_test_check_data.txt", "a")
        f.write(image_path)
        f.close()
    return size
def custom_func(idx,df=total_df):
    df = df[df["split_val"]==idx]
    # print(df.head())
    df["size"]=df["fullpath"].progress_apply(lambda x: image_size(x))
    return df
all_records = total_df["split_val"].unique()
    
pool = Pool(processes=24)
results = pool.map(custom_func, all_records)
pool.close()
pool.join()

# concatenate results into a single pd.Series
results = pd.concat(results)

results.to_csv("/nfs/turbo/umms-tocho/code/achowdur/experiment/vqmodel/test2_updated.csv",index=False)
