import os
import shutil
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
def filter_images_by_dpi(args):
    source_folder, target_folder, min_dpi = args
    os.makedirs(target_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)

        try:
            with Image.open(file_path) as img:
                if img.width >= min_dpi and img.height >= min_dpi:
                    target_path = os.path.join(target_folder, filename)
                    shutil.copy(file_path, target_path)
                    print(f"Copied '{filename}' to '{target_folder}'")
        except IOError:
            print(f"Failed to open '{filename}'. It may not be an image.")


def process_folders(source_folders, target_folders, min_dpi=100):
    pool_args = [(source_folders[i], target_folders[i], min_dpi) for i in range(len(source_folders))]
    folder_count = len(source_folders)  
    with Pool() as pool:
        
        with tqdm(total=folder_count, desc="Processing folders") as pbar:
            for _ in pool.imap_unordered(filter_images_by_dpi, pool_args):
                pbar.update(1)  
# Example usage with lists of source and target folders
BASE_DIR = "/users/rye13/finalProject/Real_Train"
TAR_DIR = "/users/rye13/finalProject/TAR_Real_Train"
source_folders = [os.path.join(BASE_DIR, i) for i in os.listdir(BASE_DIR)]
target_folders = [os.path.join(TAR_DIR, i) for i in os.listdir(BASE_DIR)]

process_folders(source_folders, target_folders)


