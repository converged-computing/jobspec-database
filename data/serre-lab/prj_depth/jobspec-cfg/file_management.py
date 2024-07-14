import os
import shutil
import glob
import pickle
import re

# Set the directory where all blender_images_* folders are located
parent_dir = '/cifs/data/tserre_lrs/projects/prj_depth/dataset'

# New directory where files will be copied
target_dir = '/cifs/data/tserre_lrs/projects/prj_depth/dataset/blender_images_all'

# Create the target directory, if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# The starting index for the new files
index = 0

# Iterate over each directory in parent directory
for directory in sorted(glob.glob(parent_dir + '/blender_images*')):
    print('\n##################################')
    print('##################################')
    print('Processing directory {}'.format(directory))
    print('index : ',index)

    # Find the highest i value in the texture_tables folder
    texture_table_files = glob.glob(directory + '/texture_tables/*.pkl')
    max_i = max(int(re.findall(r'\d+', file)[-1]) for file in texture_table_files)

    # Copy and rename the png and exr files up to max_i
    for i in range(max_i + 1):
        # Copy png file
        png_src = directory + '/{}.png'.format(i)
        png_dst = target_dir + '/{}.png'.format(index)
        shutil.copy(png_src, png_dst)

        # Copy exr file
        exr_src = directory + '/depth_maps/{}.exr'.format(i)
        exr_dst = target_dir + '/depth_maps/{}.exr'.format(index)
        shutil.copy(exr_src, exr_dst)

        index += 1

    # Copy the texture table files
    for file in texture_table_files:
        shutil.copy(file, target_dir + '/texture_tables/')