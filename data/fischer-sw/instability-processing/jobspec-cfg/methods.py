import os
import sys
import json
import glob
import logging

from multiprocessing import Pool
from functools import partial
from pathlib import Path
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import SimpleITK as sitk

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s")


def get_image_files(config, case, folder):
    """
    Function that reads image names from folder
    """
    if folder in ["raw_cases", "png_cases"]:
        dir_path = os.path.join(config["raw_data_path"])
        dat_path = os.path.join(dir_path, folder, case)
    elif folder in ["contours", "instabilities"]:
        dir_path = os.path.join(config["results_path"], "final_data")
        dat_path = os.path.join(dir_path, case, folder)
    else:
        dir_path = os.path.join(config["data_path"])
        dat_path = os.path.join(dir_path, case, folder)
    if os.path.exists(dir_path) is False:
        logging.error(f"Data directory {dir_path} doesn't exsist. Please check config for valid path.")
        return []

    if os.path.exists(dat_path) is False:
        logging.warning(f"No data found for case {case} at {dat_path}")
        return []
    
    images = glob.glob(os.path.join(dat_path, "*.tiff"))
    if images == []:
        images = glob.glob(os.path.join(dat_path, "*.png"))
    if images == []:
        logging.warning(f"No images found in folder {folder} for case {case}")
        tmp_images = []
    else:
        images = [os.path.basename(x) for x in images]
        tmp_images = [x.split(".")[0] for x in images]

    logging.info(f'Found {len(tmp_images)} images for case {case}')
    if config["images"] != []:
        img_subset = []
        for img in images:
            number = int(img.split("_")[0])
            if number in config["images"]:
                img_subset.append(img)
        tmp_images = img_subset

    tmp_images = [x.split(".")[0] for x in tmp_images]
    tmp_images.sort()
    images = [x.split(".")[0] for x in images]
    images.sort()
    # convert images to png if from .tiff folder
    if folder == "raw_cases":
        cpus = os.cpu_count()
        p = Pool(cpus)
        p.map(partial(convert2png, config=config, case=case), images)
    return tmp_images

def get_config():
    """
    Function that reads config from file
    """
    path = os.path.join(sys.path[0], "config.json")
    with open(path) as f:
        cfg = json.load(f)
    return cfg

def rename_cases(config):
    """
    Function that gets rid of " " in folder names in raw_cases dir
    """
    base_path = os.path.join(config["raw_data_path"], "raw_cases")
    cases = os.listdir(base_path)

    for cas in cases:
        if " " in cas:
            new_cas = cas.replace(" ", "_")
            os.rename(os.path.join(base_path, cas), os.path.join(base_path, new_cas))

def multi_contour_plot():
    """
    Function that creates a plot of contours for different time stamps for given cases  
    """
    # getting configuration
    config = get_config()

    if len(config["images"]) == 0:
        logging.warning("No images set to combine")
        return

    for cas in config["cases"]:
        logging.info(f"Creating multi contour image for case {cas} at times {config['images']}")

        # get image filenames
        images = get_image_files(config, cas, "contours")

        if len(images) == 0:
            continue
            
        times = []
        for img in images:
            t_stamp = int(img.split('_')[0])
            times.append(t_stamp)

        fig, axs = plt.subplots()
        axs.set_title(f"Contours {cas} at times {times}")
        cmap = mpl.colormaps['Spectral']
        
        counter = 1
        # loop through all images and combine them into one plot
        multi_img = np.zeros(read_image(config, "contours", images[0], cas).shape)
        tmp_legend = []
        tmp_labels = []
        for img in images:
            
            tmp_image = read_image(config, "contours", img, cas)
            tmp_image[tmp_image == 255] = counter
            # multi_img[tmp_image == counter] = counter
            multi_img += tmp_image
            counter += 1
            multi_img = multi_img.copy()
        
        axs.imshow(multi_img, cmap=cmap)
        uni_vals, uni_count = np.unique(multi_img, return_counts=True)
        counter = 1
        for img in images:
            t_stamp = int(img.split('_')[0])
            tmp_val = 1- (1/(max(uni_vals)) * counter)
            tmp_legend.append(Line2D([0], [0], color=cmap(tmp_val), lw=4))
            tmp_labels.append(f"t={t_stamp}")
            counter += 1

        # axs.legend(tmp_legend, tmp_labels)
        
        if config["debug"] is False:
            plt.close(fig)

        # save image into folder
        folder = "multi_contour"
        dir_path = os.path.join(config["results_path"], "final_data", cas, folder)
        if os.path.exists(dir_path) is False:
            os.makedirs(dir_path)
            logging.info(f"Creating dir {folder} for case {cas}")
        if config["save_intermediate"]:
            fig.savefig(os.path.join(dir_path, "multi_contours" + ".png"))

def calc_case_ratio():
    """
    Function that calulates the case ratio for one case
    """
    config = get_config()
    for cas in list(config["cases"]):
        #create all intermediate folders
        create_intermediate_folders(config, cas)

        tmp_conf = config.copy()
        tmp_conf["images"] = []
        background_images = get_image_files(tmp_conf, cas, "png_cases")
        if background_images == []:
            background_images = get_image_files(tmp_conf, cas, "raw_cases")
        if background_images == []:
            logging.warning(f"No images found for case {cas}")
            continue

        background_img = get_base_image(config, cas, background_images[0])

        images = get_image_files(config, cas, "png_cases")
        if images == []:
            images = get_image_files(config, cas, "raw_cases")
        if images == []:
            logging.warning(f"No images found for case {cas}")
            continue
        images.sort()

        if config["images"] == []:
            config["debug"] = False

        if config["debug"]:
            parallel = False
        else:
            if config["hpc"]:
                parallel = False
            else:
                parallel = True

        if parallel:
            cpus = os.cpu_count()
            p = Pool(cpus)
            p.map(partial(process_image, config=config, cas=cas, background=background_img), images)
        else:
            for img in images:
                status = process_image(img, config, cas, background=background_img)
        
        if config["debug"] is False:
            if len(config["images"]) == 0:
                logging.info(f"Starting to create animations for case {cas}")
                create_animation(config, cas, "contours")
                create_animation(config, cas, "instabilities")
                create_animation(config, cas, "png_cases")

        remove_empty_folders(config["data_path"])
        remove_empty_folders(config["results_path"])

def create_intermediate_folders(config, cas):
    """
    Function that creates all intermediate folders
    """
    folders = ["closed_instability", "contour", "raw_instability", "delta_data", "final_images", "histogramms", "instability", "segmented_camera", "segmented_instability"]

    for fld in folders:

        folder_path = os.path.join(config["data_path"], cas, fld)
        if os.path.exists(folder_path) is False:
            # check base path
            tmp_path = Path(*Path(folder_path).parts[:1])
            if os.path.exists(tmp_path) is False:
                logging.error(f"Path {folder_path} is not accessable. You might need an active VPN Connection to access the folder")
                exit()
            else:
                os.makedirs(folder_path)

    # create folders for results
    final_folders = ["final_data"]
    for fld in final_folders:
        folder_path = os.path.join(config["results_path"], fld, cas)
        if os.path.exists(folder_path) is False:
            os.makedirs(folder_path)
        folder_path = os.path.join(config["results_path"], fld, cas, "instabilities")
        if os.path.exists(folder_path) is False:
            os.makedirs(folder_path)
        folder_path = os.path.join(config["results_path"], fld, cas, "contours")
        if os.path.exists(folder_path) is False:
            os.makedirs(folder_path)
    
    # create png_folder if not there
    png_dir_path = os.path.join(config["raw_data_path"], "png_cases", cas)
    if os.path.exists(png_dir_path) is False:
        os.makedirs(png_dir_path)
        

def get_all_cases(config):
    """
    Function that puts all cases in config
    """
    path = os.path.join(config["raw_data_path"], "raw_cases")
    dirs = os.listdir(path)
    config["all_cases"] = dirs
    cfg_path = os.path.join(sys.path[0], "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)    


def convert2png(file, config, case) -> bool:
    """
    Function that converst .tiff images to .png images and stores them in png_cases folder for a case
    """
    
    new_dir_path = os.path.join(config["raw_data_path"], "png_cases", case)
    new_img_path = os.path.join(new_dir_path, file + ".png")
    img_path = os.path.join(config["raw_data_path"], "raw_cases", case, file)
    if os.path.exists(new_dir_path) is False:
        os.makedirs(new_dir_path)
    if os.path.exists(new_img_path):
        return True
    # im = Image.open(img_path + ".tiff")
    im = cv2.imread(img_path + ".tiff", cv2.IMREAD_GRAYSCALE)
    logging.info(f"Saved image at {new_img_path}")
    save_image(config, "png_cases", file, im, case)
    return True

def read_image(config, folder, filename, case):
    """
    Function that reads an image from directory
    """
    if folder in ["png_cases", "raw_cases"]:
        dir_path = os.path.join(config["raw_data_path"], folder, case)
    elif folder in ["contours", "instabilities"]:
        dir_path = os.path.join(config["results_path"], "final_data", case, folder)
    else:
        dir_path = os.path.join(config["data_path"], case, folder)
    img_path = os.path.join(dir_path, filename + ".png")
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        logging.warning(f"Image {img_path} does not exsist")
        return None
    return img

def process_image(img_name, config, cas, background) -> bool:
    """
    Function that processes an image
    """
    final_dir_path = os.path.join(config["results_path"], "final_data", cas)
    if os.path.exists(final_dir_path) is False:
        os.makedirs(final_dir_path)
    res_csv_folder = os.path.join(final_dir_path, "instabilities")
    if os.path.exists(res_csv_folder) is False:
        os.makedirs(res_csv_folder)
    res_csv_path = os.path.join(res_csv_folder ,img_name + ".png")
    if os.path.exists(res_csv_path) and config["new_files"] is False:
        logging.info(f"Already processed image {img_name} for case {cas}")
        return True
    logging.info("-----------------------------")
    logging.info(f"processing image {img_name}")
    logging.info("-----------------------------")

    status = True
    # hist_stat = make_histo(config, cas, "png_cases", img_name)

    base_img = get_base_image(config, cas, img_name)

    background_array = sitk.GetArrayFromImage(background)

    base_array = sitk.GetArrayFromImage(base_img)

    # trying background substraction
    # base_array = base_array - background_array
    # base_array = background_array - base_array
    

    base_array[:,300:304] = 0
    base_img = sitk.GetImageFromArray(base_array)

    try:
        logging.info(f"Starting camera segmentation for image {img_name}")
        cam_img = segment_camera(config, cas, base_img, img_name)
    except:
        logging.error(f"Something went wrong with the camera segmentation for image {img_name}")
        return False
    # get cam array
    cam_array = sitk.GetArrayFromImage(cam_img)
    # replace all found cam pixel with 0 in base image
    base_array[cam_array == 1] = 0
    base_img = sitk.GetImageFromArray(base_array)

    try:
        logging.info(f"Starting insta segmentation for image {img_name}")
        insta_img, status = segment_instability(config, cas, base_img, img_name, cam_img)
    except:
        logging.error(f"Something went wrong with the instability segmentation for image {img_name}")
        return False
    if status is False:
        logging.warning(f"No instability found in {img_name}")
        return status

    # hull = convex_hull(config, cas, insta_img, img_name)
    try:
        logging.info(f"Starting insta refining for image {img_name}")
        new_insta_img, status = refine_instability(config, cas, insta_img, img_name)
    except:
        logging.error(f"Something went wrong with the instability refinement for image {img_name}")
        return False
    if status is False:
        logging.error(f"Instability found in {img_name} is more than one blob")
        return status
    
    try:
        logging.info(f"Starting to close insta for image {img_name}")
        status, final_insta_img = close_instability(config, cas, new_insta_img, img_name, cam_img)
    except:
        logging.error(f"Something went wrong while closing the instability for image {img_name}")
        return False
    if status is False:
        logging.error(f"Closing instability in {img_name} failed")
        return status
    
    try:
        logging.info(f"Getting contour for image {img_name}")
        contours = get_contour(config, cas, final_insta_img, img_name)
    except:
        logging.error(f"Something went wrong with the contour extraction for image {img_name}")
        return False
    df = pd.DataFrame(contours)
    con_csv_folder = os.path.join(final_dir_path, "contours")
    if os.path.exists(con_csv_folder) is False:
        os.makedirs(con_csv_folder)
    con_path = os.path.join(con_csv_folder, img_name + ".png")
    plt.imsave(con_path, contours, cmap="Greys", dpi=1200)

    res_array = sitk.GetArrayFromImage(final_insta_img)
    final_img_path = os.path.join(config["data_path"], cas, "final_images")
    if os.path.exists(final_img_path) is False:
        os.makedirs(final_img_path)
    
    plt.imsave(res_csv_path, res_array, cmap="Greys", dpi=1200)

def save_image(config, folder, filename, img, case):
    """
    Function that saves images
    """
    dir_path = os.path.join(config["raw_data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    img_path = os.path.join(config["raw_data_path"], folder, case, filename + ".png")
    cv2.imwrite(img_path, img)

def get_base_image(config, case, file_name):
    """
    Function that gets base image to process
    """
    raw_img_path = os.path.join(config["raw_data_path"], "png_cases", case, file_name + ".png")
    if os.path.exists(raw_img_path):
        reader = sitk.ImageFileReader()
        reader.SetImageIO("PNGImageIO")
        reader.SetFileName(raw_img_path)
        raw_image = reader.Execute()
    else:
        raw_img_path = os.path.join(config["raw_data_path"], "raw_cases", case, file_name + ".tiff")
        reader = sitk.ImageFileReader()
        reader.SetImageIO("TIFFImageIO")
        reader.SetFileName(raw_img_path)
        raw_image = reader.Execute()

    return raw_image

def get_contour(config, case, base_image, file_name):
    """
    Function that gets the contour of the instability
    """
    # inv_base_img = sitk.InvertIntensity(base_image, maximum=1)

    base_array = sitk.GetArrayFromImage(base_image)
    # base_array -= 1

    contours, hierarchy = cv2.findContours(image=base_array, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # only get the longest contour
    length = 0
    for ele in contours:
        if len(ele) > length:
            contours = [ele]
            length = len(ele)

    contour_image = cv2.drawContours(image=np.zeros(base_array.shape), contours=contours, contourIdx=-1, color=(1, 0, 0), thickness=7, lineType=cv2.LINE_AA)
    # cv2.imshow("Contour image", contour_image)
    fig, axs = plt.subplots()
    axs.set_title(f"Contour Image {file_name.split('_')[0]}")
    pos = axs.imshow(contour_image, cmap="Greys")
    fig.colorbar(pos, ax=axs)
    if config["debug"] is False:
        plt.close(fig)

    folder = "contour"
    dir_path = os.path.join(config["data_path"], case, folder)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    
    return contour_image

def close_instability(config, case, base_image, file_name, cam_image):
    """
    Function that trys to close camera hole. WORK IN PROGRESS!
    """

    status = True
    image_array = sitk.GetArrayFromImage(base_image)
    lower = (0,0)
    upper = (0,0)
    rates = {
        "line" : [],
        "values" : []
    }
    contours, hierarchy = cv2.findContours(image=image_array, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    cont = contours[0]

    M = cv2.moments(cont)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cX -= 10
    logging.debug(f"cX: {cY}, cY: {cX}")

    contour_image = cv2.drawContours(image=np.zeros(image_array.shape), contours=contours, contourIdx=0, color=(1, 0, 0), thickness=7, lineType=cv2.LINE_AA)

    bound_rect = cv2.boundingRect(cont)
    logging.debug(bound_rect)

    tl_x = bound_rect[0]
    tl_y = bound_rect[1]
    width = bound_rect[2]
    height = bound_rect[3]

    #get array from camera image
    cam_array = sitk.GetArrayFromImage(cam_image)

    step = 75
    counter = 1
    m = 90


    # get slope and function of camera segementation. Only uses left pixels of segementation
    while abs(m) > 5 and counter < 11:    
        cam_array = sitk.GetArrayFromImage(cam_image)
        # cam_array[:,:tl_x-counter*2*step] = 0 
        # # cam_array[:,tl_x+counter*step:] = 0
        # cam_array[:,tl_x:] = 0
        tmp_y, tmp_x = np.where(cam_array == 1)
        x_lim = tmp_x.min()
        cam_array[:,:x_lim] = 0
        cam_array[:,x_lim + counter*step:] = 0

        y,x = np.where(cam_array == 1)
        coef = np.polyfit(x,y,1)
        m = coef[0]
        poly1d_fn = np.poly1d(coef)

        fig, axs = plt.subplots()
        axs.set_title(f"Test instability {counter}")
        axs.axline((0, poly1d_fn(0)), slope=m)
        pos = axs.imshow(cam_array, cmap="Greys")
        fig.colorbar(pos, ax=axs)
        if config["debug"] is False:
            plt.close(fig)

        counter += 1        
    logging.debug(f"Slope: {m}")

    if abs(m) > 5:
        logging.error(f"Camera segmentation near instability bounding box failed. Skiping {file_name}")
        status = False
        return status, base_image

    bb_div_y = poly1d_fn(tl_x)
    upper_x = tl_x + width
    upper_y = tl_y
    for i in range(int(tl_y), int(bb_div_y)):
        tmp_line = image_array[i,:]
        if len(np.where(tmp_line == 1)[0]) == 0:
            break
        test = np.where(tmp_line == 1)[0][0]
        if test < upper_x:
            upper_x = test
            upper_y = i

    upper = (upper_x, upper_y)

    lower_x = tl_x + width
    lower_y = tl_y + height

    for i in range(int(bb_div_y), int(tl_y + height)):
        tmp_line = image_array[i,:]
        if len(np.where(tmp_line == 1)[0]) == 0:
            break
        test = np.where(tmp_line == 1)[0][0]
        if test < lower_x:
            lower_x = test
            lower_y = i

    lower = (lower_x, lower_y)

    fig, axs = plt.subplots()
    axs.set_title("Contours + Center")
    axs.plot([cX],[cY],markersize=10, marker='*', color="red"),
    axs.plot([upper_x, lower_x],[upper_y, lower_y],markersize=8, marker='*', color="green"),
    axs.axline((0, poly1d_fn(0)), slope=m)
    axs.plot([tl_x, tl_x+ width, tl_x+ width, tl_x, tl_x], [tl_y, tl_y, tl_y +height, tl_y + height, tl_y])
    axs.plot([tl_x], [bb_div_y], marker="*", color="orange", markersize=10)
    # axs.legend() 
    pos = axs.imshow(contour_image, cmap="Greys")
    fig.colorbar(pos, ax=axs)

    folder = "closed_instability"
    dir_path = os.path.join(config["data_path"], case, folder)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    if config["debug"] is False:
        plt.close(fig)

    if upper != (0,0) and lower != (0,0):
        image_array = cv2.line(image_array, lower, upper, color=(1,0,0), thickness=5)
    
    tmp_image = sitk.GetImageFromArray(image_array)
    cam_seed = [(cX,cY)]
    px_val = tmp_image.GetPixel(cam_seed[0])
    logging.debug(f"Seed value: {px_val}")
    if px_val == 1:
        cam_seed = [(cX, int(poly1d_fn(cX)))]
        px_val = tmp_image.GetPixel(cam_seed[0])
        logging.debug(f"New seed value: {px_val}")
    

    fig, axs = plt.subplots()
    axs.set_title("Closed instability")
    axs.plot([cX],[cY],markersize=10, marker='*', color="red"),
    axs.plot([cX],[poly1d_fn(cX)],markersize=10, marker='*', color="red"),
    # axs.plot([upper_x, lower_x],[upper_y, lower_y],markersize=10, marker='.', color="green")
    pos = axs.imshow(image_array, cmap="Greys")
    fig.colorbar(pos, ax=axs)
    if config["debug"] is False:
        plt.close(fig)

    insta_image = sitk.ConnectedThreshold(
        image1=tmp_image,
        seedList=cam_seed,
        lower=0,
        upper=0.9,
        replaceValue=1
    )
    insta_array = sitk.GetArrayFromImage(insta_image)

    tmp_image_array = image_array + insta_array


    # fill all holes
    # get outer contour
    contours, hierarchy = cv2.findContours(image=tmp_image_array, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # only get the longest contour
    length = 0
    for ele in contours:
        if len(ele) > length:
            contours = [ele]
            length = len(ele)
    contour_image = cv2.drawContours(image=np.zeros(tmp_image_array.shape), contours=contours, contourIdx=0, color=(1, 0, 0), thickness=7, lineType=cv2.LINE_AA)

    # fill everything inside outer contour
    contour_array = sitk.GetImageFromArray(contour_image)
    insta_image = sitk.ConnectedThreshold(
        image1=contour_array,
        seedList=cam_seed,
        lower=0,
        upper=0.9,
        replaceValue=1
    )
    image_array = sitk.GetArrayFromImage(insta_image)

    # check if hole image is filled --> hole in contour that needs to closed
    uni_vals, uni_count = np.unique(image_array, return_counts=True)
    if uni_count[1]/sum(uni_count) > 0.8:
        logging.info("Hole in Contour detected. Trying to close")

        cam_image = sitk.BinaryDilate(
           image1=cam_image,
            backgroundValue=0.0,
            foregroundValue=1.0,
            boundaryToForeground=True,
            kernelRadius=(5,5)
        )
        contour_array = sitk.GetArrayFromImage(tmp_image + cam_image)
        contour_array = np.where(contour_array > 1, 1, contour_array)

        # get outer contour
        contours, hierarchy = cv2.findContours(image=contour_array, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        # only get the longest contour
        length = 0
        for ele in contours:
            if len(ele) > length:
                contours = [ele]
                length = len(ele)
        contour_image = cv2.drawContours(image=np.zeros(contour_array.shape), contours=contours, contourIdx=0, color=(1, 0, 0), thickness=7, lineType=cv2.LINE_AA)
        # fill everything inside outer contour
        contour_array = sitk.GetImageFromArray(contour_image)
        insta_image = sitk.ConnectedThreshold(
            image1=contour_array,
            seedList=cam_seed,
            lower=0,
            upper=0.9,
            replaceValue=1
        )
        image_array = sitk.GetArrayFromImage(insta_image)



    fig, axs = plt.subplots()
    axs.set_title("Filled instability")
    pos = axs.imshow(image_array, cmap="Greys")
    fig.colorbar(pos, ax=axs)
    if config["debug"] is False:
        plt.close(fig)
    
    return status, sitk.GetImageFromArray(image_array)

def refine_instability(config, case, base_image, file_name):
    """
    Function that fills holes in instability segmentation
    """
    tmp_image = base_image
    status = True

    tmp_image = sitk.BinaryErode(
        image1=tmp_image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(1,15)
    )

    tmp_image = sitk.BinaryDilate(
        image1=tmp_image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(1,15)
    )

    # get rid of small artifacts    
    tmp_image = sitk.BinaryErode(
            image1=tmp_image,
            backgroundValue=0.0,
            foregroundValue=1.0,
            kernelRadius=(7,7)
         )
    
    tmp_image = sitk.BinaryDilate(
            image1=tmp_image,
            backgroundValue=0.0,
            foregroundValue=1.0,
            kernelRadius=(7,7)
         )
    
    final_array = sitk.GetArrayViewFromImage(tmp_image)
    bla = final_array.copy()
    offset = 100
    bla[:offset, :] = 0
    bla[bla.shape[0]- offset: bla.shape[0], :] = 0
    bla[:, :offset] = 0
    bla[:, bla.shape[1]- offset: bla.shape[1]] = 0
    final_array = bla

    contours, hierarchy = cv2.findContours(image=final_array, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    tmp_array = final_array.copy()

    fig, axs = plt.subplots()
    axs.set_title(f"Contour Image after dil/er {file_name.split('_')[0]}")
    axs.imshow(tmp_array, cmap="Greys")
    # fig.colorbar(pos, ax=axs)
    if config["debug"] is False:
        plt.close(fig)

    if len(contours) != 1:

        length = 0
        rm_contours = [ele for ele in contours]
        cont_len = [len(ele) for ele in contours]
        for ele in contours:
            if len(ele) > length:
                contours = [ele]
                length = len(ele)
            
        rm_contours.remove(ele)

        for ele in rm_contours:
            bound_rect = cv2.boundingRect(ele)
            logging.debug(f"Remove {bound_rect}")

            tl_x = bound_rect[0]
            tl_y = bound_rect[1]
            width = bound_rect[2]
            height = bound_rect[3]

            tmp_array[tl_y:tl_y+height,tl_x:tl_x+width] = 0
                 
    final_array = tmp_array
    logging.debug(f"Contours_n: {len(contours)}")
    if len(contours) != 1:
        contour_image = cv2.drawContours(image=np.zeros(final_array.shape), contours=contours, contourIdx=-1, color=(1, 0, 0), thickness=7, lineType=cv2.LINE_AA)
        fig, axs = plt.subplots()
        axs.set_title(f"Contour Image {file_name.split('_')[0]}")
        pos = axs.imshow(contour_image, cmap="Greys")
        fig.colorbar(pos, ax=axs)
        if config["debug"] is False:
            plt.close(fig)
        status = False
        return base_image, status

    fig, axs = plt.subplots()
    axs.set_title("Instability after Dilate/Erode")
    pos = axs.imshow(final_array, cmap="Greys")
    fig.colorbar(pos, ax=axs)
    
    folder = "instability"
    dir_path = os.path.join(config["data_path"], case, folder)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    if config["debug"] is False:
        plt.close(fig)
    tmp_image = sitk.GetImageFromArray(final_array)    
    return tmp_image, status

def convex_hull(config, case, base_image, file_name):

    tmp_image = base_image

    image = sitk.GetArrayFromImage(tmp_image)
    
    import skimage as ski

    chull = ski.morphology.convex_hull_image(image)
    fig, axs = plt.subplots()
    axs.set_title("Convex Hull")
    pos = axs.imshow(chull, cmap="Greys")
    fig.colorbar(pos, ax=axs)

    folder = "convex_hulls"
    dir_path = os.path.join(config["data_path"], case, folder)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    if config["debug"] is False:
        plt.close(fig)

    return chull


def segment_camera(config, case, base_image, file_name):
    """
    Function that segements instability from an image
    """
    # cam corner seed
    cam_seed = [(10,10)]
    px_val = base_image.GetPixel(cam_seed[0])
    logging.debug(f"Cam seed value: {px_val}")

    # check for seed inside camera
    fin_val = 255
    fin_x = 0
    fin_y = 0
    test_y = np.linspace(750,1500,100).round(0)
    test_x = np.linspace(750,1700,100).round(0)
    for x in test_x:
        for y in test_y:
            test_seed = [(int(x),int(y))]
            px_val = base_image.GetPixel(test_seed[0])
            if px_val < fin_val:
                fin_val = px_val
                fin_x = int(x)
                fin_y = int(y)

    insta_seed = [(fin_x, fin_y)]
    # cam middle seed
    # insta_seed = [(1350,1150)]
    insta_px_val = base_image.GetPixel(insta_seed[0])
    logging.debug(f"Insta seed value: {insta_px_val}")

    fig, axs = plt.subplots()
    axs.set_title(f"Base Image {file_name.split('_')[0]}")
    pos = axs.imshow(sitk.GetArrayViewFromImage(base_image), cmap="gray")
    fig.colorbar(pos, ax=axs)
    if config["debug"] is False:
        plt.close(fig)

    upper_start = insta_px_val + 1

     # initial segmentation
    cam_image = sitk.ConnectedThreshold(
            image1=base_image,
            seedList=insta_seed,
            lower=0,
            upper=upper_start,
            replaceValue=1
        )
    # logging.info("Initial Camera Segmentation")
    insta_proc_img = sitk.LabelOverlay(base_image, cam_image)
    fig, axs = plt.subplots()
    axs.set_title(f"Initial Cam Seg {int(file_name.split('_')[0])}")
    for ele in insta_seed:
        axs.plot([ele[0]], [ele[1]], marker="*", markersize=8, color="blue")
    axs.imshow(sitk.GetArrayViewFromImage(insta_proc_img))
    if config["debug"] is False:
        plt.close(fig)

    px_n, px_count = np.unique(sitk.GetArrayFromImage(cam_image), return_counts=True)

    if len(px_n) == 1:
        return base_image, False
    
    px_count_old = px_count[1]
    delta = px_count[1] / px_count_old
    i = 0
    step = 1
    tmp_step = 0
    step_thresh = 0.5
    new_limit = upper_start
    delta_data = {
        "deltas" : [delta],
        "limits" : [upper_start+step*i],
        "px_count" : [0]
    }

    cam_array = sitk.GetArrayFromImage(cam_image)
    # get corner values to check if segementation has run to boundary
    px_offset = 50
    lr_cor_val = cam_image.GetPixel(cam_array.shape[1]- px_offset, cam_array.shape[0]- px_offset)
    ur_cor_val = cam_image.GetPixel(cam_array.shape[1]- px_offset, px_offset)

    cam_pixels = 0
    px_thresh = 4e4
    while cam_pixels < px_thresh:
    # while tmp_step < step_thresh and (lr_cor_val == 0 or ur_cor_val == 0) and cam_pixels < px_thresh:
        new_limit = upper_start+step*i

        # lr_cor_val = cam_image.GetPixel(cam_array.shape[1]- px_offset, cam_array.shape[0]- px_offset)
        # ur_cor_val = cam_image.GetPixel(cam_array.shape[1]- px_offset, px_offset)

        cam_image = sitk.ConnectedThreshold(
            image1=base_image,
            seedList=insta_seed,
            lower=0,
            upper=new_limit,
            replaceValue=1
        )
        px_n, px_count = np.unique(sitk.GetArrayFromImage(cam_image), return_counts=True)
        if len(px_n) == 1:
            return base_image, False
        delta = px_count[1] / px_count_old
        tmp_step = abs(delta_data["deltas"][-1] - delta)
        delta_data["limits"].append(new_limit)
        delta_data["deltas"].append(delta)
        tmp_px_n, tmp_px_count = np.unique(sitk.GetArrayFromImage(cam_image), return_counts=True)
        cam_pixels = tmp_px_count[1]
        delta_data["px_count"].append(cam_pixels)

        # if delta > 1.1:
        #     insta_proc_img = sitk.LabelOverlay(base_image, cam_image)
        #     fig, axs = plt.subplots()
        #     axs.set_title(f"Cam Image {int(file_name.split('_')[0])} limit {new_limit} tmp_step {tmp_step.round(2)}")
        #     axs.imshow(sitk.GetArrayViewFromImage(insta_proc_img))
        #     if config["debug"] is False:
        #         plt.close(fig)

        px_count_old = px_count[1]
        if cam_pixels > px_thresh:
            logging.debug(f"Limit: {new_limit} delta = {delta}")
            insta_proc_img = sitk.LabelOverlay(base_image, cam_image)
            fig, axs = plt.subplots()
            axs.set_title(f"Cam Image {int(file_name.split('_')[0])} limit {new_limit} tmp_step {tmp_step.round(2)}")
            axs.imshow(sitk.GetArrayViewFromImage(insta_proc_img))
            if config["debug"] is False:
                plt.close(fig)
        i += 1        

    # final segmentation
    # tmp_delta = pd.DataFrame(delta_data)
    # max_pos = tmp_delta.query("limits < 70")["deltas"].idxmax()
    # new_limit = int(tmp_delta["limits"][max_pos])
    # logging.info(f"Cam final upper value {new_limit}")
    cam_image = sitk.ConnectedThreshold(
    image1=base_image,
        seedList=insta_seed,
        lower=0,
        upper=new_limit,
        replaceValue=1
    )
    uni_vals = np.unique(sitk.GetArrayFromImage(cam_image))
    logging.debug(f"Unique values cam {uni_vals}")

    cam_proc_img = sitk.LabelOverlay(base_image, cam_image)
    fig, axs = plt.subplots()
    axs.set_title(f"Cam Image {file_name.split('_')[0]} limit {new_limit}")
    axs.imshow(sitk.GetArrayViewFromImage(cam_proc_img))

    folder = "segmented_camera"
    dir_path = os.path.join(config["data_path"], case, folder)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    if config["debug"] is False:
        plt.close(fig)

    # save delta data
    fig, axs = plt.subplots()
    axs.set_title(f"Cam Deltas {file_name.split('_')[0]}")
    axs.plot(delta_data["limits"], delta_data["deltas"])
    axs.set_xlabel("upper_limits")
    axs.set_ylabel("segmented pixel growth rate")
    folder = "cam_delta_data"
    dir_path = os.path.join(config["data_path"], case, folder)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    
    if config["debug"] is False:
        plt.close(fig)

    # show px counts

    fig, axs = plt.subplots()
    axs.set_title(f"Px Count {file_name.split('_')[0]}")
    axs.plot(delta_data["limits"], delta_data["px_count"])
    axs.set_xlabel("upper_limits")
    axs.set_ylabel("px_count")
    folder = "cam_px_data"
    dir_path = os.path.join(config["data_path"], case, folder)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    
    if config["debug"] is False:
        plt.close(fig)
    

    return cam_image

def segment_instability(config, case, base_image, file_name, cam_seg):
    """
    Function that segements camera from base image
    """
    status = True

    tmp_image = sitk.BinaryErode(
        image1=cam_seg,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(1,20)
    )

    tmp_image = sitk.BinaryDilate(
        image1=tmp_image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(1,20)
    )


    cam_array = sitk.GetArrayFromImage(tmp_image)
    boundary = 400
    max_x = cam_array.shape[0]
    max_y = cam_array.shape[1]
    # set boundary values to 0 to get rid of artifacts
    # cam_array[:boundary + max_x-boundary:,:boundary + max_y - boundary:] = 0
    cam_array[:boundary, :] = 0
    cam_array[:, :boundary] = 0
    cam_array[max_x-boundary:,:] = 0
    cam_array[:, max_y - boundary:] = 0
    # get bounding box of camera
        # get contours
    contours, hierarchy = cv2.findContours(image=cam_array, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    cont = contours[0]
    bound_rect = cv2.boundingRect(cont)
    logging.debug(bound_rect)

    tl_x = bound_rect[0]
    tl_y = bound_rect[1]
    width = bound_rect[2]
    height = bound_rect[3]

    # get seeds for instabiltiy segmentation around camera

    # get axes of camera
    y,x = np.where(cam_array == 1)
    coef = np.polyfit(x,y,1)
    m = coef[0]
    poly1d_fn = np.poly1d(coef)

    seeds = []

    # calculate seeds for instability
    # set x values relative to left camera boundary    
    seed_x = [tl_x + 50, tl_x + 150, tl_x + 225]

    for ele in seed_x:
        offset = 10
        tmp_x = ele
        tmp_y_up = int(poly1d_fn(ele) - height/2 - offset) 
        tmp_y_low = int(poly1d_fn(ele) + height/2 + offset)
        # shift y up and down until they or far enough away from camera 
        while cam_array[tmp_y_up, tmp_x] != 0 and cam_array[tmp_y_low, tmp_x] != 0:
            offset += 10
            tmp_y_up = int(poly1d_fn(ele) - height/2 - offset) 
            tmp_y_low = int(poly1d_fn(ele) + height/2 + offset)

        # add a bit of extra distance
        offset += 30
        tmp_y_up = int(poly1d_fn(ele) - height/2 - offset) 
        tmp_y_low = int(poly1d_fn(ele) + height/2 + offset)
        seeds.append((ele, tmp_y_up))
        seeds.append((ele, tmp_y_low))

    logging.debug(f"Insta Seeds : {seeds}")
    fig, axs = plt.subplots()
    axs.set_title("Cam position")
    for ele in seeds:
        axs.plot([ele[0]], [ele[1]], marker="*", markersize=8, color="blue")
    axs.axline((0, poly1d_fn(0)), slope=m)
    axs.plot([tl_x, tl_x+ width, tl_x+ width, tl_x, tl_x], [tl_y, tl_y, tl_y +height, tl_y + height, tl_y])
    pos = axs.imshow(cam_array, cmap="Greys")
    fig.colorbar(pos, ax=axs)
    if config["debug"] is False:
        plt.close(fig)

    insta_seed = seeds

    px_vals = np.array([base_image.GetPixel(ele) for ele in insta_seed])
    px_val = px_vals[(px_vals > 0) & (px_vals < 200)]
    if px_val.size == 0:
        return base_image, False
    else:
        px_val = int(px_val.mean().round())
    logging.debug(f"Insta seed value: {px_val}")
    lower_limit = 1
    if px_val > 230:
        return base_image, False
    else:
        upper_start = px_val + 20
    # initial segmentation
    insta_image = sitk.ConnectedThreshold(
            image1=base_image,
            seedList=insta_seed,
            lower=lower_limit,
            upper=upper_start,
            replaceValue=1
        )
    insta_array = sitk.GetArrayFromImage(insta_image)
    px_n, px_count = np.unique(insta_array, return_counts=True)

    if len(px_n) == 1:
        return base_image, False
    
    px_count_old = px_count[1]
    delta = px_count[1] / px_count_old
    i = 0
    step = 5
    tmp_step = 0
    step_thresh = 0.5
    new_limit = upper_start
    delta_data = {
        "deltas" : [delta],
        "limits" : [upper_start+step*i]
    }

    # get corner values to check if segementation has run to boundary
    px_offset = 50
    lr_cor_val = insta_image.GetPixel(insta_array.shape[1]- px_offset, insta_array.shape[0]- px_offset)
    ur_cor_val = insta_image.GetPixel(insta_array.shape[1]- px_offset, px_offset)

    while tmp_step < step_thresh and (lr_cor_val == 0 or ur_cor_val == 0):
        new_limit = upper_start+step*i
        lr_cor_val = insta_image.GetPixel(insta_array.shape[1]- px_offset, insta_array.shape[0]- px_offset)
        ur_cor_val = insta_image.GetPixel(insta_array.shape[1]- px_offset, px_offset)

        insta_image = sitk.ConnectedThreshold(
            image1=base_image,
            seedList=insta_seed,
            lower=lower_limit,
            upper=new_limit,
            replaceValue=1
        )
        px_n, px_count = np.unique(sitk.GetArrayFromImage(insta_image), return_counts=True)
        if len(px_n) == 1:
            return base_image, False
        delta = px_count[1] / px_count_old
        tmp_step = abs(delta_data["deltas"][-1] - delta)
        delta_data["limits"].append(new_limit)
        delta_data["deltas"].append(delta)
        px_count_old = px_count[1]
        if tmp_step > step_thresh and (lr_cor_val == 1 or ur_cor_val == 1):
            logging.debug(f"Limit: {new_limit} delta = {delta}")
            insta_proc_img = sitk.LabelOverlay(base_image, insta_image)
            fig, axs = plt.subplots()
            axs.set_title(f"Insta Image {int(file_name.split('_')[0])} limit {new_limit} last_delta {delta.round(2)}")
            axs.imshow(sitk.GetArrayViewFromImage(insta_proc_img))
            
            if config["debug"] is False:
                plt.close(fig)
        i += 1
        
        # get image for every iteration
         
        # insta_proc_img = sitk.LabelOverlay(base_image, insta_image)
        # fig, axs = plt.subplots()
        # axs.set_title(f"Insta Image {int(file_name.split('_')[0])} limit {new_limit} last_delta {delta.round(2)}")
        # axs.imshow(sitk.GetArrayViewFromImage(insta_proc_img))
        # if config["debug"] is False:
        #     plt.close(fig)
        

    # final segmentation
    new_limit = upper_start+step*(i-4)
    insta_image = sitk.ConnectedThreshold(
    image1=base_image,
        seedList=insta_seed,
        lower=lower_limit,
        upper=new_limit,
        replaceValue=1
    )
    seg_array = sitk.GetArrayFromImage(insta_image)
    uni_vals, uni_counts = np.unique(seg_array, return_counts=True)

    if len(uni_vals) > 1 and uni_counts[-1] < 100:
        return base_image, False 
    logging.debug(f"Unique values insta {uni_vals}, {uni_counts}")

    insta_proc_img = sitk.LabelOverlay(base_image, insta_image)
    
    fig, axs = plt.subplots()
    axs.set_title(f"Insta Image {int(file_name.split('_')[0])} limit {new_limit} last_delta {delta.round(2)}")
    for ele in insta_seed:
        axs.plot([ele[0]], [ele[1]], marker="*", markersize=8, color="blue")
    axs.imshow(sitk.GetArrayViewFromImage(insta_proc_img))
    folder = "segmented_instability"
    dir_path = os.path.join(config["data_path"], case, folder)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}") 
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    if config["debug"] is False:
        plt.close(fig)

    # save raw segmentation
    tmp_path = os.path.join(config["data_path"], case ,"raw_instability", file_name + ".png")
    plt.imsave(tmp_path, sitk.GetArrayFromImage(insta_image), cmap="Greys", dpi=1200)
    logging.info(f"Saved raw instability {file_name}")

    # save delta data
    fig, axs = plt.subplots()
    axs.set_title(f"Insta Deltas {file_name.split('_')[0]}")
    axs.plot(delta_data["limits"], delta_data["deltas"])
    axs.set_xlabel("upper_limits")
    axs.set_ylabel("segmented pixel growth rate")
    folder = "insta_delta_data"
    dir_path = os.path.join(config["data_path"], case, folder)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    
    if config["debug"] is False:
        plt.close(fig)

    return insta_image, status

def make_histo(config, case, folder, name) -> bool:
    """
    Function that creates histogram from image values
    """
    
    base_path=os.path.join(config["data_path"], case, folder)
    if os.path.exists(base_path) is False:
        os.makedirs(base_path)
    hist_path = os.path.join(base_path, name + "_hist.png")
    if os.path.exists(hist_path):
        logging.info(f"Already created histogramm for image {name} in folder {folder}")
        return True
    img = read_image(config, folder, name, case)
    nums, counts = np.unique(img, return_counts=True)
    plt.hist(nums, nums, weights=counts)   
    plt.savefig(hist_path)
    logging.info(f"Created histogramm for image {name} in folder {folder}")
    if config["debug"] is False:
        plt.close()
    return True

def add_area2results():
    """
    Function that adds the area from the segementation algorithm to the finger data results file
    """
    config = get_config()

    # read finger data results file
    res_path = os.path.join(config["results_path"], "finger_data", "all_data.csv")
    if os.path.exists(res_path) is False:
        logging.error(f"No results file found at {res_path}. You might need an active VPN connection.")
    else:
        res = pd.read_csv(res_path, delimiter='\t')
    tmp = res.copy()
    tmp["A_insta_segment"] = np.nan
    tmp["img_x"] = np.nan
    tmp["img_y"] = np.nan
    # loop through each line and try to add area 
    for idx, row in res.iterrows():
        # build path to resulting segmentation image
        tmp_path = os.path.join(config["results_path"], "final_data", row.case, "instabilities", row.img_name + ".png")
        if os.path.exists(tmp_path):
            img = read_image(config, "instabilities", str(row.img_name), str(row.case))
            tmp_vals, tmp_counts = np.unique(img, return_counts=True)
            tmp.loc[idx, "A_insta_segment"] = float(tmp_counts[0])
            tmp.loc[idx, "img_x"] = float(img.shape[0])
            tmp.loc[idx, "img_y"] = float(img.shape[1])
            logging.info(f"Added {float(tmp_counts[0])} for img {row.img_n} for case {row.case}")
        else:
            continue
    logging.info(tmp.info())
    tmp.to_csv(res_path, sep="\t")
    logging.info(f"Saved results at {res_path}")

def remove_empty_folders(path_abs):
    """
    Function that removes all empty folders from a path downwards
    """
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)
            logging.info(f"Removed {path}")

def create_animation(config, case, data_folder):
    """
    Function that creates animation video for images
    """
    
    images = get_image_files(config, case, data_folder)
    if images == []:
        logging.warning(f"No images found to create animation for case {case}.")
        return
    animation_folder_path = os.path.join(config["results_path"], "animations", case)
    if os.path.exists(animation_folder_path) is False:
        os.makedirs(animation_folder_path)

    if data_folder == "png_cases":
        dat_path = os.path.join(config["raw_data_path"], data_folder, case)
    else:
        dat_path = os.path.join(config["results_path"], "final_data", case, data_folder)
    frame = cv2.imread(os.path.join(dat_path, images[0]+ ".png"))
    height, width, layers = frame.shape
    fps = 5
    video_path = os.path.join(animation_folder_path, data_folder + ".avi")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for image in images:
        img = cv2.imread(os.path.join(dat_path, image + ".png"))
        out.write(img)
    cv2.destroyAllWindows()
    out.release()
    logging.info(f"Created {data_folder} video for case {case}")

if __name__ == "__main__":
    # config = get_config()
    # calc_case_ratio()
    # config = get_config()
    # get_all_cases(config)
    # rename_cases(config)
    # multi_contour_plot()
    add_area2results()
