import os
import shutil
import json
import glob
import logging
import math
import platform

from multiprocessing import Pool
from functools import partial
from methods import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s")


def proc_cases():

    config = get_config()

    for cas in list(config["cases"]):

        create_intermediate_finger_folders(config, cas)

        logging.info("-------------------")
        logging.info(f"Starting processing case {cas}")
        logging.info("-------------------")

        images = get_image_files(config, cas, "png_cases")
        if images == []:
            images = get_image_files(config, cas, "raw_cases")
        if images == []:
            logging.warning(f"No images found for case {cas}")
            continue
        images.sort()

        tmp_conf = config.copy()
        tmp_conf["images"] = []
        background_images = get_image_files(tmp_conf, cas, "png_cases")
        if background_images == []:
            background_images = get_image_files(tmp_conf, cas, "raw_cases")
        if background_images == []:
            logging.warning(f"No images found for case {cas}")
            continue

        background_img = get_base_image(config, cas, background_images[0])
        res = pd.DataFrame({})
        
        if config["debug"] is False and len(config["images"]) == 0:
            logging.info(f"Running on platform {platform.system()}")
            if platform.system() == "Windows":
                parallel = True
            else:
                parallel = False
        else:
            parallel = False
            
        if len(config["images"]) == 0:
            config["debug"] = False
        # else:
        #     config["debug"] = True

        if parallel:
            cpus = os.cpu_count()
            p = Pool(cpus)
            for data in p.map(partial(get_fingers, config=config, case=cas, background_img=background_img), images):
                logging.debug(f"Multi res : {data}")
                res = pd.concat([res,data])
        else:
            for img in images:
                data = get_fingers(img, config, cas, background_img=background_img)
                res = pd.concat([res, data])
                
        logging.info(f"Res : {res}")
        if len(res["ratio"]) > 4:
            fig, axs = plt.subplots()
            axs.set_title(f"Finger ratio {cas}")
            axs.plot(res["img_n"], res["ratio"])
            axs.set_ylim(0,1)
            fig_path = os.path.join(config["results_path"], "finger_data", cas, "plots", "ratio.png")
            plt.savefig(fig_path)
            logging.info(f"Saved ratio fig at {fig_path}")
            if config["debug"] is False:
                plt.close(fig)
            else:
                plt.show()

            if len(config["images"]) == 0:
                csv_path = os.path.join(config["results_path"], "finger_data", cas, "ratio", "ratio.csv")
                res.to_csv(csv_path, index=False, sep="\t")
                logging.info(f"Saved ratio data at {csv_path}")



def save_intermediate(config, case, img_name, array, folder):
    """
    Function that saves intermediate to folder
    """
    im_path = os.path.join(config["data_path"], case, folder, img_name + ".png")
    if os.path.exists(im_path) is False:
        logging.info(f"Saved {img_name} at {folder}")
        plt.imsave(im_path, array, cmap="Greys", dpi=1200)

def substract_background(config, case, img_name, background_img) -> sitk.Image:
    """
    Function that substracts background (first image of series) from current image 
    """

    def plot():
        fig, axs = plt.subplots()
        axs.set_title(f"Background substracted {img_name.split('_')[0]}")
        axs.imshow(sitk.GetArrayFromImage(image), cmap="Greys")
        pos = axs.imshow(sitk.GetArrayFromImage(image), cmap="Greys")
        fig.colorbar(pos, ax=axs)
        if config["debug"] is False:
            plt.close(fig)
        else:
            plt.show()

    img_path = os.path.join(config["data_path"], case, "background_substracted" ,img_name + ".png")
    image = None
    if os.path.exists(img_path):
        image = read_image(config, "background_substracted", img_name, case)
        image = sitk.GetImageFromArray(cv2.bitwise_not(image))
    if image is not None and config["new_files"] is False:
        logging.info(f"Background substraction already done for image {img_name}")
        plot()
        return image

    if os.path.exists(img_path) and config["new_files"] is False:
        logging.info(f"Already did background substraction for image {img_name}")
        image = read_image(config, "background_substracted", img_name, case)
        image = sitk.GetImageFromArray(cv2.bitwise_not(image))
        plot()
        return image

    # get background image array
    background_array = sitk.GetArrayFromImage(background_img)
    base_img = get_base_image(config, case, img_name)
    base_array = sitk.GetArrayFromImage(base_img)

    # substract bckground from image
    new_image = base_array - background_array

    new_image = delete_border(sitk.GetImageFromArray(new_image))

    # plt background substracted image
    fig, axs = plt.subplots()
    axs.set_title(f"Background substracted {img_name.split('_')[0]}")
    pos = axs.imshow(sitk.GetArrayFromImage(new_image), cmap="Greys")
    fig.colorbar(pos, ax=axs)
    if config["debug"] is False:
        plt.close(fig)
    else:
        plt.show()

    if config["save_intermediate"]:
        save_intermediate(config, case, img_name, sitk.GetArrayFromImage(new_image), "background_substracted")

    return new_image

def delete_border(img) -> sitk.Image:
    """
    Function that sets the outer border values of image to 0
    """
    img_array = sitk.GetArrayFromImage(img)
    offset = 100
    img_array[:offset, :] = 0
    img_array[img_array.shape[0]- offset: img_array.shape[0], :] = 0
    img_array[:,:offset] = 0
    img_array[:, img_array.shape[1]- offset: img_array.shape[1]] = 0

    final_img = sitk.GetImageFromArray(img_array)
    
    return final_img

def create_intermediate_finger_folders(config, cas):
    """
    Function that creates all intermediate folders
    """
    # create intermediate folders
    folders = ["background_substracted", "finger_image", "int_window", "cleaned_artifacts", "closed_fingers"]
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

    # create raw_data folder
    folder_path = os.path.join(config["raw_data_path"], "png_cases", cas)
    if os.path.exists(folder_path) is False:
        os.makedirs(folder_path)

    # create folders for results
    final_folders = ["finger_data"]
    for fld in final_folders:
        folder_path = os.path.join(config["results_path"], fld, cas)
        if os.path.exists(folder_path) is False:
            os.makedirs(folder_path)
        folder_path = os.path.join(config["results_path"], fld, cas, "ratio")
        if os.path.exists(folder_path) is False:
            os.makedirs(folder_path)
        folder_path = os.path.join(config["results_path"], fld, cas, "plots")
        if os.path.exists(folder_path) is False:
            os.makedirs(folder_path)



def int_window(config, case, img_name, base_image):
    """
    Function that applys intensity window to image
    """
    lower_threshold = 70
    upper_threshold = 250

    def plot():
        fig, axs = plt.subplots()
        axs.set_title(f"Intensity window {img_name.split('_')[0]}")
        axs.imshow(sitk.GetArrayViewFromImage(image), cmap="Greys")
        if config["debug"] is False:
            plt.close(fig)
        else:
            plt.show()

    img_path = os.path.join(config["data_path"], case, "int_window" ,img_name + ".png")
    image = None
    if os.path.exists(img_path):
        image = read_image(config, "int_window", img_name, case)
        image = sitk.GetImageFromArray(cv2.bitwise_not(image))
    if image is not None and config["new_files"] is False:
        plot()
        logging.info(f"Intensity filtering already done for image {img_name}")
        return image

    if os.path.exists(img_path) and config["new_files"] is False:
        logging.info(f"Already did intensity window for image {img_name}")
        image = read_image(config, "int_window", img_name, case)
        image = sitk.GetImageFromArray(cv2.bitwise_not(image))
        plot()
        return image
    
    if case in list(config["limits"].keys()):
        lower_threshold = config["limits"][case]["low"]
        upper_threshold = config["limits"][case]["high"]
        logging.info(f"Found int limits: {lower_threshold}, {upper_threshold}")
    else:
        logging.info(f"Using default limits {lower_threshold}, {upper_threshold}")


    # Create a binary threshold filter
    threshold_filter = sitk.ThresholdImageFilter()

    # Set the lower and upper thresholds
    threshold_filter.SetLower(lower_threshold)
    threshold_filter.SetUpper(upper_threshold)
    # threshold_filter.SetInsideValue(1)  # Value for pixels within the threshold range
    threshold_filter.SetOutsideValue(0)  # Value for pixels outside the threshold range

    # Apply the threshold filter to the input image
    output_image = threshold_filter.Execute(base_image)

    output_image = sitk.BinaryThreshold(
        image1=output_image,
        lowerThreshold=lower_threshold,
        upperThreshold=upper_threshold,
        insideValue=1,
        outsideValue=0
    )

    fig, axs = plt.subplots()
    axs.set_title(f"Intensity window {img_name.split('_')[0]}")
    axs.imshow(sitk.GetArrayViewFromImage(output_image), cmap="Greys")
    if config["debug"] is False:
        plt.close(fig)
    else:
        plt.show()
    if config["save_intermediate"]:
        save_intermediate(config, case, img_name, sitk.GetArrayFromImage(output_image), "int_window")

    return output_image

def reset_cases():
    """
    Function that resets case while deleting all images within tmp_data and png_cases folder
    """
    config = get_config()

    for cas in config["cases"]:
        logging.info("----------------------")
        logging.info(f"Start cleaning case {cas}")
        logging.info("----------------------")

        raw_path = os.path.dirname(os.path.join(config["data_path"]))
        folders = glob.glob("*" + os.sep + "*"+ os.sep + cas + "*" + os.sep, root_dir=raw_path) + glob.glob("*" + os.sep + "*" +cas + "*" + os.sep, root_dir=raw_path)
        for fld_path in folders:
            # dont remove raw data
            if "raw_cases" in fld_path or "png_cases" in fld_path:
                continue
            else:
                # only remove calculated data
                shutil.rmtree(os.path.join(raw_path, fld_path), ignore_errors=True)
                logging.info(f"Removed {fld_path} for case {cas}")
    logging.info("Cleaning cases finished")
    

def clean_artifacts(config, case, img_name, base_image) -> sitk.Image:
    """
    Function that gets rid of small artifacts within the image by Erosion and Dilation
    """

    def plot():
        fig, axs = plt.subplots()
        axs.set_title(f"Cleaned artifacts {img_name.split('_')[0]}")
        axs.imshow(sitk.GetArrayViewFromImage(image), cmap="Greys")
        if config["debug"] is False:
            plt.close(fig)
        else:
            plt.show()

    img_path = os.path.join(config["data_path"], case, "cleaned_artifacts" ,img_name + ".png")
    image = None
    if os.path.exists(img_path):
        image = read_image(config, "cleaned_artifacts", img_name, case)
        image = sitk.GetImageFromArray(cv2.bitwise_not(image))
    if image is not None and config["new_files"] is False:
        plot()
        logging.info(f"Cleaning already done for image {img_name}")
        return image
    
    if os.path.exists(img_path) and config["new_files"] is False:
        logging.info(f"Already clean artifacts for image {img_name}")
        image = read_image(config, "cleaned_artifacts", img_name, case)
        image = sitk.GetImageFromArray(cv2.bitwise_not(image))
        plot()
        return image

    px_len = 3
    times = 1

    # get rid of small artifacts by dilation and erosion
    image = sitk.BinaryErode(
        image1=base_image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(px_len,px_len)
    )

    image = sitk.BinaryDilate(
        image1=image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(px_len,px_len)
    )

    for i in range(times - 1):
        image = sitk.BinaryErode(
        image1=image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(px_len,px_len)
        )

        image = sitk.BinaryDilate(
            image1=image,
            backgroundValue=0.0,
            foregroundValue=1.0,
            boundaryToForeground=True,
            kernelRadius=(px_len,px_len)
        )

    # get rid of border artifacts created from cleaning
    image = delete_border(image)

    plot()

    if config["save_intermediate"]:
        save_intermediate(config, case, img_name, sitk.GetArrayFromImage(image), "cleaned_artifacts")
    
    return image

def close_finger(config, case, img_name, base_img) -> sitk.Image:
    """
    Function that closes all the fingers within an image
    """

    def plot():
        fig, axs = plt.subplots()
        axs.set_title(f"Finger closed {img_name.split('_')[0]}")
        axs.imshow(sitk.GetArrayViewFromImage(image), cmap="Greys")
        if config["debug"] is False:
            plt.close(fig)
        else:
            plt.show()

    img_path = os.path.join(config["data_path"], case, "closed_fingers" ,img_name + ".png")
    image = None
    if os.path.exists(img_path):
        image = read_image(config, "closed_fingers", img_name, case)
        image = sitk.GetImageFromArray(cv2.bitwise_not(image))
    if image is not None and config["new_files"] is False:
        plot()
        logging.info(f"Fingers already closed for image {img_name}")
        return image
    
    if os.path.exists(img_path) and config["new_files"] is False:
        logging.info(f"Already closed fingers for image {img_name}")
        image = read_image(config, "closed_fingers", img_name, case)
        image = sitk.GetImageFromArray(cv2.bitwise_not(image))
        plot()
        return image
    

    px_len = 25

    image = sitk.BinaryDilate(
        image1=base_img,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(px_len,px_len)
    )

    image = sitk.BinaryErode(
        image1=image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(px_len,px_len)
    )

    plot()
    if config["save_intermediate"]:
        save_intermediate(config, case, img_name, sitk.GetArrayFromImage(image), "closed_fingers")

    return image

def diff_image(config, case, img_name, cleaned_image, closed_fingers, base_image):
    """
    Function that plots the finger image
    """

    calc_img = closed_fingers - cleaned_image

    # clos_arr = sitk.GetArrayFromImage(closed_fingers)
    # clos_arr[np.where(clos_arr == 255)] = 1

    # clean_arr = sitk.GetArrayFromImage(cleaned_image)
    # clean_arr[np.where(clean_arr == 255)] = 1

    # calc_img = sitk.GetImageFromArray(clos_arr - clean_arr)

    # diff_img = sitk.LabelOverlay(base_image, calc_img)
    diff_img = calc_img

    fig, axs = plt.subplots()
    axs.set_title(f"Finger diff {img_name.split('_')[0]}")
    axs.imshow(sitk.GetArrayViewFromImage(diff_img), cmap="Greys")
    if config["debug"] is False:
        plt.close(fig)
    else:
        plt.show()
    if config["save_intermediate"]:
        save_intermediate(config, case, img_name, sitk.GetArrayFromImage(diff_img), "finger_image")

    return diff_img

def combine_results():
    """
    Function that combines all results into one .csv file
    """
    config = get_config()
    # create empty results DataFrame
    res = pd.DataFrame({})

    # find all csv files within results folder for all processed cases
    path = os.path.join(config["results_path"], "finger_data")
    csvs = glob.glob("*" + os.sep+ "*"+ os.sep+ "*.csv", root_dir=path)
    for file in csvs:
        tmp_path = os.path.join(path, file)
        if os.path.exists(tmp_path):
            tmp_data = pd.read_csv(tmp_path, sep="\t")
            res = pd.concat([res, tmp_data], ignore_index=True)
    # res = res.reset_index()
    # Drop all columns with no values in at least one of the columns
    res.dropna()
    logging.info(res.info())

    # adopt path for windows systems
    bigdata_path = "//gssnas/bigdata"
    res.finger_img_path = res.finger_img_path.replace({'/bigdata': bigdata_path}, regex=True)
    res.total_img_path = res.total_img_path.replace({'/bigdata': bigdata_path}, regex=True)
    res.finger_img_path = res.finger_img_path.replace({'/': '\\\\'}, regex=True)
    res.total_img_path = res.total_img_path.replace({'/': '\\\\'}, regex=True)

    # Check if path transformation was successfull
    if os.path.exists(res.finger_img_path[10]):
        logging.info("Path transformation successfull")
    else:
        logging.warning("Path transfomration not successfull")
    
    # Save combined pandas dataframe to csv file
    res_path = os.path.join(path, "all_data.csv")
    res.to_csv(res_path, index=False, sep="\t")
    logging.info(f"Saved combined data to {res_path}")


def get_fingers(img_name, config, case, background_img) -> pd.DataFrame():
    """
    Method that gets the finger information for one image
    """
    res = {
        "img_n" : [int(img_name.split("_")[0])],
        "img_name" : [img_name],
        "case" : [case]
    }

    logging.info(f"Substracting background for image {img_name}")
    base_image = substract_background(config, case, img_name, background_img)

    logging.info(f"Applying intensity window for image {img_name}")
    windowed_image = int_window(config, case, img_name, base_image)
    
    logging.info(f"Cleaning artifacts for image {img_name}")
    cleaned_image = clean_artifacts(config,case, img_name, windowed_image)

    logging.info(f"Closing fingers for image {img_name}")
    closed_fingers = close_finger(config,case, img_name, cleaned_image)

    logging.info(f"Calculating finger ratio for {img_name}")
    fing_px_val, fing_px_n = np.unique(sitk.GetArrayFromImage(cleaned_image), return_counts=True)
    cls_px_val, cls_px_n = np.unique(sitk.GetArrayFromImage(closed_fingers), return_counts=True)

    logging.info(f"Calculating diff image for {img_name}")
    diff_img = diff_image(config, case, img_name, cleaned_image, closed_fingers, base_image)

    if len(fing_px_val) > 1 and len(cls_px_val) > 1:
        ratio = round(fing_px_n[1]/cls_px_n[1], 3)
        res["A_finger"] = [fing_px_n[1]]
        res["A_total"] = [cls_px_n[1]]
        res["d_finger"] = [math.sqrt(4*res["A_finger"][0]/math.pi)]
        res["d_total"] = [math.sqrt(4*res["A_total"][0]/math.pi)]
        res["ratio"] = [ratio]
        if case not in list(config["limits"].keys()):
            res["thr_low"] = 70
            res["thr_high"] = 250
        else:
            res["thr_low"] = config["limits"][case]["low"]
            res["thr_high"] = config["limits"][case]["high"]
        res["finger_img_path"] = [os.path.join(config["data_path"], case, "cleaned_artifacts", img_name + ".png")]
        res["total_img_path"] = [os.path.join(config["data_path"], case, "finger_image", img_name + ".png")]
        logging.info(f"Finger ratio for {img_name}: {ratio}")
        return pd.DataFrame(res)
    else:
        logging.warning(f"No ratio calculation possible for image {img_name}")
        return pd.DataFrame(res)
if __name__ == "__main__":
    # reset_cases()
    proc_cases()
    # combine_results()
