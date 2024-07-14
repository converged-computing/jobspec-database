#!/Python 3.7.1
# -*- coding: utf-8 -*-
"""
this file is to download CONUS 2.5km data from noaa NDFD database. 
"""

from bs4 import *
from urllib.request import *
import re
import requests
import os 
import shutil
import multiprocessing 
from itertools import repeat
import tqdm
import logging
# for Unix Path
from pathlib import Path

def initFile(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
        # os.mknod requires root
        #os.mknod(file_path)
    open(file_path, 'a').close()

def initDir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
    logger.info(f'the directory {dir_path} is created!')

def getYear(url, path2save, year):
    content = urlopen(url).read()
    soup = BeautifulSoup(content, features="lxml")
    #the file path helps to create dir
    folderPath_list = []
    #help with the urls
    url_list = []
    for j in soup.findAll('a'):
        if j['href'][:4] == year:
            folder_path = Path(path2save + year + '/' + j['href'])
            folderPath_list.append(folder_path)
            url_path = url + j['href']
            url_list.append(url_path)
            logger.info('the year {0} folder has following underfolder {1}'.format(year, j['href']))
            logger.info(f'the url path of the folder is {url_path}.')
    return folderPath_list, url_list

def getMonth(url, folderPath, year):
    content_new = urlopen(url).read()
    soup_new = BeautifulSoup(content_new, features="lxml")
    folder_path_list = []
    day_url_list = []
    for j in soup_new.findAll('a'):
        #select the folder with number
        if any(i.isdigit() for i in j['href']):
            folder_path = Path(folderPath + j['href'])
            folder_path_list.append(folder_path)
            url_path = url + j['href']
            day_url_list.append(url_path)
            logger.info('the year folder {0} : folder {1}  havs following underfolder: {2}.'.format(year, folderPath, j['href']))
            logger.info(f'the url path of the folder is {url_path}.')
        else:
            logger.debug('the folder{} is not going to be created.'.format(j['href']))

    return folder_path_list, day_url_list

def getParaFile(Para_dict, url, folder_path):
    content = urlopen(url).read()
    soup_new = BeautifulSoup(content, features="lxml")
    for j in soup_new.findAll('a'):
        if j['href'][:6] in Para_dict.keys():
            file_download_url = folder_path + Para_dict.get(j['href'][:6])
            file_save_name = Path(file_download_url + '/' + j['href'] + '.dms')
            logger.info('the file {} is sorted in folder {}.'.format(j['href'], file_download_url))
            logger.info('start downloading the file {}.'.format(j['href']))
            #download the file
            downloadfile = URLopener()
            downloadfile.retrieve(file_download_url, file_save_name)


def creatListFolder(folder_list):
    for folder_path in folder_list:
        initDir(folder_path)
        

def CreatYearFolder(year_list, path2save):
    for year in year_list:
        path = path2save / year
        initDir(path)
    

if __name__ == '__main__':
    #get logger
    logfile_path = Path('/Users/sk/Downloads/debug_save_path_conus2.5/CONUS_2.5_Download_v2.log')
    initFile(logfile_path)
    logging_level = logging.INFO
    logger = logging.getLogger('conus_2.5 Download test')
    logger.setLevel(logging_level)
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(logging_level)
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('start to download the file!')

    #set path to save the data
    path2save = Path('/Users/sk/Downloads/debug_save_path_conus2.5/v2/')
    initDir(path2save)
    url = "https://nomads.ncdc.noaa.gov/data/ndfd/"
    #give the wanted year data
    year_list = ['2015', '2016']
    CreatYearFolder(year_list, path2save)
    #seperate run because the low speed. in order: ['YBUZ98', 'YCUZ98', 'YEUZ98']
    Para_dict = {'YBUZ98': 'YBU/', 'YCUZ98': 'YCU/', 'YEUZ98': 'YEU/'}

    #multiprocessing pool
    cores = multiprocessing.cpu_count()
    p = multiprocessing.Pool(processes=cores)

    for year in year_list:
        folderPath_list, url_list= getYear(url, path2save, year)
        #using multiprocessing to create folder
        p.starmap(initDir, (folderPath_list))
        #p.join()
        logger.info(f'Month folder finished! Now is going to create Day folder:')
        folder_path_list, day_url_list = p.starmap(getMonth, (url_list, folderPath_list, [year]*len(url_list)))
        logger.debug(f'folder shape: {folder_path_list}.shape.')
        p.join()
        r = p.starmap(initDir, (folder_path_list))
        p.join()
        logger.info(f'day folder finished! Now is starting to create para folder!')
        for key, value in Para_dict.items():
            r = p.starmap(initDir, list(map(lambda x: x+value, folder_path_list)))
            p.join()
            logger.info(f'the paramter {value} folders are created!')
        
        r = p.starmap_async(getParaFile, zip(repeat(Para_dict),day_url_list, folder_path_list))
            
        p.close()
        p.join()
        
        
        

                    
                    



   
