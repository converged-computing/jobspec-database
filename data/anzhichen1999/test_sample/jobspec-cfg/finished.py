# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:23:27 2024
@author: Admin

2024-05-11
@author: caz
"""
import os
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor
import csv
import pandas as pd
from urllib import parse
from threading import Lock
def get_search_list(keyword, lat, lon, file_path):
    global history_list
    keyword_str = parse.quote(keyword)
    page = 0
    z = 448792
    url = (
        f"https://www.google.com/search?tbm=map&authuser=0&hl=zh-CN&gl=us&pb=!4m12!1m3!1d{z}!"
        f"2d{lon}!3d{lat}!2m3!1f0!2f0!3f0!3m2!1i1920!2i637!4f13.1!7i20!8i0!10b1!12m16!1m1!"
        "18b1!2m3!5m1!6e2!20e3!10b1!12b1!13b1!16b1!17m1!3e1!20m3!5e2!6b1!14b1!19m4!2m3!"
        "1i360!2i120!4i8!20m57!2m2!1i203!2i100!3m2!2i4!5b1!6m6!1m2!1i86!2i86!1m2!1i408!"
        "2i240!7m42!1m3!1e1!2b0!3e3!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e8!2b0!3e3!1m3!"
        "1e10!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e9!2b1!3e2!1m3!1e10!2b0!3e3!1m3!1e10!2b1!3e2!"
        "1m3!1e10!2b0!3e4!2b1!4b1!9b0!22m6!1sSlrAZfGlA8-S0PEPk52koA8%3A3007!2s1i%3A0%2Ct%3A20588!"
        "2Cp%3ASlrAZfGlA8-S0PEPk52koA8%3A3007!4m1!2i20588!7e81!12e3!24m94!1m29!13m9!2b1!"
        "3b1!4b1!6i1!8b1!9b1!14b1!20b1!25b1!18m18!3b1!4b1!5b1!6b1!9b1!12b1!13b1!14b1!"
        "15b1!17b1!20b1!21b1!22b1!25b1!27m1!1b0!28b0!31b0!10m1!8e3!11m1!3e1!14m1!3b1!"
        "17b1!20m2!1e3!1e6!24b1!25b1!26b1!29b1!30m1!2b1!36b1!39m3!2m2!2i1!3i1!43b1!"
        "52b1!54m1!1b1!55b1!56m2!1b1!3b1!65m5!3m4!1m3!1m2!1i224!2i298!71b1!72m17!1m5!"
        "1b1!2b1!3b1!5b1!7b1!4b1!8m8!1m6!4m1!1e1!4m1!1e3!4m1!1e4!3sother_user_reviews!"
        "9b1!89b1!103b1!113b1!114m3!1b1!2m1!1b1!117b1!122m1!1b1!26m4!2m3!1i80!2i92!4i8!"
        "30m28!1m6!1m2!1i0!2i0!2m2!1i530!2i637!1m6!1m2!1i1870!2i0!2m2!1i1920!2i637!"
        "1m6!1m2!1i0!2i0!2m2!1i1920!2i20!1m6!1m2!1i0!2i617!2m2!1i1920!2i637!31b1!34m18!"
        "2b1!3b1!4b1!6b1!8m6!1b1!3b1!4b1!5b1!6b1!7b1!9b1!12b1!14b1!20b1!23b1!25b1!26b1!"
        "37m1!1e81!42b1!46m1!1e10!47m0!49m7!3b1!6m2!1b1!2b1!7m2!1e3!2b1!50m26!1m21!2m7!"
        "1u3!4z6JCl5Lia5Lit!5e1!9s0ahUKEwiF2tqos5OEAxX8ADQIHZgiDgUQ_KkBCNUFKBY!"
        "10m2!3m1!1e1!2m7!1u2!4z6K-E5YiG5pyA6auY!5e1!9s0ahUKEwiF2tqos5OEAxX8ADQIHZgiDgUQ_KkBCNYFKBc!"
        "10m2!2m1!1e1!3m1!1u3!3m1!1u2!4BIAE!2e2!3m2!1b1!3b1!59BQ2dBd0Fn!61b1!67m2!7b1!10b1!69i680&q={keyword_str}&tch=1&ech=35&psi=SlrAZfGlA8-S0PEPk52koA8.1707104844998.1"
    )
    result = 0
    while 1:
        try:
            response = requests.get(url, headers=headers, timeout=(7, 15))
            res0 = response.text.split('/*""*/')[0]
            res0_dict = json.loads(res0)
            data_dict = json.loads(res0_dict["d"].split('\n')[1])
            
            search_list = data_dict[0][1]
            if len(search_list) <= 1:
                break
            for search in search_list:
                if len(search) == 15:
                    para = search[14][10]
                    if para in history_list:
                        continue
                    name = search[14][11]
                    try:
                        score = search[14][4][7]
                        comments_num = search[14][4][8]
                    except:
                        score = ''
                        comments_num = 0
                    addr = search[14][39]
                    country = search[14][30]
                    latt = search[14][9][2]
                    lonn = search[14][9][3]
                    tags = '|'.join(search[14][13]) if search[14][13] else ''
                    lock.acquire()
                    with open(file_path, mode='a', encoding='utf-8-sig', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([keyword, lat, lon, page, para, name, score, comments_num, 
                                          addr, country, latt, lonn, tags])
                    history_list.append(para)
                    with open('history.log', mode='a', encoding='utf-8-sig', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([para])
                    lock.release()
                else:
                    pass
            url = url.replace(f'!8i{page*20}!', f'!8i{page*20+20}!') if f'!8i{page*20}' in url else url.replace('!7i20', f'!7i20!8i{page*20+20}')
            page += 1
            print('page:', keyword, lat, lon, page)
        except Exception as e:
            result += 1
            print(keyword, lat, lon, page, e)
            if result >= 5:
                break
            print(keyword, lat, lon, page, e)
            continue
    print('corr:', keyword, lat, lon)
def get_key_words():
    df = pd.read_csv(Filtered_Data_Path)
    selected_columns = ['LOCATION_NAME', 'LATITUDE', 'LONGITUDE']
    df_selected_columns = df[selected_columns]
    # --------- 测试代码 ----------------
    start_row = 0  # 指定开始行数
    num_rows = 10  # 指定要读取的行数
    df_selected_rows = df_selected_columns.iloc[start_row:start_row + num_rows]
    # --------- 测试代码 ----------------
    keyword_list = []
    #for index, row in df_selected_columns.iterrows():  # -----正式使用时开启，并把测试代码注释掉
    for index, row in df_selected_rows.iterrows():  # --------- 测试代码 ----------------
        dic = {}
        kyeword_name = row[0]
        keyword_lat = row[1]
        keyword_lon = row[2]
        RANGE = 0.1  # ### ---根据需要：修改范围参数---  ###
        west_lon, south_lat, east_lon, north_lat = float(keyword_lon) - RANGE, float(keyword_lat) - RANGE, \
            float(keyword_lon) + RANGE, float(keyword_lat) + RANGE
        lon_gap = 0.06  # ### ---根据需要：修改间距参数---  ###
        lat_gap = 0.03  # ### ---根据需要：修改间距参数---  ###
        coor_list = []
        while west_lon < east_lon + lon_gap:
            south_lat_temp = south_lat
            while south_lat_temp < north_lat:
                coor_list.append([south_lat_temp, west_lon])
                south_lat_temp += lat_gap
            coor_list.append([north_lat, west_lon])
            west_lon += lon_gap
        dic[kyeword_name] = coor_list
        keyword_list.append(dic)
    return keyword_list

def location_search(keyword_list):
    with ThreadPoolExecutor(50) as t:
        for keyword_dic in keyword_list:
            for keyword, coor_list in keyword_dic.items():
                file_path = f'{Location_File_Path}/{keyword}_result.csv'
                with open(file_path, mode='w', encoding='utf-8-sig', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['keyword', 'lat', 'lon', 'page', 'para', 'name', 'score', 'comments_num',
                                     'addr', 'country', 'latt', 'lonn', 'tags'])
                for lat, lon in coor_list:
                    lat = "{:0<.15f}".format(lat)
                    lon = "{:0<.15f}".format(lon)
                    t.submit(get_search_list, keyword, lat, lon, file_path)
                print('keyword:', keyword)
                print("完成")
# ------------------------
def get_comments(file_path,para, name, lat, lon, lock, para2=''):
    url = "https://www.google.com/maps/rpc/listugcposts"
    result = 0
    while 1:
        try:
            params = {
                "authuser": "0",
                "hl": "zh-CN",
                "gl": "us",
                "pb": f"!1m7!1s{para}!3s!6m4!4m1!1e1!4m1!1e3!2m2!1i10!2s{para2}!3e2!5m2!1sdUXAZZXwC4HE0PEP_NSiqA8!7e81!8m5!1b1!2b1!3b1!5b1!7b1!11m6!1e3!2e1!3szh-CN!4sus!6m1!1i2!13m1!1e1"
            }
            response = requests.get(url, headers=headers, params=params, timeout=(7, 15))
            res0 = response.text.split('\n')[1]
            data_dict = json.loads(res0)
            comments_list = data_dict[2]
            for comments in comments_list:
                try:
                    text = comments[0][2][-1][0][0]
                    text = text.replace('\n','')
                except:
                    text = ''
                try:
                    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(comments[0][1][2]/1000000))
                except:
                    date = ''
                try:
                    score = comments[0][2][0][0]
                except:
                    score = ''
                lock.acquire()
                with open(file_path, mode='a', encoding='utf-8-sig', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([name, lat, lon, text, date, score])
                lock.release()
            para2 = data_dict[1]
            if para2 == None:
                break
            print(para, name, lat, lon, para2)
        except Exception as e:
            result += 1
            print(file_path, e)
            if result >= 10:
                break
            continue
        return result
def get_file_path(path):
    files_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            files_name = os.path.join(root, name)
            if files_name.endswith(".csv"):
                files_list.append(files_name)
    return files_list
def get_comment_data(Location_Path):
    files_list = get_file_path(Location_Path)
    for file_path in files_list:
        try:
            df = pd.read_csv(file_path)
            head_list = df.columns.tolist()
            if 'keyword' in head_list:
                df.drop(columns=['keyword', 'lat', 'lon', 'page', 'comments_num'], inplace=True)
                df.drop_duplicates(inplace=True)
                # file_name = file_path.split('/')[-1]
                # file_path_2 = f'{Location_File_Commentdata_Path}/{file_name}'
                # print(file_path_2)
                with open(file_path, mode='w', encoding='utf-8-sig', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['name', 'lat', 'lon', 'text', 'date', 'score'])
                with ThreadPoolExecutor(50) as t:
                    futures = []
                    for i in range(len(df)):
                        if 'para' in df.columns:
                            para = df.at[i, 'para']
                            name = df.at[i, 'name']
                            lat = df.at[i, 'latt']
                            lon = df.at[i, 'lonn']
                            future = t.submit(get_comments, file_path, para, name, lat, lon, lock)
                            futures.append(future)
            else:
                print(f'此文件评论已经下载过{file_path}')
                continue
        except KeyError:
            continue
# ------------------------
def data_clean():
    path_list = get_file_path(Location_File_Path)
    # path_list = get_file_path(Location_File_Commentdata_Path)
    for path in path_list:
        df = pd.read_csv(path)
        head_list = df.columns.tolist()
        if 'keyword' in head_list:
            continue
        else:
            if 'name' in head_list:
                df.drop(columns=['name', 'lat', 'lon', 'score'], inplace=True)
                df.dropna(subset=['text'], inplace=True)
                df.drop_duplicates(subset=['text'], inplace=True)
                english_pattern = '[a-zA-Z0-9]'
                if 'text' in df.columns and not df['text'].empty:
                    english_df = df[df['text'].str.contains(english_pattern, na=False)]
                    if not english_df.empty:
                        english_df.to_csv(path, index=False)
                else:
                    continue
            else:
                continue
def filtered_data_clean(Filtered_Data_Path):
    df = pd.read_csv(Filtered_Data_Path)
    data_info = df[['PLACEKEY', 'LOCATION_NAME']]
    data_info_unique = data_info.drop_duplicates(subset=['LOCATION_NAME'])
    filtered_data_dict = {}
    for index, row in data_info_unique.iterrows():
        location_name = row['LOCATION_NAME']
        placekey = row['PLACEKEY']
        filtered_data_dict[location_name] = placekey
    return filtered_data_dict
def has_data_after_header(reader):
    try:
        next_row = next(reader)
        return True
    except StopIteration:
        return False
def save_comment(filtered_path,location_path,save_path):
    save_name = filtered_path.split('/')[-1]
    save_name = f'Comment_{save_name}'
    path_list = get_file_path(location_path)
    filtered_data_dict = filtered_data_clean(filtered_path)
    with open(f'{save_path}/{save_name}', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for path in path_list:
            name = path.split('/')[-1].split('.csv')[0]
            name = name.split('_result')[0]
            placekey = filtered_data_dict[name]
            row_list = [placekey, name]
            with open(path, mode='r') as csvfile:
                reader = csv.reader(csvfile)
                head = next(reader)
                if 'comments_num' in head:
                    continue
                else:
                    if has_data_after_header(reader):
                        for row in reader:
                            new_row = []
                            for ro in row:
                                ro = f'[{ro}]'
                                new_row.append(ro)
                            new_row = ','.join(new_row)
                            row_list.append(new_row)
                        writer.writerow(row_list)
                        print(f'成功保存----> {name} 全部评论')
                    else:
                        print("CSV文件除了表头之外没有其他数据。")

if __name__ == '__main__':
    lock = Lock()
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    Filtered_Data_Path = r'unique_SD_location.csv'  # ### ---原始关键词文件路径---  ###
    Location_File_Path = r'data'  # ### ---下载的地点检索数据路径---  ###
    Save_Comment_Path = r'result_file'  # ### ---保存评论路径---  ###

    # Location_File_Commentdata_Path = '/Users/rongtian/Desktop/2144883396941630292/commentdata'  # ##### 另一套保存路径 可不用   ###
    history_list = []
    with open('history.log', mode='r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    for line in lines:
        history_list.append(line.strip())

    '''  下面三部分代码，在需要的情况下可以注释掉两个，然后单独运行。。。'''
    # -----------地点检索-------------------------
    keyword_list = get_key_words()
    location_search(keyword_list)
    print('地点检索已全部下载完成')


    # -----------评论爬取-------------------------
    get_comment_data(Location_File_Path)
    print('地点评论已全部下载完成')

    # get_comment_data(Location_File_Path,Location_File_Commentdata_Path)  ###### 另一套保存地址 可不用


    # -----------评论数据清洗 保存-----------------
    data_clean()
    save_comment(Filtered_Data_Path,Location_File_Path,Save_Comment_Path)
    print('评论数据已全部下载完成')

    # save_comment(Filtered_Data_Path,Location_File_Commentdata_Path,Save_Comment_Path)  ###### 另一套保存地址 可不用

