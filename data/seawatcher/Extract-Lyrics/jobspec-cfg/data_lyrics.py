import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
# import urllib.requests
import urllib.parse
import urllib.error
from urllib.request import Request, urlopen
import json
from bs4 import BeautifulSoup
import ssl
from googlesearch import search
import time
import argparse


def scrape_lyrics(url):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()

    soup = BeautifulSoup(webpage, 'html.parser')

    html = soup.prettify('utf-8')
    song_json = {"Lyrics": [], "Comments": []}

    for title in soup.find_all('title'):
        song_json["Title"] = title.text.strip()

    for div in soup.find_all('div', attrs={'class': 'lyrics'}):
        song_json["Lyrics"].append(div.text.strip().split("\n"));

    return song_json


def Extract(track_name, artist_name):
    track_name = str(track_name)
    artist_name = str(artist_name)
    query = "genius lyrics " + track_name + " " + artist_name
    url = ''
    for j in search(query, tld="co.in", num=1, stop=1, pause=3):
        url = j

        if(url.find('genius') == -1):
            print("Song Not Found: %s,%s" %(track_name, artist_name))
            flag = False
            song_json = []
            return track_name, artist_name, song_json, flag
            # continue

        try:
            song_json = scrape_lyrics(url)
            if len(song_json['Lyrics']) != 0:
                flag = True
                # with open(Track_name + " " + artist_name + '.json', 'w') as outfile:
                #     json.dump(song_json, outfile, indent = 4, ensure_ascii = False)
            else:
                while (len(song_json['Lyrics']) == 0):
                    song_json = scrape_lyrics(url)
                if len(song_json['Lyrics']) != 0:
                    flag = True
                    # with open(Track_name + " " + artist_name + '.json', 'w') as outfile:
                        # json.dump(song_json, outfile, indent = 4, ensure_ascii = False)
                else:
                    flag = False
                    print(track_name + artist_name)

        except:
            print("Song Not Found in Genius: %s" % (track_name + " " + artist_name))
            flag = False
            song_json = []

        return track_name, artist_name, song_json, flag


def save_lyrics(dataset_path, json_path):
    # dataset_path: dataset path
    # json_path: the path to save json

    data_list = pd.read_excel(dataset_path)

    # dictionary to store data
    data = {
        "Artist": [],
        "Title": [],
        "Lyric": [],
        "Mood": []
    }

    num = 0

    for row_index, song in data_list.iterrows():
        title_name, artist_name, song_lyric, flag = Extract(song.Title, song.Artist)
        if flag:
            data["Title"].append(title_name)
            data["Artist"].append(artist_name)
            data["Lyric"].append(song_lyric)
            data["Mood"].append(song.Mood)
            num += 1
            print("Succeed %04d : " % num + title_name)
        time.sleep(70)

    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)
        print('Saved json/lyric_{:03d}.json: '.format(ID), len(data["Title"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    id_parser = parser.add_argument('--id',
                                    metavar='ID',
                                    action='store',
                                    help='xlsx file ID',
                                    type=int)
    id_args = parser.parse_args()
    ID = id_args.id

    dataset_path = 'output/lyric_{:03d}.xlsx'.format(ID)
    json_path = 'json/lyric_{:03d}.json'.format(ID)

    save_lyrics(dataset_path, json_path)
