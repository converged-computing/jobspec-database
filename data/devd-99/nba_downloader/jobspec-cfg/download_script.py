import json
from nba_video import download_nba_video
import json
from datetime import datetime

game_dates = {}
with open('game_date.json', 'r') as file:
        data = json.load(file)
        for i,v in enumerate(data):
            for k in data[i]:
                if k not in game_dates:
                    game_dates[k] = data[i][k]

def getDate(gId):
    if gId in game_dates:
        date = game_dates[gId]
    retDate = datetime.strptime(date, '%b %d, %Y').strftime('%Y/%m/%d')
    return retDate


with open('playDict.json', 'r') as file:
    playDict = json.load(file)

games_downloaded = 0
for game in playDict:

    # if(games_downloaded>=10):
    #     break
    print(f"Downloading game# {games_downloaded+1}")

    actions = playDict[game]["game"]["actions"]

    for action in actions:
        action_type = action["actionType"]
        if action_type =="Missed Shot":
            event_id = action["actionNumber"]
            date = getDate(game)
            

            url = download_nba_video(game, event_id, date)
            if(url):
                print("Video downloaded successfully. URL:", url)
    games_downloaded+=1

