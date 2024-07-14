import pandas as pd
import os

TIME_OFFSET = pd.Timedelta('2 hours') # The time offset between the photo name and the actual time (metadata)

def read_original_file(file_path):
    data = pd.read_csv(file_path, delimiter=',')
    df = pd.DataFrame(data)
    df['datum'] = pd.to_datetime(df['datum'])
    
    # Remove all dates before 2022-06-18
    df = df[df['datum'] > '2022-06-18']
    
    return df

def photo_name_to_date(photo_name, dt, round_to_half_hour=True):
    date = photo_name.split("_")[1]
    date = pd.to_datetime(date, format='%Y%m%d%H%M') + dt
    
    # I think this is good to round the time to the nearest half hour
    # We should predict the future, so we should round up
    if round_to_half_hour:
        if date.minute < 30:
            date = date.replace(minute=0)
        else:
            date = date.replace(minute=30)
    
    return date

def get_closest_date(df, date):
    closest_date = df.iloc[(df['datum']-date).abs().argsort()[:1]] #find a fast way to get the closest date
    return closest_date        

def get_weather_data(df, date):
    data = df[df['datum'] == date]
    if data.empty:        
        closest_date = get_closest_date(df, date)
        data = closest_date
    
    if len(data) > 1:
        data = data.iloc[0]
        
    return data

if __name__ == '__main__':
    df = read_original_file('1838.txt')
    new_columns = ['name'] + list(df.columns)
    new_df = pd.DataFrame(columns=new_columns)
    
    for image_name in os.listdir('maribor_letalisce_36'):
        if image_name.endswith('.jpg'):
            date = photo_name_to_date(image_name, TIME_OFFSET)
            weather_data = get_weather_data(df, date)
                    
            try:
                new_df.loc[len(new_df)] = [image_name] + weather_data.values.tolist()[0]
            except:
                print("Error with image: ", image_name)
                
    new_df.to_csv('dataframe.csv', index=False)
            
                        
    #photo_name = "CM00006_202206200340_23.jpg"
    #date = photo_name_to_date(photo_name, TIME_OFFSET)
    #weather_data = get_weather_data(df, date)
    #new_columns = ['name'] + list(df.columns)
    #new_df = pd.DataFrame(columns=new_columns)
    #new_df.loc[0] = [photo_name] + weather_data.values.tolist()[0]
    #print(new_df)
    
    