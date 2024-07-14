import PIL.Image, pandas as pd

MAIN_DIR = 'data/run6/temp'
TEMP_DIR = 'data/run6'
logs = ['log_0OI2J.csv', 'log_b8RLi.csv', 'log_GOU8i.csv', 'log_pjxuw.csv', 
        'log_R9hB0.csv', 'log_8rN8I.csv', 'log_F8hN7.csv', 'log_h0xPt.csv', 'log_QBA0b.csv']
merge_volumes = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
directories = ['images_0', 'images_10', 'images_20', 'images_30', 'images_40', 'images_50', 'images_60']

for log in logs:
    for m_vol, dir in zip(merge_volumes, directories):
        df = pd.read_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/logger/{log}')
        df = df[df.m_vol == m_vol]
        names_list = df['name'].values
        for name in names_list:
            image = PIL.Image.open(f'/gpfs/scratch/hebal100/{MAIN_DIR}/{dir}/{name}.png')
            image.save(f'/gpfs/scratch/hebal100/{TEMP_DIR}/{dir}/{name}.png')
