import json
import numpy as np

BASE_DIR = '/work/07016/cw38637/ls6/nlp'
videos = [
    'family-man',
    'harry-potter',
    'matrix',
    'perfect-storm',
    'polar-express'
]

for video in videos:
    with open(f'{BASE_DIR}/{video}-content.txt', 'w') as f:
        f.write('Given the following image captions from a video: ')
        video_info = json.load(open(f'{BASE_DIR}/{video}-video.txt', 'r'))
        for index, caption in enumerate(video_info['captions']):
            f.write(f'{index + 1}) {caption} ')
        f.write('and the following transcriptions: ')
        transcriptions = open(f'{BASE_DIR}/{video}-audio.txt', 'r').readlines()
        for index, line in enumerate(transcriptions):
            words = line.strip().split(' ', 3)
            f.write(f'{index + 1}) {words[3]} ({words[2]}) ')
        f.write('and given that the sentiments of the video are: ')
        keys = np.array(video_info['sentiments'][0])
        values = np.array(video_info['sentiments'][1])
        ranks = np.argsort(values)[::-1]
        top_k = 5
        top_keys = keys[ranks[:top_k]]
        top_values = values[ranks[:top_k]]
        top_percents = (top_values / np.sum(top_values)) * 100
        for index in range(top_k):
            f.write(f'{top_keys[index]} {top_percents[index]:.2f}% ')
        f.write('Describe the music that would fit such a video. Your output will be fed to a text to music model. To help you out, here are some prompts that worked well with the model: 1) Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach 2) classic reggae track with an electronic guitar solo 3) earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves 4) lofi slow bpm electro chill with organic samples 5) violins and synths that inspire awe at the finiteness of life and the universe 6) 80s electronic track with melodic synthesizers, catchy beat and groovy bass Give me only the description of the music without any explanation.\n')
