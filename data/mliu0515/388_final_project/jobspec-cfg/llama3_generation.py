import time
import json
from llama import Llama
import time

start = time.time()

ckpt_dir = '/work/07016/cw38637/ls6/llama3-weights/Meta-Llama-3-8B-Instruct/'
tokenizer_path = '/work/07016/cw38637/ls6/llama3-weights/Meta-Llama-3-8B-Instruct/tokenizer.model'
temperature = 0.6
top_p = 0.9
max_seq_len = 4096
max_batch_size = 5

generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size
)

BASE_DIR = '/work/07016/cw38637/ls6/nlp'
videos = [
    'family-man',
    'harry-potter',
    'matrix',
    'perfect-storm',
    'polar-express'
]

contents = []
for video in videos:
    name = f'{BASE_DIR}/{video}-content.txt'
    content = open(name, 'r').readlines()[0].strip()
    contents.append(content)

questions = []

for content in contents:
    questions.append([{
        "role": "user",
        "content": content
    }])

answers = generator.chat_completion(
    questions,
    temperature=temperature,
    top_p=top_p
)

end = time.time()
print(f'time: {end-start:.2f}s\n')
print(answers)

for idx in range(len(videos)):
    with open(f'{BASE_DIR}/{video}-result.txt','w') as f:
        video = videos[idx]
        question = questions[idx][0]['content']
        answer = answers[idx]['generation']['content']
        result = {
            'video': video,
            'question': question,
            'answer': answer
        }
        json.dump(result, f)

# contents = [
#     "Given the following image captions from a video: 1) a black and white photo of a woman driving a car 2) a black and white photo of a woman driving a car 3) a black and white photo of a woman sitting in the driver's seat of a car 4) a black and white photo of a woman sitting in the driver's seat of a car 5) a black and white photo of a woman sitting in the driver's seat of a car 6) a black and white photo of a woman sitting in the driver's seat of a car 7) a black and white photo of a woman driving a car 8) a black and white photo of a woman driving a car 9) a black and white photo of a woman driving a car 10) a black and white photo of a woman driving a car 11) a black and white photo of a car driving down the road at night 12) a black and white photo of a woman in a car 13) a black and white photo of a woman in a car 14) a black and white photo of a woman in a car 15) a black and white photo of a woman in a car 16) a black and white photo of a woman in a car 17) a black and white photo of a woman in a car 18) a black and white photo of a woman in a car 19) a black and white photo of a woman in a car 20) a black and white photo of a woman in a car and the following transcriptions: 1) I said you could. I was the last I saw. (SPEAKER_01) 2) Wait a minute. I did see her some time later driving. (SPEAKER_01) 3) I think you'd better come over here to my office, quick. (SPEAKER_01) 4) Carolyn, get Mr. Cassidy for me. (SPEAKER_00) 5) After all, Cassidy, I told you, all that cash. (SPEAKER_01) 6) I'm not taking the responsibility. (SPEAKER_01) 7) Oh, for heaven's sake, girl works for you for 10 years, you trust her. (SPEAKER_01) 8) All right, yes, you'd better come over. (SPEAKER_01) and given that the sentiments of the video are: suspenseful 95.00%% afraid 5.00%% Describe the music that would fit such a video. Your output will be fed to a text to music model. To help you out, here are some prompts that worked well with the model: 1) Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach 2) classic reggae track with an electronic guitar solo 3) earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves 4) lofi slow bpm electro chill with organic samples 5) violins and synths that inspire awe at the finiteness of life and the universe 6) 80s electronic track with melodic synthesizers, catchy beat and groovy bass Give me only the description of the music without any explanation.",

#     "Given the following image captions from a video: 1) a yellow taxi cab is parked in front of a building 2) a yellow taxi cab driving down the street 3) a man in a suit and tie sitting in the back seat of a car 4) a man in a suit sitting in the back seat of a car 5) a yellow taxi cab is driving down the road 6) a man in a suit getting out of a taxi cab 7) a man sitting in the back of a yellow taxi cab 8) a man in a suit and tie riding a yellow taxi 9) a man in a suit and tie is running down the street 10) a woman is sitting in the driver's seat of a car 11) there is a close up of a person in a car 12) a man with a mustache is walking down the street 13) a man in a suit is walking down the street in front of a yellow taxi 14) a man in a suit and tie running down the street 15) a man is standing next to a taxi cab 16) a man is standing next to a yellow taxi cab 17) a man is hugging another man in front of a yellow taxi 18) two men are leaning on the hood of a taxi cab 19) two men are fighting in front of a yellow taxi 20) a man in a suit is getting out of a yellow taxi 21) a man is getting out of a yellow car 22) a man is being pushed out of a car by another man 23) a man in a car is being pushed by another man 24) a taxi cab with a man sitting in the driver's seat 25) a man standing next to a yellow taxi cab 26) a man in a suit and tie is running down the street 27) a man in a suit and tie is walking down the street 28) a man in a suit and tie running down the street 29) a man with a mustache is walking in front of a car 30) a man with a moustache is making a funny face 31) a man is walking down the street with a bag 32) an old car is driving down the street in front of a park 33) a man with a moustache standing in front of a building 34) a man in a black jacket is walking down the street 35) a group of people playing frisbee in a park 36) a man with a briefcase running through a park 37) a man and a woman walking down a path in a park 38) a man running across a grassy field with a suitcase and the following transcriptions: 1) Hey! (SPEAKER_00) 2) Hey! (SPEAKER_00) 3) Where they going? Come here! (SPEAKER_00) 4) Hey! No! (SPEAKER_00) 5) No, no, no, no! (SPEAKER_00) 6) Get my money! Get my money! (SPEAKER_00) 7) Please, please, please! (SPEAKER_00) 8) Please! (SPEAKER_00) 9) He should have paid you! He should have paid you! (SPEAKER_00) 10) I'm sorry! I'm so sorry! (SPEAKER_00) 11) I'm sorry! (SPEAKER_00) 12) Idiot! (SPEAKER_00) and given that the sentiments of the video are: action 94.44%% sad 2.78%% sympathetic 2.78%% Describe the music that would fit such a video. Your output will be fed to a text to music model. To help you out, here are some prompts that worked well with the model: 1) Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach 2) classic reggae track with an electronic guitar solo 3) earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves 4) lofi slow bpm electro chill with organic samples 5) violins and synths that inspire awe at the finiteness of life and the universe 6) 80s electronic track with melodic synthesizers, catchy beat and groovy bass Give me only the description of the music without any explanation.",

#     "Given the following image captions from a video: 1) a black background with an airplane flying in the air 2) a black background with an airplane flying in the air 3) a black background with an airplane flying in the air 4) two men in suits standing next to each other 5) two men in suits sitting next to each other 6) a man in a suit and tie talking to another man in a suit 7) two men in suits standing at a counter 8) two men in suits standing at a counter 9) two men in suits sitting at a counter 10) a man with a mustache in a purple suit and bow tie 11) a man in a purple suit and bow tie 12) a man in a purple suit with a moustache and bow tie 13) a man in a purple suit and bow tie is looking away from the camera 14) a blurry image of an orange door with a sign on it 15) a man with glasses standing in front of a mirror 16) a man with glasses and a mustache is standing in front of a door 17) a man with a mustache and glasses is smoking a pipe 18) a man in a suit and tie smoking a pipe 19) a man sitting on a couch in an orange room 20) a man sitting on a couch in an orange room 21) a man sitting on a chair in an orange room 22) a man standing at the front desk of a hotel 23) a man standing at the top of a set of stairs 24) a man in a suit standing at the top of a set of stairs 25) the front page of a newspaper with the words immigrant claims fortune 26) the front page of a newspaper with the words immigrant claims fortune 27) the front page of a newspaper with the words immigrant claims fortune and the following transcriptions: 1) Who's this interesting old fellow? (SPEAKER_02) 2) I inquired of Monsieur Jean. (SPEAKER_02) 3) To my surprise, he was distinctly taken aback. (SPEAKER_02) 4) Don't you know? He asked. (SPEAKER_01) 5) Don't you recognize him? (SPEAKER_01) 6) He did look familiar. (SPEAKER_01) 7) That's Mr. Mustafa himself. He arrived early this morning. (SPEAKER_00) 8) This name will no doubt be familiar to the more seasoned persons among you. (SPEAKER_02) 9) Mr. Zero Mustafa was, at one time, the richest man in Zubrovka. (SPEAKER_02) and given that the sentiments of the video are: surprised 50.00%% action 36.36%% neutral 13.64%% Describe the music that would fit such a video. Your output will be fed to a text to music model. To help you out, here are some prompts that worked well with the model: 1) Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach 2) classic reggae track with an electronic guitar solo 3) earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves 4) lofi slow bpm electro chill with organic samples 5) violins and synths that inspire awe at the finiteness of life and the universe 6) 80s electronic track with melodic synthesizers, catchy beat and groovy bass Give me only the description of the music without any explanation.",

#     "Given the following image captions from a video: 1) the sun is setting over the city of san francisco 2) a woman in lingerie standing in front of a crowd 3)a video of people dancing in a club 4) a group of people sitting at a table at night 5) a group of people standing around a table at night 6) a group of people sitting around a table at night 7) a young man sitting at a table in a dimly lit room 8) a young man sitting at a table in a dimly lit room 9) a group of people sitting around a table at night 10) a group of people sitting around a table at a bar 11) a young man sitting at a table in a dimly lit room 12) a young man is looking at the camera with his mouth open. and the following transcriptions: 1) I was crashing there for a little bit while taking care of some things but she's done (SPEAKER_00) 2) for the summer so she's back in her parents place. (SPEAKER_00) 3) The homeless rock star Palazzo. (SPEAKER_00) 4) Alright. (SPEAKER_00) 5) What's your plan for the summer? (SPEAKER_00) 6) Mark. (SPEAKER_00) and given that the sentiments of the video are: excited 50.00%% energetic 35.71%% sensual 14.29%% Describe the music that would fit such a video. Your output will be fed to a text to music model. To help you out, here are some prompts that worked well with the model: 1) Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach 2) classic reggae track with an electronic guitar solo 3) earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves 4) lofi slow bpm electro chill with organic samples 5) violins and synths that inspire awe at the finiteness of life and the universe 6) 80s electronic track with melodic synthesizers, catchy beat and groovy bass Give me only the description of the music without any explanation.",

#     "Given the following image captions from a video: 1) a man and woman standing on the deck of a boat at sunset 2) a man and woman standing on the deck of a boat at sunset 3) a woman with red hair standing in front of a ship 4) a woman with red hair standing in front of a ship 5) leonardo dicaprio in titanic 6) leonardo dicaprio in 'the great gatsby' 7) a woman with red hair standing in front of a ship 8) a woman with red hair standing in front of a ship 9) leonardo dicaprio in 'titanic' 10) leonardo dicaprio in the movie titanic 11) a woman with red hair standing in front of a ship 12) a woman with red hair standing in front of a ship 13) a man and woman standing on the deck of a boat at sunset 14) a man and woman standing on the deck of a boat at sunset 15) a close up of a woman with red hair 16) a woman with red hair looking at a man 17) a man and a woman looking at each other in a scene from the movie titanic 18) leonardo dicaprio and mia farrow in titanic 19) a woman looking into the eyes of a man 20) a man and a woman are looking at each other 21) a man and a woman standing next to each other 22) leonardo dicaprio and meryl streep in titanic and the following transcriptions: 1) I said you might be after me. (SPEAKER_01) 2) Give me your hand. (SPEAKER_00) 3) Now close your eyes. (SPEAKER_00) 4) Go on. (SPEAKER_00) 5) Step up. (SPEAKER_00) 6) Now hold on to the railing. (SPEAKER_00) 7) Keep your eyes closed. (SPEAKER_00) and given that the sentiments of the video are: romantic 100.00% Describe the music that would fit such a video. Your output will be fed to a text to music model. To help you out, here are some prompts that worked well with the model: 1) Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach 2) classic reggae track with an electronic guitar solo 3) earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves 4) lofi slow bpm electro chill with organic samples 5) violins and synths that inspire awe at the finiteness of life and the universe 6) 80s electronic track with melodic synthesizers, catchy beat and groovy bass Give me only the description of the music without any explanation."
# ]
